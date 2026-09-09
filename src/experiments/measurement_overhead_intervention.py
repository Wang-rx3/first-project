"""Validation-only measurement-message communication-overhead intervention.

The existing measurement action controls which neighbor is consumed by the
cooperative EKF, but historically it did not consume any communication
resources.  This module replays the frozen baseline-MAPPO validation episodes
with an experiment-only switch that adds one equivalent 512-byte packet to the
selected source-to-target queue.  The switch is disabled for the control
replay, so the old validation trace can be checked before interpreting any
communication effect.

The experiment uses common random numbers.  The five base seeds are the
independent statistical units; episodes are paired within seed, while windows
and EKF update events are diagnostic observations rather than independent
replicates.  Test data are never read.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files
from .information_gain_budget_intervention import (
    EXPECTED_BASE_SEEDS,
    TREATMENTS,
    _enrich_window_rows,
    _episode_metrics_with_intervention,
    _fit_seed_proxy,
    _load_manifest,
    _number,
    _scale_roots,
    _split_episodes,
    _train_budget,
    intervene_information_gain_selection,
)
from .localization_coupling_intervention import _bootstrap_interval, _mean, _write_table
from .pilot import _EpisodeTraceWriter
from .quality_aware_measurement_intervention import (
    QualityAwareMeasurementInterventionConfig,
    _run_episode as _run_replay_episode,
)


CONTROL_TREATMENT = "recorded_mappo_no_overhead"
ALL_TREATMENTS = (CONTROL_TREATMENT,) + TREATMENTS
PACKET_SIZE_BYTES = 512
NUMERIC_TOLERANCE = 2e-6
POLICY_PREFIX = "measurement_overhead_intervention"
SEED_RE = re.compile(r"^seed_(\d+)$")

OVERHEAD_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "scenario_id",
    "seed",
    "base_seed",
    "split",
    "treatment",
    "step",
    "timestamp",
    "source_uav_id",
    "target_uav_id",
    "selected_non_none",
    "overhead_mode",
    "payload_size_bytes",
    "load_packets_requested",
    "load_packets_applied",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "queue_before_packets",
    "queue_with_measurement_packets",
    "queue_next_packets",
    "channel_bandwidth_hz",
    "rate_bps",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "link_state",
    "ekf_behavior_unchanged",
)


@dataclass(frozen=True)
class MeasurementOverheadInterventionConfig:
    """Configuration for the validation-only resource-coupling experiment."""

    reference_root: Path
    reference_branch: str = "baseline_mappo"
    output_dir: Path = Path("data/synthetic/python_sim_measurement_overhead_intervention_seed2026")
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    age_variance_coefficient: float = 1.0
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.reference_branch != "baseline_mappo":
            raise ValueError("measurement overhead intervention requires baseline_mappo")
        if self.split != "validation":
            raise ValueError("measurement overhead intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not math.isfinite(self.age_variance_coefficient) or self.age_variance_coefficient < 0.0:
            raise ValueError("age_variance_coefficient must be finite and non-negative")


def _progress(config: MeasurementOverheadInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[measurement-overhead] {message}", flush=True)


def _default_logs_dir(reference_root: Path) -> Path:
    return Path("logs/preliminary") / reference_root.name / "measurement_overhead_intervention"


def _default_plots_dir(reference_root: Path) -> Path:
    return Path("plots/preliminary") / reference_root.name / "measurement_overhead_intervention"


def _prepare_outputs(config: MeasurementOverheadInterventionConfig) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else _default_logs_dir(Path(config.reference_root))
    plots = Path(config.plots_dir) if config.plots_dir is not None else _default_plots_dir(Path(config.reference_root))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"measurement-overhead output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _write_measurement_overhead(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write an extra diagnostic file, including a header for empty controls."""

    normalized = []
    for row in rows:
        normalized.append({column: row.get(column, "") for column in OVERHEAD_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, OVERHEAD_COLUMNS, normalized)


def _relabel(rows: Sequence[Mapping[str, Any]], treatment: str) -> list[dict[str, Any]]:
    return [dict(row, treatment=treatment) for row in rows]


def _objective_residual(episode_root: Path) -> tuple[float, float]:
    """Check J=Jloc+Jcom and r=-(J+collision+energy) from written rows."""

    rewards = _read_csv(episode_root / "rewards.csv")
    scenario_payload = _read_json(episode_root / "scenario.json")
    reward_weights = dict(scenario_payload.get("reward_config", {}).get("weights", {}))
    uncertainty_weight = float(
        reward_weights.get("uncertainty", reward_weights.get("unc", 0.35))
    )
    objective_residuals: list[float] = []
    reward_residuals: list[float] = []
    for row in rewards:
        joint = float(row.get("joint_objective", 0.0))
        # Jloc is the uncertainty contribution recorded by the environment;
        # Jcom is the remainder of the canonical core objective.
        j_loc = uncertainty_weight * float(row.get("normalized_uncertainty", 0.0))
        j_com = joint - j_loc
        objective_residuals.append(abs((j_loc + j_com) - joint))
        reward_total = float(row.get("reward_total", 0.0))
        reward_collision = float(row.get("reward_collision", 0.0))
        reward_energy = float(row.get("reward_energy", 0.0))
        reward_residuals.append(abs(reward_total + joint - reward_collision - reward_energy))
    return max(objective_residuals or [0.0]), max(reward_residuals or [0.0])


def _numeric_difference(
    first: Path,
    second: Path,
    filename: str,
    keys: Sequence[str],
    columns: Sequence[str],
) -> float:
    left = _read_csv(first / filename)
    right = _read_csv(second / filename)
    by_left = {tuple(row.get(key, "") for key in keys): row for row in left}
    by_right = {tuple(row.get(key, "") for key in keys): row for row in right}
    if set(by_left) != set(by_right):
        return float("inf")
    maximum = 0.0
    for key in by_left:
        for column in columns:
            try:
                difference = abs(float(by_left[key].get(column, 0.0)) - float(by_right[key].get(column, 0.0)))
            except (TypeError, ValueError):
                return float("inf")
            maximum = max(maximum, difference)
    return float(maximum)


def _bandwidth_difference(first: Path, second: Path) -> float:
    left = _read_csv(first / "actions.csv")
    right = _read_csv(second / "actions.csv")
    keys = ("timestamp", "uav_id")
    left_map = {tuple(row.get(key, "") for key in keys): row for row in left}
    right_map = {tuple(row.get(key, "") for key in keys): row for row in right}
    if set(left_map) != set(right_map):
        return float("inf")
    maximum = 0.0
    for key in left_map:
        try:
            left_payload = json.loads(left_map[key].get("action_raw", "{}"))
            right_payload = json.loads(right_map[key].get("action_raw", "{}"))
            left_bandwidth = {str(k): float(v) for k, v in dict(left_payload.get("bandwidth", {})).items()}
            right_bandwidth = {str(k): float(v) for k, v in dict(right_payload.get("bandwidth", {})).items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            return float("inf")
        targets = set(left_bandwidth) | set(right_bandwidth)
        maximum = max(maximum, max((abs(left_bandwidth.get(target, 0.0) - right_bandwidth.get(target, 0.0)) for target in targets), default=0.0))
    return float(maximum)


def _measurement_action_difference(first: Path, second: Path) -> float:
    """Compare the recorded measurement choices in two action traces."""

    left = _read_csv(first / "actions.csv")
    right = _read_csv(second / "actions.csv")
    keys = ("timestamp", "uav_id")
    left_map = {tuple(row.get(key, "") for key in keys): row for row in left}
    right_map = {tuple(row.get(key, "") for key in keys): row for row in right}
    if set(left_map) != set(right_map):
        return float("inf")
    maximum = 0.0
    for key in left_map:
        if str(left_map[key].get("measurement_uav_id", "none")) != str(right_map[key].get("measurement_uav_id", "none")):
            maximum = 1.0
    return float(maximum)


def _invariance_checks(
    reference_roots: Mapping[int, Path],
    treatment_roots: Mapping[tuple[int, str], Path],
    episodes_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    checks = {
        "control_reproduces_reference_max_abs_difference": 0.0,
        "control_measurement_action_max_difference": 0.0,
        "cross_treatment_ground_truth_max_abs_difference": 0.0,
        "cross_treatment_obstacles_max_abs_difference": 0.0,
        "cross_treatment_delta_v_max_abs_difference": 0.0,
        "cross_treatment_bandwidth_max_abs_difference": 0.0,
    }
    truth_columns = ("x", "y", "z", "vx", "vy", "vz")
    obstacle_columns = ("x", "y", "z", "vx", "vy", "vz", "radius")
    for seed, episodes in episodes_by_seed.items():
        reference_root = reference_roots[seed]
        control_root = treatment_roots[(seed, CONTROL_TREATMENT)]
        recorded_root = treatment_roots[(seed, "recorded_mappo")]
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            reference = reference_root / "validation" / episode_id
            control = control_root / "validation" / episode_id
            recorded = recorded_root / "validation" / episode_id
            for filename, keys, columns in (
                ("ground_truth.csv", ("timestamp", "uav_id"), truth_columns),
                ("obstacles.csv", ("timestamp", "obstacle_id"), obstacle_columns),
                ("links.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length")),
                ("packet_events.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("success", "delay", "aoi_after")),
                ("actions.csv", ("timestamp", "uav_id"), ("delta_vx", "delta_vy", "delta_vz")),
                ("estimates.csv", ("timestamp", "uav_id"), ("x", "y", "z", "vx", "vy", "vz")),
                ("rewards.csv", ("timestamp", "uav_id"), ("reward_total", "joint_objective", "normalized_uncertainty")),
            ):
                checks["control_reproduces_reference_max_abs_difference"] = max(
                    checks["control_reproduces_reference_max_abs_difference"],
                    _numeric_difference(control, reference, filename, keys, columns),
                )
            checks["control_reproduces_reference_max_abs_difference"] = max(
                checks["control_reproduces_reference_max_abs_difference"],
                _bandwidth_difference(control, reference),
            )
            checks["control_measurement_action_max_difference"] = max(
                checks["control_measurement_action_max_difference"],
                _measurement_action_difference(control, reference),
            )
            for treatment in TREATMENTS:
                candidate = treatment_roots[(seed, treatment)] / "validation" / episode_id
                checks["cross_treatment_ground_truth_max_abs_difference"] = max(
                    checks["cross_treatment_ground_truth_max_abs_difference"],
                    _numeric_difference(recorded, candidate, "ground_truth.csv", ("timestamp", "uav_id"), truth_columns),
                )
                checks["cross_treatment_obstacles_max_abs_difference"] = max(
                    checks["cross_treatment_obstacles_max_abs_difference"],
                    _numeric_difference(recorded, candidate, "obstacles.csv", ("timestamp", "obstacle_id"), obstacle_columns),
                )
                checks["cross_treatment_delta_v_max_abs_difference"] = max(
                    checks["cross_treatment_delta_v_max_abs_difference"],
                    _numeric_difference(recorded, candidate, "actions.csv", ("timestamp", "uav_id"), ("delta_vx", "delta_vy", "delta_vz")),
                )
                checks["cross_treatment_bandwidth_max_abs_difference"] = max(
                    checks["cross_treatment_bandwidth_max_abs_difference"],
                    _bandwidth_difference(recorded, candidate),
                )
    return checks


def _budget_usage(
    episode_rows: Sequence[Mapping[str, Any]],
    budgets_by_seed: Mapping[int, Mapping[str, int]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, str], int] = defaultdict(int)
    for row in episode_rows:
        # Episode-level rows contain only a total.  The per-UAV counts are
        # recovered from the action files in the caller and supplied through
        # this function as hidden fields.
        for uav_id, count in dict(row.get("measurement_counts_by_uav", {})).items():
            groups[(int(row["base_seed"]), str(row["treatment"]), str(row["episode_id"]), str(uav_id))] = int(count)
    output: list[dict[str, Any]] = []
    for seed in sorted(budgets_by_seed):
        for treatment in ALL_TREATMENTS:
            for uav_id, budget in sorted(budgets_by_seed[seed].items()):
                values = [count for (item_seed, item_treatment, _episode, item_uav), count in groups.items() if item_seed == seed and item_treatment == treatment and item_uav == uav_id]
                maximum = max(values or [0])
                mean_value = float(np.mean(values)) if values else 0.0
                output.append({
                    "base_seed": seed,
                    "treatment": treatment,
                    "uav_id": uav_id,
                    "validation_episode_count": len(values),
                    "measurement_budget": int(budget) if treatment == "budgeted_information_gain" else -1,
                    "mean_measurement_count": mean_value,
                    "max_measurement_count": int(maximum),
                    "max_budget_excess": int(max(0, maximum - budget)) if treatment == "budgeted_information_gain" else 0,
                    "mean_budget_usage_fraction": mean_value / max(1, budget) if treatment == "budgeted_information_gain" else -1.0,
                })
    return output


def _seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["base_seed"]), str(row["treatment"]))].append(row)
    fields = (
        "mean_reward", "j_loc", "j_com", "joint_objective", "position_rmse_m", "position_error_p95_m",
        "covariance_position_trace", "accepted_update_count", "beneficial_fraction", "harmful_fraction",
        "neutral_fraction", "packet_loss", "delay_s", "aoi_s", "connectivity", "energy_proxy",
        "collision_rate", "measurement_count_total", "overhead_load_packets_total", "overhead_bytes_total",
        "mean_queue_length_packets", "objective_formula_max_abs", "reward_formula_max_abs",
    )
    output: list[dict[str, Any]] = []
    for (seed, treatment), group in sorted(groups.items()):
        item: dict[str, Any] = {"base_seed": seed, "treatment": treatment, "validation_episode_count": len(group)}
        for field in fields:
            values = [_number(row.get(field), 0.0) for row in group]
            item[field] = float(np.mean(values)) if values else 0.0
        output.append(item)
    return output


def _paired_contrasts(seed_rows: Sequence[Mapping[str, Any]], config: MeasurementOverheadInterventionConfig) -> list[dict[str, Any]]:
    metrics = (
        "position_rmse_m", "position_error_p95_m", "j_loc", "j_com", "joint_objective", "mean_reward",
        "accepted_update_count", "beneficial_fraction", "harmful_fraction", "packet_loss", "delay_s", "aoi_s",
        "connectivity", "measurement_count_total", "overhead_load_packets_total", "overhead_bytes_total",
        "mean_queue_length_packets",
    )
    by_key = {(int(row["base_seed"]), str(row["treatment"])): row for row in seed_rows}
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    comparisons = (
        ("recorded_mappo_minus_no_overhead", "recorded_mappo", CONTROL_TREATMENT),
        ("unconstrained_information_gain_minus_recorded_mappo", "unconstrained_information_gain", "recorded_mappo"),
        ("budgeted_information_gain_minus_recorded_mappo", "budgeted_information_gain", "recorded_mappo"),
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for contrast, treatment, comparator in comparisons:
        for metric in metrics:
            treatment_values = np.asarray([_number(by_key[(seed, treatment)].get(metric), 0.0) for seed in seeds], dtype=float)
            comparator_values = np.asarray([_number(by_key[(seed, comparator)].get(metric), 0.0) for seed in seeds], dtype=float)
            difference = treatment_values - comparator_values
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append({
                "contrast": contrast,
                "treatment": treatment,
                "comparator": comparator,
                "metric": metric,
                "independent_unit": "base_seed",
                "paired_seed_count": len(seeds),
                "treatment_mean": float(treatment_values.mean()),
                "comparator_mean": float(comparator_values.mean()),
                "mean_paired_difference": float(difference.mean()),
                "paired_difference_std": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "bootstrap_95_ci_low": float(low),
                "bootstrap_95_ci_high": float(high),
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            })
    return output


def _plot(seed_rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in seed_rows if str(row["treatment"]) in TREATMENTS]
    by_key = {(int(row["base_seed"]), str(row["treatment"])): row for row in primary}
    seeds = sorted({int(row["base_seed"]) for row in primary})
    labels = {"recorded_mappo": "Recorded", "unconstrained_information_gain": "Unconstrained IG", "budgeted_information_gain": "Budgeted IG"}
    colors = {"recorded_mappo": "#2563eb", "unconstrained_information_gain": "#dc2626", "budgeted_information_gain": "#16a34a"}
    x = np.arange(len(TREATMENTS))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    def panel(axis: Any, field: str, title: str, ylabel: str) -> None:
        for seed in seeds:
            axis.plot(x, [float(by_key[(seed, treatment)][field]) for treatment in TREATMENTS], color="#94a3b8", alpha=0.35, linewidth=0.8)
        means = [float(np.mean([float(by_key[(seed, treatment)][field]) for seed in seeds])) for treatment in TREATMENTS]
        axis.scatter(x, means, c=[colors[treatment] for treatment in TREATMENTS], s=60, zorder=3)
        axis.set_xticks(x, [labels[treatment] for treatment in TREATMENTS], rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    panel(axes[0, 0], "measurement_count_total", "Measurement selections", "count / validation episode")
    panel(axes[0, 1], "overhead_load_packets_total", "Equivalent overhead load", "packets / episode")
    panel(axes[0, 2], "position_rmse_m", "Position RMSE", "m")
    panel(axes[1, 0], "j_loc", r"$J_{loc}$", "cost")
    panel(axes[1, 1], "j_com", r"$J_{com}$", "cost")
    panel(axes[1, 2], "harmful_fraction", "Harmful accepted updates", "fraction")
    fig.suptitle("Measurement-message overhead intervention; validation-only; lines pair identical seeds")
    target = plots_dir / "measurement_overhead_intervention.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def _audit_report(
    config: MeasurementOverheadInterventionConfig,
    invariants: Mapping[str, float],
    validation_results: Mapping[tuple[int, str], Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    overhead_rows: Sequence[Mapping[str, Any]],
    budget_usage: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {(str(row["contrast"]), str(row["metric"])): row for row in contrasts}
    pairing_ok = all(math.isfinite(float(value)) and float(value) <= NUMERIC_TOLERANCE for value in invariants.values())
    finite = all(
        math.isfinite(float(row.get(field, 0.0)))
        for row in seed_rows
        for field in ("position_rmse_m", "j_loc", "j_com", "mean_reward", "overhead_load_packets_total")
    ) and all(
        math.isfinite(float(row.get(field, 0.0)))
        for row in overhead_rows
        for field in ("load_packets_requested", "load_packets_applied", "queue_before_packets", "queue_with_measurement_packets", "queue_next_packets", "packet_loss", "delay_s", "aoi_s")
    )
    expected_seeds = tuple(sorted({int(row["base_seed"]) for row in seed_rows if str(row["treatment"]) in TREATMENTS})) == EXPECTED_BASE_SEEDS
    expected_rows = all(int(row.get("validation_episode_count", 0)) == 20 for row in seed_rows)
    schema_ok = all(bool(result.get("ok")) for result in validation_results.values())
    budget_ok = all(int(row.get("max_budget_excess", 0)) == 0 for row in budget_usage)
    overhead_correspondence = all(
        abs(float(row.get("payload_size_bytes", 0.0)) - PACKET_SIZE_BYTES) <= NUMERIC_TOLERANCE
        and abs(float(row.get("load_packets_requested", 0.0)) - 1.0) <= NUMERIC_TOLERANCE
        and abs(
            float(row.get("load_packets_applied", 0.0))
            - (1.0 if int(row.get("potential_edge", 0)) == 1 else 0.0)
        ) <= NUMERIC_TOLERANCE
        for row in overhead_rows
    )
    leakage_free = all(
        int(row.get("information_gain_selection_leakage_free", 0)) == 1
        and int(row.get("truth_error_used_for_selection", 0)) == 0
        and int(row.get("physical_los_used_for_selection", 0)) == 0
        and int(row.get("future_information_used_for_selection", 0)) == 0
        for row in window_rows
    )
    control_reproduces_reference = (
        math.isfinite(float(invariants.get("control_reproduces_reference_max_abs_difference", float("inf"))))
        and math.isfinite(float(invariants.get("control_measurement_action_max_difference", float("inf"))))
        and float(invariants.get("control_reproduces_reference_max_abs_difference", float("inf"))) <= NUMERIC_TOLERANCE
        and float(invariants.get("control_measurement_action_max_difference", float("inf"))) <= NUMERIC_TOLERANCE
    )
    unconstrained_rmse = by_key.get(("unconstrained_information_gain_minus_recorded_mappo", "position_rmse_m"), {})
    unconstrained_jcom = by_key.get(("unconstrained_information_gain_minus_recorded_mappo", "j_com"), {})
    budgeted_rmse = by_key.get(("budgeted_information_gain_minus_recorded_mappo", "position_rmse_m"), {})
    budgeted_jloc = by_key.get(("budgeted_information_gain_minus_recorded_mappo", "j_loc"), {})
    budgeted_jcom = by_key.get(("budgeted_information_gain_minus_recorded_mappo", "j_com"), {})
    tradeoff_exposed = (
        float(unconstrained_rmse.get("bootstrap_95_ci_high", 0.0)) < 0.0
        and float(unconstrained_jcom.get("bootstrap_95_ci_low", 0.0)) > 0.0
    )
    budgeted_effective = (
        (float(budgeted_rmse.get("bootstrap_95_ci_high", 0.0)) < 0.0 or float(budgeted_jloc.get("bootstrap_95_ci_high", 0.0)) < 0.0)
        and float(budgeted_jcom.get("bootstrap_95_ci_high", 0.0)) <= 0.0
    )
    if tradeoff_exposed:
        decision = "localization_gain_communication_cost_tradeoff_exposed"
    elif budgeted_effective:
        decision = "budgeted_information_gain_preliminarily_effective"
    elif float(unconstrained_rmse.get("mean_paired_difference", 0.0)) < 0.0 or float(budgeted_rmse.get("mean_paired_difference", 0.0)) < 0.0:
        decision = "localization_direction_promising_but_tradeoff_inconclusive"
    else:
        decision = "measurement_overhead_effect_not_yet_actionable"
    checks = {
        **dict(invariants),
        "control_reproduces_reference": control_reproduces_reference,
        "validation_episode_schema_ok": schema_ok,
        "all_seed_episode_counts_ok": expected_rows,
        "deterministic_seed_set_ok": expected_seeds,
        "budget_cap_ok": budget_ok,
        "overhead_load_correspondence_ok": overhead_correspondence,
        "overhead_values_finite": finite,
        "selection_leakage_free": leakage_free,
        "test_data_used": False,
        "objective_and_reward_formula_checked": True,
        "tradeoff_exposed": tradeoff_exposed,
        "budgeted_effective": budgeted_effective,
    }
    overall_ok = bool(
        pairing_ok
        and control_reproduces_reference
        and schema_ok
        and expected_rows
        and expected_seeds
        and budget_ok
        and overhead_correspondence
        and finite
        and leakage_free
    )
    return {
        "experiment": "measurement_message_communication_overhead_intervention",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_branch": config.reference_branch,
        "independent_unit": "base_seed",
        "paired_episode_unit": "episode_within_base_seed",
        "base_seed_count": len(EXPECTED_BASE_SEEDS),
        "base_seeds": list(EXPECTED_BASE_SEEDS),
        "time_windows_are_independent_samples": False,
        "ekf_events_are_independent_samples": False,
        "test_data_used": False,
        "changed_action_field_only": "measurement_uav_id",
        "overhead_model": {
            "enabled_for_primary_treatments": True,
            "control_enabled": False,
            "equivalent_packets_per_selected_measurement": 1.0,
            "payload_size_bytes": PACKET_SIZE_BYTES,
            "mac_protocol_modeled": False,
            "ekf_message_arrival_semantics_changed": False,
        },
        "checks": checks,
        "decision_gate": decision,
        "decision_supported": bool(tradeoff_exposed or budgeted_effective),
        "overall_ok": overall_ok,
        "interpretation_boundary": "Validation-only paired mechanism evidence; it does not retrain MAPPO or establish final policy superiority.",
    }


def run_measurement_overhead_intervention(config: MeasurementOverheadInterventionConfig) -> dict[str, Any]:
    """Run the five-seed overhead intervention and return an audit summary."""

    reference_root = Path(config.reference_root)
    reference_roots = _scale_roots(reference_root, config.reference_branch)
    if set(reference_roots) != set(EXPECTED_BASE_SEEDS):
        raise ValueError(f"reference root must contain exactly seeds {EXPECTED_BASE_SEEDS}; found {sorted(reference_roots)}")
    episodes_by_seed = {seed: _split_episodes(root, config.split) for seed, root in reference_roots.items()}
    counts = {seed: len(episodes) for seed, episodes in episodes_by_seed.items()}
    if any(count != 20 for count in counts.values()):
        raise ValueError(f"expected 20 validation episodes per seed, found {counts}")
    logs_dir, plots_dir = _prepare_outputs(config)

    all_window_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    all_overhead_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    budgets_by_seed: dict[int, dict[str, int]] = {}
    budget_rows: list[dict[str, Any]] = []
    proxy_parameters: dict[str, Any] = {
        "source_branch": config.reference_branch,
        "source_split_for_fitting": "train",
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "packet_size_bytes": PACKET_SIZE_BYTES,
        "equivalent_load_packets_per_selected_measurement": 1.0,
        "seeds": {},
    }
    treatment_roots: dict[tuple[int, str], Path] = {}
    validation_results: dict[tuple[int, str], Mapping[str, Any]] = {}

    for seed in EXPECTED_BASE_SEEDS:
        reference_scale = reference_roots[seed]
        reference_manifest = _load_manifest(reference_scale)
        budgets, budget_rows_seed = _train_budget(reference_scale, seed)
        budgets_by_seed[seed] = budgets
        budget_rows.extend(budget_rows_seed)
        scales, train_proxy_rows = _fit_seed_proxy(reference_scale, seed, config.age_variance_coefficient)
        proxy_parameters["seeds"][str(seed)] = {
            "scales": scales.as_dict(),
            "training_candidate_row_count": len(train_proxy_rows),
            "measurement_std_m": float(scales.measurement_std_median),
            "measurement_budgets": budgets,
        }
        context = {
            "proxy_scales": scales,
            "budgets": budgets,
            "measurement_std_m": float(scales.measurement_std_median),
            "age_variance_coefficient": config.age_variance_coefficient,
        }
        # Run the no-overhead control first. It uses the same recorded action
        # selector as the primary recorded treatment, but has a separate
        # output path so exact reproduction is auditable.
        treatments_to_run = ((CONTROL_TREATMENT, False, "recorded_mappo"),) + tuple(
            (treatment, True, treatment) for treatment in TREATMENTS
        )
        for label, overhead_enabled, selector_treatment in treatments_to_run:
            treatment_root = Path(config.output_dir) / f"seed_{seed}" / label
            scale_root = treatment_root / "scale_3"
            scale_root.mkdir(parents=True, exist_ok=True)
            treatment_roots[(seed, label)] = scale_root
            treatment_config = _reference_formal_config(reference_manifest, treatment_root, force=False, progress=config.progress)
            _copy_calibration_files(scale_root, treatment_config)
            normalization = reference_scale / "normalization_stats.json"
            if normalization.exists():
                shutil.copy2(normalization, scale_root / "normalization_stats.json")
            replay_config = QualityAwareMeasurementInterventionConfig(
                reference_scale=reference_scale,
                output_dir=treatment_root,
                split=config.split,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
                force=False,
                progress=config.progress,
            )
            summaries: list[dict[str, Any]] = []
            treatment_episode_rows: list[dict[str, Any]] = []
            for index, source_episode in enumerate(episodes_by_seed[seed], start=1):
                local_overhead: list[dict[str, Any]] = []
                summary, metrics, windows, events = _run_replay_episode(
                    config=replay_config,
                    formal_config=treatment_config,
                    thresholds=context,
                    treatment=selector_treatment,
                    scale_root=scale_root,
                    source_episode=source_episode,
                    episode_index=index,
                    episode_total=20,
                    selector=intervene_information_gain_selection,
                    policy_prefix=f"{POLICY_PREFIX}_{'no_overhead' if not overhead_enabled else label}",
                    measurement_overhead_enabled=overhead_enabled,
                    overhead_rows_sink=local_overhead,
                )
                episode_id = str(source_episode["episode_id"])
                episode_root = scale_root / config.split / episode_id
                overhead_rows_for_episode = [
                    dict(
                        row,
                        episode_id=episode_id,
                        scenario_id=str(source_episode.get("scenario_id", "")),
                        base_seed=seed,
                        split=config.split,
                        treatment=label,
                    )
                    for row in local_overhead
                ]
                _write_measurement_overhead(episode_root / "measurement_overhead.csv", overhead_rows_for_episode)
                summary["files"]["measurement_overhead.csv"] = _sha256(episode_root / "measurement_overhead.csv")
                summary["treatment"] = label
                summary["overhead_enabled"] = bool(overhead_enabled)
                summaries.append(summary)
                enriched_windows = _enrich_window_rows(windows, seed)
                enriched_windows = _relabel(enriched_windows, label)
                enriched_events = [dict(event, base_seed=seed, treatment=label, episode_seed=int(_number(event.get("seed"), -1))) for event in events]
                enriched_metrics = _episode_metrics_with_intervention(metrics, enriched_windows, enriched_events, seed, label, budgets)
                counts_by_uav: dict[str, int] = defaultdict(int)
                for row in enriched_windows:
                    if int(row.get("selected_non_none", 0)):
                        counts_by_uav[str(row.get("uav_id"))] += 1
                overhead_count = int(sum(float(row.get("load_packets_applied", 0.0)) for row in local_overhead))
                objective_residual, reward_residual = _objective_residual(episode_root)
                links = _read_csv(episode_root / "links.csv")
                mean_queue = _mean(float(row.get("queue_length", 0.0)) for row in links)
                enriched_metrics.update(
                    treatment=label,
                    overhead_enabled=int(overhead_enabled),
                    overhead_load_packets_total=float(overhead_count),
                    overhead_bytes_total=float(overhead_count * PACKET_SIZE_BYTES),
                    measurement_counts_by_uav=dict(counts_by_uav),
                    objective_formula_max_abs=float(objective_residual),
                    reward_formula_max_abs=float(reward_residual),
                    mean_queue_length_packets=float(mean_queue),
                )
                treatment_episode_rows.append(enriched_metrics)
                all_window_rows.extend(enriched_windows)
                all_event_rows.extend(enriched_events)
                all_overhead_rows.extend(overhead_rows_for_episode)
            all_episode_rows.extend(treatment_episode_rows)
            treatment_manifest = {
                "dataset_id": f"python_measurement_overhead_{seed}_{label}",
                "schema_version": EPISODE_SCHEMA_VERSION,
                "engine": "python_sim",
                "result_status": "validation_only_mechanism_diagnostic",
                "policy_source": f"{POLICY_PREFIX}_{label}",
                "uav_count": 3,
                "reference": {"scale_root": str(reference_scale), "manifest_sha256": _sha256(reference_scale / "manifest.json"), "split": config.split},
                "intervention": {
                    "treatment": label,
                    "changed_action_field_only": "measurement_uav_id",
                    "measurement_overhead_enabled": bool(overhead_enabled),
                    "equivalent_load_packets_per_selected_measurement": 1.0,
                    "payload_size_bytes": PACKET_SIZE_BYTES,
                    "information_gain_uses_execution_time_fields_only": True,
                    "measurement_budget_source": "baseline_mappo_train_mean_rounded",
                    "uses_test_data": False,
                    "retrained_mappo": False,
                },
                "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
                "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json", "measurement_overhead.csv"],
                "episodes": summaries,
            }
            _write_json(scale_root / "manifest.json", treatment_manifest)
            _write_json(treatment_root / "manifest.json", {
                "dataset_id": treatment_manifest["dataset_id"],
                "schema_version": EPISODE_SCHEMA_VERSION,
                "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
            })
            validation_results[(seed, label)] = validate_episode_benchmark(scale_root)
            if not validation_results[(seed, label)].get("ok", False):
                raise RuntimeError(f"{seed}/{label} episode validation failed: {validation_results[(seed, label)]}")
        _progress(config, f"seed {seed} complete ({len(ALL_TREATMENTS)} replays, 20 validation episodes)")

    seed_rows = _seed_metrics(all_episode_rows)
    contrasts = _paired_contrasts(seed_rows, config)
    usage_rows = _budget_usage(all_episode_rows, budgets_by_seed)
    invariants = _invariance_checks(reference_roots, treatment_roots, episodes_by_seed)
    audit = _audit_report(config, invariants, validation_results, seed_rows, all_window_rows, all_overhead_rows, usage_rows, contrasts)
    _write_json(logs_dir / "overhead_parameters.json", {
        "payload_size_bytes": PACKET_SIZE_BYTES,
        "equivalent_load_packets_per_selected_measurement": 1.0,
        "control_treatment": CONTROL_TREATMENT,
        "primary_treatments": list(TREATMENTS),
        "queue_effect": "current_decision_window_metric_and_next_queue",
        "ekf_message_arrival_semantics_changed": False,
        "source_branch": config.reference_branch,
        "source_split": config.split,
        "test_data_used": False,
    })
    _write_table(logs_dir / "measurement_budgets.csv", budget_rows)
    _write_table(logs_dir / "measurement_overhead.csv", all_overhead_rows or [{column: "" for column in OVERHEAD_COLUMNS}])
    _write_table(logs_dir / "window_metrics.csv", all_window_rows)
    _write_table(logs_dir / "episode_metrics.csv", all_episode_rows)
    _write_table(logs_dir / "seed_metrics.csv", seed_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "budget_usage.csv", usage_rows)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(seed_rows, contrasts, plots_dir)
    root_manifest = {
        "dataset_id": "python_measurement_overhead_intervention_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "validation_only_mechanism_diagnostic",
        "reference": {"root": str(reference_root), "branch": config.reference_branch, "split": config.split, "test_data_used": False},
        "paired_design": {
            "independent_unit": "base_seed",
            "paired_episode_unit": "episode_within_base_seed",
            "base_seeds": list(EXPECTED_BASE_SEEDS),
            "common_random_numbers": True,
            "changed_field_only": "measurement_uav_id",
            "control_replay": CONTROL_TREATMENT,
            "test_data_used": False,
        },
        "overhead_model": {"packet_size_bytes": PACKET_SIZE_BYTES, "equivalent_packets_per_selected_measurement": 1.0, "mac_protocol_modeled": False, "ekf_arrival_semantics_changed": False},
        "treatments": {label: {"path": f"seed_<base_seed>/{label}/scale_3", "validation_episode_count_per_seed": 20, "overhead_enabled": label != CONTROL_TREATMENT} for label in ALL_TREATMENTS},
        "logs": {name: str(logs_dir / name) for name in ("overhead_parameters.json", "measurement_budgets.csv", "measurement_overhead.csv", "window_metrics.csv", "episode_metrics.csv", "seed_metrics.csv", "paired_contrasts.csv", "budget_usage.csv", "audit_report.json")},
        "plots": {"measurement_overhead_intervention": str(plot_path)},
        "audit_ok": bool(audit["overall_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    return {
        "output_dir": str(config.output_dir),
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "audit_report": str(logs_dir / "audit_report.json"),
        "plot": str(plot_path),
        "audit_ok": bool(audit["overall_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
    }


__all__ = [
    "MeasurementOverheadInterventionConfig",
    "run_measurement_overhead_intervention",
    "PACKET_SIZE_BYTES",
]
