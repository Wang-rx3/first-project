"""Paired validation intervention for cooperative EKF localization coupling.

Every reference episode is replayed with common random numbers and identical
recorded actions.  The sole intervention disables consumption of an otherwise
eligible delayed neighbor-state/UWB pair by the local EKF.  Communication,
message buffering, graph visibility, sensing and motion remain unchanged.
Episodes are the independent statistical units; window rows are diagnostics.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .bandwidth_intervention import (
    _action_invariance,
    _max_numeric_difference,
    load_recorded_actions,
)
from .baseline import _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


TREATMENTS = ("cooperative_enabled", "ekf_fusion_disabled")
POLICY_PREFIX = "localization_coupling_intervention"
_NUMERIC_TOLERANCE = 2e-6
_NEES_95_LOWER = 0.2158
_NEES_95_UPPER = 9.3484


@dataclass(frozen=True)
class LocalizationCouplingInterventionConfig:
    """Configuration for the validation-only paired coupling experiment."""

    reference_scale: Path
    output_dir: Path
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("localization coupling intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: LocalizationCouplingInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[localization-coupling] {message}", flush=True)


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "localization_coupling"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "localization_coupling"


def _prepare_output(config: LocalizationCouplingInterventionConfig) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir else _default_logs_dir(Path(config.reference_scale))
    plots = Path(config.plots_dir) if config.plots_dir else _default_plots_dir(Path(config.reference_scale))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"intervention output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float(default)


def _quantile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float(default)


def _position_diagnostics(environment: Any) -> tuple[list[float], list[float], list[float]]:
    errors: list[float] = []
    traces: list[float] = []
    nees: list[float] = []
    for uav_id in environment.uav_ids:
        delta = np.asarray(environment.states[uav_id].position_m, dtype=float) - np.asarray(
            environment.ekfs[uav_id].position, dtype=float
        )
        covariance = np.asarray(environment.ekfs[uav_id].covariance[:3, :3], dtype=float)
        errors.append(float(np.linalg.norm(delta)))
        traces.append(float(np.trace(covariance)))
        nees.append(float(delta @ np.linalg.pinv(covariance) @ delta))
    return errors, traces, nees


def _run_episode(
    *,
    config: LocalizationCouplingInterventionConfig,
    formal_config: Any,
    treatment: str,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    episode_id = str(source_episode["episode_id"])
    scenario_id = str(source_episode["scenario_id"])
    seed = int(source_episode["seed"])
    source_root = Path(config.reference_scale) / config.split / episode_id
    scenario_payload = _read_json(source_root / "scenario.json")
    scenario = scenario_payload.get("scenario_generation")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"missing scenario_generation in {source_root / 'scenario.json'}")
    recorded_windows = load_recorded_actions(source_root)
    enabled = treatment == "cooperative_enabled"

    _seed_everything(seed)
    environment = _make_environment(
        formal_config,
        seed,
        metric_scales=formal_config.metric_scales,
        cooperative_ekf_enabled=enabled,
    )
    observations, info = environment.reset(seed=seed, scenario_config=dict(scenario))
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / config.split / episode_id,
        episode_id=episode_id,
        scenario_id=scenario_id,
        split=config.split,
        seed=seed,
        phase=config.split,
        environment=environment,
        normalization_fitted=True,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=f"{POLICY_PREFIX}_{treatment}",
    )
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()
    window_rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    all_traces: list[float] = []
    all_nees: list[float] = []
    rewards_history: list[float] = []
    effective_edges_seen = 0
    max_neighbor_updates = 0

    for step, actions in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        if set(actions) != set(environment.uav_ids):
            raise ValueError(f"recorded UAV set mismatch in {episode_id} step {step}")
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        rewards_history.extend(float(value) for value in rewards.values())
        effective_edges_seen += len(next_info["central_state"]["effective_edges"])
        max_neighbor_updates = max(
            max_neighbor_updates,
            max(ekf.valid_neighbor_count for ekf in environment.ekfs.values()),
        )
        timestamp = (step + 1) * environment.config.motion.dt_s
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            info=next_info,
            graph_traces=None,
            policy_source=f"{POLICY_PREFIX}_{treatment}",
            action_start=step * environment.config.motion.dt_s,
            timestamp=timestamp,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        errors, traces, nees = _position_diagnostics(environment)
        all_errors.extend(errors)
        all_traces.extend(traces)
        all_nees.extend(nees)
        diagnostics = next_info["ekf_diagnostics"]
        arrived = sum(
            int(event.get("status") == "arrived")
            for event in next_info["message_buffer"]["events"]
        )
        components = next_info["objective_components"]
        window_rows.append(
            {
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "seed": seed,
                "split": config.split,
                "treatment": treatment,
                "step": step,
                "timestamp": timestamp,
                "arrived_message_count": arrived,
                "ekf_update_opportunities": int(next_info["step_cooperative_ekf_update_opportunities"]),
                "ekf_updates_blocked": int(next_info["step_cooperative_ekf_updates_blocked"]),
                "uwb_updates_attempted": sum(int(item["uwb_update_attempted"]) for item in diagnostics.values()),
                "uwb_updates_accepted": sum(int(item["uwb_update_accepted"]) for item in diagnostics.values()),
                "uwb_updates_rejected": sum(int(item["uwb_update_rejected"]) for item in diagnostics.values()),
                "neighbor_updates_cumulative": sum(ekf.valid_neighbor_count for ekf in environment.ekfs.values()),
                "mean_position_error_m": _mean(errors),
                "max_position_error_m": max(errors, default=0.0),
                "mean_position_covariance_trace_m2": _mean(traces),
                "mean_position_nees": _mean(nees),
                "nees_95_coverage": _mean(_NEES_95_LOWER <= value <= _NEES_95_UPPER for value in nees),
                "j_loc": float(components["j_loc"]),
                "j_com": float(components["j_com"]),
                "joint_objective": float(components["joint_objective"]),
            }
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=_mean(rewards_history),
        effective_edges_seen=effective_edges_seen,
        max_neighbor_updates=max_neighbor_updates,
        source_episode_id=episode_id,
        source_scenario_sha256=_sha256(source_root / "scenario.json"),
        treatment=treatment,
    )
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    metrics.update(
        treatment=treatment,
        position_error_p95_m=_quantile(all_errors, 0.95),
        mean_position_covariance_trace_m2=_mean(all_traces),
        mean_position_nees=_mean(all_nees),
        nees_95_coverage=_mean(_NEES_95_LOWER <= value <= _NEES_95_UPPER for value in all_nees),
        arrived_message_count=sum(int(row["arrived_message_count"]) for row in window_rows),
        ekf_update_opportunities=sum(int(row["ekf_update_opportunities"]) for row in window_rows),
        ekf_updates_blocked=sum(int(row["ekf_updates_blocked"]) for row in window_rows),
        uwb_updates_attempted=sum(int(row["uwb_updates_attempted"]) for row in window_rows),
        uwb_updates_accepted=sum(int(row["uwb_updates_accepted"]) for row in window_rows),
        uwb_updates_rejected=sum(int(row["uwb_updates_rejected"]) for row in window_rows),
    )
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"rmse={metrics['position_rmse_m']:.6f} updates={metrics['uwb_updates_accepted']} "
        f"elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, window_rows


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    _write_csv(path, tuple(rows[0].keys()), [dict(row) for row in rows])


def _bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]], config: LocalizationCouplingInterventionConfig
) -> list[dict[str, Any]]:
    metric_names = (
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "mean_position_covariance_trace_m2",
        "mean_position_nees",
        "nees_95_coverage",
        "uwb_updates_accepted",
        "j_com",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for metric in metric_names:
        enabled = np.asarray([float(by_key[(seed, "cooperative_enabled")][metric]) for seed in seeds])
        disabled = np.asarray([float(by_key[(seed, "ekf_fusion_disabled")][metric]) for seed in seeds])
        difference = disabled - enabled
        low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
        output.append(
            {
                "contrast": "ekf_fusion_disabled_minus_cooperative_enabled",
                "metric": metric,
                "paired_episodes": len(seeds),
                "cooperative_enabled_mean": float(enabled.mean()),
                "ekf_fusion_disabled_mean": float(disabled.mean()),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "paired_difference_std": float(difference.std(ddof=1)) if difference.size > 1 else 0.0,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            }
        )
    return output


def _audit(
    config: LocalizationCouplingInterventionConfig,
    treatment_roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    factual_reference_max = 0.0
    cross_truth_max = 0.0
    cross_obstacle_max = 0.0
    cross_link_max = 0.0
    cross_packet_max = 0.0
    cross_action_max = 0.0
    measurement_equal = True
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        enabled = treatment_roots["cooperative_enabled"] / config.split / episode_id
        disabled = treatment_roots["ekf_fusion_disabled"] / config.split / episode_id
        factual_reference_max = max(
            factual_reference_max,
            _max_numeric_difference(
                enabled / "estimates.csv",
                reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                enabled / "rewards.csv",
                reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
        )
        cross_truth_max = max(
            cross_truth_max,
            _max_numeric_difference(
                enabled / "ground_truth.csv", disabled / "ground_truth.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
        )
        cross_obstacle_max = max(
            cross_obstacle_max,
            _max_numeric_difference(
                enabled / "obstacles.csv", disabled / "obstacles.csv",
                key_columns=("timestamp", "obstacle_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz", "radius"),
            ),
        )
        cross_link_max = max(
            cross_link_max,
            _max_numeric_difference(
                enabled / "links.csv", disabled / "links.csv",
                key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                value_columns=("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length"),
            ),
        )
        cross_packet_max = max(
            cross_packet_max,
            _max_numeric_difference(
                enabled / "packet_events.csv", disabled / "packet_events.csv",
                key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                value_columns=("success", "delay", "aoi_after"),
            ),
        )
        action_difference, action_equal = _action_invariance(enabled / "actions.csv", disabled / "actions.csv")
        cross_action_max = max(cross_action_max, action_difference)
        measurement_equal &= action_equal

    by_treatment = {
        treatment: [row for row in episode_rows if row["treatment"] == treatment]
        for treatment in TREATMENTS
    }
    enabled_updates = sum(int(row["uwb_updates_accepted"]) for row in by_treatment["cooperative_enabled"])
    enabled_opportunities = sum(int(row["ekf_update_opportunities"]) for row in by_treatment["cooperative_enabled"])
    disabled_updates = sum(int(row["uwb_updates_accepted"]) for row in by_treatment["ekf_fusion_disabled"])
    disabled_blocked = sum(int(row["ekf_updates_blocked"]) for row in by_treatment["ekf_fusion_disabled"])
    contrast_map = {str(row["metric"]): row for row in contrasts}
    rmse = contrast_map["position_rmse_m"]
    enabled_rmse = float(rmse["cooperative_enabled_mean"])
    practical_threshold = max(0.05, 0.05 * enabled_rmse)
    rmse_difference = float(rmse["mean_paired_difference"])
    rmse_low = float(rmse["bootstrap_95_ci_low"])
    rmse_high = float(rmse["bootstrap_95_ci_high"])
    median_opportunities = float(np.median([int(row["ekf_update_opportunities"]) for row in by_treatment["cooperative_enabled"]]))
    if median_opportunities < 1.0:
        decision = "insufficient_cooperative_update_opportunities_strengthen_scenario"
    elif rmse_difference > practical_threshold and rmse_low > 0.0:
        decision = "cooperative_localization_coupling_verified_compare_measurement_selection"
    elif rmse_difference < -practical_threshold and rmse_high < 0.0:
        decision = "cooperative_fusion_harmful_audit_uwb_nlos_ci_cross_correlation"
    else:
        decision = "localization_response_too_small_audit_ekf_update_strength"

    invariant_values = (
        factual_reference_max, cross_truth_max, cross_obstacle_max,
        cross_link_max, cross_packet_max, cross_action_max,
    )
    deterministic_ok = all(value <= _NUMERIC_TOLERANCE for value in invariant_values) and measurement_equal
    update_gate = (
        enabled_opportunities > 0
        and enabled_updates > 0
        and disabled_updates == 0
        and disabled_blocked == enabled_opportunities
    )
    checks = {
        "enabled_reproduces_reference_max_abs_difference": factual_reference_max,
        "cross_treatment_ground_truth_max_abs_difference": cross_truth_max,
        "cross_treatment_obstacle_max_abs_difference": cross_obstacle_max,
        "cross_treatment_link_kpi_max_abs_difference": cross_link_max,
        "cross_treatment_packet_event_max_abs_difference": cross_packet_max,
        "cross_treatment_delta_v_max_abs_difference": cross_action_max,
        "measurement_actions_identical": measurement_equal,
        "enabled_update_opportunities": enabled_opportunities,
        "enabled_updates_accepted": enabled_updates,
        "disabled_updates_accepted": disabled_updates,
        "disabled_updates_blocked": disabled_blocked,
        "median_opportunities_per_episode": median_opportunities,
        "position_rmse_practical_threshold_m": practical_threshold,
        "position_rmse_disabled_minus_enabled": rmse_difference,
        "position_rmse_bootstrap_95_ci": [rmse_low, rmse_high],
    }
    return {
        "experiment": "paired_cooperative_localization_coupling_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "time_steps_used_as_independent_samples": False,
        "test_data_used": False,
        "checks": checks,
        "deterministic_pairing_ok": deterministic_ok,
        "update_intervention_ok": update_gate,
        "decision_gate": decision,
        "overall_ok": deterministic_ok and update_gate,
        "interpretation_boundary": (
            "This validation intervention isolates received-neighbor-state consumption by the "
            "Distributed EKF. It does not establish GAT/MAPPO superiority."
        ),
    }


def _plot(
    episode_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"cooperative_enabled": "#2563eb", "ekf_fusion_disabled": "#dc2626"}
    labels = {"cooperative_enabled": "Cooperative EKF", "ekf_fusion_disabled": "Fusion disabled"}
    metrics = (
        ("position_rmse_m", "Position RMSE (m)"),
        ("j_loc", r"$J_{loc}$"),
        ("mean_position_covariance_trace_m2", r"Mean position covariance trace ($m^2$)"),
        ("mean_position_nees", "Mean position NEES"),
        ("nees_95_coverage", "NEES 95% coverage"),
        ("uwb_updates_accepted", "Accepted cooperative updates"),
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
    seeds = sorted({int(row["seed"]) for row in episode_rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for seed in seeds:
            values = [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS]
            axis.plot((0, 1), values, color="#94a3b8", linewidth=0.8, alpha=0.45)
        means = [
            _mean(float(by_key[(seed, treatment)][metric]) for seed in seeds)
            for treatment in TREATMENTS
        ]
        axis.scatter((0, 1), means, c=[colors[item] for item in TREATMENTS], s=55, zorder=3)
        axis.set_xticks((0, 1), ("Enabled", "Disabled"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if metric == "mean_position_nees":
            axis.axhspan(_NEES_95_LOWER, _NEES_95_UPPER, color="#16a34a", alpha=0.08)
    fig.suptitle("Paired Moderate validation episodes; lines connect identical seeds")
    target = plots_dir / "localization_coupling.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_localization_coupling_intervention(
    config: LocalizationCouplingInterventionConfig,
) -> dict[str, Any]:
    """Run the paired validation-only cooperative localization intervention."""

    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("the first localization coupling intervention requires three UAVs")
    episodes = [
        item for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen intervention requires exactly 20 validation episodes, found {len(episodes)}")

    logs_dir, plots_dir = _prepare_output(config)
    formal_config = _reference_formal_config(
        manifest, Path(config.output_dir), force=False, progress=config.progress
    )
    treatment_roots: dict[str, Path] = {}
    all_episode_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    treatment_manifests: dict[str, Any] = {}
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        treatment_roots[treatment] = scale_root
        treatment_config = _reference_formal_config(
            manifest, treatment_root, force=False, progress=config.progress
        )
        _copy_calibration_files(scale_root, treatment_config)
        normalization = reference_scale / "normalization_stats.json"
        if normalization.exists():
            shutil.copy2(normalization, scale_root / "normalization_stats.json")
        summaries: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metric, windows = _run_episode(
                config=config,
                formal_config=formal_config,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            metrics.append(metric)
            all_window_rows.extend(windows)
        all_episode_rows.extend(metrics)
        treatment_manifest = {
            "dataset_id": f"python_localization_coupling_{treatment}",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "engine": "python_sim",
            "result_status": "engineering_mechanism_diagnostic_only",
            "policy_source": f"{POLICY_PREFIX}_{treatment}",
            "uav_count": 3,
            "reference": {
                "scale_root": str(reference_scale),
                "manifest_sha256": _sha256(reference_scale / "manifest.json"),
                "split": config.split,
            },
            "intervention": {
                "treatment": treatment,
                "cooperative_ekf_enabled": treatment == "cooperative_enabled",
                "fixed_fields": ["scenario", "seed", "actions", "communication", "sensors", "motion"],
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", treatment_manifest)
        _write_json(
            treatment_root / "manifest.json",
            {
                "dataset_id": treatment_manifest["dataset_id"],
                "schema_version": EPISODE_SCHEMA_VERSION,
                "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
            },
        )
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {
            "scale_root": str(scale_root),
            "manifest": str(scale_root / "manifest.json"),
            "validation_ok": True,
        }

    contrasts = _paired_contrasts(all_episode_rows, config)
    audit = _audit(config, treatment_roots, episodes, all_episode_rows, contrasts)
    _write_table(logs_dir / "window_metrics.csv", all_window_rows)
    _write_table(logs_dir / "episode_metrics.csv", all_episode_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(all_episode_rows, all_window_rows, plots_dir)
    root_manifest = {
        "dataset_id": "python_localization_coupling_moderate_validation_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_mechanism_diagnostic_only",
        "reference": {
            "scale_root": str(reference_scale),
            "manifest_sha256": _sha256(reference_scale / "manifest.json"),
            "split": config.split,
            "episode_count": len(episodes),
        },
        "paired_design": {
            "independent_unit": "episode",
            "paired_by": ["scenario_id", "seed"],
            "time_steps_are_independent_replicates": False,
            "common_random_numbers": True,
            "changed_mechanism_only": "cooperative_ekf_neighbor_fusion",
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "test_data_used": False,
        },
        "treatments": treatment_manifests,
        "logs": {
            "window_metrics": str(logs_dir / "window_metrics.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"localization_coupling": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"localization coupling audit failed: {audit['checks']}")
    return {
        "output_dir": str(config.output_dir),
        "reference_scale": str(reference_scale),
        "paired_episodes": len(episodes),
        "treatments": list(TREATMENTS),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "decision_gate": audit["decision_gate"],
        "audit_ok": True,
        "plot": str(plot_path),
    }


__all__ = [
    "LocalizationCouplingInterventionConfig",
    "TREATMENTS",
    "run_localization_coupling_intervention",
]
