"""Paired validation intervention for cooperative-measurement selection density.

The factual treatment replays the frozen MAPPO actions.  The comparator keeps
the recorded bandwidth and velocity increments but deterministically rotates
through measurement candidates visible in the actor observation before each
step.  No future packet success, message arrival, ground truth, or test data
is used.  Episodes, rather than windows, are the independent statistical
units.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _write_csv, _write_json
from .bandwidth_intervention import _max_numeric_difference, load_recorded_actions
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .localization_coupling_intervention import (
    _NEES_95_LOWER,
    _NEES_95_UPPER,
    _bootstrap_interval,
    _mean,
    _position_diagnostics,
    _quantile,
    _write_table,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


TREATMENTS = ("recorded_mappo", "round_robin_candidate")
POLICY_PREFIX = "measurement_density_intervention"
_NUMERIC_TOLERANCE = 2e-6


@dataclass(frozen=True)
class MeasurementDensityInterventionConfig:
    """Configuration for the validation-only paired density intervention."""

    reference_scale: Path
    output_dir: Path
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    minimum_selection_fraction: float = 0.80
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("measurement density intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 <= self.minimum_selection_fraction <= 1.0:
            raise ValueError("minimum_selection_fraction must lie in [0, 1]")


def _progress(config: MeasurementDensityInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[measurement-density] {message}", flush=True)


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "measurement_density"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "measurement_density"


def _prepare_output(config: MeasurementDensityInterventionConfig) -> tuple[Path, Path]:
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


def actor_visible_measurement_candidates(
    observation: Mapping[str, Any], uav_ids: Sequence[str]
) -> list[str]:
    """Return sorted candidates encoded in the current actor-visible mask."""

    mask = np.asarray(observation.get("action_mask", ()), dtype=bool).reshape(-1)
    candidates = [
        str(uav_id)
        for index, uav_id in enumerate(uav_ids)
        if index < mask.size and bool(mask[index])
    ]
    declared = observation.get("candidate_measurement_sources")
    if declared is not None:
        declared_set = {str(value) for value in declared}
        candidates = [value for value in candidates if value in declared_set]
    return sorted(set(candidates))


def intervene_measurement_selection(
    recorded_actions: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    treatment: str,
    round_robin_cursors: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Change only measurement selection using information visible before step."""

    if treatment not in TREATMENTS:
        raise ValueError(f"unknown measurement-density treatment: {treatment}")
    actions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        recorded = recorded_actions[uav_id]
        candidates = actor_visible_measurement_candidates(observations[uav_id], uav_ids)
        recorded_source = str(recorded.get("measurement_uav_id", "none"))
        selected_source = recorded_source
        if treatment == "round_robin_candidate":
            if candidates:
                cursor = int(round_robin_cursors.get(uav_id, 0))
                selected_source = candidates[cursor % len(candidates)]
                round_robin_cursors[uav_id] = cursor + 1
            else:
                selected_source = "none"
        actions[uav_id] = {
            "measurement_uav_id": selected_source,
            "bandwidth": {
                str(target): float(value)
                for target, value in dict(recorded.get("bandwidth", {})).items()
            },
            "delta_v": np.asarray(recorded["delta_v"], dtype=float).reshape(3),
        }
        metadata[uav_id] = {
            "candidate_count": len(candidates),
            "candidate_uav_ids": candidates,
            "candidate_available": bool(candidates),
            "recorded_measurement_uav_id": recorded_source,
            "selected_measurement_uav_id": selected_source,
            "selected_non_none": selected_source != "none",
            "selection_changed": selected_source != recorded_source,
            "actor_visible_only": True,
        }
    return actions, metadata


def _gate_row(
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
    split: str,
    treatment: str,
    step: int,
    timestamp: float,
    uav_id: str,
    metadata: Mapping[str, Any],
    gate_event: Mapping[str, Any] | None,
    environment: Any,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    event = dict(gate_event or {})
    delta = np.asarray(environment.states[uav_id].position_m, dtype=float) - np.asarray(
        environment.ekfs[uav_id].position, dtype=float
    )
    covariance = np.asarray(environment.ekfs[uav_id].covariance[:3, :3], dtype=float)
    nees = float(delta @ np.linalg.pinv(covariance) @ delta)
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": split,
        "treatment": treatment,
        "step": step,
        "timestamp": timestamp,
        "uav_id": uav_id,
        "candidate_count": int(metadata["candidate_count"]),
        "candidate_available": int(bool(metadata["candidate_available"])),
        "recorded_measurement_uav_id": str(metadata["recorded_measurement_uav_id"]),
        "selected_measurement_uav_id": str(metadata["selected_measurement_uav_id"]),
        "selected_non_none": int(bool(metadata["selected_non_none"])),
        "selection_changed": int(bool(metadata["selection_changed"])),
        "uwb_valid": int(bool(event.get("uwb_valid", False))),
        "effective_edge": int(bool(event.get("effective_edge", False))),
        "message_arrived": int(bool(event.get("message_arrived", False))),
        "update_period_passed": int(bool(event.get("update_period_passed", False))),
        "new_packet": int(bool(event.get("new_packet", False))),
        "ekf_update_attempted": int(bool(event.get("ekf_update_attempted", False))),
        "ekf_update_accepted": int(bool(event.get("ekf_update_accepted", False))),
        "failure_stage": str(
            event.get(
                "failure_stage",
                "no_candidate" if not metadata["candidate_available"] else "measurement_none",
            )
        ),
        "position_error_m": float(np.linalg.norm(delta)),
        "position_covariance_trace_m2": float(np.trace(covariance)),
        "position_nees": nees,
        "j_loc": float(components["j_loc"]),
        "j_com": float(components["j_com"]),
        "joint_objective": float(components["joint_objective"]),
    }


def _run_episode(
    *,
    config: MeasurementDensityInterventionConfig,
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

    _seed_everything(seed)
    environment = _make_environment(
        formal_config,
        seed,
        metric_scales=formal_config.metric_scales,
        cooperative_ekf_enabled=True,
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
    cursors = {uav_id: 0 for uav_id in environment.uav_ids}
    rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    all_traces: list[float] = []
    all_nees: list[float] = []
    rewards_history: list[float] = []
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()

    for step, recorded in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        actions, selection_metadata = intervene_measurement_selection(
            recorded,
            observations,
            environment.uav_ids,
            treatment,
            cursors,
        )
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
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
        rewards_history.extend(float(value) for value in rewards.values())
        errors, traces, nees = _position_diagnostics(environment)
        all_errors.extend(errors)
        all_traces.extend(traces)
        all_nees.extend(nees)
        gate_by_target = {
            str(event["target_uav_id"]): event
            for event in next_info.get("measurement_gate_events", [])
        }
        components = next_info["objective_components"]
        for uav_id in environment.uav_ids:
            rows.append(
                _gate_row(
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    seed=seed,
                    split=config.split,
                    treatment=treatment,
                    step=step,
                    timestamp=timestamp,
                    uav_id=uav_id,
                    metadata=selection_metadata[uav_id],
                    gate_event=gate_by_target.get(uav_id),
                    environment=environment,
                    components=components,
                )
            )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=_mean(rewards_history),
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
        actor_rows=len(rows),
        candidate_available_rows=sum(int(row["candidate_available"]) for row in rows),
        selected_non_none_rows=sum(int(row["selected_non_none"]) for row in rows),
        selected_with_candidate_rows=sum(
            int(row["candidate_available"] and row["selected_non_none"]) for row in rows
        ),
        selection_fraction_given_candidate=(
            sum(int(row["candidate_available"] and row["selected_non_none"]) for row in rows)
            / max(1, sum(int(row["candidate_available"]) for row in rows))
        ),
        uwb_valid_count=sum(int(row["uwb_valid"]) for row in rows),
        effective_edge_count=sum(int(row["effective_edge"]) for row in rows),
        message_arrived_count=sum(int(row["message_arrived"]) for row in rows),
        update_period_passed_count=sum(int(row["update_period_passed"]) for row in rows),
        new_packet_count=sum(int(row["new_packet"]) for row in rows),
        ekf_updates_attempted=sum(int(row["ekf_update_attempted"]) for row in rows),
        ekf_updates_accepted=sum(int(row["ekf_update_accepted"]) for row in rows),
    )
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"selection={metrics['selection_fraction_given_candidate']:.3f} "
        f"updates={metrics['ekf_updates_accepted']} rmse={metrics['position_rmse_m']:.6f} "
        f"elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, rows


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]], config: MeasurementDensityInterventionConfig
) -> list[dict[str, Any]]:
    metrics = (
        "selection_fraction_given_candidate",
        "ekf_updates_accepted",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "mean_position_covariance_trace_m2",
        "mean_position_nees",
        "nees_95_coverage",
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
    for metric in metrics:
        factual = np.asarray([float(by_key[(seed, "recorded_mappo")][metric]) for seed in seeds])
        intervention = np.asarray(
            [float(by_key[(seed, "round_robin_candidate")][metric]) for seed in seeds]
        )
        difference = intervention - factual
        low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
        output.append(
            {
                "contrast": "round_robin_candidate_minus_recorded_mappo",
                "metric": metric,
                "paired_episodes": len(seeds),
                "recorded_mappo_mean": float(factual.mean()),
                "round_robin_candidate_mean": float(intervention.mean()),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "paired_difference_std": float(difference.std(ddof=1)) if difference.size > 1 else 0.0,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            }
        )
    return output


def _gate_funnel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stages = (
        ("candidate_available", "candidate_available"),
        ("selected_non_none", "selected_non_none"),
        ("uwb_valid", "uwb_valid"),
        ("effective_edge", "effective_edge"),
        ("message_arrived", "message_arrived"),
        ("update_period_passed", "update_period_passed"),
        ("new_packet", "new_packet"),
        ("ekf_update_attempted", "ekf_update_attempted"),
        ("ekf_update_accepted", "ekf_update_accepted"),
    )
    output: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        selected = [row for row in rows if row["treatment"] == treatment]
        previous = len(selected)
        for order, (stage, column) in enumerate(stages, start=1):
            passed = sum(int(row[column]) for row in selected)
            output.append(
                {
                    "treatment": treatment,
                    "stage_order": order,
                    "stage": stage,
                    "actor_rows": len(selected),
                    "passed_count": passed,
                    "fraction_of_all_rows": passed / max(1, len(selected)),
                    "fraction_of_previous_stage": passed / max(1, previous),
                }
            )
            previous = passed
    return output


def _bandwidth_difference(first: Path, second: Path) -> float:
    left = _read_csv(first)
    right = _read_csv(second)
    keys = ("timestamp", "uav_id")
    left_map = {tuple(row[key] for key in keys): row for row in left}
    right_map = {tuple(row[key] for key in keys): row for row in right}
    if left_map.keys() != right_map.keys():
        return float("inf")
    maximum = 0.0
    for key in left_map:
        left_values = json.loads(left_map[key]["bandwidth_allocations_json"])
        right_values = json.loads(right_map[key]["bandwidth_allocations_json"])
        if set(left_values) != set(right_values):
            return float("inf")
        for target in left_values:
            maximum = max(maximum, abs(float(left_values[target]) - float(right_values[target])))
    return maximum


def _audit(
    config: MeasurementDensityInterventionConfig,
    treatment_roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    factual_reference_max = 0.0
    truth_max = obstacle_max = link_max = packet_max = delta_v_max = bandwidth_max = 0.0
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        factual = treatment_roots["recorded_mappo"] / config.split / episode_id
        comparator = treatment_roots["round_robin_candidate"] / config.split / episode_id
        factual_reference_max = max(
            factual_reference_max,
            _max_numeric_difference(
                factual / "estimates.csv", reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                factual / "rewards.csv", reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
        )
        truth_max = max(truth_max, _max_numeric_difference(
            factual / "ground_truth.csv", comparator / "ground_truth.csv",
            key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz"),
        ))
        obstacle_max = max(obstacle_max, _max_numeric_difference(
            factual / "obstacles.csv", comparator / "obstacles.csv",
            key_columns=("timestamp", "obstacle_id"), value_columns=("x", "y", "z", "vx", "vy", "vz", "radius"),
        ))
        link_max = max(link_max, _max_numeric_difference(
            factual / "links.csv", comparator / "links.csv",
            key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
            value_columns=("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length"),
        ))
        packet_max = max(packet_max, _max_numeric_difference(
            factual / "packet_events.csv", comparator / "packet_events.csv",
            key_columns=("timestamp", "src_uav_id", "dst_uav_id"), value_columns=("success", "delay", "aoi_after"),
        ))
        delta_v_max = max(delta_v_max, _max_numeric_difference(
            factual / "actions.csv", comparator / "actions.csv",
            key_columns=("timestamp", "uav_id"), value_columns=("delta_vx", "delta_vy", "delta_vz"),
        ))
        bandwidth_max = max(bandwidth_max, _bandwidth_difference(
            factual / "actions.csv", comparator / "actions.csv"
        ))

    contrast_map = {str(row["metric"]): row for row in contrasts}
    selection = contrast_map["selection_fraction_given_candidate"]
    updates = contrast_map["ekf_updates_accepted"]
    rmse = contrast_map["position_rmse_m"]
    j_loc = contrast_map["j_loc"]
    round_robin_selection = float(selection["round_robin_candidate_mean"])
    update_difference = float(updates["mean_paired_difference"])
    update_low = float(updates["bootstrap_95_ci_low"])
    factual_updates = float(updates["recorded_mappo_mean"])
    update_ratio = float(updates["round_robin_candidate_mean"]) / max(1e-12, factual_updates)
    rmse_difference = float(rmse["mean_paired_difference"])
    rmse_low = float(rmse["bootstrap_95_ci_low"])
    rmse_high = float(rmse["bootstrap_95_ci_high"])
    factual_rmse = float(rmse["recorded_mappo_mean"])
    practical_threshold = max(0.05, 0.05 * factual_rmse)
    j_loc_difference = float(j_loc["mean_paired_difference"])

    updates_increased = update_difference > 0.0 and update_low > 0.0 and update_ratio >= 2.0
    rmse_improved = rmse_difference < -practical_threshold and rmse_high < 0.0
    rmse_worsened = rmse_difference > practical_threshold and rmse_low > 0.0
    j_loc_insensitive = abs(j_loc_difference) <= 1e-12
    if not updates_increased:
        decision = "selection_changed_but_updates_not_increased_use_gate_funnel"
    elif rmse_improved and j_loc_insensitive:
        decision = "sparse_selection_contributes_but_j_loc_structurally_insensitive"
    elif rmse_improved:
        decision = "measurement_density_improves_localization_compare_quality_aware_selection"
    elif rmse_worsened:
        decision = "always_measurement_harmful_design_quality_aware_measurement_mask"
    else:
        decision = "updates_increased_but_localization_response_small_audit_measurement_quality"

    finite = all(
        np.isfinite(float(row[key]))
        for row in episode_rows
        for key in (
            "position_rmse_m", "position_error_p95_m", "j_loc",
            "mean_position_covariance_trace_m2", "mean_position_nees",
        )
    )
    # Changing only ``measurement_uav_id`` is intentionally allowed to alter
    # queue state, packet outcomes, and downstream link KPIs.  Treating those
    # fields as cross-treatment invariants would reject the causal pathway
    # this intervention is meant to measure.  The exogenous scene and every
    # non-intervened control action remain strict invariants.
    invariants = (
        factual_reference_max, truth_max, obstacle_max, delta_v_max, bandwidth_max,
    )
    pairing_ok = all(value <= _NUMERIC_TOLERANCE for value in invariants)
    selection_ok = round_robin_selection >= config.minimum_selection_fraction
    checks = {
        "recorded_reproduces_reference_max_abs_difference": factual_reference_max,
        "cross_treatment_ground_truth_max_abs_difference": truth_max,
        "cross_treatment_obstacle_max_abs_difference": obstacle_max,
        "cross_treatment_link_kpi_max_abs_difference": link_max,
        "cross_treatment_packet_event_max_abs_difference": packet_max,
        "downstream_link_kpi_difference_expected": True,
        "downstream_packet_event_difference_expected": True,
        "cross_treatment_delta_v_max_abs_difference": delta_v_max,
        "cross_treatment_bandwidth_max_abs_difference": bandwidth_max,
        "round_robin_selection_fraction_given_candidate": round_robin_selection,
        "minimum_selection_fraction": config.minimum_selection_fraction,
        "accepted_update_mean_paired_difference": update_difference,
        "accepted_update_bootstrap_95_ci": [update_low, float(updates["bootstrap_95_ci_high"])],
        "accepted_update_ratio": update_ratio,
        "position_rmse_mean_paired_difference_m": rmse_difference,
        "position_rmse_bootstrap_95_ci_m": [rmse_low, rmse_high],
        "position_rmse_practical_threshold_m": practical_threshold,
        "j_loc_mean_paired_difference": j_loc_difference,
    }
    return {
        "experiment": "paired_cooperative_measurement_selection_density_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "paired_episode_count": len(episodes),
        "time_steps_used_as_independent_samples": False,
        "test_data_used": False,
        "future_information_used_by_round_robin": False,
        "checks": checks,
        "deterministic_pairing_ok": pairing_ok,
        "round_robin_selection_coverage_ok": selection_ok,
        "all_localization_metrics_finite": finite,
        "decision_gate": decision,
        "overall_ok": bool(pairing_ok and selection_ok and finite),
        "interpretation_boundary": (
            "This validation intervention tests an executable actor-visible measurement-selection "
            "baseline. It does not retrain MAPPO or establish policy superiority."
        ),
    }


def _plot(
    episode_rows: Sequence[Mapping[str, Any]],
    funnel_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"recorded_mappo": "Recorded MAPPO", "round_robin_candidate": "Round-robin"}
    colors = {"recorded_mappo": "#2563eb", "round_robin_candidate": "#dc2626"}
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
    seeds = sorted({int(row["seed"]) for row in episode_rows})
    panels = (
        ("selection_fraction_given_candidate", "Non-none selection / candidate window"),
        ("ekf_updates_accepted", "Accepted cooperative EKF updates"),
        ("position_rmse_m", "Position RMSE (m)"),
        ("position_error_p95_m", "Position error P95 (m)"),
        ("j_loc", r"$J_{loc}$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel()[:5], panels):
        for seed in seeds:
            values = [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS]
            axis.plot((0, 1), values, color="#94a3b8", linewidth=0.8, alpha=0.45)
        means = [
            _mean(float(by_key[(seed, treatment)][metric]) for seed in seeds)
            for treatment in TREATMENTS
        ]
        axis.scatter((0, 1), means, c=[colors[item] for item in TREATMENTS], s=55, zorder=3)
        axis.set_xticks((0, 1), [labels[item] for item in TREATMENTS])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    axis = axes.ravel()[5]
    stages = [
        "candidate_available", "selected_non_none", "uwb_valid", "effective_edge",
        "message_arrived", "update_period_passed", "new_packet", "ekf_update_accepted",
    ]
    for treatment in TREATMENTS:
        rows = {str(row["stage"]): row for row in funnel_rows if row["treatment"] == treatment}
        values = [float(rows[stage]["fraction_of_all_rows"]) for stage in stages]
        axis.plot(range(len(stages)), values, marker="o", linewidth=1.8,
                  color=colors[treatment], label=labels[treatment])
    axis.set_xticks(range(len(stages)),
                    ("Candidate", "Selected", "UWB", "Effective", "Arrived", "Period", "New", "Accepted"),
                    rotation=35, ha="right")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Measurement gate funnel")
    axis.set_ylabel("Fraction of actor rows")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.suptitle("Paired Moderate validation episodes; lines connect identical seeds")
    target = plots_dir / "measurement_density_intervention.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_measurement_density_intervention(
    config: MeasurementDensityInterventionConfig,
) -> dict[str, Any]:
    """Run the paired validation-only measurement-density intervention."""

    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("the first measurement-density intervention requires three UAVs")
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
    treatment_manifests: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
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
        treatment_metrics: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metrics, windows = _run_episode(
                config=config,
                formal_config=formal_config,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            treatment_metrics.append(metrics)
            window_rows.extend(windows)
        episode_rows.extend(treatment_metrics)
        treatment_manifest = {
            "dataset_id": f"python_measurement_density_{treatment}",
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
                "changed_action_field_only": "measurement_uav_id",
                "round_robin_uses_actor_visible_candidates_only": treatment == "round_robin_candidate",
                "fixed_fields": ["scenario", "seed", "bandwidth", "delta_v", "communication", "sensors", "motion"],
                "uses_future_information": False,
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            # ``_reference_formal_config`` contains Windows ``Path`` objects
            # in nested dataclass fields.  Normalize the complete payload
            # before writing the intervention manifest so a validation-only
            # replay cannot fail after all episodes have already been run.
            "config": _jsonable({**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)}),
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", treatment_manifest)
        _write_json(treatment_root / "manifest.json", {
            "dataset_id": treatment_manifest["dataset_id"],
            "schema_version": EPISODE_SCHEMA_VERSION,
            "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
        })
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {
            "scale_root": str(scale_root),
            "manifest": str(scale_root / "manifest.json"),
            "validation_ok": True,
        }

    contrasts = _paired_contrasts(episode_rows, config)
    funnel = _gate_funnel(window_rows)
    audit = _audit(config, treatment_roots, episodes, episode_rows, contrasts)
    _write_table(logs_dir / "window_metrics.csv", window_rows)
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "gate_funnel.csv", funnel)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(episode_rows, funnel, plots_dir)
    root_manifest = {
        "dataset_id": "python_measurement_density_moderate_validation_seed2026",
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
            "changed_field_only": "measurement_uav_id",
            "round_robin_uses_actor_visible_candidates_only": True,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "test_data_used": False,
        },
        "treatments": treatment_manifests,
        "logs": {
            "window_metrics": str(logs_dir / "window_metrics.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "gate_funnel": str(logs_dir / "gate_funnel.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"measurement_density_intervention": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"measurement density audit failed: {audit['checks']}")
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
    "MeasurementDensityInterventionConfig",
    "TREATMENTS",
    "actor_visible_measurement_candidates",
    "intervene_measurement_selection",
    "run_measurement_density_intervention",
]
