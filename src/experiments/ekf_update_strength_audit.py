"""Read-only audit of cooperative relative-range EKF update strength."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .bandwidth_intervention import _action_invariance, _max_numeric_difference, load_recorded_actions
from .baseline import _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .localization_coupling_intervention import _mean, _position_diagnostics, _quantile
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


POLICY_SOURCE = "ekf_update_strength_audit"
_NUMERIC_TOLERANCE = 2e-6
_FORMULA_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EKFUpdateStrengthAuditConfig:
    reference_scale: Path
    output_dir: Path
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("EKF update-strength audit is validation-only")


def _progress(config: EKFUpdateStrengthAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[ekf-update-strength] {message}", flush=True)


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "ekf_update_strength"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "ekf_update_strength"


def _prepare_output(config: EKFUpdateStrengthAuditConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir else _default_logs_dir(Path(config.reference_scale))
    plots = Path(config.plots_dir) if config.plots_dir else _default_plots_dir(Path(config.reference_scale))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"audit output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    scale_root = root / "scale_3"
    scale_root.mkdir(parents=True, exist_ok=True)
    return scale_root, logs, plots


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    _write_csv(path, tuple(rows[0].keys()), [dict(row) for row in rows])


def _json_vector(value: Any) -> str:
    return json.dumps(np.asarray(value, dtype=float).reshape(-1).tolist(), separators=(",", ":"))


def _event_row(
    *,
    event: Mapping[str, Any],
    episode_id: str,
    scenario_id: str,
    seed: int,
    step: int,
    event_index: int,
) -> dict[str, Any]:
    before = np.asarray(event.get("state_before", []), dtype=float)
    after = np.asarray(event.get("state_after", []), dtype=float)
    correction = np.asarray(event.get("state_correction", []), dtype=float)
    state_closure = float(np.linalg.norm((after - before) - correction)) if before.size else 0.0
    effective_closure = abs(
        float(event["effective_measurement_variance_m2"])
        - (
            float(event["measurement_variance_m2"])
            + float(event["neighbor_direction_variance_m2"])
            + float(event["unknown_cross_bound_m2"])
        )
    )
    innovation_closure = abs(
        float(event["innovation_variance_m2"])
        - (
            float(event["local_direction_variance_m2"])
            + float(event["effective_measurement_variance_m2"])
        )
    )
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "split": "validation",
        "step": step,
        "event_index": event_index,
        "measurement_time_s": float(event["measurement_time_s"]),
        "source_uav_id": str(event["source_uav_id"]),
        "target_uav_id": str(event["target_uav_id"]),
        "los": int(bool(event["los"])),
        "accepted": int(bool(event["accepted"])),
        "rejection_reason": str(event.get("rejection_reason", "none")),
        "message_age_s": float(event["message_age_s"]),
        "message_delay_s": float(event["message_delay_s"]),
        "ci_weight": float(event["ci_weight"]),
        "measured_range_m": float(event["measured_range_m"]),
        "predicted_range_m": float(event["predicted_range_m"]),
        "nlos_bias_m": float(event["nlos_bias_m"]),
        "innovation_m": float(event["innovation_m"]),
        "nis": float(event["nis"]),
        "measurement_variance_m2": float(event["measurement_variance_m2"]),
        "local_direction_variance_m2": float(event["local_direction_variance_m2"]),
        "neighbor_direction_variance_m2": float(event["neighbor_direction_variance_m2"]),
        "unknown_cross_bound_m2": float(event["unknown_cross_bound_m2"]),
        "effective_measurement_variance_m2": float(event["effective_measurement_variance_m2"]),
        "innovation_variance_m2": float(event["innovation_variance_m2"]),
        "kalman_gain_norm": float(event["kalman_gain_norm"]),
        "state_correction_norm": float(event["state_correction_norm"]),
        "position_correction_norm_m": float(event["position_correction_norm_m"]),
        "position_correction_x_m": float(event["position_correction_x_m"]),
        "position_correction_y_m": float(event["position_correction_y_m"]),
        "position_correction_z_m": float(event["position_correction_z_m"]),
        "true_position_error_before_m": float(event["true_position_error_before_m"]),
        "true_position_error_after_m": float(event["true_position_error_after_m"]),
        "true_error_change_after_minus_before_m": float(
            event["true_position_error_after_m"] - event["true_position_error_before_m"]
        ),
        "position_covariance_trace_before_m2": float(event["position_covariance_trace_before_m2"]),
        "position_covariance_trace_after_linear_m2": float(event["position_covariance_trace_after_linear_m2"]),
        "position_covariance_trace_after_final_m2": float(event["position_covariance_trace_after_final_m2"]),
        "covariance_restored": int(bool(event["covariance_restored"])),
        "j_loc_before": float(event["j_loc_before"]),
        "j_loc_after": float(event["j_loc_after"]),
        "j_loc_change": float(event["j_loc_after"] - event["j_loc_before"]),
        "effective_variance_formula_residual": float(effective_closure),
        "innovation_variance_formula_residual": float(innovation_closure),
        "state_correction_formula_residual": float(state_closure),
        "state_before_json": _json_vector(before),
        "state_after_json": _json_vector(after),
        "state_correction_json": _json_vector(correction),
    }


def _run_episode(
    *,
    config: EKFUpdateStrengthAuditConfig,
    formal_config: Any,
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
    observations, _info = environment.reset(seed=seed, scenario_config=dict(scenario))
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
        policy_source=POLICY_SOURCE,
    )
    _progress(config, f"episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()
    event_rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    all_traces: list[float] = []
    all_nees: list[float] = []
    rewards_history: list[float] = []
    update_windows = 0

    for step, actions in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        rewards_history.extend(float(value) for value in rewards.values())
        timestamp = (step + 1) * environment.config.motion.dt_s
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            info=next_info,
            graph_traces=None,
            policy_source=POLICY_SOURCE,
            action_start=step * environment.config.motion.dt_s,
            timestamp=timestamp,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        errors, traces, nees = _position_diagnostics(environment)
        all_errors.extend(errors)
        all_traces.extend(traces)
        all_nees.extend(nees)
        events = list(next_info.get("relative_range_update_events", []))
        update_windows += int(bool(events))
        for event_index, event in enumerate(events):
            event_rows.append(
                _event_row(
                    event=event,
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    seed=seed,
                    step=step,
                    event_index=event_index,
                )
            )
        observations = next_observations
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=_mean(rewards_history),
        source_episode_id=episode_id,
        source_scenario_sha256=_sha256(source_root / "scenario.json"),
        audit_event_count=len(event_rows),
    )
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    corrections = [float(row["position_correction_norm_m"]) for row in event_rows]
    gains = [float(row["kalman_gain_norm"]) for row in event_rows]
    metrics.update(
        update_event_count=len(event_rows),
        update_window_count=update_windows,
        position_error_p95_m=_quantile(all_errors, 0.95),
        mean_position_covariance_trace_m2=_mean(all_traces),
        mean_position_nees=_mean(all_nees),
        mean_position_correction_m=_mean(corrections),
        median_position_correction_m=_quantile(corrections, 0.5),
        max_position_correction_m=max(corrections, default=0.0),
        mean_kalman_gain_norm=_mean(gains),
        mean_true_error_change_m=_mean(
            float(row["true_error_change_after_minus_before_m"]) for row in event_rows
        ),
        los_event_fraction=_mean(int(row["los"]) for row in event_rows),
        covariance_restored_fraction=_mean(int(row["covariance_restored"]) for row in event_rows),
        max_abs_j_loc_change=max((abs(float(row["j_loc_change"])) for row in event_rows), default=0.0),
    )
    _progress(
        config,
        f"episode {episode_index}/{episode_total} complete updates={len(event_rows)} "
        f"rmse={metrics['position_rmse_m']:.6f} elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, event_rows


def _audit(
    *,
    config: EKFUpdateStrengthAuditConfig,
    scale_root: Path,
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replay_max = 0.0
    action_max = 0.0
    measurement_actions_equal = True
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        replay = scale_root / config.split / episode_id
        replay_max = max(
            replay_max,
            _max_numeric_difference(
                replay / "estimates.csv", reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                replay / "rewards.csv", reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
            _max_numeric_difference(
                replay / "links.csv", reference / "links.csv",
                key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                value_columns=("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi"),
            ),
        )
        difference, equal = _action_invariance(replay / "actions.csv", reference / "actions.csv")
        action_max = max(action_max, difference)
        measurement_actions_equal &= equal

    finite = all(
        np.isfinite(float(row[key]))
        for row in event_rows
        for key in (
            "innovation_m", "nis", "measurement_variance_m2",
            "local_direction_variance_m2", "neighbor_direction_variance_m2",
            "unknown_cross_bound_m2", "effective_measurement_variance_m2",
            "innovation_variance_m2", "kalman_gain_norm",
            "position_correction_norm_m", "true_position_error_before_m",
            "true_position_error_after_m", "j_loc_before", "j_loc_after",
        )
    )
    nonnegative = all(
        float(row[key]) >= 0.0
        for row in event_rows
        for key in (
            "measurement_variance_m2", "local_direction_variance_m2",
            "neighbor_direction_variance_m2", "unknown_cross_bound_m2",
            "effective_measurement_variance_m2", "innovation_variance_m2",
            "kalman_gain_norm", "position_correction_norm_m",
        )
    )
    max_effective_residual = max(
        (float(row["effective_variance_formula_residual"]) for row in event_rows), default=float("inf")
    )
    max_innovation_residual = max(
        (float(row["innovation_variance_formula_residual"]) for row in event_rows), default=float("inf")
    )
    max_state_residual = max(
        (float(row["state_correction_formula_residual"]) for row in event_rows), default=float("inf")
    )
    episode_counts = np.asarray([int(row["update_event_count"]) for row in episode_rows], dtype=float)
    episode_coverage = float(np.mean(episode_counts > 0))
    median_events = float(np.median(episode_counts))
    total_windows = len(episode_rows) * int(_read_json(Path(config.reference_scale) / "manifest.json").get("config", {}).get("max_steps", 50))
    event_window_fraction = float(sum(int(row["update_window_count"]) for row in episode_rows) / max(1, total_windows))
    opportunity_sparse = bool(episode_coverage < 0.80 or median_events < 5.0 or event_window_fraction < 0.10)
    median_correction = _quantile((float(row["position_correction_norm_m"]) for row in event_rows), 0.5)
    corrections_small = bool(median_correction < 0.01)
    cross_ratios = [
        float(row["unknown_cross_bound_m2"]) / max(1e-12, float(row["effective_measurement_variance_m2"]))
        for row in event_rows
    ]
    median_cross_ratio = _quantile(cross_ratios, 0.5)
    cross_bound_dominant = bool(median_cross_ratio >= 0.50)
    max_j_loc_change = max((abs(float(row["j_loc_change"])) for row in event_rows), default=float("inf"))
    jloc_insensitive = bool(max_j_loc_change <= 1e-12)
    covariance_restore_fraction = _mean(int(row["covariance_restored"]) for row in event_rows)

    if opportunity_sparse:
        decision = "cooperative_update_opportunities_sparse_design_paired_density_intervention"
    elif corrections_small:
        decision = "relative_range_corrections_too_small_audit_variance_and_gate"
    elif jloc_insensitive:
        decision = "j_loc_structurally_insensitive_select_cross_correlation_aware_covariance_method"
    else:
        decision = "update_strength_and_objective_response_sufficient_compare_measurement_selection"

    invariance_ok = (
        replay_max <= _NUMERIC_TOLERANCE
        and action_max <= _NUMERIC_TOLERANCE
        and measurement_actions_equal
    )
    formula_ok = (
        finite and nonnegative
        and max_effective_residual <= _FORMULA_TOLERANCE
        and max_innovation_residual <= _FORMULA_TOLERANCE
        and max_state_residual <= _FORMULA_TOLERANCE
    )
    return {
        "experiment": "cooperative_ekf_update_strength_and_jloc_sensitivity_audit",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "time_steps_or_update_events_used_as_independent_samples": False,
        "test_data_used": False,
        "checks": {
            "reference_replay_max_abs_difference": replay_max,
            "action_max_abs_difference": action_max,
            "measurement_actions_identical": measurement_actions_equal,
            "episode_count": len(episode_rows),
            "total_update_events": len(event_rows),
            "episodes_with_updates": int(np.sum(episode_counts > 0)),
            "episode_update_coverage": episode_coverage,
            "median_updates_per_episode": median_events,
            "update_window_fraction": event_window_fraction,
            "median_position_correction_m": median_correction,
            "median_unknown_cross_bound_fraction": median_cross_ratio,
            "covariance_restored_fraction": covariance_restore_fraction,
            "max_abs_j_loc_change": max_j_loc_change,
            "max_effective_variance_formula_residual": max_effective_residual,
            "max_innovation_variance_formula_residual": max_innovation_residual,
            "max_state_correction_formula_residual": max_state_residual,
        },
        "cause_flags": {
            "cooperative_update_opportunities_sparse": opportunity_sparse,
            "relative_range_corrections_small": corrections_small,
            "unknown_cross_bound_dominant": cross_bound_dominant,
            "j_loc_insensitive_to_relative_range_update": jloc_insensitive,
        },
        "invariance_ok": invariance_ok,
        "formula_closure_ok": formula_ok,
        "decision_gate": decision,
        "overall_ok": bool(invariance_ok and formula_ok and len(event_rows) == 50),
        "interpretation_boundary": (
            "Update events are mechanism diagnostics nested within 20 episodes; "
            "they are not 50 independent statistical replicates."
        ),
    }


def _plot(
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    counts = [int(row["update_event_count"]) for row in episode_rows]
    axes[0, 0].bar(np.arange(1, len(counts) + 1), counts, color="#2563eb")
    axes[0, 0].set(title="Eligible updates per episode", xlabel="Episode", ylabel="Count")

    variance_labels = ("Measurement", "Neighbor", "Cross bound")
    variance_values = (
        [float(row["measurement_variance_m2"]) for row in event_rows],
        [float(row["neighbor_direction_variance_m2"]) for row in event_rows],
        [float(row["unknown_cross_bound_m2"]) for row in event_rows],
    )
    axes[0, 1].boxplot(variance_values, labels=variance_labels, showfliers=False)
    axes[0, 1].set(title="Effective variance decomposition", ylabel=r"Variance ($m^2$)")

    gains = [float(row["kalman_gain_norm"]) for row in event_rows]
    corrections = [float(row["position_correction_norm_m"]) for row in event_rows]
    colors = ["#16a34a" if int(row["los"]) else "#dc2626" for row in event_rows]
    axes[0, 2].scatter(gains, corrections, c=colors, alpha=0.75)
    axes[0, 2].set(title="Gain and position correction", xlabel="Kalman gain norm", ylabel="Correction (m)")

    before = [float(row["true_position_error_before_m"]) for row in event_rows]
    after = [float(row["true_position_error_after_m"]) for row in event_rows]
    for left, right in zip(before, after):
        axes[1, 0].plot((0, 1), (left, right), color="#94a3b8", alpha=0.45, linewidth=0.8)
    axes[1, 0].set_xticks((0, 1), ("Before", "After"))
    axes[1, 0].set(title="Truth error around update (diagnostic only)", ylabel="Position error (m)")

    cov_before = [float(row["position_covariance_trace_before_m2"]) for row in event_rows]
    cov_linear = [float(row["position_covariance_trace_after_linear_m2"]) for row in event_rows]
    cov_final = [float(row["position_covariance_trace_after_final_m2"]) for row in event_rows]
    axes[1, 1].boxplot(
        (cov_before, cov_linear, cov_final),
        labels=("Before", "Linear", "Final"),
        showfliers=False,
    )
    axes[1, 1].set(title="Covariance restoration", ylabel=r"Position covariance trace ($m^2$)")

    jloc_changes = [float(row["j_loc_change"]) for row in event_rows]
    axes[1, 2].scatter(np.arange(1, len(jloc_changes) + 1), jloc_changes, color="#7c3aed", s=18)
    axes[1, 2].axhline(0.0, color="#111827", linewidth=0.9)
    axes[1, 2].set(title=r"Per-update $J_{loc}$ response", xlabel="Update event", ylabel=r"$\Delta J_{loc}$")

    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    fig.suptitle("Cooperative EKF update-strength audit (20 paired validation episodes)")
    target = plots_dir / "ekf_update_strength.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_ekf_update_strength_audit(config: EKFUpdateStrengthAuditConfig) -> dict[str, Any]:
    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("the first update-strength audit requires three UAVs")
    episodes = [
        item for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen audit requires exactly 20 validation episodes, found {len(episodes)}")

    scale_root, logs_dir, plots_dir = _prepare_output(config)
    formal_config = _reference_formal_config(manifest, Path(config.output_dir), force=False, progress=config.progress)
    _copy_calibration_files(scale_root, formal_config)
    normalization = reference_scale / "normalization_stats.json"
    if normalization.exists():
        shutil.copy2(normalization, scale_root / "normalization_stats.json")

    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for index, source_episode in enumerate(episodes, start=1):
        summary, metrics, events = _run_episode(
            config=config,
            formal_config=formal_config,
            scale_root=scale_root,
            source_episode=source_episode,
            episode_index=index,
            episode_total=len(episodes),
        )
        summaries.append(summary)
        episode_rows.append(metrics)
        event_rows.extend(events)

    scale_manifest = {
        "dataset_id": "python_ekf_update_strength_audit_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "result_status": "engineering_mechanism_diagnostic_only",
        "policy_source": POLICY_SOURCE,
        "uav_count": 3,
        "reference": {
            "scale_root": str(reference_scale),
            "manifest_sha256": _sha256(reference_scale / "manifest.json"),
            "split": config.split,
        },
        "config": {**asdict(formal_config), "output_dir": str(formal_config.output_dir)},
        "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
        "episodes": summaries,
    }
    _write_json(scale_root / "manifest.json", scale_manifest)
    validation = validate_episode_benchmark(scale_root)
    if not validation["ok"]:
        raise RuntimeError(f"audit replay validation failed: {validation}")

    _write_table(logs_dir / "update_events.csv", event_rows)
    _write_table(logs_dir / "episode_summary.csv", episode_rows)
    audit = _audit(
        config=config,
        scale_root=scale_root,
        episodes=episodes,
        episode_rows=episode_rows,
        event_rows=event_rows,
    )
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(episode_rows, event_rows, plots_dir)
    root_manifest = {
        "dataset_id": "python_ekf_update_strength_audit_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_mechanism_diagnostic_only",
        "reference": scale_manifest["reference"],
        "design": {
            "independent_unit": "episode",
            "episode_count": len(episodes),
            "update_events_are_independent_replicates": False,
            "filter_or_reward_changed": False,
            "test_data_used": False,
        },
        "scale": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json"},
        "logs": {
            "update_events": str(logs_dir / "update_events.csv"),
            "episode_summary": str(logs_dir / "episode_summary.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"ekf_update_strength": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"EKF update-strength audit failed: {audit['checks']}")
    return {
        "output_dir": str(config.output_dir),
        "reference_scale": str(reference_scale),
        "episode_count": len(episodes),
        "update_event_count": len(event_rows),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "decision_gate": audit["decision_gate"],
        "cause_flags": audit["cause_flags"],
        "audit_ok": True,
        "plot": str(plot_path),
    }


__all__ = ["EKFUpdateStrengthAuditConfig", "run_ekf_update_strength_audit"]
