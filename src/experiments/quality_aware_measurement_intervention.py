"""Paired validation intervention for actor-visible quality-aware measurements.

Thresholds and normalization ranges are fitted exclusively from the frozen
training split.  Validation episodes are replayed with common random numbers
under recorded MAPPO, actor-visible round-robin, and quality-aware selection.
Physical LOS/NLOS and ground-truth error are offline labels only.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .bandwidth_intervention import _max_numeric_difference, load_recorded_actions
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .cooperative_update_quality_audit import _quality_event_row
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
from .measurement_density_intervention import (
    _NUMERIC_TOLERANCE,
    _bandwidth_difference,
    _gate_row,
    actor_visible_measurement_candidates,
    intervene_measurement_selection,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


TREATMENTS = ("recorded_mappo", "round_robin_candidate", "quality_aware_candidate")
POLICY_PREFIX = "quality_aware_measurement_intervention"
_HIGH_IS_GOOD = ("rssi_dbm", "snr_db")
_LOW_IS_GOOD = ("packet_loss", "delay_s", "aoi_s", "queue_length_packets")
_RISK_FIELDS = _HIGH_IS_GOOD + _LOW_IS_GOOD + ("neighbor_covariance_trace_m2",)


@dataclass(frozen=True)
class QualityAwareMeasurementInterventionConfig:
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
            raise ValueError("quality-aware measurement intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: QualityAwareMeasurementInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[quality-aware-measurement] {message}", flush=True)


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "quality_aware_measurement"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "quality_aware_measurement"


def _prepare_output(config: QualityAwareMeasurementInterventionConfig) -> tuple[Path, Path]:
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


def _position_covariance_trace(row: Mapping[str, str]) -> float:
    covariance = np.asarray(json.loads(row["covariance_flat_json"]), dtype=float)
    side = int(round(np.sqrt(covariance.size)))
    covariance = covariance.reshape(side, side)
    return float(np.trace(covariance[:3, :3]))


def fit_training_quality_thresholds(reference_scale: Path) -> dict[str, Any]:
    """Fit all gates/scales from train only; no outcome label is read."""

    train_root = Path(reference_scale) / "train"
    episode_roots = sorted(path for path in train_root.glob("episode_*") if path.is_dir())
    if not episode_roots:
        raise ValueError(f"no training episodes found under {train_root}")
    values: dict[str, list[float]] = {field: [] for field in _HIGH_IS_GOOD + _LOW_IS_GOOD}
    covariance_values: list[float] = []
    candidate_rows = 0
    for episode_root in episode_roots:
        for row in _read_csv(episode_root / "links.csv"):
            if int(float(row["candidate_edge"])) != 1:
                continue
            candidate_rows += 1
            values["rssi_dbm"].append(float(row["rssi"]))
            values["snr_db"].append(float(row["snr"]))
            values["packet_loss"].append(float(row["packet_loss"]))
            values["delay_s"].append(float(row["delay"]))
            values["aoi_s"].append(float(row["aoi"]))
            values["queue_length_packets"].append(float(row["queue_length"]))
        for row in _read_csv(episode_root / "estimates.csv"):
            covariance_values.append(_position_covariance_trace(row))
    if candidate_rows == 0 or not covariance_values:
        raise ValueError("training split does not contain candidate links or covariance rows")
    values["neighbor_covariance_trace_m2"] = covariance_values
    quantiles: dict[str, dict[str, float]] = {}
    for field, field_values in values.items():
        array = np.asarray(field_values, dtype=float)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"non-finite or empty training values for {field}")
        q25, q75 = np.quantile(array, (0.25, 0.75))
        quantiles[field] = {"q25": float(q25), "q75": float(q75)}
    return {
        "source_split": "train",
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "training_episode_count": len(episode_roots),
        "candidate_link_row_count": candidate_rows,
        "covariance_row_count": len(covariance_values),
        "quantiles": quantiles,
        "eligibility": {
            "link_state": "connected",
            "rssi_dbm": ">= train q25",
            "snr_db": ">= train q25",
            "packet_loss": "<= train q75",
            "delay_s": "<= train q75",
            "aoi_s": "<= train q75",
            "queue_length_packets": "<= train q75",
            "neighbor_covariance_trace_m2": "<= train q75 when a received neighbor covariance exists",
        },
        "score": "mean of seven train-IQR normalized risk components; missing neighbor covariance receives risk 1",
    }


def _risk(value: float, field: str, thresholds: Mapping[str, Any]) -> float:
    limits = thresholds["quantiles"][field]
    q25, q75 = float(limits["q25"]), float(limits["q75"])
    width = max(abs(q75 - q25), 1e-12)
    if field in _HIGH_IS_GOOD:
        return float(np.clip((q75 - value) / width, 0.0, 1.0))
    return float(np.clip((value - q25) / width, 0.0, 1.0))


def _neighbor_covariance_trace(observation: Mapping[str, Any], source: str) -> float | None:
    messages = observation.get("neighbor_messages", {})
    if not isinstance(messages, Mapping) or source not in messages:
        return None
    message = messages[source]
    if not isinstance(message, Mapping) or "covariance" not in message:
        return None
    covariance = np.asarray(message["covariance"], dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] < 3 or covariance.shape[1] < 3:
        return None
    value = float(np.trace(covariance[:3, :3]))
    return value if np.isfinite(value) else None


def quality_candidate_scores(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Score only candidates and fields already present in the actor observation."""

    candidates = actor_visible_measurement_candidates(observation, uav_ids)
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        incoming = {}
    rows: list[dict[str, Any]] = []
    for source in candidates:
        link = incoming.get(source, {})
        if not isinstance(link, Mapping):
            link = {}
        metrics = {
            "rssi_dbm": float(link.get("rssi_dbm", -np.inf)),
            "snr_db": float(link.get("snr_db", -np.inf)),
            "packet_loss": float(link.get("packet_loss", np.inf)),
            "delay_s": float(link.get("delay_s", np.inf)),
            "aoi_s": float(link.get("aoi_s", np.inf)),
            "queue_length_packets": float(link.get("queue_length_packets", np.inf)),
        }
        covariance_trace = _neighbor_covariance_trace(observation, source)
        quantiles = thresholds["quantiles"]
        eligible = str(link.get("link_state", "unavailable")) == "connected"
        eligible &= metrics["rssi_dbm"] >= float(quantiles["rssi_dbm"]["q25"])
        eligible &= metrics["snr_db"] >= float(quantiles["snr_db"]["q25"])
        for field in _LOW_IS_GOOD:
            eligible &= metrics[field] <= float(quantiles[field]["q75"])
        if covariance_trace is not None:
            eligible &= covariance_trace <= float(quantiles["neighbor_covariance_trace_m2"]["q75"])
        risks = {field: _risk(value, field, thresholds) for field, value in metrics.items()}
        risks["neighbor_covariance_trace_m2"] = (
            _risk(covariance_trace, "neighbor_covariance_trace_m2", thresholds)
            if covariance_trace is not None else 1.0
        )
        score = float(np.mean(list(risks.values())))
        rows.append({
            "source_uav_id": source,
            "eligible": bool(eligible),
            "quality_risk_score": score,
            "link_state": str(link.get("link_state", "unavailable")),
            **metrics,
            "neighbor_covariance_trace_m2": covariance_trace if covariance_trace is not None else -1.0,
            "neighbor_covariance_available": covariance_trace is not None,
            "risk_components": risks,
            "actor_visible_only": True,
        })
    return sorted(rows, key=lambda row: (float(row["quality_risk_score"]), str(row["source_uav_id"])))


def intervene_quality_measurement_selection(
    recorded_actions: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    treatment: str,
    thresholds: Mapping[str, Any],
    round_robin_cursors: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if treatment in ("recorded_mappo", "round_robin_candidate"):
        return intervene_measurement_selection(
            recorded_actions, observations, uav_ids, treatment, round_robin_cursors
        )
    if treatment != "quality_aware_candidate":
        raise ValueError(f"unknown quality-aware treatment: {treatment}")
    actions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        recorded = recorded_actions[uav_id]
        scores = quality_candidate_scores(observations[uav_id], uav_ids, thresholds)
        eligible = [row for row in scores if bool(row["eligible"])]
        selected = str(eligible[0]["source_uav_id"]) if eligible else "none"
        recorded_source = str(recorded.get("measurement_uav_id", "none"))
        actions[uav_id] = {
            "measurement_uav_id": selected,
            "bandwidth": {str(key): float(value) for key, value in dict(recorded.get("bandwidth", {})).items()},
            "delta_v": np.asarray(recorded["delta_v"], dtype=float).reshape(3),
        }
        selected_row = next((row for row in scores if row["source_uav_id"] == selected), None)
        metadata[uav_id] = {
            "candidate_count": len(scores),
            "candidate_uav_ids": [row["source_uav_id"] for row in scores],
            "candidate_available": bool(scores),
            "eligible_candidate_count": len(eligible),
            "recorded_measurement_uav_id": recorded_source,
            "selected_measurement_uav_id": selected,
            "selected_non_none": selected != "none",
            "selection_changed": selected != recorded_source,
            "selected_quality_risk_score": (
                float(selected_row["quality_risk_score"]) if selected_row is not None else -1.0
            ),
            "candidate_quality_json": json.dumps(scores, sort_keys=True),
            "actor_visible_only": True,
            "uses_physical_los": False,
            "uses_truth_error": False,
            "uses_future_information": False,
        }
    return actions, metadata


def _run_episode(
    *,
    config: QualityAwareMeasurementInterventionConfig,
    formal_config: Any,
    thresholds: Mapping[str, Any],
    treatment: str,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
    selector: Callable[..., tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = intervene_quality_measurement_selection,
    policy_prefix: str = POLICY_PREFIX,
    measurement_overhead_enabled: bool = False,
    overhead_rows_sink: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        measurement_overhead_enabled=measurement_overhead_enabled,
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
        policy_source=f"{policy_prefix}_{treatment}",
    )
    cursors = {uav_id: 0 for uav_id in environment.uav_ids}
    windows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    errors: list[float] = []
    traces: list[float] = []
    nees_values: list[float] = []
    rewards_history: list[float] = []
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()
    for step, recorded in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        actions, metadata = selector(
            recorded, observations, environment.uav_ids, treatment, thresholds, cursors
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
            policy_source=f"{policy_prefix}_{treatment}",
            action_start=step * environment.config.motion.dt_s,
            timestamp=timestamp,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if overhead_rows_sink is not None:
            for overhead_row in next_info.get("measurement_overhead_events", []):
                overhead_rows_sink.append(
                    {
                        **dict(overhead_row),
                        "episode_id": episode_id,
                        "scenario_id": scenario_id,
                        "seed": int(seed),
                        "split": config.split,
                        "treatment": treatment,
                        "step": int(step),
                    }
                )
        rewards_history.extend(float(value) for value in rewards.values())
        current_errors, current_traces, current_nees = _position_diagnostics(environment)
        errors.extend(current_errors)
        traces.extend(current_traces)
        nees_values.extend(current_nees)
        gate_by_target = {
            str(event["target_uav_id"]): event
            for event in next_info.get("measurement_gate_events", [])
        }
        components = next_info["objective_components"]
        for uav_id in environment.uav_ids:
            row = _gate_row(
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=seed,
                split=config.split,
                treatment=treatment,
                step=step,
                timestamp=timestamp,
                uav_id=uav_id,
                metadata=metadata[uav_id],
                gate_event=gate_by_target.get(uav_id),
                environment=environment,
                components=components,
            )
            row.update(
                eligible_candidate_count=int(metadata[uav_id].get("eligible_candidate_count", -1)),
                selected_quality_risk_score=float(metadata[uav_id].get("selected_quality_risk_score", -1.0)),
                candidate_quality_json=str(metadata[uav_id].get("candidate_quality_json", "[]")),
                actor_visible_only=int(bool(metadata[uav_id].get("actor_visible_only", True))),
                physical_los_used_for_selection=int(bool(metadata[uav_id].get("uses_physical_los", False))),
                truth_error_used_for_selection=int(bool(metadata[uav_id].get("uses_truth_error", False))),
                future_information_used_for_selection=int(bool(metadata[uav_id].get("uses_future_information", False))),
            )
            windows.append(row)
        for event_index, event in enumerate(next_info.get("relative_range_update_events", [])):
            events.append(_quality_event_row(
                event=event,
                treatment=treatment,
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=seed,
                step=step,
                event_index=event_index,
            ))
        observations = next_observations
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
    accepted = [row for row in events if int(row["accepted"])]
    counts = {
        label: sum(row["effect_class"] == label for row in accepted)
        for label in ("beneficial", "harmful", "approximately_neutral")
    }
    total = max(1, len(accepted))
    treatment_windows = [row for row in windows if row["treatment"] == treatment]
    candidate_count = sum(int(row["candidate_available"]) for row in treatment_windows)
    metrics.update(
        treatment=treatment,
        accepted_update_count=len(accepted),
        ekf_updates_accepted=len(accepted),
        beneficial_fraction=counts["beneficial"] / total,
        harmful_fraction=counts["harmful"] / total,
        neutral_fraction=counts["approximately_neutral"] / total,
        nlos_update_fraction=_mean(1 - int(row["los"]) for row in accepted),
        stale_update_fraction=_mean(float(row["message_age_s"]) > 0.5 for row in accepted),
        position_error_p95_m=_quantile(errors, 0.95),
        mean_position_covariance_trace_m2=_mean(traces),
        mean_position_nees=_mean(nees_values),
        nees_95_coverage=_mean(_NEES_95_LOWER <= value <= _NEES_95_UPPER for value in nees_values),
        selection_fraction_given_candidate=(
            sum(int(row["candidate_available"] and row["selected_non_none"]) for row in treatment_windows)
            / max(1, candidate_count)
        ),
        eligible_fraction_given_candidate=(
            sum(int(row["eligible_candidate_count"] > 0) for row in treatment_windows)
            / max(1, candidate_count)
            if treatment == "quality_aware_candidate" else -1.0
        ),
    )
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"updates={len(accepted)} harmful={metrics['harmful_fraction']:.3f} "
        f"rmse={metrics['position_rmse_m']:.6f} elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, windows, events


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]], config: QualityAwareMeasurementInterventionConfig
) -> list[dict[str, Any]]:
    metrics = (
        "selection_fraction_given_candidate", "accepted_update_count",
        "beneficial_fraction", "harmful_fraction", "neutral_fraction",
        "nlos_update_fraction", "stale_update_fraction", "position_rmse_m",
        "position_error_p95_m", "j_loc", "mean_position_covariance_trace_m2",
        "mean_position_nees", "nees_95_coverage",
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for comparator in ("recorded_mappo", "round_robin_candidate"):
        for metric in metrics:
            baseline = np.asarray([float(by_key[(seed, comparator)][metric]) for seed in seeds])
            quality = np.asarray([float(by_key[(seed, "quality_aware_candidate")][metric]) for seed in seeds])
            difference = quality - baseline
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append({
                "contrast": f"quality_aware_candidate_minus_{comparator}",
                "metric": metric,
                "paired_episodes": len(seeds),
                "comparator_mean": float(baseline.mean()),
                "quality_aware_candidate_mean": float(quality.mean()),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "paired_difference_std": float(difference.std(ddof=1)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            })
    return output


def _invariance_checks(
    config: QualityAwareMeasurementInterventionConfig,
    roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    checks = {
        "recorded_reproduces_reference_max_abs_difference": 0.0,
        "cross_treatment_ground_truth_max_abs_difference": 0.0,
        "cross_treatment_obstacle_max_abs_difference": 0.0,
        "cross_treatment_link_kpi_max_abs_difference": 0.0,
        "cross_treatment_packet_event_max_abs_difference": 0.0,
        "cross_treatment_delta_v_max_abs_difference": 0.0,
        "cross_treatment_bandwidth_max_abs_difference": 0.0,
    }
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        factual = roots["recorded_mappo"] / config.split / episode_id
        checks["recorded_reproduces_reference_max_abs_difference"] = max(
            checks["recorded_reproduces_reference_max_abs_difference"],
            _max_numeric_difference(
                factual / "estimates.csv", reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                factual / "rewards.csv", reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
        )
        for treatment in ("round_robin_candidate", "quality_aware_candidate"):
            comparator = roots[treatment] / config.split / episode_id
            pairs = (
                ("cross_treatment_ground_truth_max_abs_difference", "ground_truth.csv", ("timestamp", "uav_id"), ("x", "y", "z", "vx", "vy", "vz")),
                ("cross_treatment_obstacle_max_abs_difference", "obstacles.csv", ("timestamp", "obstacle_id"), ("x", "y", "z", "vx", "vy", "vz", "radius")),
                ("cross_treatment_link_kpi_max_abs_difference", "links.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length")),
                ("cross_treatment_packet_event_max_abs_difference", "packet_events.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("success", "delay", "aoi_after")),
                ("cross_treatment_delta_v_max_abs_difference", "actions.csv", ("timestamp", "uav_id"), ("delta_vx", "delta_vy", "delta_vz")),
            )
            for name, filename, keys, values in pairs:
                checks[name] = max(checks[name], _max_numeric_difference(
                    factual / filename, comparator / filename,
                    key_columns=keys, value_columns=values,
                ))
            checks["cross_treatment_bandwidth_max_abs_difference"] = max(
                checks["cross_treatment_bandwidth_max_abs_difference"],
                _bandwidth_difference(factual / "actions.csv", comparator / "actions.csv"),
            )
    return checks


def _audit(
    config: QualityAwareMeasurementInterventionConfig,
    roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    invariants = _invariance_checks(config, roots, episodes)
    contrast_map = {(row["contrast"], row["metric"]): row for row in contrasts}
    prefix = "quality_aware_candidate_minus_round_robin_candidate"
    harmful = contrast_map[(prefix, "harmful_fraction")]
    rmse = contrast_map[(prefix, "position_rmse_m")]
    updates = contrast_map[(prefix, "accepted_update_count")]
    recorded_updates = _mean(float(row["accepted_update_count"]) for row in episode_rows if row["treatment"] == "recorded_mappo")
    round_robin_updates = float(updates["comparator_mean"])
    quality_updates = float(updates["quality_aware_candidate_mean"])
    harmful_reduced = float(harmful["mean_paired_difference"]) < 0.0 and float(harmful["bootstrap_95_ci_high"]) < 0.0
    factual_rmse = _mean(float(row["position_rmse_m"]) for row in episode_rows if row["treatment"] == "recorded_mappo")
    practical = max(0.05, 0.05 * factual_rmse)
    rmse_improved = float(rmse["mean_paired_difference"]) < -practical and float(rmse["bootstrap_95_ci_high"]) < 0.0
    rmse_worsened = float(rmse["mean_paired_difference"]) > practical and float(rmse["bootstrap_95_ci_low"]) > 0.0
    update_collapse = quality_updates < max(recorded_updates, 0.5 * round_robin_updates)
    if harmful_reduced and rmse_improved:
        decision = "quality_aware_selection_effective_integrate_mask_then_retrain"
    elif harmful_reduced and not rmse_improved:
        decision = "harmful_updates_reduced_rmse_inconclusive_add_random_seeds"
    elif update_collapse:
        decision = "quality_thresholds_too_strict_refit_from_training_only"
    elif rmse_worsened:
        decision = "quality_score_harmful_stop_and_reaudit_visible_proxies"
    else:
        decision = "actor_visible_quality_proxies_not_discriminative_consider_message_age_or_los_estimator"
    finite = all(
        np.isfinite(float(row[key]))
        for row in episode_rows
        for key in ("position_rmse_m", "position_error_p95_m", "j_loc", "mean_position_nees")
    )
    leakage_free = all(
        int(row["physical_los_used_for_selection"]) == 0
        and int(row["truth_error_used_for_selection"]) == 0
        and int(row["future_information_used_for_selection"]) == 0
        for row in window_rows if row["treatment"] == "quality_aware_candidate"
    )
    pairing_ok = all(value <= _NUMERIC_TOLERANCE for value in invariants.values())
    thresholds_train_only = (
        thresholds.get("source_split") == "train"
        and not thresholds.get("validation_used_for_fitting")
        and not thresholds.get("test_used_for_fitting")
    )
    return {
        "experiment": "paired_actor_visible_quality_aware_measurement_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "paired_episode_count": len(episodes),
        "event_rows_used_as_independent_samples": False,
        "test_data_used": False,
        "checks": {
            **invariants,
            "recorded_accepted_updates_mean": recorded_updates,
            "round_robin_accepted_updates_mean": round_robin_updates,
            "quality_aware_accepted_updates_mean": quality_updates,
            "quality_to_round_robin_update_ratio": quality_updates / max(round_robin_updates, 1e-12),
            "quality_minus_round_robin_harmful_fraction": float(harmful["mean_paired_difference"]),
            "harmful_fraction_bootstrap_95_ci": [float(harmful["bootstrap_95_ci_low"]), float(harmful["bootstrap_95_ci_high"])],
            "quality_minus_round_robin_position_rmse_m": float(rmse["mean_paired_difference"]),
            "position_rmse_bootstrap_95_ci_m": [float(rmse["bootstrap_95_ci_low"]), float(rmse["bootstrap_95_ci_high"])],
            "position_rmse_practical_threshold_m": practical,
        },
        "thresholds_fitted_from_train_only": bool(thresholds_train_only),
        "selection_information_leakage_free": bool(leakage_free),
        "deterministic_pairing_ok": bool(pairing_ok),
        "all_metrics_finite": bool(finite),
        "decision_gate": decision,
        "overall_ok": bool(thresholds_train_only and leakage_free and pairing_ok and finite),
        "interpretation_boundary": (
            "Physical LOS/NLOS and true error are offline labels only. Validation episodes are paired units; "
            "this intervention does not retrain MAPPO or prove final policy superiority."
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

    colors = {
        "recorded_mappo": "#2563eb",
        "round_robin_candidate": "#dc2626",
        "quality_aware_candidate": "#16a34a",
    }
    labels = {
        "recorded_mappo": "Recorded MAPPO",
        "round_robin_candidate": "Round-robin",
        "quality_aware_candidate": "Quality-aware",
    }
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
    seeds = sorted({int(row["seed"]) for row in episode_rows})
    panels = (
        ("selection_fraction_given_candidate", "Non-none selection fraction"),
        ("accepted_update_count", "Accepted cooperative updates"),
        ("harmful_fraction", "Harmful update fraction"),
        ("position_rmse_m", "Position RMSE (m)"),
        ("position_error_p95_m", "Position error P95 (m)"),
        ("mean_position_nees", "Mean position NEES"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel()[:6], panels):
        for seed in seeds:
            axis.plot(
                range(len(TREATMENTS)),
                [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS],
                color="#94a3b8", alpha=0.35, linewidth=0.8,
            )
        means = [_mean(float(row[metric]) for row in episode_rows if row["treatment"] == treatment) for treatment in TREATMENTS]
        axis.scatter(range(len(TREATMENTS)), means, s=55, c=[colors[t] for t in TREATMENTS], zorder=3)
        axis.set_xticks(range(len(TREATMENTS)), ["MAPPO", "Round-robin", "Quality-aware"], rotation=15)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axis = axes.ravel()[6]
    accepted = [row for row in event_rows if int(row["accepted"])]
    x = np.arange(3)
    width = 0.24
    for offset, treatment in enumerate(TREATMENTS):
        selected = [row for row in accepted if row["treatment"] == treatment]
        fractions = [
            _mean(row["effect_class"] == category for row in selected)
            for category in ("beneficial", "harmful", "approximately_neutral")
        ]
        axis.bar(x + (offset - 1) * width, fractions, width, color=colors[treatment], label=labels[treatment])
    axis.set_xticks(x, ["Beneficial", "Harmful", "Neutral"])
    axis.set_ylabel("Fraction of accepted updates")
    axis.set_title("Offline update-effect labels")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    axis = axes.ravel()[7]
    for treatment in TREATMENTS:
        selected = [row for row in accepted if row["treatment"] == treatment]
        axis.scatter(
            [float(row["message_age_s"]) for row in selected],
            [float(row["true_error_change_after_minus_before_m"]) for row in selected],
            s=10, alpha=0.35, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.axvline(0.5, color="#f59e0b", linestyle="--", linewidth=1)
    axis.set_xlabel("Message age (s), offline diagnostic")
    axis.set_ylabel("Error change after-before (m)")
    axis.set_title("Freshness and update effect")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    target = plots_dir / "quality_aware_measurement.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_quality_aware_measurement_intervention(
    config: QualityAwareMeasurementInterventionConfig,
) -> dict[str, Any]:
    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo" or int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("reference must be the frozen three-UAV MAPPO run")
    episodes = [
        item for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen intervention requires exactly 20 validation episodes, found {len(episodes)}")
    logs_dir, plots_dir = _prepare_output(config)
    thresholds = fit_training_quality_thresholds(reference_scale)
    _write_json(logs_dir / "quality_thresholds.json", thresholds)
    formal_config = _reference_formal_config(manifest, Path(config.output_dir), force=False, progress=config.progress)
    roots: dict[str, Path] = {}
    treatment_manifests: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        roots[treatment] = scale_root
        treatment_config = _reference_formal_config(manifest, treatment_root, force=False, progress=config.progress)
        _copy_calibration_files(scale_root, treatment_config)
        normalization = reference_scale / "normalization_stats.json"
        if normalization.exists():
            shutil.copy2(normalization, scale_root / "normalization_stats.json")
        summaries: list[dict[str, Any]] = []
        metrics_rows: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metrics, windows, events = _run_episode(
                config=config,
                formal_config=formal_config,
                thresholds=thresholds,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            metrics_rows.append(metrics)
            window_rows.extend(windows)
            event_rows.extend(events)
        episode_rows.extend(metrics_rows)
        treatment_manifest = {
            "dataset_id": f"python_quality_aware_measurement_{treatment}",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "engine": "python_sim",
            "result_status": "engineering_mechanism_diagnostic_only",
            "policy_source": f"{POLICY_PREFIX}_{treatment}",
            "uav_count": 3,
            "reference": {"scale_root": str(reference_scale), "manifest_sha256": _sha256(reference_scale / "manifest.json"), "split": config.split},
            "intervention": {
                "treatment": treatment,
                "changed_action_field_only": "measurement_uav_id",
                "quality_threshold_source": "train",
                "actor_visible_only": treatment == "quality_aware_candidate",
                "physical_los_or_truth_used_for_selection": False,
                "uses_future_information": False,
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", treatment_manifest)
        _write_json(treatment_root / "manifest.json", {
            "dataset_id": treatment_manifest["dataset_id"], "schema_version": EPISODE_SCHEMA_VERSION,
            "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
        })
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {"scale_root": str(scale_root), "validation_ok": True}
    contrasts = _paired_contrasts(episode_rows, config)
    audit = _audit(config, roots, episodes, episode_rows, window_rows, contrasts, thresholds)
    _write_table(logs_dir / "window_metrics.csv", window_rows)
    _write_table(logs_dir / "update_events.csv", event_rows)
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(episode_rows, event_rows, plots_dir)
    root_manifest = {
        "dataset_id": "python_quality_aware_measurement_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_mechanism_diagnostic_only",
        "reference": {"scale_root": str(reference_scale), "manifest_sha256": _sha256(reference_scale / "manifest.json"), "split": config.split, "episode_count": len(episodes)},
        "paired_design": {
            "independent_unit": "episode", "paired_by": ["scenario_id", "seed"],
            "common_random_numbers": True, "event_rows_are_independent_replicates": False,
            "quality_threshold_source": "train", "validation_used_for_threshold_fitting": False,
            "test_data_used": False, "changed_field_only": "measurement_uav_id",
        },
        "quality_thresholds": thresholds,
        "treatments": treatment_manifests,
        "logs": {
            "quality_thresholds": str(logs_dir / "quality_thresholds.json"),
            "window_metrics": str(logs_dir / "window_metrics.csv"),
            "update_events": str(logs_dir / "update_events.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"quality_aware_measurement": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"quality-aware measurement audit failed: {audit['checks']}")
    return {
        "output_dir": str(config.output_dir), "reference_scale": str(reference_scale),
        "paired_episodes": len(episodes), "treatments": list(TREATMENTS),
        "schema_version": EPISODE_SCHEMA_VERSION, "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir), "decision_gate": audit["decision_gate"],
        "audit_ok": True, "plot": str(plot_path),
    }


__all__ = [
    "QualityAwareMeasurementInterventionConfig", "TREATMENTS",
    "fit_training_quality_thresholds", "quality_candidate_scores",
    "intervene_quality_measurement_selection", "run_quality_aware_measurement_intervention",
]
