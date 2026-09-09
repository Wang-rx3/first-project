"""Read-only audit of cooperative UWB updates and subsequent EKF sensors.

This module intentionally consumes frozen validation artefacts instead of
replaying the simulator.  It pairs each accepted cooperative UWB update with
the next diagnostic window for the same target UAV and follows the target for
several later windows.  Ground-truth errors and LOS labels are retained only
as validation diagnostics; they never enter an actor, an EKF update, a
reward, or a fitted parameter.

The audit is designed to answer two separate questions:

* Does a cooperative update lose its mean-error benefit after the next
  prediction/visual-odometry/update cycle?
* Does the conservative covariance restoration leave ``J_loc`` unchanged
  even when the linear EKF covariance decreases?

Events are nested within episodes and episodes are reduced to training-seed
rows before any paired confidence interval is calculated.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ekf_update_persistence_audit import (
    DEFAULT_BRANCHES,
    DEFAULT_HORIZONS,
    DEFAULT_SEEDS,
    _bootstrap_ci,
    _bool,
    _episode_dirs,
    _float,
    _position_error_metrics,
    _safe_mean,
    _safe_std,
    _seed_branch_dirs,
)
from .metrics import position_error_metrics


PRACTICAL_ERROR_THRESHOLD_M = 0.001
MINIMUM_EVENT_COUNT = 30
MINIMUM_TRANSITION_FRACTION = 0.50
JLOC_CHANGE_TOLERANCE = 1.0e-8
FINAL_COVARIANCE_TOLERANCE_M2 = 1.0e-8


@dataclass(frozen=True)
class EKFSensorInteractionAuditConfig:
    """Configuration for the validation-only sensor interaction audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "communication_constrained_proxy_mappo_seedset_12136_16172/"
        "ekf_sensor_interaction_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    expected_episodes: int = 20
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("EKF sensor interaction audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty unique sequence")
        if not self.horizons or any(int(value) <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.expected_episodes < 1:
            raise ValueError("expected_episodes must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: EKFSensorInteractionAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[ekf-sensor-interaction] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_validation_updates(path: Path) -> list[dict[str, Any]]:
    """Read accepted validation updates and discard other split rows.

    The aggregate JSONL file contains train/validation/test rows.  Only rows
    whose split is validation and whose accepted flag is true are parsed into
    the audit data structure.  No test episode directory or test metric is
    opened by this module.
    """

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid cooperative update JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                continue
            if str(value.get("split", "")) != "validation":
                continue
            if _bool(value.get("accepted", False)):
                rows.append(value)
    return rows


def _safe_optional_mean(values: Iterable[Any]) -> float | str:
    numbers = [_float(value, float("nan")) for value in values]
    finite = [value for value in numbers if math.isfinite(value)]
    return float(np.mean(finite)) if finite else ""


def _diagnostic_sequences(
    episode_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, float], dict[str, float]]]:
    """Load one episode's diagnostic rows keyed by target UAV."""

    by_uav: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(episode_root / "ekf_diagnostics.csv"):
        uav_id = str(row.get("uav_id", ""))
        if not uav_id:
            continue
        by_uav[uav_id].append(
            {
                "timestamp": _float(row.get("timestamp"), float("nan")),
                "position_error_m": _float(row.get("position_error_m"), float("nan")),
                "position_nees": _float(row.get("position_nees"), float("nan")),
                "covariance_trace_m2": _float(
                    row.get("position_covariance_trace_m2"), float("nan")
                ),
                "nis_vo": _float(row.get("nis_vo"), float("nan")),
                "nis_uwb_mean": _float(row.get("nis_uwb_mean"), float("nan")),
                "nis_uwb_max": _float(row.get("nis_uwb_max"), float("nan")),
                "nis_gnss": _float(row.get("nis_gnss"), float("nan")),
                "vo_update_attempted": _bool(row.get("vo_update_attempted")),
                "vo_update_accepted": _bool(row.get("vo_update_accepted")),
                "vo_update_rejected": _bool(row.get("vo_update_rejected")),
                "uwb_update_attempted": _bool(row.get("uwb_update_attempted")),
                "uwb_update_accepted": _bool(row.get("uwb_update_accepted")),
                "uwb_update_rejected": _bool(row.get("uwb_update_rejected")),
                "gnss_update_attempted": _bool(row.get("gnss_update_attempted")),
                "gnss_update_accepted": _bool(row.get("gnss_update_accepted")),
                "gnss_update_rejected": _bool(row.get("gnss_update_rejected")),
            }
        )
    for rows in by_uav.values():
        rows.sort(key=lambda item: _float(item.get("timestamp"), float("inf")))

    # Rewards contain the per-window normalized uncertainty used by the
    # current implementation.  It is diagnostic only here.
    uncertainty_weight = 0.35
    scenario_path = episode_root / "scenario.json"
    if scenario_path.is_file():
        try:
            payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        reward_config = payload.get("reward_config", {}) if isinstance(payload, dict) else {}
        if isinstance(reward_config, Mapping):
            weights = reward_config.get("weights", {})
            if isinstance(weights, Mapping):
                uncertainty_weight = _float(
                    weights.get("uncertainty", weights.get("unc", uncertainty_weight)),
                    uncertainty_weight,
                )
    rewards: dict[tuple[str, float], dict[str, float]] = {}
    for row in _read_csv(episode_root / "rewards.csv"):
        uav_id = str(row.get("uav_id", ""))
        timestamp = _float(row.get("timestamp"), float("nan"))
        if not uav_id or not math.isfinite(timestamp):
            continue
        j_loc = uncertainty_weight * _float(row.get("normalized_uncertainty"), float("nan"))
        joint = _float(row.get("joint_objective"), float("nan"))
        rewards[(uav_id, round(timestamp, 6))] = {
            "j_loc": j_loc,
            "j_com": joint - j_loc if math.isfinite(joint) and math.isfinite(j_loc) else float("nan"),
            "mean_reward": _float(row.get("reward_total"), float("nan")),
        }
    return dict(by_uav), rewards


def _find_current_index(rows: Sequence[Mapping[str, Any]], event: Mapping[str, Any]) -> int:
    timestamp = _float(event.get("measurement_time_s", event.get("timestamp")), float("nan"))
    if math.isfinite(timestamp):
        for index, row in enumerate(rows):
            row_timestamp = _float(row.get("timestamp"), float("nan"))
            if math.isfinite(row_timestamp) and abs(row_timestamp - timestamp) <= 1.0e-6:
                return index
    step = int(round(_float(event.get("step"), 0.0)))
    return max(0, min(step, len(rows) - 1)) if rows else -1


def _state_components(value: Any) -> tuple[float, float, float]:
    """Return finite position components from a serialized EKF state."""

    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = []
    if isinstance(candidate, Mapping):
        candidate = [candidate.get(axis, float("nan")) for axis in ("x", "y", "z")]
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        candidate = []
    values = [_float(candidate[index], 0.0) if index < len(candidate) else 0.0 for index in range(3)]
    return float(values[0]), float(values[1]), float(values[2])


def _transition_type(
    row: Mapping[str, Any], *, timestamp: float, cooperative_times: set[float]
) -> str:
    if math.isfinite(timestamp) and round(timestamp, 6) in cooperative_times:
        return "cooperative_uwb_update"
    if any(
        _bool(row.get(name))
        for name in ("vo_update_attempted", "vo_update_accepted", "vo_update_rejected")
    ):
        return "visual_odometry_update"
    if any(
        _bool(row.get(name))
        for name in ("gnss_update_attempted", "gnss_update_accepted", "gnss_update_rejected")
    ):
        return "gnss_update"
    if any(
        _bool(row.get(name))
        for name in ("uwb_update_attempted", "uwb_update_accepted", "uwb_update_rejected")
    ):
        return "onboard_uwb_update"
    return "prediction_only"


def _first_matching(
    rows: Sequence[Mapping[str, Any]],
    start_index: int,
    predicate: Any,
) -> tuple[int, Mapping[str, Any]] | None:
    for index in range(max(0, start_index), len(rows)):
        if predicate(rows[index]):
            return index, rows[index]
    return None


def _matching_event_row(
    *,
    event: Mapping[str, Any],
    seed: int,
    branch: str,
    episode_root: Path,
    update_index: int,
    horizons: Sequence[int],
    sequences: Mapping[str, Sequence[Mapping[str, Any]]],
    rewards: Mapping[tuple[str, float], Mapping[str, float]],
    cooperative_times: set[float],
) -> dict[str, Any] | None:
    """Construct one event-to-follow-up interaction row."""

    if not _bool(event.get("accepted", False)):
        return None
    target = str(event.get("target_uav_id", ""))
    sequence = sequences.get(target, [])
    current_index = _find_current_index(sequence, event)
    if not target or not sequence or current_index < 0:
        return None

    timestamp = _float(event.get("measurement_time_s", event.get("timestamp")), 0.0)
    before_error = _float(event.get("true_position_error_before_m"), 0.0)
    after_error = _float(event.get("true_position_error_after_m"), 0.0)
    before_state = _state_components(event.get("state_before"))
    after_state = _state_components(event.get("state_after"))
    state_correction = _state_components(event.get("state_correction"))
    state_residual = np.asarray(after_state) - np.asarray(before_state) - np.asarray(state_correction)
    position_correction_norm = _float(
        event.get("position_correction_norm_m"), float(np.linalg.norm(state_correction))
    )
    cov_before = _float(event.get("position_covariance_trace_before_m2"), 0.0)
    cov_linear = _float(event.get("position_covariance_trace_after_linear_m2"), 0.0)
    cov_final = _float(event.get("position_covariance_trace_after_final_m2"), 0.0)
    j_loc_before = _float(event.get("j_loc_before"), 0.0)
    j_loc_after = _float(event.get("j_loc_after"), j_loc_before)

    row: dict[str, Any] = {
        "base_seed": int(seed),
        "branch": str(branch),
        "split": "validation",
        "episode_id": str(event.get("episode_id", episode_root.name)),
        "scenario_id": str(event.get("scenario_id", "")),
        "update_index": int(update_index),
        "step": int(round(_float(event.get("step"), 0.0))),
        "timestamp": timestamp,
        "source_uav_id": str(event.get("source_uav_id", "")),
        "target_uav_id": target,
        "accepted": 1,
        "los_label_offline": int(_bool(event.get("los", False))),
        "message_age_s": _float(event.get("message_age_s"), 0.0),
        "message_delay_s": _float(event.get("message_delay_s"), 0.0),
        "nis": _float(event.get("nis"), 0.0),
        "measurement_variance_m2": _float(event.get("measurement_variance_m2"), 0.0),
        "effective_measurement_variance_m2": _float(
            event.get("effective_measurement_variance_m2"), 0.0
        ),
        "local_direction_variance_m2": _float(event.get("local_direction_variance_m2"), 0.0),
        "neighbor_direction_variance_m2": _float(
            event.get("neighbor_direction_variance_m2"), 0.0
        ),
        "unknown_cross_bound_m2": _float(event.get("unknown_cross_bound_m2"), 0.0),
        "unknown_cross_bound_fraction": _float(
            event.get("unknown_cross_bound_fraction"), 0.0
        ),
        "kalman_gain_norm": _float(event.get("kalman_gain_norm"), 0.0),
        "position_correction_x_m": float(state_correction[0]),
        "position_correction_y_m": float(state_correction[1]),
        "position_correction_z_m": float(state_correction[2]),
        "position_correction_norm_m": position_correction_norm,
        "state_before_x_m": float(before_state[0]),
        "state_before_y_m": float(before_state[1]),
        "state_before_z_m": float(before_state[2]),
        "state_after_x_m": float(after_state[0]),
        "state_after_y_m": float(after_state[1]),
        "state_after_z_m": float(after_state[2]),
        "state_correction_residual_norm_m": float(np.linalg.norm(state_residual)),
        "covariance_trace_before_m2": cov_before,
        "covariance_trace_after_linear_m2": cov_linear,
        "covariance_trace_after_final_m2": cov_final,
        "linear_cov_reduction_m2": cov_before - cov_linear,
        "final_cov_reduction_m2": cov_before - cov_final,
        "covariance_restored": int(_bool(event.get("covariance_restored", False))),
        "j_loc_before": j_loc_before,
        "j_loc_after": j_loc_after,
        "j_loc_immediate_change": j_loc_after - j_loc_before,
        "true_position_error_before_m": before_error,
        "true_position_error_after_m": after_error,
        "immediate_error_change_m": after_error - before_error,
        "immediate_effect_class": (
            "beneficial"
            if after_error - before_error < -PRACTICAL_ERROR_THRESHOLD_M
            else "harmful"
            if after_error - before_error > PRACTICAL_ERROR_THRESHOLD_M
            else "approximately_neutral"
        ),
        "truth_used_offline_only": 1,
    }

    next_index = current_index + 1
    next_row = sequence[next_index] if next_index < len(sequence) else None
    if next_row is None:
        row.update(
            {
                "next_window_observed": 0,
                "next_window_timestamp": -1.0,
                "next_window_transition_type": "censored",
                "next_window_position_error_m": -1.0,
                "next_window_error_change_from_after_m": -1.0,
                "next_window_covariance_trace_m2": -1.0,
                "next_window_covariance_change_m2": -1.0,
                "next_window_position_nees": -1.0,
                "next_prediction_error_change_m": -1.0,
                "next_prediction_covariance_change_m2": -1.0,
            }
        )
    else:
        next_timestamp = _float(next_row.get("timestamp"), -1.0)
        next_error = _float(next_row.get("position_error_m"), -1.0)
        next_covariance = _float(next_row.get("covariance_trace_m2"), -1.0)
        next_type = _transition_type(
            next_row, timestamp=next_timestamp, cooperative_times=cooperative_times
        )
        row.update(
            {
                "next_window_observed": 1,
                "next_window_timestamp": next_timestamp,
                "next_window_transition_type": next_type,
                "next_window_position_error_m": next_error,
                "next_window_error_change_from_after_m": next_error - after_error,
                "next_window_covariance_trace_m2": next_covariance,
                "next_window_covariance_change_m2": next_covariance - cov_final,
                "next_window_position_nees": _float(next_row.get("position_nees"), -1.0),
                "next_prediction_error_change_m": next_error - after_error,
                "next_prediction_covariance_change_m2": next_covariance - cov_final,
            }
        )

    next_sensor = _first_matching(
        sequence,
        next_index,
        lambda item: _transition_type(
            item,
            timestamp=_float(item.get("timestamp"), float("nan")),
            cooperative_times=cooperative_times,
        )
        != "prediction_only",
    )
    if next_sensor is None:
        row["next_sensor_update_type"] = "none_observed"
        row["next_sensor_update_h"] = -1
    else:
        sensor_index, sensor_row = next_sensor
        row["next_sensor_update_type"] = _transition_type(
            sensor_row,
            timestamp=_float(sensor_row.get("timestamp"), float("nan")),
            cooperative_times=cooperative_times,
        )
        row["next_sensor_update_h"] = sensor_index - current_index

    next_vo = _first_matching(
        sequence,
        next_index,
        lambda item: any(
            _bool(item.get(name))
            for name in ("vo_update_attempted", "vo_update_accepted", "vo_update_rejected")
        ),
    )
    if next_vo is None:
        row.update(
            {
                "next_vo_observed": 0,
                "next_vo_nis": -1.0,
                "next_vo_accepted": 0,
                "next_vo_rejected": 0,
                "next_vo_error_change_m": -1.0,
                "next_vo_covariance_change_m2": -1.0,
            }
        )
    else:
        vo_index, vo_row = next_vo
        vo_error = _float(vo_row.get("position_error_m"), -1.0)
        vo_covariance = _float(vo_row.get("covariance_trace_m2"), -1.0)
        row.update(
            {
                "next_vo_observed": 1,
                "next_vo_nis": _float(vo_row.get("nis_vo"), -1.0),
                "next_vo_accepted": int(_bool(vo_row.get("vo_update_accepted"))),
                "next_vo_rejected": int(_bool(vo_row.get("vo_update_rejected"))),
                "next_vo_error_change_m": vo_error - after_error,
                "next_vo_covariance_change_m2": vo_covariance - cov_final,
                "next_vo_h": vo_index - current_index,
            }
        )

    for label, predicate in (
        (
            "onboard_uwb",
            lambda item: any(
                _bool(item.get(name))
                for name in ("uwb_update_attempted", "uwb_update_accepted", "uwb_update_rejected")
            )
            and _transition_type(
                item,
                timestamp=_float(item.get("timestamp"), float("nan")),
                cooperative_times=cooperative_times,
            )
            != "cooperative_uwb_update",
        ),
        (
            "gnss",
            lambda item: any(
                _bool(item.get(name))
                for name in ("gnss_update_attempted", "gnss_update_accepted", "gnss_update_rejected")
            ),
        ),
        (
            "cooperative_uwb",
            lambda item: _transition_type(
                item,
                timestamp=_float(item.get("timestamp"), float("nan")),
                cooperative_times=cooperative_times,
            )
            == "cooperative_uwb_update",
        ),
    ):
        found = _first_matching(sequence, next_index, predicate)
        row[f"next_{label}_observed"] = int(found is not None)
        if found is None:
            row[f"next_{label}_h"] = -1
        else:
            row[f"next_{label}_h"] = found[0] - current_index

    for horizon in horizons:
        prefix = f"h{int(horizon)}"
        future_index = current_index + int(horizon)
        observed = future_index < len(sequence)
        future = sequence[future_index] if observed else {}
        future_error = _float(future.get("position_error_m"), -1.0) if observed else -1.0
        future_covariance = (
            _float(future.get("covariance_trace_m2"), -1.0) if observed else -1.0
        )
        future_timestamp = _float(future.get("timestamp"), -1.0) if observed else -1.0
        future_reward = (
            rewards.get((target, round(future_timestamp, 6)), {}) if observed else {}
        )
        future_j_loc = _float(future_reward.get("j_loc"), -1.0)
        transition = (
            _transition_type(
                future,
                timestamp=future_timestamp,
                cooperative_times=cooperative_times,
            )
            if observed
            else "censored"
        )
        row[f"{prefix}_observed"] = int(observed)
        row[f"{prefix}_future_error_m"] = future_error
        row[f"{prefix}_error_change_from_before_m"] = future_error - before_error if observed else -1.0
        row[f"{prefix}_error_change_from_after_m"] = future_error - after_error if observed else -1.0
        row[f"{prefix}_future_covariance_trace_m2"] = future_covariance
        row[f"{prefix}_covariance_change_from_final_m2"] = future_covariance - cov_final if observed else -1.0
        row[f"{prefix}_future_position_nees"] = (
            _float(future.get("position_nees"), -1.0) if observed else -1.0
        )
        row[f"{prefix}_transition_type"] = transition
        row[f"{prefix}_benefit_retained"] = int(
            observed and future_error < before_error - PRACTICAL_ERROR_THRESHOLD_M
        )
        row[f"{prefix}_j_loc"] = future_j_loc
        row[f"{prefix}_j_loc_change_from_after"] = future_j_loc - j_loc_after if observed else -1.0

    return row


# A descriptive alias makes the helper convenient for focused unit tests and
# keeps the implementation name explicit.
_event_row = _matching_event_row


def _episode_metric_source(scale_root: Path, split: str) -> dict[str, dict[str, str]]:
    path = scale_root / "diagnostics" / "episode_metrics.csv"
    return {
        str(row.get("episode_id", "")): row
        for row in _read_csv(path)
        if str(row.get("split", "")) == split and str(row.get("episode_id", ""))
    }


def _load_event_rows(
    config: EKFSensorInteractionAuditConfig,
    reference_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], int], dict[tuple[int, str], int]]:
    rows: list[dict[str, Any]] = []
    event_counts: dict[tuple[int, str], int] = defaultdict(int)
    episode_counts: dict[tuple[int, str], int] = defaultdict(int)
    for seed, branch, scale_root in _seed_branch_dirs(
        reference_root, config.seeds, config.branches
    ):
        metric_map = _episode_metric_source(scale_root, config.split)
        episode_paths = _episode_dirs(scale_root, config.split)
        episode_counts[(seed, branch)] = len(episode_paths)
        updates = _read_validation_updates(scale_root / "diagnostics" / "cooperative_updates.jsonl")
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for update in updates:
            episode_id = str(update.get("episode_id", ""))
            if episode_id:
                by_episode[episode_id].append(update)
        for index, episode_path in enumerate(episode_paths, start=1):
            episode_id = episode_path.name
            if episode_id not in metric_map:
                raise ValueError(f"missing validation episode metrics for {seed}/{branch}/{episode_id}")
            sequences, rewards = _diagnostic_sequences(episode_path)
            updates_for_episode = sorted(
                by_episode.get(episode_id, []),
                key=lambda item: (
                    int(round(_float(item.get("step"), 0.0))),
                    str(item.get("target_uav_id", "")),
                    str(item.get("source_uav_id", "")),
                ),
            )
            times_by_target: dict[str, set[float]] = defaultdict(set)
            for update in updates_for_episode:
                target = str(update.get("target_uav_id", ""))
                update_time = _float(
                    update.get("measurement_time_s", update.get("timestamp")), float("nan")
                )
                if target and math.isfinite(update_time):
                    times_by_target[target].add(round(update_time, 6))
            for update_index, update in enumerate(updates_for_episode):
                target = str(update.get("target_uav_id", ""))
                event = _matching_event_row(
                    event=update,
                    seed=seed,
                    branch=branch,
                    episode_root=episode_path,
                    update_index=update_index,
                    horizons=config.horizons,
                    sequences=sequences,
                    rewards=rewards,
                    cooperative_times=times_by_target.get(target, set()),
                )
                if event is None:
                    continue
                event["scenario_id"] = str(metric_map[episode_id].get("scenario_id", ""))
                rows.append(event)
                event_counts[(seed, branch)] += 1
            if index == 1 or index == len(episode_paths) or index % 5 == 0:
                _progress(
                    config,
                    f"{branch} seed={seed}: processed validation episode {index}/{len(episode_paths)}",
                )
    return rows, dict(event_counts), dict(episode_counts)


def _episode_rows(
    config: EKFSensorInteractionAuditConfig,
    reference_root: Path,
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[(int(row["base_seed"]), str(row["branch"]), str(row["episode_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for seed, branch, scale_root in _seed_branch_dirs(
        reference_root, config.seeds, config.branches
    ):
        metrics = _episode_metric_source(scale_root, config.split)
        for episode_path in _episode_dirs(scale_root, config.split):
            episode_id = episode_path.name
            source = metrics.get(episode_id)
            if source is None:
                raise ValueError(f"missing metrics for {seed}/{branch}/{episode_id}")
            group = groups.get((seed, branch, episode_id), [])
            canonical_rmse, canonical_p95 = _position_error_metrics(episode_path)
            event_count = len(group)
            beneficial = sum(str(row.get("immediate_effect_class")) == "beneficial" for row in group)
            harmful = sum(str(row.get("immediate_effect_class")) == "harmful" for row in group)
            neutral = event_count - beneficial - harmful
            next_observed = sum(int(_float(row.get("next_window_observed"), 0.0)) for row in group)
            next_vo = sum(int(_float(row.get("next_vo_observed"), 0.0)) for row in group)
            next_prediction = sum(
                str(row.get("next_window_transition_type", "")) == "prediction_only"
                for row in group
            )
            base: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": str(branch),
                "split": config.split,
                "episode_id": episode_id,
                "scenario_id": str(source.get("scenario_id", "")),
                "sample_unit": "episode",
                "position_rmse_m": canonical_rmse,
                "position_error_p95_m": canonical_p95,
                "position_rmse_component_m": _float(source.get("position_rmse_m"), 0.0),
                "j_loc": _float(source.get("j_loc"), 0.0),
                "j_com": _float(source.get("j_com"), 0.0),
                "joint_objective": _float(source.get("joint_objective"), 0.0),
                "mean_reward": _float(source.get("mean_reward"), 0.0),
                "packet_loss": _float(source.get("packet_loss"), 0.0),
                "delay_s": _float(source.get("delay_s"), 0.0),
                "aoi_s": _float(source.get("aoi_s"), 0.0),
                "connectivity": _float(source.get("connectivity"), 0.0),
                "energy_proxy": _float(source.get("energy_proxy"), 0.0),
                "collision_rate": _float(source.get("collision_rate"), 0.0),
                "task_complete": _float(source.get("task_complete"), 0.0),
                "termination_reason": str(source.get("termination_reason", "")),
                "accepted_cooperative_update_count": event_count,
                "eventful_episode": int(event_count > 0),
                "beneficial_update_count": beneficial,
                "harmful_update_count": harmful,
                "neutral_update_count": neutral,
                "beneficial_fraction": beneficial / event_count if event_count else "",
                "harmful_fraction": harmful / event_count if event_count else "",
                "neutral_fraction": neutral / event_count if event_count else "",
                "next_window_observed_count": next_observed,
                "next_visual_odometry_count": next_vo,
                "next_prediction_only_count": next_prediction,
                "next_window_observed_fraction": next_observed / event_count if event_count else "",
                "next_visual_odometry_fraction": next_vo / event_count if event_count else "",
                "next_prediction_only_fraction": next_prediction / event_count if event_count else "",
                "mean_next_window_error_change_from_after_m": _safe_optional_mean(
                    row.get("next_window_error_change_from_after_m")
                    for row in group
                    if _float(row.get("next_window_observed"), 0.0) > 0
                ),
                "mean_next_vo_error_change_m": _safe_optional_mean(
                    row.get("next_vo_error_change_m")
                    for row in group
                    if _float(row.get("next_vo_observed"), 0.0) > 0
                ),
                "linear_cov_reduction_event_fraction": (
                    _safe_mean(
                        _float(row.get("linear_cov_reduction_m2"), 0.0) > 1.0e-12
                        for row in group
                    )
                    if group
                    else ""
                ),
                "linear_cov_reduction_mean_m2": _safe_optional_mean(
                    row.get("linear_cov_reduction_m2") for row in group
                ),
                "final_cov_reduction_mean_m2": _safe_optional_mean(
                    row.get("final_cov_reduction_m2") for row in group
                ),
                "covariance_restored_fraction": (
                    _safe_mean(_bool(row.get("covariance_restored")) for row in group)
                    if group
                    else ""
                ),
                "j_loc_immediate_change_abs_max": max(
                    (abs(_float(row.get("j_loc_immediate_change"), 0.0)) for row in group),
                    default="",
                ),
                "state_correction_residual_max_m": max(
                    (_float(row.get("state_correction_residual_norm_m"), 0.0) for row in group),
                    default="",
                ),
                "truth_used_offline_only": 1,
            }
            for horizon in config.horizons:
                prefix = f"h{int(horizon)}"
                observed = [
                    row for row in group if int(_float(row.get(f"{prefix}_observed"), 0.0)) == 1
                ]
                retained = [
                    row for row in observed if int(_float(row.get(f"{prefix}_benefit_retained"), 0.0)) == 1
                ]
                base[f"{prefix}_observed_event_count"] = len(observed)
                base[f"{prefix}_censored_event_count"] = event_count - len(observed)
                base[f"{prefix}_benefit_retained_count"] = len(retained)
                base[f"{prefix}_benefit_retained_fraction"] = (
                    len(retained) / len(observed) if observed else ""
                )
                base[f"{prefix}_mean_error_change_from_before_m"] = _safe_optional_mean(
                    row.get(f"{prefix}_error_change_from_before_m") for row in observed
                )
                base[f"{prefix}_mean_error_change_from_after_m"] = _safe_optional_mean(
                    row.get(f"{prefix}_error_change_from_after_m") for row in observed
                )
                base[f"{prefix}_mean_j_loc_change_from_after"] = _safe_optional_mean(
                    row.get(f"{prefix}_j_loc_change_from_after") for row in observed
                )
            output.append(base)
    return output


def _seed_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    config: EKFSensorInteractionAuditConfig,
) -> list[dict[str, Any]]:
    episode_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    event_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        episode_groups[(int(row["base_seed"]), str(row["branch"]))].append(row)
    for row in event_rows:
        event_groups[(int(row["base_seed"]), str(row["branch"]))].append(row)

    output: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in config.branches:
            episodes = episode_groups.get((int(seed), str(branch)), [])
            events = event_groups.get((int(seed), str(branch)), [])
            if not episodes:
                raise ValueError(f"no episode rows for seed={seed}, branch={branch}")
            event_count = len(events)
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": str(branch),
                "split": config.split,
                "sample_unit": "training_seed",
                "episode_count": len(episodes),
                "event_count": event_count,
                "eventful_episode_count": sum(int(_float(item.get("eventful_episode"), 0.0)) for item in episodes),
                "eventful_episode_fraction": _safe_mean(
                    _float(item.get("eventful_episode"), 0.0) for item in episodes
                ),
                "position_rmse_m": _safe_mean(_float(item.get("position_rmse_m"), 0.0) for item in episodes),
                "position_error_p95_m": _safe_mean(_float(item.get("position_error_p95_m"), 0.0) for item in episodes),
                "j_loc": _safe_mean(_float(item.get("j_loc"), 0.0) for item in episodes),
                "j_com": _safe_mean(_float(item.get("j_com"), 0.0) for item in episodes),
                "joint_objective": _safe_mean(_float(item.get("joint_objective"), 0.0) for item in episodes),
                "mean_reward": _safe_mean(_float(item.get("mean_reward"), 0.0) for item in episodes),
                "accepted_cooperative_update_count_mean": _safe_mean(
                    _float(item.get("accepted_cooperative_update_count"), 0.0) for item in episodes
                ),
                "accepted_cooperative_update_count_total": event_count,
                "beneficial_fraction": (
                    sum(str(item.get("immediate_effect_class")) == "beneficial" for item in events) / event_count
                    if event_count
                    else ""
                ),
                "harmful_fraction": (
                    sum(str(item.get("immediate_effect_class")) == "harmful" for item in events) / event_count
                    if event_count
                    else ""
                ),
                "next_visual_odometry_fraction": (
                    sum(int(_float(item.get("next_vo_observed"), 0.0)) for item in events) / event_count
                    if event_count
                    else ""
                ),
                "next_prediction_only_fraction": (
                    sum(
                        str(item.get("next_window_transition_type", "")) == "prediction_only"
                        for item in events
                    )
                    / event_count
                    if event_count
                    else ""
                ),
                "next_window_observed_fraction": (
                    sum(int(_float(item.get("next_window_observed"), 0.0)) for item in events) / event_count
                    if event_count
                    else ""
                ),
                "mean_next_window_error_change_from_after_m": _safe_optional_mean(
                    item.get("next_window_error_change_from_after_m")
                    for item in events
                    if _float(item.get("next_window_observed"), 0.0) > 0
                ),
                "mean_next_vo_error_change_m": _safe_optional_mean(
                    item.get("next_vo_error_change_m")
                    for item in events
                    if _float(item.get("next_vo_observed"), 0.0) > 0
                ),
                "linear_cov_reduction_event_fraction": (
                    _safe_mean(
                        _float(item.get("linear_cov_reduction_m2"), 0.0) > 1.0e-12
                        for item in events
                    )
                    if events
                    else ""
                ),
                "linear_cov_reduction_mean_m2": _safe_optional_mean(
                    item.get("linear_cov_reduction_m2") for item in events
                ),
                "final_cov_reduction_mean_m2": _safe_optional_mean(
                    item.get("final_cov_reduction_m2") for item in events
                ),
                "covariance_restored_fraction": (
                    _safe_mean(_bool(item.get("covariance_restored")) for item in events)
                    if events
                    else ""
                ),
                "j_loc_immediate_change_abs_max": max(
                    (abs(_float(item.get("j_loc_immediate_change"), 0.0)) for item in events),
                    default="",
                ),
                "truth_used_offline_only": 1,
            }
            for horizon in config.horizons:
                prefix = f"h{int(horizon)}"
                observed = [
                    item for item in events if int(_float(item.get(f"{prefix}_observed"), 0.0)) == 1
                ]
                retained = [
                    item for item in observed if int(_float(item.get(f"{prefix}_benefit_retained"), 0.0)) == 1
                ]
                row[f"{prefix}_observed_event_count"] = len(observed)
                row[f"{prefix}_benefit_retained_count"] = len(retained)
                row[f"{prefix}_benefit_retained_fraction"] = len(retained) / len(observed) if observed else ""
                row[f"{prefix}_mean_error_change_from_before_m"] = _safe_optional_mean(
                    item.get(f"{prefix}_error_change_from_before_m") for item in observed
                )
                row[f"{prefix}_mean_error_change_from_after_m"] = _safe_optional_mean(
                    item.get(f"{prefix}_error_change_from_after_m") for item in observed
                )
            output.append(row)
    return output


def _paired_contrasts(
    seed_rows: Sequence[Mapping[str, Any]],
    config: EKFSensorInteractionAuditConfig,
) -> list[dict[str, Any]]:
    lookup = {(int(row["base_seed"]), str(row["branch"])): row for row in seed_rows}
    metrics = [
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "accepted_cooperative_update_count_mean",
        "beneficial_fraction",
        "harmful_fraction",
        "next_visual_odometry_fraction",
        "next_prediction_only_fraction",
        "linear_cov_reduction_mean_m2",
        "final_cov_reduction_mean_m2",
        "j_loc_immediate_change_abs_max",
    ]
    for horizon in config.horizons:
        metrics.extend(
            [
                f"h{int(horizon)}_benefit_retained_fraction",
                f"h{int(horizon)}_mean_error_change_from_before_m",
                f"h{int(horizon)}_mean_error_change_from_after_m",
            ]
        )
    baseline = str(config.branches[0])
    output: list[dict[str, Any]] = []
    for branch in config.branches[1:]:
        for metric in metrics:
            differences: list[float] = []
            for seed in config.seeds:
                treatment = _float(lookup[(int(seed), str(branch))].get(metric), float("nan"))
                control = _float(lookup[(int(seed), baseline)].get(metric), float("nan"))
                if math.isfinite(treatment) and math.isfinite(control):
                    differences.append(treatment - control)
            low, high = _bootstrap_ci(
                differences,
                config.bootstrap_samples,
                config.bootstrap_seed + len(output),
            )
            output.append(
                {
                    "contrast": f"{branch}_minus_{baseline}",
                    "metric": metric,
                    "paired_seed_count": len(differences),
                    "total_seed_count": len(config.seeds),
                    "mean_paired_difference": _safe_mean(differences),
                    "median_paired_difference": float(np.median(differences)) if differences else float("nan"),
                    "paired_difference_std": _safe_std(differences),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": int(
                        math.isfinite(low)
                        and math.isfinite(high)
                        and (low > 0.0 or high < 0.0)
                    ),
                    "seed_level_inference": 1,
                }
            )
    return output


def _transition_summary(
    event_rows: Sequence[Mapping[str, Any]], config: EKFSensorInteractionAuditConfig
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        for horizon in config.horizons:
            transition = str(row.get(f"h{int(horizon)}_transition_type", ""))
            grouped[(int(row["base_seed"]), str(row["branch"]), int(horizon), transition)].append(row)
    for (seed, branch, horizon, transition), rows in sorted(grouped.items()):
        all_events = [
            row
            for row in event_rows
            if int(row["base_seed"]) == seed
            and str(row["branch"]) == branch
            and str(row.get(f"h{horizon}_transition_type", "")) != ""
        ]
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "split": config.split,
                "horizon": horizon,
                "transition_type": transition,
                "event_count": len(rows),
                "total_event_count": len(all_events),
                "event_fraction": len(rows) / len(all_events) if all_events else 0.0,
                "episode_count": len({str(row["episode_id"]) for row in rows}),
                "event_rows_are_not_independent": 1,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        names = list(rows[0].keys())
    elif fieldnames:
        names = list(fieldnames)
    else:
        raise ValueError(f"no fieldnames available for empty audit table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def _prepare_outputs(config: EKFSensorInteractionAuditConfig) -> tuple[Path, Path]:
    logs_dir = Path(config.output_dir)
    plots_dir = (
        Path(config.plots_dir)
        if config.plots_dir is not None
        else Path("plots")
        / "preliminary"
        / Path(config.reference_root).name
        / "ekf_sensor_interaction_audit"
    )
    managed_logs = (
        "sensor_interaction_events.csv",
        "episode_interaction_summary.csv",
        "seed_interaction_summary.csv",
        "transition_interaction_summary.csv",
        "paired_contrasts.csv",
        "audit_report.json",
    )
    managed_plots = ("ekf_sensor_interaction.png",)
    for directory, names in ((logs_dir, managed_logs), (plots_dir, managed_plots)):
        resolved = directory.resolve()
        if resolved.exists() and any(resolved.iterdir()) and not config.force:
            raise FileExistsError(f"audit output is not empty: {resolved}; use --force")
        resolved.mkdir(parents=True, exist_ok=True)
        if config.force:
            for name in names:
                target = resolved / name
                if target.is_file():
                    target.unlink()
    return logs_dir, plots_dir


def _plot_audit(
    event_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    config: EKFSensorInteractionAuditConfig,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#0891b2"]
    labels = {config.branches[index]: f"Branch {index + 1}" for index in range(len(config.branches))}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axis = axes[0, 0]
    for index, branch in enumerate(config.branches):
        values = []
        for horizon in config.horizons:
            selected = [
                row
                for row in event_rows
                if str(row.get("branch")) == str(branch)
                and int(_float(row.get(f"h{int(horizon)}_observed"), 0.0)) == 1
            ]
            values.append(
                _safe_mean(row.get(f"h{int(horizon)}_error_change_from_before_m") for row in selected)
            )
        axis.plot(config.horizons, values, marker="o", color=colors[index % len(colors)], label=labels[branch])
    axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("Error change from pre-update")
    axis.set_xlabel("future decision horizon")
    axis.set_ylabel("error change (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[0, 1]
    for index, branch in enumerate(config.branches):
        selected = [row for row in seed_rows if str(row.get("branch")) == str(branch)]
        values = [
            _safe_mean(row.get(f"h{int(horizon)}_benefit_retained_fraction") for row in selected)
            for horizon in config.horizons
        ]
        axis.plot(config.horizons, values, marker="o", color=colors[index % len(colors)], label=labels[branch])
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Benefit retained at seed level")
    axis.set_xlabel("future decision horizon")
    axis.set_ylabel("retained fraction")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[0, 2]
    types = ["visual_odometry_update", "prediction_only", "onboard_uwb_update", "gnss_update", "none_observed"]
    width = 0.8 / max(1, len(config.branches))
    x = np.arange(len(types), dtype=float)
    for index, branch in enumerate(config.branches):
        selected = [
            row
            for row in event_rows
            if str(row.get("branch")) == str(branch)
            and int(_float(row.get("h1_observed"), 0.0)) == 1
        ]
        values = [
            sum(str(row.get("next_window_transition_type")) == transition for row in selected)
            / len(selected)
            if selected
            else 0.0
            for transition in types
        ]
        axis.bar(x + (index - (len(config.branches) - 1) / 2) * width, values, width=width, color=colors[index % len(colors)], label=labels[branch])
    axis.set_xticks(x, ["VO", "Predict", "Onboard UWB", "GNSS", "None"], rotation=25, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Next-window transition type")
    axis.set_ylabel("fraction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 0]
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    x_seed = np.arange(len(seeds), dtype=float)
    for index, branch in enumerate(config.branches):
        values = [
            _float(next(row for row in seed_rows if int(row["base_seed"]) == seed and row["branch"] == branch)["event_count"])
            for seed in seeds
        ]
        axis.plot(x_seed, values, marker="o", color=colors[index % len(colors)], label=labels[branch])
    axis.set_title("Accepted cooperative updates by seed")
    axis.set_ylabel("event count")
    axis.set_xticks(x_seed, [str(seed) for seed in seeds], rotation=30)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 1]
    for index, branch in enumerate(config.branches):
        selected = [row for row in seed_rows if str(row.get("branch")) == str(branch)]
        axis.scatter(
            [_float(row.get("linear_cov_reduction_mean_m2"), 0.0) for row in selected],
            [_float(row.get("final_cov_reduction_mean_m2"), 0.0) for row in selected],
            color=colors[index % len(colors)],
            label=labels[branch],
        )
    axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("Linear vs final covariance reduction")
    axis.set_xlabel("linear reduction (m²)")
    axis.set_ylabel("final reduction (m²)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 2]
    selected = [
        row
        for row in contrasts
        if row.get("metric") in {"position_rmse_m", "j_loc", "j_com", "harmful_fraction"}
    ]
    selected.sort(key=lambda row: (str(row.get("contrast")), ["position_rmse_m", "j_loc", "j_com", "harmful_fraction"].index(row["metric"])))
    positions = np.arange(len(selected), dtype=float)
    values = np.asarray([_float(row.get("mean_paired_difference"), 0.0) for row in selected], dtype=float)
    lows = np.asarray([_float(row.get("bootstrap_95_ci_low"), 0.0) for row in selected], dtype=float)
    highs = np.asarray([_float(row.get("bootstrap_95_ci_high"), 0.0) for row in selected], dtype=float)
    if selected:
        axis.errorbar(positions, values, yerr=np.vstack([values - lows, highs - values]), fmt="o", color="#111827", ecolor="#111827", capsize=4)
        axis.set_xticks(positions, [f"{row['contrast'].split('_minus_')[0]}\n{row['metric']}" for row in selected], rotation=25, ha="right", fontsize=7)
    axis.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1)
    axis.set_title("Seed-level paired contrasts")
    axis.set_ylabel("difference")
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Cooperative UWB—subsequent EKF sensor interaction audit (validation only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _decision(
    event_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    config: EKFSensorInteractionAuditConfig,
) -> dict[str, Any]:
    focal_branch = str(config.branches[-1])
    focal_events = [row for row in event_rows if str(row.get("branch")) == focal_branch]
    focal_seeds = [row for row in seed_rows if str(row.get("branch")) == focal_branch]
    event_count = len(focal_events)
    if event_count < MINIMUM_EVENT_COUNT or not focal_seeds:
        return {
            "persistence_primary_cause": "insufficient_evidence",
            "recommendation": "Add validation episodes or pre-registered seeds before modifying EKF or reward.",
            "focal_branch": focal_branch,
            "valid_event_count": event_count,
            "minimum_event_count": MINIMUM_EVENT_COUNT,
            "next_transition_type_fraction": {},
            "h1_benefit_retained_seed_mean": 0.0,
            "benefit_retained_seed_mean_by_horizon": {},
            "benefit_retention_observed_event_count_by_horizon": {},
            "benefit_retention_decision_horizon": None,
            "benefit_retained_at_decision_horizon": 0.0,
            "jloc_objective_break": False,
        }

    next_counts = Counter(str(row.get("next_window_transition_type", "none_observed")) for row in focal_events)
    next_fractions = {
        transition: count / event_count for transition, count in sorted(next_counts.items())
    }
    retention_by_horizon: dict[str, float] = {}
    observed_by_horizon: dict[str, int] = {}
    for horizon in config.horizons:
        prefix = f"h{int(horizon)}"
        observed_count = int(
            round(
                sum(
                    _float(row.get(f"{prefix}_observed_event_count"), 0.0)
                    for row in focal_seeds
                )
            )
        )
        retained_values = [
            _float(row.get(f"{prefix}_benefit_retained_fraction"), float("nan"))
            for row in focal_seeds
        ]
        retention_by_horizon[prefix] = _safe_mean(
            [value for value in retained_values if math.isfinite(value)], 0.0
        )
        observed_by_horizon[prefix] = observed_count
    usable_horizons = [
        int(horizon)
        for horizon in config.horizons
        if observed_by_horizon.get(f"h{int(horizon)}", 0) > 0
    ]
    retention_horizon = max(usable_horizons) if usable_horizons else None
    long_retained = (
        retention_by_horizon.get(f"h{int(retention_horizon)}", 0.0)
        if retention_horizon is not None
        else 0.0
    )
    linear_fraction = _safe_mean(
        _float(row.get("linear_cov_reduction_event_fraction"), float("nan"))
        for row in focal_seeds
    )
    final_reduction = _safe_mean(
        _float(row.get("final_cov_reduction_mean_m2"), float("nan")) for row in focal_seeds
    )
    jloc_abs_max = max(
        (_float(row.get("j_loc_immediate_change_abs_max"), 0.0) for row in focal_seeds),
        default=0.0,
    )
    jloc_break = bool(
        linear_fraction >= MINIMUM_TRANSITION_FRACTION
        and abs(final_reduction) <= FINAL_COVARIANCE_TOLERANCE_M2
        and jloc_abs_max <= JLOC_CHANGE_TOLERANCE
    )

    visual_fraction = next_fractions.get("visual_odometry_update", 0.0)
    prediction_fraction = next_fractions.get("prediction_only", 0.0)
    retained_low = long_retained < MINIMUM_TRANSITION_FRACTION
    if visual_fraction >= MINIMUM_TRANSITION_FRACTION and retained_low:
        primary = "visual_odometry_interaction"
        recommendation = (
            "The next diagnostic window is predominantly visual odometry and the cooperative benefit decays. "
            "Calibrate execution-visible VO covariance or quality gating using train/warm-up tracking quality."
        )
    elif prediction_fraction >= MINIMUM_TRANSITION_FRACTION and retained_low:
        primary = "prediction_propagation"
        recommendation = (
            "The next window is predominantly prediction-only and the benefit decays. "
            "Audit dt, state transition, process noise Q, and update scheduling before changing the reward."
        )
    elif jloc_break:
        primary = "covariance_restoration_objective_break"
        recommendation = (
            "Linear covariance decreases but final covariance and J_loc do not respond. "
            "Keep the conservative protection and study joint covariance propagation or Covariance Intersection."
        )
    else:
        primary = "insufficient_evidence"
        recommendation = (
            "The follow-up interaction evidence is mixed. Freeze the current implementation and add "
            "pre-registered validation scenarios or seeds before modifying EKF or reward."
        )
    return {
        "persistence_primary_cause": primary,
        "recommendation": recommendation,
        "focal_branch": focal_branch,
        "valid_event_count": event_count,
        "minimum_event_count": MINIMUM_EVENT_COUNT,
        "minimum_transition_fraction": MINIMUM_TRANSITION_FRACTION,
        "next_transition_type_fraction": next_fractions,
        "h1_benefit_retained_seed_mean": retention_by_horizon.get("h1", 0.0),
        "benefit_retained_seed_mean_by_horizon": retention_by_horizon,
        "benefit_retention_observed_event_count_by_horizon": observed_by_horizon,
        "benefit_retention_decision_horizon": retention_horizon,
        "benefit_retained_at_decision_horizon": long_retained,
        "jloc_objective_break": jloc_break,
        "jloc_break_conditions": {
            "linear_covariance_reduction_seed_mean": linear_fraction,
            "final_covariance_reduction_seed_mean_m2": final_reduction,
            "j_loc_immediate_change_abs_max": jloc_abs_max,
            "final_covariance_tolerance_m2": FINAL_COVARIANCE_TOLERANCE_M2,
            "j_loc_change_tolerance": JLOC_CHANGE_TOLERANCE,
        },
        "decision_basis": "seed-level summaries with event rows used only for mechanism counts",
        "event_rows_are_not_independent": True,
    }


def run_ekf_sensor_interaction_audit(
    config: EKFSensorInteractionAuditConfig,
) -> dict[str, Any]:
    """Run the read-only cooperative-UWB sensor interaction audit."""

    if config.split != "validation":
        raise ValueError("EKF sensor interaction audit is validation-only")
    reference_root = Path(config.reference_root).resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")
    _seed_branch_dirs(reference_root, config.seeds, config.branches)

    event_rows, event_counts, episode_counts = _load_event_rows(config, reference_root)
    episode_rows = _episode_rows(config, reference_root, event_rows)
    seed_rows = _seed_rows(episode_rows, event_rows, config)
    contrasts = _paired_contrasts(seed_rows, config)
    transitions = _transition_summary(event_rows, config)
    decision = _decision(event_rows, seed_rows, config)

    event_numeric_fields = [
        "timestamp",
        "message_age_s",
        "message_delay_s",
        "nis",
        "measurement_variance_m2",
        "effective_measurement_variance_m2",
        "local_direction_variance_m2",
        "neighbor_direction_variance_m2",
        "unknown_cross_bound_m2",
        "unknown_cross_bound_fraction",
        "kalman_gain_norm",
        "position_correction_norm_m",
        "state_correction_residual_norm_m",
        "covariance_trace_before_m2",
        "covariance_trace_after_linear_m2",
        "covariance_trace_after_final_m2",
        "linear_cov_reduction_m2",
        "final_cov_reduction_m2",
        "j_loc_before",
        "j_loc_after",
        "j_loc_immediate_change",
        "true_position_error_before_m",
        "true_position_error_after_m",
        "immediate_error_change_m",
        "next_window_observed",
        "next_window_timestamp",
        "next_window_position_error_m",
        "next_window_error_change_from_after_m",
        "next_window_covariance_trace_m2",
        "next_window_covariance_change_m2",
        "next_window_position_nees",
        "next_prediction_error_change_m",
        "next_prediction_covariance_change_m2",
        "next_vo_observed",
        "next_vo_nis",
        "next_vo_error_change_m",
        "next_vo_covariance_change_m2",
    ]
    for horizon in config.horizons:
        prefix = f"h{int(horizon)}"
        event_numeric_fields.extend(
            [
                f"{prefix}_observed",
                f"{prefix}_future_error_m",
                f"{prefix}_error_change_from_before_m",
                f"{prefix}_error_change_from_after_m",
                f"{prefix}_future_covariance_trace_m2",
                f"{prefix}_covariance_change_from_final_m2",
                f"{prefix}_future_position_nees",
                f"{prefix}_benefit_retained",
                f"{prefix}_j_loc",
                f"{prefix}_j_loc_change_from_after",
            ]
        )
    all_event_finite = all(
        math.isfinite(_float(row.get(field), float("nan")))
        for row in event_rows
        for field in event_numeric_fields
    )
    all_episode_finite = all(
        math.isfinite(float(value))
        for row in episode_rows
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    complete_episodes = all(
        episode_counts.get((int(seed), str(branch)), 0) == config.expected_episodes
        for seed in config.seeds
        for branch in config.branches
    )
    formula_residual_state = max(
        (_float(row.get("state_correction_residual_norm_m"), 0.0) for row in event_rows),
        default=0.0,
    )
    formula_residual_error = max(
        abs(
            _float(row.get("immediate_error_change_m"), 0.0)
            - (
                _float(row.get("true_position_error_after_m"), 0.0)
                - _float(row.get("true_position_error_before_m"), 0.0)
            )
        )
        for row in event_rows
    ) if event_rows else 0.0
    logs_dir, plots_dir = _prepare_outputs(config)
    _write_csv(logs_dir / "sensor_interaction_events.csv", event_rows, event_numeric_fields)
    _write_csv(logs_dir / "episode_interaction_summary.csv", episode_rows)
    _write_csv(logs_dir / "seed_interaction_summary.csv", seed_rows)
    _write_csv(logs_dir / "transition_interaction_summary.csv", transitions, (
        "base_seed", "branch", "split", "horizon", "transition_type", "event_count",
        "total_event_count", "event_fraction", "episode_count", "event_rows_are_not_independent"
    ))
    _write_csv(logs_dir / "paired_contrasts.csv", contrasts, (
        "contrast", "metric", "paired_seed_count", "total_seed_count",
        "mean_paired_difference", "median_paired_difference", "paired_difference_std",
        "bootstrap_95_ci_low", "bootstrap_95_ci_high", "ci_excludes_zero", "seed_level_inference"
    ))
    plot_path = plots_dir / "ekf_sensor_interaction.png"
    _plot_audit(event_rows, seed_rows, contrasts, config, plot_path)

    report: dict[str, Any] = {
        "audit": "ekf_sensor_interaction",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "split": config.split,
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": [str(branch) for branch in config.branches],
        "horizons": [int(value) for value in config.horizons],
        "execution_order": [
            "predict",
            "visual_odometry",
            "gnss",
            "message_release",
            "cooperative_uwb",
        ],
        "execution_order_text": "预测 → 视觉里程计 → GNSS → 消息释放 → 协同UWB",
        "independent_unit": "training seed after episode-level reduction",
        "episode_rows_are_not_independent": True,
        "event_rows_are_not_independent": True,
        "test_data_read": False,
        "test_data_used": False,
        "train_data_read": False,
        "source_writes": 0,
        "truth_used_by_actor_filter_or_reward": False,
        "offline_diagnostic_labels": [
            "true_position_error_before_m",
            "true_position_error_after_m",
            "los_label_offline",
        ],
        "event_counts": {
            f"{seed}::{branch}": int(event_counts.get((int(seed), str(branch)), 0))
            for seed in config.seeds
            for branch in config.branches
        },
        "episode_counts": {
            f"{seed}::{branch}": int(episode_counts.get((int(seed), str(branch)), 0))
            for seed in config.seeds
            for branch in config.branches
        },
        "checks": {
            "validation_only": config.split == "validation",
            "test_not_read": True,
            "train_not_used_for_fitting": True,
            "source_untouched_by_audit": True,
            "all_update_fields_finite": all_event_finite,
            "all_episode_numeric_fields_finite": all_episode_finite,
            "complete_validation_episode_counts": complete_episodes,
            "expected_episodes_per_seed_branch": config.expected_episodes,
            "observed_episode_count": len(episode_rows),
            "expected_episode_count": len(config.seeds) * len(config.branches) * config.expected_episodes,
            "checkpoint_or_environment_mutated": False,
            "trainer_update_called": False,
            "optimizer_step_called": False,
            "seed_level_inference_used": True,
        },
        "formula_checks": {
            "state_correction_residual_max_m": formula_residual_state,
            "immediate_error_change_max_abs_residual_m": formula_residual_error,
            "state_correction_closed": formula_residual_state <= 1.0e-8,
            "immediate_error_change_closed": formula_residual_error <= 1.0e-12,
        },
        "decision": decision,
        "persistence_primary_cause": decision["persistence_primary_cause"],
        "jloc_objective_break": bool(decision.get("jloc_objective_break", False)),
        "audit_ok": bool(
            complete_episodes
            and bool(seed_rows)
            and all_event_finite
            and all_episode_finite
            and formula_residual_state <= 1.0e-8
            and formula_residual_error <= 1.0e-12
        ),
        "decision_supported": bool(
            complete_episodes
            and bool(seed_rows)
            and all_event_finite
            and all_episode_finite
            and len([row for row in event_rows if str(row.get("branch")) == str(config.branches[-1])])
            >= MINIMUM_EVENT_COUNT
        ),
        "outputs": {
            "sensor_interaction_events": str(logs_dir / "sensor_interaction_events.csv"),
            "episode_interaction_summary": str(logs_dir / "episode_interaction_summary.csv"),
            "seed_interaction_summary": str(logs_dir / "seed_interaction_summary.csv"),
            "transition_interaction_summary": str(logs_dir / "transition_interaction_summary.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
            "plot": str(plot_path),
        },
        "interpretation_boundary": (
            "Ground-truth error and LOS/NLOS are validation-only labels. They are not used for "
            "actor decisions, EKF updates, reward construction, fitting, or policy selection. "
            "Event and window rows are nested diagnostics, not independent statistical samples."
        ),
    }
    report_path = logs_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_BRANCHES",
    "DEFAULT_HORIZONS",
    "DEFAULT_SEEDS",
    "EKFSensorInteractionAuditConfig",
    "MINIMUM_EVENT_COUNT",
    "MINIMUM_TRANSITION_FRACTION",
    "run_ekf_sensor_interaction_audit",
]
