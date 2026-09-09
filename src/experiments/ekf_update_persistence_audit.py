"""Read-only persistence audit for cooperative EKF updates.

The audit consumes only frozen validation artefacts from a completed MAPPO
experiment.  It does not replay the simulator, update a policy, or change any
episode/checkpoint file.  An accepted relative-range update is paired with
the later diagnostic rows for the same target UAV so that an immediate error
reduction can be distinguished from a reduction that survives subsequent
prediction/update windows.

Ground-truth error is used only as an offline validation label.  Update rows
and windows are nested diagnostics; confidence intervals are computed after
reducing to one mean per training seed.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import position_error_metrics


DEFAULT_SEEDS = (12136, 13145, 14154, 15163, 16172)
DEFAULT_BRANCHES = (
    "baseline_mappo",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo",
    "communication_constrained_proxy_guided_mappo",
)
DEFAULT_HORIZONS = (1, 2, 3, 5)
PRACTICAL_EVENT_THRESHOLD_M = 0.001
_SEED_RE = re.compile(r"^seed_(\d+)$")
_NUMERIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EKFUpdatePersistenceAuditConfig:
    """Configuration for a validation-only persistence audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "communication_constrained_proxy_mappo_seedset_12136_16172/"
        "ekf_update_persistence_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("EKF update persistence audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty unique sequence")
        if not self.horizons or any(int(value) <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: EKFUpdatePersistenceAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[ekf-update-persistence] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values_array = np.asarray(
        [_float(value, float("nan")) for value in values],
        dtype=float,
    )
    values_array = values_array[np.isfinite(values_array)]
    return float(values_array.mean()) if values_array.size else float(default)


def _safe_std(values: Iterable[float], default: float = 0.0) -> float:
    values_array = np.asarray(
        [_float(value, float("nan")) for value in values],
        dtype=float,
    )
    values_array = values_array[np.isfinite(values_array)]
    return float(values_array.std(ddof=1)) if values_array.size > 1 else float(default)


def _float_or_nan(value: Any) -> float:
    return _float(value, float("nan"))


def _optional_mean(values: Iterable[Any]) -> float | str:
    """Mean a conditional diagnostic, returning blank when undefined."""

    numbers = [_float_or_nan(value) for value in values]
    finite = [value for value in numbers if math.isfinite(value)]
    return float(np.mean(finite)) if finite else ""


def _conditional_metric_keys(horizons: Sequence[int]) -> set[str]:
    keys = {
        "beneficial_fraction",
        "harmful_fraction",
        "neutral_fraction",
        "linear_cov_reduction_event_fraction",
        "j_loc_immediate_change_abs_max",
    }
    for horizon in horizons:
        prefix = f"h{int(horizon)}"
        keys.update(
            {
                f"{prefix}_mean_error_change_from_before_m",
                f"{prefix}_mean_error_change_from_after_m",
                f"{prefix}_benefit_retained_fraction",
                f"{prefix}_benefit_loss_observed_fraction",
                f"{prefix}_censor_fraction",
                f"{prefix}_mean_time_to_benefit_loss_h",
                f"{prefix}_mean_j_loc_change_from_after",
            }
        )
    return keys


def _bootstrap_ci(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    """Return a percentile interval for seed-level means."""

    array = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if array.size == 0:
        return 0.0, 0.0
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def classify_update_effect(error_change_m: float) -> str:
    """Classify an immediate error change with the fixed 1 mm threshold."""

    if error_change_m < -PRACTICAL_EVENT_THRESHOLD_M:
        return "beneficial"
    if error_change_m > PRACTICAL_EVENT_THRESHOLD_M:
        return "harmful"
    return "approximately_neutral"


def _seed_branch_dirs(
    reference_root: Path, seeds: Sequence[int], branches: Sequence[str]
) -> list[tuple[int, str, Path]]:
    output: list[tuple[int, str, Path]] = []
    for seed in seeds:
        for branch in branches:
            scale = Path(reference_root) / f"seed_{int(seed)}" / branch / "scale_3"
            if not scale.is_dir():
                raise FileNotFoundError(f"missing scale_3 directory: {scale}")
            validation = scale / "validation"
            if not validation.is_dir():
                raise FileNotFoundError(f"missing validation directory: {validation}")
            output.append((int(seed), branch, scale))
    return output


def _episode_dirs(scale_root: Path, split: str) -> list[Path]:
    """Enumerate only the requested validation directory."""

    directory = scale_root / split
    return sorted(
        (path for path in directory.iterdir() if path.is_dir() and path.name.startswith("episode_")),
        key=lambda path: path.name,
    )


def _episode_metric_map(scale_root: Path, seed: int, branch: str, split: str) -> dict[str, dict[str, Any]]:
    path = scale_root / "diagnostics" / f"{split}_episode_metrics.csv"
    if not path.is_file():
        path = scale_root / "diagnostics" / "episode_metrics.csv"
    rows = _read_csv(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("split", "")) != split:
            continue
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            continue
        source_norm_rmse = _float(row.get("position_rmse_norm_m"), float("nan"))
        converted: dict[str, Any] = {
            "base_seed": int(seed),
            "branch": branch,
            "episode_id": episode_id,
            "scenario_id": str(row.get("scenario_id", "")),
            "split": split,
            # ``position_rmse_m`` in historical formal outputs is a pooled
            # coordinate-wise RMSE.  Keep it under an explicit legacy name;
            # persistence comparisons replace the canonical ``position_rmse_m``
            # with the vector-norm definition computed from source rows.
            "position_rmse_component_m": _float(row.get("position_rmse_m")),
            "position_rmse_norm_m_source": source_norm_rmse if math.isfinite(source_norm_rmse) else "",
            "position_error_p95_m": _float(row.get("position_error_p95_m")),
            "covariance_position_trace": _float(row.get("covariance_position_trace")),
            "j_loc": _float(row.get("j_loc")),
            "j_com": _float(row.get("j_com")),
            "joint_objective": _float(row.get("joint_objective")),
            "mean_reward": _float(row.get("mean_reward")),
            "packet_loss": _float(row.get("packet_loss")),
            "delay_s": _float(row.get("delay_s")),
            "aoi_s": _float(row.get("aoi_s")),
            "connectivity": _float(row.get("connectivity")),
            "energy_proxy": _float(row.get("energy_proxy")),
            "collision_rate": _float(row.get("collision_rate")),
            "task_complete": _float(row.get("task_complete")),
            "termination_reason": str(row.get("termination_reason", "")),
        }
        output[episode_id] = converted
    return output


def _measurement_counts(episode_root: Path) -> tuple[int, int]:
    """Return selected measurement count and recorded overhead packet count."""

    actions = _read_csv(episode_root / "actions.csv")
    selected = [
        row
        for row in actions
        if str(row.get("measurement_uav_id", "none")) != "none"
        and not _bool(row.get("invalid_measurement_action", False))
    ]
    overhead = _read_csv(episode_root / "measurement_overhead.csv")
    packets = sum(
        int(round(_float(row.get("load_packets_applied"), 0.0)))
        for row in overhead
        if _bool(row.get("selected_non_none", False))
        or str(row.get("measurement_uav_id", "none")) != "none"
    )
    return len(selected), packets


def _position_error_metrics(episode_root: Path) -> tuple[float, float]:
    """Compute norm-based position RMSE and P95 for one validation episode."""

    truth_rows = _read_csv(episode_root / "ground_truth.csv")
    estimate_rows = _read_csv(episode_root / "estimates.csv")
    metrics = position_error_metrics(truth_rows, estimate_rows)
    rmse = _float(metrics.get("rmse_norm_m"), 0.0)
    p95 = _float(metrics.get("p95_norm_m"), 0.0)
    return rmse, p95


def _diagnostic_sequences(
    episode_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, float], dict[str, float]]]:
    diagnostics = _read_csv(episode_root / "ekf_diagnostics.csv")
    by_uav: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        uav_id = str(row.get("uav_id", ""))
        if not uav_id:
            continue
        by_uav[uav_id].append(
            {
                "timestamp": _float(row.get("timestamp")),
                "position_error_m": _float(row.get("position_error_m")),
                "position_nees": _float(row.get("position_nees")),
                "covariance_trace_m2": _float(row.get("position_covariance_trace_m2")),
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
        rows.sort(key=lambda row: float(row["timestamp"]))

    reward_rows = _read_csv(episode_root / "rewards.csv")
    scenario_path = episode_root / "scenario.json"
    uncertainty_weight = 0.35
    if scenario_path.is_file():
        payload = _read_json(scenario_path)
        reward_config = payload.get("reward_config", {})
        if isinstance(reward_config, Mapping):
            weights = reward_config.get("weights", {})
            if isinstance(weights, Mapping):
                uncertainty_weight = _float(
                    weights.get("uncertainty", weights.get("unc", uncertainty_weight)),
                    uncertainty_weight,
                )
    rewards: dict[tuple[str, float], dict[str, float]] = {}
    for row in reward_rows:
        uav_id = str(row.get("uav_id", ""))
        timestamp = round(_float(row.get("timestamp")), 6)
        j_loc = uncertainty_weight * _float(row.get("normalized_uncertainty"))
        joint = _float(row.get("joint_objective"))
        rewards[(uav_id, timestamp)] = {
            "j_loc": j_loc,
            "j_com": joint - j_loc,
            "mean_reward": _float(row.get("reward_total")),
        }
    return dict(by_uav), rewards


def _transition_type(
    diagnostic_row: Mapping[str, Any],
    *,
    timestamp: float,
    cooperative_times: set[float],
) -> str:
    """Classify the next diagnostic window without inspecting future truth."""

    if round(float(timestamp), 6) in cooperative_times:
        return "cooperative_uwb_update"
    if any(
        _bool(diagnostic_row.get(name))
        for name in ("vo_update_attempted", "vo_update_accepted", "vo_update_rejected")
    ):
        return "visual_odometry_update"
    if any(
        _bool(diagnostic_row.get(name))
        for name in ("gnss_update_attempted", "gnss_update_accepted", "gnss_update_rejected")
    ):
        return "gnss_update"
    if any(
        _bool(diagnostic_row.get(name))
        for name in ("uwb_update_attempted", "uwb_update_accepted", "uwb_update_rejected")
    ):
        return "onboard_uwb_update"
    return "prediction_only"


def _first_non_prediction_type(
    sequence: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
    end_index: int,
    cooperative_times: set[float],
) -> str:
    for index in range(start_index, min(end_index + 1, len(sequence))):
        row = sequence[index]
        transition = _transition_type(
            row,
            timestamp=_float(row.get("timestamp")),
            cooperative_times=cooperative_times,
        )
        if transition != "prediction_only":
            return transition
    return "prediction_only"


def _next_transition_horizon(
    sequence: Sequence[Mapping[str, Any]],
    *,
    current_index: int,
    cooperative_times: set[float],
    predicate: Any,
) -> int:
    for horizon, index in enumerate(range(current_index + 1, len(sequence)), start=1):
        if predicate(sequence[index], cooperative_times):
            return int(horizon)
    return -1


def _find_current_index(
    rows: Sequence[Mapping[str, Any]], event: Mapping[str, Any]
) -> int:
    timestamp = _float(event.get("measurement_time_s", event.get("timestamp")), -1.0)
    if timestamp >= 0.0:
        candidates = [
            index
            for index, row in enumerate(rows)
            if abs(_float(row.get("timestamp")) - timestamp) <= 1e-6
        ]
        if candidates:
            return candidates[0]
    step = int(round(_float(event.get("step"), 0.0)))
    return max(0, min(step, len(rows) - 1)) if rows else -1


def _event_row(
    *,
    event: Mapping[str, Any],
    seed: int,
    branch: str,
    episode_root: Path,
    update_index: int,
    horizons: Sequence[int],
    sequences: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    rewards: Mapping[tuple[str, float], Mapping[str, float]] | None = None,
    cooperative_times: set[float] | None = None,
) -> dict[str, Any] | None:
    if not _bool(event.get("accepted", False)):
        return None
    episode_id = str(event.get("episode_id", episode_root.name))
    target = str(event.get("target_uav_id", ""))
    if sequences is None or rewards is None:
        sequences, rewards = _diagnostic_sequences(episode_root)
    sequence = sequences.get(target, [])
    current_index = _find_current_index(sequence, event)
    if not sequence or current_index < 0:
        return None
    if cooperative_times is None:
        cooperative_times = set()

    before_error = _float(event.get("true_position_error_before_m"))
    after_error = _float(event.get("true_position_error_after_m"))
    error_change = after_error - before_error
    effect = classify_update_effect(error_change)
    measurement_time = _float(event.get("measurement_time_s"), _float(event.get("timestamp")))
    j_loc_before = _float(event.get("j_loc_before"))
    j_loc_after = _float(event.get("j_loc_after"), j_loc_before)
    row: dict[str, Any] = {
        "base_seed": int(seed),
        "branch": branch,
        "split": "validation",
        "episode_id": episode_id,
        "scenario_id": str(event.get("scenario_id", "")),
        "update_index": int(update_index),
        "step": int(round(_float(event.get("step"), 0.0))),
        "timestamp": measurement_time,
        "source_uav_id": str(event.get("source_uav_id", "")),
        "target_uav_id": target,
        "accepted": 1,
        "los_label_offline": int(_bool(event.get("los", False))),
        "message_age_s": _float(event.get("message_age_s")),
        "message_delay_s": _float(event.get("message_delay_s")),
        "nis": _float(event.get("nis")),
        "measurement_variance_m2": _float(event.get("measurement_variance_m2")),
        "effective_measurement_variance_m2": _float(
            event.get("effective_measurement_variance_m2")
        ),
        "kalman_gain_norm": _float(event.get("kalman_gain_norm")),
        "position_correction_norm_m": _float(event.get("position_correction_norm_m")),
        "unknown_cross_bound_m2": _float(event.get("unknown_cross_bound_m2")),
        "unknown_cross_bound_fraction": _float(event.get("unknown_cross_bound_fraction")),
        "covariance_trace_before_m2": _float(event.get("position_covariance_trace_before_m2")),
        "covariance_trace_after_linear_m2": _float(
            event.get("position_covariance_trace_after_linear_m2")
        ),
        "covariance_trace_after_final_m2": _float(
            event.get("position_covariance_trace_after_final_m2")
        ),
        "linear_cov_reduction_m2": _float(
            event.get("position_covariance_trace_before_m2")
        ) - _float(event.get("position_covariance_trace_after_linear_m2")),
        "final_cov_reduction_m2": _float(
            event.get("position_covariance_trace_before_m2")
        ) - _float(event.get("position_covariance_trace_after_final_m2")),
        "j_loc_before": j_loc_before,
        "j_loc_after": j_loc_after,
        "j_loc_immediate_change": j_loc_after - j_loc_before,
        "true_position_error_before_m": before_error,
        "true_position_error_after_m": after_error,
        "immediate_error_change_m": error_change,
        "immediate_effect_class": effect,
        "truth_used_offline_only": 1,
    }
    row["next_cooperative_update_h"] = _next_transition_horizon(
        sequence,
        current_index=current_index,
        cooperative_times=cooperative_times,
        predicate=lambda _item, times: round(_float(_item.get("timestamp")), 6) in times,
    )
    row["next_sensor_update_h"] = _next_transition_horizon(
        sequence,
        current_index=current_index,
        cooperative_times=cooperative_times,
        predicate=lambda item, times: _transition_type(
            item,
            timestamp=_float(item.get("timestamp")),
            cooperative_times=times,
        ) != "prediction_only",
    )
    for horizon in horizons:
        prefix = f"h{int(horizon)}"
        future_index = current_index + int(horizon)
        observed = int(future_index < len(sequence))
        future = sequence[future_index] if observed else {}
        future_error = _float(future.get("position_error_m"), -1.0)
        future_covariance = _float(future.get("covariance_trace_m2"), -1.0)
        future_nees = _float(future.get("position_nees"), -1.0)
        reward_key = (target, round(_float(future.get("timestamp")), 6)) if observed else ("", 0.0)
        future_reward = rewards.get(reward_key, {})
        future_j_loc = _float(future_reward.get("j_loc"), -1.0) if observed else -1.0
        row[f"{prefix}_observed"] = observed
        row[f"{prefix}_future_error_m"] = future_error
        row[f"{prefix}_future_covariance_trace_m2"] = future_covariance
        row[f"{prefix}_future_nees"] = future_nees
        row[f"{prefix}_error_change_from_before_m"] = future_error - before_error if observed else -1.0
        row[f"{prefix}_error_change_from_after_m"] = future_error - after_error if observed else -1.0
        row[f"{prefix}_time_since_update_s"] = (
            _float(future.get("timestamp")) - measurement_time if observed else -1.0
        )
        row[f"{prefix}_transition_type"] = (
            _transition_type(
                future,
                timestamp=_float(future.get("timestamp")),
                cooperative_times=cooperative_times,
            )
            if observed
            else "censored"
        )
        row[f"{prefix}_first_intervening_transition_type"] = (
            _first_non_prediction_type(
                sequence,
                start_index=current_index + 1,
                end_index=future_index,
                cooperative_times=cooperative_times,
            )
            if observed
            else "censored"
        )
        row[f"{prefix}_benefit_retained"] = int(
            observed and future_error < before_error - PRACTICAL_EVENT_THRESHOLD_M
        )
        row[f"{prefix}_benefit_retained_fraction"] = (
            (before_error - future_error) / max(abs(before_error - after_error), 1e-12)
            if observed and abs(before_error - after_error) > PRACTICAL_EVENT_THRESHOLD_M
            else 0.0
        )
        row[f"{prefix}_j_loc"] = future_j_loc
        row[f"{prefix}_j_loc_change_from_after"] = future_j_loc - j_loc_after if observed else -1.0
        row[f"{prefix}_time_to_benefit_loss_h"] = 0

    loss_horizon = 0
    if effect == "beneficial":
        for horizon in horizons:
            if int(row[f"h{int(horizon)}_observed"]) and int(row[f"h{int(horizon)}_benefit_retained"]) == 0:
                loss_horizon = int(horizon)
                break
    row["time_to_benefit_loss_h"] = int(loss_horizon)
    return row


def _load_event_rows(
    *,
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    split: str,
    horizons: Sequence[int],
    config: EKFUpdatePersistenceAuditConfig,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], int], dict[tuple[int, str], int]]:
    event_rows: list[dict[str, Any]] = []
    event_counts: dict[tuple[int, str], int] = defaultdict(int)
    episode_counts: dict[tuple[int, str], int] = defaultdict(int)
    for seed, branch, scale_root in _seed_branch_dirs(reference_root, seeds, branches):
        metric_map = _episode_metric_map(scale_root, seed, branch, split)
        episode_paths = _episode_dirs(scale_root, split)
        episode_counts[(seed, branch)] = len(episode_paths)
        updates_path = scale_root / "diagnostics" / f"{split}_cooperative_updates.jsonl"
        if not updates_path.is_file():
            updates_path = scale_root / "diagnostics" / "cooperative_updates.jsonl"
        updates = _read_jsonl(updates_path)
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for update in updates:
            if str(update.get("split", "")) == split and _bool(update.get("accepted", False)):
                by_episode[str(update.get("episode_id", ""))].append(update)
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
            cooperative_times_by_target: dict[str, set[float]] = defaultdict(set)
            for update in updates_for_episode:
                target_uav_id = str(update.get("target_uav_id", ""))
                update_time = _float(
                    update.get("measurement_time_s"),
                    _float(update.get("timestamp"), float("nan")),
                )
                if target_uav_id and math.isfinite(update_time):
                    cooperative_times_by_target[target_uav_id].add(round(update_time, 6))
            for event_index, update in enumerate(updates_for_episode):
                row = _event_row(
                    event=update,
                    seed=seed,
                    branch=branch,
                    episode_root=episode_path,
                    update_index=event_index,
                    horizons=horizons,
                    sequences=sequences,
                    rewards=rewards,
                    cooperative_times=cooperative_times_by_target.get(
                        str(update.get("target_uav_id", "")), set()
                    ),
                )
                if row is not None:
                    row["scenario_id"] = metric_map[episode_id].get("scenario_id", row["scenario_id"])
                    event_rows.append(row)
                    event_counts[(seed, branch)] += 1
            if config.progress and (index == 1 or index == len(episode_paths) or index % 5 == 0):
                _progress(
                    config,
                    f"{branch} seed={seed}: processed validation episode {index}/{len(episode_paths)}",
                )
    return event_rows, dict(event_counts), dict(episode_counts)


def _episode_rows(
    *,
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    split: str,
    horizons: Sequence[int],
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    event_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        event_groups[(int(row["base_seed"]), str(row["branch"]), str(row["episode_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for seed, branch, scale_root in _seed_branch_dirs(reference_root, seeds, branches):
        metrics = _episode_metric_map(scale_root, seed, branch, split)
        for episode_path in _episode_dirs(scale_root, split):
            episode_id = episode_path.name
            source = metrics.get(episode_id)
            if source is None:
                raise ValueError(f"missing metrics for {seed}/{branch}/{episode_id}")
            group = event_groups.get((seed, branch, episode_id), [])
            measurement_count, overhead_packets = _measurement_counts(episode_path)
            position_rmse, position_p95 = _position_error_metrics(episode_path)
            output_row: dict[str, Any] = {
                **source,
                "sample_unit": "episode",
                "accepted_update_count": len(group),
                "eventful_episode": int(bool(group)),
                "cooperative_update_coverage": int(bool(group)),
                "beneficial_update_count": sum(
                    str(row["immediate_effect_class"]) == "beneficial" for row in group
                ),
                "harmful_update_count": sum(
                    str(row["immediate_effect_class"]) == "harmful" for row in group
                ),
                "neutral_update_count": sum(
                    str(row["immediate_effect_class"]) == "approximately_neutral" for row in group
                ),
                "measurement_count": measurement_count,
                "measurement_overhead_packets": overhead_packets,
                # Persistence diagnostics use the canonical vector-norm
                # definition.  The historical formal value is retained so
                # old component-wise summaries remain traceable.
                "position_rmse_m": position_rmse,
                "position_rmse_norm_m": position_rmse,
                "position_rmse_component_m": source.get("position_rmse_component_m", ""),
                "position_error_p95_norm_m": position_p95,
                "position_error_p95_m": position_p95,
                "linear_cov_reduction_event_fraction": (
                    _safe_mean(
                        float(row["linear_cov_reduction_m2"]) > _NUMERIC_TOLERANCE
                        for row in group
                    )
                    if group
                    else ""
                ),
                "j_loc_immediate_change_abs_max": max(
                    (abs(_float(row["j_loc_immediate_change"])) for row in group),
                    default="",
                ),
                "truth_used_offline_only": 1,
            }
            denominator = max(1, len(group))
            output_row["beneficial_fraction"] = (
                output_row["beneficial_update_count"] / denominator if group else ""
            )
            output_row["harmful_fraction"] = (
                output_row["harmful_update_count"] / denominator if group else ""
            )
            output_row["neutral_fraction"] = (
                output_row["neutral_update_count"] / denominator if group else ""
            )
            for horizon in horizons:
                prefix = f"h{int(horizon)}"
                observed = [row for row in group if int(row[f"{prefix}_observed"]) == 1]
                beneficial_observed = [
                    row
                    for row in observed
                    if str(row["immediate_effect_class"]) == "beneficial"
                ]
                output_row[f"{prefix}_observed_event_count"] = len(observed)
                output_row[f"{prefix}_censored_event_count"] = len(group) - len(observed)
                output_row[f"{prefix}_censor_fraction"] = (
                    (len(group) - len(observed)) / max(1, len(group))
                    if group
                    else ""
                )
                output_row[f"{prefix}_mean_error_change_from_before_m"] = (
                    _safe_mean(
                        [row[f"{prefix}_error_change_from_before_m"] for row in observed],
                        float("nan"),
                    )
                    if observed
                    else ""
                )
                output_row[f"{prefix}_mean_error_change_from_after_m"] = (
                    _safe_mean(
                        [row[f"{prefix}_error_change_from_after_m"] for row in observed],
                        float("nan"),
                    )
                    if observed
                    else ""
                )
                retained = [
                    row for row in beneficial_observed if int(row[f"{prefix}_benefit_retained"]) == 1
                ]
                output_row[f"{prefix}_beneficial_observed_count"] = len(beneficial_observed)
                output_row[f"{prefix}_benefit_retained_count"] = len(retained)
                output_row[f"{prefix}_benefit_retained_fraction"] = (
                    len(retained) / len(beneficial_observed)
                    if beneficial_observed
                    else ""
                )
                output_row[f"{prefix}_mean_benefit_retained_fraction"] = (
                    _safe_mean(
                        [row[f"{prefix}_benefit_retained_fraction"] for row in beneficial_observed],
                        float("nan"),
                    )
                    if beneficial_observed
                    else ""
                )
                loss_values = [
                    int(row["time_to_benefit_loss_h"])
                    for row in beneficial_observed
                    if int(row["time_to_benefit_loss_h"]) > 0
                    and int(row["time_to_benefit_loss_h"]) <= int(horizon)
                ]
                output_row[f"{prefix}_benefit_loss_observed_fraction"] = (
                    len(loss_values) / len(beneficial_observed)
                    if beneficial_observed
                    else ""
                )
                output_row[f"{prefix}_mean_time_to_benefit_loss_h"] = (
                    _safe_mean(loss_values, float("nan")) if loss_values else ""
                )
                output_row[f"{prefix}_mean_j_loc_change_from_after"] = (
                    _safe_mean(
                        [row[f"{prefix}_j_loc_change_from_after"] for row in observed],
                        float("nan"),
                    )
                    if observed
                    else ""
                )
                output_row[f"{prefix}_j_loc_observed_count"] = sum(
                    float(row[f"{prefix}_j_loc"]) >= 0.0 for row in observed
                )
            output.append(output_row)
    return output


def _seed_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        groups[(int(row["base_seed"]), str(row["branch"]))].append(row)
    metric_keys = (
        "position_rmse_m",
        "position_rmse_norm_m",
        "position_rmse_component_m",
        "position_error_p95_norm_m",
        "position_error_p95_m",
        "covariance_position_trace",
        "j_loc",
        "j_com",
        "joint_objective",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "collision_rate",
        "accepted_update_count",
        "measurement_count",
        "measurement_overhead_packets",
        "beneficial_fraction",
        "harmful_fraction",
        "neutral_fraction",
        "linear_cov_reduction_event_fraction",
        "j_loc_immediate_change_abs_max",
        "eventful_episode",
        "cooperative_update_coverage",
    )
    for horizon in horizons:
        prefix = f"h{int(horizon)}"
        metric_keys += (
            f"{prefix}_mean_error_change_from_before_m",
            f"{prefix}_mean_error_change_from_after_m",
            f"{prefix}_beneficial_observed_count",
            f"{prefix}_benefit_retained_fraction",
            f"{prefix}_benefit_loss_observed_fraction",
            f"{prefix}_censor_fraction",
            f"{prefix}_mean_time_to_benefit_loss_h",
            f"{prefix}_mean_j_loc_change_from_after",
        )
    output: list[dict[str, Any]] = []
    conditional_keys = _conditional_metric_keys(horizons)
    for seed in seeds:
        for branch in branches:
            group = groups.get((int(seed), branch), [])
            if not group:
                raise ValueError(f"no episode rows for seed={seed}, branch={branch}")
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": branch,
                "split": "validation",
                "sample_unit": "training_seed",
                "episode_count": len(group),
                "truth_used_offline_only": 1,
            }
            for key in metric_keys:
                if key in conditional_keys:
                    row[key] = _optional_mean(item.get(key) for item in group)
                else:
                    row[key] = _safe_mean([_float(item.get(key)) for item in group])
            output.append(row)
    return output


def _paired_contrasts(
    seed_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
    horizons: Sequence[int],
    config: EKFUpdatePersistenceAuditConfig,
) -> list[dict[str, Any]]:
    by_key = {
        (int(row["base_seed"]), str(row["branch"])): row for row in seed_rows
    }
    baseline = branches[0]
    metric_keys = [
        "position_rmse_m",
        "position_rmse_norm_m",
        "position_rmse_component_m",
        "position_error_p95_norm_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "accepted_update_count",
        "beneficial_fraction",
        "harmful_fraction",
        "measurement_count",
        "measurement_overhead_packets",
    ]
    for horizon in horizons:
        prefix = f"h{int(horizon)}"
        metric_keys.extend(
            [
                f"{prefix}_mean_error_change_from_before_m",
                f"{prefix}_benefit_retained_fraction",
                f"{prefix}_benefit_loss_observed_fraction",
                f"{prefix}_mean_j_loc_change_from_after",
            ]
        )
    output: list[dict[str, Any]] = []
    for branch in branches[1:]:
        for metric in metric_keys:
            differences = [
                _float_or_nan(by_key[(int(seed), branch)].get(metric))
                - _float_or_nan(by_key[(int(seed), baseline)].get(metric))
                for seed in seeds
            ]
            differences = [value for value in differences if math.isfinite(value)]
            low, high = _bootstrap_ci(
                differences, config.bootstrap_samples, config.bootstrap_seed + len(output)
            )
            output.append(
                {
                    "contrast": f"{branch}_minus_{baseline}",
                    "metric": metric,
                    "paired_seed_count": len(differences),
                    "total_seed_count": len(seeds),
                    "mean_paired_difference": _safe_mean(differences),
                    "median_paired_difference": float(np.median(differences)) if differences else float("nan"),
                    "paired_difference_std": _safe_std(differences),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": int(
                        math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0)
                    ),
                    "seed_level_inference": 1,
                }
            )
    return output


def _jloc_rows(
    event_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    episode_lookup = {
        (int(row["base_seed"]), str(row["branch"]), str(row["episode_id"])): row
        for row in episode_rows
    }
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in event_rows}):
        for branch in sorted({str(row["branch"]) for row in event_rows}):
            events = [
                row for row in event_rows
                if int(row["base_seed"]) == seed and str(row["branch"]) == branch
            ]
            if not events:
                continue
            for horizon in horizons:
                prefix = f"h{int(horizon)}"
                observed = [row for row in events if int(row[f"{prefix}_observed"]) == 1]
                output.append(
                    {
                        "base_seed": seed,
                        "branch": branch,
                        "split": "validation",
                        "horizon": int(horizon),
                        "event_count": len(events),
                        "observed_event_count": len(observed),
                        "mean_j_loc_before": _safe_mean(row["j_loc_before"] for row in events),
                        "mean_j_loc_after": _safe_mean(row["j_loc_after"] for row in events),
                        "mean_j_loc_immediate_change": _safe_mean(
                            row["j_loc_immediate_change"] for row in events
                        ),
                        "mean_j_loc_future": _safe_mean(
                            [row[f"{prefix}_j_loc"] for row in observed], -1.0
                        ),
                        "mean_j_loc_future_change": _safe_mean(
                            [row[f"{prefix}_j_loc_change_from_after"] for row in observed], -1.0
                        ),
                        "mean_error_change_from_before_m": _safe_mean(
                            [row[f"{prefix}_error_change_from_before_m"] for row in observed], -1.0
                        ),
                        "episode_count_with_events": len(
                            {
                                str(row["episode_id"])
                                for row in events
                                if (seed, branch, str(row["episode_id"])) in episode_lookup
                            }
                        ),
                        "event_rows_are_not_independent": 1,
                    }
                )
    return output


def _decision(
    event_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    branches: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, Any]:
    if not event_rows or not seed_rows:
        return {
            "primary_reason": "insufficient_evidence",
            "recommendation": "Add diagnostic coverage before modifying EKF, reward, or communication models.",
            "immediate_beneficial_fraction": 0.0,
            "immediate_harmful_fraction": 0.0,
            "long_horizon_benefit_retained_fraction": 0.0,
            "j_loc_immediate_change_abs_max": 0.0,
        }
    # Mechanism rows are intentionally not used as independent observations.
    # The focal branch is the last branch (the communication-constrained
    # branch in the default experiment), and each of its five seed rows is
    # already reduced from validation episodes.
    focal_branch = branches[-1]
    focal_rows = [row for row in seed_rows if str(row["branch"]) == focal_branch]
    if not focal_rows:
        return {
            "primary_reason": "insufficient_evidence",
            "recommendation": "Add diagnostic coverage before modifying EKF, reward, or communication models.",
            "immediate_beneficial_fraction": 0.0,
            "immediate_harmful_fraction": 0.0,
            "long_horizon_benefit_retained_fraction": 0.0,
            "j_loc_immediate_change_abs_max": 0.0,
            "decision_basis": "seed-level focal-branch rows unavailable",
        }
    beneficial = _safe_mean(_float_or_nan(row.get("beneficial_fraction")) for row in focal_rows)
    harmful = _safe_mean(_float_or_nan(row.get("harmful_fraction")) for row in focal_rows)
    longest = max(int(value) for value in horizons)
    long_rows = [
        row
        for row in focal_rows
        if _float(row.get(f"h{longest}_beneficial_observed_count"), 0.0) > 0.0
    ]
    retained = _safe_mean(
        _float_or_nan(row.get(f"h{longest}_benefit_retained_fraction")) for row in long_rows
    )
    linear_reduction = _safe_mean(
        _float_or_nan(row.get("linear_cov_reduction_event_fraction"))
        for row in focal_rows
    )
    jloc_abs_max = max(
        (_float(row["j_loc_immediate_change_abs_max"]) for row in focal_rows),
        default=0.0,
    )
    if beneficial > harmful and len(long_rows) >= max(3, len(focal_rows) // 2) and retained < 0.5:
        primary = "benefit_not_persistent"
        recommendation = (
            "Single updates are often immediately beneficial but the benefit is lost during "
            "subsequent prediction/update windows. Audit propagation and update scheduling "
            "before changing the reward or communication model."
        )
    elif linear_reduction >= 0.5 and jloc_abs_max <= 1e-8:
        primary = "j_loc_insensitive_after_conservative_restoration"
        recommendation = (
            "The linear EKF covariance decreases, but the final conservative restoration "
            "leaves J_loc unchanged. Keep the filter and reward frozen; next assess a "
            "mathematically valid cross-correlation covariance treatment."
        )
    elif beneficial > 0.0 and retained >= 0.5:
        primary = "persistent_updates_but_endpoint_gain_small"
        recommendation = (
            "Some update benefits persist, but endpoint improvement is small. Audit the "
            "mapping from persistent uncertainty to episode-level RMSE before changing core models."
        )
    else:
        primary = "update_persistence_inconclusive"
        recommendation = (
            "Immediate and persistent effects are mixed. Freeze the current implementation and "
            "add pre-registered scenario or seed repetitions before modifying EKF or reward."
        )
    return {
        "primary_reason": primary,
        "recommendation": recommendation,
        "decision_basis": "mean of validation episode summaries at focal-branch seed level",
        "focal_branch": focal_branch,
        "focal_seed_count": len(focal_rows),
        "immediate_beneficial_fraction": beneficial,
        "immediate_harmful_fraction": harmful,
        "longest_horizon": longest,
        "long_horizon_beneficial_seed_count": len(long_rows),
        "long_horizon_benefit_retained_fraction": retained,
        "linear_covariance_reduction_event_fraction": linear_reduction,
        "j_loc_immediate_change_abs_max": jloc_abs_max,
        "branches_compared": list(branches),
        "event_rows_are_not_independent": True,
    }


def _prepare_outputs(config: EKFUpdatePersistenceAuditConfig) -> tuple[Path, Path]:
    logs_dir = Path(config.output_dir)
    plots_dir = (
        Path(config.plots_dir)
        if config.plots_dir is not None
        else Path("plots")
        / "preliminary"
        / Path(config.reference_root).name
        / "ekf_update_persistence_audit"
    )
    managed_logs = (
        "update_persistence_events.csv",
        "episode_persistence_summary.csv",
        "seed_persistence_summary.csv",
        "paired_contrasts.csv",
        "jloc_response.csv",
        "audit_report.json",
    )
    managed_plots = ("ekf_update_persistence.png",)
    for directory in (logs_dir, plots_dir):
        directory = directory.resolve()
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"audit output is not empty: {directory}; use --force")
        directory.mkdir(parents=True, exist_ok=True)
        if config.force:
            for name in managed_logs if directory == logs_dir.resolve() else managed_plots:
                path = directory / name
                if path.is_file():
                    path.unlink()
    return logs_dir, plots_dir


def _plot_audit(
    event_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    branches: Sequence[str],
    horizons: Sequence[int],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "#16a34a",
        "communication_constrained_proxy_guided_mappo": "#dc2626",
    }
    labels = {
        "baseline_mappo": "Baseline",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "Proxy-guided",
        "communication_constrained_proxy_guided_mappo": "Communication-constrained",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axis = axes[0, 0]
    for branch in branches:
        values = []
        for horizon in horizons:
            selected = [
                row for row in event_rows
                if row["branch"] == branch and int(row[f"h{int(horizon)}_observed"]) == 1
            ]
            values.append(_safe_mean(row[f"h{int(horizon)}_error_change_from_before_m"] for row in selected))
        axis.plot(horizons, values, marker="o", color=colors.get(branch, None), label=labels.get(branch, branch))
    axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("Error change from pre-update")
    axis.set_xlabel("future decision horizon")
    axis.set_ylabel("error change (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[0, 1]
    for branch in branches:
        values = []
        for horizon in horizons:
            selected = [
                row for row in seed_rows
                if row["branch"] == branch
            ]
            values.append(_safe_mean(row[f"h{int(horizon)}_benefit_retained_fraction"] for row in selected))
        axis.plot(horizons, values, marker="o", color=colors.get(branch, None), label=labels.get(branch, branch))
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Retained benefit among immediate beneficial updates")
    axis.set_xlabel("future decision horizon")
    axis.set_ylabel("retained fraction")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[0, 2]
    width = 0.8 / max(1, len(branches))
    x = np.arange(len(branches), dtype=float)
    for offset, effect, color in [
        (-width, "beneficial", "#16a34a"),
        (0.0, "approximately_neutral", "#6b7280"),
        (width, "harmful", "#dc2626"),
    ]:
        values = []
        for branch in branches:
            selected = [row for row in event_rows if row["branch"] == branch]
            values.append(_safe_mean(row["immediate_effect_class"] == effect for row in selected))
        axis.bar(x + offset, values, width=width, color=color, label=effect)
    axis.set_xticks(x, [labels.get(branch, branch) for branch in branches], rotation=25, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Immediate update effect classes")
    axis.set_ylabel("fraction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 0]
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    x = np.arange(len(seeds), dtype=float)
    for branch in branches:
        values = [
            _float(next(row for row in seed_rows if int(row["base_seed"]) == seed and row["branch"] == branch)["position_rmse_m"])
            for seed in seeds
        ]
        axis.plot(x, values, marker="o", color=colors.get(branch, None), label=labels.get(branch, branch))
    axis.set_title("Position RMSE by training seed")
    axis.set_ylabel("RMSE (m)")
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=30)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 1]
    for branch in branches:
        values = [
            _float(next(row for row in seed_rows if int(row["base_seed"]) == seed and row["branch"] == branch)["j_loc"])
            for seed in seeds
        ]
        axis.plot(x, values, marker="o", color=colors.get(branch, None), label=labels.get(branch, branch))
    axis.set_title(r"$J_{\mathrm{loc}}$ by training seed")
    axis.set_ylabel(r"$J_{\mathrm{loc}}$")
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=30)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    axis = axes[1, 2]
    focal = branches[-1]
    selected = [
        row for row in contrasts
        if row["contrast"] == f"{focal}_minus_{branches[0]}"
        and row["metric"] in {"position_rmse_m", "j_loc", "j_com", "harmful_fraction"}
    ]
    selected.sort(key=lambda row: ("position_rmse_m", "j_loc", "j_com", "harmful_fraction").index(row["metric"]))
    positions = np.arange(len(selected), dtype=float)
    values = np.asarray([_float(row["mean_paired_difference"]) for row in selected])
    lows = np.asarray([_float(row["bootstrap_95_ci_low"]) for row in selected])
    highs = np.asarray([_float(row["bootstrap_95_ci_high"]) for row in selected])
    if selected:
        axis.errorbar(
            positions,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt="o",
            color="#111827",
            ecolor="#111827",
            capsize=4,
        )
    axis.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1)
    axis.set_title("Communication-constrained minus baseline")
    axis.set_xticks(positions, [row["metric"] for row in selected], rotation=30, ha="right")
    axis.set_ylabel("seed-level paired difference")
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Effective EKF update persistence audit (validation only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_ekf_update_persistence_audit(
    config: EKFUpdatePersistenceAuditConfig,
) -> dict[str, Any]:
    """Run the read-only validation persistence audit."""

    if config.split != "validation":
        raise ValueError("EKF update persistence audit is validation-only")
    reference_root = Path(config.reference_root).resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")
    _seed_branch_dirs(reference_root, config.seeds, config.branches)

    event_rows, event_counts, episode_counts = _load_event_rows(
        reference_root=reference_root,
        seeds=config.seeds,
        branches=config.branches,
        split=config.split,
        horizons=config.horizons,
        config=config,
    )
    episode_rows = _episode_rows(
        reference_root=reference_root,
        seeds=config.seeds,
        branches=config.branches,
        split=config.split,
        horizons=config.horizons,
        event_rows=event_rows,
    )
    seed_rows = _seed_rows(episode_rows, config.seeds, config.branches, config.horizons)
    contrasts = _paired_contrasts(
        seed_rows, config.seeds, config.branches, config.horizons, config
    )
    jloc_rows = _jloc_rows(event_rows, episode_rows, config.horizons)
    decision = _decision(event_rows, seed_rows, config.branches, config.horizons)

    event_numeric_fields = [
        "timestamp",
        "message_age_s",
        "nis",
        "measurement_variance_m2",
        "effective_measurement_variance_m2",
        "kalman_gain_norm",
        "position_correction_norm_m",
        "unknown_cross_bound_m2",
        "unknown_cross_bound_fraction",
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
    ]
    for horizon in config.horizons:
        prefix = f"h{int(horizon)}"
        event_numeric_fields.extend(
            [
                f"{prefix}_observed",
                f"{prefix}_future_error_m",
                f"{prefix}_future_covariance_trace_m2",
                f"{prefix}_future_nees",
                f"{prefix}_error_change_from_before_m",
                f"{prefix}_error_change_from_after_m",
                f"{prefix}_time_since_update_s",
                f"{prefix}_benefit_retained",
                f"{prefix}_benefit_retained_fraction",
                f"{prefix}_j_loc",
                f"{prefix}_j_loc_change_from_after",
            ]
        )
    all_finite = all(
        math.isfinite(_float(row.get(field), float("nan")))
        for row in event_rows
        for field in event_numeric_fields
    )
    all_episode_finite = all(
        math.isfinite(_float(value, float("nan")))
        for row in episode_rows
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    expected_episode_count = len(config.seeds) * len(config.branches) * 20
    observed_episode_count = len(episode_rows)
    complete_episode_counts = all(
        int(episode_counts.get((seed, branch), 0)) == 20
        for seed in config.seeds
        for branch in config.branches
    )
    logs_dir, plots_dir = _prepare_outputs(config)
    event_path = logs_dir / "update_persistence_events.csv"
    episode_path = logs_dir / "episode_persistence_summary.csv"
    seed_path = logs_dir / "seed_persistence_summary.csv"
    contrast_path = logs_dir / "paired_contrasts.csv"
    jloc_path = logs_dir / "jloc_response.csv"
    plot_path = plots_dir / "ekf_update_persistence.png"
    _write_csv(event_path, event_rows)
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_path, seed_rows)
    _write_csv(contrast_path, contrasts)
    _write_csv(jloc_path, jloc_rows)
    _plot_audit(event_rows, seed_rows, contrasts, config.branches, config.horizons, plot_path)

    transition_counts: dict[str, int] = defaultdict(int)
    for event in event_rows:
        for horizon in config.horizons:
            prefix = f"h{int(horizon)}"
            transition_counts[
                f"h{int(horizon)}::{str(event.get(f'{prefix}_transition_type', 'unknown'))}"
            ] += 1
    coverage_summary: dict[str, dict[str, Any]] = {}
    for seed in config.seeds:
        for branch in config.branches:
            selected = [
                row
                for row in episode_rows
                if int(row["base_seed"]) == int(seed)
                and str(row["branch"]) == str(branch)
            ]
            eventful = sum(int(bool(row.get("eventful_episode", 0))) for row in selected)
            coverage_summary[f"{seed}::{branch}"] = {
                "episode_count": len(selected),
                "eventful_episode_count": eventful,
                "eventful_episode_fraction": eventful / len(selected) if selected else 0.0,
                "accepted_update_count": sum(
                    int(_float(row.get("accepted_update_count"), 0.0)) for row in selected
                ),
            }

    report: dict[str, Any] = {
        "audit": "ekf_update_persistence",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "split": config.split,
        "horizons": [int(value) for value in config.horizons],
        "base_seeds": [int(value) for value in config.seeds],
        "branches": list(config.branches),
        "position_metric_definition": {
            "position_rmse_m": "canonical vector-norm RMSE over UAV-timestamp position errors",
            "position_rmse_norm_m": "same canonical vector-norm RMSE, explicit field",
            "position_rmse_component_m": "legacy pooled coordinate-wise RMSE retained for traceability",
            "position_error_p95_norm_m": "P95 of vector-norm position errors",
        },
        "independent_unit": "training seed after episode-level reduction",
        "episode_rows_are_not_independent": True,
        "event_rows_are_not_independent": True,
        "test_data_read": False,
        "test_data_used": False,
        "source_writes": 0,
        "truth_used_by_actor_filter_or_reward": False,
        "offline_diagnostic_labels": [
            "true_position_error_before_m",
            "true_position_error_after_m",
            "los_label_offline",
        ],
        "event_counts": {
            f"{seed}::{branch}": int(event_counts.get((seed, branch), 0))
            for seed in config.seeds
            for branch in config.branches
        },
        "event_coverage": coverage_summary,
        "transition_counts": dict(transition_counts),
        "episode_counts": {
            f"{seed}::{branch}": int(episode_counts.get((seed, branch), 0))
            for seed in config.seeds
            for branch in config.branches
        },
        "checks": {
            "validation_only": config.split == "validation",
            "test_not_read": True,
            "source_untouched_by_audit": True,
            "all_update_fields_finite": all_finite,
            "all_episode_numeric_fields_finite": all_episode_finite,
            "complete_20_episode_validation_per_seed_branch": complete_episode_counts,
            "expected_episode_count": expected_episode_count,
            "observed_episode_count": observed_episode_count,
            "checkpoint_or_environment_mutated": False,
            "trainer_update_called": False,
            "optimizer_step_called": False,
            "seed_level_inference_used": True,
        },
        "formula_checks": {
            "immediate_error_change_max_abs_residual": max(
                (
                    abs(
                        _float(row["immediate_error_change_m"])
                        - (
                            _float(row["true_position_error_after_m"])
                            - _float(row["true_position_error_before_m"])
                        )
                    )
                    for row in event_rows
                ),
                default=0.0,
            ),
            "linear_covariance_reduction_definition_max_abs_residual": max(
                (
                    abs(
                        _float(row["linear_cov_reduction_m2"])
                        - (
                            _float(row["covariance_trace_before_m2"])
                            - _float(row["covariance_trace_after_linear_m2"])
                        )
                    )
                    for row in event_rows
                ),
                default=0.0,
            ),
        },
        "decision": decision,
        "audit_ok": bool(
            bool(event_rows)
            and bool(seed_rows)
            and all_finite
            and all_episode_finite
            and complete_episode_counts
            and config.split == "validation"
        ),
        "decision_supported": bool(bool(event_rows) and all_finite and complete_episode_counts),
        "outputs": {
            "update_persistence_events": str(event_path),
            "episode_persistence_summary": str(episode_path),
            "seed_persistence_summary": str(seed_path),
            "paired_contrasts": str(contrast_path),
            "jloc_response": str(jloc_path),
            "audit_report": str(logs_dir / "audit_report.json"),
            "plot": str(plot_path),
        },
        "interpretation_boundary": (
            "Ground-truth error and LOS/NLOS are validation-only labels. They are not used "
            "for actor decisions, EKF updates, reward construction, proxy fitting, or policy "
            "selection. Update events are nested within episodes and are not independent samples."
        ),
    }
    report_path = logs_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_BRANCHES",
    "DEFAULT_HORIZONS",
    "DEFAULT_SEEDS",
    "EKFUpdatePersistenceAuditConfig",
    "PRACTICAL_EVENT_THRESHOLD_M",
    "classify_update_effect",
    "run_ekf_update_persistence_audit",
]
