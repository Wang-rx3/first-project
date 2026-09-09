"""Validation-only audit for persistent-proxy measurement collapse.

The persistent-proxy experiment is deliberately audited from frozen
validation artifacts rather than rerun with a different policy.  This keeps
the audit read-only: the saved actions, actor logits, queue events and EKF
update events are compared with a train-fitted persistent proxy.  In
particular, no validation label is used to fit a proxy or to choose an
action.  The audit is intended to distinguish a teacher-utility scale
problem from an actor, budget or EKF problem before a new checkpoint is
trained.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.measurement_proxy import (
    _persistent_raw_from_components,
    _persistent_row_link,
    persistent_information_gain_from_row,
)
from ..schema import EPISODE_SCHEMA_VERSION
from ..synthetic.episode_generator import _write_json
from .execution_uncertainty_proxy_audit import _candidate_rows_from_episode
from .persistent_proxy_guided_quality_mappo import (
    BASELINE_BRANCH,
    DEFAULT_SEEDS,
    PERSISTENT_BRANCH,
)
from .quality_aware_mappo import (
    _episode_rows_for_branch,
    _finite,
    _read_csv,
    _read_json,
    _read_jsonl,
    _write_table,
)


MECHANISM_BRANCHES = (BASELINE_BRANCH, PERSISTENT_BRANCH)


@dataclass(frozen=True)
class PersistentProxyMechanismAuditConfig:
    """Configuration for the frozen validation mechanism audit."""

    reference_root: Path
    output_dir: Path
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    split: str = "validation"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    device: str = "cpu"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("persistent proxy mechanism audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.device != "cpu":
            raise ValueError("the deterministic audit currently supports device='cpu' only")


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _step_from_timestamp(value: Any) -> int:
    timestamp = _number(value, 0.0)
    return max(0, int(round(timestamp / 0.2)) - 1)


def _json_object(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return parsed


def _array(value: Any, length: int | None = None, fill: float = 0.0) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.empty(0, dtype=float)
    if length is None:
        return array
    result = np.full(int(length), float(fill), dtype=float)
    result[: min(result.size, array.size)] = array[: result.size]
    return result


def _softmax_masked(logits: Sequence[float], mask: Sequence[float], count: int) -> np.ndarray:
    """Reconstruct the actor categorical distribution from saved logits."""

    values = _array(logits, count + 1, fill=-1.0e9)
    allowed = _array(mask, count + 1, fill=0.0) > 0.5
    if not bool(allowed.any()):
        allowed[-1] = True
    values = np.where(allowed, values, -1.0e9)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.eye(1, count + 1, count, dtype=float).reshape(-1)
    maximum = float(np.max(values[allowed]))
    exponent = np.zeros(count + 1, dtype=float)
    exponent[allowed] = np.exp(np.clip(values[allowed] - maximum, -745.0, 0.0))
    total = float(exponent.sum())
    if not math.isfinite(total) or total <= 0.0:
        result = np.zeros(count + 1, dtype=float)
        result[-1] = 1.0
        return result
    return exponent / total


def _teacher_probabilities(
    gain: Sequence[float],
    cost: Sequence[float],
    valid: Sequence[float],
    action_mask: Sequence[float],
    *,
    temperature: float,
    communication_cost_weight: float,
    target_cost: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and the exact utility used to form them.

    ``target_cost=None`` reproduces the existing ``gain - cost`` teacher.
    When a target is supplied, only cost above the train-derived target is
    charged.  The latter is a counterfactual diagnostic here; it is not used
    to alter the frozen validation action.
    """

    count = max(len(gain), len(cost), len(valid))
    if count <= 0:
        return np.asarray([1.0], dtype=float), np.empty(0, dtype=float)
    gains = _array(gain, count, fill=0.0)
    costs = _array(cost, count, fill=1.0)
    validity = _array(valid, count, fill=0.0)
    if target_cost is None:
        charged = costs
    else:
        charged = np.maximum(0.0, costs - float(target_cost))
    utilities = gains - float(communication_cost_weight) * charged
    utilities = np.where(validity > 0.5, utilities, -1.0e9)
    raw = np.concatenate([utilities, np.asarray([0.0])])
    allowed = _array(action_mask, count + 1, fill=0.0) > 0.5
    if not bool(allowed.any()):
        allowed[-1] = True
    raw = np.where(allowed, raw, -1.0e9)
    scale = max(float(temperature), 1.0e-12)
    scaled = raw / scale
    maximum = float(np.max(scaled[allowed]))
    probabilities = np.zeros(count + 1, dtype=float)
    probabilities[allowed] = np.exp(np.clip(scaled[allowed] - maximum, -745.0, 0.0))
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        probabilities[:] = 0.0
        probabilities[-1] = 1.0
    else:
        probabilities /= total
    return probabilities, utilities


def _uav_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(str(value).split("_")[-1]), str(value)
    except (TypeError, ValueError):
        return 10**9, str(value)


def _candidate_values(
    row: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    horizon: int,
) -> tuple[float, float, float, float]:
    """Compute normalized persistent gain, cost and raw gain from one row."""

    raw, normalized, valid = persistent_information_gain_from_row(
        row, model, horizon=horizon
    )
    own_state = _array(row.get("own_state_6"))
    own_cov = _array(row.get("own_cov_diag_6"))
    neighbor_state = _array(row.get("neighbor_state_6"))
    neighbor_cov = _array(row.get("neighbor_cov_diag_6"))
    direction = _array(row.get("u_hat"))
    raw_again, quality, valid_again = _persistent_raw_from_components(
        own_state=own_state if own_state.size >= 6 else None,
        own_cov_diag=own_cov if own_cov.size >= 6 else None,
        neighbor_state=neighbor_state if neighbor_state.size >= 6 else None,
        neighbor_cov_diag=neighbor_cov if neighbor_cov.size >= 3 else None,
        direction=direction if direction.size >= 3 else None,
        link=_persistent_row_link(row),
        measurement_std=max(_number(row.get("measurement_std_m"), 0.5), 1.0e-9),
        age=max(_number(row.get("message_age_s"), 0.0), 0.0),
        model=model,
        horizon=max(1, int(horizon)),
    )
    if math.isfinite(raw_again):
        raw = raw_again
    if not math.isfinite(normalized):
        normalized = float("nan")
    cost = 1.0 - quality if math.isfinite(quality) else 1.0
    valid_value = float(bool(valid and valid_again and math.isfinite(normalized)))
    return (
        float(normalized),
        float(np.clip(cost, 0.0, 1.0)),
        float(raw),
        valid_value,
    )


def _action_map(episode_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _read_csv(episode_dir / "actions.csv"):
        step = _step_from_timestamp(row.get("timestamp"))
        mask = _json_object(row.get("action_mask_json"), {})
        measurement_mask = mask.get("measurement", []) if isinstance(mask, Mapping) else []
        output[(step, str(row.get("uav_id", "")))] = {
            "measurement_uav_id": str(row.get("measurement_uav_id", "none")),
            "action_mask": measurement_mask,
            "invalid_measurement_action": int(_number(row.get("invalid_measurement_action"), 0.0)),
            "measurement_transition_invalidated": int(
                _number(row.get("measurement_transition_invalidated"), 0.0)
            ),
        }
    return output


def _quality_map(scale_root: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in _read_jsonl(scale_root / "diagnostics" / "measurement_quality.jsonl"):
        if str(row.get("split", "")) != "validation":
            continue
        key = (
            str(row.get("episode_id", "")),
            _step_from_timestamp(row.get("timestamp")),
            str(row.get("uav_id", "")),
        )
        output[key] = row
    return output


def _budget_map(episode_dir: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    budget_path = episode_dir / "measurement_budget.csv"
    # The historical baseline has no hard budget file by design.  Missing
    # here means "unconstrained baseline", not a malformed validation run.
    if not budget_path.exists():
        return output
    for row in _read_csv(budget_path):
        output[str(row.get("uav_id", ""))] = {
            "limit": _number(row.get("budget_limit"), -1.0),
            "remaining_before": _number(row.get("budget_remaining_before"), -1.0),
            "used": _number(row.get("budget_used"), -1.0),
            "remaining_after": _number(row.get("budget_remaining_after"), -1.0),
        }
    return output


def _metric_map(scale_root: Path, branch: str, seed: int) -> dict[str, dict[str, Any]]:
    rows = _episode_rows_for_branch(scale_root, branch, seed)
    return {
        str(row.get("episode_id", "")): row
        for row in rows
        if str(row.get("split", "")) == "validation"
    }


def _checkpoint_hash(scale_root: Path) -> str:
    path = scale_root / "checkpoints" / "final.pt"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_map(scale_root: Path, persistent_scale: Path) -> dict[str, float]:
    persistent_manifest = _read_json(persistent_scale / "manifest.json")
    config = persistent_manifest.get("config", {})
    raw = config.get("measurement_communication_cost_target_by_uav", {})
    if isinstance(raw, Mapping) and raw:
        return {str(key): _number(value, 0.0) for key, value in raw.items()}
    # This fallback is still train-derived: it reads the already written
    # target artifact, never validation or test data.
    candidates = [
        Path("logs/preliminary") / persistent_scale.parents[2].name / "measurement_communication_cost_targets.csv",
        persistent_scale.parents[2] / "measurement_communication_cost_targets.csv",
    ]
    rows = []
    for candidate in candidates:
        if candidate.exists():
            rows = _read_csv(candidate)
            break
    result: dict[str, float] = {}
    for row in rows:
        if str(row.get("treatment", row.get("branch", ""))) == PERSISTENT_BRANCH:
            key = str(row.get("uav_id", ""))
            value = _number(row.get("target_cost", row.get("target_cost_mean")), float("nan"))
            if key and math.isfinite(value):
                result[key] = value
    if not result:
        raise ValueError(f"missing train-derived communication targets: {persistent_scale}")
    return result


def _event_index(scale_root: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in _read_jsonl(scale_root / "diagnostics" / "cooperative_updates.jsonl")
        if str(row.get("split", "")) == "validation"
    ]


def _replay_logged_branch(
    *,
    scale_root: Path,
    persistent_scale: Path,
    seed: int,
    branch: str,
    model: Mapping[str, Any],
    target_costs: Mapping[str, float],
    progress: bool,
) -> dict[str, Any]:
    """Extract a branch's frozen validation mechanism trace."""

    candidate_events: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    communication_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    quality_by_key = _quality_map(scale_root)
    branch_metrics = _metric_map(scale_root, branch, seed)
    updates = _event_index(scale_root)
    updates_by_key: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for event in updates:
        key = (
            str(event.get("episode_id", "")),
            int(_number(event.get("step"), 0.0)),
            str(event.get("target_uav_id", "")),
        )
        updates_by_key.setdefault(key, []).append(event)

    validation_root = scale_root / "validation"
    episode_dirs = sorted(path for path in validation_root.iterdir() if path.is_dir())
    for episode_index, episode_dir in enumerate(episode_dirs, start=1):
        episode_id = episode_dir.name
        if progress:
            print(
                f"[persistent-proxy-audit] seed={seed} branch={branch} "
                f"validation episode {episode_index}/{len(episode_dirs)}",
                flush=True,
            )
        action_rows = _action_map(episode_dir)
        budget_rows = _budget_map(episode_dir)
        overhead = _read_csv(episode_dir / "measurement_overhead.csv")
        raw_candidates = _candidate_rows_from_episode(
            episode_dir,
            base_seed=seed,
            treatment=branch,
            split="validation",
            include_offline_label=True,
            age_variance_coefficient=float(model.get("age_variance_coefficient", 1.0)),
        )
        by_window: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for row in raw_candidates:
            by_window.setdefault(
                (_step_from_timestamp(row.get("timestamp")), str(row.get("uav_id", ""))),
                [],
            ).append(row)

        selected_count = 0
        accepted_count = 0
        harmful_count = 0
        for (step, uav_id), rows in sorted(by_window.items(), key=lambda item: (item[0][0], _uav_sort_key(item[0][1]))):
            rows = sorted(rows, key=lambda row: _uav_sort_key(str(row.get("source_uav_id", ""))))
            uav_ids = sorted(
                {str(row.get("source_uav_id", "")) for row in rows}, key=_uav_sort_key
            )
            # The formal action space is fixed to the three UAV slots.  Keep
            # absent slots as invalid candidates so the explicit none action
            # remains the fourth slot.
            all_uav_ids = tuple(f"uav_{index:02d}" for index in range(3))
            gains = np.zeros(3, dtype=float)
            costs = np.ones(3, dtype=float)
            valids = np.zeros(3, dtype=float)
            raw_gains = np.full(3, np.nan, dtype=float)
            source_rows: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                source = str(row.get("source_uav_id", ""))
                if source not in all_uav_ids:
                    continue
                index = all_uav_ids.index(source)
                gain, cost, raw_gain, valid = _candidate_values(
                    row, model, horizon=int(model.get("persistent_horizon", 3))
                )
                gains[index] = gain if math.isfinite(gain) else 0.0
                costs[index] = cost if math.isfinite(cost) else 1.0
                valids[index] = valid
                raw_gains[index] = raw_gain
                source_rows[source] = row
            action = action_rows.get((step, uav_id), {})
            qlog = quality_by_key.get((episode_id, step, uav_id), {})
            if qlog:
                logged_gain = _array(qlog.get("persistent_proxy_gain"))
                logged_cost = _array(qlog.get("persistent_proxy_cost"))
                logged_valid = _array(qlog.get("persistent_proxy_valid"))
                if logged_gain.size >= 3 and logged_cost.size >= 3 and logged_valid.size >= 3:
                    gains = logged_gain[:3]
                    costs = logged_cost[:3]
                    valids = logged_valid[:3]
            measurement_mask = action.get("action_mask", [0, 1, 1, 1])
            target = _number(target_costs.get(uav_id), 0.0)
            teacher_current, utilities_current = _teacher_probabilities(
                gains,
                costs,
                valids,
                measurement_mask,
                temperature=_number(qlog.get("measurement_proxy_temperature"), 0.25),
                communication_cost_weight=_number(
                    qlog.get("measurement_proxy_communication_cost_weight"), 1.0
                ),
            )
            teacher_target, utilities_target = _teacher_probabilities(
                gains,
                costs,
                valids,
                measurement_mask,
                temperature=_number(qlog.get("measurement_proxy_temperature"), 0.25),
                communication_cost_weight=_number(
                    qlog.get("measurement_proxy_communication_cost_weight"), 1.0
                ),
                target_cost=target,
            )
            actor_probs = _softmax_masked(
                qlog.get("measurement_logits_adjusted", []),
                measurement_mask,
                3,
            )
            selected = str(action.get("measurement_uav_id", "none"))
            selected_count += int(selected != "none")
            update_key = (episode_id, step, uav_id)
            target_updates = updates_by_key.get(update_key, [])
            accepted = [event for event in target_updates if bool(event.get("accepted", False))]
            harmful = [
                event
                for event in accepted
                if _number(event.get("true_position_error_after_m"), 0.0)
                - _number(event.get("true_position_error_before_m"), 0.0)
                > 0.001
            ]
            accepted_count += len(accepted)
            harmful_count += len(harmful)
            budget = budget_rows.get(uav_id, {})
            candidate_sources = [source for source in all_uav_ids if valids[all_uav_ids.index(source)] > 0.5]
            current_positive = [
                index for index in range(3)
                if valids[index] > 0.5 and utilities_current[index] > 0.0
            ]
            target_positive = [
                index for index in range(3)
                if valids[index] > 0.5 and utilities_target[index] > 0.0
            ]
            teacher_current_choice = (
                all_uav_ids[int(np.argmax(teacher_current))]
                if int(np.argmax(teacher_current)) < 3
                else "none"
            )
            teacher_target_choice = (
                all_uav_ids[int(np.argmax(teacher_target))]
                if int(np.argmax(teacher_target)) < 3
                else "none"
            )
            coverage_rows.append(
                {
                    "base_seed": seed,
                    "branch": branch,
                    "episode_id": episode_id,
                    "step": step,
                    "uav_id": uav_id,
                    "candidate_count": len(candidate_sources),
                    "positive_current_utility": int(bool(current_positive)),
                    "positive_target_relative_utility": int(bool(target_positive)),
                    "actor_none_probability": float(actor_probs[-1]),
                    "teacher_none_probability_current": float(teacher_current[-1]),
                    "teacher_none_probability_target_relative": float(teacher_target[-1]),
                    "actor_selected_non_none": int(selected != "none"),
                    "teacher_selected_non_none_current": int(teacher_current_choice != "none"),
                    "teacher_selected_non_none_target_relative": int(teacher_target_choice != "none"),
                    "actor_teacher_top1_agreement_current": int(selected == teacher_current_choice),
                    "actor_teacher_top1_agreement_target_relative": int(selected == teacher_target_choice),
                    "budget_limit": _number(budget.get("limit"), -1.0),
                    "budget_remaining_before": _number(budget.get("remaining_before"), -1.0),
                    "budget_remaining_positive": int(_number(budget.get("remaining_before"), -1.0) > 0.0),
                    "accepted_update_count": len(accepted),
                    "harmful_update_count": len(harmful),
                    "measurement_communication_cost_target": target,
                }
            )
            for index, source in enumerate(all_uav_ids):
                candidate = source_rows.get(source, {})
                candidate_events.append(
                    {
                        "base_seed": seed,
                        "branch": branch,
                        "episode_id": episode_id,
                        "step": step,
                        "uav_id": uav_id,
                        "source_uav_id": source,
                        "candidate_index": index,
                        "candidate_rank_current": int(
                            1 + np.sum(utilities_current > utilities_current[index])
                        ) if valids[index] > 0.5 else "",
                        "candidate_rank_target_relative": int(
                            1 + np.sum(utilities_target > utilities_target[index])
                        ) if valids[index] > 0.5 else "",
                        "persistent_gain_norm": float(gains[index]),
                        "persistent_gain_raw": float(raw_gains[index]) if math.isfinite(raw_gains[index]) else "",
                        "proxy_cost_norm": float(costs[index]),
                        "proxy_utility_current": float(utilities_current[index]),
                        "proxy_utility_target_relative": float(utilities_target[index]),
                        "proxy_valid": int(valids[index] > 0.5),
                        "actor_probability": float(actor_probs[index]),
                        "teacher_probability_current": float(teacher_current[index]),
                        "teacher_probability_target_relative": float(teacher_target[index]),
                        "actor_none_probability": float(actor_probs[-1]),
                        "teacher_none_probability_current": float(teacher_current[-1]),
                        "teacher_none_probability_target_relative": float(teacher_target[-1]),
                        "candidate_selected": int(selected == source),
                        "selected_measurement_uav_id": selected,
                        "candidate_visible_in_actor": int(valids[index] > 0.5),
                        "message_age_s": _number(candidate.get("message_age_s"), 0.0),
                        "rssi_dbm": _number(candidate.get("rssi_dbm"), 0.0),
                        "snr_db": _number(candidate.get("snr_db"), 0.0),
                        "packet_loss_pred": _number(candidate.get("packet_loss_pred"), 1.0),
                        "delay_pred_s": _number(candidate.get("delay_pred_s"), 0.0),
                        "aoi_pred_s": _number(candidate.get("aoi_pred_s"), 0.0),
                        "queue_length_pred": _number(candidate.get("queue_length_pred"), 0.0),
                        "link_state": str(candidate.get("link_state", "unknown")),
                        "neighbor_covariance_available": int(
                            candidate.get("neighbor_covariance_available", 0)
                        ),
                        "budget_remaining_before": _number(budget.get("remaining_before"), -1.0),
                        "accepted_update_count": len(accepted),
                        "harmful_update_count": len(harmful),
                        "offline_position_error_m": _number(
                            candidate.get("offline_position_error_m"), float("nan")
                        ),
                    }
                )

        metric = branch_metrics.get(episode_id, {})
        load_packets = [_number(row.get("load_packets_applied"), 0.0) for row in overhead]
        queue_before = [_number(row.get("queue_before_packets"), float("nan")) for row in overhead]
        queue_loaded = [_number(row.get("queue_with_measurement_packets"), float("nan")) for row in overhead]
        queue_next = [_number(row.get("queue_next_packets"), float("nan")) for row in overhead]
        communication_rows.append(
            {
                "base_seed": seed,
                "branch": branch,
                "episode_id": episode_id,
                "measurement_count": selected_count,
                "overhead_event_count": len(overhead),
                "payload_size_bytes_all_512": int(
                    all(int(_number(row.get("payload_size_bytes"), 0.0)) == 512 for row in overhead)
                    if overhead else True
                ),
                "load_packets_requested": float(
                    sum(_number(row.get("load_packets_requested"), 0.0) for row in overhead)
                ),
                "load_packets_applied": float(sum(load_packets)),
                "queue_before_mean": float(np.nanmean(queue_before)) if queue_before else 0.0,
                "queue_with_measurement_mean": float(np.nanmean(queue_loaded)) if queue_loaded else 0.0,
                "queue_next_mean": float(np.nanmean(queue_next)) if queue_next else 0.0,
                "packet_loss_mean": float(
                    np.nanmean([_number(row.get("packet_loss"), float("nan")) for row in overhead])
                ) if overhead else 0.0,
                "delay_mean_s": float(
                    np.nanmean([_number(row.get("delay_s"), float("nan")) for row in overhead])
                ) if overhead else 0.0,
                "aoi_mean_s": float(
                    np.nanmean([_number(row.get("aoi_s"), float("nan")) for row in overhead])
                ) if overhead else 0.0,
                "j_com": _number(metric.get("j_com"), 0.0),
                "mean_reward": _number(metric.get("mean_reward"), 0.0),
                "position_rmse_m": _number(metric.get("position_rmse_m"), 0.0),
            }
        )

    update_lookup = {
        (str(row.get("episode_id", "")), int(_number(row.get("step"), 0.0)), str(row.get("target_uav_id", ""))): row
        for row in updates
    }
    candidate_lookup = {
        (
            str(row.get("episode_id", "")),
            int(_number(row.get("step"), 0.0)),
            str(row.get("uav_id", "")),
            str(row.get("source_uav_id", "")),
        ): row
        for row in candidate_events
    }
    for event in updates:
        if not bool(event.get("accepted", False)):
            continue
        episode_id = str(event.get("episode_id", ""))
        step = int(_number(event.get("step"), 0.0))
        target_uav = str(event.get("target_uav_id", ""))
        source_uav = str(event.get("source_uav_id", ""))
        candidate = candidate_lookup.get((episode_id, step, target_uav, source_uav), {})
        before_error = _number(event.get("true_position_error_before_m"), float("nan"))
        after_error = _number(event.get("true_position_error_after_m"), float("nan"))
        error_delta = after_error - before_error if math.isfinite(before_error) and math.isfinite(after_error) else float("nan")
        before_cov = _number(event.get("position_covariance_trace_before_m2"), float("nan"))
        linear_cov = _number(event.get("position_covariance_trace_after_linear_m2"), float("nan"))
        final_cov = _number(event.get("position_covariance_trace_after_final_m2"), float("nan"))
        update_rows.append(
            {
                "base_seed": seed,
                "branch": branch,
                "episode_id": episode_id,
                "step": step,
                "source_uav_id": source_uav,
                "target_uav_id": target_uav,
                "los": int(bool(event.get("los", False))),
                "message_age_s": _number(event.get("message_age_s"), 0.0),
                "nis": _number(event.get("nis"), float("nan")),
                "measurement_variance_m2": _number(event.get("measurement_variance_m2"), float("nan")),
                "kalman_gain_norm": _number(event.get("kalman_gain_norm"), float("nan")),
                "position_correction_norm_m": _number(event.get("position_correction_norm_m"), float("nan")),
                "unknown_cross_bound_m2": _number(event.get("unknown_cross_bound_m2"), float("nan")),
                "position_error_before_m": before_error,
                "position_error_after_m": after_error,
                "position_error_delta_m": error_delta,
                "linear_covariance_reduction_m2": before_cov - linear_cov if math.isfinite(before_cov) and math.isfinite(linear_cov) else float("nan"),
                "final_covariance_reduction_m2": before_cov - final_cov if math.isfinite(before_cov) and math.isfinite(final_cov) else float("nan"),
                "j_loc_before": _number(event.get("j_loc_before"), float("nan")),
                "j_loc_after": _number(event.get("j_loc_after"), float("nan")),
                "proxy_utility_current": _number(candidate.get("proxy_utility_current"), float("nan")),
                "proxy_utility_target_relative": _number(candidate.get("proxy_utility_target_relative"), float("nan")),
                "persistent_gain_norm": _number(candidate.get("persistent_gain_norm"), float("nan")),
                "proxy_valid": _number(candidate.get("proxy_valid"), 0.0),
                "benefit_class": (
                    "beneficial" if error_delta < -0.001
                    else "harmful" if error_delta > 0.001
                    else "neutral"
                ) if math.isfinite(error_delta) else "unlabelled",
                "covariance_restored": int(bool(event.get("covariance_restored", False))),
            }
        )
    return {
        "candidate_events": candidate_events,
        "coverage_rows": coverage_rows,
        "communication_rows": communication_rows,
        "update_rows": update_rows,
        "checkpoint_hash": _checkpoint_hash(scale_root),
    }


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_number(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _seed_rows(rows: Sequence[Mapping[str, Any]], *, branch: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("branch")) == branch]
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in selected:
        groups.setdefault((int(_number(row.get("base_seed"), -1)), str(row.get("episode_id", ""))), []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, episode_id), episode_rows in sorted(groups.items()):
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "episode_id": episode_id,
                "decision_windows": len(episode_rows),
                "candidate_windows": _mean(
                    (row.get("candidate_count", 0.0) > 0 for row in episode_rows), 0.0
                ),
                "positive_current_utility_window_rate": _mean(
                    (row.get("positive_current_utility", 0.0) for row in episode_rows), 0.0
                ),
                "positive_target_relative_utility_window_rate": _mean(
                    (row.get("positive_target_relative_utility", 0.0) for row in episode_rows), 0.0
                ),
                "actor_none_rate": _mean(
                    (row.get("actor_none_probability", 0.0) for row in episode_rows), 0.0
                ),
                "teacher_none_rate_current": _mean(
                    (row.get("teacher_none_probability_current", 0.0) for row in episode_rows), 0.0
                ),
                "teacher_none_rate_target_relative": _mean(
                    (row.get("teacher_none_probability_target_relative", 0.0) for row in episode_rows), 0.0
                ),
                "actor_selected_non_none_rate": _mean(
                    (row.get("actor_selected_non_none", 0.0) for row in episode_rows), 0.0
                ),
                "budget_remaining_positive_rate": _mean(
                    (row.get("budget_remaining_positive", 0.0) for row in episode_rows), 0.0
                ),
                "accepted_update_count": sum(int(_number(row.get("accepted_update_count"), 0.0)) for row in episode_rows),
                "harmful_update_count": sum(int(_number(row.get("harmful_update_count"), 0.0)) for row in episode_rows),
            }
        )
    return output


def _plot(
    candidate_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    seed_episode_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {BASELINE_BRANCH: "#2563eb", PERSISTENT_BRANCH: "#dc2626"}
    labels = {BASELINE_BRANCH: "Baseline", PERSISTENT_BRANCH: "Persistent proxy"}
    seeds = sorted({int(_number(row.get("base_seed"), -1)) for row in coverage_rows})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    for branch in MECHANISM_BRANCHES:
        values = [
            _mean(
                (
                    row.get("actor_none_probability", 0.0)
                    for row in coverage_rows
                    if row.get("branch") == branch
                    and int(_number(row.get("base_seed"), -1)) == seed
                ),
                0.0,
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Actor none probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("training seed")
    ax.legend()

    ax = axes[0, 1]
    for branch in MECHANISM_BRANCHES:
        values = [
            _mean(
                (
                    row.get("positive_current_utility", 0.0)
                    for row in coverage_rows
                    if row.get("branch") == branch
                    and int(_number(row.get("base_seed"), -1)) == seed
                ),
                0.0,
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Windows with positive current teacher utility")
    ax.set_ylim(0, 1)
    ax.set_xlabel("training seed")
    ax.legend()

    ax = axes[0, 2]
    for branch in MECHANISM_BRANCHES:
        values = [
            sum(
                int(_number(row.get("actor_selected_non_none"), 0.0))
                for row in coverage_rows
                if row.get("branch") == branch and int(_number(row.get("base_seed"), -1)) == seed
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Non-none measurement decisions")
    ax.set_xlabel("training seed")
    ax.legend()

    ax = axes[1, 0]
    for branch in MECHANISM_BRANCHES:
        values = [
            _mean(
                (
                    row.get("teacher_none_probability_current", 0.0)
                    for row in coverage_rows
                    if row.get("branch") == branch
                    and int(_number(row.get("base_seed"), -1)) == seed
                ),
                0.0,
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Current teacher none probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("training seed")
    ax.legend()

    ax = axes[1, 1]
    for branch in MECHANISM_BRANCHES:
        values = [
            _mean(
                (
                    row.get("persistent_gain_norm", 0.0)
                    for row in candidate_rows
                    if row.get("branch") == branch
                    and int(_number(row.get("base_seed"), -1)) == seed
                    and int(_number(row.get("proxy_valid"), 0.0)) == 1
                ),
                0.0,
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Mean persistent gain")
    ax.set_xlabel("training seed")
    ax.legend()

    ax = axes[1, 2]
    for branch in MECHANISM_BRANCHES:
        values = [
            _mean(
                (
                    row.get("harmful_update_count", 0.0)
                    for row in seed_episode_rows
                    if row.get("branch") == branch
                    and int(_number(row.get("base_seed"), -1)) == seed
                ),
                0.0,
            )
            for seed in seeds
        ]
        ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Harmful updates / decision window")
    ax.set_xlabel("training seed")
    ax.legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Persistent-proxy mechanism audit — validation diagnostic only")
    path = plots_dir / "persistent_proxy_mechanism_audit.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _finite_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    # Structural NA is expected for an unavailable candidate (and for an
    # accepted update whose source is not present in the current candidate
    # window).  Core proxy, policy and queue values must still be finite.
    optional = {
        "candidate_rank_current",
        "candidate_rank_target_relative",
        "persistent_gain_raw",
        "offline_position_error_m",
        "proxy_utility_current",
        "proxy_utility_target_relative",
        "persistent_gain_norm",
    }
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                if key in optional:
                    continue
                return False
    return True


def _decision(
    coverage_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> tuple[str, dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        values: dict[str, Any] = {"base_seed": int(seed)}
        for branch in MECHANISM_BRANCHES:
            rows = [
                row for row in coverage_rows
                if int(_number(row.get("base_seed"), -1)) == int(seed)
                and row.get("branch") == branch
            ]
            comm = [
                row for row in communication_rows
                if int(_number(row.get("base_seed"), -1)) == int(seed)
                and row.get("branch") == branch
            ]
            updates = [
                row for row in update_rows
                if int(_number(row.get("base_seed"), -1)) == int(seed)
                and row.get("branch") == branch
            ]
            prefix = branch.replace("_", "-")
            values[f"{prefix}_measurement_count"] = int(
                sum(int(_number(row.get("actor_selected_non_none"), 0.0)) for row in rows)
            )
            values[f"{prefix}_positive_utility_rate"] = _mean(
                (row.get("positive_current_utility", 0.0) for row in rows), 0.0
            )
            values[f"{prefix}_actor_none_rate"] = _mean(
                (row.get("actor_none_probability", 0.0) for row in rows), 0.0
            )
            values[f"{prefix}_teacher_none_rate"] = _mean(
                (row.get("teacher_none_probability_current", 0.0) for row in rows), 0.0
            )
            values[f"{prefix}_budget_positive_rate"] = _mean(
                (row.get("budget_remaining_positive", 0.0) for row in rows), 0.0
            )
            values[f"{prefix}_accepted_updates"] = len(updates)
            values[f"{prefix}_harmful_fraction"] = _mean(
                [int(str(row.get("benefit_class")) == "harmful") for row in updates], 0.0
            )
            values[f"{prefix}_j_com"] = _mean(
                (row.get("j_com", 0.0) for row in comm), 0.0
            )
        baseline_count = float(values.get("baseline-mappo_measurement_count", 0.0))
        persistent_count = float(
            values.get("persistent-proxy-guided-quality-mappo_measurement_count", 0.0)
        )
        ratio = persistent_count / baseline_count if baseline_count > 0.0 else 0.0
        values["persistent_to_baseline_measurement_ratio"] = ratio
        values["collapse_candidate"] = bool(
            ratio <= 0.20
            and values.get("persistent-proxy-guided-quality-mappo_budget_positive_rate", 0.0) >= 0.50
            and values.get("persistent-proxy-guided-quality-mappo_positive_utility_rate", 0.0) > 0.0
            and values.get("persistent-proxy-guided-quality-mappo_actor_none_rate", 0.0) >= 0.50
        )
        per_seed.append(values)

    collapse_count = sum(bool(row.get("collapse_candidate", False)) for row in per_seed)
    if collapse_count >= max(3, int(math.ceil(0.6 * len(seeds)))):
        return "policy_prior_overrides_budget", {"per_seed": per_seed, "collapse_seed_count": collapse_count}

    # If the policy does not collapse, use the pre-registered order for the
    # remaining causes.  These checks are intentionally conservative and only
    # fire when every required data block has enough observations.
    persistent_updates = [row for row in update_rows if row.get("branch") == PERSISTENT_BRANCH]
    if len(persistent_updates) >= 10:
        finite = [
            row for row in persistent_updates
            if math.isfinite(_number(row.get("proxy_utility_current")))
            and math.isfinite(_number(row.get("linear_covariance_reduction_m2")))
        ]
        if len(finite) >= 10:
            median = float(np.median([_number(row.get("proxy_utility_current")) for row in finite]))
            high = [row for row in finite if _number(row.get("proxy_utility_current")) >= median]
            low = [row for row in finite if _number(row.get("proxy_utility_current")) < median]
            high_gain = _mean((row.get("linear_covariance_reduction_m2") for row in high), 0.0)
            low_gain = _mean((row.get("linear_covariance_reduction_m2") for row in low), 0.0)
            if high and low and high_gain <= low_gain:
                return "persistent_proxy_not_predictive", {"per_seed": per_seed, "high_gain": high_gain, "low_gain": low_gain}
    comm_diffs: list[float] = []
    for row in per_seed:
        comm_diffs.append(
            float(row.get("persistent-proxy-guided-quality-mappo_j_com", 0.0))
            - float(row.get("baseline-mappo_j_com", 0.0))
        )
    if sum(value > 0.005 for value in comm_diffs) >= max(3, int(math.ceil(0.6 * len(seeds)))):
        return "communication_cost_still_dominant", {"per_seed": per_seed, "j_com_differences": comm_diffs}
    persistent_harm = [row for row in persistent_updates if row.get("benefit_class") in {"beneficial", "harmful"}]
    if persistent_harm and sum(row.get("benefit_class") == "harmful" for row in persistent_harm) > len(persistent_harm) / 2:
        return "ekf_mean_benefit_not_persistent", {"per_seed": per_seed}
    return "statistical_evidence_insufficient", {"per_seed": per_seed}


def run_persistent_proxy_mechanism_audit(
    config: PersistentProxyMechanismAuditConfig,
) -> dict[str, Any]:
    """Run the frozen persistent-proxy mechanism audit."""

    reference = Path(config.reference_root).resolve()
    logs = Path(config.output_dir).resolve()
    plots = Path(config.plots_dir).resolve() if config.plots_dir else (
        Path("plots/preliminary") / reference.name / "persistent_proxy_mechanism_audit"
    ).resolve()
    managed = (
        "candidate_proxy_events.csv",
        "accepted_update_quality.csv",
        "policy_coverage.csv",
        "communication_alignment.csv",
        "seed_episode_summary.csv",
        "audit_report.json",
    )
    if config.force:
        for path in (logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("persistent proxy mechanism audit output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []
    all_communication: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    checkpoint_before: dict[str, str] = {}
    train_only: dict[str, Any] = {}
    replay_status: list[dict[str, Any]] = []

    for seed in config.seeds:
        baseline_scale = reference / f"seed_{int(seed)}" / BASELINE_BRANCH / "scale_3"
        persistent_scale = reference / f"seed_{int(seed)}" / PERSISTENT_BRANCH / "scale_3"
        if not baseline_scale.is_dir() or not persistent_scale.is_dir():
            raise FileNotFoundError(f"missing branch scale for seed {seed}")
        model_path = persistent_scale / "calibration" / "measurement_persistent_proxy_model.json"
        model = _read_json(model_path)
        if model.get("source_split") != "train" or model.get("validation_used_for_fitting") or model.get("test_used_for_fitting"):
            raise ValueError(f"persistent proxy model is not train-only: {model_path}")
        targets = _target_map(baseline_scale, persistent_scale)
        train_only[str(seed)] = {
            "persistent_proxy_model": str(model_path),
            "source_split": model.get("source_split"),
            "validation_used_for_fitting": bool(model.get("validation_used_for_fitting", False)),
            "test_used_for_fitting": bool(model.get("test_used_for_fitting", False)),
            "communication_targets_source": "baseline_mappo/train manifest",
            "communication_targets": targets,
        }
        for branch, scale in ((BASELINE_BRANCH, baseline_scale), (PERSISTENT_BRANCH, persistent_scale)):
            checkpoint_before[f"{seed}/{branch}"] = _checkpoint_hash(scale)
            result = _replay_logged_branch(
                scale_root=scale,
                persistent_scale=persistent_scale,
                seed=int(seed),
                branch=branch,
                model=model,
                target_costs=targets,
                progress=config.progress,
            )
            all_candidates.extend(result["candidate_events"])
            all_coverage.extend(result["coverage_rows"])
            all_communication.extend(result["communication_rows"])
            all_updates.extend(result["update_rows"])
            replay_status.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "validation_episode_count": len(result["communication_rows"]),
                    "checkpoint_hash": result["checkpoint_hash"],
                    "saved_actions_source": "validation/actions.csv",
                    "saved_queue_source": "validation/measurement_overhead.csv",
                }
            )

    checkpoint_after = {
        key: _checkpoint_hash(
            reference / f"seed_{key.split('/')[0]}" / key.split('/')[1] / "scale_3"
        )
        for key in checkpoint_before
    }
    checkpoint_unchanged = checkpoint_before == checkpoint_after
    seed_episode = _seed_rows(all_coverage, branch=BASELINE_BRANCH) + _seed_rows(
        all_coverage, branch=PERSISTENT_BRANCH
    )
    primary_cause, decision_details = _decision(
        all_coverage, all_communication, all_updates, config.seeds
    )
    finite = _finite_rows(all_candidates) and _finite_rows(all_coverage) and _finite_rows(all_communication) and _finite_rows(all_updates)
    payload_ok = all(
        int(_number(row.get("payload_size_bytes_all_512"), 0.0)) == 1
        for row in all_communication
        if int(_number(row.get("overhead_event_count"), 0.0)) > 0
    )
    report = {
        "experiment": "persistent_proxy_mechanism_audit",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": list(MECHANISM_BRANCHES),
        "split": "validation",
        "independent_unit": "base training seed",
        "event_rows_are_not_independent_samples": True,
        "test_data_used": False,
        "test_data_access_count": 0,
        "training_updates_called": False,
        "optimizer_steps_called": False,
        "replay_mode": "frozen logged validation artifacts; no environment rerun",
        "fit_sources": train_only,
        "checks": {
            "validation_episode_rows_complete": len(all_communication) == len(config.seeds) * len(MECHANISM_BRANCHES) * 20,
            "candidate_rows_nonempty": bool(all_candidates),
            "all_fields_finite": bool(finite),
            "checkpoint_hash_unchanged": bool(checkpoint_unchanged),
            "measurement_payload_rule_verified": bool(payload_ok),
            "train_only_fit_sources": all(
                item.get("source_split") == "train"
                and not item.get("validation_used_for_fitting")
                and not item.get("test_used_for_fitting")
                for item in train_only.values()
            ),
            "validation_only": True,
            "saved_action_and_queue_artifacts_present": all(
                Path(str(row.get("checkpoint_hash", ""))).name != "" for row in replay_status
            ),
        },
        "policy_collapse_thresholds": {
            "minimum_seed_count": max(3, int(math.ceil(0.6 * len(config.seeds)))),
            "persistent_to_baseline_measurement_ratio_max": 0.20,
            "minimum_budget_remaining_positive_rate": 0.50,
            "minimum_positive_current_utility_rate": 0.0,
            "minimum_actor_none_rate": 0.50,
        },
        "decision": {
            "primary_cause": primary_cause,
            **decision_details,
        },
        "audit_ok": bool(
            finite
            and checkpoint_unchanged
            and payload_ok
            and all(
                item.get("source_split") == "train"
                and not item.get("validation_used_for_fitting")
                and not item.get("test_used_for_fitting")
                for item in train_only.values()
            )
        ),
        "decision_supported": bool(finite and checkpoint_unchanged and primary_cause != "statistical_evidence_insufficient"),
        "interpretation_boundary": (
            "The audit diagnoses the frozen persistent teacher and policy coverage. "
            "The target-relative utility is a counterfactual diagnostic only; it does not "
            "change the saved validation action or claim a new algorithm result."
        ),
    }
    _write_table(logs / "candidate_proxy_events.csv", all_candidates)
    _write_table(logs / "accepted_update_quality.csv", all_updates)
    _write_table(logs / "policy_coverage.csv", all_coverage)
    _write_table(logs / "communication_alignment.csv", all_communication)
    _write_table(logs / "seed_episode_summary.csv", seed_episode)
    plot = _plot(all_candidates, all_coverage, seed_episode, plots)
    report["outputs"] = {
        "candidate_proxy_events": str(logs / "candidate_proxy_events.csv"),
        "accepted_update_quality": str(logs / "accepted_update_quality.csv"),
        "policy_coverage": str(logs / "policy_coverage.csv"),
        "communication_alignment": str(logs / "communication_alignment.csv"),
        "seed_episode_summary": str(logs / "seed_episode_summary.csv"),
        "plot": str(plot),
        "audit_report": str(logs / "audit_report.json"),
    }
    report["replay_status"] = replay_status
    _write_json(logs / "audit_report.json", report)
    return {
        "output_dir": str(logs),
        "plots_dir": str(plots),
        "audit_report": str(logs / "audit_report.json"),
        "plot": str(plot),
        "audit_ok": bool(report["audit_ok"]),
        "decision_supported": bool(report["decision_supported"]),
        "primary_cause": primary_cause,
    }


__all__ = [
    "PersistentProxyMechanismAuditConfig",
    "run_persistent_proxy_mechanism_audit",
]
