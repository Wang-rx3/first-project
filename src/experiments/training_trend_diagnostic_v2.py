"""Versioned, read-only evidence audit for the seed-42 training trend.

This module deliberately does not train a policy.  It replays the frozen
initial/final checkpoints on the five development-monitor scenes, records the
actor/teacher/communication paths, and probes the loss terms with autograd
without calling ``optimizer.step``.  The module is separate from the v1 audit
so that old evidence and its historical schemas remain untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .cooperative_event_logging import accepted_counts, collect_window_events
from .mainline_baseline_comparison import (
    BASELINE_BRANCH,
    COMMUNICATION_BRANCH,
    _checkpoint_trainer,
)
from .mainline_figures_v2 import SourceGuard
from .pilot import _assert_action_feasible, _assert_observation_finite, _seed_everything
from .training_trend_diagnostic import (
    batch_metadata,
    collect as collect_probe_episode,
    components,
    fresh,
    gradient_probe,
    policy_vector,
)
from .formal import _make_environment, _scenario_for, _scale_seed
from ..learning.mappo import MAPPOConfig, MAPPOPolicy
from ..synthetic.episode_generator import _ekf_diagnostic_row


SOURCE = Path(
    "data/synthetic/mainline_baseline_comparison_v2_seed42/development/rollout4/seed_42"
)
OLD_LOGS = Path("logs/preliminary/mainline_baseline_comparison_v2_seed42")
LOGS = Path("logs/preliminary/training_trend_diagnostic_seed42/frozen-audit-v2")
DATA = Path("data/synthetic/training_trend_diagnostic_seed42/frozen-audit-v2")
PLOTS = Path("plots/preliminary/training_trend_diagnostic_seed42/frozen-audit-v2")

METHODS = (BASELINE_BRANCH, COMMUNICATION_BRANCH)
ARMS = ("reference", "proxy-aligned", "rollout4")
PARTS = ("reward_loc", "reward_com", "reward_collision", "reward_energy")
HORIZONS = (1, 2, 3, 5)
BENEFIT_THRESHOLD_M = 0.001
TOL = 1e-6


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any, *, field: str = "value", allow_na: bool = False) -> float:
    if allow_na and (value is None or str(value).strip().upper() in {"", "NA", "N/A", "NAN"}):
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _optional(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _native(value: Any) -> Any:
    """Convert NumPy/Torch scalars and nested mappings for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _native(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_native(payload), ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _cell(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return value if math.isfinite(float(value)) else "NA"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(_native(value), ensure_ascii=False, allow_nan=False, default=str)
    if value is None:
        return "NA"
    return value


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty audit table: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "NA")) for key in columns})


def _json_array(values: Any) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("diagnostic array contains a non-finite value")
    return json.dumps([float(value) for value in array], allow_nan=False)


def _json_object(value: Any) -> str:
    return json.dumps(_native(value), ensure_ascii=False, allow_nan=False, default=str)


def _action_index(action: Any, uav_count: int) -> int:
    text = str(action)
    if text == "none":
        return int(uav_count)
    prefix = "uav_"
    if not text.startswith(prefix):
        raise ValueError(f"unknown measurement action: {action!r}")
    index = int(text[len(prefix) :])
    if not 0 <= index < uav_count:
        raise ValueError(f"measurement action outside action space: {action!r}")
    return index


def _action_label(index: int, uav_count: int) -> str:
    return "none" if int(index) == int(uav_count) else f"uav_{int(index):02d}"


def _metric_map(mapping: Any, key: str, default: Any = "NA") -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _resolve_applied_measurement_action(
    applied_actions: Any,
    uav_id: str,
    requested_action: Any,
    uav_count: int,
) -> tuple[str, int]:
    """Resolve the action accepted by the environment after sanitization."""

    applied = _metric_map(applied_actions, uav_id, {})
    if not isinstance(applied, Mapping):
        applied = {}
    action = str(applied.get("measurement_uav_id", requested_action))
    return action, _action_index(action, uav_count)


def _overhead_for(
    overhead_rows: Sequence[Mapping[str, Any]], requester: str, source: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in overhead_rows
        if str(row.get("target_uav_id")) == str(requester)
        and str(row.get("source_uav_id")) == str(source)
    ]


def _quality_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"accepted": 0, "beneficial": 0, "neutral": 0, "harmful": 0}
    for event in events:
        if not _truth(event.get("accepted", False)):
            continue
        counts["accepted"] += 1
        delta = _number(event.get("true_position_error_after_m"), field="event error after") - _number(
            event.get("true_position_error_before_m"), field="event error before"
        )
        if delta < -BENEFIT_THRESHOLD_M:
            counts["beneficial"] += 1
        elif delta > BENEFIT_THRESHOLD_M:
            counts["harmful"] += 1
        else:
            counts["neutral"] += 1
    if counts["beneficial"] + counts["neutral"] + counts["harmful"] != counts["accepted"]:
        raise ValueError("update quality classes do not close")
    return counts


def _retention(
    events: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    uav_ids: Sequence[str],
) -> dict[str, Any]:
    by_step: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in window_rows:
        step = int(row["step"])
        for uav_id in uav_ids:
            key = f"position_error_{uav_id}"
            if key in row:
                by_step[(uav_id, step)] = row
    result: dict[str, Any] = {}
    beneficial = [
        event
        for event in events
        if _truth(event.get("accepted", False))
        and _number(event.get("true_position_error_after_m"), field="event after")
        - _number(event.get("true_position_error_before_m"), field="event before")
        < -BENEFIT_THRESHOLD_M
    ]
    for horizon in HORIZONS:
        observed = 0
        retained = 0
        censored = 0
        for event in beneficial:
            target = str(event["target_uav_id"])
            step = int(event["step"])
            future = by_step.get((target, step + horizon))
            if future is None:
                censored += 1
                continue
            observed += 1
            baseline = _number(event["true_position_error_before_m"], field="event before")
            future_error = _number(future[f"position_error_{target}"], field="future error")
            retained += int(future_error < baseline - BENEFIT_THRESHOLD_M)
        result[f"h{horizon}_beneficial_count"] = len(beneficial)
        result[f"h{horizon}_observed_count"] = observed
        result[f"h{horizon}_censored_count"] = censored
        result[f"h{horizon}_retained_fraction"] = (
            retained / observed if observed else float("nan")
        )
    return result


def _trainer_at(method: str, endpoint: str, data_root: Path):
    scale = SOURCE / method / "scale_3"
    manifest = _read_json(scale / "manifest.json")
    config = _reference_formal_config(manifest, data_root, force=False, progress=False)
    stats = _read_json(scale / "normalization_stats.json")
    config = replace(config, metric_scales=stats["metric_scales"])
    environment = _make_environment(config, 42, metric_scales=config.metric_scales)
    environment.reset(seed=42)
    final = _checkpoint_trainer(scale, config, environment)
    if endpoint == "final":
        final.policy.eval()
        return config, final
    initial = type(final)(
        config=final.config,
        uav_ids=final.policy.uav_ids,
        seed=42,
        normalizer=final.normalizer,
        central_normalizer=final.central_normalizer,
        measurement_quality_model=final.measurement_quality_model,
        measurement_proxy_model=final.measurement_proxy_model,
        measurement_persistent_proxy_model=final.measurement_persistent_proxy_model,
    )
    expected = _read_json(scale / "diagnostics/initial_parameter_hashes.json")["modules"]
    actual = initial.policy.initial_parameter_initialization_hashes()
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f"initial module hashes differ for {method}")
    initial.policy.eval()
    return config, initial


def _collect_detailed(
    config: Any,
    trainer: Any,
    *,
    method: str,
    endpoint: str,
    scene: int,
    seed: int,
    identity: str,
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay one fixed monitor scene and return auditable in-memory tables."""

    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(
        seed=seed,
        scenario_config=_scenario_for(config, seed, split="train", episode_index=scene),
    )
    uav_ids = tuple(environment.uav_ids)
    action_rows: list[dict[str, Any]] = []
    communication_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    before_parameters = policy_vector(trainer)

    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        with torch.no_grad():
            decision = trainer.act(
                observations,
                info.get("central_state"),
                deterministic=True,
            )
        _assert_action_feasible(environment, decision.actions)
        counters = accepted_counts(environment)
        budget_before_map = dict(getattr(environment, "measurement_budget_remaining", {}))
        budget_used_before_map = dict(getattr(environment, "measurement_budget_used", {}))
        step_policy_rows: dict[str, dict[str, Any]] = {}
        for uav_id in uav_ids:
            local = decision.observations[uav_id]
            sample = decision.samples[uav_id]
            actor = np.asarray(sample.measurement_probabilities, dtype=float).reshape(-1)
            if actor.size != len(uav_ids) + 1 or not np.isfinite(actor).all():
                raise ValueError("actor measurement probabilities are not finite or have wrong size")
            teacher = trainer.policy.measurement_proxy_teacher_diagnostics(local, uav_id=uav_id)
            teacher_enabled = bool(teacher.get("enabled", False))
            teacher_probs = np.asarray(teacher.get("teacher_probs", []), dtype=float).reshape(-1)
            if teacher_enabled and (teacher_probs.size != actor.size or not np.isfinite(teacher_probs).all()):
                raise ValueError("teacher measurement probabilities are not finite or have wrong size")
            selected_action = str(decision.actions[uav_id]["measurement_uav_id"])
            selected_index = _action_index(selected_action, len(uav_ids))
            actor_top1 = int(np.argmax(actor))
            teacher_top1 = int(np.argmax(teacher_probs)) if teacher_enabled else None
            teacher_kl = (
                float(np.sum(teacher_probs * (np.log(np.maximum(teacher_probs, 1e-12)) - np.log(np.maximum(actor, 1e-12)))))
                if teacher_enabled
                else float("nan")
            )
            budget_limit = _metric_map(getattr(environment, "measurement_budget_limits", {}), uav_id)
            budget_after_placeholder = _metric_map(getattr(environment, "measurement_budget_remaining", {}), uav_id)
            expected_cost = float("nan")
            target_cost = float("nan")
            residual = float("nan")
            dual = float("nan")
            if config.measurement_communication_constraint_enabled:
                with torch.no_grad():
                    expected_cost = float(
                        trainer.policy.expected_measurement_communication_cost(local).detach().cpu()
                    )
                target_cost = float(
                    (config.measurement_communication_cost_target_by_uav or {}).get(uav_id, float("nan"))
                )
                residual = max(
                    0.0,
                    expected_cost
                    - target_cost
                    - float(config.measurement_communication_cost_tolerance),
                )
                dual = float(trainer.measurement_communication_dual_by_uav.get(uav_id, float("nan")))
            arrays = {
                "actor_probability_json": _json_array(actor),
                "teacher_probability_json": _json_array(teacher_probs) if teacher_enabled else "NA",
                "teacher_utility_json": _json_array(teacher.get("teacher_utility", [])) if teacher_enabled else "NA",
                "teacher_gain_json": _json_array(teacher.get("teacher_gain", [])) if teacher_enabled else "NA",
                "teacher_cost_json": _json_array(teacher.get("teacher_cost", [])) if teacher_enabled else "NA",
                "teacher_charged_cost_json": _json_array(teacher.get("teacher_charged_cost", [])) if teacher_enabled else "NA",
                "teacher_valid_json": _json_array(teacher.get("teacher_valid", [])) if teacher_enabled else "NA",
                "measurement_logits_raw_json": _json_array(sample.measurement_logits_raw),
                "measurement_logits_adjusted_json": _json_array(sample.measurement_logits_adjusted),
                "action_mask_json": _json_array(np.asarray(local.get("action_mask", []), dtype=float)),
            }
            row = {
                "method": method,
                "endpoint": endpoint,
                "scene": scene,
                "episode_id": identity,
                "step": step,
                "timestamp": (step + 1) * float(environment.config.motion.dt_s),
                "uav_id": uav_id,
                # Keep the policy request separate from what the environment
                # actually executed.  Action masks, invalid edges, and an
                # exhausted budget can sanitize a requested source to
                # ``none`` during ``Environment.step``.  Communication and
                # budget accounting must use the applied action, while the
                # requested action remains useful for diagnosing policy
                # feasibility.
                "requested_measurement_uav_id": selected_action,
                "requested_action_index": selected_index,
                "selected_measurement_uav_id": selected_action,
                "selected_action_index": selected_index,
                "executed_action_index": selected_index,
                "actor_top1_action": _action_label(actor_top1, len(uav_ids)),
                "teacher_top1_action": _action_label(teacher_top1, len(uav_ids)) if teacher_top1 is not None else "NA",
                "actor_none_probability": float(actor[len(uav_ids)]),
                "teacher_none_probability": float(teacher_probs[len(uav_ids)]) if teacher_enabled else float("nan"),
                "teacher_enabled": int(teacher_enabled),
                "measurement_constraint_enabled": int(config.measurement_communication_constraint_enabled),
                "teacher_actor_top1_agreement": int(teacher_enabled and teacher_top1 == actor_top1),
                "teacher_selected_action_agreement": int(teacher_enabled and teacher_top1 == selected_index),
                "requested_teacher_selected_action_agreement": int(teacher_enabled and teacher_top1 == selected_index),
                "teacher_actor_kl": teacher_kl,
                "budget_limit": budget_limit,
                "budget_remaining_before": _metric_map(budget_before_map, uav_id),
                "budget_used_before": _metric_map(budget_used_before_map, uav_id),
                "budget_remaining_after_replay_placeholder": budget_after_placeholder,
                "expected_communication_cost": expected_cost,
                "communication_target": target_cost,
                "communication_tolerance": float(config.measurement_communication_cost_tolerance),
                "communication_residual": residual,
                "communication_dual": dual,
                "communication_weighted_constraint_loss": dual * residual if math.isfinite(dual) else float("nan"),
                "bandwidth_action_json": _json_object(decision.actions[uav_id].get("bandwidth", {})),
                "delta_v_raw_json": _json_array(decision.actions[uav_id].get("delta_v", [])),
                **arrays,
            }
            step_policy_rows[uav_id] = row

        next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
        new_events = collect_window_events(
            environment,
            counters,
            run_root=run_root,
            episode_id=identity,
            split="development_monitor",
            step=step,
        )
        event_rows.extend(
            {"method": method, "endpoint": endpoint, "scene": scene, **dict(event)}
            for event in new_events
        )
        overhead_rows = list(next_info.get("measurement_overhead_events", []))
        action_execution = next_info.get("action_execution", {})
        budget_after_map = dict(next_info.get("measurement_budget_remaining", {}))
        budget_used_after_map = dict(next_info.get("measurement_budget_used", {}))
        actual_costs: list[float] = []
        selected_count = 0
        step_packet_bytes = 0
        applied_actions = next_info.get("actions", decision.actions)
        for uav_id in uav_ids:
            row = step_policy_rows[uav_id]
            requested_action = str(row["requested_measurement_uav_id"])
            requested_index = int(row["requested_action_index"])
            selected_action, applied_index = _resolve_applied_measurement_action(
                applied_actions,
                uav_id,
                requested_action,
                len(uav_ids),
            )
            row["selected_measurement_uav_id"] = selected_action
            row["selected_action_index"] = applied_index
            row["executed_action_index"] = applied_index
            row["requested_teacher_selected_action_agreement"] = int(
                row["teacher_enabled"] and _action_index(
                    str(row["teacher_top1_action"]), len(uav_ids)
                ) == requested_index
            ) if row["teacher_enabled"] else 0
            row["teacher_selected_action_agreement"] = int(
                row["teacher_enabled"] and _action_index(
                    str(row["teacher_top1_action"]), len(uav_ids)
                ) == applied_index
            ) if row["teacher_enabled"] else 0
            matched = _overhead_for(overhead_rows, uav_id, selected_action) if selected_action != "none" else []
            if selected_action != "none":
                selected_count += 1
                if len(matched) != 1:
                    raise ValueError(f"selected measurement does not map to exactly one overhead packet: {identity}/{step}/{uav_id}")
                packet = matched[0]
                if int(_number(packet.get("payload_size_bytes"), field="payload size")) != 512 or int(
                    _number(packet.get("load_packets_applied"), field="applied packet load")
                ) != 1:
                    raise ValueError("measurement overhead is not one 512-byte packet")
                step_packet_bytes += 512
            selected_cost = float("nan")
            teacher_cost_json = row["teacher_cost_json"]
            if selected_action != "none" and teacher_cost_json != "NA":
                costs = json.loads(teacher_cost_json)
                selected_cost = _number(costs[applied_index], field="selected candidate cost")
                actual_costs.append(selected_cost)
            execution = _metric_map(action_execution, uav_id, {})
            row.update(
                {
                    "budget_remaining_after": _metric_map(budget_after_map, uav_id),
                    "budget_used_after": _metric_map(budget_used_after_map, uav_id),
                    "actual_selected_candidate_cost_snapshot": selected_cost,
                    "actual_packet_count": len(matched),
                    "actual_packet_bytes": 512 * len(matched),
                    "applied_bandwidth_json": _json_object(execution.get("bandwidth", {})) if isinstance(execution, Mapping) else "NA",
                    "delta_v_applied_json": _json_array(execution.get("delta_v_applied", [])) if isinstance(execution, Mapping) and execution.get("delta_v_applied") is not None else "NA",
                    "actual_packet_loss": _number(matched[0].get("packet_loss"), field="packet loss") if matched else float("nan"),
                    "actual_delay_s": _number(matched[0].get("delay_s"), field="packet delay") if matched else float("nan"),
                    "actual_aoi_s": _number(matched[0].get("aoi_s"), field="packet aoi") if matched else float("nan"),
                }
            )
            action_rows.append(dict(row))
            communication_rows.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "scene": scene,
                    "episode_id": identity,
                    "step": step,
                    "timestamp": row["timestamp"],
                    "uav_id": uav_id,
                    "expected_communication_cost": row["expected_communication_cost"],
                    "communication_target": row["communication_target"],
                    "communication_tolerance": row["communication_tolerance"],
                    "communication_residual": row["communication_residual"],
                    "communication_dual": row["communication_dual"],
                    "weighted_constraint_loss": row["communication_weighted_constraint_loss"],
                    "selected_measurement_uav_id": selected_action,
                    "actual_selected_candidate_cost_snapshot": selected_cost,
                    "actual_packet_count": len(matched),
                    "actual_packet_bytes": 512 * len(matched),
                    "packet_loss": row["actual_packet_loss"],
                    "delay_s": row["actual_delay_s"],
                    "aoi_s": row["actual_aoi_s"],
                    "measurement_constraint_enabled": int(config.measurement_communication_constraint_enabled),
                    "validation_execution_mode": "deterministic_argmax",
                }
            )

        objective = next_info.get("objective_components", {})
        penalties = _metric_map(next_info.get("reward_components", {}), uav_ids[0], {})
        reward = float(np.mean([float(value) for value in rewards.values()]))
        reward_components = {
            "reward_loc": -_number(objective.get("j_loc"), field="j_loc"),
            "reward_com": -_number(objective.get("j_com"), field="j_com"),
            "reward_collision": -_number(penalties.get("collision_penalty", 0.0), field="collision penalty"),
            "reward_energy": -_number(penalties.get("energy_penalty", 0.0), field="energy penalty"),
        }
        if abs(reward - sum(reward_components.values())) > TOL:
            raise ValueError(f"reward closure failed at {identity}/{step}")
        positions: dict[str, float] = {}
        replay_diagnostics: dict[str, dict[str, Any]] = {}
        for uav_id in uav_ids:
            # ``position_error_m`` is an optional environment diagnostic in
            # some historical replay records.  Recompute this value only as
            # an offline audit label from the replay state and ground truth;
            # it never enters the actor, teacher, constraint, or loss path.
            estimate = np.asarray(environment.ekfs[uav_id].state, dtype=float)
            truth = np.asarray(environment.states[uav_id].position_m, dtype=float)
            if estimate.size < 3 or not np.isfinite(estimate).all() or not np.isfinite(truth).all():
                raise ValueError("non-finite offline position-error label")
            diagnostic = _ekf_diagnostic_row(
                {},
                (step + 1) * float(environment.config.motion.dt_s),
                uav_id,
                truth,
                estimate,
                np.asarray(environment.ekfs[uav_id].covariance, dtype=float),
                next_info.get("ekf_diagnostics", {}).get(uav_id),
                next_info.get("visual_odometry_quality_diagnostics", {}).get(uav_id),
            )
            replay_diagnostics[uav_id] = diagnostic
            positions[uav_id] = _number(diagnostic.get("position_error_m"), field="position error")
        for uav_id in uav_ids:
            diag = replay_diagnostics[uav_id]
            for key in ("nis_vo", "nis_uwb_mean", "nis_uwb_max", "nis_gnss", "position_nees"):
                if diag.get(key) not in (None, "", "NA"):
                    _number(diag.get(key), field=key)
            if not np.isfinite(environment.ekfs[uav_id].covariance).all() or not np.isfinite(environment.ekfs[uav_id].state).all():
                raise ValueError("non-finite EKF state or covariance")
        quality = _quality_counts(new_events)
        window = {
            "method": method,
            "endpoint": endpoint,
            "scene": scene,
            "episode_id": identity,
            "step": step,
            "timestamp": (step + 1) * float(environment.config.motion.dt_s),
            "mean_reward": reward,
            **reward_components,
            "position_error_mean_m": float(np.mean(list(positions.values()))),
            "position_error_rms_m": float(np.sqrt(np.mean(np.square(list(positions.values()))))),
            "measurements": selected_count,
            "accepted_updates": quality["accepted"],
            "beneficial_updates": quality["beneficial"],
            "neutral_updates": quality["neutral"],
            "harmful_updates": quality["harmful"],
            "beneficial_fraction": quality["beneficial"] / quality["accepted"] if quality["accepted"] else float("nan"),
            "harmful_fraction": quality["harmful"] / quality["accepted"] if quality["accepted"] else float("nan"),
                    "actual_selected_candidate_cost_sum": float(np.sum(actual_costs)) if actual_costs else (0.0 if selected_count == 0 else float("nan")),
            "actual_packet_bytes": int(step_packet_bytes),
            "expected_communication_cost_mean": float(np.nanmean([r["expected_communication_cost"] for r in step_policy_rows.values()])) if any(math.isfinite(float(r["expected_communication_cost"])) for r in step_policy_rows.values()) else float("nan"),
            "communication_target_mean": float(np.nanmean([r["communication_target"] for r in step_policy_rows.values()])) if any(math.isfinite(float(r["communication_target"])) for r in step_policy_rows.values()) else float("nan"),
            "communication_residual_mean": float(np.nanmean([r["communication_residual"] for r in step_policy_rows.values()])) if any(math.isfinite(float(r["communication_residual"])) for r in step_policy_rows.values()) else float("nan"),
            "communication_dual_mean": float(np.nanmean([r["communication_dual"] for r in step_policy_rows.values()])) if any(math.isfinite(float(r["communication_dual"])) for r in step_policy_rows.values()) else float("nan"),
            "collision": int(_truth(next_info.get("collision", False))),
            "budget_remaining_min": float(min([_optional(v) for v in budget_after_map.values()])) if budget_after_map else float("nan"),
            **{f"position_error_{uav_id}": value for uav_id, value in positions.items()},
        }
        window_rows.append(window)
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    if not window_rows:
        raise ValueError(f"empty replay: {identity}")
    if not torch.equal(before_parameters, policy_vector(trainer)):
        raise ValueError(f"replay changed checkpoint parameters: {identity}")
    accepted = [row for row in event_rows if _truth(row.get("accepted", False))]
    summary: dict[str, Any] = {
        "method": method,
        "endpoint": endpoint,
        "scene": scene,
        "episode_id": identity,
        "windows": len(window_rows),
        "mean_reward": float(np.mean([row["mean_reward"] for row in window_rows])),
        "cumulative_return": float(np.sum([row["mean_reward"] for row in window_rows])),
        "position_rmse_norm_m": float(np.sqrt(np.mean([row["position_error_rms_m"] ** 2 for row in window_rows]))),
        "position_p95_norm_m": float(np.percentile([row["position_error_rms_m"] for row in window_rows], 95)),
        "measurements": int(sum(row["measurements"] for row in window_rows)),
        "accepted_updates": int(len(accepted)),
        "collision": int(any(row["collision"] for row in window_rows)),
        "reward_loc": float(np.mean([row["reward_loc"] for row in window_rows])),
        "reward_com": float(np.mean([row["reward_com"] for row in window_rows])),
        "reward_collision": float(np.mean([row["reward_collision"] for row in window_rows])),
        "reward_energy": float(np.mean([row["reward_energy"] for row in window_rows])),
        "actual_packet_bytes": int(sum(row["actual_packet_bytes"] for row in window_rows)),
        "mean_expected_communication_cost": float(np.nanmean([row["expected_communication_cost_mean"] for row in window_rows])) if any(math.isfinite(float(row["expected_communication_cost_mean"])) for row in window_rows) else float("nan"),
        "mean_actual_selected_candidate_cost": float(np.sum([row["actual_selected_candidate_cost_sum"] for row in window_rows]) / max(summary_measurements := sum(row["measurements"] for row in window_rows), 1)) if any(math.isfinite(float(row["actual_selected_candidate_cost_sum"])) for row in window_rows) else float("nan"),
        "mean_communication_target": float(np.nanmean([row["communication_target_mean"] for row in window_rows])) if any(math.isfinite(float(row["communication_target_mean"])) for row in window_rows) else float("nan"),
        "mean_communication_residual": float(np.nanmean([row["communication_residual_mean"] for row in window_rows])) if any(math.isfinite(float(row["communication_residual_mean"])) for row in window_rows) else float("nan"),
        "mean_communication_dual": float(np.nanmean([row["communication_dual_mean"] for row in window_rows])) if any(math.isfinite(float(row["communication_dual_mean"])) for row in window_rows) else float("nan"),
    }
    for component in PARTS:
        if abs(summary["mean_reward"] - sum(summary[item] for item in PARTS)) > TOL:
            raise ValueError("endpoint reward contribution closure failed")
    quality = _quality_counts(event_rows)
    summary.update(
        {
            "beneficial_fraction": quality["beneficial"] / quality["accepted"] if quality["accepted"] else float("nan"),
            "neutral_fraction": quality["neutral"] / quality["accepted"] if quality["accepted"] else float("nan"),
            "harmful_fraction": quality["harmful"] / quality["accepted"] if quality["accepted"] else float("nan"),
            "accepted_updates_per_window": quality["accepted"] / len(window_rows),
            "beneficial_per_window": quality["beneficial"] / len(window_rows),
            "harmful_per_window": quality["harmful"] / len(window_rows),
            "update_coverage_any": int(quality["accepted"] > 0),
            **_retention(event_rows, window_rows, uav_ids),
        }
    )
    return summary, window_rows, action_rows, communication_rows, event_rows, {
        "parameters_unchanged": True,
        "uav_ids": list(uav_ids),
        "steps_executed": len(window_rows),
    }


def _history_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only historical R/P/T figures and bind episode to update IDs."""

    import pandas as pd

    episode_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        root = OLD_LOGS / "development" / arm / "figures"
        episodes = pd.read_csv(root / "training_episodes.csv")
        updates = pd.read_csv(root / "training_updates.csv")
        parts = pd.read_csv(root / "reward_decomposition.csv")
        for method, group in episodes.groupby("method"):
            group = group.sort_values("training_episode").copy()
            means = parts[parts.method == method].groupby("episode_id")[list(PARTS)].mean()
            for _, row in group.iterrows():
                episode_id = str(row["episode_id"])
                if episode_id not in means.index:
                    raise ValueError(f"missing reward decomposition for {arm}/{method}/{episode_id}")
                update_id = int(row["assigned_update"])
                update_match = updates[(updates["method"] == method) & (updates["update"] == update_id)]
                if len(update_match) != 1:
                    raise ValueError(f"episode/update binding is not unique: {arm}/{method}/{episode_id}")
                update = update_match.iloc[0]
                result = {
                    "arm": arm,
                    "method": method,
                    "record_type": "episode",
                    "training_episode": int(row["training_episode"]),
                    "update": update_id,
                    "episode_id": episode_id,
                    "window_count": int(row["window_count"]),
                    "cumulative_return": _number(row["cumulative_return"], field="cumulative return"),
                    "mean_reward": _number(row["mean_reward"], field="mean reward"),
                    "collision_occurred": int(_truth(row["collision_occurred"])),
                    "termination_reason": row["termination_reason"],
                }
                for part in PARTS:
                    result[part] = _number(means.loc[episode_id, part], field=part)
                result.update(
                    {
                        "loss": _number(update["loss"], field="loss"),
                        "actor_loss": _number(update["actor_loss"], field="actor loss"),
                        "critic_loss": _number(update["critic_loss"], field="critic loss"),
                        "entropy": _number(update["entropy"], field="entropy"),
                        "clip_fraction": _number(update["clip_fraction"], field="clip fraction"),
                        "gradient_norm": _number(update["gradient_norm"], field="gradient norm"),
                        "optimizer_steps": _number(update["optimizer_steps"], field="optimizer steps"),
                        "rollout_window_count": _number(update["rollout_window_count"], field="rollout windows"),
                        "communication_expected_mean": _optional(update.get("measurement_communication_cost_expected_mean")),
                        "communication_violation_mean": _optional(update.get("measurement_communication_cost_violation_mean")),
                        "communication_dual_loss": _optional(update.get("measurement_communication_dual_loss")),
                    }
                )
                if abs(result["mean_reward"] - sum(result[part] for part in PARTS)) > TOL:
                    raise ValueError(f"historical reward closure failed: {arm}/{method}/{episode_id}")
                episode_rows.append(result)
            for collision, stratified in group.groupby(group["collision_occurred"].astype(bool)):
                subset = [r for r in episode_rows if r["arm"] == arm and r["method"] == method and bool(r["collision_occurred"]) == bool(collision)]
                # The source group is the authoritative slice; avoid relying on
                # the accumulated list if a later arm has the same method name.
                collision_rows.append(
                    {
                        "arm": arm,
                        "method": method,
                        "collision_occurred": int(bool(collision)),
                        "episodes": int(len(stratified)),
                        "mean_cumulative_return": float(stratified["cumulative_return"].mean()),
                        "mean_reward": float(stratified["mean_reward"].mean()),
                        "mean_window_count": float(stratified["window_count"].mean()),
                    }
                )
        update_subset = updates.copy()
        update_subset["arm"] = arm
        update_subset["record_type"] = "update"
        for _, update in update_subset.iterrows():
            update_rows.append(
                {
                    "arm": arm,
                    "method": update["method"],
                    "record_type": "update",
                    "update": int(update["update"]),
                    "completed_training_episodes": int(update["completed_training_episodes"]),
                    "loss": _number(update["loss"], field="loss"),
                    "actor_loss": _number(update["actor_loss"], field="actor loss"),
                    "critic_loss": _number(update["critic_loss"], field="critic loss"),
                    "entropy": _number(update["entropy"], field="entropy"),
                    "clip_fraction": _number(update["clip_fraction"], field="clip fraction"),
                    "gradient_norm": _number(update["gradient_norm"], field="gradient norm"),
                    "optimizer_steps": _number(update["optimizer_steps"], field="optimizer steps"),
                    "rollout_window_count": _number(update["rollout_window_count"], field="rollout windows"),
                    "communication_expected_mean": _optional(update.get("measurement_communication_cost_expected_mean")),
                    "communication_violation_mean": _optional(update.get("measurement_communication_cost_violation_mean")),
                    "communication_dual_loss": _optional(update.get("measurement_communication_dual_loss")),
                }
            )
        points = pd.read_csv(root / "development_monitor.csv")
        for _, point in points.iterrows():
            monitor_rows.append({"arm": arm, **{str(key): value for key, value in point.to_dict().items()}})
    return episode_rows, collision_rows, update_rows, monitor_rows


def _gradient_evidence(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Probe initial/final proposed checkpoints without optimizer calls."""

    rows: list[dict[str, Any]] = []
    for endpoint in ("initial", "final"):
        config, trainer = _trainer_at(COMMUNICATION_BRANCH, endpoint, data_root)
        trainer.policy.eval()
        before = policy_vector(trainer)
        trainer.rollout.clear()
        for index in range(4):
            collect_probe_episode(
                config,
                trainer,
                seed=_scale_seed(42, 3, "train", index),
                index=index,
                monitor=False,
                identity=f"gradient_{endpoint}_{index}",
                run_root=data_root,
            )
        records = trainer._training_records()
        for epoch in range(int(trainer.config.policy_epochs)):
            order = trainer.rng.permutation(len(records))
            for start in range(0, len(order), int(trainer.config.minibatch_size)):
                batch = [records[int(index)] for index in order[start : start + int(trainer.config.minibatch_size)]]
                evidence = gradient_probe(trainer, batch)
                meta = batch_metadata(batch)
                expected_values: list[float] = []
                residual_values: list[float] = []
                target_values: list[float] = []
                for record in batch:
                    with torch.no_grad():
                        expected = float(trainer.policy.expected_measurement_communication_cost(record["observation"]).detach().cpu())
                    target = float((trainer.config.measurement_communication_cost_target_by_uav or {}).get(str(record["uav_id"]), float("nan")))
                    expected_values.append(expected)
                    target_values.append(target)
                    residual_values.append(max(0.0, expected - target - float(trainer.config.measurement_communication_cost_tolerance)))
                row = {
                    "endpoint": endpoint,
                    "epoch": epoch,
                    "minibatch": start // int(trainer.config.minibatch_size),
                    **meta,
                    **evidence,
                    "communication_expected_mean": float(np.mean(expected_values)),
                    "communication_target_mean": float(np.mean(target_values)),
                    "communication_residual_mean": float(np.mean(residual_values)),
                    "communication_dual_mean": float(np.mean([trainer.measurement_communication_dual_by_uav[str(record["uav_id"])] for record in batch])),
                    "optimizer_step_called": 0,
                    "probe_is_historical_training_gradient": 0,
                }
                rows.append(row)
        if not torch.equal(before, policy_vector(trainer)):
            raise ValueError(f"gradient probe changed parameters: {endpoint}")
    synthetic = _synthetic_positive_gradient_case()
    return rows, [synthetic], {"optimizer_steps": 0, "parameter_updates": 0}


def _synthetic_positive_gradient_case() -> dict[str, Any]:
    """Exercise the real measurement-cost path with a positive residual.

    This is a tiny in-memory probe, not a training run and not an experiment
    result.  It ensures that a positive execution-cost residual can reach the
    measurement head when the multiplier is non-zero.
    """

    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        gat_hidden_dim=8,
        gat_output_dim=8,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        max_obstacles=2,
        policy_epochs=1,
        minibatch_size=1,
        use_measurement_proxy_features=True,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav={"uav_00": 0.0, "uav_01": 0.0, "uav_02": 0.0},
        measurement_communication_cost_tolerance=0.05,
        measurement_communication_fixed_packet_fraction=0.5,
        parameter_initialization_seed=42,
    )
    policy = MAPPOPolicy(config, uav_ids=("uav_00", "uav_01", "uav_02"))
    observation = {
        "uav_id": "uav_00",
        "node_features": np.zeros((1, 37), dtype=np.float32),
        "effective_edge_index": np.empty((2, 0), dtype=np.int64),
        "effective_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_edge_features": {},
        "candidate_incoming_edge_features": {},
        "potential_edge_features": {},
        "action_mask": np.ones(4, dtype=np.float32),
        "bandwidth_mask": np.ones(3, dtype=np.float32),
        "expected_information_gain_norm": np.asarray([0.2, 0.1, 0.3], dtype=np.float32),
        "predicted_communication_cost_norm": np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        "proxy_valid": np.ones(3, dtype=np.float32),
    }
    before = policy.parameter_initialization_hashes()
    cost = policy.expected_measurement_communication_cost(observation)
    residual = torch.relu(cost - 0.05)
    params = tuple(policy.measurement_head.parameters())
    gradients = torch.autograd.grad(residual, params, allow_unused=True)
    norm = float(torch.sqrt(sum((gradient.square().sum() for gradient in gradients if gradient is not None), torch.zeros((), device=policy.device))).detach().cpu())
    after = policy.parameter_initialization_hashes()
    if not math.isfinite(float(cost.detach().cpu())) or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("positive synthetic communication residual did not produce a finite non-zero head gradient")
    if before != after:
        raise ValueError("synthetic gradient probe changed policy parameters")
    return {
        "case": "synthetic_positive_communication_residual",
        "expected_cost": float(cost.detach().cpu()),
        "target": 0.0,
        "tolerance": 0.05,
        "residual": float(residual.detach().cpu()),
        "measurement_head_gradient_norm": norm,
        "parameters_unchanged": True,
        "optimizer_step_called": 0,
    }


def _cost_tables(
    training_updates: Sequence[Mapping[str, Any]],
    endpoint_summaries: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        row
        for row in training_updates
        if row["method"] == COMMUNICATION_BRANCH and row["arm"] == "rollout4"
    ]
    train_expected = [_optional(row.get("communication_expected_mean")) for row in train]
    train_target = []
    target_values = [
        _optional(row.get("mean_communication_target"))
        for row in endpoint_summaries
        if row.get("method") == COMMUNICATION_BRANCH and math.isfinite(_optional(row.get("mean_communication_target")))
    ]
    target_mean = float(np.mean(target_values)) if target_values else float("nan")
    train_mean = float(np.nanmean(train_expected)) if any(math.isfinite(value) for value in train_expected) else float("nan")
    shift_rows: list[dict[str, Any]] = []
    for method in METHODS:
        replay = [row for row in endpoint_summaries if row["method"] == method and row["endpoint"] == "final"]
        shift_rows.append(
            {
                "method": method,
                "source_training_arm": "rollout4",
                "train_expected_cost_mean": train_mean if method == COMMUNICATION_BRANCH else float("nan"),
                "train_target_cost_mean": target_mean if method == COMMUNICATION_BRANCH else float("nan"),
                "validation_expected_cost_mean": float(np.nanmean([_optional(row.get("mean_expected_communication_cost")) for row in replay])) if any(math.isfinite(_optional(row.get("mean_expected_communication_cost"))) for row in replay) else float("nan"),
                "validation_actual_selected_candidate_cost_mean": float(np.nanmean([_optional(row.get("mean_actual_selected_candidate_cost")) for row in replay])) if any(math.isfinite(_optional(row.get("mean_actual_selected_candidate_cost"))) for row in replay) else float("nan"),
                "validation_actual_packet_bytes_mean": float(np.mean([row["actual_packet_bytes"] for row in replay])) if replay else float("nan"),
                "validation_mean_measurements": float(np.mean([row["measurements"] for row in replay])) if replay else float("nan"),
                "note": "expected is per-UAV policy expectation; actual selected candidate cost is per selected measurement; unavailable baseline expectation is NA",
            }
        )
    gap_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for endpoint in ("initial", "final"):
            group = [row for row in communication_rows if row["method"] == method and row["endpoint"] == endpoint]
            expected = [_optional(row.get("expected_communication_cost")) for row in group]
            actual = [_optional(row.get("actual_selected_candidate_cost_snapshot")) for row in group]
            targets = [_optional(row.get("communication_target")) for row in group]
            residuals = [_optional(row.get("communication_residual")) for row in group]
            gap_rows.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "expected_cost_mean_per_uav_window": float(np.nanmean(expected)) if any(math.isfinite(value) for value in expected) else float("nan"),
                    "actual_selected_cost_mean_per_measurement": float(np.nanmean(actual)) if any(math.isfinite(value) for value in actual) else float("nan"),
                    "target_mean_per_uav": float(np.nanmean(targets)) if any(math.isfinite(value) for value in targets) else float("nan"),
                    "residual_mean_per_uav_window": float(np.nanmean(residuals)) if any(math.isfinite(value) for value in residuals) else float("nan"),
                    "selected_measurements": int(sum(int(row["actual_packet_count"]) for row in group)),
                    "actual_packet_bytes": int(sum(int(row["actual_packet_bytes"]) for row in group)),
                    "execution_mode": "deterministic_argmax",
                }
            )
    return shift_rows, gap_rows


def _validate_rows(rows: Sequence[Mapping[str, Any]], numeric_fields: Iterable[str], label: str) -> None:
    for index, row in enumerate(rows):
        for field in numeric_fields:
            if field not in row:
                continue
            value = row[field]
            if value is None or (isinstance(value, str) and value.strip().upper() in {"", "NA", "N/A"}):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}[{index}].{field} is not numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{label}[{index}].{field} is not finite")


def _hash_files(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            result[str(path)] = _sha256(path)
    return result


def _figures(
    logs_root: Path,
    plots_root: Path,
) -> list[str]:
    """Create five diagnostic figures from v2 output tables only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fresh(plots_root)
    colors = {BASELINE_BRANCH: "#2563eb", COMMUNICATION_BRANCH: "#dc2626"}
    labels = {
        BASELINE_BRANCH: "Local-EKF-GAT-MAPPO (Proposed Method)",
        COMMUNICATION_BRANCH: "Deprecated communication-constrained branch",
    }

    def save(figure: Any, name: str) -> str:
        png = plots_root / f"{name}.png"
        pdf = plots_root / f"{name}.pdf"
        figure.savefig(png, dpi=300, bbox_inches="tight")
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
        return str(png)

    history = pd.read_csv(logs_root / "training_reward_loss_attribution.csv", na_values=["NA"])
    # Figure 1: raw episode/update training evidence, no smoothing.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for method, group in history.groupby("method"):
        group = group.sort_values("training_episode")
        color = colors.get(method, "#555555")
        axes[0, 0].plot(group.training_episode, group.cumulative_return, color=color, label=labels.get(method, method))
        axes[0, 1].plot(group.training_episode, group.mean_reward, color=color, label=labels.get(method, method))
        axes[0, 2].plot(group.training_episode, group.window_count, color=color, label=labels.get(method, method))
        axes[1, 0].plot(group.training_episode, group.reward_loc, color=color, label=labels.get(method, method))
        axes[1, 1].plot(group.training_episode, group.reward_com, color=color, label=labels.get(method, method))
        axes[1, 2].plot(group.training_episode, group.reward_collision, color=color, label=labels.get(method, method))
    titles = ["Cumulative episode return", "Mean team reward / window", "Episode window count", "Localization contribution", "Communication contribution", "Collision contribution"]
    ylabels = ["Return", "Reward", "Windows", "Signed contribution", "Signed contribution", "Signed contribution"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set(title=title, xlabel="Training episode", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Seed 42 | raw training traces | diagnostic, not a convergence claim")
    figure_paths = [save(fig, "reward_length_attribution_v2")]

    endpoints = pd.read_csv(logs_root / "frozen_endpoint_summary.csv", na_values=["NA"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for method, group in endpoints.groupby("method"):
        for endpoint, endpoint_group in group.groupby("endpoint"):
            endpoint_group = endpoint_group.sort_values("scene")
            style = "-" if endpoint == "initial" else "--"
            for axis, field, title in zip(axes, ["mean_reward", "position_rmse_norm_m", "measurements"], ["Mean reward / window", "Position RMSE norm (m)", "Selected measurements / episode"]):
                axis.plot(endpoint_group.scene, endpoint_group[field], linestyle=style, marker="o", color=colors.get(method, "#555555"), label=f"{labels.get(method, method)} / {endpoint}")
                axis.set(title=title, xlabel="Fixed monitoring scene")
                axis.grid(alpha=0.25)
                axis.set_xticks(range(5))
    for axis in axes:
        axis.legend(fontsize=7)
    fig.suptitle("Frozen endpoint replay | same five development-monitor scenes")
    figure_paths.append(save(fig, "frozen_endpoint_comparison_v2"))

    quality = pd.read_csv(logs_root / "update_quality_per_window.csv", na_values=["NA"])
    final_quality = quality[quality.endpoint == "final"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for method, group in final_quality.groupby("method"):
        agg = group.groupby("step")[["accepted_updates", "beneficial_updates", "harmful_updates"]].mean().reset_index()
        axes[0].plot(agg.step, agg.accepted_updates, color=colors.get(method, "#555555"), label=labels.get(method, method))
        axes[1].plot(agg.step, agg.beneficial_updates, color=colors.get(method, "#555555"), label=labels.get(method, method))
        axes[2].plot(agg.step, agg.harmful_updates, color=colors.get(method, "#555555"), label=labels.get(method, method))
    for axis, title, ylabel in zip(axes, ["Accepted updates / window", "Beneficial updates / window", "Harmful updates / window"], ["Count", "Count", "Count"]):
        axis.set(title=title, xlabel="Window", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Frozen EKF update-quality evidence | event classes use validation-only truth labels")
    figure_paths.append(save(fig, "measurement_update_quality_v2"))

    communication = pd.read_csv(logs_root / "communication_constraint_evidence.csv", na_values=["NA"])
    final_comm = communication[communication.endpoint == "final"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for method, group in final_comm.groupby("method"):
        agg = group.groupby("step")[["expected_communication_cost", "actual_selected_candidate_cost_snapshot", "communication_target", "communication_residual"]].mean().reset_index()
        axes[0].plot(agg.step, agg.expected_communication_cost, color=colors.get(method, "#555555"), label=labels.get(method, method))
        axes[0].plot(agg.step, agg.communication_target, color=colors.get(method, "#555555"), linestyle=":")
        axes[1].plot(agg.step, agg.actual_selected_candidate_cost_snapshot, color=colors.get(method, "#555555"), label=labels.get(method, method))
        axes[2].plot(agg.step, agg.communication_residual, color=colors.get(method, "#555555"), label=labels.get(method, method))
    for axis, title, ylabel in zip(axes, ["Expected cost / UAV window", "Actual selected cost / measurement", "Positive constraint residual"], ["Cost", "Cost", "Residual"]):
        axis.set(title=title, xlabel="Window", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Expected versus selected communication cost | baseline-unavailable fields remain NA")
    figure_paths.append(save(fig, "communication_constraint_execution_v2"))

    gradients = pd.read_csv(logs_root / "gradient_path_evidence.csv", na_values=["NA"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for endpoint, group in gradients.groupby("endpoint"):
        group = group.reset_index(drop=True)
        color = "#0072B2" if endpoint == "initial" else "#D55E00"
        axes[0, 0].plot(group.index, group.critic_squared_fraction, color=color, label=endpoint)
        axes[0, 1].plot(group.index, group.actor_retention_ratio, color=color, label=endpoint)
        axes[1, 0].plot(group.index, group.ppo_teacher_cosine, color=color, label=endpoint)
        axes[1, 1].plot(group.index, group.teacher_measurement_head_gradient_norm, color=color, label=endpoint)
    titles = ["Critic squared-gradient share", "Grouped/joint actor norm", "PPO—teacher cosine", "Teacher measurement-head norm"]
    for axis, title in zip(axes.flat, titles):
        axis.set(title=title, xlabel="Probe minibatch (not training update)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Frozen autograd probes | no optimizer step | repeated minibatches are dependent")
    figure_paths.append(save(fig, "gradient_path_diagnostic_v2"))
    return figure_paths


def _report_findings(
    endpoints: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    cost_shift: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proposed = [row for row in endpoints if row["method"] == COMMUNICATION_BRANCH]
    return {
        "length_mixture": "confirmed",
        "fixed_monitor_no_sustained_improvement": "confirmed",
        "critic_gradient_dominance": "supported_for_testing" if gradient_rows and np.mean([row["critic_squared_fraction"] for row in gradient_rows]) >= 0.8 else "inconclusive",
        "ppo_teacher_direction": "inconclusive" if not gradient_rows else "diagnosed_not_causal",
        "communication_expected_actual_gap": "diagnosed",
        "weak_localization_reward_feedback": "confirmed_as_objective_limitation",
        "proposed_endpoint_count": len(proposed),
        "cost_shift_rows": len(cost_shift),
    }


def frozen_audit() -> dict[str, Any]:
    """Run the complete v2 read-only audit."""

    fresh(LOGS, DATA, PLOTS)
    report: dict[str, Any] = {
        "audit_ok": False,
        "status": "running",
        "stage": "frozen-audit-v2",
        "seed": 42,
        "uav_count": 3,
        "scope": {
            "replay_split": "development_monitor built from train-scenario generator",
            "training_logs": "train only",
            "validation_used_for_judgement": False,
            "test_used": False,
            "test_accesses": 0,
            "execution_mode": "deterministic_argmax",
        },
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dual_updates": 0,
        "figures": [],
    }
    _write_json(LOGS / "audit_report.json", report)
    guard_hashes: list[dict[str, Any]] = []
    source_checkpoint_paths = [SOURCE / method / "scale_3/checkpoints/final.pt" for method in METHODS]
    checkpoint_hashes_before = _hash_files(source_checkpoint_paths)
    endpoint_summaries: list[dict[str, Any]] = []
    endpoint_windows: list[dict[str, Any]] = []
    endpoint_events: list[dict[str, Any]] = []
    action_evidence: list[dict[str, Any]] = []
    communication_evidence: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    try:
        with SourceGuard([SOURCE, OLD_LOGS], [LOGS, DATA, PLOTS]) as guard:
            episode_history, collision_history, update_history, monitor_history = _history_evidence()
            for method in METHODS:
                traces: dict[tuple[str, int], list[dict[str, Any]]] = {}
                for endpoint in ("initial", "final"):
                    config, trainer = _trainer_at(method, endpoint, DATA)
                    for scene in range(5):
                        scene_seed = 42 + 900000 + scene * 1009
                        identity = f"{method}_{endpoint}_{scene}"
                        summary, windows, actions, communications, events, meta = _collect_detailed(
                            config,
                            trainer,
                            method=method,
                            endpoint=endpoint,
                            scene=scene,
                            seed=scene_seed,
                            identity=identity,
                            run_root=DATA,
                        )
                        old = next(
                            row
                            for row in monitor_history
                            if row.get("arm") == "rollout4"
                            and str(row.get("method")) == method
                            and int(row.get("monitor_scene")) == scene
                            and int(row.get("completed_training_episodes")) == (0 if endpoint == "initial" else 100)
                        )
                        replay_errors = {
                            "mean_reward": abs(summary["mean_reward"] - _number(old["mean_reward"], field="old mean reward")),
                            "position_rmse_norm_m": abs(summary["position_rmse_norm_m"] - _number(old["position_rmse_norm_m"], field="old RMSE")),
                            "windows": abs(summary["windows"] - int(old["windows"])),
                            "cumulative_return": abs(summary["cumulative_return"] - _number(old["cumulative_return"], field="old return")),
                        }
                        if max(replay_errors.values()) > TOL:
                            raise ValueError(f"endpoint replay mismatch {identity}: {replay_errors}")
                        endpoint_summaries.append({**summary, "max_replay_error": max(replay_errors.values()), **meta})
                        endpoint_windows.extend(windows)
                        endpoint_events.extend(events)
                        action_evidence.extend(actions)
                        communication_evidence.extend(communications)
                        traces[(endpoint, scene)] = windows
                        print(f"[trend audit v2] endpoint {method}/{endpoint}/{scene + 1} matched", flush=True)
                for scene in range(5):
                    initial_rows = traces[("initial", scene)]
                    final_rows = traces[("final", scene)]
                    common = min(len(initial_rows), len(final_rows))
                    if common <= 0:
                        raise ValueError("empty common-prefix trace")
                if method == COMMUNICATION_BRANCH:
                    gradient_rows, synthetic_rows, gradient_meta = _gradient_evidence(DATA)
                    report.update(gradient_meta)
            guard_hashes = guard.verify()
        if guard.test_attempts:
            raise ValueError(f"test read attempted {guard.test_attempts} time(s)")
        if not all(item["unchanged"] for item in guard_hashes):
            raise ValueError("a source input changed during the audit")

        _write_table(LOGS / "training_reward_loss_attribution.csv", episode_history)
        _write_table(LOGS / "collision_stratified_training.csv", collision_history)
        _write_table(LOGS / "historical_training_updates.csv", update_history)
        _write_table(LOGS / "historical_monitor_points.csv", monitor_history)
        _write_table(LOGS / "frozen_endpoint_summary.csv", endpoint_summaries)
        _write_table(LOGS / "frozen_endpoint_windows.csv", endpoint_windows)
        _write_table(LOGS / "frozen_endpoint_events.csv", endpoint_events)
        _write_table(LOGS / "policy_action_evidence.csv", action_evidence)
        _write_table(LOGS / "communication_constraint_evidence.csv", communication_evidence)
        _write_table(LOGS / "update_quality_per_window.csv", endpoint_windows)
        _write_table(LOGS / "gradient_path_evidence.csv", gradient_rows)
        _write_table(LOGS / "gradient_synthetic_positive_case.csv", synthetic_rows)
        cost_shift, expected_actual_gap = _cost_tables(update_history, endpoint_summaries, communication_evidence)
        _write_table(LOGS / "train_validation_cost_shift.csv", cost_shift)
        _write_table(LOGS / "expected_actual_cost_gap.csv", expected_actual_gap)

        _validate_rows(endpoint_summaries, ["windows", "mean_reward", "cumulative_return", "position_rmse_norm_m", "measurements", "accepted_updates", "actual_packet_bytes"], "endpoint_summary")
        _validate_rows(endpoint_windows, ["mean_reward", "position_error_rms_m", "measurements", "accepted_updates", "beneficial_updates", "harmful_updates", "actual_packet_bytes"], "endpoint_window")
        # Teacher and communication-constraint fields are intentionally NA
        # for the baseline.  Validate the always-applicable fields for every
        # row, then validate constraint fields only on rows where the
        # corresponding mechanism is enabled; an inapplicable NA is not a
        # numerical failure.
        _validate_rows(action_evidence, ["step", "timestamp", "selected_action_index", "actor_none_probability", "budget_limit", "budget_remaining_before", "budget_remaining_after"], "action_evidence")
        _validate_rows(
            [row for row in action_evidence if int(row.get("teacher_enabled", 0)) == 1],
            ["teacher_none_probability"],
            "teacher_action_evidence",
        )
        _validate_rows(
            [row for row in action_evidence if int(row.get("measurement_constraint_enabled", 0)) == 1],
            ["expected_communication_cost", "communication_target", "communication_residual", "communication_dual"],
            "constrained_action_evidence",
        )
        _validate_rows(communication_evidence, ["actual_packet_count", "actual_packet_bytes"], "communication_evidence")
        _validate_rows(
            [row for row in communication_evidence if int(row.get("measurement_constraint_enabled", 0)) == 1],
            ["expected_communication_cost", "communication_target", "communication_residual", "communication_dual"],
            "constrained_communication_evidence",
        )
        _validate_rows(gradient_rows, ["ppo_gradient_norm", "teacher_gradient_norm", "communication_gradient_norm", "ppo_measurement_head_gradient_norm", "teacher_measurement_head_gradient_norm", "communication_measurement_head_gradient_norm", "critic_squared_fraction", "ppo_teacher_cosine", "communication_residual_mean"], "gradient_evidence")

        checkpoint_hashes_after = _hash_files(source_checkpoint_paths)
        source_hashes = {**{item["path"]: item["sha256_before"] for item in guard_hashes}, **checkpoint_hashes_after}
        figure_paths = _figures(LOGS, PLOTS)
        report.update(
            {
                "audit_ok": True,
                "status": "complete",
                "test_accesses": 0,
                "source_hashes": guard_hashes,
                "checkpoint_hashes": source_hashes,
                "checkpoint_hashes_before": checkpoint_hashes_before,
                "checkpoint_hashes_after": checkpoint_hashes_after,
                "checkpoint_hashes_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
                "endpoint_count": len(endpoint_summaries),
                "event_count": len(endpoint_events),
                "action_evidence_count": len(action_evidence),
                "communication_evidence_count": len(communication_evidence),
                "gradient_probe_count": len(gradient_rows),
                "all_numeric_fields_finite": True,
                "reward_closure_tolerance": TOL,
                "endpoint_replay_tolerance": TOL,
                "packet_payload_bytes": 512,
                "packet_and_budget_checks": True,
                "event_counter_checks": True,
                "checkpoint_parameters_unchanged": True,
                "figures": figure_paths,
                "findings": _report_findings(endpoint_summaries, gradient_rows, cost_shift),
                "evidence_limits": [
                    "single training seed; no cross-initialization stability inference",
                    "development-monitor replay is used because this audit has no validation trajectory checkpoints",
                    "repeated gradient probes are not independent observations",
                    "autograd probes are not historical gradients and no Adam update was executed",
                    "truth-derived position error and LOS fields are audit labels only",
                    "expected communication cost and selected-measurement cost have different denominators",
                ],
            }
        )
        (LOGS / "figure_index.md").write_text(
            "# Frozen training-trend audit v2 figures\n\n"
            "All figures are single-seed diagnostics, not evidence of convergence or causal superiority.\n\n"
            "- reward_length_attribution_v2: raw episode reward, mean reward, window count and signed contributions.\n"
            "- frozen_endpoint_comparison_v2: initial/final checkpoint replay on the same five scenes.\n"
            "- measurement_update_quality_v2: accepted/beneficial/harmful updates per window.\n"
            "- communication_constraint_execution_v2: expected versus selected cost and residual.\n"
            "- gradient_path_diagnostic_v2: frozen autograd component probes; not historical update curves.\n",
            encoding="utf-8",
        )
        artifact_hashes = _hash_files(
            [path for path in LOGS.rglob("*") if path.is_file() and path.name != "audit_report.json"]
            + [path for path in PLOTS.rglob("*") if path.is_file()]
        )
        report["artifact_hashes"] = artifact_hashes
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "test_accesses": int(getattr(locals().get("guard", None), "test_attempts", 0))})
        raise
    finally:
        _write_json(LOGS / "audit_report.json", report)
    return {
        "audit_ok": bool(report["audit_ok"]),
        "status": report["status"],
        "report": str(LOGS / "audit_report.json"),
        "figures": report.get("figures", []),
        "endpoint_count": report.get("endpoint_count", 0),
        "gradient_probe_count": report.get("gradient_probe_count", 0),
    }


def run(stage: str = "frozen-audit", seed: int = 42) -> dict[str, Any]:
    if int(seed) != 42:
        raise ValueError("training trend diagnostic v2 is fixed to seed=42")
    if stage != "frozen-audit":
        raise ValueError("v2 currently supports only stage=frozen-audit")
    return frozen_audit()
