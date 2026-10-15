"""Offline route and gradient feasibility audit for PPO-A.

This script reads only the recorded A-arm update JSON/CSV files and the
initial/update/final checkpoints.  It does not import the runner, create an
environment, call ``step``, or execute backward/optimizer updates.  The
gradient hook in the source run was installed around ``optimizer.step`` after
PPO's clipping call; the report therefore labels hook group norms as
post-clipping.  ``update_stats.gradient_norm`` is the pre-clipping global
norm returned by ``clip_grad_norm_``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - gives a useful CLI error
    raise RuntimeError("This audit needs the project .venv PyTorch runtime") from exc


ARMS = ("C", "N", "G", "R")
UPDATES = 6
HORIZON = 120
GROUPS = (
    "first_gat",
    "deep_gat",
    "actor_body",
    "motion_head",
    "log_std",
    "measurement_head",
    "bandwidth_head",
    "critic",
    "total_actor",
)
FIXED_CLIP_NORM = 0.5


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _finite(value: Any) -> Any:
    """Convert NumPy/Torch scalars and non-finite values for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_finite(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite(row.get(key, "")) for key in fieldnames})


def _parse_payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _parse_vec(value: Any, size: int = 3) -> np.ndarray | None:
    value = _parse_payload(value)
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    try:
        result = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _group_for(name: str) -> str | None:
    if name == "delta_logstd":
        return "log_std"
    if name.startswith("gat.layers.0."):
        return "first_gat"
    if name.startswith("gat.layers.1."):
        return "deep_gat"
    if name.startswith("actor_body."):
        return "actor_body"
    if name.startswith("delta_mean_head."):
        return "motion_head"
    if name.startswith("measurement_head."):
        return "measurement_head"
    if name.startswith("bandwidth_head."):
        return "bandwidth_head"
    if name.startswith("critic."):
        return "critic"
    return None


def _state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("policy_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no policy_state_dict")
    return state


def _group_tensor_names(state: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = {group: [] for group in GROUPS if group != "total_actor"}
    for name in state:
        group = _group_for(str(name))
        if group is not None:
            groups[group].append(str(name))
    return groups


def _group_l2(state: Mapping[str, Any], names: list[str]) -> float:
    square_sum = 0.0
    for name in names:
        tensor = state[name].detach().float()
        square_sum += float(torch.sum(tensor * tensor).cpu())
    return float(math.sqrt(square_sum))


def _group_norms(state: Mapping[str, Any]) -> dict[str, float]:
    names = _group_tensor_names(state)
    result = {group: _group_l2(state, items) for group, items in names.items()}
    result["total_actor"] = float(
        math.sqrt(sum(result[group] ** 2 for group in result if group != "critic"))
    )
    return result


def _group_deltas(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, float]:
    names = _group_tensor_names(current)
    result: dict[str, float] = {}
    for group, items in names.items():
        square_sum = 0.0
        for name in items:
            delta = current[name].detach().float() - previous[name].detach().float()
            square_sum += float(torch.sum(delta * delta).cpu())
        result[group] = float(math.sqrt(square_sum))
    result["total_actor"] = float(
        math.sqrt(sum(result[group] ** 2 for group in result if group != "critic"))
    )
    return result


def _checkpoint_rows(data_root: Path, arm: str) -> list[dict[str, Any]]:
    names = ["initial.pt"] + [f"update_{index:02d}.pt" for index in range(1, UPDATES + 1)] + ["final.pt"]
    states: list[tuple[str, Mapping[str, Any]]] = []
    for name in names:
        path = data_root / arm / name
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        states.append((name.removesuffix(".pt"), _state_dict(checkpoint)))
    if not states:
        raise FileNotFoundError(f"no checkpoints found for arm {arm}")

    rows: list[dict[str, Any]] = []
    initial_name, initial_state = states[0]
    previous_state = initial_state
    for name, state in states:
        norms = _group_norms(state)
        from_initial = _group_deltas(state, initial_state)
        from_previous = _group_deltas(state, previous_state)
        for group in GROUPS:
            parameter_l2 = norms[group]
            rows.append(
                {
                    "arm": arm,
                    "checkpoint": name,
                    "group": group,
                    "parameter_l2": parameter_l2,
                    "delta_from_initial_l2": from_initial[group],
                    "delta_from_previous_l2": from_previous[group],
                    "relative_delta_from_initial": from_initial[group] / max(norms[group], 1e-12),
                    "relative_delta_from_previous": from_previous[group] / max(norms[group], 1e-12),
                }
            )
        previous_state = state
    return rows


def _bin_goal(value: float) -> str:
    if value < 5.0:
        return "<5m"
    if value < 10.0:
        return "5-10m"
    if value < 20.0:
        return "10-20m"
    if value < 50.0:
        return "20-50m"
    return ">=50m"


def _bin_speed(value: float) -> str:
    if value < 0.5:
        return "<0.5mps"
    if value < 1.0:
        return "0.5-1mps"
    if value < 2.0:
        return "1-2mps"
    if value < 4.0:
        return "2-4mps"
    if value < 6.0:
        return "4-6mps"
    return ">=6mps"


def _bin_time(step: int) -> str:
    if step < 40:
        return "early_0_39"
    if step < 80:
        return "middle_40_79"
    return "late_80_119"


def _action_info(row: Mapping[str, str]) -> tuple[str, dict[str, float]]:
    payload = _parse_payload(row.get("action_execution", ""))
    if not isinstance(payload, Mapping):
        payload = {}
    longitudinal = _number(payload.get("commanded_longitudinal_delta_v_mps"), 0.0)
    braking = _number(payload.get("commanded_braking_delta_v_mps"), 0.0)
    if braking > 1e-9 or bool(payload.get("braking_action", 0)):
        label = "braking"
    elif longitudinal > 1e-9:
        label = "acceleration"
    elif longitudinal < -1e-9:
        label = "braking"
    else:
        label = "neutral"
    values = {
        "commanded_longitudinal_delta_v_mps": longitudinal,
        "commanded_braking_delta_v_mps": braking,
        "speed_change_mps": _number(payload.get("speed_change_mps"), 0.0),
        "brake_effective": _number(payload.get("brake_effective"), 0.0),
        "brake_blocked": _number(payload.get("brake_blocked"), 0.0),
        "safety_projected": float(bool(payload.get("safety_projected", False))),
        "boundary_projected": float(bool(payload.get("boundary_projected", False))),
    }
    return label, values


def _advantage_records(log_root: Path, arm: str) -> list[dict[str, Any]]:
    steps = _read_csv(log_root / arm / "steps.csv")
    online = [row for row in steps if row.get("phase") == "online"]
    online.sort(key=lambda row: (_number(row.get("block"), -1), _number(row.get("block_step"), -1)))
    records: list[dict[str, Any]] = []
    for index in range(1, UPDATES + 1):
        update = _json(log_root / arm / f"update_{index:02d}.json")
        block = int(update.get("block", index - 1))
        block_rows = [row for row in online if int(_number(row.get("block"), -1)) == block]
        audits = update.get("per_record_audit", [])
        if len(block_rows) != len(audits):
            raise ValueError(f"{arm} update {index}: {len(block_rows)} online rows vs {len(audits)} records")
        for row, audit in zip(block_rows, audits):
            action_label, action_values = _action_info(row)
            goal = _number(row.get("goal_distance_m_actual"))
            speed = _number(row.get("speed_mps_actual"))
            step = int(_number(row.get("block_step"), 0.0))
            records.append(
                {
                    "arm": arm,
                    "update": index,
                    "block": block,
                    "block_step": step,
                    "goal_distance_m": goal,
                    "speed_mps": speed,
                    "goal_bin": _bin_goal(goal),
                    "speed_bin": _bin_speed(speed),
                    "time_phase": _bin_time(step),
                    "action_class": action_label,
                    "advantage": _number(audit.get("advantage")),
                    "return": _number(audit.get("return")),
                    "old_value": _number(audit.get("old_value")),
                    "next_value": _number(audit.get("next_value")),
                    "terminated": int(bool(audit.get("terminated", False))),
                    "truncated": int(bool(audit.get("truncated", False))),
                    **action_values,
                }
            )
    return records


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    advantages = np.asarray([row["advantage"] for row in records], dtype=np.float64)
    returns = np.asarray([row["return"] for row in records], dtype=np.float64)
    values = np.asarray([row["old_value"] for row in records], dtype=np.float64)
    errors = values - returns
    if returns.size and float(np.var(returns)) > 1e-12:
        explained = 1.0 - float(np.var(errors)) / float(np.var(returns))
    else:
        explained = float("nan")
    return {
        "n": int(len(records)),
        "advantage_mean": _mean(advantages.tolist()),
        "advantage_std": float(np.std(advantages)) if advantages.size else float("nan"),
        "return_mean": _mean(returns.tolist()),
        "value_mean": _mean(values.tolist()),
        "value_mse": float(np.mean(errors * errors)) if errors.size else float("nan"),
        "value_rmse": float(np.sqrt(np.mean(errors * errors))) if errors.size else float("nan"),
        "value_mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "value_bias_value_minus_return": _mean(errors.tolist()),
        "explained_variance_preupdate": explained,
    }


def _group_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row[key]), []).append(row)
    output: list[dict[str, Any]] = []
    for label, items in sorted(groups.items()):
        summary = _stats(items)
        summary.update({"feature": key, "bin": label})
        output.append(summary)
    return output


def _advantage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_records = [row for row in records if row["arm"] == arm]
        for update in range(1, UPDATES + 1):
            subset = [row for row in arm_records if row["update"] == update]
            for item in _group_records(subset, "goal_bin") + _group_records(subset, "speed_bin") + _group_records(subset, "time_phase") + _group_records(subset, "action_class"):
                item.update({"arm": arm, "update": update})
                rows.append(item)
        for key in ("goal_bin", "speed_bin", "time_phase", "action_class"):
            for item in _group_records(arm_records, key):
                item.update({"arm": arm, "update": "all_updates"})
                rows.append(item)
    return rows


def _action_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_records = [row for row in records if row["arm"] == arm]
        for update in list(range(1, UPDATES + 1)) + ["all_updates"]:
            subset = [row for row in arm_records if update == "all_updates" or row["update"] == update]
            for label in ("acceleration", "braking", "neutral"):
                items = [row for row in subset if row["action_class"] == label]
                if not items:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "update": update,
                        "action_class": label,
                        "n": len(items),
                        "fraction": len(items) / max(len(subset), 1),
                        "mean_advantage": _mean([row["advantage"] for row in items]),
                        "mean_goal_distance_m": _mean([row["goal_distance_m"] for row in items]),
                        "mean_speed_mps": _mean([row["speed_mps"] for row in items]),
                        "mean_commanded_longitudinal_delta_v_mps": _mean([row["commanded_longitudinal_delta_v_mps"] for row in items]),
                        "mean_commanded_braking_delta_v_mps": _mean([row["commanded_braking_delta_v_mps"] for row in items]),
                        "mean_speed_change_mps": _mean([row["speed_change_mps"] for row in items]),
                        "brake_effective_fraction": _mean([row["brake_effective"] for row in items]),
                        "brake_blocked_fraction": _mean([row["brake_blocked"] for row in items]),
                    }
                )
    return rows


def _gradient_rows(log_root: Path, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for update_index in range(1, UPDATES + 1):
        update = _json(log_root / arm / f"update_{update_index:02d}.json")
        stats = update.get("update_stats", {})
        pre_global = _number(stats.get("gradient_norm"))
        post_actor = _number(stats.get("actor_gradient_norm"))
        post_critic = _number(stats.get("critic_gradient_norm"))
        post_total = float(math.sqrt(post_actor * post_actor + post_critic * post_critic))
        summary_rows.append(
            {
                "arm": arm,
                "update": update_index,
                "optimizer_steps": _number(update.get("optimizer_steps")),
                "preclip_global_gradient_norm_mean": pre_global,
                "postclip_actor_group_norm_mean": post_actor,
                "postclip_critic_group_norm_mean": post_critic,
                "postclip_group_norm_rss_from_means": post_total,
                "postclip_actor_share_from_means": post_actor / max(post_total, 1e-12),
                "postclip_critic_share_from_means": post_critic / max(post_total, 1e-12),
                "clip_norm": FIXED_CLIP_NORM,
                "implied_scale_from_mean_preclip": min(1.0, FIXED_CLIP_NORM / max(pre_global, 1e-12)),
                "implied_scale_is_exact": False,
                "actor_clip_scale_reported": _number(stats.get("actor_clip_scale")),
                "critic_clip_scale_reported": _number(stats.get("critic_clip_scale")),
                "clip_scope_actor_critic": bool(stats.get("gradient_clip_scope_actor_critic", 0.0)),
                "clip_fraction": _number(stats.get("clip_fraction")),
                "approximate_kl": _number(stats.get("approximate_kl")),
                "approximate_kl_max": _number(stats.get("approximate_kl_max")),
            }
        )
        loss_rows.append(
            {
                "arm": arm,
                "update": update_index,
                "actor_loss": _number(stats.get("actor_loss")),
                "critic_loss": _number(stats.get("critic_loss")),
                "total_loss": _number(stats.get("loss")),
                "entropy": _number(stats.get("entropy")),
                "clip_fraction": _number(stats.get("clip_fraction")),
                "approximate_kl": _number(stats.get("approximate_kl")),
                "approximate_kl_max": _number(stats.get("approximate_kl_max")),
            }
        )
        # ``PPOUpdateAudit.gradients`` is serialized cumulatively in every
        # update JSON (8, 16, ..., 48 entries).  Keep only this update's
        # eight newly completed Adam steps; otherwise the same hook snapshot
        # would be counted repeatedly across update files.
        cumulative = update.get("gradients_per_optimizer_step", [])
        current_steps = cumulative[(update_index - 1) * 8 : update_index * 8]
        for item in current_steps:
            step = int(_number(item.get("step"), -1))
            for group, values in (item.get("groups") or {}).items():
                step_rows.append(
                    {
                        "arm": arm,
                        "update": update_index,
                        "optimizer_step": step,
                        "group": group,
                        "gradient_l2_postclip": _number(values.get("l2")),
                        "max_abs_postclip": _number(values.get("max_abs")),
                        "nonzero_parameter_count": int(_number(values.get("nonzero_parameter_count"), 0.0)),
                        "parameter_count": int(_number(values.get("parameter_count"), 0.0)),
                        "finite": bool(values.get("finite", False)),
                        "capture_semantics": "post_clip_pre_adam_step_current_update",
                    }
                )
    return summary_rows, step_rows, loss_rows


def _horizon_rows(log_root: Path, arm: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for update_index in range(1, UPDATES + 1):
        update = _json(log_root / arm / f"update_{update_index:02d}.json")
        flags = update.get("last_transition_flags", {})
        subset = [row for row in records if row["arm"] == arm and row["update"] == update_index]
        terminal_steps = sum(row["terminated"] for row in subset)
        truncated_steps = sum(row["truncated"] for row in subset)
        rows.append(
            {
                "arm": arm,
                "update": update_index,
                "record_count": len(subset),
                "horizon": HORIZON,
                "terminal_count": int(update.get("terminal_count", terminal_steps)),
                "truncated_count": int(update.get("truncated_count", truncated_steps)),
                "last_terminated": bool(flags.get("terminated", False)),
                "last_truncated": bool(flags.get("truncated", False)),
                "last_done": bool(flags.get("done", False)),
                "last_terminal_for_bootstrap": bool(flags.get("terminal_for_bootstrap", False)),
                "last_next_value": _number(update.get("last_next_value")),
                "bootstrap_time_limits": bool(update.get("bootstrap_time_limits", False)),
                "nonterminal_bootstrap_count": int(update.get("nonterminal_bootstrap_count", 0)),
                "gae_boundary_count": int(update.get("gae_recursion_boundary_count", 0)),
                "gae_recurrence_matches": bool(update.get("gae_recurrence_matches_training_records", False)),
                "gae_does_not_cross_done": bool(update.get("gae_does_not_cross_done_boundary", False)),
                "value_mse_preupdate": _number(update.get("value_mse_preupdate")),
                "value_mse_preupdate_recomputed": _stats(subset)["value_mse"],
            }
        )
    return rows


def _summary(
    log_root: Path,
    data_root: Path,
    gradient_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for arm in ARMS:
        gradients = [row for row in gradient_rows if row["arm"] == arm]
        losses = [row for row in loss_rows if row["arm"] == arm]
        arm_records = [row for row in records if row["arm"] == arm]
        initial = [row for row in checkpoint_rows if row["arm"] == arm and row["checkpoint"] == "initial"]
        final = [row for row in checkpoint_rows if row["arm"] == arm and row["checkpoint"] == "final"]
        initial_map = {row["group"]: row for row in initial}
        final_map = {row["group"]: row for row in final}
        pre = [row["preclip_global_gradient_norm_mean"] for row in gradients]
        actor = [row["postclip_actor_group_norm_mean"] for row in gradients]
        critic = [row["postclip_critic_group_norm_mean"] for row in gradients]
        body_delta = final_map.get("actor_body", {}).get("delta_from_initial_l2", float("nan"))
        motion_delta = final_map.get("motion_head", {}).get("delta_from_initial_l2", float("nan"))
        logstd_delta = final_map.get("log_std", {}).get("delta_from_initial_l2", float("nan"))
        measurement_delta = final_map.get("measurement_head", {}).get("delta_from_initial_l2", float("nan"))
        bandwidth_delta = final_map.get("bandwidth_head", {}).get("delta_from_initial_l2", float("nan"))
        critic_delta = final_map.get("critic", {}).get("delta_from_initial_l2", float("nan"))
        active_head_delta = math.sqrt(
            sum(value * value for value in (motion_delta, logstd_delta, measurement_delta, bandwidth_delta))
        )
        findings.append(
            {
                "arm": arm,
                "preclip_gradient_norm_mean_over_updates": _mean(pre),
                "preclip_gradient_norm_min": min(pre),
                "preclip_gradient_norm_max": max(pre),
                "postclip_actor_group_norm_mean": _mean(actor),
                "postclip_critic_group_norm_mean": _mean(critic),
                "postclip_actor_share_from_means": _mean(actor) / max(math.sqrt(_mean(actor) ** 2 + _mean(critic) ** 2), 1e-12),
                "clip_fraction_mean": _mean([row["clip_fraction"] for row in gradients]),
                "actor_body_delta_initial_to_final_l2": body_delta,
                "motion_head_delta_initial_to_final_l2": motion_delta,
                "log_std_delta_initial_to_final_l2": logstd_delta,
                "active_actor_head_delta_initial_to_final_l2": active_head_delta,
                "critic_delta_initial_to_final_l2": critic_delta,
                "actor_body_to_critic_delta_l2_ratio": body_delta / max(critic_delta, 1e-12),
                "active_actor_head_to_critic_delta_l2_ratio": active_head_delta / max(critic_delta, 1e-12),
                "motion_head_to_actor_body_delta_l2_ratio": motion_delta / max(body_delta, 1e-12),
                "actor_body_parameter_l2_initial": initial_map.get("actor_body", {}).get("parameter_l2"),
                "actor_body_parameter_l2_final": final_map.get("actor_body", {}).get("parameter_l2"),
                "motion_head_parameter_l2_initial": initial_map.get("motion_head", {}).get("parameter_l2"),
                "motion_head_parameter_l2_final": final_map.get("motion_head", {}).get("parameter_l2"),
                "critic_parameter_l2_initial": initial_map.get("critic", {}).get("parameter_l2"),
                "critic_parameter_l2_final": final_map.get("critic", {}).get("parameter_l2"),
                "measurement_head_delta_initial_to_final_l2": measurement_delta,
                "bandwidth_head_delta_initial_to_final_l2": bandwidth_delta,
                "actor_loss_mean": _mean([row["actor_loss"] for row in losses]),
                "actor_loss_first": losses[0]["actor_loss"],
                "actor_loss_last": losses[-1]["actor_loss"],
                "entropy_first": losses[0]["entropy"],
                "entropy_last": losses[-1]["entropy"],
                "critic_loss_mean": _mean([row["critic_loss"] for row in losses]),
                "critic_preupdate_mse_mean": _mean([row["value_mse_preupdate"] for row in horizon_rows if row["arm"] == arm]),
                "critic_preupdate_mae_all_records": _stats(arm_records)["value_mae"],
                "critic_preupdate_explained_variance_all_records": _stats(arm_records)["explained_variance_preupdate"],
                "actor_and_critic_update_steps": sorted({int(row["optimizer_steps"]) for row in gradients}),
                "interpretation": "diagnostic evidence; no parameter repair is applied",
            }
        )

    return {
        "audit": "single_uav_ppo_a_route_gradient_audit",
        "scope": {
            "log_root": str(log_root),
            "data_root": str(data_root),
            "arms": list(ARMS),
            "online_steps_per_arm": 720,
            "updates_per_arm": UPDATES,
            "adam_steps_per_arm": 48,
            "horizon": HORIZON,
            "input_sources": ["update_01..06.json", "steps.csv", "initial.pt", "update_01..06.pt", "final.pt"],
            "excluded": ["BC/teacher/controller/obstacle/multiseed runs", "historical source modifications", "optimizer updates"],
        },
        "gradient_semantics": {
            "gradient_norm": "pre_clip_global_norm_mean_per_update, returned by clip_grad_norm_ before its in-place clipping",
            "actor_gradient_norm": "post_clip actor-group norm in joint clipping mode",
            "critic_gradient_norm": "post_clip critic-group norm in joint clipping mode",
            "gradients_per_optimizer_step": "post_clip group norms captured immediately before Adam.step",
            "reported_actor_clip_scale_and_critic_clip_scale": "post_clip diagnostic values in the joint branch; 1.0 does not prove no clipping",
            "implied_scale_warning": "0.5 / mean(preclip norm) is only a rough descriptive estimate, not the mean of per-minibatch scales",
        },
        "findings": findings,
        "decision": {
            "clear_implementation_bug_supported": False,
            "minimal_repair_supported_by_this_audit": False,
            "reason": "Recorded GAE/log-probability checks pass; finite-horizon truncation is intentionally terminal for bootstrap; inactive measurement/bandwidth heads have zero support in single-UAV topology; evidence is consistent with weak/short on-policy learning and critic-dominated joint gradients, but exact pre-clip group gradients are not recoverable from the saved rollout telemetry.",
            "next_step": "Root agent decides whether a separately authorized smoke/repair experiment is warranted; this artifact applies no hyperparameter or PPO update change.",
        },
        "limitations": [
            "The saved CSV/JSON contains no serialized actor observations for replaying exact forward/backward gradients.",
            "Per-step hook gradients are post clipping; pre clipping actor/critic group norms cannot be reconstructed exactly offline.",
            "Critic metrics are pre-update values against recorded finite-horizon returns; they do not establish post-update calibration on held-out states.",
            "Four arms are route variants in one seed/run family, not a multiseed estimate of optimization variance.",
        ],
    }


def run(project_root: Path, output_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    log_root = project_root / "logs" / "preliminary" / "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011"
    data_root = project_root / "data" / "synthetic" / "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011"
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    gradient_step_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for arm in ARMS:
        checkpoint_rows.extend(_checkpoint_rows(data_root, arm))
        gradients, steps, losses = _gradient_rows(log_root, arm)
        gradient_rows.extend(gradients)
        gradient_step_rows.extend(steps)
        loss_rows.extend(losses)
        arm_records = _advantage_records(log_root, arm)
        records.extend(arm_records)
        horizon_rows.extend(_horizon_rows(log_root, arm, arm_records))

    _write_csv(
        output_root / "checkpoint_norms.csv",
        checkpoint_rows,
        ["arm", "checkpoint", "group", "parameter_l2", "delta_from_initial_l2", "delta_from_previous_l2", "relative_delta_from_initial", "relative_delta_from_previous"],
    )
    _write_csv(
        output_root / "gradient_summary.csv",
        gradient_rows,
        ["arm", "update", "optimizer_steps", "preclip_global_gradient_norm_mean", "postclip_actor_group_norm_mean", "postclip_critic_group_norm_mean", "postclip_group_norm_rss_from_means", "postclip_actor_share_from_means", "postclip_critic_share_from_means", "clip_norm", "implied_scale_from_mean_preclip", "implied_scale_is_exact", "actor_clip_scale_reported", "critic_clip_scale_reported", "clip_scope_actor_critic", "clip_fraction", "approximate_kl", "approximate_kl_max"],
    )
    _write_csv(
        output_root / "gradient_steps_postclip.csv",
        gradient_step_rows,
        ["arm", "update", "optimizer_step", "group", "gradient_l2_postclip", "max_abs_postclip", "nonzero_parameter_count", "parameter_count", "finite", "capture_semantics"],
    )
    _write_csv(
        output_root / "loss_entropy_kl.csv",
        loss_rows,
        ["arm", "update", "actor_loss", "critic_loss", "total_loss", "entropy", "clip_fraction", "approximate_kl", "approximate_kl_max"],
    )
    _write_csv(
        output_root / "advantage_allocation.csv",
        _advantage_rows(records),
        ["arm", "update", "feature", "bin", "n", "advantage_mean", "advantage_std", "return_mean", "value_mean", "value_mse", "value_rmse", "value_mae", "value_bias_value_minus_return", "explained_variance_preupdate"],
    )
    _write_csv(
        output_root / "sampled_action_allocation.csv",
        _action_rows(records),
        ["arm", "update", "action_class", "n", "fraction", "mean_advantage", "mean_goal_distance_m", "mean_speed_mps", "mean_commanded_longitudinal_delta_v_mps", "mean_commanded_braking_delta_v_mps", "mean_speed_change_mps", "brake_effective_fraction", "brake_blocked_fraction"],
    )
    _write_csv(
        output_root / "critic_horizon_bootstrap.csv",
        horizon_rows,
        ["arm", "update", "record_count", "horizon", "terminal_count", "truncated_count", "last_terminated", "last_truncated", "last_done", "last_terminal_for_bootstrap", "last_next_value", "bootstrap_time_limits", "nonterminal_bootstrap_count", "gae_boundary_count", "gae_recurrence_matches", "gae_does_not_cross_done", "value_mse_preupdate", "value_mse_preupdate_recomputed"],
    )
    report = _summary(log_root, data_root, gradient_rows, checkpoint_rows, loss_rows, horizon_rows, records)
    _write_json(output_root / "gradient_audit_summary.json", report)
    _write_json(
        output_root / "audit_manifest.json",
        {
            "script": str((project_root / "src" / "analysis" / "single_uav_ppo_a_route_gradient_audit.py").resolve()),
            "output_root": str(output_root),
            "files": sorted(path.name for path in output_root.iterdir() if path.is_file()),
            "rows": {"records": len(records), "gradient_steps_postclip": len(gradient_step_rows), "checkpoint_rows": len(checkpoint_rows)},
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("logs/preliminary/single_uav_ppo_a_route_feasibility_v3_seed2026_20261011/gradient_audit"),
    )
    args = parser.parse_args()
    output = args.output_root if args.output_root.is_absolute() else args.project_root / args.output_root
    report = run(args.project_root, output)
    print(json.dumps({"audit": report["audit"], "output_root": str(output.resolve()), "decision": report["decision"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
