"""Pure contracts for post-step, actual stable-goal validation.

The simulator reports goal-distance and speed in the learning-reward
component map.  The stable flags/counter are reported at the top level of
``next_info``.  Keeping this small extraction/aggregation module free of
numpy, torch, environments, and policy imports makes the schema testable
without accidentally running an experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


STABLE_CONTRACT_VERSION = "single_uav_actual_stable_validation_v2"
STABLE_LOG_SCHEMA_VERSION = "goal_recovery_actual_stable_v1"

# Names intentionally describe the post-step values rather than the pre-step
# values logged by the correction evaluator.
ACTUAL_STABLE_FIELDS = (
    "actual_goal_distance_m",
    "actual_speed_mps",
    "goal_threshold_m",
    "speed_threshold_mps",
    "goal_in_radius",
    "goal_speed_ok",
    "goal_condition",
    "goal_stable",
    "goal_reached",
    "stable_steps",
    "stable_hold_steps",
    # Explicit names used by the stable-validation summary schema.
    "actual_stable_goal_distance_m",
    "actual_stable_speed_mps",
    "actual_stable_goal_threshold_m",
    "actual_stable_speed_threshold_mps",
    "actual_stable_goal_in_radius",
    "actual_stable_goal_speed_ok",
    "actual_stable_goal_condition",
    "actual_stable_goal_stable",
    "actual_stable_goal_reached",
    "actual_stable_steps",
    "actual_stable_hold_steps",
)

EPISODE_AGGREGATE_FIELDS = (
    "episodes",
    "entered_1m_episodes",
    "stable_goal_episodes",
    "entered_1m_and_stable_goal_episodes",
    "entered_1m_not_stable_episodes",
    "stable_goal_rate",
    "entered_1m_rate",
    "actual_stable_validation",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number(mapping: Mapping[str, Any], key: str, *, integer: bool = False) -> int | float:
    if key not in mapping:
        raise KeyError(key)
    value = mapping[key]
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{key} must be finite")
    if integer:
        if int(result) != result:
            raise ValueError(f"{key} must be an integer")
        return int(result)
    return result


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    if key not in mapping:
        raise KeyError(key)
    value = mapping[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"0", "1", "false", "true"}:
        return value.strip().lower() in {"1", "true"}
    raise ValueError(f"{key} must be boolean")


def extract_next_info_stable_contract(next_info: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact post-step stable contract from environment info.

    ``current_goal_distance`` and ``single_uav_speed_mps`` are deliberately
    read only from ``next_info['learning_reward_components']``.  Thresholds
    are read from that same map.  Stable booleans and counters are deliberately
    read from top-level ``next_info``; using fallback paths would make a stale
    or pre-step value look like an actual post-step observation.
    """

    info = _mapping(next_info, "next_info")
    components = _mapping(info["learning_reward_components"], "learning_reward_components")
    distance = float(_number(components, "current_goal_distance"))
    speed = float(_number(components, "single_uav_speed_mps"))
    distance_threshold = float(_number(components, "single_uav_goal_threshold_m"))
    speed_threshold = float(_number(components, "single_uav_goal_speed_threshold_mps"))
    in_radius = _boolean(info, "single_uav_goal_in_radius")
    speed_ok = _boolean(info, "single_uav_goal_speed_ok")
    condition = _boolean(info, "single_uav_goal_condition")
    stable = _boolean(info, "single_uav_goal_stable")
    reached = _boolean(info, "single_uav_goal_reached")
    stable_steps = int(_number(info, "single_uav_stable_steps", integer=True))
    hold_steps = int(_number(info, "single_uav_stable_hold_steps", integer=True))
    result = {
        "actual_goal_distance_m": distance,
        "actual_speed_mps": speed,
        "goal_threshold_m": distance_threshold,
        "speed_threshold_mps": speed_threshold,
        "goal_in_radius": in_radius,
        "goal_speed_ok": speed_ok,
        "goal_condition": condition,
        "goal_stable": stable,
        "goal_reached": reached,
        "stable_steps": stable_steps,
        "stable_hold_steps": hold_steps,
        "actual_stable_goal_distance_m": distance,
        "actual_stable_speed_mps": speed,
        "actual_stable_goal_threshold_m": distance_threshold,
        "actual_stable_speed_threshold_mps": speed_threshold,
        "actual_stable_goal_in_radius": in_radius,
        "actual_stable_goal_speed_ok": speed_ok,
        "actual_stable_goal_condition": condition,
        "actual_stable_goal_stable": stable,
        "actual_stable_goal_reached": reached,
        "actual_stable_steps": stable_steps,
        "actual_stable_hold_steps": hold_steps,
    }
    return result


# Short aliases are kept for callers/tests that refer to the operation rather
# than the wire-level name.
extract_actual_stable_fields = extract_next_info_stable_contract
extract_stable_fields = extract_next_info_stable_contract


def actual_stable_episode_fields(
    *,
    next_info: Mapping[str, Any],
    entered_1m: bool | int,
) -> dict[str, Any]:
    """Return episode-row stable fields with an explicit actual prefix."""

    result = extract_next_info_stable_contract(next_info)
    result["actual_stable_entered_1m"] = bool(entered_1m)
    return result


def aggregate_actual_stable_episodes(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the exact stable-validation episode counters and rates."""

    total = len(summaries)
    entered = sum(bool(row.get("entered_1m", row.get("actual_stable_entered_1m", False))) for row in summaries)
    stable = sum(bool(row.get("stable_goal", row.get("actual_stable_goal_stable", False))) for row in summaries)
    entered_stable = sum(
        bool(row.get("entered_1m", row.get("actual_stable_entered_1m", False)))
        and bool(row.get("stable_goal", row.get("actual_stable_goal_stable", False)))
        for row in summaries
    )
    output: dict[str, Any] = {
        "episodes": total,
        "entered_1m_episodes": entered,
        "stable_goal_episodes": stable,
        "entered_1m_and_stable_goal_episodes": entered_stable,
        "entered_1m_not_stable_episodes": entered - entered_stable,
        "stable_goal_rate": stable / total if total else 0.0,
        "entered_1m_rate": entered / total if total else 0.0,
        "actual_stable_validation": True,
    }
    return output


aggregate_stable_episodes = aggregate_actual_stable_episodes


def sha256_file(path: str | Path) -> str:
    """Return a file hash without importing experiment/runtime code."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Hash JSON-compatible values deterministically for manifest contracts."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_canonical_stable_fields(next_info: Mapping[str, Any]) -> dict[str, Any]:
    """Copy, without recomputation, the exact fields used by stable_goal."""
    info = _mapping(next_info, "next_info")
    components = _mapping(info.get("learning_reward_components"), "learning_reward_components")
    distance = float(_number(components, "current_goal_distance"))
    speed = float(_number(components, "single_uav_speed_mps"))
    distance_threshold = float(_number(components, "single_uav_goal_threshold_m"))
    speed_threshold = float(_number(components, "single_uav_goal_speed_threshold_mps"))
    hold = int(_number(components, "single_uav_stable_hold_steps", integer=True))
    distance_ok = _boolean(info, "single_uav_goal_in_radius")
    speed_ok = _boolean(info, "single_uav_goal_speed_ok")
    candidate = _boolean(info, "single_uav_goal_condition")
    counter = int(_number(info, "single_uav_stable_steps", integer=True))
    stable = _boolean(info, "single_uav_goal_stable")
    if candidate != (distance_ok and speed_ok):
        raise ValueError("canonical stable condition flags are inconsistent")
    if not candidate and counter != 0:
        raise ValueError("stable counter must reset when candidate is false")
    if hold < 1 or counter < 0 or (stable and (not candidate or counter < hold)):
        raise ValueError("canonical stable counter/flag is inconsistent")
    return {
        "post_step_actual_speed_mps": speed,
        "goal_distance_for_stable_m": distance,
        "goal_speed_for_stable_mps": speed,
        "stable_goal_distance_threshold_m": distance_threshold,
        "stable_speed_threshold_mps": speed_threshold,
        "stable_hold_steps_required": hold,
        "goal_distance_ok": int(distance_ok),
        "goal_speed_ok": int(speed_ok),
        "stable_candidate_step": int(candidate),
        "stable_hold_counter": counter,
        "stable_goal_now": int(stable),
        "goal_distance_source": "post_step_estimated_state",
        "goal_speed_source": "simulator_actual_state",
        "stable_criterion_timing": "post_environment_step_before_termination_return",
        "stable_log_schema_version": STABLE_LOG_SCHEMA_VERSION,
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _first_step(rows: Sequence[Mapping[str, Any]], field: str) -> int | str:
    return next((int(row["step"]) for row in rows if int(row.get(field, 0) or 0)), "")


def aggregate_episode_stable_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inside = [row for row in rows if int(row.get("goal_distance_ok", 0) or 0)]
    entry = inside[0] if inside else None
    speeds = [float(row["post_step_actual_speed_mps"]) for row in inside]
    tangential = [abs(float(row["post_step_actual_tangential_velocity_mps"])) for row in inside]
    return {
        "first_entered_1m_step": int(entry["step"]) if entry else "",
        "first_goal_distance_ok_step": _first_step(rows, "goal_distance_ok"),
        "first_goal_speed_ok_step": _first_step(rows, "goal_speed_ok"),
        "first_stable_candidate_step": _first_step(rows, "stable_candidate_step"),
        "max_stable_hold_counter": max((int(row["stable_hold_counter"]) for row in rows), default=0),
        "stable_goal_step": _first_step(rows, "stable_goal_now"),
        "actual_speed_at_first_entered_1m_mps": float(entry["post_step_actual_speed_mps"]) if entry else float("nan"),
        "radial_velocity_at_first_entered_1m_mps": float(entry["post_step_actual_radial_velocity_mps"]) if entry else float("nan"),
        "tangential_velocity_at_first_entered_1m_mps": float(entry["post_step_actual_tangential_velocity_mps"]) if entry else float("nan"),
        "mean_actual_speed_inside_1m_mps": _mean(speeds),
        "min_actual_speed_inside_1m_mps": min(speeds, default=float("nan")),
        "max_actual_speed_inside_1m_mps": max(speeds, default=float("nan")),
        "fraction_inside_1m_speed_ok": sum(int(row["goal_speed_ok"]) for row in inside) / len(inside) if inside else float("nan"),
        "mean_tangential_velocity_inside_1m_mps": _mean(tangential),
        "correction_active_fraction_inside_1m": sum(int(row["goal_recovery_correction_active"]) for row in inside) / len(inside) if inside else float("nan"),
        "mean_correction_norm_inside_1m_mps": _mean([float(row["goal_recovery_correction_norm_mps"]) for row in inside]),
        "entered_1m": int(bool(inside)),
        "stable_goal": int(any(int(row.get("stable_goal_now", 0) or 0) for row in rows)),
    }


def aggregate_actual_stable_validation(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entered = [row for row in episodes if int(row.get("entered_1m", 0) or 0)]
    entry = [float(row["actual_speed_at_first_entered_1m_mps"]) for row in entered]
    inside = [float(row["mean_actual_speed_inside_1m_mps"]) for row in entered if math.isfinite(float(row["mean_actual_speed_inside_1m_mps"]))]
    threshold = float(episodes[0]["stable_speed_threshold_mps"]) if episodes else float("nan")
    counters = [int(row.get("max_stable_hold_counter", 0) or 0) for row in episodes]
    distribution = {f"counter_{index}": sum(value == index for value in counters) for index in range(5)}
    distribution["counter_5_plus"] = sum(value >= 5 for value in counters)
    finite_mean = lambda field: _mean([float(row[field]) for row in entered if math.isfinite(float(row[field]))])
    return {
        "entered_1m_episodes": len(entered),
        "stable_goal_episodes": sum(int(row.get("stable_goal", 0) or 0) for row in episodes),
        "entry_actual_speed_mean": _mean(entry), "entry_actual_speed_min": min(entry, default=float("nan")),
        "entry_actual_speed_max": max(entry, default=float("nan")),
        "inside_1m_actual_speed_mean": _mean(inside), "inside_1m_actual_speed_min": min(inside, default=float("nan")),
        "inside_1m_actual_speed_max": max(inside, default=float("nan")),
        "episodes_entry_speed_above_threshold": sum(value > threshold for value in entry),
        "episodes_ever_speed_ok_inside_1m": sum(float(row["fraction_inside_1m_speed_ok"]) > 0 for row in entered),
        "max_stable_hold_counter_distribution": distribution,
        "mean_tangential_velocity_inside_1m": finite_mean("mean_tangential_velocity_inside_1m_mps"),
        "mean_correction_active_fraction_inside_1m": finite_mean("correction_active_fraction_inside_1m"),
        "mean_correction_norm_inside_1m": finite_mean("mean_correction_norm_inside_1m_mps"),
    }


def load_previous_formal_summary(root: Path) -> dict[str, Any]:
    path = Path(root).resolve() / "evaluation_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "single_uav_goal_recovery_correction_evaluation" or payload.get("formal") is not True or payload.get("condition") != "paired" or int(payload.get("episode_count", 0)) != 20:
        raise ValueError("reference must be the 20-episode paired formal evaluation")
    return payload


def episode_identity(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, int, int, str]]:
    result = [(
        int(_number(row, "episode_index", integer=True)), str(row.get("episode_id", "")),
        int(_number(row, "scenario_seed", integer=True)), int(_number(row, "noise_seed", integer=True)),
        str(row.get("scenario_hash", "")),
    ) for row in rows]
    if any(not row[1] or not row[4] for row in result) or len(set(result)) != len(result):
        raise ValueError("malformed or duplicate episode identity")
    return result


def assert_previous_formal_identity(reference: Mapping[str, Any], *, checkpoint_path: Path,
                                    checkpoint_sha256: str, policy_hash: str,
                                    scenario_manifest_hash: str,
                                    current_episodes: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    checkpoint = _mapping(reference.get("checkpoint"), "checkpoint")
    results = _mapping(reference.get("results"), "results")
    on = _mapping(results.get("on"), "results.on")
    checks = {
        "same_checkpoint_absolute_path_as_previous_formal": checkpoint_path.resolve() == Path(str(checkpoint.get("resolved_path", checkpoint.get("path", "")))).resolve(),
        "same_checkpoint_sha256_as_previous_formal": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash_as_previous_formal": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest_as_previous_formal": scenario_manifest_hash == str(reference.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity_as_previous_formal": episode_identity(current_episodes) == episode_identity(on.get("episodes", [])),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError("previous formal identity mismatch: " + ", ".join(failed))
    return checks


__all__ = [
    "ACTUAL_STABLE_FIELDS",
    "EPISODE_AGGREGATE_FIELDS",
    "STABLE_CONTRACT_VERSION",
    "STABLE_LOG_SCHEMA_VERSION",
    "aggregate_actual_stable_validation",
    "aggregate_episode_stable_fields",
    "assert_previous_formal_identity",
    "actual_stable_episode_fields",
    "aggregate_actual_stable_episodes",
    "aggregate_stable_episodes",
    "canonical_json_hash",
    "extract_actual_stable_fields",
    "extract_canonical_stable_fields",
    "extract_next_info_stable_contract",
    "extract_stable_fields",
    "sha256_file",
    "episode_identity",
    "load_previous_formal_summary",
]
