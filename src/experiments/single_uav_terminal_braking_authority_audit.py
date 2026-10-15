"""Offline audit of terminal-capture braking authority from existing logs.

The module is deliberately file-only.  It does not import the simulator,
controller, policy, Teacher, or any training code, and it never performs a
rollout.  Raw input artifacts are read-only and derived artifacts are written
to a fresh output directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE = "single_uav_terminal_braking_authority_audit"
VERSION = "terminal_braking_authority_audit_v1"
DEFAULT_TERMINAL_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_validation_seed2026"
)
DEFAULT_OUTPUT_ROOT = Path(
    "logs/preliminary/single_uav_terminal_braking_authority_audit_final_seed2026"
)
DT_S = 0.2
EPSILON = 1.0e-9
ACTOR_NEUTRAL_EPSILON_MPS = 1.0e-6
MIN_EPISODE_BRAKING_SAMPLES = 3
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026

REQUIRED_STEP_FIELDS = {
    "episode_id",
    "step",
    "post_bypass_recovery",
    "terminal_capture_active",
    "radial_velocity_error_mps",
    "radial_braking_delta_v_mps",
    "terminal_tangential_damping_delta_v_mps",
    "terminal_raw_correction_delta_v_mps",
    "terminal_correction_clipped",
    "goal_recovery_correction_delta_v_mps",
    "goal_recovery_applied_delta_v_mps",
    "actor_raw_delta_v_mps",
    "corrected_delta_v_pre_transform_mps",
    "corrected_delta_v_pre_safety_mps",
    "applied_delta_v_mps",
    "safety_projection_applied",
    "boundary_projection_applied",
    "action_clipped",
    "collision",
    "goal_distance_m",
    "goal_distance_after_m",
    "current_radial_velocity_mps",
    "current_tangential_velocity_mps",
    "terminal_desired_radial_velocity_mps",
    "radial_propulsion_delta_v_mps",
    "radial_propulsion_scale",
    "terminal_blend",
    "post_step_actual_radial_velocity_mps",
    "post_step_actual_tangential_velocity_mps",
    "post_step_actual_speed_mps",
}

REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "passed_obstacle",
    "post_bypass_recovery",
    "entered_1m",
    "minimum_goal_distance_m",
}

STATUS_FLAGS: dict[str, Any] = {
    "source_control_algorithm_modified": False,
    "goal_recovery_controller_modified": False,
    "terminal_capture_controller_modified": False,
    "actor_modified": False,
    "gat_modified": False,
    "critic_modified": False,
    "action_transform_modified": False,
    "reward_modified": False,
    "dynamics_modified": False,
    "stable_goal_criterion_modified": False,
    "obstacle_logic_modified": False,
    "training_performed": False,
    "ppo_updates": 0,
    "mappo_updates": 0,
    "bc_retraining_executed": False,
    "dagger_collection_executed": False,
    "teacher_rollout_executed": False,
    "formal_evaluation_executed_by_codex": False,
    "smoke_evaluation_executed_by_codex": False,
    "rollout_executed": False,
    "offline_log_analysis_executed": True,
}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def integer(value: Any) -> int:
    number = finite_float(value)
    if number is None or int(number) != number:
        raise ValueError(f"expected integer value, got {value!r}")
    return int(number)


def parse_vector(value: Any) -> tuple[float, float, float] | None:
    """Parse one finite three-vector without evaluating file content."""
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        text = str(value).strip()
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                raw = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    values = [finite_float(item) for item in raw]
    if any(item is None for item in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def scale(vector: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(float(value) * factor for value in vector)  # type: ignore[return-value]


def recover_goal_direction(
    radial_braking_delta_v_mps: Sequence[float],
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float, float] | None:
    """Recover +goal direction from a braking vector known to point outward.

    ``GoalRecoveryCorrection.compute`` defines radial braking as
    ``gain * min(radial_error, 0) * goal_direction``.  A non-zero braking
    vector therefore points in ``-goal_direction``.
    """
    magnitude = norm(radial_braking_delta_v_mps)
    if not math.isfinite(magnitude) or magnitude <= epsilon:
        return None
    return scale(radial_braking_delta_v_mps, -1.0 / magnitude)


def braking_projection(
    delta_v_mps: Sequence[float], goal_direction: Sequence[float]
) -> float:
    """Positive magnitude of the component opposing +goal direction."""
    return max(0.0, -dot(delta_v_mps, goal_direction))


def signed_goal_projection(
    delta_v_mps: Sequence[float], goal_direction: Sequence[float]
) -> float:
    return dot(delta_v_mps, goal_direction)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    if denominator <= EPSILON:
        return None
    return numerator / denominator


def quantile(values: Iterable[float], probability: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    probability = min(1.0, max(0.0, float(probability)))
    position = (len(finite) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def descriptive_statistics(values: Iterable[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p10": None, "p25": None, "median": None, "p75": None,
            "p90": None, "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "p10": quantile(finite, 0.10),
        "p25": quantile(finite, 0.25),
        "median": statistics.median(finite),
        "p75": quantile(finite, 0.75),
        "p90": quantile(finite, 0.90),
        "max": max(finite),
    }


def robust_median_lower_bound(
    values: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> float | None:
    """Deterministic one-sided 95% bootstrap lower bound of the median."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2 or replicates <= 0:
        return None
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(finite, k=len(finite)))
        for _ in range(replicates)
    ]
    return quantile(medians, 0.05)


def _read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = list(reader.fieldnames or ())
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path.name} missing required fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    return rows, fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_from(row: Mapping[str, Any], field: str) -> tuple[float, float, float]:
    value = parse_vector(row.get(field))
    if value is None:
        raise ValueError(
            f"invalid finite three-vector in {field} for "
            f"{row.get('episode_id')} step {row.get('step')}"
        )
    return value


def classify_actor_radial(value: float) -> str:
    if value < -ACTOR_NEUTRAL_EPSILON_MPS:
        return "assist"
    if value > ACTOR_NEUTRAL_EPSILON_MPS:
        return "oppose"
    return "neutral"


def derive_braking_sample(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if integer(row.get("post_bypass_recovery")) != 1:
        return None
    if integer(row.get("terminal_capture_active")) != 1:
        return None
    radial_error = finite_float(row.get("radial_velocity_error_mps"))
    braking_vector = _vector_from(row, "radial_braking_delta_v_mps")
    if radial_error is None or not (radial_error < 0.0 or norm(braking_vector) > EPSILON):
        return None
    goal_direction = recover_goal_direction(braking_vector)
    if goal_direction is None:
        return None

    requested = braking_projection(braking_vector, goal_direction)
    terminal_raw = _vector_from(row, "terminal_raw_correction_delta_v_mps")
    post_cap_vector = _vector_from(row, "goal_recovery_correction_delta_v_mps")
    applied_correction_vector = _vector_from(row, "goal_recovery_applied_delta_v_mps")
    actor = _vector_from(row, "actor_raw_delta_v_mps")
    pre_transform = _vector_from(row, "corrected_delta_v_pre_transform_mps")
    post_transform = _vector_from(row, "corrected_delta_v_pre_safety_mps")
    final_applied = _vector_from(row, "applied_delta_v_mps")
    tangential = _vector_from(row, "terminal_tangential_damping_delta_v_mps")

    raw_radial = braking_projection(terminal_raw, goal_direction)
    post_cap = braking_projection(post_cap_vector, goal_direction)
    correction_applied = braking_projection(applied_correction_vector, goal_direction)
    pre_transform_braking = braking_projection(pre_transform, goal_direction)
    post_transform_braking = braking_projection(post_transform, goal_direction)
    final_braking = braking_projection(final_applied, goal_direction)
    actor_signed = signed_goal_projection(actor, goal_direction)
    terminal_clip_ratio = finite_float(row.get("terminal_correction_clip_ratio"))
    clean = (
        integer(row.get("safety_projection_applied")) == 0
        and integer(row.get("boundary_projection_applied")) == 0
        and integer(row.get("collision")) == 0
    )
    saturated = integer(row.get("terminal_correction_clipped")) == 1

    return {
        "episode_id": str(row.get("episode_id")),
        "episode_index": integer(row.get("episode_index", 0)),
        "step": integer(row.get("step")),
        "clean": int(clean),
        "saturated": int(saturated),
        "action_clipped": integer(row.get("action_clipped")),
        "safety_projection_applied": integer(row.get("safety_projection_applied")),
        "boundary_projection_applied": integer(row.get("boundary_projection_applied")),
        "goal_direction": list(goal_direction),
        "goal_distance_m": finite_float(row.get("goal_distance_m")),
        "goal_distance_after_m": finite_float(row.get("goal_distance_after_m")),
        "terminal_blend": finite_float(row.get("terminal_blend")),
        "radial_velocity_error_mps": radial_error,
        "current_radial_velocity_mps": finite_float(row.get("current_radial_velocity_mps")),
        "current_tangential_velocity_mps": finite_float(row.get("current_tangential_velocity_mps")),
        "terminal_desired_radial_velocity_mps": finite_float(row.get("terminal_desired_radial_velocity_mps")),
        "radial_propulsion_scale": finite_float(row.get("radial_propulsion_scale")),
        "radial_propulsion_norm_mps": norm(_vector_from(row, "radial_propulsion_delta_v_mps")),
        "requested_radial_braking_mps": requested,
        "terminal_raw_combined_radial_braking_mps": raw_radial,
        "post_correction_cap_radial_braking_mps": post_cap,
        "applied_correction_radial_braking_mps": correction_applied,
        "pre_transform_radial_braking_mps": pre_transform_braking,
        "post_transform_radial_braking_mps": post_transform_braking,
        "final_applied_radial_braking_mps": final_braking,
        "command_side_effective_radial_deceleration_mps2": final_braking / DT_S,
        "requested_tangential_damping_norm_mps": norm(tangential),
        "terminal_raw_correction_norm_mps": finite_float(row.get("terminal_raw_correction_norm_mps")),
        "terminal_correction_clip_ratio": terminal_clip_ratio,
        "radial_budget_loss_due_to_cap_mps": max(0.0, requested - post_cap),
        "tangential_budget_loss_due_to_cap_mps": (
            max(0.0, norm(tangential) * (1.0 - terminal_clip_ratio))
            if terminal_clip_ratio is not None else None
        ),
        "correction_cap_retention": safe_ratio(post_cap, requested),
        "alpha_retention": safe_ratio(correction_applied, post_cap),
        "transform_retention": safe_ratio(post_transform_braking, pre_transform_braking),
        "final_retention": safe_ratio(final_braking, requested),
        "loss_due_to_correction_cap_mps": max(0.0, requested - post_cap),
        "loss_due_to_action_transform_mps": max(0.0, pre_transform_braking - post_transform_braking),
        "loss_due_to_safety_boundary_mps": max(0.0, post_transform_braking - final_braking),
        "actor_radial_delta_v_mps": actor_signed,
        "actor_relation": classify_actor_radial(actor_signed),
        "post_step_actual_radial_velocity_mps": finite_float(row.get("post_step_actual_radial_velocity_mps")),
        "post_step_actual_tangential_velocity_mps": finite_float(row.get("post_step_actual_tangential_velocity_mps")),
        "post_step_actual_speed_mps": finite_float(row.get("post_step_actual_speed_mps")),
        "observed_radial_deceleration_crosscheck_mps2": None,
        "observed_speed_deceleration_crosscheck_mps2": None,
        "goal_axis_rotation_deg": None,
    }


def add_consecutive_crosschecks(
    derived: list[dict[str, Any]], all_rows: Sequence[Mapping[str, Any]]
) -> None:
    source: dict[tuple[str, int], Mapping[str, Any]] = {
        (str(row.get("episode_id")), integer(row.get("step"))): row for row in all_rows
    }
    directions: dict[tuple[str, int], tuple[float, float, float]] = {
        (str(item["episode_id"]), int(item["step"])): tuple(item["goal_direction"])
        for item in derived
    }
    for item in derived:
        key = (str(item["episode_id"]), int(item["step"]))
        previous_key = (key[0], key[1] - 1)
        previous = source.get(previous_key)
        if previous is None:
            continue
        previous_radial = finite_float(previous.get("post_step_actual_radial_velocity_mps"))
        previous_speed = finite_float(previous.get("post_step_actual_speed_mps"))
        current_radial = item.get("post_step_actual_radial_velocity_mps")
        current_speed = item.get("post_step_actual_speed_mps")
        if previous_radial is not None and current_radial is not None:
            item["observed_radial_deceleration_crosscheck_mps2"] = max(
                0.0, (previous_radial - float(current_radial)) / DT_S
            )
        if previous_speed is not None and current_speed is not None:
            item["observed_speed_deceleration_crosscheck_mps2"] = max(
                0.0, (previous_speed - float(current_speed)) / DT_S
            )
        previous_direction = directions.get(previous_key)
        if previous_direction is not None:
            cosine = max(-1.0, min(1.0, dot(previous_direction, item["goal_direction"])))
            item["goal_axis_rotation_deg"] = math.degrees(math.acos(cosine))


def aggregate_episode_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["episode_id"])].append(sample)
    output: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        rows = grouped[episode_id]
        clean = [row for row in rows if integer(row["clean"]) == 1]
        saturated_clean = [row for row in clean if integer(row["saturated"]) == 1]
        effective = [float(row["command_side_effective_radial_deceleration_mps2"]) for row in saturated_clean]
        all_effective = [float(row["command_side_effective_radial_deceleration_mps2"]) for row in clean]
        output.append({
            "episode_id": episode_id,
            "all_braking_samples": len(rows),
            "clean_braking_samples": len(clean),
            "saturated_clean_braking_samples": len(saturated_clean),
            "eligible_for_conservative_estimate": int(len(saturated_clean) >= MIN_EPISODE_BRAKING_SAMPLES),
            "saturated_clean_effective_deceleration_mean_mps2": statistics.fmean(effective) if effective else None,
            "saturated_clean_effective_deceleration_p10_mps2": quantile(effective, 0.10),
            "saturated_clean_effective_deceleration_median_mps2": statistics.median(effective) if effective else None,
            "clean_effective_deceleration_mean_mps2": statistics.fmean(all_effective) if all_effective else None,
            "clean_effective_deceleration_p10_mps2": quantile(all_effective, 0.10),
            "clean_effective_deceleration_median_mps2": statistics.median(all_effective) if all_effective else None,
        })
    return output


def actor_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {relation: sum(row["actor_relation"] == relation for row in samples) for relation in ("assist", "oppose", "neutral")}
    total = len(samples)
    return {
        "count": total,
        **{f"{key}_count": value for key, value in counts.items()},
        **{f"{key}_fraction": value / total if total else None for key, value in counts.items()},
        "signed_radial_delta_v_mps": descriptive_statistics(
            float(row["actor_radial_delta_v_mps"]) for row in samples
        ),
        "assist_magnitude_mps": descriptive_statistics(
            -float(row["actor_radial_delta_v_mps"])
            for row in samples if row["actor_relation"] == "assist"
        ),
        "oppose_magnitude_mps": descriptive_statistics(
            float(row["actor_radial_delta_v_mps"])
            for row in samples if row["actor_relation"] == "oppose"
        ),
    }


def _subset_statistics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "requested_radial_braking_mps",
        "terminal_raw_combined_radial_braking_mps",
        "post_correction_cap_radial_braking_mps",
        "applied_correction_radial_braking_mps",
        "pre_transform_radial_braking_mps",
        "post_transform_radial_braking_mps",
        "final_applied_radial_braking_mps",
        "command_side_effective_radial_deceleration_mps2",
        "correction_cap_retention",
        "alpha_retention",
        "transform_retention",
        "final_retention",
        "loss_due_to_correction_cap_mps",
        "loss_due_to_action_transform_mps",
        "loss_due_to_safety_boundary_mps",
        "requested_tangential_damping_norm_mps",
        "radial_budget_loss_due_to_cap_mps",
        "tangential_budget_loss_due_to_cap_mps",
        "observed_radial_deceleration_crosscheck_mps2",
        "observed_speed_deceleration_crosscheck_mps2",
        "goal_axis_rotation_deg",
    )
    return {field: descriptive_statistics(row.get(field) for row in samples) for field in fields}


def _stall_analysis(
    episode_rows: Sequence[Mapping[str, Any]],
    all_steps: Sequence[Mapping[str, Any]],
    derived_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for row in episode_rows:
        minimum = finite_float(row.get("minimum_goal_distance_m"))
        if (
            integer(row.get("passed_obstacle")) == 1
            and integer(row.get("post_bypass_recovery")) == 1
            and integer(row.get("entered_1m")) == 0
            and minimum is not None
        ):
            candidates.append((minimum, row))
    if not candidates:
        return {"found": False, "confirmed": False, "reason": "no matching episode"}
    minimum, episode = min(candidates, key=lambda item: item[0])
    episode_id = str(episode["episode_id"])
    rows = [row for row in all_steps if str(row.get("episode_id")) == episode_id]
    active = [row for row in rows if integer(row.get("terminal_capture_active")) == 1]
    if not active:
        return {"found": True, "confirmed": False, "episode_id": episode_id, "reason": "no active terminal rows"}
    closest = min(active, key=lambda row: finite_float(row.get("goal_distance_after_m")) or math.inf)
    tail = sorted(active, key=lambda row: integer(row.get("step")))[-10:]
    closest_key = (episode_id, integer(closest["step"]))
    closest_derived = derived_by_key.get(closest_key, {})
    tail_propulsion_scales = [finite_float(row.get("radial_propulsion_scale")) for row in tail]
    tail_propulsion_norms = [norm(_vector_from(row, "radial_propulsion_delta_v_mps")) for row in tail]
    scale_stats = descriptive_statistics(tail_propulsion_scales)
    propulsion_stats = descriptive_statistics(tail_propulsion_norms)
    confirmed = bool(
        1.0 < minimum <= 1.5
        and scale_stats["median"] is not None
        and float(scale_stats["median"]) <= 0.10
        and propulsion_stats["median"] is not None
        and float(propulsion_stats["median"]) <= 0.05
    )
    return {
        "found": True,
        "confirmed": confirmed,
        "confirmation_rule": "1m<minimum<=1.5m and last-10-active median propulsion scale<=0.10 and propulsion norm<=0.05m/s",
        "episode_id": episode_id,
        "minimum_goal_distance_m": minimum,
        "closest_active_step": integer(closest["step"]),
        "closest_active_state": {
            "goal_distance_m": finite_float(closest.get("goal_distance_m")),
            "goal_distance_after_m": finite_float(closest.get("goal_distance_after_m")),
            "current_radial_velocity_mps": finite_float(closest.get("current_radial_velocity_mps")),
            "current_tangential_velocity_mps": finite_float(closest.get("current_tangential_velocity_mps")),
            "terminal_blend": finite_float(closest.get("terminal_blend")),
            "terminal_desired_radial_velocity_mps": finite_float(closest.get("terminal_desired_radial_velocity_mps")),
            "radial_propulsion_scale": finite_float(closest.get("radial_propulsion_scale")),
            "radial_propulsion_norm_mps": norm(_vector_from(closest, "radial_propulsion_delta_v_mps")),
            "radial_braking_norm_mps": norm(_vector_from(closest, "radial_braking_delta_v_mps")),
            "final_applied_radial_braking_mps": closest_derived.get("final_applied_radial_braking_mps"),
            "post_step_actual_speed_mps": finite_float(closest.get("post_step_actual_speed_mps")),
        },
        "last_10_active_steps": {
            "count": len(tail),
            "radial_propulsion_scale": scale_stats,
            "radial_propulsion_norm_mps": propulsion_stats,
            "goal_distance_after_m": descriptive_statistics(finite_float(row.get("goal_distance_after_m")) for row in tail),
            "current_radial_velocity_mps": descriptive_statistics(finite_float(row.get("current_radial_velocity_mps")) for row in tail),
            "current_tangential_velocity_mps": descriptive_statistics(finite_float(row.get("current_tangential_velocity_mps")) for row in tail),
            "post_step_actual_speed_mps": descriptive_statistics(finite_float(row.get("post_step_actual_speed_mps")) for row in tail),
        },
    }


def build_summary(
    all_steps: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    samples: list[dict[str, Any]],
    episode_aggregates: list[dict[str, Any]],
    *,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    clean = [row for row in samples if integer(row["clean"]) == 1]
    saturated = [row for row in samples if integer(row["saturated"]) == 1]
    unsaturated = [row for row in samples if integer(row["saturated"]) == 0]
    saturated_clean = [row for row in clean if integer(row["saturated"]) == 1]
    clean_action_clipped = [row for row in clean if integer(row["action_clipped"]) == 1]
    eligible_episodes = [
        row for row in episode_aggregates
        if integer(row["eligible_for_conservative_estimate"]) == 1
    ]
    episode_medians = [
        float(row["saturated_clean_effective_deceleration_median_mps2"])
        for row in eligible_episodes
        if row["saturated_clean_effective_deceleration_median_mps2"] is not None
    ]
    candidate_p10 = quantile(episode_medians, 0.10)
    candidate_p25 = quantile(episode_medians, 0.25)
    candidate_bootstrap = robust_median_lower_bound(episode_medians)
    candidates = {
        "candidate_1_episode_median_p10_mps2": candidate_p10,
        "candidate_2_episode_median_p25_mps2": candidate_p25,
        "candidate_3_bootstrap_95pct_lower_bound_of_episode_median_mps2": candidate_bootstrap,
    }
    ready = bool(
        len(saturated_clean) >= 20
        and len(eligible_episodes) >= 5
        and candidate_p10 is not None
        and candidate_p10 > EPSILON
    )
    recommended_name = "candidate_1_episode_median_p10_mps2" if ready else None
    recommended = candidate_p10 if ready else None
    v_initial = 5.2
    v_final = 0.5
    stopping_distance = (
        (v_initial * v_initial - v_final * v_final) / (2.0 * recommended)
        if recommended is not None and recommended > EPSILON else None
    )
    median_episode_authority = quantile(episode_medians, 0.50)
    median_stopping_distance = (
        (v_initial * v_initial - v_final * v_final) / (2.0 * median_episode_authority)
        if median_episode_authority is not None and median_episode_authority > EPSILON else None
    )
    empirical_authority_margin = (
        max(0.0, stopping_distance - median_stopping_distance)
        if stopping_distance is not None and median_stopping_distance is not None else None
    )
    one_decision_window_travel = v_initial * DT_S
    margin = (
        empirical_authority_margin + one_decision_window_travel
        if empirical_authority_margin is not None else None
    )
    derived_by_key = {
        (str(row["episode_id"]), int(row["step"])): row for row in samples
    }
    budget_ratio = [
        safe_ratio(
            float(row["requested_tangential_damping_norm_mps"]),
            float(row["requested_radial_braking_mps"]),
        )
        for row in saturated_clean
    ]
    summary = {
        "stage": STAGE,
        "version": VERSION,
        "input_provenance": dict(input_provenance),
        "data_dictionary": {
            "observational_unit": "one logged control decision window",
            "cluster_unit": "episode_id",
            "dt_s": DT_S,
            "delta_v_units": "m/s per decision window",
            "effective_deceleration_definition": "max(0,-dot(final_applied_delta_v, EKF_goal_direction))/dt",
            "goal_direction_reconstruction": "-radial_braking_vector/norm(radial_braking_vector), justified by controller source",
            "clean_definition": "no safety projection, no boundary projection, collision=0",
            "saturated_clean_definition": "clean and terminal_correction_clipped=1",
            "missing_policy": "no imputation; invalid required vectors fail closed; optional cross-checks remain null",
        },
        "dynamics_contract": {
            "source": "src/simulation/motion_model.py:143-179,290-365",
            "dt_s": DT_S,
            "action_type": "three-dimensional velocity increment",
            "execution_equation": "delta=norm_clip(command,a_max*dt); v_next=speed_clip(v+delta); applied=v_next-v; safety/boundary may recompute v from executed displacement",
            "interpretation": "final applied delta-v divided by dt is exact executed simulator acceleration over the window; its EKF-goal-axis projection is command-side radial authority, not an axis-rotation-free observed radial derivative",
        },
        "action_chain_stages": [
            {"stage": 0, "field": "actor_raw_delta_v_mps", "meaning": "frozen Actor velocity-increment action"},
            {"stage": 1, "field": "radial_braking_delta_v_mps / radial_propulsion_delta_v_mps / terminal_tangential_damping_delta_v_mps", "meaning": "uncapped terminal component requests"},
            {"stage": 2, "field": "terminal_raw_correction_delta_v_mps", "meaning": "uncapped radial plus tangential correction"},
            {"stage": 3, "field": "goal_recovery_correction_delta_v_mps", "meaning": "correction after the 0.6 m/s norm cap, before alpha"},
            {"stage": 4, "field": "goal_recovery_applied_delta_v_mps / corrected_delta_v_pre_transform_mps", "meaning": "alpha-scaled correction / Actor plus that correction"},
            {"stage": 5, "field": "corrected_delta_v_pre_safety_mps", "meaning": "Actor parameterization/action-transform output"},
            {"stage": 6, "field": "motion_boundary_delta_v_mps / safety_projection_delta_v_mps", "meaning": "downstream boundary and safety execution deltas"},
            {"stage": 7, "field": "applied_delta_v_mps", "meaning": "actual dynamics-executed velocity increment"},
        ],
        "sample_counts": {
            "total_step_rows": len(all_steps),
            "all_braking_samples": len(samples),
            "clean_braking_samples": len(clean),
            "saturated_braking_samples": len(saturated),
            "unsaturated_braking_samples": len(unsaturated),
            "saturated_clean_braking_samples": len(saturated_clean),
            "clean_action_clipped_samples": len(clean_action_clipped),
            "episodes_with_braking": len(episode_aggregates),
            "episodes_eligible_for_conservative_estimate": len(eligible_episodes),
            "minimum_samples_per_eligible_episode": MIN_EPISODE_BRAKING_SAMPLES,
        },
        "all_braking_statistics": _subset_statistics(samples),
        "clean_braking_statistics": _subset_statistics(clean),
        "saturated_clean_braking_statistics": _subset_statistics(saturated_clean),
        "actor_radial_contribution_clean": actor_summary(clean),
        "episode_level_saturated_clean_effective_deceleration_mps2": {
            "basis": "per-episode median over saturated clean braking samples",
            **descriptive_statistics(episode_medians),
        },
        "conservative_a_eff_candidates": candidates,
        "readiness": {
            "status": "READY" if ready else "NOT_READY",
            "rule": "at least 20 saturated-clean steps, at least 5 episodes with >=3 such steps, and positive episode-median p10",
            "recommended_candidate": recommended_name,
            "recommended_conservative_a_eff_mps2": recommended,
            "calculation_basis": "p10 across per-episode median command-side effective radial deceleration in saturated clean windows" if ready else None,
            "sample_count": len(saturated_clean),
            "episode_count": len(eligible_episodes),
            "uncertainty": "episode clustering retained; recommendation is a conservative empirical command-side authority, not a hardware flight-dynamics identification",
            "recommended_stopping_margin_m": margin,
            "stopping_margin_basis": "one 0.2 s decision-window travel at 5.2 m/s plus the distance gap between p10 and median episode authority",
        },
        "stopping_distance_sanity": {
            "initial_speed_mps": v_initial,
            "final_speed_mps": v_final,
            "recommended_a_eff_mps2": recommended,
            "stopping_distance_m": stopping_distance,
            "terminal_radius_m": 8.0,
            "radius_minus_stopping_distance_m": 8.0 - stopping_distance if stopping_distance is not None else None,
            "eight_meters_physically_sufficient_under_simple_second_order_model": stopping_distance is not None and stopping_distance <= 8.0,
            "one_decision_window_travel_m": one_decision_window_travel,
            "empirical_authority_uncertainty_margin_m": empirical_authority_margin,
            "recommended_stopping_margin_m": margin,
            "stopping_distance_plus_recommended_margin_m": stopping_distance + margin if stopping_distance is not None and margin is not None else None,
            "eight_meters_robust_after_recommended_margin": stopping_distance is not None and margin is not None and stopping_distance + margin <= 8.0,
            "interpretation": "8 m is only nominally sufficient by 0.11 m in the idealized equation; one decision window alone travels 1.04 m, so 8 m is not a robust trigger distance",
            "limitations": "excludes braking-onset delay, alpha ramp, tangential path length, goal-axis rotation, Actor opposition, and future disturbances",
        },
        "radial_tangential_budget_competition": {
            "saturated_clean_requested_radial_braking_mps": descriptive_statistics(row["requested_radial_braking_mps"] for row in saturated_clean),
            "saturated_clean_requested_tangential_damping_norm_mps": descriptive_statistics(row["requested_tangential_damping_norm_mps"] for row in saturated_clean),
            "tangential_to_radial_requested_ratio": descriptive_statistics(budget_ratio),
            "radial_budget_loss_due_to_cap_mps": descriptive_statistics(row["radial_budget_loss_due_to_cap_mps"] for row in saturated_clean),
            "tangential_budget_loss_due_to_cap_mps": descriptive_statistics(row["tangential_budget_loss_due_to_cap_mps"] for row in saturated_clean),
            "separation_method": "uniform norm clip ratio applies to the summed orthogonal radial/tangential components; component losses are reconstructed from logged pre-cap vectors and clip ratio",
        },
        "goal_boundary_stall": _stall_analysis(episodes, all_steps, derived_by_key),
        "status_flags": dict(STATUS_FLAGS),
    }
    return summary


STEP_OUTPUT_FIELDS = (
    "episode_id", "episode_index", "step", "clean", "saturated", "action_clipped",
    "safety_projection_applied", "boundary_projection_applied", "goal_direction",
    "goal_distance_m", "goal_distance_after_m", "terminal_blend", "radial_velocity_error_mps",
    "current_radial_velocity_mps", "current_tangential_velocity_mps",
    "terminal_desired_radial_velocity_mps", "radial_propulsion_scale", "radial_propulsion_norm_mps",
    "requested_radial_braking_mps", "terminal_raw_combined_radial_braking_mps",
    "post_correction_cap_radial_braking_mps", "applied_correction_radial_braking_mps",
    "pre_transform_radial_braking_mps", "post_transform_radial_braking_mps",
    "final_applied_radial_braking_mps", "command_side_effective_radial_deceleration_mps2",
    "requested_tangential_damping_norm_mps", "terminal_raw_correction_norm_mps",
    "terminal_correction_clip_ratio", "radial_budget_loss_due_to_cap_mps",
    "tangential_budget_loss_due_to_cap_mps", "correction_cap_retention", "alpha_retention",
    "transform_retention", "final_retention", "loss_due_to_correction_cap_mps",
    "loss_due_to_action_transform_mps", "loss_due_to_safety_boundary_mps",
    "actor_radial_delta_v_mps", "actor_relation", "post_step_actual_radial_velocity_mps",
    "post_step_actual_tangential_velocity_mps", "post_step_actual_speed_mps",
    "observed_radial_deceleration_crosscheck_mps2", "observed_speed_deceleration_crosscheck_mps2",
    "goal_axis_rotation_deg",
)


EPISODE_OUTPUT_FIELDS = (
    "episode_id", "all_braking_samples", "clean_braking_samples",
    "saturated_clean_braking_samples", "eligible_for_conservative_estimate",
    "saturated_clean_effective_deceleration_mean_mps2",
    "saturated_clean_effective_deceleration_p10_mps2",
    "saturated_clean_effective_deceleration_median_mps2",
    "clean_effective_deceleration_mean_mps2", "clean_effective_deceleration_p10_mps2",
    "clean_effective_deceleration_median_mps2",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            for key, value in list(rendered.items()):
                if isinstance(value, (list, tuple, dict)):
                    rendered[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(rendered)


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, str)):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_report(summary: Mapping[str, Any]) -> str:
    counts = summary["sample_counts"]
    stats = summary["saturated_clean_braking_statistics"]
    actor = summary["actor_radial_contribution_clean"]
    readiness = summary["readiness"]
    stopping = summary["stopping_distance_sanity"]
    budget = summary["radial_tangential_budget_competition"]
    stall = summary["goal_boundary_stall"]
    lines = [
        "# Single-UAV Terminal Capture Effective Braking Authority Audit",
        "",
        "## Verdict",
        "",
        f"**{readiness['status']}**",
        "",
        f"Recommended conservative a_eff: {_format(readiness['recommended_conservative_a_eff_mps2'])} m/s².",
        f"Basis: {readiness.get('calculation_basis') or 'insufficient data for freezing'}.",
        "",
        "## Data and dynamics contract",
        "",
        f"- Existing step rows scanned: {counts['total_step_rows']}.",
        f"- dt: {DT_S} s (`src/simulation/motion_model.py`, `MotionConfig.dt_s`).",
        "- Action is a 3-D velocity increment. `applied_delta_v_mps` is the executed velocity change after motion limits and safety/boundary reconciliation.",
        "- Dividing its goal-axis braking projection by dt gives executed simulator acceleration over that window. It is not treated as an axis-rotation-free observed radial derivative.",
        "",
        "## Action chain field map",
        "",
        "| Stage | Log field | Physical meaning |",
        "|---:|---|---|",
        "| 0 | `actor_raw_delta_v_mps` | Frozen Actor velocity increment before external correction (already policy-parameterized, not pre-tanh latent) |",
        "| 1 | `radial_braking_delta_v_mps`, `radial_propulsion_delta_v_mps`, `terminal_tangential_damping_delta_v_mps` | Uncapped terminal component requests |",
        "| 2 | `terminal_raw_correction_delta_v_mps` | Uncapped radial plus tangential correction |",
        "| 3 | `goal_recovery_correction_delta_v_mps` | Correction after 0.6 m/s norm cap, before alpha |",
        "| 4 | `goal_recovery_applied_delta_v_mps`; `corrected_delta_v_pre_transform_mps` | Alpha-scaled correction; Actor plus that correction |",
        "| 5 | `corrected_delta_v_pre_safety_mps` | Output of `actor_action_transform` |",
        "| 6 | `motion_boundary_delta_v_mps`; `safety_projection_delta_v_mps` | Difference introduced by nominal dynamics/speed-boundary handling and scene safety projection |",
        "| 7 | `applied_delta_v_mps` | Final dynamics-executed velocity increment |",
        "",
        "## Sample counts",
        "",
        f"- All braking samples: {counts['all_braking_samples']}.",
        f"- Clean braking samples: {counts['clean_braking_samples']}.",
        f"- Saturated clean braking samples: {counts['saturated_clean_braking_samples']}.",
        f"- Eligible episodes: {counts['episodes_eligible_for_conservative_estimate']}.",
        "",
        "## Saturated clean braking authority",
        "",
        "| Quantity | mean | p10 | p25 | median | p75 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, field in (
        ("Requested radial Δv", "requested_radial_braking_mps"),
        ("Post-0.6 radial Δv", "post_correction_cap_radial_braking_mps"),
        ("Post-transform radial Δv", "post_transform_radial_braking_mps"),
        ("Final applied radial Δv", "final_applied_radial_braking_mps"),
        ("Effective radial deceleration", "command_side_effective_radial_deceleration_mps2"),
        ("Correction-cap retention", "correction_cap_retention"),
        ("Transform retention", "transform_retention"),
        ("Final retention", "final_retention"),
    ):
        item = stats[field]
        lines.append(
            f"| {label} | {_format(item['mean'])} | {_format(item['p10'])} | "
            f"{_format(item['p25'])} | {_format(item['median'])} | "
            f"{_format(item['p75'])} | {_format(item['p90'])} |"
        )
    lines.extend([
        "",
        "## Actor contribution",
        "",
        f"- Assist: {_format(actor['assist_fraction'])} ({actor['assist_count']}/{actor['count']}).",
        f"- Oppose: {_format(actor['oppose_fraction'])} ({actor['oppose_count']}/{actor['count']}).",
        f"- Neutral: {_format(actor['neutral_fraction'])} ({actor['neutral_count']}/{actor['count']}).",
        "",
        "## Conservative candidates and stopping-distance sanity",
        "",
    ])
    for name, value in summary["conservative_a_eff_candidates"].items():
        lines.append(f"- `{name}`: {_format(value)} m/s².")
    lines.extend([
        f"- 5.2 → 0.5 m/s stopping distance using recommendation: {_format(stopping['stopping_distance_m'])} m.",
        f"- 8 m minus computed distance: {_format(stopping['radius_minus_stopping_distance_m'])} m.",
        f"- Simple second-order 8 m sufficiency: {_format(stopping['eight_meters_physically_sufficient_under_simple_second_order_model'])}.",
        f"- Recommended evidence-derived stopping margin: {_format(readiness['recommended_stopping_margin_m'])} m.",
        f"- Stopping distance plus margin: {_format(stopping['stopping_distance_plus_recommended_margin_m'])} m.",
        f"- 8 m robust after margin: {_format(stopping['eight_meters_robust_after_recommended_margin'])}.",
        "- This sanity calculation excludes alpha/onset delay, tangential travel, Actor interaction and disturbances.",
        "",
        "## Radial/tangential budget competition",
        "",
        f"- Requested tangential/radial ratio median: {_format(budget['tangential_to_radial_requested_ratio']['median'])}.",
        f"- Radial cap loss median: {_format(budget['radial_budget_loss_due_to_cap_mps']['median'])} m/s.",
        f"- Tangential cap loss median: {_format(budget['tangential_budget_loss_due_to_cap_mps']['median'])} m/s.",
        "",
        "## Goal-boundary stall",
        "",
        f"- Matching episode found: {_format(stall.get('found'))}.",
        f"- Audit confirmation rule passed: {_format(stall.get('confirmed'))}.",
        f"- Episode: {stall.get('episode_id', 'N/A')}; minimum distance: {_format(stall.get('minimum_goal_distance_m'))} m.",
        "",
        "## Limitations",
        "",
        "- Step rows are repeated observations; the recommended value is derived from per-episode medians.",
        "- Reconstructed goal direction is available only when logged radial braking is non-zero.",
        "- Pre-step radial velocity is EKF-derived while post-step actual fields use simulator truth; their direct difference is not used as the primary estimate.",
        "- Consecutive actual radial/speed differences are reported only as cross-checks with goal-axis rotation diagnostics.",
        "- No rollout, environment step, training, controller modification, or parameter change was performed.",
        "",
        "## Status flags",
        "",
        "```json",
        json.dumps(summary["status_flags"], indent=2, ensure_ascii=False),
        "```",
        "",
    ])
    return "\n".join(lines)


def run_audit(terminal_root: Path, output_root: Path) -> dict[str, Any]:
    terminal_root = terminal_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if not terminal_root.is_dir():
        raise FileNotFoundError(terminal_root)
    if output_root == terminal_root or terminal_root in output_root.parents:
        raise ValueError("output root must be separate from and outside the input root")
    steps_path = terminal_root / "steps.csv"
    episodes_path = terminal_root / "terminal_capture_validation_episodes.csv"
    evaluation_summary_path = terminal_root / "evaluation_summary.json"
    summary_path = terminal_root / "terminal_capture_validation_summary.json"
    steps, _ = _read_csv(steps_path, REQUIRED_STEP_FIELDS)
    episodes, _ = _read_csv(episodes_path, REQUIRED_EPISODE_FIELDS)
    if not evaluation_summary_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("canonical evaluation and terminal-capture summaries are required")
    with evaluation_summary_path.open("r", encoding="utf-8") as stream:
        evaluation_summary = json.load(stream)
    with summary_path.open("r", encoding="utf-8") as stream:
        source_summary = json.load(stream)
    if not (
        evaluation_summary.get("stage") == "single_uav_goal_recovery_correction_evaluation"
        and evaluation_summary.get("formal") is True
        and evaluation_summary.get("condition") == "on"
        and evaluation_summary.get("terminal_capture_validation") is True
        and int(evaluation_summary.get("episode_count", 0)) == 20
    ):
        raise ValueError("input is not the canonical ON formal20 terminal-capture evaluation")
    if not (
        source_summary.get("stage") == "single_uav_terminal_capture_validation"
        and source_summary.get("candidate_canonical_defaults") is True
        and int(source_summary.get("episodes", 0)) == 20
    ):
        raise ValueError("terminal-capture summary is missing canonical formal20 identity")
    episode_ids = {str(row["episode_id"]) for row in episodes}
    seen: set[tuple[str, int]] = set()
    for row in steps:
        key = (str(row["episode_id"]), integer(row["step"]))
        if key in seen:
            raise ValueError(f"duplicate step key: {key}")
        if key[0] not in episode_ids:
            raise ValueError(f"step has no episode row: {key[0]}")
        seen.add(key)
    samples = [sample for row in steps if (sample := derive_braking_sample(row)) is not None]
    add_consecutive_crosschecks(samples, steps)
    episode_aggregates = aggregate_episode_rows(samples)
    provenance = {
        "terminal_root": str(terminal_root),
        "steps_csv_sha256": _sha256(steps_path),
        "episodes_csv_sha256": _sha256(episodes_path),
        "evaluation_summary_sha256": _sha256(evaluation_summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "evaluation_stage": evaluation_summary.get("stage"),
        "source_stage": source_summary.get("stage"),
        "source_formal": evaluation_summary.get("formal"),
        "step_rows": len(steps),
        "episode_rows": len(episodes),
    }
    summary = build_summary(
        steps, episodes, samples, episode_aggregates, input_provenance=provenance
    )
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "braking_authority_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    _write_csv(output_root / "braking_authority_steps.csv", samples, STEP_OUTPUT_FIELDS)
    _write_csv(output_root / "braking_authority_episodes.csv", episode_aggregates, EPISODE_OUTPUT_FIELDS)
    (output_root / "braking_authority_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-root", type=Path, default=DEFAULT_TERMINAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(args.terminal_root, args.output_root)
    print(json.dumps({
        "status": summary["readiness"]["status"],
        "recommended_conservative_a_eff_mps2": summary["readiness"]["recommended_conservative_a_eff_mps2"],
        "output_root": str(args.output_root.resolve()),
        **STATUS_FLAGS,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
