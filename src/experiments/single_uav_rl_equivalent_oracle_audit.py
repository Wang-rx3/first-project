"""Read-only RL-equivalent oracle and actor action audit for one UAV.

This experiment is deliberately diagnostic.  It does not train a policy, add
rollout transitions, or change the task definition.  It compares the existing
truth-state scripted controller with the same controller driven only by the
local observation that is available to the RL actor, then audits the complete
actor action path for two frozen checkpoints.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import MotionConfig, clip_vector_norm
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _assert_observation_finite,
    _copy_calibration_files,
    _episode_metrics,
    _make_environment,
    _scenario_for,
    _scale_seed,
    _write_attention_summary,
    _write_episode_metrics,
    _write_scale_manifest,
)
from .pilot import (
    _EpisodeTraceWriter,
    _append_jsonl,
    _assert_action_feasible,
    _seed_everything,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_scripted_feasibility import (
    ScriptedOracleController,
    _canonical_hash,
    _goal_from_scenario,
    _paired_scenarios,
    _position_from_scenario,
    _prepare_root,
    _scenario_payload,
    _start_distance,
)


TRUTH_ORACLE = "truth_oracle"
ESTIMATED_ORACLE = "estimated_state_oracle"
BASELINE_RL = "baseline_rl"
FINE_RL = "fine_tuned_rl"
POLICY_NAMES: tuple[str, ...] = (
    TRUTH_ORACLE,
    ESTIMATED_ORACLE,
    BASELINE_RL,
    FINE_RL,
)
RL_POLICY_NAMES = (BASELINE_RL, FINE_RL)
POLICY_LABELS = {
    TRUTH_ORACLE: "truth scripted/oracle",
    ESTIMATED_ORACLE: "RL-equivalent estimated-state oracle",
    BASELINE_RL: "coef=0.25 baseline RL",
    FINE_RL: "navigation-preserving fine-tuned RL",
}

DISTANCE_BINS: tuple[tuple[str, str], ...] = (
    ("gt_8m", ">8 m"),
    ("8_5m", "8–5 m"),
    ("5_3m", "5–3 m"),
    ("3_1m", "3–1 m"),
    ("le_1m", "≤1 m"),
)
_DISTANCE_BIN_LABELS = dict(DISTANCE_BINS)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean(values: Iterable[Any]) -> float:
    numbers: list[float] = []
    for value in values:
        number = _number(value)
        if np.isfinite(number):
            numbers.append(number)
    return float(np.mean(numbers)) if numbers else float("nan")


def _percentile(values: Iterable[Any], percentile: float) -> float:
    numbers = np.asarray(
        [_number(value) for value in values if np.isfinite(_number(value))],
        dtype=float,
    )
    return float(np.percentile(numbers, percentile)) if numbers.size else float("nan")


def _maximum(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = np.asarray(
        [_number(value) for value in values if np.isfinite(_number(value))],
        dtype=float,
    )
    return float(np.max(numbers)) if numbers.size else float(default)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _reward_number(reward: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in reward:
            value = _number(reward.get(key))
            if np.isfinite(value):
                return value
    return float("nan")


def _distance_bin(distance_m: float) -> str:
    distance = float(distance_m)
    if not np.isfinite(distance):
        return "unknown"
    if distance > 8.0:
        return "gt_8m"
    if distance > 5.0:
        return "8_5m"
    if distance > 3.0:
        return "5_3m"
    if distance > 1.0:
        return "3_1m"
    return "le_1m"


def _unit(vector: np.ndarray, tolerance: float = 1.0e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > tolerance else np.zeros(3, dtype=float)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-9 or right_norm <= 1.0e-9:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = _cosine(left, right)
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _extract_rl_visible_state(
    observation: Mapping[str, Any],
    *,
    uav_id: str = "uav_00",
    expected_node_dim: int | None = None,
) -> dict[str, Any]:
    """Extract only the actor-visible self node from a raw local observation.

    The self node layout is produced by ``build_graph_batch`` and is part of
    the actor contract: estimated EKF state [0:15], covariance diagonal
    [15:30], estimated goal delta [30:33], local obstacle features, and
    optional motion features.  Deliberately do not inspect ``estimate`` or
    ``goal`` convenience fields, and never fall back to simulator truth.
    """

    if not isinstance(observation, Mapping):
        raise ValueError("RL-equivalent oracle requires a mapping observation")
    if "node_features" not in observation:
        raise ValueError("RL-equivalent oracle requires actor-visible node_features")
    node_features = np.asarray(observation["node_features"], dtype=float)
    if node_features.ndim != 2 or node_features.shape[0] < 1 or node_features.shape[1] < 33:
        raise ValueError(
            "actor-visible node_features must have shape (nodes, >=33) for the self node"
        )
    if expected_node_dim is not None and node_features.shape[1] != int(expected_node_dim):
        raise ValueError(
            f"actor node feature dimension {node_features.shape[1]} does not match "
            f"checkpoint/environment dimension {expected_node_dim}"
        )
    node_ids = list(observation.get("node_ids", []))
    own_index = node_ids.index(uav_id) if uav_id in node_ids else 0
    if own_index >= node_features.shape[0]:
        raise ValueError("uav_00 node id is not represented in node_features")
    row = node_features[own_index]
    estimate = row[:15].copy()
    goal_delta = row[30:33].copy()
    if not np.isfinite(estimate).all() or not np.isfinite(goal_delta).all():
        raise ValueError("actor-visible estimated state contains non-finite values")
    position = estimate[:3]
    velocity = estimate[3:6]
    distance = float(np.linalg.norm(goal_delta))
    return {
        "position_m": position,
        "velocity_mps": velocity,
        "goal_delta_m": goal_delta,
        "distance_m": distance,
        "goal_direction": _unit(goal_delta),
        "radial_velocity_mps": float(np.dot(velocity, _unit(goal_delta))),
        "node_index": int(own_index),
        "node_feature_dim": int(node_features.shape[1]),
        "state_source": "actor_visible_node_features_only",
    }


class EstimatedStateOracleController(ScriptedOracleController):
    """The scripted oracle driven only by RL-visible EKF/node features.

    The base class supplies the speed schedule and static configuration.  This
    override intentionally does not call ``ScriptedOracleController.action``;
    that method reads simulator truth.  Every control quantity below comes
    from ``observations[uav_00][node_features]``.
    """

    def action(
        self,
        observations: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.uav_id not in observations:
            raise ValueError(f"missing observation for {self.uav_id}")
        visible = _extract_rl_visible_state(observations[self.uav_id])
        distance = float(visible["distance_m"])
        direction = np.asarray(visible["goal_direction"], dtype=float)
        velocity = np.asarray(visible["velocity_mps"], dtype=float)
        desired_speed = self.desired_speed(distance)
        desired_velocity = direction * desired_speed
        raw_delta = desired_velocity - velocity
        legal_delta = clip_vector_norm(raw_delta, self.max_delta_v_mps)
        speed = float(np.linalg.norm(velocity))
        velocity_direction = _unit(velocity)
        demanded_longitudinal = float(np.dot(velocity_direction, raw_delta))
        demanded_braking = max(0.0, -demanded_longitudinal)
        return {
            self.uav_id: {
                "delta_v": legal_delta.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }, {
            "estimated_goal_distance_before_m": distance,
            "estimated_position_m": visible["position_m"].tolist(),
            "estimated_velocity_mps": velocity.tolist(),
            "estimated_goal_delta_m": visible["goal_delta_m"].tolist(),
            "estimated_goal_direction": direction.tolist(),
            "estimated_radial_velocity_mps": visible["radial_velocity_mps"],
            "desired_speed_mps": desired_speed,
            "desired_velocity_mps": desired_velocity.tolist(),
            "raw_delta_v_mps": raw_delta.tolist(),
            "raw_delta_v_norm_mps": float(np.linalg.norm(raw_delta)),
            "legal_delta_v_mps": legal_delta.tolist(),
            "legal_delta_v_norm_mps": float(np.linalg.norm(legal_delta)),
            "controller_delta_v_capped": int(
                np.linalg.norm(raw_delta) > self.max_delta_v_mps + 1.0e-9
            ),
            "demanded_braking_delta_v_mps": demanded_braking,
            "pre_speed_mps": speed,
            "state_source": "actor_visible_node_features_only",
            "truth_state_read_for_control": False,
        }


def _transition_geometry(
    *,
    pre_position: np.ndarray,
    post_position: np.ndarray,
    pre_velocity: np.ndarray,
    post_velocity: np.ndarray,
    goal: np.ndarray,
    commanded_delta: np.ndarray,
    applied_delta: np.ndarray,
    dt_s: float,
) -> dict[str, Any]:
    pre_goal = goal - pre_position
    post_goal = goal - post_position
    pre_distance = float(np.linalg.norm(pre_goal))
    post_distance = float(np.linalg.norm(post_goal))
    pre_direction = _unit(pre_goal)
    post_direction = _unit(post_goal)
    pre_speed = float(np.linalg.norm(pre_velocity))
    post_speed = float(np.linalg.norm(post_velocity))
    pre_radial_velocity = float(np.dot(pre_velocity, pre_direction))
    post_radial_velocity = float(np.dot(post_velocity, post_direction))
    pre_tangent = pre_velocity - pre_radial_velocity * pre_direction
    post_tangent = post_velocity - post_radial_velocity * post_direction
    action_projection = float(np.dot(commanded_delta, pre_direction))
    applied_projection = float(np.dot(applied_delta, pre_direction))
    velocity_direction = _unit(pre_velocity)
    applied_velocity_projection = float(np.dot(applied_delta, velocity_direction))
    commanded_velocity_projection = float(np.dot(commanded_delta, velocity_direction))
    tangential_action = commanded_delta - action_projection * pre_direction
    applied_tangential_action = applied_delta - applied_projection * pre_direction
    return {
        "true_goal_distance_before_m": pre_distance,
        "true_goal_distance_m": post_distance,
        "true_delta_goal_distance_m": pre_distance - post_distance,
        "true_radial_progress_mps": (pre_distance - post_distance) / max(dt_s, 1.0e-12),
        "true_radial_velocity_before_mps": pre_radial_velocity,
        "radial_velocity_toward_mps": post_radial_velocity,
        "tangential_speed_before_mps": float(np.linalg.norm(pre_tangent)),
        "tangential_speed_mps": float(np.linalg.norm(post_tangent)),
        "total_speed_before_mps": pre_speed,
        "total_speed_mps": post_speed,
        "action_projection_toward_goal_mps": action_projection,
        "applied_action_projection_toward_goal_mps": applied_projection,
        "braking_projection_mps": max(0.0, -commanded_velocity_projection),
        "applied_braking_projection_mps": max(0.0, -applied_velocity_projection),
        "tangential_action_mps": float(np.linalg.norm(tangential_action)),
        "applied_tangential_action_mps": float(np.linalg.norm(applied_tangential_action)),
        "cosine_action_goal": _cosine(commanded_delta, pre_direction),
        "cosine_applied_action_goal": _cosine(applied_delta, pre_direction),
        "cosine_velocity_goal": _cosine(pre_velocity, pre_direction),
        "cosine_post_velocity_goal": _cosine(post_velocity, post_direction),
        "action_norm_mps": float(np.linalg.norm(commanded_delta)),
        "applied_action_norm_mps": float(np.linalg.norm(applied_delta)),
        "distance_bin": _distance_bin(pre_distance),
        "inside_8m": int(post_distance <= 8.0),
        "inside_5m": int(post_distance <= 5.0),
        "inside_3m": int(post_distance <= 3.0),
        "inside_1m": int(post_distance <= 1.0),
    }


def _base_step_row(
    *,
    policy: str,
    episode_index: int,
    seed: int,
    step_index: int,
    timestamp_s: float,
    scenario_hash: str,
    geometry: Mapping[str, Any],
    estimated_before: Mapping[str, Any] | None,
    estimated_after: Mapping[str, Any] | None,
    control: Mapping[str, Any],
    execution: Mapping[str, Any],
    next_info: Mapping[str, Any],
    reward: Mapping[str, Any],
) -> dict[str, Any]:
    estimated_before_distance = (
        _number(estimated_before.get("distance_m"))
        if isinstance(estimated_before, Mapping)
        else float("nan")
    )
    estimated_after_distance = (
        _number(estimated_after.get("distance_m"))
        if isinstance(estimated_after, Mapping)
        else _reward_number(reward, "current_goal_distance")
    )
    applied = np.asarray(
        execution.get("delta_v_applied", control.get("legal_delta_v_mps", (0.0, 0.0, 0.0))),
        dtype=float,
    ).reshape(3)
    commanded = np.asarray(
        control.get("legal_delta_v_mps", (0.0, 0.0, 0.0)), dtype=float
    ).reshape(3)
    row = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_index": int(episode_index),
        "seed": int(seed),
        "step_index": int(step_index),
        "timestamp_s": float(timestamp_s),
        "scenario_hash": scenario_hash,
        "estimated_goal_distance_before_m": estimated_before_distance,
        "estimated_goal_distance_m": estimated_after_distance,
        "estimated_distance_error_m": (
            estimated_after_distance - _number(geometry.get("true_goal_distance_m"))
            if np.isfinite(estimated_after_distance)
            else float("nan")
        ),
        "desired_speed_mps": _number(control.get("desired_speed_mps")),
        "target_speed_mps": _reward_number(reward, "single_uav_target_speed_mps"),
        "delta_v_demanded_norm_mps": _number(control.get("raw_delta_v_norm_mps")),
        "delta_v_commanded_norm_mps": float(np.linalg.norm(commanded)),
        "delta_v_applied_norm_mps": float(np.linalg.norm(applied)),
        "braking_demanded_mps": _number(control.get("demanded_braking_delta_v_mps"), 0.0),
        "commanded_braking_delta_v_mps": _number(
            execution.get("commanded_braking_delta_v_mps"),
            max(0.0, -float(np.dot(_unit(np.asarray(control.get("pre_velocity_mps", (0.0, 0.0, 0.0)), dtype=float)), commanded))),
        ),
        "applied_braking_delta_v_mps": _number(
            execution.get("braking_delta_v_mps"),
            float(geometry.get("applied_braking_projection_mps", 0.0)),
        ),
        "actual_speed_reduction_mps": _number(execution.get("speed_reduction_mps"), 0.0),
        "braking_action": int(_number(execution.get("braking_action"), 0.0) > 0.5),
        "effective_braking": int(_number(execution.get("brake_effective"), 0.0) > 0.5),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "controller_delta_v_capped": int(control.get("controller_delta_v_capped", 0)),
        "braking_progress_signal": _reward_number(reward, "single_uav_braking_progress_signal"),
        "braking_progress_reward": _reward_number(reward, "single_uav_braking_progress_reward"),
        "target_velocity_action_reward": _reward_number(reward, "single_uav_target_velocity_action_reward"),
        "goal_progress_signal": _reward_number(reward, "single_uav_goal_progress_signal"),
        "goal_progress_reward": _reward_number(
            reward, "single_uav_goal_progress_reward", "goal_progress_reward"
        ),
        "goal_direction_signal": _reward_number(reward, "single_uav_goal_direction_signal"),
        "goal_direction_reward": _reward_number(reward, "single_uav_goal_direction_reward"),
        "stable_condition": int(
            bool(next_info.get("single_uav_goal_in_radius", False))
            and bool(next_info.get("single_uav_goal_speed_ok", False))
        ),
        "goal_stable": int(bool(next_info.get("single_uav_goal_stable", False))),
        "consecutive_stable_steps": int(round(_number(next_info.get("single_uav_stable_steps"), 0.0))),
        "termination_reason": str(next_info.get("termination_reason", "running")),
        **dict(geometry),
        "estimated_inside_8m": int(np.isfinite(estimated_after_distance) and estimated_after_distance <= 8.0),
        "estimated_inside_5m": int(np.isfinite(estimated_after_distance) and estimated_after_distance <= 5.0),
        "estimated_inside_3m": int(np.isfinite(estimated_after_distance) and estimated_after_distance <= 3.0),
        "estimated_inside_1m": int(np.isfinite(estimated_after_distance) and estimated_after_distance <= 1.0),
    }
    if isinstance(estimated_before, Mapping):
        row["estimated_radial_velocity_before_mps"] = _number(
            estimated_before.get("radial_velocity_mps")
        )
    if isinstance(estimated_after, Mapping):
        row["estimated_radial_velocity_mps"] = _number(
            estimated_after.get("radial_velocity_mps")
        )
    return row


def _oracle_state_from_state(state: Any, goal: np.ndarray) -> dict[str, Any]:
    position = np.asarray(state.position_m, dtype=float).copy()
    velocity = np.asarray(state.velocity_mps, dtype=float).copy()
    delta = goal - position
    return {
        "position_m": position,
        "velocity_mps": velocity,
        "goal_delta_m": delta,
        "distance_m": float(np.linalg.norm(delta)),
        "goal_direction": _unit(delta),
        "radial_velocity_mps": float(np.dot(velocity, _unit(delta))),
    }


def _oracle_state_from_environment(environment: Any) -> dict[str, Any]:
    return _oracle_state_from_state(
        environment.states["uav_00"],
        np.asarray(environment.goals["uav_00"], dtype=float).copy(),
    )


def _geometry_with_estimates(
    *,
    pre_position: np.ndarray,
    post_position: np.ndarray,
    pre_velocity: np.ndarray,
    post_velocity: np.ndarray,
    goal: np.ndarray,
    commanded: np.ndarray,
    applied: np.ndarray,
    dt_s: float,
) -> dict[str, Any]:
    return _transition_geometry(
        pre_position=pre_position,
        post_position=post_position,
        pre_velocity=pre_velocity,
        post_velocity=post_velocity,
        goal=goal,
        commanded_delta=commanded,
        applied_delta=applied,
        dt_s=dt_s,
    )


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = _json_text(value)
                output[column] = value
            writer.writerow(output)


def _entry_step(rows: Sequence[Mapping[str, Any]], threshold: float) -> int | None:
    for row in rows:
        distance = _number(row.get("estimated_goal_distance_m"))
        if np.isfinite(distance) and distance <= threshold:
            return int(row.get("step_index", 0)) + 1
    return None


def _speed_at_entry(rows: Sequence[Mapping[str, Any]], threshold: float) -> float:
    for row in rows:
        distance = _number(row.get("estimated_goal_distance_m"))
        if np.isfinite(distance) and distance <= threshold:
            return _number(row.get("total_speed_mps"))
    return float("nan")


def _episode_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    writer_summary: Mapping[str, Any],
) -> dict[str, Any]:
    estimated_distances = np.asarray(
        [_number(row.get("estimated_goal_distance_m")) for row in rows], dtype=float
    )
    true_distances = np.asarray(
        [_number(row.get("true_goal_distance_m")) for row in rows], dtype=float
    )
    estimated_distances = estimated_distances[np.isfinite(estimated_distances)]
    true_distances = true_distances[np.isfinite(true_distances)]
    holds = [
        int(round(_number(row.get("consecutive_stable_steps"), 0.0)))
        for row in rows
    ]
    max_hold = max(holds, default=0)
    termination_reason = str(writer_summary.get("termination_reason", "running"))

    def mean_in_distance(max_distance: float) -> float:
        return _mean(
            row.get("total_speed_mps")
            for row in rows
            if np.isfinite(_number(row.get("estimated_goal_distance_m")))
            and _number(row.get("estimated_goal_distance_m")) <= max_distance
        )

    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_id": str(writer_summary.get("episode_id", "")),
        "scenario_id": str(writer_summary.get("scenario_id", "")),
        "episode_index": int(rows[0].get("episode_index", 0)) if rows else 0,
        "seed": int(rows[0].get("seed", writer_summary.get("seed", 0))) if rows else int(writer_summary.get("seed", 0)),
        "scenario_hash": str(rows[0].get("scenario_hash", "")) if rows else "",
        "steps": len(rows),
        "entered_8m": int(estimated_distances.size and np.min(estimated_distances) <= 8.0),
        "entered_5m": int(estimated_distances.size and np.min(estimated_distances) <= 5.0),
        "entered_3m": int(estimated_distances.size and np.min(estimated_distances) <= 3.0),
        "entered_1m": int(estimated_distances.size and np.min(estimated_distances) <= 1.0),
        "entered_8m_true": int(true_distances.size and np.min(true_distances) <= 8.0),
        "entered_5m_true": int(true_distances.size and np.min(true_distances) <= 5.0),
        "entered_3m_true": int(true_distances.size and np.min(true_distances) <= 3.0),
        "entered_1m_true": int(true_distances.size and np.min(true_distances) <= 1.0),
        "stable_goal": int(any(int(row.get("goal_stable", 0)) for row in rows)),
        "minimum_goal_distance_m": float(np.min(estimated_distances)) if estimated_distances.size else float("nan"),
        "minimum_estimated_goal_distance_m": float(np.min(estimated_distances)) if estimated_distances.size else float("nan"),
        "minimum_true_goal_distance_m": float(np.min(true_distances)) if true_distances.size else float("nan"),
        "mean_speed_mps": _mean(row.get("total_speed_mps") for row in rows),
        "mean_speed_within_1m_mps": mean_in_distance(1.0),
        "mean_speed_at_8m_entry_mps": _speed_at_entry(rows, 8.0),
        "mean_speed_at_5m_entry_mps": _speed_at_entry(rows, 5.0),
        "mean_speed_at_3m_entry_mps": _speed_at_entry(rows, 3.0),
        "mean_speed_at_1m_entry_mps": _speed_at_entry(rows, 1.0),
        "maximum_consecutive_goal_hold_steps": max_hold,
        "maximum_consecutive_stable_steps": max_hold,
        "hold_steps_ge_1": int(max_hold >= 1),
        "hold_steps_ge_3": int(max_hold >= 3),
        "hold_steps_ge_5": int(max_hold >= 5),
        "steps_to_8m": _entry_step(rows, 8.0),
        "steps_to_5m": _entry_step(rows, 5.0),
        "steps_to_3m": _entry_step(rows, 3.0),
        "steps_to_1m": _entry_step(rows, 1.0),
        "action_clipping_fraction": _mean(row.get("action_clipped") for row in rows),
        "effective_braking_fraction": _mean(row.get("effective_braking") for row in rows),
        "braking_action_fraction": _mean(row.get("braking_action") for row in rows),
        "maximum_action_norm_mps": _maximum(row.get("action_norm_mps") for row in rows),
        "maximum_applied_action_norm_mps": _maximum(row.get("applied_action_norm_mps") for row in rows),
        "maximum_braking_projection_mps": _maximum(row.get("braking_projection_mps") for row in rows),
        "maximum_applied_braking_projection_mps": _maximum(row.get("applied_braking_projection_mps") for row in rows),
        "timeout": int(termination_reason == "time_limit"),
        "termination_reason": termination_reason,
        "collision_occurred": bool(writer_summary.get("collision_occurred", False)),
    }


def _aggregate_episode_summaries(
    summaries: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    selected = [row for row in summaries if str(row.get("policy")) == policy]
    count = len(selected)

    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0) or 0) for row in selected))

    def mean_field(name: str) -> float:
        return _mean(row.get(name) for row in selected)

    def mean_steps(name: str) -> float:
        return _mean(row.get(name) for row in selected if row.get(name) is not None)

    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": count,
        "entered_8m_episodes": count_field("entered_8m"),
        "entered_8m_rate": _rate(count_field("entered_8m"), count),
        "entered_5m_episodes": count_field("entered_5m"),
        "entered_5m_rate": _rate(count_field("entered_5m"), count),
        "entered_3m_episodes": count_field("entered_3m"),
        "entered_3m_rate": _rate(count_field("entered_3m"), count),
        "entered_1m_episodes": count_field("entered_1m"),
        "entered_1m_rate": _rate(count_field("entered_1m"), count),
        "stable_goal_episodes": count_field("stable_goal"),
        "stable_goal_rate": _rate(count_field("stable_goal"), count),
        "minimum_goal_distance_m": min(
            (_number(row.get("minimum_goal_distance_m")) for row in selected),
            default=float("nan"),
        ),
        "mean_minimum_goal_distance_m": mean_field("minimum_goal_distance_m"),
        "mean_true_minimum_goal_distance_m": mean_field("minimum_true_goal_distance_m"),
        "mean_speed_mps": mean_field("mean_speed_mps"),
        "mean_speed_within_1m_mps": mean_field("mean_speed_within_1m_mps"),
        "mean_speed_at_8m_entry_mps": mean_field("mean_speed_at_8m_entry_mps"),
        "mean_speed_at_5m_entry_mps": mean_field("mean_speed_at_5m_entry_mps"),
        "mean_speed_at_3m_entry_mps": mean_field("mean_speed_at_3m_entry_mps"),
        "mean_speed_at_1m_entry_mps": mean_field("mean_speed_at_1m_entry_mps"),
        "mean_maximum_consecutive_goal_hold_steps": mean_field("maximum_consecutive_goal_hold_steps"),
        "maximum_consecutive_goal_hold_steps_max": int(
            round(_maximum((row.get("maximum_consecutive_goal_hold_steps") for row in selected), 0.0))
        ),
        "episodes_with_hold_steps_ge_1": count_field("hold_steps_ge_1"),
        "episodes_with_hold_steps_ge_1_rate": _rate(count_field("hold_steps_ge_1"), count),
        "episodes_with_hold_steps_ge_3": count_field("hold_steps_ge_3"),
        "episodes_with_hold_steps_ge_3_rate": _rate(count_field("hold_steps_ge_3"), count),
        "episodes_with_hold_steps_ge_5": count_field("hold_steps_ge_5"),
        "episodes_with_hold_steps_ge_5_rate": _rate(count_field("hold_steps_ge_5"), count),
        "timeout_episodes": count_field("timeout"),
        "timeout_rate": _rate(count_field("timeout"), count),
        "mean_steps_to_8m": mean_steps("steps_to_8m"),
        "mean_steps_to_5m": mean_steps("steps_to_5m"),
        "mean_steps_to_3m": mean_steps("steps_to_3m"),
        "mean_steps_to_1m": mean_steps("steps_to_1m"),
        "mean_action_norm_mps": mean_field("mean_action_norm_mps"),
        "mean_action_clipping_fraction": mean_field("action_clipping_fraction"),
        "mean_effective_braking_fraction": mean_field("effective_braking_fraction"),
        "mean_braking_action_fraction": mean_field("braking_action_fraction"),
        "maximum_braking_projection_mps": _maximum(
            (row.get("maximum_braking_projection_mps") for row in selected), 0.0
        ),
    }


def _aggregate_actor_rows(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("policy")) == policy]
    if not selected:
        return {"policy": policy, "policy_label": POLICY_LABELS[policy], "rows": 0}
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "rows": len(selected),
        "mean_actor_raw_delta_norm_mps": _mean(row.get("actor_raw_delta_norm_mps") for row in selected),
        "maximum_actor_raw_delta_norm_mps": _maximum(row.get("actor_raw_delta_norm_mps") for row in selected),
        "mean_actor_tanh_norm": _mean(row.get("actor_tanh_norm") for row in selected),
        "mean_actor_scaled_delta_norm_mps": _mean(row.get("actor_scaled_delta_norm_mps") for row in selected),
        "maximum_actor_scaled_delta_norm_mps": _maximum(row.get("actor_scaled_delta_norm_mps") for row in selected),
        "mean_applied_delta_norm_mps": _mean(row.get("applied_action_norm_mps") for row in selected),
        "maximum_applied_delta_norm_mps": _maximum(row.get("applied_action_norm_mps") for row in selected),
        "mean_action_scaling_residual_norm_mps": _mean(row.get("action_scaling_residual_norm_mps") for row in selected),
        "maximum_action_scaling_residual_norm_mps": _maximum(row.get("action_scaling_residual_norm_mps") for row in selected),
        "mean_action_clipping": _mean(row.get("action_clipped") for row in selected),
        "mean_action_projection_toward_goal_mps": _mean(row.get("action_projection_toward_goal_mps") for row in selected),
        "mean_braking_projection_mps": _mean(row.get("braking_projection_mps") for row in selected),
        "maximum_braking_projection_mps": _maximum(row.get("braking_projection_mps") for row in selected),
        "mean_tangential_action_mps": _mean(row.get("tangential_action_mps") for row in selected),
        "positive_radial_action_fraction": _mean(
            row.get("positive_radial_action") for row in selected
        ),
        "negative_radial_action_fraction": _mean(
            row.get("negative_radial_action") for row in selected
        ),
        "mean_radial_velocity_mps": _mean(row.get("radial_velocity_toward_mps") for row in selected),
        "negative_radial_velocity_fraction": _mean(
            row.get("negative_radial_velocity") for row in selected
        ),
        "mean_effective_braking": _mean(row.get("effective_braking") for row in selected),
    }


def _distance_bin_summary(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, label in DISTANCE_BINS:
        selected = [
            row
            for row in rows
            if str(row.get("policy")) == policy and str(row.get("distance_bin")) == name
        ]
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "distance_bin": name,
                "distance_bin_label": label,
                "rows": len(selected),
                "mean_action_norm_mps": _mean(row.get("action_norm_mps") for row in selected),
                "action_norm_p50_mps": _percentile((row.get("action_norm_mps") for row in selected), 50),
                "action_norm_p90_mps": _percentile((row.get("action_norm_mps") for row in selected), 90),
                "action_norm_max_mps": _maximum(row.get("action_norm_mps") for row in selected),
                "mean_radial_action_projection_mps": _mean(
                    row.get("action_projection_toward_goal_mps") for row in selected
                ),
                "mean_braking_projection_mps": _mean(
                    row.get("braking_projection_mps") for row in selected
                ),
                "mean_tangential_action_mps": _mean(
                    row.get("tangential_action_mps") for row in selected
                ),
                "positive_radial_action_fraction": _mean(
                    row.get("positive_radial_action") for row in selected
                ),
                "negative_radial_action_fraction": _mean(
                    row.get("negative_radial_action") for row in selected
                ),
                "mean_radial_velocity_mps": _mean(
                    row.get("radial_velocity_toward_mps") for row in selected
                ),
                "negative_radial_velocity_fraction": _mean(
                    row.get("negative_radial_velocity") for row in selected
                ),
                "mean_braking_magnitude_mps": _mean(
                    row.get("braking_projection_mps") for row in selected
                ),
                "action_clipping_fraction": _mean(
                    row.get("action_clipped") for row in selected
                ),
                "effective_braking_fraction": _mean(
                    row.get("effective_braking") for row in selected
                ),
                "mean_total_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
                "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in selected),
            }
        )
    return output


def _actor_step_row(
    *,
    policy: str,
    episode_index: int,
    seed: int,
    step_index: int,
    timestamp_s: float,
    scenario_hash: str,
    raw_observation: Mapping[str, Any],
    decision: Any,
    trainer: Any,
    pre_position: np.ndarray,
    post_position: np.ndarray,
    pre_velocity: np.ndarray,
    post_velocity: np.ndarray,
    goal: np.ndarray,
    execution: Mapping[str, Any],
    next_observations: Mapping[str, Mapping[str, Any]],
    next_info: Mapping[str, Any],
    reward: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a row for the full frozen actor action transformation chain."""

    import torch

    uav_id = "uav_00"
    local_actor_observation = decision.observations[uav_id]
    sample = decision.samples[uav_id]
    with torch.no_grad():
        _measurement, _bandwidth, delta_distribution, _mask = trainer.policy._actor_distributions(
            local_actor_observation
        )
        base_distribution = delta_distribution.base_dist
        distribution_mean = base_distribution.mean.detach().cpu().numpy().astype(float)
        distribution_std = base_distribution.scale.detach().cpu().numpy().astype(float)

    raw_delta = np.asarray(sample.delta_raw, dtype=float).reshape(3)
    tanh_delta = np.tanh(raw_delta)
    component_limit = float(trainer.policy._delta_component_limit())
    scaled_delta = tanh_delta * component_limit
    commanded_delta = np.asarray(sample.delta_v, dtype=float).reshape(3)
    applied_delta = np.asarray(
        execution.get("delta_v_applied", commanded_delta), dtype=float
    ).reshape(3)
    pre_direction = _unit(goal - pre_position)
    pre_velocity_direction = _unit(pre_velocity)
    control = {
        "legal_delta_v_mps": commanded_delta.tolist(),
        "raw_delta_v_norm_mps": float(np.linalg.norm(raw_delta)),
        "pre_velocity_mps": pre_velocity.tolist(),
        "demanded_braking_delta_v_mps": max(
            0.0, -float(np.dot(pre_velocity_direction, commanded_delta))
        ),
        "desired_speed_mps": float("nan"),
    }
    geometry = _geometry_with_estimates(
        pre_position=pre_position,
        post_position=post_position,
        pre_velocity=pre_velocity,
        post_velocity=post_velocity,
        goal=goal,
        commanded=commanded_delta,
        applied=applied_delta,
        dt_s=float(trainer.policy.config.dt_s)
        if hasattr(trainer.policy.config, "dt_s")
        else 0.2,
    )
    row = _base_step_row(
        policy=policy,
        episode_index=episode_index,
        seed=seed,
        step_index=step_index,
        timestamp_s=timestamp_s,
        scenario_hash=scenario_hash,
        geometry=geometry,
        estimated_before=_extract_rl_visible_state(
            raw_observation, expected_node_dim=int(trainer.config.node_feature_dim)
        ),
        estimated_after=_extract_rl_visible_state(
            next_observations[uav_id], expected_node_dim=int(trainer.config.node_feature_dim)
        ),
        control=control,
        execution=execution,
        next_info=next_info,
        reward=reward,
    )
    row.update(
        {
            "actor_observation_json": _json_text(local_actor_observation),
            "raw_environment_observation_json": _json_text(raw_observation),
            "actor_raw_delta_json": _json_text(raw_delta),
            "actor_distribution_mean_json": _json_text(distribution_mean),
            "actor_distribution_std_json": _json_text(distribution_std),
            "actor_tanh_output_json": _json_text(tanh_delta),
            "actor_scaled_delta_v_json": _json_text(scaled_delta),
            "action_applied_json": _json_text(applied_delta),
            "actor_action_deterministic": 1,
            "actor_action_squash": "tanh",
            "actor_action_scale": "component_limit=max_delta_v/sqrt(3)",
            "action_scale_component_limit_mps": component_limit,
            "actor_raw_delta_norm_mps": float(np.linalg.norm(raw_delta)),
            "actor_distribution_mean_norm": float(np.linalg.norm(distribution_mean)),
            "actor_distribution_std_norm": float(np.linalg.norm(distribution_std)),
            "actor_tanh_norm": float(np.linalg.norm(tanh_delta)),
            "actor_scaled_delta_norm_mps": float(np.linalg.norm(scaled_delta)),
            "action_scaling_residual_norm_mps": float(
                np.linalg.norm(commanded_delta - scaled_delta)
            ),
            "actor_observation_node_feature_dim": int(
                np.asarray(local_actor_observation["node_features"]).shape[1]
            ),
            "actor_observation_state_source": "trainer.normalizer(raw_environment_observation)",
            "actor_distribution_kind": "Normal_then_TanhTransform",
            "positive_radial_action": int(
                float(geometry["action_projection_toward_goal_mps"]) > 1.0e-4
            ),
            "negative_radial_action": int(
                float(geometry["action_projection_toward_goal_mps"]) < -1.0e-4
            ),
            "negative_radial_velocity": int(
                float(geometry["radial_velocity_toward_mps"]) < -1.0e-4
            ),
        }
    )
    return row


def _complete_geometry_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add common action-direction flags to oracle and actor rows."""

    result = dict(row)
    result["positive_radial_action"] = int(
        _number(result.get("action_projection_toward_goal_mps"), 0.0) > 1.0e-4
    )
    result["negative_radial_action"] = int(
        _number(result.get("action_projection_toward_goal_mps"), 0.0) < -1.0e-4
    )
    result["negative_radial_velocity"] = int(
        _number(result.get("radial_velocity_toward_mps"), 0.0) < -1.0e-4
    )
    return result


def _runner_geometry(
    environment: Any,
    *,
    goal: np.ndarray,
    pre_state: Any,
    post_state: Any,
    commanded: np.ndarray,
    applied: np.ndarray,
) -> dict[str, Any]:
    return _geometry_with_estimates(
        pre_position=np.asarray(pre_state.position_m, dtype=float).copy(),
        post_position=np.asarray(post_state.position_m, dtype=float).copy(),
        pre_velocity=np.asarray(pre_state.velocity_mps, dtype=float).copy(),
        post_velocity=np.asarray(post_state.velocity_mps, dtype=float).copy(),
        goal=goal,
        commanded=commanded,
        applied=applied,
        dt_s=float(environment.config.motion.dt_s),
    )


def _run_oracle_episode(
    *,
    controller_kind: str,
    config: FormalExperimentConfig,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
    progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(
        seed=seed, scenario_config=copy.deepcopy(dict(scenario))
    )
    episode_id = f"episode_{serial:06d}"
    policy = TRUTH_ORACLE if controller_kind == TRUTH_ORACLE else ESTIMATED_ORACLE
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="rl_equivalent_oracle_audit",
        environment=environment,
        normalization_fitted=False,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=policy,
    )
    controller = (
        ScriptedOracleController(environment)
        if controller_kind == TRUTH_ORACLE
        else EstimatedStateOracleController(environment)
    )
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        pre_state = environment.states["uav_00"].copy()
        if controller_kind == TRUTH_ORACLE:
            action, control = controller.action()
        else:
            action, control = controller.action(observations)
        control = dict(control)
        control["pre_velocity_mps"] = pre_state.velocity_mps.tolist()
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        applied = np.asarray(
            execution.get("delta_v_applied", action["uav_00"]["delta_v"]),
            dtype=float,
        ).reshape(3)
        commanded = np.asarray(action["uav_00"]["delta_v"], dtype=float).reshape(3)
        geometry = _runner_geometry(
            environment,
            goal=goal,
            pre_state=pre_state,
            post_state=environment.states["uav_00"],
            commanded=commanded,
            applied=applied,
        )
        estimated_before = (
            _oracle_state_from_state(pre_state, goal)
            if controller_kind == TRUTH_ORACLE
            else _extract_rl_visible_state(observations["uav_00"])
        )
        estimated_after = _extract_rl_visible_state(next_observations["uav_00"])
        row = _base_step_row(
            policy=policy,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            geometry=geometry,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            control=control,
            execution=execution,
            next_info=next_info,
            reward=next_info.get("learning_reward_components", {}),
        )
        row = _complete_geometry_flags(row)
        row.update(
            {
                "controller_state_source": (
                    "simulator_truth_position_velocity_and_configured_goal"
                    if controller_kind == TRUTH_ORACLE
                    else "actor_visible_node_features_only"
                ),
                "truth_state_read_for_control": bool(controller_kind == TRUTH_ORACLE),
                "estimated_true_distance_error_before_m": (
                    _number(estimated_before.get("distance_m"))
                    - _number(geometry.get("true_goal_distance_before_m"))
                ),
                "estimated_true_direction_angular_error_deg": _angle_deg(
                    np.asarray(estimated_before.get("goal_direction", (0.0, 0.0, 0.0)), dtype=float),
                    _unit(goal - pre_state.position_m),
                ),
                "estimated_true_radial_velocity_error_mps": (
                    _number(estimated_before.get("radial_velocity_mps"))
                    - float(geometry.get("true_radial_velocity_before_mps", 0.0))
                ),
            }
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=policy,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    summary = writer.finalize()
    summary.update(
        {
            "seed": int(seed),
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "policy_source": policy,
            "controller_state_source": (
                "simulator_truth_position_velocity_and_configured_goal"
                if controller_kind == TRUTH_ORACLE
                else "actor_visible_node_features_only"
            ),
            "truth_state_read_for_control": bool(controller_kind == TRUTH_ORACLE),
            "paired_episode_index": int(episode_index),
            "paired_scenario_hash": scenario_hash,
        }
    )
    return summary, rows


def _run_frozen_rl_episode(
    *,
    policy: str,
    trainer: Any,
    config: FormalExperimentConfig,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one deterministic checkpoint episode without any trainer update."""

    import torch

    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(
        seed=seed, scenario_config=copy.deepcopy(dict(scenario))
    )
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="rl_equivalent_oracle_audit",
        environment=environment,
        normalization_fitted=bool(
            trainer.normalizer.fitted and trainer.central_normalizer.fitted
        ),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=policy,
    )
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    actor_jsonl = scale_root / "diagnostics" / "actor_observations.jsonl"
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        pre_state = environment.states["uav_00"].copy()
        with torch.no_grad():
            decision = trainer.act(
                observations,
                info["central_state"],
                deterministic=True,
            )
        _assert_action_feasible(environment, decision.actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(
            decision.actions
        )
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        commanded = np.asarray(
            decision.actions["uav_00"].get("delta_v", (0.0, 0.0, 0.0)), dtype=float
        ).reshape(3)
        applied = np.asarray(
            execution.get("delta_v_applied", commanded), dtype=float
        ).reshape(3)
        geometry = _runner_geometry(
            environment,
            goal=goal,
            pre_state=pre_state,
            post_state=environment.states["uav_00"],
            commanded=commanded,
            applied=applied,
        )
        row = _actor_step_row(
            policy=policy,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            raw_observation=observations["uav_00"],
            decision=decision,
            trainer=trainer,
            pre_position=pre_state.position_m.copy(),
            post_position=environment.states["uav_00"].position_m.copy(),
            pre_velocity=pre_state.velocity_mps.copy(),
            post_velocity=environment.states["uav_00"].velocity_mps.copy(),
            goal=goal,
            execution=execution,
            next_observations=next_observations,
            next_info=next_info,
            reward=next_info.get("learning_reward_components", {}),
        )
        # Reuse the already computed geometry so the row's geometry and the
        # action-chain values are guaranteed to describe the same transition.
        row.update(geometry)
        row = _complete_geometry_flags(row)
        row.update(
            {
                "controller_state_source": "not_applicable_frozen_actor",
                "truth_state_read_for_control": False,
                "estimated_true_distance_error_before_m": (
                    _number(_extract_rl_visible_state(observations["uav_00"]).get("distance_m"))
                    - float(geometry["true_goal_distance_before_m"])
                ),
                "estimated_true_direction_angular_error_deg": _angle_deg(
                    np.asarray(
                        _extract_rl_visible_state(observations["uav_00"]).get(
                            "goal_direction", (0.0, 0.0, 0.0)
                        ),
                        dtype=float,
                    ),
                    _unit(goal - pre_state.position_m),
                ),
                "estimated_true_radial_velocity_error_mps": (
                    _number(_extract_rl_visible_state(observations["uav_00"]).get("radial_velocity_mps"))
                    - float(geometry["true_radial_velocity_before_mps"])
                ),
            }
        )
        rows.append(row)
        _append_jsonl(
            actor_jsonl,
            {
                "policy": policy,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "seed": seed,
                "step_index": step,
                "scenario_hash": scenario_hash,
                "actor_observation": decision.observations["uav_00"],
                "raw_environment_observation": observations["uav_00"],
                "actor_raw_delta": row.get("actor_raw_delta_json"),
                "actor_distribution_mean": row.get("actor_distribution_mean_json"),
                "actor_distribution_std": row.get("actor_distribution_std_json"),
                "actor_tanh_output": row.get("actor_tanh_output_json"),
                "actor_scaled_delta_v": row.get("actor_scaled_delta_v_json"),
                "action_applied": row.get("action_applied_json"),
            },
        )
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=decision.actions,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=decision.graph_traces,
            policy_source=policy,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    summary = writer.finalize()
    summary.update(
        {
            "seed": int(seed),
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "policy_source": policy,
            "controller_state_source": "not_applicable_frozen_actor",
            "truth_state_read_for_control": False,
            "paired_episode_index": int(episode_index),
            "paired_scenario_hash": scenario_hash,
        }
    )
    return summary, rows


def _write_normalization_copy(scale_root: Path, payload: Mapping[str, Any]) -> None:
    normalizer = payload.get("normalization_stats")
    central = payload.get("central_normalization_stats")
    if isinstance(normalizer, Mapping):
        _write_json(
            scale_root / "normalization_stats.json",
            _jsonable(
                {
                    "normalization_stats": dict(normalizer),
                    "central_normalization_stats": dict(central)
                    if isinstance(central, Mapping)
                    else None,
                    "source": "frozen_checkpoint",
                }
            ),
        )


def _effective_contract(
    config: FormalExperimentConfig,
    *,
    environment: Any,
    policy_type: str,
) -> dict[str, Any]:
    motion = environment.config.motion
    reward = environment.config.reward
    return {
        "policy_type": policy_type,
        "uav_count": int(config.uav_count),
        "seed": int(config.seed),
        "scenario_seed": config.scenario_seed,
        "gnss_mode": str(config.gnss_mode),
        "max_steps": int(config.max_steps),
        "dt_s": float(motion.dt_s),
        "max_delta_v_per_step_mps": float(motion.max_acceleration_mps2 * motion.dt_s),
        "max_acceleration_mps2": float(motion.max_acceleration_mps2),
        "max_speed_mps": float(motion.max_speed_mps),
        "action_physical_meaning": motion.action_contract(),
        "environment_safety_projection_enabled": bool(
            environment.config.safety_projection_enabled
        ),
        "braking_safety_projection_enabled": bool(
            reward.single_uav_braking_safety_projection_enabled
        ),
        "observation_state_source": (
            "actor receives normalized local observation; self node is EKF estimate"
            if policy_type == "frozen_rl"
            else "truth simulator state"
            if policy_type == TRUTH_ORACLE
            else "actor-visible EKF node features only"
        ),
        "localization_estimate_source": "EKF state estimate in node_features[0:15]",
        "estimated_goal_source": "goal_delta = configured goal - EKF estimated position in node_features[30:33]",
        "goal_distance_for_rl_observation": "norm(node_features[30:33])",
        "goal_radius_m": float(reward.single_uav_goal_threshold_m),
        "stable_speed_threshold_mps": float(reward.single_uav_goal_speed_threshold_mps),
        "stable_hold_steps": int(reward.single_uav_stable_hold_steps),
        "stable_goal_definition": (
            "estimated goal distance <= goal radius and actual speed <= stable speed "
            "threshold for stable_hold_steps consecutive transitions"
        ),
        "goal_termination": "collision or stable_goal; otherwise time_limit at max_steps",
        "single_uav_learning_mode": str(reward.single_uav_learning_mode),
        "single_uav_speed_control_mode": str(reward.single_uav_speed_control_mode),
        "near_goal_radius_m": float(reward.single_uav_near_goal_radius_m),
        "target_speed_far_mps": float(reward.single_uav_target_speed_far_mps),
        "braking_acceleration_mps2": float(reward.single_uav_braking_acceleration_mps2),
        "braking_margin_m": float(reward.single_uav_braking_margin_m),
        "braking_distance_speed_profile": str(
            reward.single_uav_braking_distance_speed_profile
        ),
        "goal_hold_reward": float(reward.single_uav_goal_hold_reward),
        "braking_progress_reward_coef": float(
            reward.single_uav_braking_progress_reward_coef
        ),
        "braking_action_reward_coef": float(
            reward.single_uav_braking_action_reward_coef
        ),
        "target_velocity_action_reward_coef": float(
            reward.single_uav_target_velocity_action_reward_coef
        ),
        "braking_curriculum_fraction": float(
            config.single_uav_braking_curriculum_fraction
        ),
        "terminal_braking_curriculum_fraction": float(
            config.single_uav_terminal_braking_curriculum_fraction
        ),
        "random_stream_mode": str(config.random_stream_mode),
        "safety_execution_mode": str(config.safety_execution_mode),
        "dynamic_obstacle_motion_features_enabled": bool(
            config.dynamic_obstacle_motion_features_enabled
        ),
        "dynamic_obstacle_swept_projection_enabled": bool(
            config.dynamic_obstacle_swept_projection_enabled
        ),
        "include_remaining_time": bool(config.include_remaining_time),
        "encoder_type": str(config.encoder_type),
        "node_feature_dim": int(config.node_feature_dim),
        "max_obstacles": int(config.max_obstacles),
    }


def _config_snapshot(config: FormalExperimentConfig) -> dict[str, Any]:
    """Return the checkpoint-level fields relevant to this audit."""

    names = (
        "uav_count",
        "seed",
        "scenario_seed",
        "gnss_mode",
        "max_steps",
        "single_uav_learning_mode",
        "single_uav_speed_control_mode",
        "single_uav_near_goal_radius_m",
        "single_uav_goal_threshold_m",
        "single_uav_goal_speed_threshold_mps",
        "single_uav_stable_hold_steps",
        "single_uav_target_speed_far_mps",
        "single_uav_braking_acceleration_mps2",
        "single_uav_braking_margin_m",
        "single_uav_braking_distance_speed_profile",
        "single_uav_braking_safety_projection_enabled",
        "single_uav_braking_curriculum_fraction",
        "single_uav_terminal_braking_curriculum_fraction",
        "single_uav_goal_hold_reward",
        "single_uav_braking_progress_reward_coef",
        "single_uav_braking_action_reward_coef",
        "single_uav_target_velocity_action_reward_coef",
        "dynamic_obstacle_motion_features_enabled",
        "dynamic_obstacle_swept_projection_enabled",
        "safety_execution_mode",
        "random_stream_mode",
        "include_remaining_time",
        "encoder_type",
        "max_obstacles",
        "learning_reward_clip",
    )
    return {name: _jsonable(getattr(config, name)) for name in names}


def _configuration_audit(
    *,
    baseline_config: FormalExperimentConfig,
    fine_config: FormalExperimentConfig,
    scenarios: Sequence[Mapping[str, Any]],
    baseline_checkpoint: Path,
    fine_checkpoint: Path,
    baseline_sha256: str,
    fine_sha256: str,
) -> dict[str, Any]:
    baseline_env = _make_environment(
        baseline_config,
        _scale_seed(baseline_config.seed, 1, "test"),
        metric_scales=baseline_config.metric_scales,
    )
    fine_env = _make_environment(
        fine_config,
        _scale_seed(fine_config.seed, 1, "test"),
        metric_scales=fine_config.metric_scales,
    )
    truth_env = _make_environment(
        baseline_config,
        _scale_seed(baseline_config.seed, 1, "test"),
        metric_scales=baseline_config.metric_scales,
    )
    baseline_contract = _effective_contract(
        baseline_config, environment=baseline_env, policy_type="frozen_rl"
    )
    fine_contract = _effective_contract(
        fine_config, environment=fine_env, policy_type="frozen_rl"
    )
    truth_contract = _effective_contract(
        baseline_config, environment=truth_env, policy_type=TRUTH_ORACLE
    )
    training_only_fields = {
        "braking_curriculum_fraction",
        "terminal_braking_curriculum_fraction",
    }
    comparison_fields = [
        key
        for key in baseline_contract
        if key
        not in {
            "policy_type",
            "observation_state_source",
            "localization_estimate_source",
            "estimated_goal_source",
            "goal_distance_for_rl_observation",
        }
        and key not in training_only_fields
    ]
    effective_differences = {
        key: {"baseline": baseline_contract.get(key), "fine_tuned": fine_contract.get(key)}
        for key in comparison_fields
        if _json_text(baseline_contract.get(key)) != _json_text(fine_contract.get(key))
    }
    old_summary_path = Path(
        "data/synthetic/single_uav_scripted_feasibility_seed2026/summary.json"
    )
    previous_contract: Mapping[str, Any] = {}
    if old_summary_path.is_file():
        try:
            loaded = json.loads(old_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping) and isinstance(loaded.get("task_contract"), Mapping):
                previous_contract = loaded["task_contract"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous_contract = {}
    safety_fields = (
        "environment_safety_projection_enabled",
        "braking_safety_projection_enabled",
    )
    previous_safety_comparison = {
        field: {
            "previous_feasibility": previous_contract.get(field),
            "current_baseline": baseline_contract.get(field),
            "matches": previous_contract.get(field) == baseline_contract.get(field),
        }
        for field in safety_fields
    }
    scenario_hashes = [str(item["scenario_hash"]) for item in scenarios]
    obstacle_hashes = [
        _canonical_hash(_scenario_payload(item["scenario"]).get("obstacle_specs", []))
        for item in scenarios
    ]
    observation_contract = {
        "raw_local_observation_is_source": True,
        "trainer_network_input": "trainer.normalizer.transform(raw local observation)",
        "node_feature_dim": int(baseline_config.node_feature_dim),
        "self_node_selection": "node_ids.index('uav_00'), falling back to index 0 only because a one-UAV local graph has one self node",
        "node_feature_slices": {
            "estimated_ekf_state": {
                "slice": "[0:15]",
                "fields": [
                    "position_x",
                    "position_y",
                    "position_z",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                    "roll",
                    "pitch",
                    "yaw",
                    "remaining_ekf_state_9_to_14",
                ],
            },
            "covariance_diagonal": {"slice": "[15:30]", "fields": ["EKF covariance diagonal"]},
            "estimated_goal_delta": {
                "slice": "[30:33]",
                "fields": ["goal_x - estimated_x", "goal_y - estimated_y", "goal_z - estimated_z"],
            },
            "local_obstacle_features": {"slice": "[33:37]", "fields": ["clearance", "box_count", "sphere_count", "timestep"]},
            "braking_features": {"slice": "[37:39]", "fields": ["normalized_safe_target_speed", "normalized_brake_margin"]},
        },
        "estimated_oracle_decision_inputs": [
            "node_features only",
            "estimated position = self node [0:3]",
            "estimated velocity = self node [3:6]",
            "estimated goal delta = self node [30:33]",
            "estimated distance = norm(estimated goal delta)",
            "estimated goal direction = estimated goal delta / estimated distance",
        ],
        "forbidden_controller_inputs": [
            "environment.states",
            "environment.goals",
            "simulator ground-truth position/velocity",
            "ground-truth goal-relative vector",
        ],
    }
    return {
        "baseline": {
            "checkpoint_path": str(baseline_checkpoint.resolve()),
            "checkpoint_sha256": baseline_sha256,
            "checkpoint_config": _config_snapshot(baseline_config),
            "effective_contract": baseline_contract,
        },
        "fine_tuned": {
            "checkpoint_path": str(fine_checkpoint.resolve()),
            "checkpoint_sha256": fine_sha256,
            "checkpoint_config": _config_snapshot(fine_config),
            "effective_contract": fine_contract,
        },
        "truth_oracle_environment_contract": truth_contract,
        "baseline_vs_fine_effective_environment_differences": effective_differences,
        "test_effective_environment_contract_matches": not bool(effective_differences),
        "baseline_vs_fine_training_only_differences": {
            key: {
                "baseline": baseline_contract.get(key),
                "fine_tuned": fine_contract.get(key),
            }
            for key in sorted(training_only_fields)
            if _json_text(baseline_contract.get(key))
            != _json_text(fine_contract.get(key))
        },
        "previous_feasibility_safety_comparison": previous_safety_comparison,
        "paired_scenario_audit": {
            "episodes": len(scenarios),
            "same_scenario_passed_to_all_policies": True,
            "same_seed_indices": True,
            "all_curriculum_used_flags_false": all(
                not bool(item.get("curriculum_used", False)) for item in scenarios
            ),
            "scenario_hashes": scenario_hashes,
            "unique_scenario_hash_count": len(set(scenario_hashes)),
            "unique_obstacle_geometry_hash_count": len(set(obstacle_hashes)),
            "start_distance_m": [float(item["start_distance_m"]) for item in scenarios],
        },
        "observation_contract": observation_contract,
        "action_contract": {
            "actor_distribution": "Normal(mean, exp(delta_logstd)) followed by TanhTransform",
            "pre_tanh_field": "PolicySample.delta_raw",
            "post_tanh_field": "tanh(delta_raw)",
            "scaled_action": "tanh(delta_raw) * (max_delta_v / sqrt(3))",
            "component_limit_mps": float(
                baseline_contract["max_delta_v_per_step_mps"] / math.sqrt(3.0)
            ),
            "environment_interface": "environment.step({'uav_00': {'delta_v': ..., 'measurement_uav_id': ..., 'bandwidth': ...}})",
            "post_action_execution": "info.action_execution.uav_00.delta_v_applied",
            "environment_action_clipping": "recorded from action_execution.action_clipped",
        },
        "controller_state_source_contract": {
            TRUTH_ORACLE: "simulator truth position/velocity and configured goal",
            ESTIMATED_ORACLE: "actor-visible self node features only; no truth fallback",
            BASELINE_RL: "frozen actor input from normalized local observation",
            FINE_RL: "frozen actor input from normalized local observation",
        },
    }


def _write_role_manifest(
    *,
    role: str,
    role_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    checkpoint_path: Path | None,
    checkpoint_sha_before: str | None,
    checkpoint_sha_after: str | None,
    policy_hash_before: str | None,
    policy_hash_after: str | None,
    controller_uses_truth: bool,
) -> dict[str, Any]:
    scale_root = role_root / "scale_1"
    _write_scale_manifest(scale_root=scale_root, config=config, episodes=episode_summaries)
    path = scale_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "single_uav_rl_equivalent_oracle_audit",
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "policy_source": role,
            "policy_label": POLICY_LABELS[role],
            "feasibility_contract": {
                "normal_far_distance_test_only": True,
                "curriculum_used": False,
                "controller_uses_simulator_truth": bool(controller_uses_truth),
                "controller_state_source": (
                    "simulator_truth_position_velocity_and_configured_goal"
                    if controller_uses_truth
                    else "actor_visible_node_features_only"
                    if role == ESTIMATED_ORACLE
                    else "frozen_actor_network"
                ),
                "action_interface": "environment.step(delta_v)",
                "direct_position_assignment": False,
                "direct_velocity_assignment": False,
                "stable_judgement_reused": True,
            },
            "read_only_contract": [
                "no MAPPOTrainer.update call",
                "no rollout transition added",
                "no optimizer state used for evaluation",
                "no reward, curriculum, dynamics, action-space, or stable criterion change",
            ],
            "checkpoint": (
                {
                    "path": str(checkpoint_path.resolve()),
                    "sha256_before": checkpoint_sha_before,
                    "sha256_after": checkpoint_sha_after,
                    "unchanged": checkpoint_sha_before == checkpoint_sha_after,
                }
                if checkpoint_path is not None
                else {"not_applicable": True}
            ),
            "policy_identity": (
                {
                    "hash_before": policy_hash_before,
                    "hash_after": policy_hash_after,
                    "unchanged": policy_hash_before == policy_hash_after,
                }
                if policy_hash_before is not None
                else {"not_applicable": True},
            ),
        }
    )
    _write_json(path, _jsonable(manifest))
    _write_json(
        role_root / "policy_identity.json",
        _jsonable(
            {
                "role": role,
                "checkpoint": manifest["checkpoint"],
                "policy_identity": manifest["policy_identity"],
                "policy_updates_performed": False,
            }
        ),
    )
    return manifest


def _rows_by_policy_step(
    rows: Sequence[Mapping[str, Any]], policy: str, field: str
) -> tuple[list[float], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if str(row.get("policy")) != policy:
            continue
        step = int(row.get("step_index", -1))
        value = _number(row.get(field))
        if step >= 0 and np.isfinite(value):
            grouped.setdefault(step, []).append(value)
    steps = sorted(grouped)
    return [float(step) for step in steps], [float(np.mean(grouped[step])) for step in steps]


def _plot_audit_outputs(
    *,
    step_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        TRUTH_ORACLE: "#009E73",
        ESTIMATED_ORACLE: "#E69F00",
        BASELINE_RL: "#0072B2",
        FINE_RL: "#D55E00",
    }
    paths: dict[str, str] = {}

    def save(figure: Any, name: str) -> None:
        path = output_dir / name
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths[name.rsplit(".", 1)[0]] = str(path.resolve())

    def time_plot(field: str, title: str, ylabel: str, name: str, *, zero_line: bool = False) -> None:
        figure, axis = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        for policy in POLICY_NAMES:
            steps, values = _rows_by_policy_step(step_rows, policy, field)
            if steps:
                axis.plot(
                    [step * 0.2 for step in steps],
                    values,
                    color=colors[policy],
                    linewidth=2.0,
                    label=POLICY_LABELS[policy],
                )
        if zero_line:
            axis.axhline(0.0, color="#374151", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        save(figure, name)

    time_plot(
        "true_goal_distance_m",
        "Paired policies: true goal distance vs time",
        "True goal distance (m)",
        "goal_distance_vs_time.png",
    )
    time_plot(
        "radial_velocity_toward_mps",
        "Paired policies: radial velocity toward goal",
        "Radial velocity (m/s)",
        "radial_velocity_vs_time.png",
        zero_line=True,
    )
    time_plot(
        "total_speed_mps",
        "Paired policies: total speed vs time",
        "Total speed (m/s)",
        "total_speed_vs_time.png",
    )

    def scatter_plot(x_field: str, y_field: str, title: str, xlabel: str, ylabel: str, name: str) -> None:
        figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
        for policy in POLICY_NAMES:
            x_values: list[float] = []
            y_values: list[float] = []
            for row in step_rows:
                if str(row.get("policy")) != policy:
                    continue
                x_value = _number(row.get(x_field))
                y_value = _number(row.get(y_field))
                if np.isfinite(x_value) and np.isfinite(y_value):
                    x_values.append(x_value)
                    y_values.append(y_value)
            if x_values:
                axis.scatter(
                    x_values,
                    y_values,
                    s=10,
                    alpha=0.24,
                    color=colors[policy],
                    label=POLICY_LABELS[policy],
                    edgecolors="none",
                )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        save(figure, name)

    scatter_plot(
        "true_goal_distance_before_m",
        "action_projection_toward_goal_mps",
        "Action projection toward goal by true distance",
        "True goal distance before action (m)",
        "Action projection toward goal (m/s)",
        "action_projection_toward_goal_vs_distance.png",
    )
    scatter_plot(
        "true_goal_distance_before_m",
        "braking_projection_mps",
        "Braking projection magnitude by true distance",
        "True goal distance before action (m)",
        "Opposite-velocity action projection (m/s)",
        "braking_projection_vs_distance.png",
    )

    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    bins = np.linspace(0.0, 0.65, 27)
    for policy in POLICY_NAMES:
        values = [
            _number(row.get("action_norm_mps"))
            for row in step_rows
            if str(row.get("policy")) == policy
            and np.isfinite(_number(row.get("action_norm_mps")))
        ]
        if values:
            axis.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                color=colors[policy],
                label=POLICY_LABELS[policy],
            )
    axis.set_title("Action norm distribution")
    axis.set_xlabel("Commanded delta-v norm (m/s)")
    axis.set_ylabel("Step count")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save(figure, "action_norm_distribution.png")

    estimated_rows = [
        row
        for row in step_rows
        if str(row.get("policy")) == ESTIMATED_ORACLE
        and np.isfinite(_number(row.get("estimated_goal_distance_m")))
        and np.isfinite(_number(row.get("true_goal_distance_m")))
    ]
    figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    if estimated_rows:
        x = [_number(row.get("true_goal_distance_m")) for row in estimated_rows]
        y = [_number(row.get("estimated_goal_distance_m")) for row in estimated_rows]
        axis.scatter(x, y, s=12, alpha=0.28, color=colors[ESTIMATED_ORACLE], edgecolors="none")
        lower = min(x + y)
        upper = max(x + y)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.0, label="zero error")
    axis.set_title("Estimated-state oracle: estimated vs true goal distance")
    axis.set_xlabel("True goal distance (m)")
    axis.set_ylabel("Estimated goal distance (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save(figure, "estimated_vs_true_goal_distance.png")

    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    if estimated_rows:
        x = [_number(row.get("true_goal_distance_before_m")) for row in estimated_rows]
        y = [_number(row.get("estimated_true_direction_angular_error_deg")) for row in estimated_rows]
        valid = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
        if valid:
            axis.scatter(
                [pair[0] for pair in valid],
                [pair[1] for pair in valid],
                s=12,
                alpha=0.3,
                color=colors[ESTIMATED_ORACLE],
                edgecolors="none",
            )
    axis.set_title("Estimated goal-direction angular error by distance")
    axis.set_xlabel("True goal distance before action (m)")
    axis.set_ylabel("Angular error (degrees)")
    axis.grid(alpha=0.25)
    save(figure, "direction_angular_error_vs_distance.png")
    return paths


def _write_role_outputs(
    *,
    role: str,
    role_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path | None,
    checkpoint_sha_before: str | None,
    checkpoint_sha_after: str | None,
    policy_hash_before: str | None,
    policy_hash_after: str | None,
    controller_uses_truth: bool,
) -> dict[str, Any]:
    scale_root = role_root / "scale_1"
    standard_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    for writer_summary in episode_summaries:
        episode_path = Path(str(writer_summary["path"]))
        metric = _episode_metrics(episode_path, writer_summary)
        metric.update(
            {
                "policy": role,
                "policy_label": POLICY_LABELS[role],
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "curriculum_used": False,
            }
        )
        standard_rows.append(metric)
        episode_rows = [
            row
            for row in step_rows
            if str(row.get("episode_index")) == str(
                writer_summary.get("paired_episode_index", row.get("episode_index"))
            )
        ]
        # The row-level episode index is unique within a role.  When the
        # caller has not added paired metadata, use the episode id suffix.
        if not episode_rows:
            serial_text = str(writer_summary.get("episode_id", "")).rsplit("_", 1)[-1]
            try:
                serial = int(serial_text)
            except ValueError:
                serial = -1
            episode_rows = [row for row in step_rows if int(row.get("episode_index", -1)) == serial]
        feasibility_rows.append(
            _episode_summary(episode_rows, policy=role, writer_summary=writer_summary)
        )
    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", standard_rows)
    _write_rows_csv(
        scale_root / "diagnostics" / "feasibility_episode_metrics.csv", feasibility_rows
    )
    _write_rows_csv(scale_root / "diagnostics" / "step_metrics.csv", step_rows)
    _write_rows_csv(
        scale_root / "diagnostics" / "distance_bin_metrics.csv",
        _distance_bin_summary(step_rows, role),
    )
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", [])
    manifest = _write_role_manifest(
        role=role,
        role_root=role_root,
        config=config,
        episode_summaries=episode_summaries,
        checkpoint_path=checkpoint_path,
        checkpoint_sha_before=checkpoint_sha_before,
        checkpoint_sha_after=checkpoint_sha_after,
        policy_hash_before=policy_hash_before,
        policy_hash_after=policy_hash_after,
        controller_uses_truth=controller_uses_truth,
    )
    validation = validate_episode_benchmark(scale_root)
    return {
        "role": role,
        "policy_label": POLICY_LABELS[role],
        "episodes": len(episode_summaries),
        "schema_validation": validation,
        "episode_metrics_path": str(
            (scale_root / "diagnostics" / "episode_metrics.csv").resolve()
        ),
        "feasibility_episode_metrics_path": str(
            (scale_root / "diagnostics" / "feasibility_episode_metrics.csv").resolve()
        ),
        "step_metrics_path": str(
            (scale_root / "diagnostics" / "step_metrics.csv").resolve()
        ),
        "distance_bin_metrics_path": str(
            (scale_root / "diagnostics" / "distance_bin_metrics.csv").resolve()
        ),
        "manifest": str((scale_root / "manifest.json").resolve()),
        "checkpoint_unchanged": (
            checkpoint_sha_before == checkpoint_sha_after
            if checkpoint_sha_before is not None
            else True
        ),
        "policy_unchanged": (
            policy_hash_before == policy_hash_after
            if policy_hash_before is not None
            else True
        ),
        "no_training_update_call": True,
        "no_transition_added": True,
        "episode_summaries": [dict(row) for row in episode_summaries],
        "feasibility_episode_metrics": feasibility_rows,
    }


def _write_root_manifest(output_root: Path, roles: Mapping[str, Mapping[str, Any]]) -> None:
    _write_json(
        output_root / "manifest.json",
        _jsonable(
            {
                "dataset_id": "single_uav_rl_equivalent_oracle_audit",
                "schema_version": EPISODE_SCHEMA_VERSION,
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "roles": {
                    role: {
                        "path": role,
                        "scale": "scale_1",
                        "manifest": f"{role}/scale_1/manifest.json",
                        "schema_ok": bool(
                            report.get("schema_validation", {}).get("ok", False)
                        ),
                    }
                    for role, report in roles.items()
                },
            }
        ),
    )


def _classify_root_cause(
    *,
    aggregates: Mapping[str, Mapping[str, Any]],
    actor_aggregates: Mapping[str, Mapping[str, Any]],
    config_audit: Mapping[str, Any],
) -> dict[str, Any]:
    truth = aggregates.get(TRUTH_ORACLE, {})
    estimated = aggregates.get(ESTIMATED_ORACLE, {})
    baseline = aggregates.get(BASELINE_RL, {})
    fine = aggregates.get(FINE_RL, {})
    estimated_stable = _number(estimated.get("stable_goal_rate"), 0.0)
    truth_stable = _number(truth.get("stable_goal_rate"), 0.0)
    scaling_residual = max(
        _number(actor_aggregates.get(BASELINE_RL, {}).get("maximum_action_scaling_residual_norm_mps"), 0.0),
        _number(actor_aggregates.get(FINE_RL, {}).get("maximum_action_scaling_residual_norm_mps"), 0.0),
    )
    scaled_vs_applied_gap = max(
        _number(actor_aggregates.get(BASELINE_RL, {}).get("mean_actor_scaled_delta_norm_mps"), 0.0)
        - _number(actor_aggregates.get(BASELINE_RL, {}).get("mean_applied_delta_norm_mps"), 0.0),
        _number(actor_aggregates.get(FINE_RL, {}).get("mean_actor_scaled_delta_norm_mps"), 0.0)
        - _number(actor_aggregates.get(FINE_RL, {}).get("mean_applied_delta_norm_mps"), 0.0),
    )
    direction_deficit = max(
        0.0,
        0.5
        - _number(
            actor_aggregates.get(BASELINE_RL, {}).get("positive_radial_action_fraction"),
            0.0,
        ),
    )
    magnitude_deficit = (
        _number(actor_aggregates.get(BASELINE_RL, {}).get("mean_actor_scaled_delta_norm_mps"), 0.0)
        < 0.25 * _number(truth.get("maximum_braking_projection_mps"), 0.6)
    )
    if truth_stable >= 0.9 and estimated_stable < 0.5:
        primary = "B"
        primary_text = "truth oracle succeeds but RL-equivalent estimated-state oracle degrades; inspect observation/localization sufficiency"
    elif scaling_residual > 1.0e-5 or scaled_vs_applied_gap > 1.0e-4:
        primary = "mapping"
        primary_text = "a non-negligible action scaling/execution gap is present"
    elif direction_deficit > 0.0:
        primary = "D"
        primary_text = "actor action projection toward the goal is frequently insufficient or negative"
    elif magnitude_deficit:
        primary = "C"
        primary_text = "actor action magnitude is materially below the available braking authority"
    else:
        primary = "A"
        primary_text = "the actor action output requires further causal separation; no single dominant failure was detected"
    return {
        "primary_class": primary,
        "primary_interpretation": primary_text,
        "classes": {
            "A_actor_raw_action_small": bool(magnitude_deficit),
            "B_observation_or_localization": bool(truth_stable >= 0.9 and estimated_stable < 0.5),
            "C_actor_action_magnitude_or_authority": bool(magnitude_deficit),
            "D_direction_learning": bool(direction_deficit > 0.0),
            "action_mapping_gap": bool(scaling_residual > 1.0e-5 or scaled_vs_applied_gap > 1.0e-4),
        },
        "supporting_rates": {
            "truth_stable_rate": truth_stable,
            "estimated_oracle_stable_rate": estimated_stable,
            "baseline_stable_rate": _number(baseline.get("stable_goal_rate"), 0.0),
            "fine_tuned_stable_rate": _number(fine.get("stable_goal_rate"), 0.0),
        },
        "no_automatic_next_training_step": True,
        "config_audit_test_environment_equal": bool(
            config_audit.get("test_effective_environment_contract_matches", False)
        ),
    }


def evaluate_single_uav_rl_equivalent_oracle_audit(
    baseline_checkpoint: str | Path,
    fine_tuned_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the frozen four-policy observation/action audit."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    output_root = Path(output_dir)
    logs_root = (
        Path(logs_dir)
        if logs_dir is not None
        else Path("logs/preliminary") / output_root.name
    )
    plots_root = (
        Path(plots_dir)
        if plots_dir is not None
        else Path("plots/preliminary") / output_root.name
    )
    _prepare_root(output_root, force=force)
    _prepare_root(logs_root, force=force)
    _prepare_root(plots_root, force=force)

    baseline_path = Path(baseline_checkpoint)
    fine_path = Path(fine_tuned_checkpoint)
    if not baseline_path.is_file() or not fine_path.is_file():
        raise FileNotFoundError("both frozen RL checkpoints must exist")
    import torch

    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    fine_payload = torch.load(fine_path, map_location="cpu", weights_only=False)
    if not isinstance(baseline_payload, Mapping) or not isinstance(fine_payload, Mapping):
        raise ValueError("checkpoint payloads must be mappings")
    baseline_payload = dict(baseline_payload)
    fine_payload = dict(fine_payload)
    baseline_sha = _sha256(baseline_path)
    fine_sha = _sha256(fine_path)
    baseline_config = _config_from_checkpoint(
        baseline_payload, output_dir=output_root / BASELINE_RL, progress=progress
    )
    fine_config = _config_from_checkpoint(
        fine_payload, output_dir=output_root / FINE_RL, progress=progress
    )
    if baseline_config.uav_count != 1 or fine_config.uav_count != 1:
        raise ValueError("the RL-equivalent audit requires two 1-UAV checkpoints")
    for name in ("seed", "scenario_seed", "max_steps", "gnss_mode"):
        if getattr(baseline_config, name) != getattr(fine_config, name):
            raise ValueError(f"baseline and fine-tuned configs disagree on {name}")

    scenarios = _paired_scenarios(baseline_config, int(episodes))
    scenario_manifest = [
        {
            "episode_index": int(item["episode_index"]),
            "seed": int(item["seed"]),
            "scenario_hash": str(item["scenario_hash"]),
            "start_distance_m": float(item["start_distance_m"]),
            "curriculum_used": bool(item.get("curriculum_used", False)),
        }
        for item in scenarios
    ]
    _write_json(output_root / "paired_scenario_manifest.json", _jsonable(scenario_manifest))
    _write_json(logs_root / "paired_scenario_manifest.json", _jsonable(scenario_manifest))
    config_audit = _configuration_audit(
        baseline_config=baseline_config,
        fine_config=fine_config,
        scenarios=scenarios,
        baseline_checkpoint=baseline_path,
        fine_checkpoint=fine_path,
        baseline_sha256=baseline_sha,
        fine_sha256=fine_sha,
    )
    _write_json(logs_root / "configuration_audit.json", _jsonable(config_audit))
    _write_json(output_root / "configuration_audit.json", _jsonable(config_audit))

    role_reports: dict[str, dict[str, Any]] = {}
    role_summaries: dict[str, list[dict[str, Any]]] = {}
    role_step_rows: dict[str, list[dict[str, Any]]] = {}

    for role, controller_kind, serial_base in (
        (TRUTH_ORACLE, TRUTH_ORACLE, 100000),
        (ESTIMATED_ORACLE, ESTIMATED_ORACLE, 200000),
    ):
        role_root = output_root / role
        scale_root = role_root / "scale_1"
        scale_root.mkdir(parents=True, exist_ok=True)
        _copy_calibration_files(scale_root, baseline_config)
        summaries: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for item in scenarios:
            if progress:
                print(
                    f"[rl-equivalent-audit][{role}] episode "
                    f"{item['episode_index']}/{len(scenarios)}",
                    flush=True,
                )
            summary, episode_rows = _run_oracle_episode(
                controller_kind=controller_kind,
                config=baseline_config,
                scale_root=scale_root,
                scenario=item["scenario"],
                scenario_hash=str(item["scenario_hash"]),
                episode_index=int(item["episode_index"]),
                seed=int(item["seed"]),
                serial=serial_base + int(item["episode_index"]),
                progress=progress,
            )
            summaries.append(summary)
            rows.extend(episode_rows)
        report = _write_role_outputs(
            role=role,
            role_root=role_root,
            config=baseline_config,
            episode_summaries=summaries,
            step_rows=rows,
            checkpoint_path=None,
            checkpoint_sha_before=None,
            checkpoint_sha_after=None,
            policy_hash_before=None,
            policy_hash_after=None,
            controller_uses_truth=role == TRUTH_ORACLE,
        )
        role_reports[role] = report
        role_summaries[role] = report["feasibility_episode_metrics"]
        role_step_rows[role] = rows

    for role, payload, serial_base in (
        (BASELINE_RL, baseline_payload, 300000),
        (FINE_RL, fine_payload, 400000),
    ):
        role_root = output_root / role
        scale_root = role_root / "scale_1"
        scale_root.mkdir(parents=True, exist_ok=True)
        config = baseline_config if role == BASELINE_RL else fine_config
        _copy_calibration_files(scale_root, config)
        _write_normalization_copy(scale_root, payload)
        trainer, policy_hashes, _ = _load_frozen_trainer(
            payload, config, metric_scales=config.metric_scales
        )
        trainer.policy.eval()
        summaries = []
        rows = []
        for item in scenarios:
            if progress:
                print(
                    f"[rl-equivalent-audit][{role}] episode "
                    f"{item['episode_index']}/{len(scenarios)}",
                    flush=True,
                )
            summary, episode_rows = _run_frozen_rl_episode(
                policy=role,
                trainer=trainer,
                config=config,
                scale_root=scale_root,
                scenario=item["scenario"],
                scenario_hash=str(item["scenario_hash"]),
                episode_index=int(item["episode_index"]),
                seed=int(item["seed"]),
                serial=serial_base + int(item["episode_index"]),
            )
            summaries.append(summary)
            rows.extend(episode_rows)
        checkpoint_after = _sha256(Path(baseline_path if role == BASELINE_RL else fine_path))
        policy_after = _state_dict_hash(trainer.policy.state_dict())
        report = _write_role_outputs(
            role=role,
            role_root=role_root,
            config=config,
            episode_summaries=summaries,
            step_rows=rows,
            checkpoint_path=baseline_path if role == BASELINE_RL else fine_path,
            checkpoint_sha_before=baseline_sha if role == BASELINE_RL else fine_sha,
            checkpoint_sha_after=checkpoint_after,
            policy_hash_before=policy_hashes.get("loaded_policy_hash"),
            policy_hash_after=policy_after,
            controller_uses_truth=False,
        )
        role_reports[role] = report
        role_summaries[role] = report["feasibility_episode_metrics"]
        role_step_rows[role] = rows

    all_step_rows = [
        row
        for role in POLICY_NAMES
        for row in role_step_rows.get(role, [])
    ]
    all_episode_summaries = [
        row
        for role in POLICY_NAMES
        for row in role_summaries.get(role, [])
    ]
    _write_rows_csv(logs_root / "step_metrics.csv", all_step_rows)
    _write_rows_csv(logs_root / "episode_metrics.csv", all_episode_summaries)
    distance_bins = [
        row
        for policy in POLICY_NAMES
        for row in _distance_bin_summary(all_step_rows, policy)
    ]
    _write_rows_csv(logs_root / "distance_bin_metrics.csv", distance_bins)
    actor_rows = [
        row
        for policy in RL_POLICY_NAMES
        for row in role_step_rows.get(policy, [])
        if "actor_raw_delta_json" in row
    ]
    _write_rows_csv(logs_root / "actor_action_audit.csv", actor_rows)
    actor_aggregates = {
        policy: _aggregate_actor_rows(actor_rows, policy) for policy in RL_POLICY_NAMES
    }
    _write_rows_csv(
        logs_root / "actor_action_aggregate.csv",
        list(actor_aggregates.values()),
    )
    _write_rows_csv(logs_root / "distance_bin_summary.csv", distance_bins)
    plot_paths = _plot_audit_outputs(step_rows=all_step_rows, output_dir=plots_root)
    aggregates = {
        policy: _aggregate_episode_summaries(all_episode_summaries, policy)
        for policy in POLICY_NAMES
    }
    classification = _classify_root_cause(
        aggregates=aggregates,
        actor_aggregates=actor_aggregates,
        config_audit=config_audit,
    )
    _write_json(logs_root / "root_cause_classification.json", _jsonable(classification))
    roles_for_manifest = {
        role: role_reports[role] for role in POLICY_NAMES if role in role_reports
    }
    _write_root_manifest(output_root, roles_for_manifest)
    schema_validation = {
        role: report.get("schema_validation", {})
        for role, report in role_reports.items()
    }
    overall_schema_ok = all(bool(report.get("ok", False)) for report in schema_validation.values())
    read_only_integrity = {
        "all_checkpoint_files_unchanged": all(
            bool(report.get("checkpoint_unchanged", True)) for report in role_reports.values()
        ),
        "all_loaded_policy_states_unchanged": all(
            bool(report.get("policy_unchanged", True)) for report in role_reports.values()
        ),
        "no_training_update_call": all(
            bool(report.get("no_training_update_call", False)) for report in role_reports.values()
        ),
        "no_transition_added": all(
            bool(report.get("no_transition_added", False)) for report in role_reports.values()
        ),
    }
    summary = {
        "dataset_id": "single_uav_rl_equivalent_oracle_audit",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "paired_design": {
            "unit": "normal far-distance test episode",
            "episodes": len(scenarios),
            "same_scenario_passed_to_all_policies": True,
            "same_seed_indices": True,
            "curriculum_used": False,
            "scenario_hashes_recorded": True,
        },
        "configuration_audit": config_audit,
        "policies": role_reports,
        "aggregate_metrics": aggregates,
        "actor_action_aggregate": actor_aggregates,
        "distance_bin_summary": distance_bins,
        "root_cause_classification": classification,
        "read_only_integrity": read_only_integrity,
        "schema_validation": schema_validation,
        "overall_schema_ok": overall_schema_ok,
        "plots": plot_paths,
        "outputs": {
            "data": str(output_root.resolve()),
            "logs": str(logs_root.resolve()),
            "plots": str(plots_root.resolve()),
        },
        "claim_boundary": "read-only observation/action causality diagnostic; no RL training result",
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(logs_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "TRUTH_ORACLE",
    "ESTIMATED_ORACLE",
    "BASELINE_RL",
    "FINE_RL",
    "POLICY_NAMES",
    "EstimatedStateOracleController",
    "evaluate_single_uav_rl_equivalent_oracle_audit",
]
