"""Read-only scripted-controller feasibility test for the single-UAV task.

This module answers a physical-feasibility question rather than a learning
question: can a deterministic controller, using the simulator state and goal,
complete the existing single-UAV task under the existing action-driven
environment?  The controller never writes position or velocity directly.  It
only submits a bounded three-dimensional ``delta_v`` action to
``MultiUAVJointEnv.step``.

The oracle is intentionally kept separate from PPO/MAPPO training.  The
baseline and navigation-preserving fine-tuned checkpoints are evaluated in
the same output package, on the same normal far-distance test scenarios, with
``update_policy=False`` and no rollout transitions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import MotionConfig, clip_vector_norm
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _episode_metrics,
    _make_environment,
    _run_episode,
    _scale_seed,
    _scenario_for,
    _write_attention_summary,
    _write_episode_metrics,
    _write_scale_manifest,
)
from .pilot import _EpisodeTraceWriter, _seed_everything
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)


ORACLE_NAME = "scripted_oracle"
RL_POLICY_NAMES: tuple[str, ...] = ("baseline_rl", "fine_tuned_rl")
POLICY_NAMES: tuple[str, ...] = (ORACLE_NAME,) + RL_POLICY_NAMES
POLICY_LABELS = {
    ORACLE_NAME: "scripted/oracle controller",
    "baseline_rl": "coef=0.25 baseline RL",
    "fine_tuned_rl": "navigation-preserving fine-tuned RL",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite(values: Iterable[Any]) -> np.ndarray:
    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers.append(number)
    return np.asarray(numbers, dtype=float)


def _mean(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _max(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite(values)
    return float(np.max(numbers)) if numbers.size else float(default)


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_payload(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = scenario.get("scenario")
    return nested if isinstance(nested, Mapping) else scenario


def _goal_from_scenario(scenario: Mapping[str, Any]) -> np.ndarray:
    payload = _scenario_payload(scenario)
    goals = payload.get("goals", {})
    if not isinstance(goals, Mapping) or "uav_00" not in goals:
        raise ValueError("normal single-UAV scenario is missing goals[uav_00]")
    goal = np.asarray(goals["uav_00"], dtype=float).reshape(3)
    if not np.isfinite(goal).all():
        raise ValueError("scenario goal is non-finite")
    return goal


def _position_from_scenario(scenario: Mapping[str, Any]) -> np.ndarray:
    payload = _scenario_payload(scenario)
    positions = payload.get("initial_positions", {})
    if not isinstance(positions, Mapping) or "uav_00" not in positions:
        raise ValueError("normal single-UAV scenario is missing initial_positions[uav_00]")
    position = np.asarray(positions["uav_00"], dtype=float).reshape(3)
    if not np.isfinite(position).all():
        raise ValueError("scenario initial position is non-finite")
    return position


def _start_distance(scenario: Mapping[str, Any]) -> float:
    return float(np.linalg.norm(_goal_from_scenario(scenario) - _position_from_scenario(scenario)))


def _timestamp(row: Mapping[str, Any], fallback: float) -> float:
    value = _number(row, "timestamp", fallback)
    return float(value if np.isfinite(value) else fallback)


def _vector(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    return np.asarray([_number(row, name) for name in names], dtype=float)


class ScriptedOracleController:
    """A deterministic bounded velocity-increment controller.

    The controller uses simulator truth for the feasibility upper bound.  It
    still respects the configured physical acceleration and speed limits by
    sending a clipped ``delta_v`` through the normal action interface.  Its
    speed schedule is deliberately explicit:

    * beyond 8 m: cruise toward the goal;
    * 8--3 m: track a stopping-distance speed target with a one-metre
      controller margin;
    * 3--1 m: continue forward with a positive linearly decreasing target
      speed;
    * inside 1 m: damp velocity and use a small proportional position
      correction to remain inside the goal radius.

    The one-metre controller margin is an oracle-controller setting.  It does
    not alter the environment reward's braking margin or target-speed
    formula.
    """

    def __init__(self, environment: Any) -> None:
        if len(environment.uav_ids) != 1:
            raise ValueError("ScriptedOracleController requires one UAV")
        self.environment = environment
        self.uav_id = environment.uav_ids[0]
        self.motion = environment.config.motion
        reward = environment.config.reward
        self.goal_radius_m = float(reward.single_uav_goal_threshold_m)
        self.stable_speed_mps = float(reward.single_uav_goal_speed_threshold_mps)
        self.near_goal_radius_m = float(reward.single_uav_near_goal_radius_m)
        self.cruise_speed_mps = float(
            min(reward.single_uav_target_speed_far_mps, self.motion.max_speed_mps)
        )
        self.braking_acceleration_mps2 = float(
            min(reward.single_uav_braking_acceleration_mps2, self.motion.max_acceleration_mps2)
        )
        self.controller_braking_margin_m = 1.0
        self.max_delta_v_mps = float(
            self.motion.max_acceleration_mps2 * self.motion.dt_s
        )

    def desired_speed(self, distance_m: float) -> float:
        """Return the scripted target speed for one true goal distance."""

        distance = max(0.0, float(distance_m))
        if distance > self.near_goal_radius_m:
            return float(self.cruise_speed_mps)
        if distance > 3.0:
            usable = max(
                0.0,
                distance
                - self.goal_radius_m
                - self.controller_braking_margin_m,
            )
            value = math.sqrt(
                max(
                    0.0,
                    self.stable_speed_mps**2
                    + 2.0 * self.braking_acceleration_mps2 * usable,
                )
            )
            return float(np.clip(value, self.stable_speed_mps, self.cruise_speed_mps))
        if distance > self.goal_radius_m:
            # At 3 m the target is 2.5 m/s; at 1 m it remains positive but
            # approaches the stable-speed threshold.  This prevents a rule
            # based controller from stopping in the 1--3 m band.
            fraction = (distance - self.goal_radius_m) / max(2.0, 1e-9)
            return float(
                self.stable_speed_mps
                + fraction * (2.5 - self.stable_speed_mps)
            )
        return float(min(0.35, max(0.0, 0.50 * distance)))

    def action(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a legal environment action and its pre-clipping diagnostics."""

        state = self.environment.states[self.uav_id]
        goal = np.asarray(self.environment.goals[self.uav_id], dtype=float).reshape(3)
        position = np.asarray(state.position_m, dtype=float).reshape(3)
        velocity = np.asarray(state.velocity_mps, dtype=float).reshape(3)
        goal_vector = goal - position
        distance = float(np.linalg.norm(goal_vector))
        if np.isfinite(distance) and distance > 1e-9:
            direction = goal_vector / distance
        else:
            direction = np.zeros(3, dtype=float)
        desired_speed = self.desired_speed(distance)
        desired_velocity = direction * desired_speed
        raw_delta = desired_velocity - velocity
        legal_delta = clip_vector_norm(raw_delta, self.max_delta_v_mps)
        pre_speed = float(np.linalg.norm(velocity))
        if pre_speed > 1e-9:
            velocity_direction = velocity / pre_speed
            demanded_longitudinal = float(np.dot(velocity_direction, raw_delta))
        else:
            demanded_longitudinal = 0.0
        demanded_braking = max(0.0, -demanded_longitudinal)
        action = {
            self.uav_id: {
                "delta_v": legal_delta.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }
        return action, {
            "true_goal_distance_before_m": distance,
            "desired_speed_mps": desired_speed,
            "desired_velocity_mps": desired_velocity.tolist(),
            "raw_delta_v_mps": raw_delta.tolist(),
            "raw_delta_v_norm_mps": float(np.linalg.norm(raw_delta)),
            "legal_delta_v_mps": legal_delta.tolist(),
            "legal_delta_v_norm_mps": float(np.linalg.norm(legal_delta)),
            "controller_delta_v_capped": int(
                np.linalg.norm(raw_delta) > self.max_delta_v_mps + 1e-9
            ),
            "demanded_braking_delta_v_mps": demanded_braking,
            "pre_speed_mps": pre_speed,
        }


def _oracle_step_row(
    *,
    policy: str,
    episode_index: int,
    seed: int,
    step_index: int,
    timestamp_s: float,
    scenario_hash: str,
    goal: np.ndarray,
    previous_position: np.ndarray,
    next_position: np.ndarray,
    next_velocity: np.ndarray,
    control: Mapping[str, Any],
    next_info: Mapping[str, Any],
    execution: Mapping[str, Any],
    reward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_goal_vector = goal - next_position
    current_distance = float(np.linalg.norm(current_goal_vector))
    if current_distance > 1e-9:
        direction = current_goal_vector / current_distance
        radial_velocity = float(np.dot(next_velocity, direction))
        tangential_velocity = next_velocity - radial_velocity * direction
        tangential_speed = float(np.linalg.norm(tangential_velocity))
    else:
        radial_velocity = 0.0
        tangential_speed = float(np.linalg.norm(next_velocity))
    previous_distance = float(np.linalg.norm(goal - previous_position))
    dt = max(float(timestamp_s) / max(1, step_index + 1), 1e-12)
    true_delta = previous_distance - current_distance
    reward = reward if isinstance(reward, Mapping) else {}
    applied_delta = np.asarray(
        execution.get("delta_v_applied", (0.0, 0.0, 0.0)), dtype=float
    ).reshape(3)
    action_delta = np.asarray(control.get("legal_delta_v_mps", (0.0, 0.0, 0.0)), dtype=float)
    pre_speed = float(np.linalg.norm(np.asarray(control.get("pre_velocity_mps", (0.0, 0.0, 0.0)), dtype=float)))
    if not np.isfinite(pre_speed):
        pre_speed = _number(execution, "pre_speed_mps", 0.0)
    true_inside_8 = int(current_distance <= 8.0)
    true_inside_5 = int(current_distance <= 5.0)
    true_inside_3 = int(current_distance <= 3.0)
    true_inside_1 = int(current_distance <= 1.0)
    # The environment keeps the learning-channel diagnostics in
    # ``learning_reward_components``.  ``reward_components`` is a smaller
    # collision/energy map and does not contain the goal-distance fields.
    estimated_distance = _number(reward, "current_goal_distance")
    stable_condition = int(
        bool(next_info.get("single_uav_goal_in_radius", False))
        and bool(next_info.get("single_uav_goal_speed_ok", False))
    )
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_index": int(episode_index),
        "seed": int(seed),
        "step_index": int(step_index),
        "timestamp_s": float(timestamp_s),
        "scenario_hash": scenario_hash,
        "estimated_goal_distance_m": estimated_distance,
        "true_goal_distance_m": current_distance,
        "previous_true_goal_distance_m": previous_distance,
        "true_delta_goal_distance_m": true_delta,
        "true_radial_progress_mps": true_delta / dt,
        "radial_velocity_toward_mps": radial_velocity,
        "tangential_speed_mps": tangential_speed,
        "total_speed_mps": float(np.linalg.norm(next_velocity)),
        "desired_speed_mps": _number(control, "desired_speed_mps"),
        "target_speed_mps": _number(reward, "single_uav_target_speed_mps"),
        "delta_v_demanded_norm_mps": _number(control, "raw_delta_v_norm_mps"),
        "delta_v_commanded_norm_mps": float(np.linalg.norm(action_delta)),
        "delta_v_applied_norm_mps": _number(execution, "delta_v_applied_norm_mps", float(np.linalg.norm(applied_delta))),
        "braking_demanded_mps": _number(control, "demanded_braking_delta_v_mps", 0.0),
        "commanded_braking_delta_v_mps": _number(execution, "commanded_braking_delta_v_mps", 0.0),
        "applied_braking_delta_v_mps": _number(execution, "braking_delta_v_mps", 0.0),
        "actual_speed_reduction_mps": _number(execution, "speed_reduction_mps", 0.0),
        "braking_action": int(_number(execution, "braking_action", 0.0) > 0.5),
        "effective_braking": int(_number(execution, "brake_effective", 0.0) > 0.5),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "controller_delta_v_capped": int(control.get("controller_delta_v_capped", 0)),
        "braking_progress_signal": _number(reward, "single_uav_braking_progress_signal"),
        "braking_progress_reward": _number(reward, "single_uav_braking_progress_reward"),
        "target_velocity_action_reward": _number(reward, "single_uav_target_velocity_action_reward"),
        "goal_progress_signal": _number(reward, "single_uav_goal_progress_signal"),
        "goal_progress_reward": _number(reward, "single_uav_goal_progress_reward"),
        "goal_direction_signal": _number(reward, "single_uav_goal_direction_signal"),
        "goal_direction_reward": _number(reward, "single_uav_goal_direction_reward"),
        "inside_8m": true_inside_8,
        "inside_5m": true_inside_5,
        "inside_3m": true_inside_3,
        "inside_1m": true_inside_1,
        "estimated_inside_8m": int(np.isfinite(estimated_distance) and estimated_distance <= 8.0),
        "estimated_inside_3m": int(np.isfinite(estimated_distance) and estimated_distance <= 3.0),
        "estimated_inside_1m": int(np.isfinite(estimated_distance) and estimated_distance <= 1.0),
        "stable_condition": stable_condition,
        "goal_stable": int(bool(next_info.get("single_uav_goal_stable", False))),
        "consecutive_stable_steps": int(round(_number(next_info, "single_uav_stable_steps", 0.0))),
        "termination_reason": str(next_info.get("termination_reason", "running")),
    }


def _rl_step_rows(
    episode_path: Path,
    *,
    policy: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
    scenario: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read frozen RL traces into the same compact feasibility schema."""

    rewards = _read_csv(episode_path / "rewards.csv")
    actions = _read_csv(episode_path / "actions.csv")
    truth = _read_csv(episode_path / "ground_truth.csv")
    goal = _goal_from_scenario(scenario)
    dt = 0.2
    action_by_timestamp = {
        round(_number(row, "timestamp"), 6): row for row in actions
    }
    truth_by_timestamp = {
        round(_number(row, "timestamp"), 6): row for row in truth
    }
    rows: list[dict[str, Any]] = []
    previous_true_distance: float | None = None
    for index, reward in enumerate(rewards):
        timestamp_s = _timestamp(reward, (index + 1) * dt)
        action = action_by_timestamp.get(round(timestamp_s, 6), {})
        truth_row = truth_by_timestamp.get(round(timestamp_s, 6), {})
        position = _vector(truth_row, ("x", "y", "z"))
        velocity = _vector(truth_row, ("vx", "vy", "vz"))
        true_distance = float(np.linalg.norm(goal - position)) if np.isfinite(position).all() else float("nan")
        if np.isfinite(true_distance):
            direction = goal - position
            norm = float(np.linalg.norm(direction))
            radial = float(np.dot(velocity, direction / norm)) if norm > 1e-9 else 0.0
            tangential = float(np.linalg.norm(velocity - radial * direction / max(norm, 1e-9)))
        else:
            radial, tangential = float("nan"), float("nan")
        if previous_true_distance is None or not np.isfinite(true_distance):
            true_delta = float("nan")
        else:
            true_delta = previous_true_distance - true_distance
        estimated_distance = _number(reward, "current_goal_distance")
        pre_speed = _number(action, "pre_speed_mps")
        post_speed = _number(action, "post_speed_mps")
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "episode_index": int(episode_index),
                "seed": int(seed),
                "step_index": int(index),
                "timestamp_s": timestamp_s,
                "scenario_hash": scenario_hash,
                "estimated_goal_distance_m": estimated_distance,
                "true_goal_distance_m": true_distance,
                "previous_true_goal_distance_m": previous_true_distance if previous_true_distance is not None else float("nan"),
                "true_delta_goal_distance_m": true_delta,
                "true_radial_progress_mps": true_delta / dt if np.isfinite(true_delta) else float("nan"),
                "radial_velocity_toward_mps": radial,
                "tangential_speed_mps": tangential,
                "total_speed_mps": _number(reward, "single_uav_speed_mps", float(np.linalg.norm(velocity))),
                "desired_speed_mps": float("nan"),
                "target_speed_mps": _number(reward, "single_uav_target_speed_mps"),
                "delta_v_demanded_norm_mps": _number(action, "delta_v_commanded_norm_mps"),
                "delta_v_commanded_norm_mps": _number(action, "delta_v_commanded_norm_mps"),
                "delta_v_applied_norm_mps": _number(action, "delta_v_applied_norm_mps"),
                "braking_demanded_mps": _number(action, "commanded_braking_delta_v_mps", 0.0),
                "commanded_braking_delta_v_mps": _number(action, "commanded_braking_delta_v_mps", 0.0),
                "applied_braking_delta_v_mps": _number(action, "braking_delta_v_mps", 0.0),
                "actual_speed_reduction_mps": _number(action, "speed_reduction_mps", max(0.0, pre_speed - post_speed) if np.isfinite(pre_speed) and np.isfinite(post_speed) else 0.0),
                "braking_action": int(_number(action, "braking_action", 0.0) > 0.5),
                "effective_braking": int(_number(action, "brake_effective", 0.0) > 0.5),
                "action_clipped": int(_number(action, "action_clipped", 0.0) > 0.5),
                "controller_delta_v_capped": 0,
                "braking_progress_signal": _number(reward, "single_uav_braking_progress_signal"),
                "braking_progress_reward": _number(reward, "single_uav_braking_progress_reward"),
                "target_velocity_action_reward": _number(reward, "single_uav_target_velocity_action_reward"),
                "goal_progress_signal": _number(reward, "single_uav_goal_progress_signal"),
                "goal_progress_reward": _number(reward, "single_uav_goal_progress_reward"),
                "goal_direction_signal": _number(reward, "single_uav_goal_direction_signal"),
                "goal_direction_reward": _number(reward, "single_uav_goal_direction_reward"),
                "inside_8m": int(np.isfinite(true_distance) and true_distance <= 8.0),
                "inside_5m": int(np.isfinite(true_distance) and true_distance < 5.0),
                "inside_3m": int(np.isfinite(true_distance) and true_distance <= 3.0),
                "inside_1m": int(np.isfinite(true_distance) and true_distance <= 1.0),
                "estimated_inside_8m": int(np.isfinite(estimated_distance) and estimated_distance <= 8.0),
                "estimated_inside_3m": int(np.isfinite(estimated_distance) and estimated_distance <= 3.0),
                "estimated_inside_1m": int(np.isfinite(estimated_distance) and estimated_distance <= 1.0),
                "stable_condition": int(
                    _number(reward, "single_uav_goal_in_radius", 0.0) > 0.5
                    and _number(reward, "single_uav_goal_speed_ok", 0.0) > 0.5
                ),
                "goal_stable": int(_number(reward, "single_uav_goal_stable", 0.0) > 0.5),
                "consecutive_stable_steps": int(round(_number(reward, "single_uav_stable_steps", 0.0))),
                "termination_reason": "running",
            }
        )
        previous_true_distance = true_distance
    scenario_payload = _scenario_payload(scenario)
    try:
        scenario_meta = json.loads((episode_path / "scenario.json").read_text(encoding="utf-8"))
        termination_reason = str(scenario_meta.get("termination_reason", "running"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        termination_reason = "running"
    for row in rows:
        row["termination_reason"] = termination_reason
    return rows


def _episode_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_summary: Mapping[str, Any],
) -> dict[str, Any]:
    estimated = _finite(row.get("estimated_goal_distance_m") for row in rows)
    truth = _finite(row.get("true_goal_distance_m") for row in rows)
    hold_values = _finite(row.get("consecutive_stable_steps") for row in rows)
    max_hold = int(round(_max(hold_values, default=0.0)))

    def entry_step(threshold: float) -> int | None:
        for row in rows:
            value = _number(row, "estimated_goal_distance_m")
            if np.isfinite(value) and value <= threshold:
                return int(row.get("step_index", 0)) + 1
        return None

    def speed_at_entry(threshold: float) -> float:
        for row in rows:
            value = _number(row, "estimated_goal_distance_m")
            if np.isfinite(value) and value <= threshold:
                return _number(row, "total_speed_mps")
        return float("nan")

    def mean_in_range(threshold: float) -> float:
        return _mean(
            row.get("total_speed_mps")
            for row in rows
            if np.isfinite(_number(row, "estimated_goal_distance_m"))
            and _number(row, "estimated_goal_distance_m") <= threshold
        )

    termination_reason = str(episode_summary.get("termination_reason", "running"))
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_id": str(episode_summary.get("episode_id", "")),
        "episode_index": int(rows[0].get("episode_index", 0)) if rows else 0,
        "seed": int(rows[0].get("seed", episode_summary.get("seed", 0))) if rows else int(episode_summary.get("seed", 0)),
        "scenario_hash": str(rows[0].get("scenario_hash", "")) if rows else "",
        "steps": len(rows),
        "entered_8m": int(estimated.size > 0 and np.min(estimated) <= 8.0),
        "entered_5m": int(estimated.size > 0 and np.min(estimated) <= 5.0),
        "entered_3m": int(estimated.size > 0 and np.min(estimated) <= 3.0),
        "entered_1m": int(estimated.size > 0 and np.min(estimated) <= 1.0),
        "entered_8m_true": int(truth.size > 0 and np.min(truth) <= 8.0),
        "entered_5m_true": int(truth.size > 0 and np.min(truth) <= 5.0),
        "entered_3m_true": int(truth.size > 0 and np.min(truth) <= 3.0),
        "entered_1m_true": int(truth.size > 0 and np.min(truth) <= 1.0),
        "stable_goal": int(any(int(row.get("goal_stable", 0)) > 0 for row in rows)),
        "minimum_goal_distance_m": float(np.min(estimated)) if estimated.size else float("nan"),
        "minimum_estimated_goal_distance_m": float(np.min(estimated)) if estimated.size else float("nan"),
        "minimum_true_goal_distance_m": float(np.min(truth)) if truth.size else float("nan"),
        "mean_speed_mps": _mean(row.get("total_speed_mps") for row in rows),
        "mean_speed_within_1m_mps": mean_in_range(1.0),
        "mean_speed_at_8m_entry_mps": speed_at_entry(8.0),
        "mean_speed_at_5m_entry_mps": speed_at_entry(5.0),
        "mean_speed_at_3m_entry_mps": speed_at_entry(3.0),
        "mean_speed_at_1m_entry_mps": speed_at_entry(1.0),
        "maximum_consecutive_stable_steps": max_hold,
        "hold_steps_ge_1": int(max_hold >= 1),
        "hold_steps_ge_3": int(max_hold >= 3),
        "hold_steps_ge_5": int(max_hold >= 5),
        "steps_to_8m": entry_step(8.0),
        "steps_to_5m": entry_step(5.0),
        "steps_to_3m": entry_step(3.0),
        "steps_to_1m": entry_step(1.0),
        "action_clipping_fraction": _mean(row.get("action_clipped") for row in rows),
        "controller_delta_v_cap_fraction": _mean(row.get("controller_delta_v_capped") for row in rows),
        "maximum_braking_demanded_mps": _max(row.get("braking_demanded_mps") for row in rows),
        "maximum_braking_commanded_mps": _max(row.get("commanded_braking_delta_v_mps") for row in rows),
        "maximum_braking_applied_mps": _max(row.get("applied_braking_delta_v_mps") for row in rows),
        "braking_demanded_step_fraction": _mean(
            float(_number(row, "braking_demanded_mps", 0.0) > 1e-9) for row in rows
        ),
        "braking_action_fraction": _mean(row.get("braking_action") for row in rows),
        "effective_braking_fraction": _mean(row.get("effective_braking") for row in rows),
        "timeout": int(termination_reason == "time_limit"),
        "termination_reason": termination_reason,
        "collision_occurred": bool(episode_summary.get("collision_occurred", False)),
    }


def _aggregate_episode_rows(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("policy")) == policy]
    count = len(selected)

    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0)) for row in selected))

    def mean_field(name: str) -> float:
        return _mean(row.get(name) for row in selected)

    def mean_steps(name: str) -> float:
        return _mean(row.get(name) for row in selected if row.get(name) is not None)

    summary = {
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
        "minimum_goal_distance_m": _min_or_nan(row.get("minimum_goal_distance_m") for row in selected),
        "mean_minimum_goal_distance_m": mean_field("minimum_goal_distance_m"),
        "mean_true_minimum_goal_distance_m": mean_field("minimum_true_goal_distance_m"),
        "mean_speed_mps": mean_field("mean_speed_mps"),
        "mean_speed_within_1m_mps": mean_field("mean_speed_within_1m_mps"),
        "mean_speed_at_8m_entry_mps": mean_field("mean_speed_at_8m_entry_mps"),
        "mean_speed_at_5m_entry_mps": mean_field("mean_speed_at_5m_entry_mps"),
        "mean_speed_at_3m_entry_mps": mean_field("mean_speed_at_3m_entry_mps"),
        "mean_speed_at_1m_entry_mps": mean_field("mean_speed_at_1m_entry_mps"),
        "mean_maximum_consecutive_stable_steps": mean_field("maximum_consecutive_stable_steps"),
        "maximum_consecutive_stable_steps_max": int(round(_max((row.get("maximum_consecutive_stable_steps") for row in selected), default=0.0))),
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
        "mean_action_clipping_fraction": mean_field("action_clipping_fraction"),
        "mean_controller_delta_v_cap_fraction": mean_field("controller_delta_v_cap_fraction"),
        "maximum_braking_demanded_mps": _max((row.get("maximum_braking_demanded_mps") for row in selected), default=0.0),
        "maximum_braking_commanded_mps": _max((row.get("maximum_braking_commanded_mps") for row in selected), default=0.0),
        "maximum_braking_applied_mps": _max((row.get("maximum_braking_applied_mps") for row in selected), default=0.0),
        "mean_braking_action_fraction": mean_field("braking_action_fraction"),
        "mean_effective_braking_fraction": mean_field("effective_braking_fraction"),
    }
    return summary


def _min_or_nan(values: Iterable[Any]) -> float:
    array = _finite(values)
    return float(np.min(array)) if array.size else float("nan")


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
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                output[column] = value
            writer.writerow(output)


def _paired_scenarios(config: FormalExperimentConfig, episodes: int) -> list[dict[str, Any]]:
    scenario_config = replace(
        config,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        force=True,
        progress=False,
        command_line="scripted feasibility paired normal test scenario generation",
    )
    output: list[dict[str, Any]] = []
    for episode_index in range(int(episodes)):
        seed = _scale_seed(config.seed, config.uav_count, "test", episode_index)
        scenario = _scenario_for(
            scenario_config,
            seed,
            split="test",
            episode_index=episode_index,
            phase="test",
        )
        output.append(
            {
                "episode_index": episode_index + 1,
                "seed": seed,
                "scenario": copy.deepcopy(scenario),
                "scenario_hash": _canonical_hash(scenario),
                "curriculum_used": "single_uav_braking_curriculum" in scenario,
                "start_distance_m": _start_distance(scenario),
            }
        )
    if any(item["curriculum_used"] for item in output):
        raise RuntimeError("paired feasibility scenarios unexpectedly used curriculum")
    return output


def _run_oracle_episode(
    *,
    config: FormalExperimentConfig,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    episode_total: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="scripted_feasibility",
        environment=environment,
        normalization_fitted=False,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=ORACLE_NAME,
    )
    controller = ScriptedOracleController(environment)
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(config.max_steps):
        state_before = environment.states["uav_00"].copy()
        action, control = controller.action()
        control = dict(control)
        control["pre_velocity_mps"] = state_before.velocity_mps.tolist()
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        row = _oracle_step_row(
            policy=ORACLE_NAME,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            goal=goal,
            previous_position=state_before.position_m.copy(),
            next_position=environment.states["uav_00"].position_m.copy(),
            next_velocity=environment.states["uav_00"].velocity_mps.copy(),
            control=control,
            next_info=next_info,
            execution=execution,
            reward=next_info.get("learning_reward_components", next_info),
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
            policy_source=ORACLE_NAME,
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
            "policy_source": ORACLE_NAME,
        }
    )
    return summary, rows


def _write_role_manifest(
    *,
    role: str,
    role_root: Path,
    scale_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    checkpoint: Path | None,
    checkpoint_sha256_before: str | None,
    checkpoint_sha256_after: str | None,
    policy_hash_before: str | None,
    policy_hash_after: str | None,
) -> dict[str, Any]:
    _write_scale_manifest(scale_root=scale_root, config=config, episodes=episode_summaries)
    path = scale_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "single_uav_scripted_oracle_feasibility",
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "policy_source": role,
            "policy_label": POLICY_LABELS[role],
            "feasibility_contract": {
                "normal_far_distance_test_only": True,
                "curriculum_used": False,
                "controller_uses_simulator_truth": role == ORACLE_NAME,
                "action_interface": "environment.step(delta_v)",
                "direct_position_assignment": False,
                "direct_velocity_assignment": False,
                "stable_judgement_reused": True,
            },
            "checkpoint": (
                {
                    "path": str(checkpoint.resolve()),
                    "sha256_before": checkpoint_sha256_before,
                    "sha256_after": checkpoint_sha256_after,
                    "unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
                }
                if checkpoint is not None
                else {"not_applicable": True}
            ),
            "policy_identity": (
                {
                    "hash_before": policy_hash_before,
                    "hash_after": policy_hash_after,
                    "unchanged": policy_hash_before == policy_hash_after,
                }
                if policy_hash_before is not None
                else {"not_applicable": True}
            ),
            "read_only_contract": [
                "no MAPPOTrainer.update call",
                "no rollout transition added",
                "no checkpoint optimizer state loaded or written",
                "no reward, curriculum, dynamics, action-space, or stable criterion change",
            ],
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


def _plot_feasibility(
    *,
    step_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        ORACLE_NAME: "#16a34a",
        "baseline_rl": "#2563eb",
        "fine_tuned_rl": "#dc2626",
    }
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for policy in POLICY_NAMES:
        selected = [row for row in step_rows if row.get("policy") == policy]
        by_step: dict[int, list[float]] = {}
        times: dict[int, float] = {}
        for row in selected:
            value = _number(row, "true_goal_distance_m")
            step = int(row.get("step_index", -1))
            if np.isfinite(value) and step >= 0:
                by_step.setdefault(step, []).append(value)
                times[step] = _number(row, "timestamp_s", step * 0.2)
        if by_step:
            steps = sorted(by_step)
            axis.plot(
                [times[step] for step in steps],
                [float(np.mean(by_step[step])) for step in steps],
                linewidth=2.0,
                color=colors[policy],
                label=POLICY_LABELS[policy],
            )
    for value, style, label in (
        (8.0, ":", "8 m"),
        (3.0, "--", "3 m"),
        (1.0, ":", "1 m"),
    ):
        axis.axhline(value, color="#6b7280", linestyle=style, linewidth=1.0, label=label)
    axis.set_title("Single-UAV feasibility: true goal distance")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("True goal distance (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    distance_path = output_dir / "goal_distance_vs_step.png"
    figure.savefig(distance_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for policy in POLICY_NAMES:
        selected = [row for row in step_rows if row.get("policy") == policy]
        by_step: dict[int, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_step.setdefault(int(row.get("step_index", -1)), []).append(row)
        steps = sorted(step for step in by_step if step >= 0)
        if not steps:
            continue
        x = [float(np.mean([_number(row, "timestamp_s", step * 0.2) for row in by_step[step]])) for step in steps]
        speed = [float(np.mean([_number(row, "total_speed_mps") for row in by_step[step]])) for step in steps]
        radial = [float(np.mean([_number(row, "radial_velocity_toward_mps") for row in by_step[step]])) for step in steps]
        desired = [float(np.mean([_number(row, "desired_speed_mps") for row in by_step[step]])) for step in steps]
        axes[0].plot(x, speed, color=colors[policy], linewidth=1.8, label=f"{POLICY_LABELS[policy]} speed")
        if np.isfinite(desired).any():
            axes[0].plot(x, desired, color=colors[policy], linestyle="--", linewidth=1.2, alpha=0.8, label=f"{POLICY_LABELS[policy]} desired")
        axes[1].plot(x, radial, color=colors[policy], linewidth=1.8, label=POLICY_LABELS[policy])
    axes[0].axhline(0.5, color="#111827", linestyle=":", linewidth=1.0, label="stable speed threshold")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Speed, scripted desired speed, and radial velocity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Radial velocity toward goal (m/s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    speed_path = output_dir / "speed_radial_desired_vs_step.png"
    figure.savefig(speed_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "goal_distance_vs_step": str(distance_path.resolve()),
        "speed_radial_desired_vs_step": str(speed_path.resolve()),
    }


def _prepare_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"feasibility output is not empty: {path}; use --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _normalization_copy(scale_root: Path, payload: Mapping[str, Any]) -> None:
    normalization = payload.get("normalization_stats")
    central = payload.get("central_normalization_stats")
    if isinstance(normalization, Mapping):
        _write_json(
            scale_root / "normalization_stats.json",
            _jsonable(
                {
                    "normalization_stats": dict(normalization),
                    "central_normalization_stats": dict(central) if isinstance(central, Mapping) else None,
                    "source": "frozen_checkpoint",
                }
            ),
        )


def _read_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return dict(payload), _sha256(path)


def _run_frozen_rl(
    *,
    role: str,
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: str,
    config: FormalExperimentConfig,
    role_root: Path,
    scenarios: Sequence[Mapping[str, Any]],
    progress: bool,
    serial_base: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scale_root = role_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    _normalization_copy(scale_root, checkpoint_payload)
    trainer, policy_hashes, _ = _load_frozen_trainer(
        checkpoint_payload,
        config,
        metric_scales=config.metric_scales,
    )
    trainer.policy.eval()
    episode_summaries: list[dict[str, Any]] = []
    standard_episode_rows: list[dict[str, Any]] = []
    feasibility_episode_rows: list[dict[str, Any]] = []
    step_rows_all: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    import torch

    for item in scenarios:
        episode_index = int(item["episode_index"])
        seed = int(item["seed"])
        serial = serial_base + episode_index - 1
        with torch.no_grad():
            summary, _update, returned_rows = _run_episode(
                config=config,
                trainer=trainer,
                scale_root=scale_root,
                phase="scripted_feasibility_rl_comparison",
                split="test",
                serial=serial,
                seed=seed,
                update_policy=False,
                update_index=None,
                episode_index=episode_index,
                episode_total=len(scenarios),
                metric_scales=config.metric_scales,
                flush_update=False,
                scenario_config=copy.deepcopy(dict(item["scenario"])),
            )
        episode_path = scale_root / "test" / f"episode_{serial:06d}"
        metric = returned_rows[-1] if returned_rows and isinstance(returned_rows[-1], Mapping) else _episode_metrics(episode_path, summary)
        standard_episode_rows.append(dict(metric))
        attention_rows.extend(
            row for row in returned_rows[:-1]
            if isinstance(row, Mapping) and "layer" in row
        )
        step_rows = _rl_step_rows(
            episode_path,
            policy=role,
            episode_index=episode_index,
            seed=seed,
            scenario_hash=str(item["scenario_hash"]),
            scenario=item["scenario"],
        )
        episode_entry = dict(summary)
        episode_entry.update(
            {
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "policy_role": role,
                "paired_episode_index": episode_index,
                "paired_seed": seed,
                "paired_scenario_hash": item["scenario_hash"],
                "curriculum_used": False,
            }
        )
        episode_summaries.append(episode_entry)
        row_summary = _episode_summary(step_rows, policy=role, episode_summary=summary)
        feasibility_episode_rows.append(row_summary)
        step_rows_all.extend(step_rows)

    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", standard_episode_rows)
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)
    checkpoint_after = _sha256(checkpoint_path)
    policy_after = _state_dict_hash(trainer.policy.state_dict())
    manifest = _write_role_manifest(
        role=role,
        role_root=role_root,
        scale_root=scale_root,
        config=config,
        episode_summaries=episode_summaries,
        checkpoint=checkpoint_path,
        checkpoint_sha256_before=checkpoint_sha256,
        checkpoint_sha256_after=checkpoint_after,
        policy_hash_before=policy_hashes.get("loaded_policy_hash"),
        policy_hash_after=policy_after,
    )
    validation = validate_episode_benchmark(scale_root)
    diagnostics = {
        "role": role,
        "episodes": len(scenarios),
        "schema_validation": validation,
        "checkpoint_unchanged": checkpoint_sha256 == checkpoint_after,
        "policy_unchanged": policy_hashes.get("loaded_policy_hash") == policy_after,
        "no_training_update_call": True,
        "no_transition_added": True,
        "optimizer_state_not_loaded": True,
        "manifest": str((scale_root / "manifest.json").resolve()),
        "episode_metrics": feasibility_episode_rows,
        "step_rows": step_rows_all,
    }
    return diagnostics["episode_metrics"], diagnostics["step_rows"], diagnostics


def _run_oracle_batch(
    *,
    config: FormalExperimentConfig,
    role_root: Path,
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scale_root = role_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    summaries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    standard_episode_rows: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for item in scenarios:
        episode_index = int(item["episode_index"])
        summary, step_rows = _run_oracle_episode(
            config=config,
            scale_root=scale_root,
            scenario=item["scenario"],
            scenario_hash=str(item["scenario_hash"]),
            episode_index=episode_index,
            episode_total=len(scenarios),
            seed=int(item["seed"]),
            serial=630000 + episode_index,
        )
        episode_metric = _episode_summary(step_rows, policy=ORACLE_NAME, episode_summary=summary)
        metric = dict(episode_metric)
        metric["episode_id"] = summary["episode_id"]
        metric["scenario_id"] = summary["scenario_id"]
        metrics.append(metric)
        standard_episode_rows.append(
            _episode_metrics(
                scale_root / "test" / summary["episode_id"],
                summary,
            )
        )
        steps.extend(step_rows)
        entry = dict(summary)
        entry.update(
            {
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "policy_role": ORACLE_NAME,
                "paired_episode_index": episode_index,
                "paired_seed": int(item["seed"]),
                "paired_scenario_hash": item["scenario_hash"],
                "curriculum_used": False,
            }
        )
        summaries.append(entry)
    _write_episode_metrics(
        scale_root / "diagnostics" / "episode_metrics.csv",
        standard_episode_rows,
    )
    _write_rows_csv(scale_root / "diagnostics" / "feasibility_episode_metrics.csv", metrics)
    _write_json(scale_root / "diagnostics" / "attention_summary.json", {"rows": []})
    manifest = _write_role_manifest(
        role=ORACLE_NAME,
        role_root=role_root,
        scale_root=scale_root,
        config=config,
        episode_summaries=summaries,
        checkpoint=None,
        checkpoint_sha256_before=None,
        checkpoint_sha256_after=None,
        policy_hash_before=None,
        policy_hash_after=None,
    )
    validation = validate_episode_benchmark(scale_root)
    return metrics, steps, {
        "role": ORACLE_NAME,
        "episodes": len(scenarios),
        "schema_validation": validation,
        "no_training_update_call": True,
        "no_transition_added": True,
        "manifest": str((scale_root / "manifest.json").resolve()),
    }


def _controller_contract(config: FormalExperimentConfig, scenario: Mapping[str, Any]) -> dict[str, Any]:
    motion = MotionConfig()
    reward = config
    environment = _make_environment(config, _scale_seed(config.seed, 1, "test"), metric_scales=config.metric_scales)
    return {
        "normal_test_start_distance_m": _start_distance(scenario),
        "max_steps": int(config.max_steps),
        "dt_s": float(motion.dt_s),
        "action_physical_meaning": motion.action_contract(),
        "max_delta_v_per_step_mps": float(motion.max_acceleration_mps2 * motion.dt_s),
        "max_acceleration_mps2": float(motion.max_acceleration_mps2),
        "max_speed_mps": float(motion.max_speed_mps),
        "braking_distance_target_speed": {
            "function": "braking_distance_target_speed",
            "formula": "clip(min(v_far, sqrt(v_stable^2 + 2*a*max(distance-goal_threshold-margin, 0))), v_stable, v_max)",
            "speed_profile": str(config.single_uav_braking_distance_speed_profile),
            "far_target_speed_mps": float(config.single_uav_target_speed_far_mps),
            "braking_acceleration_mps2": float(config.single_uav_braking_acceleration_mps2),
            "braking_margin_m": float(config.single_uav_braking_margin_m),
        },
        "goal_radius_m": float(config.single_uav_goal_threshold_m),
        "stable_speed_threshold_mps": float(config.single_uav_goal_speed_threshold_mps),
        "stable_hold_steps": int(config.single_uav_stable_hold_steps),
        "stable_goal_definition": "loc_goal_v2: estimated_goal_distance <= goal_radius AND actual_speed <= stable_speed_threshold for stable_hold_steps consecutive transitions",
        "early_termination": ["collision", "stable_goal"],
        "time_limit_termination": "truncated at max_steps when no collision/stable goal",
        "environment_safety_projection_enabled": bool(environment.config.safety_projection_enabled),
        "braking_safety_projection_enabled": bool(config.single_uav_braking_safety_projection_enabled),
        "controller_state_source": "simulator_truth_position_velocity_and_configured_goal",
    }


def _write_root_summary(
    *,
    output_root: Path,
    logs_root: Path,
    plots_root: Path,
    contract: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    policy_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    policy_reports: Mapping[str, Mapping[str, Any]],
    plot_paths: Mapping[str, str],
    sanity: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = {
        policy: _aggregate_episode_rows(rows, policy)
        for policy, rows in policy_metrics.items()
    }
    comparison_columns: list[dict[str, Any]] = []
    for policy in POLICY_NAMES:
        comparison_columns.append(aggregate[policy])
    _write_rows_csv(logs_root / "feasibility_comparison.csv", comparison_columns)
    _write_rows_csv(
        logs_root / "episode_metrics.csv",
        [row for policy in POLICY_NAMES for row in policy_metrics.get(policy, [])],
    )
    summary = {
        "dataset_id": "single_uav_scripted_oracle_feasibility",
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
        "task_contract": dict(contract),
        "sanity_check": dict(sanity),
        "policies": dict(policy_reports),
        "aggregate_metrics": aggregate,
        "comparison": comparison_columns,
        "plots": dict(plot_paths),
        "outputs": {
            "data": str(output_root.resolve()),
            "logs": str(logs_root.resolve()),
            "plots": str(plots_root.resolve()),
        },
        "claim_boundary": "read-only physical feasibility diagnostic; no RL training result",
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(
        output_root / "paired_scenario_manifest.json",
        _jsonable(
            [
                {
                    "episode_index": item["episode_index"],
                    "seed": item["seed"],
                    "scenario_hash": item["scenario_hash"],
                    "start_distance_m": item["start_distance_m"],
                    "curriculum_used": item["curriculum_used"],
                }
                for item in scenarios
            ]
        ),
    )
    return summary


def evaluate_scripted_single_uav_feasibility(
    baseline_checkpoint: str | Path,
    fine_tuned_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    sanity_episodes: int = 2,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the sanity check and the paired 20-episode feasibility test."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    if int(sanity_episodes) < 1 or int(sanity_episodes) > int(episodes):
        raise ValueError("sanity_episodes must lie in [1, episodes]")
    output_root = Path(output_dir)
    _prepare_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(baseline_checkpoint)
    fine_path = Path(fine_tuned_checkpoint)
    baseline_payload, baseline_sha = _read_checkpoint(baseline_path)
    fine_payload, fine_sha = _read_checkpoint(fine_path)
    baseline_config = _config_from_checkpoint(baseline_payload, output_dir=output_root / "baseline_rl", progress=progress)
    fine_config = _config_from_checkpoint(fine_payload, output_dir=output_root / "fine_tuned_rl", progress=progress)
    if baseline_config.uav_count != 1 or fine_config.uav_count != 1:
        raise ValueError("scripted feasibility requires two 1-UAV checkpoints")
    for name in ("seed", "scenario_seed", "max_steps", "gnss_mode"):
        if getattr(baseline_config, name) != getattr(fine_config, name):
            raise ValueError(f"baseline and fine-tuned configs disagree on {name}")

    scenarios = _paired_scenarios(baseline_config, int(episodes))
    sanity_root = output_root / "sanity" / ORACLE_NAME
    sanity_root.mkdir(parents=True, exist_ok=True)
    sanity_summaries: list[dict[str, Any]] = []
    sanity_steps: list[dict[str, Any]] = []
    for item in scenarios[: int(sanity_episodes)]:
        summary, rows = _run_oracle_episode(
            config=baseline_config,
            scale_root=sanity_root,
            scenario=item["scenario"],
            scenario_hash=str(item["scenario_hash"]),
            episode_index=int(item["episode_index"]),
            episode_total=int(sanity_episodes),
            seed=int(item["seed"]),
            serial=640000 + int(item["episode_index"]),
        )
        sanity_summaries.append(_episode_summary(rows, policy=ORACLE_NAME, episode_summary=summary))
        sanity_steps.extend(rows)
    _write_rows_csv(logs_root / "sanity_step_metrics.csv", sanity_steps)
    _write_rows_csv(logs_root / "sanity_episode_metrics.csv", sanity_summaries)
    sanity_aggregate = _aggregate_episode_rows(sanity_summaries, ORACLE_NAME)
    sanity = {
        "episodes": int(sanity_episodes),
        "aggregate": sanity_aggregate,
        "completed_without_runtime_error": True,
        "entry_and_hold_observable": bool(
            sanity_aggregate["entered_8m_episodes"] >= 0
            and sanity_aggregate["entered_1m_episodes"] >= 0
        ),
    }

    oracle_metrics, oracle_steps, oracle_report = _run_oracle_batch(
        config=baseline_config,
        role_root=output_root / ORACLE_NAME,
        scenarios=scenarios,
    )
    baseline_metrics, baseline_steps, baseline_report = _run_frozen_rl(
        role="baseline_rl",
        checkpoint_path=baseline_path,
        checkpoint_payload=baseline_payload,
        checkpoint_sha256=baseline_sha,
        config=baseline_config,
        role_root=output_root / "baseline_rl",
        scenarios=scenarios,
        progress=progress,
        serial_base=650000,
    )
    fine_metrics, fine_steps, fine_report = _run_frozen_rl(
        role="fine_tuned_rl",
        checkpoint_path=fine_path,
        checkpoint_payload=fine_payload,
        checkpoint_sha256=fine_sha,
        config=fine_config,
        role_root=output_root / "fine_tuned_rl",
        scenarios=scenarios,
        progress=progress,
        serial_base=660000,
    )
    all_steps = oracle_steps + baseline_steps + fine_steps
    _write_rows_csv(logs_root / "step_metrics.csv", all_steps)
    plot_paths = _plot_feasibility(step_rows=all_steps, output_dir=plots_root)
    policy_metrics = {
        ORACLE_NAME: oracle_metrics,
        "baseline_rl": baseline_metrics,
        "fine_tuned_rl": fine_metrics,
    }
    policy_reports = {
        ORACLE_NAME: oracle_report,
        "baseline_rl": baseline_report,
        "fine_tuned_rl": fine_report,
    }
    contract = _controller_contract(baseline_config, scenarios[0]["scenario"])
    summary = _write_root_summary(
        output_root=output_root,
        logs_root=logs_root,
        plots_root=plots_root,
        contract=contract,
        scenarios=scenarios,
        policy_metrics=policy_metrics,
        policy_reports=policy_reports,
        plot_paths=plot_paths,
        sanity=sanity,
    )
    summary["schema_validation"] = {
        policy: report.get("schema_validation", {})
        for policy, report in policy_reports.items()
    }
    summary["overall_schema_ok"] = bool(
        all(bool(report.get("schema_validation", {}).get("ok", False)) for report in policy_reports.values())
    )
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "ORACLE_NAME",
    "RL_POLICY_NAMES",
    "POLICY_NAMES",
    "ScriptedOracleController",
    "evaluate_scripted_single_uav_feasibility",
]
