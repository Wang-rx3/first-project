"""Fresh single-UAV PPO baseline preparation contract.

This module is deliberately a preparation/factory layer.  It does not train,
collect rollouts, call :meth:`MultiUAVJointEnv.step`, or fit persisted
statistics.  The environment subclass is kept here so the strict success
definition is isolated from the historical multi-UAV implementation.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..learning.mappo import MAPPOTrainer, central_state_vector_dim
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from ..simulation.motion_model import MotionConfig, UAVKinematicState, apply_velocity_increment
from ..simulation.obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    is_physical_obstacle,
    point_obstacle_distance,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
)


BASELINE_ID = "single_uav_ppo_baseline_v1"
EXPECTED_NODE_FEATURE_DIM = 42
FEATURE_INDEX_MAP = {
    "estimated_state": [0, 15],
    "estimated_velocity": [3, 6],
    "estimated_attitude": [6, 9],
    "diagonal_covariance": [15, 30],
    "goal_delta": [30, 33],
    "obstacle_aggregate": [33, 37],
    "braking_features": [37, 39],
    "obstacle_body_direction": [39, 42],
}


def _finite_vector(value: Iterable[float], *, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(tuple(value), dtype=float).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class SingleUAVPPOConfig:
    """Explicit from-scratch settings for the isolated baseline.

    Values are repeated here instead of inherited from mutable historical
    experiment defaults.  The four supervisor/assistance switches are fixed
    false by validation and cannot be enabled through this loader.
    """

    baseline_id: str = BASELINE_ID
    seed: int = 2026
    uav_count: int = 1
    gnss_mode: str = "denied"
    max_steps: int = 120
    dt_s: float = 0.2
    max_speed_mps: float = 12.0
    max_acceleration_mps2: float = 3.0
    safety_distance_m: float = 2.0
    bounds_min_m: tuple[float, float, float] = (0.0, 0.0, 2.0)
    bounds_max_m: tuple[float, float, float] = (200.0, 200.0, 60.0)
    safety_projection_enabled: bool = True
    safety_execution_mode: str = "legacy"
    dynamic_obstacle_motion_features_enabled: bool = False
    dynamic_obstacle_swept_projection_enabled: bool = False
    single_uav_obstacle_direction_features_enabled: bool = True

    # Dense, estimated-goal navigation remains the learning signal.  These
    # values only describe the historical reward channel; strict success uses
    # truth state in StrictSingleUAVPPOEnv.
    single_uav_learning_mode: str = "loc_goal_v2"
    single_uav_speed_control_mode: str = "braking_distance"
    single_uav_braking_distance_speed_profile: str = "legacy"
    single_uav_goal_threshold_m: float = 1.0
    single_uav_goal_speed_threshold_mps: float = 0.5
    single_uav_near_goal_radius_m: float = 5.0
    single_uav_target_speed_far_mps: float = 6.0
    single_uav_braking_acceleration_mps2: float = 2.0
    single_uav_braking_margin_m: float = 2.0
    single_uav_stable_hold_steps: int = 5
    single_uav_goal_bonus: float = 5.0
    single_uav_goal_hold_reward: float = 0.0
    strict_success_bonus: float = 5.0
    strict_success_hold_steps: int = 5
    strict_pass_buffer_m: float = 0.75
    strict_minimum_clearance_m: float = 2.5

    # The approved baseline explicitly disables every assistance path.
    pass_commit_enabled: bool = False
    lateral_exit_enabled: bool = False
    radial_intent_enabled: bool = False
    goal_recovery_correction_enabled: bool = False
    goal_recovery_terminal_capture_enabled: bool = False

    # Explicit PPO values.  These are component assembly settings only until
    # a later, separately approved training stage.
    learning_rate: float = 1e-4
    policy_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.005
    initial_delta_std: float = 0.20
    ppo_action_logprob_mode: str = "sampled"
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64
    gat_hidden_dim: int = 32
    gat_output_dim: int = 32
    gat_heads: int = 1
    gat_layers: int = 2
    max_obstacles: int = 8
    device: str = "cpu"
    encoder_type: str = "gat"
    train_episodes: int = 100
    warmup_episodes: int = 5
    validation_episodes: int = 10
    test_episodes: int = 10
    from_scratch: bool = True
    historical_checkpoint: str | None = None
    train_potential_statistics: Mapping[str, Any] | None = None

    # A future trainer must receive frozen, train-fitted normalization and a
    # provenance record.  The preparation API leaves this pending.
    train_normalizer_path: str | None = None
    train_normalizer_provenance: Mapping[str, Any] | None = None
    action_log_path: str = "logs/preliminary/single_uav_ppo_baseline_v1_seed2026/sampled_execution.csv"

    def __post_init__(self) -> None:
        if self.baseline_id != BASELINE_ID:
            raise ValueError(f"baseline_id must be {BASELINE_ID}")
        if self.uav_count != 1 or self.max_steps != 120:
            raise ValueError("the isolated baseline requires one UAV and max_steps=120")
        if self.gnss_mode != "denied":
            raise ValueError("single-UAV PPO baseline requires gnss_mode='denied'")
        if self.safety_execution_mode != "legacy":
            raise ValueError("single-UAV PPO baseline freezes safety_execution_mode='legacy'")
        if self.dynamic_obstacle_motion_features_enabled or self.dynamic_obstacle_swept_projection_enabled:
            raise ValueError("dynamic obstacle feature/projection extensions are disabled")
        if not self.safety_projection_enabled:
            raise ValueError("base safety projection must remain enabled")
        exact_values = {
            "dt_s": 0.2,
            "max_speed_mps": 12.0,
            "max_acceleration_mps2": 3.0,
            "safety_distance_m": 2.0,
            "single_uav_goal_threshold_m": 1.0,
            "single_uav_goal_speed_threshold_mps": 0.5,
        }
        for name, expected in exact_values.items():
            if not np.isclose(float(getattr(self, name)), expected, atol=0.0, rtol=0.0):
                raise ValueError(f"{name} is frozen at {expected}")
        if tuple(self.bounds_min_m) != (0.0, 0.0, 2.0) or tuple(self.bounds_max_m) != (200.0, 200.0, 60.0):
            raise ValueError("single-UAV PPO baseline bounds are frozen")
        if not self.single_uav_obstacle_direction_features_enabled:
            raise ValueError("single-UAV PPO baseline requires obstacle direction features")
        if self.single_uav_learning_mode != "loc_goal_v2":
            raise ValueError("single-UAV PPO baseline requires loc_goal_v2 dense navigation")
        if self.single_uav_speed_control_mode != "braking_distance":
            raise ValueError("single-UAV PPO baseline requires braking_distance speed mode")
        if self.single_uav_stable_hold_steps != self.strict_success_hold_steps:
            raise ValueError("legacy and strict hold settings must both be five")
        if self.strict_success_hold_steps != 5:
            raise ValueError("strict success requires exactly five post-state holds")
        if self.ppo_action_logprob_mode != "sampled":
            raise ValueError("PPO log probability mode must be sampled")
        if self.historical_checkpoint is not None or not self.from_scratch:
            raise ValueError("this baseline loader accepts only fresh from-scratch initialization")
        if self.train_potential_statistics is not None and not isinstance(self.train_potential_statistics, Mapping):
            raise ValueError("train_potential_statistics must be a mapping when supplied")
        for name in (
            "pass_commit_enabled",
            "lateral_exit_enabled",
            "radial_intent_enabled",
            "goal_recovery_correction_enabled",
            "goal_recovery_terminal_capture_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a JSON boolean")
            if getattr(self, name):
                raise ValueError(f"{name} is disabled by the isolated baseline contract")
        for name in (
            "safety_projection_enabled",
            "dynamic_obstacle_motion_features_enabled",
            "dynamic_obstacle_swept_projection_enabled",
            "single_uav_obstacle_direction_features_enabled",
            "from_scratch",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a JSON boolean")
        for name, value in (
            ("dt_s", self.dt_s),
            ("max_speed_mps", self.max_speed_mps),
            ("max_acceleration_mps2", self.max_acceleration_mps2),
            ("safety_distance_m", self.safety_distance_m),
            ("strict_pass_buffer_m", self.strict_pass_buffer_m),
            ("strict_minimum_clearance_m", self.strict_minimum_clearance_m),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.strict_minimum_clearance_m < self.safety_distance_m:
            raise ValueError("strict minimum clearance must include the base safety distance")
        _finite_vector(self.bounds_min_m, size=3, name="bounds_min_m")
        _finite_vector(self.bounds_max_m, size=3, name="bounds_max_m")
        if any(lo >= hi for lo, hi in zip(self.bounds_min_m, self.bounds_max_m)):
            raise ValueError("bounds_min_m must be below bounds_max_m")

    @property
    def node_feature_dim(self) -> int:
        return EXPECTED_NODE_FEATURE_DIM

    @property
    def max_delta_v(self) -> float:
        return float(self.max_acceleration_mps2 * self.dt_s)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bounds_min_m"] = list(self.bounds_min_m)
        payload["bounds_max_m"] = list(self.bounds_max_m)
        payload["feature_index_map"] = copy.deepcopy(FEATURE_INDEX_MAP)
        payload["node_feature_dim"] = self.node_feature_dim
        payload["max_delta_v"] = self.max_delta_v
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SingleUAVPPOConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("single-UAV PPO config must be a JSON object")
        values = dict(payload)
        values.pop("feature_index_map", None)
        values.pop("node_feature_dim", None)
        values.pop("max_delta_v", None)
        for key in ("bounds_min_m", "bounds_max_m"):
            if key in values:
                values[key] = tuple(values[key])
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown single-UAV PPO config keys: {unknown}")
        return cls(**values)


def load_single_uav_ppo_config(path: str | Path) -> SingleUAVPPOConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read single-UAV PPO config {config_path}: {exc}") from exc
    return SingleUAVPPOConfig.from_mapping(payload)


def build_single_uav_simulation_config(config: SingleUAVPPOConfig) -> SimulationConfig:
    """Build the actual runtime graph from the new config values."""

    motion = MotionConfig(
        dt_s=config.dt_s,
        max_speed_mps=config.max_speed_mps,
        max_acceleration_mps2=config.max_acceleration_mps2,
        bounds_min_m=config.bounds_min_m,
        bounds_max_m=config.bounds_max_m,
        safety_distance_m=config.safety_distance_m,
    )
    reward = RewardConfig(
        collision_penalty=10.0,
        energy_weight=0.01,
        single_uav_learning_mode=config.single_uav_learning_mode,
        single_uav_learning_weights={"localization_progress": 0.8, "goal_progress": 0.2},
        single_uav_goal_progress_coef=1.0,
        single_uav_localization_cost_weight=0.2,
        single_uav_goal_bonus=config.single_uav_goal_bonus,
        single_uav_goal_threshold_m=config.single_uav_goal_threshold_m,
        single_uav_goal_speed_threshold_mps=config.single_uav_goal_speed_threshold_mps,
        single_uav_near_goal_radius_m=config.single_uav_near_goal_radius_m,
        single_uav_near_goal_speed_penalty_weight=0.1,
        single_uav_speed_control_mode=config.single_uav_speed_control_mode,
        single_uav_target_speed_far_mps=config.single_uav_target_speed_far_mps,
        single_uav_braking_acceleration_mps2=config.single_uav_braking_acceleration_mps2,
        single_uav_braking_margin_m=config.single_uav_braking_margin_m,
        single_uav_braking_distance_speed_profile=config.single_uav_braking_distance_speed_profile,
        single_uav_stable_hold_steps=config.single_uav_stable_hold_steps,
        single_uav_goal_hold_reward=config.single_uav_goal_hold_reward,
        single_uav_braking_safety_projection_enabled=False,
        potential_statistics=copy.deepcopy(config.train_potential_statistics),
    )
    return SimulationConfig(
        random_stream_mode="legacy",
        uav_count=1,
        max_steps=config.max_steps,
        seed=config.seed,
        motion=motion,
        reward=reward,
        gnss_mode=config.gnss_mode,
        cooperative_ekf_enabled=True,
        safety_projection_enabled=config.safety_projection_enabled,
        dynamic_obstacle_motion_features_enabled=config.dynamic_obstacle_motion_features_enabled,
        dynamic_obstacle_swept_projection_enabled=config.dynamic_obstacle_swept_projection_enabled,
        single_uav_obstacle_direction_features_enabled=config.single_uav_obstacle_direction_features_enabled,
        safety_execution_mode=config.safety_execution_mode,
    )


def _require_train_potential_statistics(config: SingleUAVPPOConfig) -> None:
    statistics = config.train_potential_statistics
    if not isinstance(statistics, Mapping):
        raise ValueError(
            "future PPO construction requires train-fitted potential_statistics; "
            "preflight does not invent delta scales"
        )
    required = ("u_center", "u_scale", "c_center", "c_scale", "delta_u_scale", "delta_goal_scale")
    missing = [key for key in required if key not in statistics]
    if missing:
        raise ValueError(f"train potential_statistics missing required keys: {missing}")
    for key in required:
        value = float(statistics[key])
        if not np.isfinite(value) or key.endswith("scale") and value <= 0.0:
            raise ValueError(f"train potential_statistics[{key}] must be finite and positive for scales")


def _obstacle_projection_bounds(start: np.ndarray, direction: np.ndarray, obstacle: Obstacle) -> tuple[float, float]:
    if isinstance(obstacle, SphereObstacle):
        center = np.asarray(obstacle.center_m, dtype=float)
        projection = float(np.dot(center - start, direction))
        return projection - float(obstacle.radius_m), projection + float(obstacle.radius_m)
    if isinstance(obstacle, BoxObstacle):
        corners = np.asarray(
            [
                [x, y, z]
                for x in (obstacle.minimum_m[0], obstacle.maximum_m[0])
                for y in (obstacle.minimum_m[1], obstacle.maximum_m[1])
                for z in (obstacle.minimum_m[2], obstacle.maximum_m[2])
            ],
            dtype=float,
        )
        projections = (corners - start) @ direction
        return float(np.min(projections)), float(np.max(projections))
    return float("inf"), float("-inf")


def _mission_relevant_obstacles(start: np.ndarray, goal: np.ndarray, obstacles: Iterable[Obstacle]) -> dict[str, Obstacle]:
    relevant: dict[str, Obstacle] = {}
    for obstacle in obstacles:
        if not obstacle.active or not is_physical_obstacle(obstacle):
            continue
        if segment_intersects_obstacle(start, goal, obstacle):
            obstacle_min, obstacle_max = _obstacle_projection_bounds(
                start, (goal - start) / max(float(np.linalg.norm(goal - start)), 1e-12), obstacle
            )
            direction = (goal - start) / max(float(np.linalg.norm(goal - start)), 1e-12)
            center_projection = float(np.dot(np.asarray(obstacle.center_m, dtype=float) - start, direction))
            if obstacle_max > 0.0 and center_projection > 0.0:
                relevant[str(obstacle.obstacle_id)] = obstacle
    return relevant


@dataclass(frozen=True)
class StrictSuccessResult:
    collision_permanent_failure: bool
    physical_pass_complete: bool
    goal_distance_ok: bool
    speed_ok: bool
    hold_steps: int
    success: bool


def evaluate_strict_success(
    *,
    collision: bool,
    physical_pass_complete: bool,
    actual_goal_distance_m: float,
    actual_speed_mps: float,
    previous_hold_steps: int,
    required_hold_steps: int = 5,
    goal_threshold_m: float = 1.0,
    speed_threshold_mps: float = 0.5,
    collision_permanent_failure: bool = False,
) -> StrictSuccessResult:
    """Pure post-state strict gate used by the isolated environment."""

    permanent = bool(collision_permanent_failure or collision)
    distance_ok = bool(np.isfinite(actual_goal_distance_m) and actual_goal_distance_m <= goal_threshold_m)
    speed_ok = bool(np.isfinite(actual_speed_mps) and actual_speed_mps <= speed_threshold_mps)
    qualifying = bool(not permanent and physical_pass_complete and distance_ok and speed_ok)
    hold = int(previous_hold_steps + 1 if qualifying else 0)
    success = bool(qualifying and hold >= int(required_hold_steps))
    return StrictSuccessResult(permanent, bool(physical_pass_complete), distance_ok, speed_ok, hold, success)


class StrictSingleUAVPPOEnv(MultiUAVJointEnv):
    """Single-UAV environment with an isolated truth-state success gate."""

    def __init__(self, config: SimulationConfig, baseline: SingleUAVPPOConfig) -> None:
        if config.uav_count != 1:
            raise ValueError("StrictSingleUAVPPOEnv requires one UAV")
        self.baseline = baseline
        # Keep loc_goal_v2 progress and speed shaping, but suppress its
        # one-step legacy success signal and bonus.  Strict success is added
        # only after the actual post-step truth state has been evaluated.
        legacy_suppressed_reward = replace(
            config.reward,
            single_uav_goal_bonus=0.0,
            single_uav_goal_hold_reward=0.0,
            single_uav_stable_hold_steps=max(10_000, baseline.max_steps + 1),
        )
        super().__init__(replace(config, reward=legacy_suppressed_reward))
        self._strict_start_position = np.zeros(3, dtype=float)
        self._strict_goal = np.zeros(3, dtype=float)
        self._strict_direction = np.zeros(3, dtype=float)
        self._strict_relevant_obstacles: dict[str, Obstacle] = {}
        self._strict_passed_obstacles: set[str] = set()
        self._strict_min_clearance = float("inf")
        self._strict_collision_permanent_failure = False
        self._strict_hold_steps = 0
        self._strict_success = False
        self._strict_bonus_awarded = False
        self._strict_seen_steps = 0
        self._ppo_reset_count = 0
        self._ppo_episode_id = ""

    def reset(self, *, seed: int | None = None, scenario_config: Mapping[str, Any] | None = None):
        observations, info = super().reset(seed=seed, scenario_config=scenario_config)
        uid = self.uav_ids[0]
        self._strict_start_position = self.states[uid].position_m.copy()
        self._strict_goal = self.goals[uid].copy()
        delta = self._strict_goal - self._strict_start_position
        distance = float(np.linalg.norm(delta))
        self._strict_direction = delta / max(distance, 1e-12)
        self._strict_relevant_obstacles = _mission_relevant_obstacles(
            self._strict_start_position, self._strict_goal, self.obstacles
        )
        self._strict_passed_obstacles = set()
        self._strict_min_clearance = float("inf")
        self._strict_collision_permanent_failure = False
        self._strict_hold_steps = 0
        self._strict_success = False
        self._strict_bonus_awarded = False
        self._strict_seen_steps = 0
        self._ppo_reset_count += 1
        scene = scenario_config or {}
        self._ppo_episode_id = str(
            scene.get("episode_id", f"{self.baseline.baseline_id}_reset_{self._ppo_reset_count}")
        )
        info = dict(info)
        info.update(self.strict_status())
        info["actor_node_feature_dim"] = int(observations[uid]["node_features"].shape[-1])
        return observations, info

    def _update_physical_pass(self, previous_position: np.ndarray, current_position: np.ndarray) -> None:
        if not self._strict_relevant_obstacles:
            return
        for obstacle_id, obstacle in self._strict_relevant_obstacles.items():
            current = next((item for item in self.obstacles if str(item.obstacle_id) == obstacle_id), obstacle)
            current_clearance = segment_obstacle_clearance(
                previous_position,
                current_position,
                current,
            )
            self._strict_min_clearance = min(self._strict_min_clearance, current_clearance)
            _minimum, downstream = _obstacle_projection_bounds(
                self._strict_start_position, self._strict_direction, current
            )
            current_projection = float(np.dot(current_position - self._strict_start_position, self._strict_direction))
            if (
                current_projection >= downstream + self.baseline.strict_pass_buffer_m
                and current_clearance >= self.baseline.strict_minimum_clearance_m
            ):
                self._strict_passed_obstacles.add(obstacle_id)

    def strict_status(self) -> dict[str, Any]:
        physical_pass_complete = bool(
            self._strict_relevant_obstacles
            and self._strict_passed_obstacles >= set(self._strict_relevant_obstacles)
        )
        return {
            "strict_collision_permanent_failure": bool(self._strict_collision_permanent_failure),
            "strict_relevant_physical_obstacle_ids": sorted(self._strict_relevant_obstacles),
            "strict_passed_physical_obstacle_ids": sorted(self._strict_passed_obstacles),
            "strict_physical_pass_complete": physical_pass_complete,
            "strict_trajectory_minimum_clearance_m": (
                float(self._strict_min_clearance) if np.isfinite(self._strict_min_clearance) else None
            ),
            "strict_hold_steps": int(self._strict_hold_steps),
            "strict_success": bool(self._strict_success),
            "strict_goal_distance_m_actual": float(
                np.linalg.norm(self.states[self.uav_ids[0]].position_m - self._strict_goal)
            ) if self.states else float("nan"),
            "strict_speed_mps_actual": float(np.linalg.norm(self.states[self.uav_ids[0]].velocity_mps)) if self.states else float("nan"),
        }

    def step(self, action):
        uid = self.uav_ids[0]
        previous_state = self.states[uid].copy()
        previous_position = self.states[uid].position_m.copy()
        observations, rewards, terminated, truncated, info = super().step(action)
        self._strict_seen_steps += 1
        current_position = self.states[uid].position_m.copy()
        self._update_physical_pass(previous_position, current_position)
        collision = bool(info.get("collision", False))
        status = evaluate_strict_success(
            collision=collision,
            physical_pass_complete=bool(
                self._strict_relevant_obstacles
                and self._strict_passed_obstacles >= set(self._strict_relevant_obstacles)
            ),
            actual_goal_distance_m=float(np.linalg.norm(current_position - self._strict_goal)),
            actual_speed_mps=float(np.linalg.norm(self.states[uid].velocity_mps)),
            previous_hold_steps=self._strict_hold_steps,
            required_hold_steps=self.baseline.strict_success_hold_steps,
            goal_threshold_m=self.baseline.single_uav_goal_threshold_m,
            speed_threshold_mps=self.baseline.single_uav_goal_speed_threshold_mps,
            collision_permanent_failure=self._strict_collision_permanent_failure,
        )
        self._strict_collision_permanent_failure = status.collision_permanent_failure
        self._strict_hold_steps = status.hold_steps
        self._strict_success = bool(
            (self._strict_success or status.success)
            and not status.collision_permanent_failure
        )
        strict_bonus_triggered = bool(self._strict_success and not self._strict_bonus_awarded)
        if strict_bonus_triggered:
            bonus = float(self.baseline.strict_success_bonus)
            learning_rewards = info.get("learning_rewards")
            if isinstance(learning_rewards, Mapping):
                # The legacy core returns its bounded learning channel and
                # exposes the preclip values in the component manifest.  Add
                # the strict one-shot bonus before applying that same clip.
                components = info.get("learning_reward_components", {})
                preclip = components.get("learning_reward_preclip_by_uav", {}) if isinstance(components, Mapping) else {}
                clip = float(self.config.reward.learning_reward_clip)
                info["learning_rewards"] = {
                    key: float(np.clip(float(preclip.get(key, value)) + bonus, -clip, clip))
                    for key, value in learning_rewards.items()
                }
                if isinstance(components, Mapping):
                    updated_components = dict(components)
                    updated_components["learning_reward_total_by_uav"] = dict(info["learning_rewards"])
                    updated_components["learning_reward_total"] = float(np.mean(list(info["learning_rewards"].values())))
                    updated_components["learning_reward_preclip_by_uav"] = {
                        key: float(preclip.get(key, value)) + bonus
                        for key, value in learning_rewards.items()
                    }
                    updated_components["single_uav_learning_reward_preclip"] = float(
                        np.mean(list(updated_components["learning_reward_preclip_by_uav"].values()))
                    )
                    updated_components["single_uav_learning_reward"] = updated_components["learning_reward_total"]
                    updated_components["single_uav_goal_bonus"] = bonus
                    updated_components["strict_success_bonus"] = bonus
                    info["learning_reward_components"] = updated_components
            self._strict_bonus_awarded = True
            info["strict_success_bonus"] = bonus
        else:
            info["strict_success_bonus"] = 0.0
        if self._strict_success:
            terminated, truncated = True, False
            info["termination_reason"] = "strict_goal_reached"
            info["task_complete"] = True
        elif self._strict_collision_permanent_failure:
            terminated, truncated = True, False
            info["termination_reason"] = "collision"
            info["task_complete"] = False
        uid = self.uav_ids[0]
        actual_distance = float(np.linalg.norm(self.states[uid].position_m - self._strict_goal))
        actual_speed = float(np.linalg.norm(self.states[uid].velocity_mps))
        strict_condition = bool(
            not self._strict_collision_permanent_failure
            and bool(
                self._strict_relevant_obstacles
                and self._strict_passed_obstacles >= set(self._strict_relevant_obstacles)
            )
            and actual_distance <= self.baseline.single_uav_goal_threshold_m
            and actual_speed <= self.baseline.single_uav_goal_speed_threshold_mps
        )
        # Replace legacy estimated-goal reporting fields with the isolated
        # truth-state contract so downstream reports cannot mistake the old
        # one-step estimated signal for baseline success.
        info["single_uav_goal_in_radius"] = bool(actual_distance <= self.baseline.single_uav_goal_threshold_m)
        info["single_uav_goal_speed_ok"] = bool(actual_speed <= self.baseline.single_uav_goal_speed_threshold_mps)
        info["single_uav_goal_condition"] = strict_condition
        info["single_uav_goal_reached"] = bool(self._strict_success)
        info["single_uav_goal_stable"] = bool(self._strict_success)
        info["single_uav_stable_steps"] = int(self._strict_hold_steps)
        info["single_uav_stable_hold_steps"] = int(self.baseline.strict_success_hold_steps)
        info["single_uav_goal_bonus_triggered"] = strict_bonus_triggered
        info["single_uav_goal_bonus"] = float(self.baseline.strict_success_bonus if strict_bonus_triggered else 0.0)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info.update(self.strict_status())
        components = info.get("learning_reward_components")
        if isinstance(components, Mapping):
            updated = dict(components)
            updated.update({key: value for key, value in info.items() if (
                key.startswith("single_uav_goal_")
                or key in {"single_uav_stable_steps", "single_uav_stable_hold_steps"}
                or key.startswith("strict_")
            )})
            info["learning_reward_components"] = updated
        info["strict_gate_source"] = "actual_post_state_truth"
        info["legacy_goal_success_ignored"] = True
        info["single_uav_ppo_episode_id"] = self._ppo_episode_id
        info["single_uav_ppo_step"] = self._strict_seen_steps - 1
        info["single_uav_ppo_pre_action_state"] = {
            name: np.asarray(getattr(previous_state, name), dtype=float).tolist()
            for name in (
                "position_m", "velocity_mps", "attitude_rad",
                "acceleration_mps2", "angular_velocity_rps",
            )
        }
        return observations, rewards, bool(terminated), bool(truncated), info


def create_single_uav_ppo_environment(config: SingleUAVPPOConfig) -> StrictSingleUAVPPOEnv:
    return StrictSingleUAVPPOEnv(build_single_uav_simulation_config(config), config)


def _require_fitted_training_normalizer(
    normalizer: ObservationNormalizer | None,
    provenance: Mapping[str, Any] | None,
    *,
    expected_node_dim: int,
) -> None:
    if normalizer is None or not normalizer.fitted:
        raise ValueError("future PPO construction requires a fitted train ObservationNormalizer")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("future PPO construction requires train-normalizer provenance")
    source_sha = provenance.get("source_sha256", provenance.get("sha256"))
    if not isinstance(source_sha, str) or not source_sha.strip():
        raise ValueError("train-normalizer provenance must include source_sha256")
    if str(provenance.get("fit_split", "")).lower() != "train":
        raise ValueError("train-normalizer provenance must identify fit_split='train'")
    feature_map = provenance.get("feature_index_map", provenance.get("feature_map"))
    if feature_map is None:
        raise ValueError("train-normalizer provenance must include feature_index_map")
    if dict(feature_map) != FEATURE_INDEX_MAP:
        raise ValueError("train-normalizer feature_index_map does not match the 42-feature actor contract")
    artifact_path = provenance.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError("train-normalizer provenance must include artifact_path for SHA verification")
    artifact = Path(artifact_path)
    if not artifact.exists() or not artifact.is_file() or _sha256_file(artifact) != source_sha:
        raise ValueError("train-normalizer source_sha256 does not match artifact_path")
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(saved, Mapping):
        raise ValueError("train-normalizer artifact must be a JSON object")
    saved_observation = saved.get("observation_normalizer", saved)
    if ObservationNormalizer.from_state_dict(saved_observation).state_dict() != normalizer.state_dict():
        raise ValueError("supplied observation statistics differ from the hashed train artifact")
    canonical_hash = provenance.get("state_sha256")
    if canonical_hash is not None:
        canonical = json.dumps(normalizer.state_dict(), sort_keys=True, separators=(",", ":"), default=str)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != str(canonical_hash):
            raise ValueError("train-normalizer state_sha256 does not match the supplied normalizer state")
    state = normalizer.state_dict()
    node = state.get("normalizers", {}).get("node_features")
    if not isinstance(node, Mapping) or np.asarray(node.get("mean", [])).size != expected_node_dim:
        raise ValueError("fitted train normalizer node feature dimension does not match 42")
    if (
        not np.isfinite(np.asarray(node["mean"], dtype=float)).all()
        or not np.isfinite(np.asarray(node["scale"], dtype=float)).all()
        or np.any(np.asarray(node["scale"], dtype=float) <= 0.0)
        or np.any(np.asarray(node["count"], dtype=int) <= 0)
    ):
        raise ValueError("all 42 Actor features require finite, fitted training statistics")


class FrozenObservationNormalizer:
    """Read-only evaluation view over train-fitted observation statistics."""

    def __init__(self, normalizer: ObservationNormalizer, provenance: Mapping[str, Any]) -> None:
        _require_fitted_training_normalizer(normalizer, provenance, expected_node_dim=EXPECTED_NODE_FEATURE_DIM)
        self._normalizer = ObservationNormalizer.from_state_dict(normalizer.state_dict())
        self.provenance = copy.deepcopy(dict(provenance))

    @property
    def fitted(self) -> bool:
        return bool(self._normalizer.fitted)

    def transform(self, observations: Any) -> Any:
        return self._normalizer.transform(observations)

    def state_dict(self) -> dict[str, Any]:
        return self._normalizer.state_dict()

    def fit(self, *_args: Any, **_kwargs: Any) -> "FrozenObservationNormalizer":
        raise RuntimeError("evaluation normalizer is frozen; fit/update is forbidden")

    update = fit


class FrozenFeatureNormalizer:
    """Read-only evaluation view over a train-fitted central normalizer."""

    def __init__(self, normalizer: FeatureNormalizer, provenance: Mapping[str, Any], expected_dim: int) -> None:
        if not normalizer.fitted:
            raise ValueError("future PPO construction requires a fitted central normalizer")
        if normalizer.feature_dim != int(expected_dim):
            raise ValueError("central normalizer feature dimension does not match the actor/critic contract")
        artifact = Path(str(provenance.get("artifact_path", "")))
        if not artifact.is_file() or _sha256_file(artifact) != provenance.get("source_sha256", provenance.get("sha256")):
            raise ValueError("central normalizer requires the hashed training artifact")
        saved = json.loads(artifact.read_text(encoding="utf-8"))
        if saved.get("central_normalizer") != normalizer.state_dict():
            raise ValueError("supplied central statistics differ from the hashed train artifact")
        self._normalizer = FeatureNormalizer.from_state_dict(normalizer.state_dict())
        self.provenance = copy.deepcopy(dict(provenance))

    @property
    def fitted(self) -> bool:
        return bool(self._normalizer.fitted)

    @property
    def feature_dim(self) -> int | None:
        return self._normalizer.feature_dim

    def transform(self, values: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._normalizer.transform(values, *args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return self._normalizer.state_dict()

    def fit(self, *_args: Any, **_kwargs: Any) -> "FrozenFeatureNormalizer":
        raise RuntimeError("evaluation central normalizer is frozen; fit/update is forbidden")

    update = fit


class SampledActionLoggingPPOTrainer(MAPPOTrainer):
    """Attach execution logging to the normal PPO collection path."""

    def reconcile_action_execution(self, decision: Any, info: Mapping[str, Any] | None = None) -> None:
        uid = self.policy.uav_ids[0]
        sample = decision.samples[uid]
        sampled = np.asarray(sample.delta_v, dtype=float).copy()
        old_logprob = float(sample.old_log_prob)
        super().reconcile_action_execution(decision, info)
        if not np.array_equal(sample.delta_v, sampled) or sample.old_log_prob != old_logprob:
            raise AssertionError("sampled PPO action/logprob changed during reconciliation")
        logger = getattr(self, "action_logger", None)
        if logger is None:
            return
        payload = info or {}
        saved = payload.get("single_uav_ppo_pre_action_state")
        if not isinstance(saved, Mapping):
            raise ValueError("execution logging requires the isolated environment's pre-action state")
        previous_state = UAVKinematicState(
            **{name: np.asarray(saved[name], dtype=float) for name in (
                "position_m", "velocity_mps", "attitude_rad",
                "acceleration_mps2", "angular_velocity_rps",
            )}
        )
        logger.append(
            episode_id=str(payload.get("single_uav_ppo_episode_id", "")),
            step=int(payload["single_uav_ppo_step"]),
            uav_id=uid,
            previous_state=previous_state,
            sampled_delta_v=sampled,
            execution=payload["action_execution"][uid],
            motion=self.logging_environment.config.motion,
            old_logprob=old_logprob,
            logprob_action="sampled",
        )


def create_single_uav_ppo_trainer(
    environment: StrictSingleUAVPPOEnv,
    config: SingleUAVPPOConfig,
    *,
    normalizer: ObservationNormalizer | None = None,
    central_normalizer: FeatureNormalizer | None = None,
    normalizer_provenance: Mapping[str, Any] | None = None,
    action_log_path: str | Path | None = None,
) -> MAPPOTrainer:
    """Assemble a trainer with explicit settings; no update is performed."""

    _require_train_potential_statistics(config)
    runtime_statistics = environment.config.reward.potential_statistics
    if not isinstance(runtime_statistics, Mapping):
        raise ValueError("future PPO environment must be built with the same train-fitted potential_statistics")
    for key in ("delta_u_scale", "delta_goal_scale"):
        if isinstance(runtime_statistics, Mapping) and not np.isclose(
            float(runtime_statistics[key]), float(config.train_potential_statistics[key])
        ):
            raise ValueError(f"runtime potential_statistics[{key}] does not match train calibration")
    _require_fitted_training_normalizer(
        normalizer, normalizer_provenance, expected_node_dim=config.node_feature_dim
    )
    expected_central_dim = central_state_vector_dim(
        1, config.node_feature_dim, 11, config.max_obstacles, False
    )
    if central_normalizer is None or not central_normalizer.fitted:
        raise ValueError("future PPO construction requires a fitted central normalizer")
    if central_normalizer.feature_dim != expected_central_dim:
        raise ValueError("fitted central normalizer dimension does not match this baseline")
    frozen_normalizer = FrozenObservationNormalizer(normalizer, normalizer_provenance or {})
    frozen_central_normalizer = FrozenFeatureNormalizer(
        central_normalizer, normalizer_provenance or {}, expected_central_dim
    )
    trainer = SampledActionLoggingPPOTrainer.from_environment(
        environment,
        seed=config.seed,
        normalizer=frozen_normalizer,
        central_normalizer=frozen_central_normalizer,
        episode_objective="legacy",
        include_remaining_time=False,
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=11,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_output_dim=config.gat_output_dim,
        gat_heads=config.gat_heads,
        gat_layers=config.gat_layers,
        actor_hidden_dim=config.actor_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        max_obstacles=config.max_obstacles,
        learning_rate=config.learning_rate,
        policy_epochs=config.policy_epochs,
        minibatch_size=config.minibatch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        max_grad_norm=config.max_grad_norm,
        entropy_coef=config.entropy_coef,
        initial_delta_std=config.initial_delta_std,
        ppo_action_logprob_mode=config.ppo_action_logprob_mode,
        encoder_type=config.encoder_type,
        device=config.device,
    )
    if trainer.config.ppo_action_logprob_mode != "sampled":
        raise RuntimeError("trainer did not receive sampled PPO logprob mode")
    if trainer.config.node_feature_dim != EXPECTED_NODE_FEATURE_DIM:
        raise RuntimeError("trainer node feature dimension is not 42")
    # Exercise the real policy encoder input path once at construction time.
    # This is a reset-only contract probe: it samples no transition and does
    # not call backward or an optimizer update.
    if environment.states:
        probe_observations = environment._observations()
    else:
        probe_observations, _probe_info = environment.reset(seed=config.seed)
    normalized_probe = trainer.normalizer.transform(probe_observations)
    actor_input_dims: list[int] = []
    if config.encoder_type == "gat":
        hook = trainer.policy.gat.register_forward_pre_hook(
            lambda _module, args: actor_input_dims.append(int(args[0].shape[-1]))
        )
        try:
            trainer.policy._local_context(normalized_probe[environment.uav_ids[0]])
        finally:
            hook.remove()
    else:
        actor_input_dims.append(int(np.asarray(normalized_probe[environment.uav_ids[0]]["node_features"]).shape[-1]))
        trainer.policy._local_context(normalized_probe[environment.uav_ids[0]])
    if not actor_input_dims or actor_input_dims[-1] != EXPECTED_NODE_FEATURE_DIM:
        raise RuntimeError("actual Actor encoder input does not have 42 features")
    trainer._single_uav_actor_contract_probe = {
        "actor_encoder_input_shape": [1, int(actor_input_dims[-1])],
        "normalization_fitted": bool(trainer.normalizer.fitted),
        "normalization_provenance": copy.deepcopy(dict(normalizer_provenance or {})),
        "environment_steps": 0,
    }
    trainer.logging_environment = environment
    trainer.action_logger = ActionExecutionCSVLogger(
        Path(config.action_log_path if action_log_path is None else action_log_path)
    )
    return trainer


def verify_actor_observation_contract(
    environment: StrictSingleUAVPPOEnv,
    observations: Mapping[str, Mapping[str, Any]],
    *,
    normalizer: ObservationNormalizer | None = None,
) -> dict[str, Any]:
    """Inspect the actual reset graph and optionally its frozen normalization."""

    if set(observations) != {"uav_00"}:
        raise ValueError("actor contract expected exactly uav_00")
    local = observations["uav_00"]
    raw = np.asarray(local.get("node_features"), dtype=float)
    if raw.ndim != 2 or raw.shape[-1] != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError(f"actual actor graph has shape {raw.shape}, expected final dimension 42")
    estimate = np.asarray(local.get("estimate"), dtype=float).reshape(-1)
    if estimate.size < 6 or not np.allclose(raw[0, 3:6], estimate[3:6]):
        raise ValueError("estimated velocity is not present in the actor's first node row")
    goal_delta = np.asarray(local.get("goal"), dtype=float).reshape(3) - estimate[:3]
    if not np.allclose(raw[0, 30:33], goal_delta):
        raise ValueError("goal delta ordering is not [30:33]")
    expected_direction = environment._local_obstacle_direction_features(estimate)
    if not np.allclose(raw[0, 39:42], expected_direction):
        raise ValueError("obstacle body direction ordering is not [39:42]")
    result: dict[str, Any] = {
        "actual_actor_node_feature_dim": int(raw.shape[-1]),
        "raw_estimated_velocity": raw[0, 3:6].tolist(),
        "raw_goal_delta": raw[0, 30:33].tolist(),
        "raw_obstacle_body_direction": raw[0, 39:42].tolist(),
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "normalization_status": "pending" if normalizer is None else ("fitted" if normalizer.fitted else "unfitted"),
    }
    if normalizer is not None:
        if not normalizer.fitted:
            raise ValueError("evaluation normalization cannot fit or update an unfitted normalizer")
        transformed = normalizer.transform(observations)
        transformed_raw = np.asarray(transformed["uav_00"]["node_features"], dtype=float)
        if transformed_raw.shape != raw.shape:
            raise ValueError("normalized actor graph changed the feature shape")
        result["normalized_actor_node_feature_dim"] = int(transformed_raw.shape[-1])
        result["normalization_roundtrip_shape_match"] = True
    return result


def probe_sampled_execution_contract(
    environment: StrictSingleUAVPPOEnv,
    config: SingleUAVPPOConfig,
    observations: Mapping[str, Mapping[str, Any]],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise the real untrained PPO sample/reconcile/buffer path.

    The probe uses reset-fixture statistics in memory only.  It calls no
    environment transition, backward pass, or optimizer update.
    """

    temporary_normalizer = ObservationNormalizer().fit(observations)
    central = info.get("central_state")
    if not isinstance(central, Mapping):
        raise ValueError("reset-only action probe requires central_state")
    from ..learning.mappo import central_state_to_vector

    temporary_central_normalizer = FeatureNormalizer().fit(
        central_state_to_vector(
            central,
            uav_count=1,
            node_feature_dim=EXPECTED_NODE_FEATURE_DIM,
            edge_feature_dim=11,
            max_obstacles=config.max_obstacles,
            include_remaining_time=False,
        )
    )
    # Assemble the untrained policy directly for this reset-only probe.  The
    # production factory deliberately requires a real train-normalizer
    # artifact and matching potential statistics; temporary probe statistics
    # must never satisfy that future-training contract.
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=config.seed,
        normalizer=temporary_normalizer,
        central_normalizer=temporary_central_normalizer,
        episode_objective="legacy",
        include_remaining_time=False,
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=11,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_output_dim=config.gat_output_dim,
        gat_heads=config.gat_heads,
        gat_layers=config.gat_layers,
        actor_hidden_dim=config.actor_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        max_obstacles=config.max_obstacles,
        learning_rate=config.learning_rate,
        policy_epochs=config.policy_epochs,
        minibatch_size=config.minibatch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        max_grad_norm=config.max_grad_norm,
        entropy_coef=config.entropy_coef,
        initial_delta_std=config.initial_delta_std,
        ppo_action_logprob_mode=config.ppo_action_logprob_mode,
        encoder_type=config.encoder_type,
        device=config.device,
    )
    if trainer.config.ppo_action_logprob_mode != "sampled":
        raise RuntimeError("reset-only policy probe did not receive sampled PPO logprob mode")
    normalized_probe = temporary_normalizer.transform(observations)
    actor_input_dims: list[int] = []
    actor_input_values: list[np.ndarray] = []

    def capture_actor_input(_module: Any, args: Any) -> None:
        actor_input_dims.append(int(args[0].shape[-1]))
        actor_input_values.append(args[0].detach().cpu().numpy().copy())

    hook = trainer.policy.gat.register_forward_pre_hook(capture_actor_input)
    try:
        trainer.policy._local_context(normalized_probe[environment.uav_ids[0]])
    finally:
        hook.remove()
    if not actor_input_dims or actor_input_dims[-1] != EXPECTED_NODE_FEATURE_DIM:
        raise RuntimeError("actual reset-only Actor encoder input does not have 42 features")
    actor_received_normalized_features = bool(np.allclose(
        actor_input_values[-1], normalized_probe[environment.uav_ids[0]]["node_features"]
    ))
    if not actor_received_normalized_features:
        raise AssertionError("Actor input differs from the normalized 42-feature observation")
    uid = "uav_00"
    decision = trainer.act(observations, central, deterministic=False)
    sample = decision.samples[uid]
    sampled = np.asarray(sample.delta_v, dtype=float).copy()
    executed = -sampled + np.asarray([0.11, -0.07, 0.05], dtype=float)
    if np.allclose(sampled, executed):
        executed = sampled + np.asarray([0.13, 0.0, 0.0], dtype=float)
    old_logprob = float(sample.old_log_prob)
    before_logprob, _ = trainer.policy.evaluate_actions(decision.observations[uid], sample)
    executed_action = copy.deepcopy(decision.actions[uid])
    executed_action["delta_v"] = executed.copy()
    execution_info = {
        "actions": {uid: executed_action},
        "applied_bandwidth": {uid: {}},
        "action_execution": {
            uid: {
                "delta_v_applied": executed.copy(),
                "safety_projected": True,
                "boundary_projected": False,
                "action_clipped": False,
            }
        },
    }
    previous_state = environment.states[uid].copy()
    motion = environment.config.motion
    _nominal_state, _nominal_info = apply_velocity_increment(
        previous_state, sampled, motion, safety_projector=None
    )
    nominal_executed = np.asarray(_nominal_info["delta_v_applied"], dtype=float)
    safety_correction = executed - nominal_executed
    total_correction = executed - sampled
    reconcile = reconcile_single_uav_sampled_execution(trainer, decision, execution_info)
    after_logprob, _ = trainer.policy.evaluate_actions(decision.observations[uid], sample)
    ratio = float(np.exp(float(after_logprob.detach().cpu()) - old_logprob))
    trainer.add_transition(
        decision,
        {uid: 0.0},
        central,
        done=False,
        terminated=False,
        truncated=False,
    )
    return {
        "probe_tag": "actual_policy_sampled_vs_executed_buffer_probe",
        "sampled_delta_v": sampled.tolist(),
        "executed_delta_v": executed.tolist(),
        "safety_correction": safety_correction.tolist(),
        "total_correction": total_correction.tolist(),
        "action_execution_changed": bool(reconcile["action_execution_changed"]),
        "logprob_action": "sampled",
        "old_logprob_before": old_logprob,
        "old_logprob_after": float(after_logprob.detach().cpu()),
        "policy_evaluation_before": float(before_logprob.detach().cpu()),
        "stored_old_logprob_after": float(sample.old_log_prob),
        "old_logprob_unchanged": bool(
            sample.old_log_prob == old_logprob
            and np.isclose(float(after_logprob.detach().cpu()), old_logprob, atol=1e-6)
        ),
        "importance_ratio": ratio,
        "ratio_is_one": bool(np.isclose(ratio, 1.0, atol=1e-5)),
        "actor_encoder_input_shape": [1, int(actor_input_dims[-1])],
        "actor_received_normalized_features": actor_received_normalized_features,
        "normalization_fitted": bool(temporary_normalizer.fitted),
        "normalization_probe_only": True,
        "buffer_transition_count": len(trainer.rollout),
        "buffer_sampled_action_preserved": bool(np.array_equal(
            trainer.rollout[0].decision.samples[uid].delta_v, sampled
        )),
        "optimizer_created_for_component_probe": True,
        "optimizer_state_entries": len(trainer.optimizer.state),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "environment_steps": 0,
    }


def _controlled_reset_perturbation_probe(
    environment: StrictSingleUAVPPOEnv,
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Perturb upstream velocity, goal, and obstacle and rebuild the graph."""

    uid = environment.uav_ids[0]
    baseline = np.asarray(observations[uid]["node_features"], dtype=float).copy()
    observation_variants = [copy.deepcopy(observations)]
    ekf = environment.ekfs[uid]
    old_velocity = ekf.state[3:6].copy()
    old_goal = environment.goals[uid].copy()
    obstacle = environment.obstacles[0] if environment.obstacles else None
    old_center = None if obstacle is None else obstacle.center_m.copy()
    try:
        ekf.state[3:6] += np.asarray([0.25, -0.10, 0.05])
        environment.graph_batch = environment._build_graph_batch()
        velocity_probe = np.asarray(environment._observations()[uid]["node_features"], dtype=float)
        observation_variants.append(copy.deepcopy(environment._observations()))
        environment.goals[uid] = old_goal + np.asarray([0.0, 1.0, 0.0])
        environment.graph_batch = environment._build_graph_batch()
        goal_probe = np.asarray(environment._observations()[uid]["node_features"], dtype=float)
        observation_variants.append(copy.deepcopy(environment._observations()))
        obstacle_probe_changed = False
        if obstacle is not None:
            obstacle.center_m = obstacle.center_m + np.asarray([0.0, 1.0, 0.0])
            environment.graph_batch = environment._build_graph_batch()
            obstacle_probe = np.asarray(environment._observations()[uid]["node_features"], dtype=float)
            observation_variants.append(copy.deepcopy(environment._observations()))
            obstacle_probe_changed = bool(not np.allclose(obstacle_probe[:, 39:42], baseline[:, 39:42]))
        else:
            obstacle_probe = baseline.copy()
    finally:
        ekf.state[3:6] = old_velocity
        environment.goals[uid] = old_goal
        if obstacle is not None and old_center is not None:
            obstacle.center_m = old_center
        environment.graph_batch = environment._build_graph_batch()
    temporary = ObservationNormalizer().fit(observation_variants)
    normalized = [temporary.transform(value)[uid]["node_features"] for value in observation_variants]
    return {
        "estimated_velocity_changed": bool(not np.allclose(velocity_probe[:, 3:6], baseline[:, 3:6])),
        "goal_delta_changed": bool(not np.allclose(goal_probe[:, 30:33], baseline[:, 30:33])),
        "obstacle_direction_changed": obstacle_probe_changed,
        "normalized_estimated_velocity_changed": bool(not np.allclose(normalized[0][:, 3:6], normalized[1][:, 3:6])),
        "normalized_goal_delta_changed": bool(not np.allclose(normalized[0][:, 30:33], normalized[2][:, 30:33])),
        "normalized_obstacle_direction_changed": bool(
            len(normalized) == 4 and not np.allclose(normalized[0][:, 39:42], normalized[3][:, 39:42])
        ),
        "normalization_fit_scope": "temporary_reset_probe_only",
        "restored_feature_shape": list(environment._observations()[uid]["node_features"].shape),
        "environment_steps": 0,
    }


def reconcile_single_uav_sampled_execution(
    trainer: MAPPOTrainer,
    decision: Any,
    info: Mapping[str, Any],
    *,
    logger: "ActionExecutionCSVLogger | None" = None,
    previous_states: Mapping[str, UAVKinematicState] | None = None,
    episode_id: str = "",
    step: int = 0,
    motion: MotionConfig | None = None,
) -> dict[str, Any]:
    """Reconcile diagnostics while preserving sampled PPO action/logprob."""

    if trainer.config.ppo_action_logprob_mode != "sampled":
        raise ValueError("single-UAV baseline requires sampled PPO reconciliation")
    uid = trainer.policy.uav_ids[0]
    sample = decision.samples[uid]
    old_log_prob = float(sample.old_log_prob)
    sampled = np.asarray(sample.delta_v, dtype=float).copy()
    trainer.reconcile_action_execution(decision, info)
    if not np.isclose(sample.old_log_prob, old_log_prob, atol=1e-7):
        raise AssertionError("sampled PPO old_log_prob changed during execution reconciliation")
    result = {
        "ppo_action_logprob_mode": "sampled",
        "sampled_delta_v": sampled.tolist(),
        "executed_delta_v": np.asarray(sample.executed_delta_v, dtype=float).tolist(),
        "old_log_prob_unchanged": True,
        "action_execution_changed": bool(sample.action_execution_changed),
    }
    if logger is not None:
        if previous_states is None or uid not in previous_states:
            raise ValueError("previous_states[uav_00] is required when action logging is enabled")
        execution = info.get("action_execution", {})
        execution = execution.get(uid, {}) if isinstance(execution, Mapping) else {}
        if motion is None:
            raise ValueError("motion is required when action logging is enabled")
        logger.append(
            episode_id=episode_id,
            probe_tag="future_transition",
            step=step,
            uav_id=uid,
            previous_state=previous_states[uid],
            sampled_delta_v=sampled,
            execution=execution,
            motion=motion,
            logprob_action="sampled",
            old_logprob=old_log_prob,
        )
        result["action_log_written"] = True
    return result


@dataclass
class ActionExecutionCSVLogger:
    """Future-use CSV logger separating sample, execution, and safety residuals."""

    path: Path
    _fieldnames: tuple[str, ...] = (
        "episode_id",
        "probe_tag",
        "step",
        "uav_id",
        "logprob_action",
        "old_logprob",
        "sampled_delta_v_x",
        "sampled_delta_v_y",
        "sampled_delta_v_z",
        "executed_delta_v_x",
        "executed_delta_v_y",
        "executed_delta_v_z",
        "safety_correction_norm_mps",
        "safety_correction_x",
        "safety_correction_y",
        "safety_correction_z",
        "total_correction_norm_mps",
        "total_correction_x",
        "total_correction_y",
        "total_correction_z",
        "reason_safety_projection",
        "reason_action_norm_clip",
        "reason_speed_clip",
        "reason_boundary_projection",
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self._fieldnames).writeheader()

    def append(
        self,
        *,
        episode_id: str = "",
        probe_tag: str = "future_transition",
        step: int,
        uav_id: str,
        previous_state: UAVKinematicState,
        sampled_delta_v: Iterable[float],
        execution: Mapping[str, Any],
        motion: MotionConfig,
        logprob_action: str = "sampled",
        old_logprob: float | None = None,
    ) -> dict[str, Any]:
        sampled = np.asarray(tuple(sampled_delta_v), dtype=float).reshape(3)
        executed = np.asarray(execution.get("delta_v_applied", sampled), dtype=float).reshape(3)
        no_safety_state, no_safety_info = apply_velocity_increment(
            previous_state, sampled, motion, safety_projector=None
        )
        nominal = np.asarray(no_safety_info["delta_v_applied"], dtype=float).reshape(3)
        total_correction = executed - sampled
        safety_correction = executed - nominal if bool(execution.get("safety_projected", False)) else np.zeros(3)
        row = {
            "episode_id": str(episode_id),
            "probe_tag": str(probe_tag),
            "step": int(step),
            "uav_id": str(uav_id),
            "logprob_action": str(logprob_action),
            "old_logprob": "" if old_logprob is None else float(old_logprob),
            "sampled_delta_v_x": float(sampled[0]),
            "sampled_delta_v_y": float(sampled[1]),
            "sampled_delta_v_z": float(sampled[2]),
            "executed_delta_v_x": float(executed[0]),
            "executed_delta_v_y": float(executed[1]),
            "executed_delta_v_z": float(executed[2]),
            "safety_correction_norm_mps": float(np.linalg.norm(safety_correction)),
            "safety_correction_x": float(safety_correction[0]),
            "safety_correction_y": float(safety_correction[1]),
            "safety_correction_z": float(safety_correction[2]),
            "total_correction_norm_mps": float(np.linalg.norm(total_correction)),
            "total_correction_x": float(total_correction[0]),
            "total_correction_y": float(total_correction[1]),
            "total_correction_z": float(total_correction[2]),
            "reason_safety_projection": int(bool(execution.get("safety_projected", False))),
            "reason_action_norm_clip": int(not np.allclose(no_safety_info.get("delta_v_commanded", sampled), sampled)),
            "reason_speed_clip": int(
                np.linalg.norm(previous_state.velocity_mps + np.asarray(no_safety_info.get("delta_v_commanded", sampled)))
                > motion.max_speed_mps + 1e-8
            ),
            "reason_boundary_projection": int(bool(execution.get("boundary_projected", False))),
        }
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self._fieldnames).writerow(row)
        return row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _preflight_fixture() -> dict[str, Any]:
    return {
        "initial_positions": {"uav_00": [30.0, 100.0, 10.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [170.0, 100.0, 10.0]},
        "obstacle_specs": [
            {
                "geometry_type": "box",
                "obstacle_id": "blocking_initial_path",
                "center_m": [100.0, 100.0, 10.0],
                "size_m": [20.0, 20.0, 20.0],
                "velocity_mps": [0.0, 0.0, 0.0],
                "physical_collision_enabled": True,
            }
        ],
    }


def prepare_single_uav_ppo(config_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Write a fresh reset-only preparation manifest.

    The output directory must be new or empty.  The function creates no
    training data, trajectories, checkpoints, fitted normalizer, or optimizer
    updates.  ``action_contract_probe.csv`` is a synthetic contract probe and
    is explicitly marked as such.
    """

    config_file = Path(config_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty preflight output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_single_uav_ppo_config(config_file)
    environment = create_single_uav_ppo_environment(config)
    observations, info = environment.reset(seed=config.seed, scenario_config=_preflight_fixture())
    actor_contract = verify_actor_observation_contract(environment, observations)
    perturbation_probe = _controlled_reset_perturbation_probe(environment, observations)
    # This object is fitted only in memory from the reset fixture.  It is
    # intentionally absent from all persisted training/evaluation artifacts.
    temporary_probe_normalizer = ObservationNormalizer().fit(observations)
    normalized_actor_contract = verify_actor_observation_contract(
        environment, observations, normalizer=temporary_probe_normalizer
    )
    sampled_execution_probe = probe_sampled_execution_contract(
        environment, config, observations, info
    )

    resolved = config.to_dict()
    project_root = Path(__file__).resolve().parents[2]
    source_paths = (
        "src/experiments/single_uav_ppo_baseline.py", "src/cli.py",
        "src/learning/mappo.py", "src/learning/normalization.py",
        "src/simulation/environment.py", "src/simulation/motion_model.py",
        "src/simulation/obstacle_model.py", "src/simulation/graph_interface.py",
    )
    source_hashes = {name: _sha256_file(project_root / name) for name in source_paths}
    resolved.update(
        {
            "config_source": str(config_file.resolve()),
            "config_sha256": _sha256_file(config_file),
            "source_sha256": source_hashes,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "normalizer_status": "pending_train_fit",
            "normalizer_provenance": None,
            "temporary_probe_normalizer": "in_memory_reset_fixture_only; not training data",
            "reset_action_probe": sampled_execution_probe,
            "preflight_only": True,
            "training_performed": False,
            "environment_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "fixture": _preflight_fixture(),
        }
    )
    checks = {
        "config_lock": True,
        "from_scratch": bool(config.from_scratch),
        "historical_checkpoint_absent": config.historical_checkpoint is None,
        "all_assistance_disabled": all(
            not bool(getattr(config, key))
            for key in (
                "pass_commit_enabled",
                "lateral_exit_enabled",
                "radial_intent_enabled",
                "goal_recovery_correction_enabled",
                "goal_recovery_terminal_capture_enabled",
            )
        ),
        "actual_environment_config_graph": {
            "uav_count": environment.config.uav_count,
            "node_feature_dim": actor_contract["actual_actor_node_feature_dim"],
            "direction_features_enabled": environment.config.single_uav_obstacle_direction_features_enabled,
            "braking_features_enabled": environment.config.reward.single_uav_speed_control_mode == "braking_distance",
            "safety_projection_enabled": environment.config.safety_projection_enabled,
            "braking_safety_projection_enabled": environment.config.reward.single_uav_braking_safety_projection_enabled,
            "external_controller_instances": [
                name for name, value in vars(environment).items()
                if type(value).__module__.startswith("src.control.")
            ],
            "environment_class": f"{type(environment).__module__}.{type(environment).__name__}",
            "trainer_factory_class": f"{SampledActionLoggingPPOTrainer.__module__}.{SampledActionLoggingPPOTrainer.__name__}",
            "automatic_execution_logger": True,
        },
        "strict_success_contract": {
            "goal_distance_source": "actual_post_state_truth",
            "speed_source": "actual_post_state_truth",
            "goal_distance_m": config.single_uav_goal_threshold_m,
            "speed_mps": config.single_uav_goal_speed_threshold_mps,
            "consecutive_post_state_steps": config.strict_success_hold_steps,
            "pass_buffer_m": config.strict_pass_buffer_m,
            "minimum_clearance_m": config.strict_minimum_clearance_m,
            "collision_is_permanent_failure": True,
            "requires_relevant_physical_obstacle": True,
        },
        "sampled_action_logprob_contract": config.ppo_action_logprob_mode == "sampled",
        "actual_sampled_action_probe": bool(
            sampled_execution_probe["ratio_is_one"]
            and sampled_execution_probe["old_logprob_unchanged"]
            and sampled_execution_probe["buffer_transition_count"] == 1
            and sampled_execution_probe["environment_steps"] == 0
            and sampled_execution_probe["optimizer_steps"] == 0
            and sampled_execution_probe["backward_calls"] == 0
        ),
        "reset_only": True,
        "environment_steps": 0,
        "formal_training": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "actor_contract": actor_contract,
        "normalized_actor_contract": normalized_actor_contract,
        "controlled_reset_perturbation_probe": perturbation_probe,
        "sampled_execution_probe": sampled_execution_probe,
        "reset_info_strict_status": {
            key: value
            for key, value in info.items()
            if str(key).startswith("strict_") or key == "actor_node_feature_dim"
        },
    }
    (output / "config.lock").write_text(
        json.dumps({
            "config_sha256": resolved["config_sha256"], "baseline_id": BASELINE_ID,
            "source_sha256": source_hashes,
            "motion_action_contract": environment.config.motion.action_contract(),
            "evaluation_steps": config.max_steps,
            "strict_success_contract": checks["strict_success_contract"],
        }, indent=2),
        encoding="utf-8",
    )
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    (output / "checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    summary = {
        "baseline_id": BASELINE_ID,
        "status": "preflight_ready_pending_train_normalization",
        "config_sha256": resolved["config_sha256"],
        "normalizer_status": "pending_train_fit",
        "environment_steps": 0,
        "formal_training": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checks_passed": bool(
            checks["config_lock"]
            and checks["from_scratch"]
            and checks["historical_checkpoint_absent"]
            and checks["all_assistance_disabled"]
            and checks["sampled_action_logprob_contract"]
            and checks["actual_sampled_action_probe"]
            and checks["reset_only"]
            and not checks["formal_training"]
            and checks["environment_steps"] == 0
            and checks["backward_calls"] == 0
            and checks["optimizer_steps"] == 0
            and all(value for key, value in perturbation_probe.items() if key.endswith("_changed"))
            and sampled_execution_probe["old_logprob_unchanged"]
            and sampled_execution_probe["ratio_is_one"]
            and sampled_execution_probe["buffer_sampled_action_preserved"]
            and sampled_execution_probe["actor_received_normalized_features"]
            and not checks["actual_environment_config_graph"]["external_controller_instances"]
        ),
        "action_contract_probe": "actual_untrained_policy_sample_reconcile_buffer_probe; no environment transition or trajectory",
    }
    (output / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "action_contract_probe.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "probe_kind",
                "sampled_action",
                "executed_action",
                "safety_correction",
                "total_correction",
                "logprob_action",
                "old_logprob_unchanged",
                "importance_ratio",
                "environment_step",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "probe_kind": sampled_execution_probe["probe_tag"],
                "sampled_action": json.dumps(sampled_execution_probe["sampled_delta_v"]),
                "executed_action": json.dumps(sampled_execution_probe["executed_delta_v"]),
                "safety_correction": json.dumps(sampled_execution_probe["safety_correction"]),
                "total_correction": json.dumps(sampled_execution_probe["total_correction"]),
                "logprob_action": sampled_execution_probe["logprob_action"],
                "old_logprob_unchanged": int(sampled_execution_probe["old_logprob_unchanged"]),
                "importance_ratio": sampled_execution_probe["importance_ratio"],
                "environment_step": 0,
            }
        )
    return {"ok": bool(summary["checks_passed"]), "output_root": str(output.resolve()), "summary": summary, "checks": checks}


# Friendly aliases keep the preparation API discoverable without creating a
# second implementation or a second historical environment path.
SingleUAVPPOEnv = StrictSingleUAVPPOEnv
strict_success_gate = evaluate_strict_success
build_single_uav_ppo_environment = create_single_uav_ppo_environment


__all__ = [
    "ActionExecutionCSVLogger",
    "BASELINE_ID",
    "EXPECTED_NODE_FEATURE_DIM",
    "FEATURE_INDEX_MAP",
    "SingleUAVPPOConfig",
    "StrictSingleUAVPPOEnv",
    "StrictSuccessResult",
    "SingleUAVPPOEnv",
    "FrozenFeatureNormalizer",
    "FrozenObservationNormalizer",
    "build_single_uav_simulation_config",
    "create_single_uav_ppo_environment",
    "create_single_uav_ppo_trainer",
    "evaluate_strict_success",
    "strict_success_gate",
    "build_single_uav_ppo_environment",
    "load_single_uav_ppo_config",
    "prepare_single_uav_ppo",
    "reconcile_single_uav_sampled_execution",
    "probe_sampled_execution_contract",
    "verify_actor_observation_contract",
]
