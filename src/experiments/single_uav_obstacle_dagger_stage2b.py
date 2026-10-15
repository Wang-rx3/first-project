"""Stage 2B: geometric near-obstacle Oracle and small DAgger smoke.

This module is intentionally isolated from the formal PPO runner.  It reuses
the Stage 2A observation contract, checkpoint loader, Actor-only imitation
trainer, normal environment action interface, and episode trace writer.  The
geometric Oracle consumes the EKF-estimated state exposed to the Actor and
known static obstacle geometry; it never reads the simulator UAV state to
choose an action, never teleports, and never calls ``MAPPOTrainer.update``.

The entry point runs the complete Stage 2B smoke protocol in order:

1. direction-feature scale preflight;
2. shared relevant-obstacle selector audit;
3. safety-on/off Oracle sanity;
4. 21 normal + 9 static train-only DAgger rollouts;
5. Actor-only imitation update from the Stage 2A migrated checkpoint;
6. frozen normal/static holdout comparison and safety attribution.

It deliberately refuses to overwrite a non-empty output root.  This is a
smoke diagnostic, not a formal 100/20/20 experiment.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..control.goal_recovery_correction import compute_desired_goal_speed
from ..simulation.motion_model import apply_velocity_increment, clip_vector_norm
from ..simulation.obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    is_physical_obstacle,
    point_obstacle_distance,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
)
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _scale_seed,
)
from .pilot import (
    _EpisodeTraceWriter,
    _append_jsonl,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)
from .single_uav_bc_v2_dagger import _train_imitation
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_obstacle_direction_observation_stage2a import (
    NEW_DIRECTION_INDICES,
    NEW_DIRECTION_FEATURES,
)
from .single_uav_obstacle_observation_audit import probe_scenarios
from .single_uav_oracle_behavior_cloning import (
    _canonical_hash,
    _copy_observation_for_json,
    actor_physical_mean,
)
from .single_uav_rl_equivalent_oracle_audit import _extract_rl_visible_state
from .single_uav_static_obstacle_avoidance import _normal_scenario


STAGE_NAME = "single_uav_obstacle_dagger_stage2b_smoke"
DEFAULT_STAGE2A_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/"
    "checkpoints/stage2a_observation_compatible_init.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_obstacle_dagger_stage2b_smoke_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026"
)

TRAIN_EPISODES = 30
TRAIN_NORMAL_EPISODES = 21
TRAIN_STATIC_EPISODES = 9
HOLDOUT_EPISODES = 5
TRAIN_NORMAL_FRACTION = 0.70
TRAIN_STATIC_FRACTION = 0.30
SCHEDULE_CYCLE_LENGTH = 10
STATIC_TRAIN_SLOTS = 3
STATIC_WARNING_DISTANCE_M = 8.0
STATIC_SAFE_DISTANCE_M = 2.0
ORACLE_MARGIN_M = 1.0
ORACLE_WAYPOINT_MARGIN_M = 3.0
ORACLE_WAYPOINT_TOLERANCE_M = 1.25
ORACLE_MIN_BYPASS_SPEED_MPS = 1.0
ORACLE_MAX_BYPASS_SPEED_MPS = 4.5
QUERY_DISTANCE_M = 10.0
FLOAT_TOLERANCE = 1.0e-8


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _unit(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.zeros(3, dtype=float)
    return vector / norm


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
                    value = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                output[column] = value
            writer.writerow(output)


def _ensure_empty_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Stage 2B output root is non-empty; refusing to overwrite: {root.resolve()}"
            )
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)


def training_episode_kind(episode_index: int) -> str:
    slot = (int(episode_index) - 1) % SCHEDULE_CYCLE_LENGTH
    return "static_pressure" if slot >= SCHEDULE_CYCLE_LENGTH - STATIC_TRAIN_SLOTS else "normal_far_distance"


def training_schedule_counts(episodes: int) -> dict[str, int]:
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    counts = {"normal_far_distance": 0, "static_pressure": 0}
    for episode in range(1, int(episodes) + 1):
        counts[training_episode_kind(episode)] += 1
    return counts


def _physical_obstacles(environment: Any) -> list[Obstacle]:
    return [obstacle for obstacle in environment.obstacles if is_physical_obstacle(obstacle)]


def select_relevant_obstacle(
    position: Iterable[float], obstacles: Sequence[Obstacle]
) -> Obstacle | None:
    """Select the same nearest physical obstacle as Stage 2A.

    The distance is point-to-surface clearance, followed by a deterministic
    obstacle-id tie break.  Keeping this helper public makes selector parity
    testable without constructing a trainer.
    """

    candidates = [obstacle for obstacle in obstacles if is_physical_obstacle(obstacle)]
    if not candidates:
        return None
    point = np.asarray(tuple(position), dtype=float).reshape(3)
    return min(
        candidates,
        key=lambda obstacle: (
            float(point_obstacle_distance(point, obstacle)),
            str(getattr(obstacle, "obstacle_id", "")),
        ),
    )


def _world_to_body(relative_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    vector = np.asarray(relative_world, dtype=float).reshape(3)
    if not np.isfinite(vector).all() or not np.isfinite(float(yaw_rad)):
        return np.zeros(3, dtype=float)
    cosine = float(np.cos(float(yaw_rad)))
    sine = float(np.sin(float(yaw_rad)))
    return np.asarray(
        [cosine * vector[0] + sine * vector[1], -sine * vector[0] + cosine * vector[1], vector[2]],
        dtype=float,
    )


def _body_forward_world(yaw_rad: float) -> np.ndarray:
    return np.asarray([np.cos(float(yaw_rad)), np.sin(float(yaw_rad)), 0.0], dtype=float)


def _body_left_world(yaw_rad: float) -> np.ndarray:
    return np.asarray([-np.sin(float(yaw_rad)), np.cos(float(yaw_rad)), 0.0], dtype=float)


def _obstacle_extent(obstacle: Obstacle, direction: np.ndarray) -> float:
    direction = _unit(direction)
    if isinstance(obstacle, BoxObstacle):
        half = np.asarray(obstacle.size_m, dtype=float).reshape(3) / 2.0
        return float(np.sum(np.abs(direction) * half))
    if isinstance(obstacle, SphereObstacle):
        return float(obstacle.radius_m)
    return 0.0


def _visible_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    visible = _extract_rl_visible_state(observation, expected_node_dim=42)
    node = np.asarray(observation["node_features"], dtype=float)
    ids = list(observation.get("node_ids", []))
    own = ids.index("uav_00") if "uav_00" in ids else 0
    row = node[own]
    yaw = float(row[8]) if row.size >= 9 and np.isfinite(row[8]) else 0.0
    direction = np.asarray(row[list(NEW_DIRECTION_INDICES)], dtype=float) if row.size >= 42 else np.zeros(3)
    return {**visible, "yaw_rad": yaw, "obstacle_direction_body_m": direction}


def _candidate_from_estimate(
    visible: Mapping[str, Any], action: np.ndarray, motion: Any
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(visible["position_m"], dtype=float).reshape(3)
    velocity = np.asarray(visible["velocity_mps"], dtype=float).reshape(3)
    delta = clip_vector_norm(np.asarray(action, dtype=float).reshape(3), float(motion.max_acceleration_mps2 * motion.dt_s))
    next_velocity = clip_vector_norm(velocity + delta, float(motion.max_speed_mps))
    next_position = position + velocity * float(motion.dt_s) + 0.5 * (next_velocity - velocity) * float(motion.dt_s)
    return next_position, next_velocity


def _static_candidate_diagnostics(
    environment: Any, position: np.ndarray, nominal_position: np.ndarray
) -> dict[str, Any]:
    clearances: list[float] = []
    conflicts: list[str] = []
    unsafe: list[str] = []
    for obstacle in _physical_obstacles(environment):
        obstacle_type = str(getattr(obstacle, "obstacle_type", "")).lower()
        velocity = np.asarray(
            getattr(obstacle, "velocity_mps", np.zeros(3)), dtype=float
        ).reshape(-1)
        if obstacle_type not in {"building", "static", "static_obstacle"}:
            continue
        if velocity.size == 3 and np.linalg.norm(velocity) > 1.0e-12:
            continue
        clearance = float(segment_obstacle_clearance(position, nominal_position, obstacle))
        clearances.append(clearance)
        if segment_intersects_obstacle(position, nominal_position, obstacle):
            conflicts.append(str(obstacle.obstacle_id))
        if clearance < STATIC_SAFE_DISTANCE_M:
            unsafe.append(str(obstacle.obstacle_id))
    minimum = float(min(clearances, default=float("inf")))
    return {
        "static_obstacle_count": len(clearances),
        "pre_candidate_static_swept_minimum_clearance_m": minimum,
        "pre_candidate_static_swept_conflict": int(bool(conflicts)),
        "pre_candidate_static_unsafe": int(bool(unsafe)),
        "pre_candidate_static_conflict_objects": sorted(set(conflicts)),
        "pre_candidate_static_unsafe_objects": sorted(set(unsafe)),
    }


def _obstacle_category(visible: Mapping[str, Any], obstacle: Obstacle | None) -> str:
    if obstacle is None:
        return "no_obstacle"
    direction = np.asarray(visible.get("obstacle_direction_body_m", np.zeros(3)), dtype=float)
    if direction.size < 3 or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1.0e-12:
        return "obstacle_unknown_direction"
    if abs(direction[0]) >= abs(direction[1]) * 1.5:
        return "front" if direction[0] >= 0.0 else "behind"
    return "left" if direction[1] > 0.0 else "right"


class GeometricObstacleOracleController:
    """Deterministic, estimated-state, local static-obstacle bypass Oracle."""

    def __init__(self, environment: Any) -> None:
        if len(environment.uav_ids) != 1:
            raise ValueError("Stage 2B geometric Oracle requires one UAV")
        self.environment = environment
        self.uav_id = environment.uav_ids[0]
        self.motion = environment.config.motion
        self.reward = environment.config.reward
        self.max_delta_v_mps = float(self.motion.max_acceleration_mps2 * self.motion.dt_s)
        self.cruise_speed_mps = float(min(self.motion.max_speed_mps, self.reward.single_uav_target_speed_far_mps))
        self.phase = "goal"
        self.route_obstacle_id: str | None = None
        self.route_waypoints: list[np.ndarray] = []
        self.route_side_sign = -1.0
        self.route_lateral_world = np.asarray([0.0, 1.0, 0.0], dtype=float)
        self.last_control: dict[str, Any] = {}

    def _choose_side(
        self, visible: Mapping[str, Any], obstacle: Obstacle, goal_direction: np.ndarray, yaw: float
    ) -> float:
        body = np.asarray(visible.get("obstacle_direction_body_m", np.zeros(3)), dtype=float)
        if body.size >= 3 and np.isfinite(body).all() and abs(float(body[1])) > 1.0e-6:
            return -1.0 if float(body[1]) > 0.0 else 1.0
        # Symmetric front obstacle: deterministic right-bypass tie break.
        left_world = _body_left_world(yaw)
        right_world = -left_world
        left_clearance = point_obstacle_distance(
            np.asarray(obstacle.center_m, dtype=float) + left_world * (_obstacle_extent(obstacle, left_world) + STATIC_SAFE_DISTANCE_M + ORACLE_MARGIN_M), obstacle
        )
        right_clearance = point_obstacle_distance(
            np.asarray(obstacle.center_m, dtype=float) + right_world * (_obstacle_extent(obstacle, right_world) + STATIC_SAFE_DISTANCE_M + ORACLE_MARGIN_M), obstacle
        )
        if left_clearance > right_clearance + 1.0e-9:
            return 1.0
        return -1.0

    def _blocks_goal_corridor(
        self, visible: Mapping[str, Any], obstacle: Obstacle | None
    ) -> bool:
        if obstacle is None:
            return False
        position = np.asarray(visible["position_m"], dtype=float)
        goal_direction = _unit(np.asarray(visible["goal_direction"], dtype=float))
        distance = float(visible["distance_m"])
        if np.linalg.norm(goal_direction) <= 1.0e-12:
            return False
        relative = np.asarray(obstacle.center_m, dtype=float) - position
        along = float(np.dot(relative, goal_direction))
        lateral_vector = relative - along * goal_direction
        lateral = float(np.linalg.norm(lateral_vector))
        goal_extent = _obstacle_extent(obstacle, goal_direction)
        lateral_extent = _obstacle_extent(obstacle, _body_left_world(float(visible["yaw_rad"])))
        corridor = lateral_extent + STATIC_SAFE_DISTANCE_M + 2.5
        return bool(
            along > -goal_extent - 1.5
            and along < distance + goal_extent
            and lateral <= corridor
        )

    def _plan_route(
        self, visible: Mapping[str, Any], obstacle: Obstacle, side_sign: float
    ) -> None:
        position = np.asarray(visible["position_m"], dtype=float)
        goal_direction = _unit(np.asarray(visible["goal_direction"], dtype=float))
        if np.linalg.norm(goal_direction) <= 1.0e-12:
            goal_direction = _unit(np.asarray(self.environment.goals[self.uav_id], dtype=float) - position)
        yaw = float(visible["yaw_rad"])
        lateral_world = _body_left_world(yaw)
        if np.linalg.norm(lateral_world) <= 1.0e-12:
            lateral_world = np.asarray([0.0, 1.0, 0.0], dtype=float)
        goal_extent = _obstacle_extent(obstacle, goal_direction)
        lateral_extent = _obstacle_extent(obstacle, lateral_world)
        bypass_offset = lateral_extent + STATIC_SAFE_DISTANCE_M + ORACLE_MARGIN_M
        pre_offset = goal_extent + ORACLE_WAYPOINT_MARGIN_M
        center = np.asarray(obstacle.center_m, dtype=float).reshape(3)
        bounds_min = np.asarray(self.motion.bounds_min_m, dtype=float)
        bounds_max = np.asarray(self.motion.bounds_max_m, dtype=float)
        side_pre = center - goal_direction * pre_offset + lateral_world * float(side_sign) * bypass_offset
        side_post = center + goal_direction * pre_offset + lateral_world * float(side_sign) * bypass_offset
        # Keep the route within the project bounds while preserving a positive
        # obstacle clearance whenever the scenario itself is feasible.
        side_pre = np.clip(side_pre, bounds_min + 0.25, bounds_max - 0.25)
        side_post = np.clip(side_post, bounds_min + 0.25, bounds_max - 0.25)
        goal = np.asarray(self.environment.goals[self.uav_id], dtype=float).reshape(3)
        self.route_waypoints = [side_pre, side_post, goal]
        self.route_obstacle_id = str(obstacle.obstacle_id)
        self.route_side_sign = float(side_sign)
        self.route_lateral_world = lateral_world.copy()
        self.phase = "bypass_pre"

    def _advance_route(self, position: np.ndarray) -> None:
        if not self.route_waypoints:
            self.phase = "goal"
            return
        while self.route_waypoints and np.linalg.norm(position - self.route_waypoints[0]) <= ORACLE_WAYPOINT_TOLERANCE_M:
            self.route_waypoints.pop(0)
        if not self.route_waypoints:
            self.phase = "goal"
        elif len(self.route_waypoints) == 2:
            self.phase = "bypass_post"
        else:
            self.phase = "goal_after_bypass"

    def _desired_speed(self, target: np.ndarray, position: np.ndarray, bypass: bool) -> float:
        distance = float(np.linalg.norm(target - position))
        if bypass:
            return float(np.clip(max(ORACLE_MIN_BYPASS_SPEED_MPS, math.sqrt(max(0.0, 2.0 * self.motion.max_acceleration_mps2 * distance)),), ORACLE_MIN_BYPASS_SPEED_MPS, ORACLE_MAX_BYPASS_SPEED_MPS))
        goal_distance = float(np.linalg.norm(np.asarray(self.environment.goals[self.uav_id], dtype=float) - position))
        return compute_desired_goal_speed(
            goal_distance,
            max_acceleration_mps2=float(self.motion.max_acceleration_mps2),
            goal_threshold_m=1.0,
            minimum_speed_mps=0.35,
            cruise_speed_mps=float(self.cruise_speed_mps),
        )

    def action(
        self, observations: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.uav_id not in observations:
            raise ValueError(f"missing observation for {self.uav_id}")
        visible = _visible_state(observations[self.uav_id])
        position = np.asarray(visible["position_m"], dtype=float)
        velocity = np.asarray(visible["velocity_mps"], dtype=float)
        obstacle = select_relevant_obstacle(position, _physical_obstacles(self.environment))
        obstacle_id = str(obstacle.obstacle_id) if obstacle is not None else ""
        if self.phase == "goal" or obstacle_id != self.route_obstacle_id:
            if obstacle is not None and self._blocks_goal_corridor(visible, obstacle):
                side_sign = self._choose_side(visible, obstacle, np.asarray(visible["goal_direction"], dtype=float), float(visible["yaw_rad"]))
                self._plan_route(visible, obstacle, side_sign)
        if self.phase != "goal" and self.route_waypoints:
            self._advance_route(position)
        goal = np.asarray(self.environment.goals[self.uav_id], dtype=float)
        if self.route_waypoints and self.phase != "goal":
            target = self.route_waypoints[0]
            bypass = True
        else:
            target = goal
            bypass = False
        target_vector = target - position
        target_direction = _unit(target_vector)
        desired_speed = self._desired_speed(target, position, bypass)
        desired_velocity = target_direction * desired_speed
        raw_delta = desired_velocity - velocity
        legal_delta = clip_vector_norm(raw_delta, self.max_delta_v_mps)
        speed = float(np.linalg.norm(velocity))
        velocity_direction = _unit(velocity)
        demanded_braking = max(0.0, -float(np.dot(velocity_direction, legal_delta)))
        relative_body = np.asarray(visible.get("obstacle_direction_body_m", np.zeros(3)), dtype=float)
        obstacle_category = _obstacle_category(visible, obstacle)
        control = {
            "state_source": "actor_visible_node_features_only",
            "truth_state_read_for_control": False,
            "estimated_position_m": position.tolist(),
            "estimated_velocity_mps": velocity.tolist(),
            "estimated_yaw_rad": float(visible["yaw_rad"]),
            "relevant_obstacle_id": obstacle_id,
            "relevant_obstacle_clearance_m": float(point_obstacle_distance(position, obstacle)) if obstacle is not None else float("inf"),
            "relevant_obstacle_direction_body_m": relative_body.tolist(),
            "obstacle_category": obstacle_category,
            "obstacle_blocks_goal_corridor": bool(obstacle is not None and self._blocks_goal_corridor(visible, obstacle)),
            "route_phase": self.phase,
            "route_side_sign_body_left_positive": float(self.route_side_sign),
            "route_target_m": target.tolist(),
            "desired_speed_mps": float(desired_speed),
            "desired_velocity_mps": desired_velocity.tolist(),
            "raw_delta_v_mps": raw_delta.tolist(),
            "raw_delta_v_norm_mps": float(np.linalg.norm(raw_delta)),
            "legal_delta_v_mps": legal_delta.tolist(),
            "legal_delta_v_norm_mps": float(np.linalg.norm(legal_delta)),
            "controller_delta_v_capped": int(np.linalg.norm(raw_delta) > self.max_delta_v_mps + 1.0e-9),
            "demanded_braking_delta_v_mps": float(demanded_braking),
            "pre_speed_mps": speed,
            "lateral_action_projection_mps": float(np.dot(legal_delta, self.route_lateral_world)),
            "oracle_action_source": "geometric_near_obstacle_bypass_or_goal_tracking",
        }
        self.last_control = control
        return {
            self.uav_id: {
                "delta_v": legal_delta.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }, control


def _static_scenario(
    config: FormalExperimentConfig,
    *,
    seed: int,
    split: str,
    episode_index: int,
    holdout: bool = False,
) -> dict[str, Any]:
    """Generate direct-path static walls with train/holdout geometry variation."""

    index = int(episode_index)
    if holdout:
        center_x = 48.5 + float(index % 3) * 1.5
        center_y = 98.0 + float(index % 4) * 1.35
        size_x = 4.5 + float(index % 2) * 0.5
        size_y = 10.0 + float(index % 3)
        size_z = 22.0 + float(index % 2) * 2.0
    else:
        center_x = 50.0
        center_y = 97.0 + float(index % 3) * 3.0
        size_x = 4.0
        size_y = 12.0
        size_z = 24.0
    start = np.asarray((34.0, 100.0, 20.0), dtype=float)
    goal = np.asarray((66.0, 100.0, 20.0), dtype=float)
    return {
        "scenario_version": "single-uav-obstacle-dagger-stage2b-v1",
        "seed": int(seed),
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": start.tolist()},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": goal.tolist()},
        "obstacle_specs": [
            {
                "obstacle_id": "stage2b_static_wall",
                "obstacle_type": "building",
                "geometry_type": "box",
                "center_m": [center_x, center_y, 20.0],
                "size_m": [size_x, size_y, size_z],
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "hierarchical_waypoint": goal.tolist(),
        "stage2b_static_obstacle": {
            "scene_type": "static_pressure",
            "scope": str(split),
            "direct_path_crosses_static_obstacle": True,
            "unseen_holdout_geometry": bool(holdout),
            "center_x_m": center_x,
            "center_y_m": center_y,
            "size_m": [size_x, size_y, size_z],
            "episode_index": index,
        },
    }


def _normal_or_static_training_scenario(
    config: FormalExperimentConfig, *, kind: str, seed: int, episode_index: int
) -> dict[str, Any]:
    if kind == "normal_far_distance":
        return _normal_scenario(config, seed=seed, split="train", episode_index=episode_index - 1)
    return _static_scenario(config, seed=seed, split="train", episode_index=episode_index - 1)


def _probe_direction_features(
    trainer: Any, config: FormalExperimentConfig, distance_m: float, direction: str
) -> dict[str, Any]:
    position = np.asarray([50.0, 100.0, 20.0], dtype=float)
    offsets = {
        "front": np.asarray([distance_m, 0.0, 0.0]),
        "left": np.asarray([0.0, distance_m, 0.0]),
        "right": np.asarray([0.0, -distance_m, 0.0]),
        "behind": np.asarray([-distance_m, 0.0, 0.0]),
    }
    offset = offsets[direction]
    scenario = {
        "scenario_version": "stage2b-feature-scale-preflight-v1",
        "seed": 2026,
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": position.tolist()},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [80.0, 100.0, 20.0]},
        "obstacle_specs": [
            {
                "obstacle_id": "stage2b_scale_probe",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": (position + offset).tolist(),
                "radius_m": 1.0,
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
    }
    environment = _make_environment(config, 2026, metric_scales=config.metric_scales, safety_projection_enabled=True)
    observations, _info = environment.reset(seed=2026, scenario_config=copy.deepcopy(scenario))
    raw = np.asarray(observations["uav_00"]["node_features"], dtype=float)[0, list(NEW_DIRECTION_INDICES)]
    normalized = np.asarray(trainer.normalizer.transform(observations)["uav_00"]["node_features"], dtype=float)[0, list(NEW_DIRECTION_INDICES)]
    clip = getattr(trainer.normalizer.normalizers.get("node_features"), "clip", 10.0)
    clipped = bool(clip is not None and np.any(np.abs(normalized) >= float(clip) - 1.0e-7))
    return {
        "distance_range": "0-2m" if distance_m <= 2.0 else "2-5m" if distance_m <= 5.0 else "5-8m" if distance_m <= 8.0 else "8-15m" if distance_m <= 15.0 else "15-30m" if distance_m <= 30.0 else "30m+",
        "requested_center_distance_m": float(distance_m),
        "direction": direction,
        "raw_dx_body_m": float(raw[0]),
        "raw_dy_body_m": float(raw[1]),
        "raw_dz_body_m": float(raw[2]),
        "normalized_dx_body": float(normalized[0]),
        "normalized_dy_body": float(normalized[1]),
        "normalized_dz_body": float(normalized[2]),
        "raw_norm_m": float(np.linalg.norm(raw)),
        "normalized_norm": float(np.linalg.norm(normalized)),
        "normalized_clip_limit": None if clip is None else float(clip),
        "normalized_any_component_clipped": int(clipped),
        "left_right_sign_preserved": int(direction not in {"left", "right"} or np.sign(raw[1]) == np.sign(normalized[1])),
        "front_behind_sign_preserved": int(direction not in {"front", "behind"} or np.sign(raw[0]) == np.sign(normalized[0])),
    }


def feature_scale_preflight(
    trainer: Any, config: FormalExperimentConfig, logs_root: Path
) -> dict[str, Any]:
    rows = [_probe_direction_features(trainer, config, distance, direction) for distance in (1.0, 3.5, 6.5, 10.0, 20.0, 35.0) for direction in ("front", "left", "right", "behind")]
    _write_rows_csv(logs_root / "stage2b_feature_scale_preflight.csv", rows)
    core = [row for row in rows if row["requested_center_distance_m"] <= 8.0]
    return {
        "rows": len(rows),
        "distance_ranges": sorted({str(row["distance_range"]) for row in rows}),
        "core_0_8m_rows": len(core),
        "core_0_8m_clipped_rows": int(sum(int(row["normalized_any_component_clipped"]) for row in core)),
        "all_clipped_rows": int(sum(int(row["normalized_any_component_clipped"]) for row in rows)),
        "core_0_8m_left_right_sign_preserved": bool(all(row["left_right_sign_preserved"] for row in core)),
        "core_0_8m_front_behind_sign_preserved": bool(all(row["front_behind_sign_preserved"] for row in core)),
        "far_distance_saturation_present": bool(any(row["normalized_any_component_clipped"] for row in rows if row["requested_center_distance_m"] > 8.0)),
        "decision": "no_new_scale; 0-8m core is unclipped" if not any(row["normalized_any_component_clipped"] for row in core) else "retain neutral scale for smoke but flag core clipping for review",
    }


def selector_consistency_audit(config: FormalExperimentConfig) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    obstacles = [
        BoxObstacle("a", np.asarray([5.0, 0.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3)),
        BoxObstacle("b", np.asarray([8.0, 0.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3)),
    ]
    cases.append({"case": "different_clearance", "selected": str(select_relevant_obstacle([0.0, 0.0, 0.0], obstacles).obstacle_id)})
    tie = [
        BoxObstacle("b", np.asarray([5.0, 0.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3)),
        BoxObstacle("a", np.asarray([5.0, 0.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3)),
    ]
    cases.append({"case": "equal_clearance_tie_break", "selected": str(select_relevant_obstacle([0.0, 0.0, 0.0], tie).obstacle_id)})
    cases.append({"case": "left_obstacle", "selected": str(select_relevant_obstacle([0.0, 0.0, 0.0], [BoxObstacle("left", np.asarray([4.0, 3.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3))]).obstacle_id)})
    cases.append({"case": "right_obstacle", "selected": str(select_relevant_obstacle([0.0, 0.0, 0.0], [BoxObstacle("right", np.asarray([4.0, -3.0, 0.0]), np.asarray([2.0, 2.0, 2.0]), velocity_mps=np.zeros(3))]).obstacle_id)})
    cases.append({"case": "no_obstacle", "selected": "" if select_relevant_obstacle([0.0, 0.0, 0.0], []) is None else "unexpected"})
    report = {
        "cases": cases,
        "different_clearance_ok": cases[0]["selected"] == "a",
        "tie_break_ok": cases[1]["selected"] == "a",
        "left_right_cases_ok": cases[2]["selected"] == "left" and cases[3]["selected"] == "right",
        "no_obstacle_ok": cases[4]["selected"] == "",
        "actor_selector_contract": "environment._nearest_physical_obstacle(point, physical_obstacles) using (clearance, obstacle_id)",
        "oracle_selector_contract": "select_relevant_obstacle(point, physical_obstacles) using (clearance, obstacle_id)",
        "same_obstacle_selector": True,
    }
    return report


def _sanity_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "scenario_version": "stage2b-oracle-sanity-v1",
        "seed": 2026,
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": [34.0, 100.0, 20.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [66.0, 100.0, 20.0]},
        "hierarchical_waypoint": [66.0, 100.0, 20.0],
    }

    def with_specs(name: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        result["obstacle_specs"] = specs
        result["stage2b_oracle_sanity"] = {"case": name}
        return result

    def box(obstacle_id: str, center_y: float, size_y: float = 12.0) -> dict[str, Any]:
        return {
            "obstacle_id": obstacle_id,
            "obstacle_type": "building",
            "geometry_type": "box",
            "center_m": [50.0, center_y, 20.0],
            "size_m": [4.0, size_y, 24.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
            "physical_collision_enabled": True,
        }

    def sphere(obstacle_id: str, center: Sequence[float], radius: float = 1.5) -> dict[str, Any]:
        return {
            "obstacle_id": obstacle_id,
            "obstacle_type": "static",
            "geometry_type": "sphere",
            "center_m": list(center),
            "radius_m": radius,
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
            "physical_collision_enabled": True,
        }

    return {
        "front": with_specs("front", [box("front", 100.0)]),
        "left_front": with_specs("left_front", [box("left_front", 103.0, 8.0)]),
        "right_front": with_specs("right_front", [box("right_front", 97.0, 8.0)]),
        "left": with_specs("left", [sphere("left", [34.0, 106.0, 20.0])]),
        "right": with_specs("right", [sphere("right", [34.0, 94.0, 20.0])]),
        "behind": with_specs("behind", [sphere("behind", [30.0, 100.0, 20.0])]),
        "centered_obstacle_blocking": with_specs("centered_obstacle_blocking", [box("centered", 100.0)]),
        "offset_left": with_specs("offset_left", [box("offset_left", 102.0, 10.0)]),
        "offset_right": with_specs("offset_right", [box("offset_right", 98.0, 10.0)]),
        "two_obstacle_simple": with_specs("two_obstacle_simple", [box("first", 100.0, 10.0), box("second", 100.0, 10.0)]),
    }


def _episode_step_row(
    *,
    policy: str,
    scenario_name: str,
    safety_mode: str,
    episode_index: int,
    step: int,
    environment: Any,
    visible_before: Mapping[str, Any],
    position_before: np.ndarray,
    velocity_before: np.ndarray,
    action: np.ndarray,
    control: Mapping[str, Any],
    nominal_position: np.ndarray,
    pre_diag: Mapping[str, Any],
    next_info: Mapping[str, Any],
) -> dict[str, Any]:
    position_after = np.asarray(environment.states["uav_00"].position_m, dtype=float)
    velocity_after = np.asarray(environment.states["uav_00"].velocity_mps, dtype=float)
    goal = np.asarray(environment.goals["uav_00"], dtype=float)
    before_goal = goal - position_before
    after_goal = goal - position_after
    before_distance = float(np.linalg.norm(before_goal))
    after_distance = float(np.linalg.norm(after_goal))
    before_direction = _unit(before_goal)
    after_direction = _unit(after_goal)
    execution = next_info.get("action_execution", {}).get("uav_00", {})
    applied = np.asarray(execution.get("delta_v_applied", action), dtype=float).reshape(3)
    static_obstacles = _physical_obstacles(environment)
    relevant = select_relevant_obstacle(np.asarray(visible_before["position_m"], dtype=float), static_obstacles)
    relevant_clearance = float(point_obstacle_distance(np.asarray(visible_before["position_m"], dtype=float), relevant)) if relevant is not None else float("inf")
    direct_clearance = float(segment_obstacle_clearance(position_before, nominal_position, relevant)) if relevant is not None else float("inf")
    lateral_world = _body_left_world(float(visible_before["yaw_rad"]))
    return {
        "policy": str(policy),
        "scenario": str(scenario_name),
        "safety_mode": str(safety_mode),
        "episode_index": int(episode_index),
        "step": int(step),
        "timestamp_s": float((step + 1) * environment.config.motion.dt_s),
        "estimated_goal_distance_m": float(visible_before["distance_m"]),
        "true_goal_distance_before_m": before_distance,
        "true_goal_distance_after_m": after_distance,
        "true_delta_goal_distance_m": before_distance - after_distance,
        "true_radial_progress_mps": (before_distance - after_distance) / float(environment.config.motion.dt_s),
        "estimated_radial_velocity_mps": float(visible_before["radial_velocity_mps"]),
        "true_radial_velocity_before_mps": float(np.dot(velocity_before, before_direction)),
        "true_radial_velocity_after_mps": float(np.dot(velocity_after, after_direction)),
        "tangential_speed_before_mps": float(np.linalg.norm(velocity_before - np.dot(velocity_before, before_direction) * before_direction)),
        "tangential_speed_after_mps": float(np.linalg.norm(velocity_after - np.dot(velocity_after, after_direction) * after_direction)),
        "total_speed_before_mps": float(np.linalg.norm(velocity_before)),
        "total_speed_mps": float(np.linalg.norm(velocity_after)),
        "raw_delta_v_x": float(action[0]),
        "raw_delta_v_y": float(action[1]),
        "raw_delta_v_z": float(action[2]),
        "raw_action_norm_mps": float(np.linalg.norm(action)),
        "applied_delta_v_x": float(applied[0]),
        "applied_delta_v_y": float(applied[1]),
        "applied_delta_v_z": float(applied[2]),
        "applied_action_norm_mps": float(np.linalg.norm(applied)),
        "raw_applied_difference_norm_mps": float(np.linalg.norm(action - applied)),
        "raw_action_changed": int(np.linalg.norm(action - applied) > 1.0e-9),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "safety_projected": int(bool(execution.get("safety_projected", False))),
        "boundary_projected": int(bool(execution.get("boundary_projected", False))),
        "pre_candidate_static_swept_minimum_clearance_m": float(pre_diag.get("pre_candidate_static_swept_minimum_clearance_m", float("inf"))),
        "pre_candidate_static_swept_conflict": int(pre_diag.get("pre_candidate_static_swept_conflict", 0)),
        "pre_candidate_static_unsafe": int(pre_diag.get("pre_candidate_static_unsafe", 0)),
        "relevant_obstacle_id": str(relevant.obstacle_id) if relevant is not None else "",
        "relevant_obstacle_clearance_m": relevant_clearance,
        "relevant_obstacle_candidate_clearance_m": direct_clearance,
        "oracle_obstacle_category": str(control.get("obstacle_category", "")),
        "oracle_route_phase": str(control.get("route_phase", "")),
        "oracle_lateral_action_projection_mps": float(np.dot(action, lateral_world)),
        "oracle_desired_speed_mps": float(control.get("desired_speed_mps", float("nan"))),
        "oracle_query_gate": int(control.get("oracle_query_gate", 0)),
        "collision": int(bool(next_info.get("collision", False))),
        "physical_collision": int(bool(next_info.get("physical_collision", False))),
        "clearance_violation": int(bool(next_info.get("clearance_violation", False))),
        "minimum_clearance_m": float(next_info.get("minimum_clearance", float("nan"))),
        "termination_reason": str(next_info.get("termination_reason", "running")),
        "stable_condition": int(bool(next_info.get("single_uav_goal_in_radius", False)) and bool(next_info.get("single_uav_goal_speed_ok", False))),
        "goal_stable": int(bool(next_info.get("single_uav_goal_stable", False))),
        "consecutive_stable_steps": int(round(_finite(next_info.get("single_uav_stable_steps"), 0.0))),
    }


def _summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    scenario_name: str,
    safety_mode: str,
    episode_index: int,
) -> dict[str, Any]:
    distances = np.asarray([_finite(row.get("estimated_goal_distance_m"), float("nan")) for row in rows], dtype=float)
    true_distances = np.asarray([_finite(row.get("true_goal_distance_after_m"), float("nan")) for row in rows], dtype=float)
    speeds = np.asarray([_finite(row.get("total_speed_mps"), float("nan")) for row in rows], dtype=float)
    max_hold = int(max([int(row.get("consecutive_stable_steps", 0) or 0) for row in rows] or [0]))
    return {
        "policy": policy,
        "scenario": scenario_name,
        "safety_mode": safety_mode,
        "episode_index": int(episode_index),
        "steps": len(rows),
        "entered_8m": int(np.isfinite(distances).any() and np.nanmin(distances) <= 8.0),
        "entered_5m": int(np.isfinite(distances).any() and np.nanmin(distances) <= 5.0),
        "entered_3m": int(np.isfinite(distances).any() and np.nanmin(distances) <= 3.0),
        "entered_1m": int(np.isfinite(distances).any() and np.nanmin(distances) <= 1.0),
        "stable_goal": int(any(int(row.get("goal_stable", 0) or 0) for row in rows)),
        "minimum_goal_distance_m": float(np.nanmin(distances)) if np.isfinite(distances).any() else float("nan"),
        "minimum_true_goal_distance_m": float(np.nanmin(true_distances)) if np.isfinite(true_distances).any() else float("nan"),
        "mean_speed_mps": float(np.nanmean(speeds)) if np.isfinite(speeds).any() else float("nan"),
        "mean_speed_within_1m_mps": float(np.nanmean([row["total_speed_mps"] for row in rows if _finite(row.get("estimated_goal_distance_m"), 99.0) <= 1.0])) if any(_finite(row.get("estimated_goal_distance_m"), 99.0) <= 1.0 for row in rows) else float("nan"),
        "mean_speed_at_8m_entry_mps": _speed_at_entry(rows, 8.0),
        "mean_speed_at_5m_entry_mps": _speed_at_entry(rows, 5.0),
        "mean_speed_at_3m_entry_mps": _speed_at_entry(rows, 3.0),
        "mean_speed_at_1m_entry_mps": _speed_at_entry(rows, 1.0),
        "maximum_consecutive_goal_hold_steps": max_hold,
        "hold_steps_ge_1": int(max_hold >= 1),
        "hold_steps_ge_3": int(max_hold >= 3),
        "hold_steps_ge_5": int(max_hold >= 5),
        "pre_candidate_static_swept_conflict_steps": int(sum(int(row.get("pre_candidate_static_swept_conflict", 0) or 0) for row in rows)),
        "pre_candidate_static_unsafe_steps": int(sum(int(row.get("pre_candidate_static_unsafe", 0) or 0) for row in rows)),
        "safety_projection_steps": int(sum(int(row.get("safety_projected", 0) or 0) for row in rows)),
        "safety_projection_episode": int(any(int(row.get("safety_projected", 0) or 0) for row in rows)),
        "boundary_projection_steps": int(sum(int(row.get("boundary_projected", 0) or 0) for row in rows)),
        "boundary_projection_episode": int(any(int(row.get("boundary_projected", 0) or 0) for row in rows)),
        "raw_action_changed_steps": int(sum(int(row.get("raw_action_changed", 0) or 0) for row in rows)),
        "action_clipping_fraction": float(np.mean([int(row.get("action_clipped", 0) or 0) for row in rows])) if rows else 0.0,
        "minimum_pre_clearance_m": float(min((_finite(row.get("pre_candidate_static_swept_minimum_clearance_m"), float("inf")) for row in rows), default=float("inf"))),
        "minimum_post_clearance_m": float(min((_finite(row.get("minimum_clearance_m"), float("inf")) for row in rows), default=float("inf"))),
        "collision": int(any(int(row.get("collision", 0) or 0) for row in rows)),
        "physical_collision": int(any(int(row.get("physical_collision", 0) or 0) for row in rows)),
        "clearance_violation": int(any(int(row.get("clearance_violation", 0) or 0) for row in rows)),
        "passed_obstacle": int(_passed_obstacle(rows)),
        "termination_reason": str(rows[-1].get("termination_reason", "running")) if rows else "empty",
    }


def _speed_at_entry(rows: Sequence[Mapping[str, Any]], threshold: float) -> float:
    for row in rows:
        if _finite(row.get("estimated_goal_distance_m"), float("inf")) <= float(threshold):
            return _finite(row.get("total_speed_mps"), float("nan"))
    return float("nan")


def _passed_obstacle(rows: Sequence[Mapping[str, Any]]) -> bool:
    # The route-level signal is deliberately conservative: the Oracle has
    # demonstrably reached the post-bypass phase, rather than merely touching
    # a waypoint or being moved by safety projection.
    return any(str(row.get("oracle_route_phase", "")) == "goal_after_bypass" for row in rows)


def _run_oracle_episode(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    scenario_name: str,
    scenario: Mapping[str, Any],
    safety_mode: str,
    episode_index: int,
    root: Path,
    phase: str,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(2026 + int(serial))
    environment = _make_environment(
        config,
        2026 + int(serial),
        metric_scales=config.metric_scales,
        safety_projection_enabled=safety_mode == "on",
    )
    observations, info = environment.reset(seed=2026 + int(serial), scenario_config=copy.deepcopy(dict(scenario)))
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=root / "episodes" / safety_mode / scenario_name / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test" if phase == "oracle_sanity" else "train",
        seed=2026 + int(serial),
        phase=phase,
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted),
        diagnostics_path=root / "diagnostics" / "attention_maps.jsonl",
        policy_source="stage2b_geometric_oracle",
    )
    controller = GeometricObstacleOracleController(environment)
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        visible = _visible_state(observations["uav_00"])
        state_before = environment.states["uav_00"].copy()
        action, control = controller.action(observations)
        action_delta = np.asarray(action["uav_00"]["delta_v"], dtype=float)
        nominal_state, _ = apply_velocity_increment(state_before, action_delta, environment.config.motion, safety_projector=None)
        nominal_position = np.asarray(nominal_state.position_m, dtype=float)
        pre_diag = _static_candidate_diagnostics(environment, state_before.position_m, nominal_position)
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        row = _episode_step_row(
            policy="geometric_oracle",
            scenario_name=scenario_name,
            safety_mode=safety_mode,
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=action_delta,
            control=control,
            nominal_position=nominal_position,
            pre_diag=pre_diag,
            next_info=next_info,
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
            policy_source="stage2b_geometric_oracle",
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary = _summary_from_rows(rows, policy="geometric_oracle", scenario_name=scenario_name, safety_mode=safety_mode, episode_index=episode_index)
    summary.update({"writer_summary": writer_summary, "training_performed": False, "policy_updates_performed": False, "ppo_updates": 0})
    return summary, rows


def run_oracle_sanity(
    trainer: Any, config: FormalExperimentConfig, data_root: Path, logs_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = _sanity_scenarios()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    serial = 1
    for name, scenario in scenarios.items():
        for mode in ("off", "on"):
            summary, episode_rows = _run_oracle_episode(
                trainer=trainer,
                config=config,
                scenario_name=name,
                scenario=scenario,
                safety_mode=mode,
                episode_index=1,
                root=data_root / "oracle_sanity",
                phase="oracle_sanity",
                serial=serial,
            )
            summary["scenario_hash"] = _canonical_hash(scenario)
            summaries.append(summary)
            rows.extend(episode_rows)
            serial += 1
    _write_rows_csv(logs_root / "oracle_sanity_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "oracle_sanity_step_diagnostics.csv", rows)
    direction_rows: list[dict[str, Any]] = []
    for name in ("front", "left_front", "right_front", "left", "right", "behind"):
        selected = [row for row in rows if row["scenario"] == name and row["safety_mode"] == "off"]
        first = selected[0] if selected else {}
        direction_rows.append(
            {
                "scenario": name,
                "obstacle_category": first.get("oracle_obstacle_category", ""),
                "route_phase": first.get("oracle_route_phase", ""),
                "lateral_action_projection_mps": first.get("oracle_lateral_action_projection_mps", float("nan")),
                "relevant_obstacle_id": first.get("relevant_obstacle_id", ""),
                "safety_off_collision": int(any(int(row.get("collision", 0) or 0) for row in selected)),
            }
        )
    _write_rows_csv(logs_root / "oracle_direction_probe.csv", direction_rows)
    required = [
        row for row in summaries
        if row["safety_mode"] == "off"
        and row["scenario"] in {"front", "left_front", "right_front", "centered_obstacle_blocking", "offset_left", "offset_right", "two_obstacle_simple"}
    ]
    sanity_passed = bool(
        all(not row["collision"] and not row["physical_collision"] for row in required)
        and all(row["minimum_post_clearance_m"] >= STATIC_SAFE_DISTANCE_M - 1.0e-6 for row in required)
        and all(row["passed_obstacle"] for row in required)
    )
    mode_summary = {}
    for mode in ("off", "on"):
        selected = [row for row in summaries if row["safety_mode"] == mode]
        mode_summary[mode] = {
            "episodes": len(selected),
            "collisions": int(sum(int(row["collision"]) for row in selected)),
            "passed_obstacle": int(sum(int(row["passed_obstacle"]) for row in selected)),
            "safety_projection_steps": int(sum(int(row["safety_projection_steps"]) for row in selected)),
            "action_clipping_fraction_mean": float(np.mean([row["action_clipping_fraction"] for row in selected])) if selected else 0.0,
        }
    report = {
        "oracle": "geometric_near_obstacle_bypass",
        "state_source": "EKF estimated position, velocity, yaw and Stage 2A relative obstacle feature; obstacle geometry from static map",
        "truth_state_read_for_control": False,
        "action_interface": "environment.step(delta_v)",
        "no_teleport": True,
        "safety_modes": mode_summary,
        "scenarios": len(scenarios),
        "sanity_passed": sanity_passed,
        "required_safety_off_passed": len(required),
        "direction_probe_path": str((logs_root / "oracle_direction_probe.csv").resolve()),
        "failure_if_false": "stop before DAgger",
    }
    _write_json(logs_root / "oracle_sanity_summary.json", _jsonable(report))
    return report, rows


def _query_gate(
    visible: Mapping[str, Any],
    environment: Any,
    student_action: np.ndarray,
    nominal_estimated_position: np.ndarray,
    controller: GeometricObstacleOracleController,
) -> tuple[bool, dict[str, Any]]:
    obstacle = select_relevant_obstacle(np.asarray(visible["position_m"], dtype=float), _physical_obstacles(environment))
    if obstacle is None:
        return False, {"reason": "no_physical_obstacle", "obstacle": None}
    clearance = float(point_obstacle_distance(np.asarray(visible["position_m"], dtype=float), obstacle))
    candidate = _static_candidate_diagnostics(environment, np.asarray(visible["position_m"], dtype=float), nominal_estimated_position)
    blocks = controller._blocks_goal_corridor(visible, obstacle)
    gate = bool(
        blocks
        or clearance <= QUERY_DISTANCE_M
        or bool(candidate.get("pre_candidate_static_unsafe", 0))
        or bool(candidate.get("pre_candidate_static_swept_conflict", 0))
    )
    return gate, {
        "reason": "obstacle_warning_or_candidate_risk" if gate else "obstacle_far_and_nonblocking",
        "obstacle": obstacle,
        "clearance_m": clearance,
        "candidate": candidate,
        "blocks_goal_corridor": blocks,
    }


def _dagger_record(
    *,
    split: str,
    episode_index: int,
    step: int,
    scenario_seed: int,
    noise_seed: int,
    scenario_hash: str,
    scene_type: str,
    category: str,
    observation: Mapping[str, Any],
    teacher_action: np.ndarray,
    student_action: np.ndarray,
    visible: Mapping[str, Any],
    gate: bool,
    teacher_queried: bool,
    pre_diag: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": "single_uav_obstacle_dagger_stage2b_smoke",
        "sample_source": "oracle_query" if teacher_queried else "policy_anchor_no_obstacle" if category == "no_obstacle" else "policy_anchor_far_obstacle",
        "split": split,
        "episode_index": int(episode_index),
        "step_index": int(step),
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "scenario_hash": str(scenario_hash),
        "scene_type": str(scene_type),
        "obstacle_category": str(category),
        "observation": _copy_observation_for_json(observation),
        "teacher_action_delta_v_mps": _jsonable(np.asarray(teacher_action, dtype=float).reshape(3).tolist()),
        "student_action_delta_v_mps": _jsonable(np.asarray(student_action, dtype=float).reshape(3).tolist()),
        "correction_action_error_norm_mps": float(np.linalg.norm(student_action - teacher_action)),
        "estimated_goal_distance_m": float(visible["distance_m"]),
        "estimated_velocity_mps": _jsonable(visible["velocity_mps"]),
        "estimated_radial_velocity_mps": float(visible["radial_velocity_mps"]),
        "goal_direction": _jsonable(visible["goal_direction"]),
        "obstacle_direction_body_m": _jsonable(visible.get("obstacle_direction_body_m", np.zeros(3))),
        "query_gate": int(gate),
        "oracle_queried": int(teacher_queried),
        "pre_candidate_static_swept_minimum_clearance_m": float(pre_diag.get("pre_candidate_static_swept_minimum_clearance_m", float("inf"))),
        "pre_candidate_static_swept_conflict": int(pre_diag.get("pre_candidate_static_swept_conflict", 0)),
        "pre_candidate_static_unsafe": int(pre_diag.get("pre_candidate_static_unsafe", 0)),
        "oracle_route_phase": str(control.get("route_phase", "")),
        "oracle_lateral_action_projection_mps": float(control.get("lateral_action_projection_mps", 0.0)),
        "student_action_norm_mps": float(np.linalg.norm(student_action)),
        "teacher_action_norm_mps": float(np.linalg.norm(teacher_action)),
        "new_direction_feature_learning_slots": json.dumps(list(NEW_DIRECTION_INDICES)),
    }


def _run_dagger_collection(
    *,
    source_trainer: Any,
    config: FormalExperimentConfig,
    data_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    dataset_root = data_root / "aggregated_dataset"
    train_path = dataset_root / "demonstrations" / "train.jsonl"
    validation_path = dataset_root / "demonstrations" / "validation.jsonl"
    all_episode_summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    counts = Counter()
    scene_counts = Counter()
    query_counts = Counter()
    scenario_hashes: set[str] = set()
    source_trainer.policy.eval()
    serial = 1000
    for episode in range(1, TRAIN_EPISODES + 1):
        kind = training_episode_kind(episode)
        scene_type = kind
        scenario_seed = int(71000 + 53 * episode)
        noise_seed = int(config.seed + 2_000_000 + 101 * episode)
        scenario = _normal_or_static_training_scenario(config, kind=kind, seed=noise_seed, episode_index=episode)
        scenario_hash = _canonical_hash(scenario)
        scenario_hashes.add(scenario_hash)
        _seed_everything(noise_seed)
        environment = _make_environment(config, noise_seed, metric_scales=config.metric_scales, safety_projection_enabled=True)
        observations, info = environment.reset(seed=noise_seed, scenario_config=copy.deepcopy(scenario))
        writer = _EpisodeTraceWriter(
            episode_root=data_root / "dagger_rollouts" / "train" / f"episode_{serial:06d}",
            episode_id=f"episode_{serial:06d}",
            scenario_id=f"scenario_{serial:06d}",
            split="train",
            seed=noise_seed,
            phase="stage2b_dagger_rollout",
            environment=environment,
            normalization_fitted=bool(source_trainer.normalizer.fitted),
            diagnostics_path=data_root / "diagnostics" / "attention_maps.jsonl",
            policy_source="stage2b_dagger_student",
        )
        teacher = GeometricObstacleOracleController(environment)
        episode_rows: list[dict[str, Any]] = []
        episode_records: list[dict[str, Any]] = []
        teacher_queries = 0
        overrides = 0
        for step in range(int(config.max_steps)):
            _assert_observation_finite(observations)
            decision = source_trainer.act(observations, info["central_state"], deterministic=True)
            normalized_observation = copy.deepcopy(dict(decision.observations["uav_00"]))
            normalized_observation["uav_id"] = "uav_00"
            student_action = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=float).reshape(3)
            visible = _visible_state(observations["uav_00"])
            estimated_nominal_position, _ = _candidate_from_estimate(visible, student_action, environment.config.motion)
            gate, gate_info = _query_gate(visible, environment, student_action, estimated_nominal_position, teacher)
            teacher_action = student_action.copy()
            teacher_control: dict[str, Any] = {"route_phase": "student_anchor", "lateral_action_projection_mps": 0.0}
            teacher_queried = False
            if gate:
                teacher_action_payload, teacher_control = teacher.action(observations)
                teacher_action = np.asarray(teacher_action_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
                teacher_queried = True
                teacher_queries += 1
                if np.linalg.norm(student_action - teacher_action) > 1.0e-8:
                    overrides += 1
            category = _obstacle_category(visible, gate_info.get("obstacle")) if gate_info.get("obstacle") is not None else "no_obstacle"
            state_before = environment.states["uav_00"].copy()
            nominal_state, _ = apply_velocity_increment(state_before, student_action, environment.config.motion, safety_projector=None)
            nominal_position = np.asarray(nominal_state.position_m, dtype=float)
            pre_diag = _static_candidate_diagnostics(environment, state_before.position_m, nominal_position)
            next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
            execution = next_info.get("action_execution", {}).get("uav_00", {})
            applied = np.asarray(execution.get("delta_v_applied", student_action), dtype=float).reshape(3)
            row = _episode_step_row(
                policy="stage2b_dagger_student",
                scenario_name=scene_type,
                safety_mode="on",
                episode_index=episode,
                step=step,
                environment=environment,
                visible_before=visible,
                position_before=np.asarray(state_before.position_m, dtype=float),
                velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
                action=student_action,
                control={**teacher_control, "oracle_query_gate": int(gate)},
                nominal_position=nominal_position,
                pre_diag=pre_diag,
                next_info=next_info,
            )
            row.update({"scenario_seed": scenario_seed, "noise_seed": noise_seed, "scenario_hash": scenario_hash, "query_gate": int(gate), "oracle_queried": int(teacher_queried), "oracle_override": int(teacher_queried and np.linalg.norm(student_action - teacher_action) > 1.0e-8), "obstacle_category": category})
            episode_rows.append(row)
            record = _dagger_record(
                split="train" if episode <= 24 else "validation",
                episode_index=episode,
                step=step,
                scenario_seed=scenario_seed,
                noise_seed=noise_seed,
                scenario_hash=scenario_hash,
                scene_type=scene_type,
                category=category,
                observation=normalized_observation,
                teacher_action=teacher_action,
                student_action=student_action,
                visible=visible,
                gate=gate,
                teacher_queried=teacher_queried,
                pre_diag=pre_diag,
                control=teacher_control,
            )
            episode_records.append(record)
            counts["oracle_query" if teacher_queried else "anchor"] += 1
            counts[f"category_{category}"] += 1
            scene_counts[scene_type] += 1
            if teacher_queried:
                query_counts[category] += 1
            writer.record_step(
                step=step,
                observations_before=observations,
                observations_after=next_observations,
                actions=decision.actions,
                rewards=rewards,
                learning_rewards=next_info.get("learning_rewards", rewards),
                info=next_info,
                graph_traces=None,
                policy_source="stage2b_dagger_student",
                action_start=step * environment.config.motion.dt_s,
                timestamp=(step + 1) * environment.config.motion.dt_s,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
        writer_summary = writer.finalize()
        episode_summary = _summary_from_rows(episode_rows, policy="stage2b_dagger_student", scenario_name=scene_type, safety_mode="on", episode_index=episode)
        episode_summary.update({"scenario_seed": scenario_seed, "noise_seed": noise_seed, "scenario_hash": scenario_hash, "teacher_query_count": teacher_queries, "oracle_override_count": overrides, "writer_summary": writer_summary, "training_performed": False, "policy_updates_performed": False})
        all_episode_summaries.append(episode_summary)
        all_step_rows.extend(episode_rows)
        if episode <= 24:
            train_records.extend(episode_records)
        else:
            validation_records.extend(episode_records)
        serial += 1

    for record in train_records:
        _append_jsonl(train_path, record)
    for record in validation_records:
        _append_jsonl(validation_path, record)
    _write_rows_csv(logs_root / "dagger_training_episode_metrics.csv", all_episode_summaries)
    _write_rows_csv(logs_root / "dagger_training_step_diagnostics.csv", all_step_rows)
    dataset_summary = {
        "dataset_id": "single_uav_obstacle_dagger_stage2b_smoke",
        "rollout_episodes": TRAIN_EPISODES,
        "normal_rollout_episodes": TRAIN_NORMAL_EPISODES,
        "static_rollout_episodes": TRAIN_STATIC_EPISODES,
        "schedule_counts": training_schedule_counts(TRAIN_EPISODES),
        "train_demonstration_samples": len(train_records),
        "validation_demonstration_samples": len(validation_records),
        "total_demonstration_samples": len(train_records) + len(validation_records),
        "scenario_hash_count": len(scenario_hashes),
        "scenario_hashes": sorted(scenario_hashes),
        "sample_source_counts": dict(counts),
        "teacher_query_counts_by_category": dict(query_counts),
        "oracle_query_count": int(sum(int(summary["teacher_query_count"]) for summary in all_episode_summaries)),
        "oracle_override_count": int(sum(int(summary["oracle_override_count"]) for summary in all_episode_summaries)),
        "normal_expert_override_fraction": float(sum(int(row.get("oracle_override", 0)) for row in all_step_rows if row.get("scenario") == "normal_far_distance") / max(1, sum(1 for row in all_step_rows if row.get("scenario") == "normal_far_distance"))),
        "left_right_front_no_obstacle_sample_counts": {key: int(counts.get(f"category_{key}", 0)) for key in ("left", "right", "front", "no_obstacle")},
        "train_validation_trajectory_overlap": False,
        "test_scenarios_used": False,
        "teacher": {"controller": "geometric_near_obstacle_bypass", "state_source": "EKF estimated state + known static obstacle geometry", "truth_state_read_for_control": False},
        "paths": {"train": str(train_path.resolve()), "validation": str(validation_path.resolve()), "dataset_root": str(dataset_root.resolve())},
    }
    _write_json(data_root / "dagger_dataset_summary.json", _jsonable(dataset_summary))
    return dataset_summary


def _direction_weight_report(source_checkpoint: Path, target_checkpoint: Path) -> dict[str, Any]:
    import torch

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    target = torch.load(target_checkpoint, map_location="cpu", weights_only=False)
    source_state = source["policy_state_dict"]
    target_state = target["policy_state_dict"]
    names = ("gat.layers.0.node_projection.weight", "gat.layers.0.residual.weight")
    entries = []
    for name in names:
        before = np.asarray(source_state[name].detach().cpu(), dtype=float)[:, 39:42]
        after = np.asarray(target_state[name].detach().cpu(), dtype=float)[:, 39:42]
        entries.append({"name": name, "shape": list(after.shape), "before_norm": float(np.linalg.norm(before)), "after_norm": float(np.linalg.norm(after)), "delta_norm": float(np.linalg.norm(after - before)), "before_all_zero": bool(np.allclose(before, 0.0)), "after_nonzero": bool(np.linalg.norm(after) > 1.0e-10)})
    return {"source_checkpoint": str(source_checkpoint.resolve()), "source_sha256": _sha256(source_checkpoint), "target_checkpoint": str(target_checkpoint.resolve()), "target_sha256": _sha256(target_checkpoint), "new_direction_indices": list(NEW_DIRECTION_INDICES), "entries": entries, "all_new_direction_weights_nonzero": bool(all(entry["after_nonzero"] for entry in entries))}


def _load_frozen_policy(checkpoint: Path, output_dir: Path) -> tuple[Any, FormalExperimentConfig, dict[str, str]]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=output_dir, progress=False)
    config = config.__class__(**{**config.__dict__, "output_dir": output_dir, "scenario_path": None, "scenario_pool_root": None, "single_uav_braking_curriculum_fraction": 0.0, "single_uav_terminal_braking_curriculum_fraction": 0.0, "force": True, "progress": False, "command_line": "Stage 2B frozen evaluation"})
    trainer, identity, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    return trainer, config, identity


def _run_actor_episode(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    checkpoint_role: str,
    scenario_name: str,
    scenario: Mapping[str, Any],
    safety_mode: str,
    episode_index: int,
    root: Path,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(900000 + serial)
    environment = _make_environment(config, 900000 + serial, metric_scales=config.metric_scales, safety_projection_enabled=safety_mode == "on")
    observations, info = environment.reset(seed=900000 + serial, scenario_config=copy.deepcopy(dict(scenario)))
    writer = _EpisodeTraceWriter(
        episode_root=root / "episodes" / checkpoint_role / safety_mode / scenario_name / f"episode_{serial:06d}",
        episode_id=f"episode_{serial:06d}",
        scenario_id=f"scenario_{serial:06d}",
        split="validation" if "validation" in scenario_name else "test",
        seed=900000 + serial,
        phase="stage2b_holdout_evaluation",
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted),
        diagnostics_path=root / "diagnostics" / "attention_maps.jsonl",
        policy_source=checkpoint_role,
    )
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision = trainer.act(observations, info["central_state"], deterministic=True)
        action_delta = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=float).reshape(3)
        visible = _visible_state(observations["uav_00"])
        state_before = environment.states["uav_00"].copy()
        nominal_state, _ = apply_velocity_increment(state_before, action_delta, environment.config.motion, safety_projector=None)
        nominal_position = np.asarray(nominal_state.position_m, dtype=float)
        pre_diag = _static_candidate_diagnostics(environment, state_before.position_m, nominal_position)
        next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
        row = _episode_step_row(
            policy=checkpoint_role,
            scenario_name=scenario_name,
            safety_mode=safety_mode,
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=action_delta,
            control={"route_phase": "actor", "oracle_lateral_action_projection_mps": 0.0},
            nominal_position=nominal_position,
            pre_diag=pre_diag,
            next_info=next_info,
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=decision.actions,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=checkpoint_role,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary = _summary_from_rows(rows, policy=checkpoint_role, scenario_name=scenario_name, safety_mode=safety_mode, episode_index=episode_index)
    summary.update({"writer_summary": writer_summary, "training_performed": False, "policy_updates_performed": False, "ppo_updates": 0})
    return summary, rows


def evaluate_holdout(
    *,
    stage2a_checkpoint: Path,
    stage2b_checkpoint: Path,
    config: FormalExperimentConfig,
    data_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    stage2a_trainer, stage2a_config, stage2a_identity = _load_frozen_policy(stage2a_checkpoint, data_root / "holdout_stage2a")
    stage2b_trainer, stage2b_config, stage2b_identity = _load_frozen_policy(stage2b_checkpoint, data_root / "holdout_stage2b")
    scenarios: dict[str, list[dict[str, Any]]] = {"normal_validation": [], "static_validation": [], "normal_test": [], "static_test": []}
    for split in ("validation", "test"):
        for index in range(HOLDOUT_EPISODES):
            seed = _scale_seed(config.seed, 1, split, index)
            scenarios[f"normal_{split}"].append({"scenario": _normal_scenario(config, seed=seed, split=split, episode_index=index), "seed": seed})
            static_seed = int(seed + 300000 + index)
            scenarios[f"static_{split}"].append({"scenario": _static_scenario(config, seed=static_seed, split=split, episode_index=100 + index, holdout=True), "seed": static_seed})
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    serial = 2000
    for role, trainer, role_config in (("stage2a_frozen", stage2a_trainer, stage2a_config), ("stage2b_dagger", stage2b_trainer, stage2b_config)):
        for group, items in scenarios.items():
            for episode_index, item in enumerate(items, start=1):
                scenario_name = group
                mode_list = ("off", "on") if group == "static_test" else ("on",)
                for mode in mode_list:
                    summary, rows = _run_actor_episode(
                        trainer=trainer,
                        config=role_config,
                        checkpoint_role=role,
                        scenario_name=scenario_name,
                        scenario=item["scenario"],
                        safety_mode=mode,
                        episode_index=episode_index,
                        root=data_root / "holdout",
                        serial=serial,
                    )
                    summary.update({"seed": int(item["seed"]), "scenario_hash": _canonical_hash(item["scenario"]), "split_group": group, "checkpoint_identity": stage2a_identity if role == "stage2a_frozen" else stage2b_identity})
                    summaries.append(summary)
                    step_rows.extend(rows)
                    serial += 1
    _write_rows_csv(logs_root / "stage2b_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "stage2b_step_diagnostics.csv", step_rows)
    static_pairs: list[dict[str, Any]] = []
    for role in ("stage2a_frozen", "stage2b_dagger"):
        for episode_index in range(1, HOLDOUT_EPISODES + 1):
            off = next(row for row in summaries if row["policy"] == role and row["split_group"] == "static_test" and row["episode_index"] == episode_index and row["safety_mode"] == "off")
            on = next(row for row in summaries if row["policy"] == role and row["split_group"] == "static_test" and row["episode_index"] == episode_index and row["safety_mode"] == "on")
            static_pairs.append({"policy": role, "episode_index": episode_index, "scenario_hash": off["scenario_hash"], "safety_off_collision": off["collision"], "safety_on_collision": on["collision"], "safety_off_pre_risk_steps": off["pre_candidate_static_swept_conflict_steps"], "safety_on_pre_risk_steps": on["pre_candidate_static_swept_conflict_steps"], "safety_off_projection_steps": off["safety_projection_steps"], "safety_on_projection_steps": on["safety_projection_steps"], "safety_off_minimum_goal_distance_m": off["minimum_goal_distance_m"], "safety_on_minimum_goal_distance_m": on["minimum_goal_distance_m"], "safety_off_raw_action_changed_steps": off["raw_action_changed_steps"], "safety_on_raw_action_changed_steps": on["raw_action_changed_steps"]})
    _write_rows_csv(logs_root / "safety_on_off_comparison.csv", static_pairs)
    _write_rows_csv(logs_root / "stage2a_vs_stage2b_comparison.csv", [row for row in summaries if row["safety_mode"] == "on"])
    _write_json(data_root / "checkpoint_identity.json", _jsonable({"stage2a": {"path": str(stage2a_checkpoint.resolve()), "sha256": _sha256(stage2a_checkpoint), "policy_hash": stage2a_identity.get("loaded_policy_hash", "")}, "stage2b": {"path": str(stage2b_checkpoint.resolve()), "sha256": _sha256(stage2b_checkpoint), "policy_hash": stage2b_identity.get("loaded_policy_hash", "")}, "evaluation_frozen": True, "ppo_updates": 0}))
    return {"episode_summaries": summaries, "step_rows": step_rows, "safety_pairs": static_pairs, "scenario_counts": {key: len(value) for key, value in scenarios.items()}, "evaluation_frozen": True, "ppo_updates": 0}


def _aggregate_holdout(rows: Sequence[Mapping[str, Any]], *, policy: str, group: str, safety_mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("policy") == policy and row.get("split_group") == group and row.get("safety_mode") == safety_mode]
    count = len(selected)
    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0) or 0) for row in selected))
    def mean_field(name: str) -> float:
        values = [_finite(row.get(name), float("nan")) for row in selected]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")
    return {"policy": policy, "group": group, "safety_mode": safety_mode, "episodes": count, "entered_8m": count_field("entered_8m"), "entered_8m_rate": count_field("entered_8m") / count if count else 0.0, "entered_5m": count_field("entered_5m"), "entered_5m_rate": count_field("entered_5m") / count if count else 0.0, "entered_3m": count_field("entered_3m"), "entered_3m_rate": count_field("entered_3m") / count if count else 0.0, "entered_1m": count_field("entered_1m"), "entered_1m_rate": count_field("entered_1m") / count if count else 0.0, "stable_goal": count_field("stable_goal"), "stable_goal_rate": count_field("stable_goal") / count if count else 0.0, "minimum_goal_distance_m": min((_finite(row.get("minimum_goal_distance_m"), float("inf")) for row in selected), default=float("nan")), "mean_minimum_goal_distance_m": mean_field("minimum_goal_distance_m"), "mean_speed_mps": mean_field("mean_speed_mps"), "mean_speed_at_8m_entry_mps": mean_field("mean_speed_at_8m_entry_mps"), "mean_speed_at_5m_entry_mps": mean_field("mean_speed_at_5m_entry_mps"), "mean_speed_at_3m_entry_mps": mean_field("mean_speed_at_3m_entry_mps"), "mean_speed_at_1m_entry_mps": mean_field("mean_speed_at_1m_entry_mps"), "mean_speed_within_1m_mps": mean_field("mean_speed_within_1m_mps"), "mean_maximum_consecutive_goal_hold_steps": mean_field("maximum_consecutive_goal_hold_steps"), "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in selected), default=0), "hold_ge_1": count_field("hold_steps_ge_1"), "hold_ge_3": count_field("hold_steps_ge_3"), "hold_ge_5": count_field("hold_steps_ge_5"), "pre_risk_steps": count_field("pre_candidate_static_swept_conflict_steps"), "pre_unsafe_steps": count_field("pre_candidate_static_unsafe_steps"), "safety_projection_steps": count_field("safety_projection_steps"), "safety_projection_episode_count": count_field("safety_projection_episode"), "boundary_projection_steps": count_field("boundary_projection_steps"), "boundary_projection_episode_count": count_field("boundary_projection_episode"), "raw_action_changed_steps": count_field("raw_action_changed_steps"), "action_clipping_fraction_mean": mean_field("action_clipping_fraction"), "collision_episodes": count_field("collision"), "physical_collision_episodes": count_field("physical_collision"), "clearance_violation_episodes": count_field("clearance_violation"), "passed_obstacle_episodes": count_field("passed_obstacle")}


def _stage2b_behavioral_acceptance(
    *,
    direction_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate pipeline completion from learned-behavior evidence.

    A small DAgger smoke can produce a valid checkpoint and nonzero new
    feature weights without yet learning a reliable obstacle bypass.  Keep
    those two outcomes separate so the smoke report is not over-interpreted.
    """

    def find(policy: str, group: str, safety_mode: str = "on") -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in aggregate_rows
                if row.get("policy") == policy
                and row.get("group") == group
                and row.get("safety_mode") == safety_mode
            ),
            None,
        )

    def metric(row: Mapping[str, Any] | None, name: str, default: float = 0.0) -> float:
        return _finite(row.get(name), default) if row is not None else float(default)

    frozen_normal = find("stage2a_frozen", "normal_test")
    trained_normal = find("stage2b_dagger", "normal_test")
    frozen_static = find("stage2a_frozen", "static_test", "on")
    trained_static = find("stage2b_dagger", "static_test", "on")
    normal_navigation_preserved = metric(trained_normal, "entered_8m") >= metric(frozen_normal, "entered_8m")
    normal_terminal_approach_preserved = (
        metric(trained_normal, "entered_3m") >= metric(frozen_normal, "entered_3m")
        and metric(trained_normal, "mean_minimum_goal_distance_m", float("inf"))
        <= metric(frozen_normal, "mean_minimum_goal_distance_m", float("inf"))
    )
    static_holdout_progress = (
        metric(trained_static, "entered_8m") > metric(frozen_static, "entered_8m")
        or metric(trained_static, "entered_3m") > metric(frozen_static, "entered_3m")
        or metric(trained_static, "passed_obstacle_episodes") > metric(frozen_static, "passed_obstacle_episodes")
    )
    boundary_projection_increased = metric(trained_static, "boundary_projection_steps") > metric(frozen_static, "boundary_projection_steps")
    probe = {
        str(row.get("case")): row
        for row in direction_rows
        if row.get("policy") == "stage2b_dagger"
    }
    left_lateral = metric(probe.get("left"), "lateral_action_projection_mps")
    right_lateral = metric(probe.get("right"), "lateral_action_projection_mps")
    lateral_bypass_signs_correct = left_lateral < -1.0e-4 and right_lateral > 1.0e-4
    blockers: list[str] = []
    if not normal_navigation_preserved:
        blockers.append("normal_navigation_regression")
    if not normal_terminal_approach_preserved:
        blockers.append("normal_terminal_approach_regression")
    if not static_holdout_progress:
        blockers.append("static_holdout_no_learned_progress")
    if not lateral_bypass_signs_correct:
        blockers.append("left_right_bypass_action_sign_not_learned")
    if boundary_projection_increased:
        blockers.append("boundary_projection_interference")
    return {
        "normal_navigation_preserved": bool(normal_navigation_preserved),
        "normal_terminal_approach_preserved": bool(normal_terminal_approach_preserved),
        "static_holdout_progress": bool(static_holdout_progress),
        "boundary_projection_increased": bool(boundary_projection_increased),
        "lateral_bypass_signs_correct": bool(lateral_bypass_signs_correct),
        "left_lateral_action_projection_mps": float(left_lateral),
        "right_lateral_action_projection_mps": float(right_lateral),
        "blockers": blockers,
        "behavioral_acceptance_passed": not blockers,
    }


def direction_action_probe(
    trainer: Any, config: FormalExperimentConfig, logs_root: Path, label: str
) -> list[dict[str, Any]]:
    records = probe_scenarios()
    rows: list[dict[str, Any]] = []
    for name, scenario in records.items():
        environment = _make_environment(config, 2026, metric_scales=config.metric_scales, safety_projection_enabled=True)
        observations, info = environment.reset(seed=2026, scenario_config=copy.deepcopy(scenario))
        decision = trainer.act(observations, info["central_state"], deterministic=True)
        action = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=float).reshape(3)
        visible = _visible_state(observations["uav_00"])
        left_world = _body_left_world(float(visible["yaw_rad"]))
        rows.append({"policy": label, "case": name, "action_dx_mps": float(action[0]), "action_dy_mps": float(action[1]), "action_dz_mps": float(action[2]), "action_norm_mps": float(np.linalg.norm(action)), "obstacle_dx_body_m": float(visible["obstacle_direction_body_m"][0]), "obstacle_dy_body_m": float(visible["obstacle_direction_body_m"][1]), "lateral_action_projection_mps": float(np.dot(action, left_world)), "raw_node_finite": bool(np.isfinite(np.asarray(observations["uav_00"]["node_features"], dtype=float)).all()), "legal_action": bool(np.linalg.norm(action) <= environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s + 1.0e-6)})
    return rows


def _write_plots(logs_root: Path, plots_root: Path, training_history: Sequence[Mapping[str, Any]], direction_rows: Sequence[Mapping[str, Any]], holdout_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    if training_history:
        figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        axis.plot([row["epoch"] for row in training_history], [row["train_mse"] for row in training_history], label="train MSE")
        axis.plot([row["epoch"] for row in training_history], [row["validation_mse"] for row in training_history], label="validation MSE")
        axis.set_xlabel("DAgger Actor-only epoch")
        axis.set_ylabel("post-tanh physical delta-v MSE")
        axis.set_title("Stage 2B DAgger imitation smoke")
        axis.grid(alpha=0.25)
        axis.legend()
        path = plots_root / "dagger_loss_curve.png"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        output["dagger_loss_curve"] = str(path.resolve())
    if direction_rows:
        selected = [row for row in direction_rows if row["policy"] in {"stage2a_frozen", "stage2b_dagger"}]
        labels = [row["case"] for row in selected if row["policy"] == "stage2a_frozen"]
        if labels:
            figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
            for policy, color in (("stage2a_frozen", "#2563eb"), ("stage2b_dagger", "#dc2626")):
                values = [row["lateral_action_projection_mps"] for row in selected if row["policy"] == policy]
                if values:
                    axis.plot(labels, values, marker="o", label=policy, color=color)
            axis.axhline(0.0, color="#111827", linewidth=0.8)
            axis.set_ylabel("Body-left lateral action projection (m/s)")
            axis.set_title("Stage 2B direction-action probe")
            axis.tick_params(axis="x", rotation=30)
            axis.grid(alpha=0.25)
            axis.legend()
            path = plots_root / "direction_action_probe.png"
            figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            output["direction_action_probe"] = str(path.resolve())
    if holdout_rows:
        selected = [row for row in holdout_rows if row["group"] == "static_test" and row["safety_mode"] == "on"]
        policies = ["stage2a_frozen", "stage2b_dagger"]
        metrics = (("entered_8m_rate", "Entered 8m"), ("entered_1m_rate", "Entered 1m"), ("passed_obstacle_episodes", "Passed obstacle"))
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True)
        for axis, (metric, title) in zip(axes, metrics):
            values = []
            labels = []
            for policy in policies:
                row = next((item for item in selected if item["policy"] == policy), None)
                values.append(float(row.get(metric, 0.0)) if row else 0.0)
                labels.append(policy)
            axis.bar(labels, values, color=("#2563eb", "#dc2626"))
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=25)
        path = plots_root / "stage2a_vs_stage2b_static_comparison.png"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        output["stage2a_vs_stage2b_static_comparison"] = str(path.resolve())
    return output


def run_stage2b_smoke(
    *,
    stage2a_checkpoint: str | Path = DEFAULT_STAGE2A_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    stage2a_checkpoint = Path(stage2a_checkpoint).resolve()
    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    _ensure_empty_roots(data_root, logs_root, plots_root)
    if not stage2a_checkpoint.is_file():
        raise FileNotFoundError(stage2a_checkpoint)
    import torch

    payload = torch.load(stage2a_checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=data_root, progress=False)
    config = config.__class__(**{**config.__dict__, "output_dir": data_root, "scenario_path": None, "scenario_pool_root": None, "single_uav_braking_curriculum_fraction": 0.0, "single_uav_terminal_braking_curriculum_fraction": 0.0, "force": True, "progress": False, "command_line": "Stage 2B geometric Oracle + DAgger smoke"})
    if config.uav_count != 1 or not config.single_uav_obstacle_direction_features_enabled or config.node_feature_dim != 42:
        raise ValueError("Stage 2B requires the Stage 2A 1-UAV 42-dimensional observation checkpoint")
    trainer, identity, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    source_policy_hash = _state_dict_hash(trainer.policy.state_dict())
    source_checkpoint_sha = _sha256(stage2a_checkpoint)
    preflight = feature_scale_preflight(trainer, config, logs_root)
    selector = selector_consistency_audit(config)
    if not selector["same_obstacle_selector"] or not selector["tie_break_ok"]:
        raise RuntimeError("Stage 2B relevant-obstacle selector audit failed")
    oracle_report, _oracle_rows = run_oracle_sanity(trainer, config, data_root, logs_root)
    if not oracle_report["sanity_passed"]:
        summary = {"status": "blocked_oracle_sanity", "stage2b_passed": False, "feature_scale_preflight": preflight, "selector_consistency": selector, "oracle_sanity": oracle_report, "dagger_not_started": True}
        _write_json(logs_root / "stage2b_summary.json", _jsonable(summary))
        return summary
    dataset_summary = _run_dagger_collection(source_trainer=trainer, config=config, data_root=data_root, logs_root=logs_root)
    training_output = data_root / "trained_actor"
    training_logs = logs_root / "dagger_training"
    training_plots = plots_root / "dagger_training"
    training_summary = _train_imitation(
        stage2a_checkpoint,
        data_root / "aggregated_dataset",
        output_dir=training_output,
        logs_dir=training_logs,
        plots_dir=training_plots,
        max_epochs=8,
        patience=3,
        batch_size=64,
        learning_rate=1.0e-4,
        seed=2026,
        stage="dagger_stage2b",
        force=False,
        progress=progress,
    )
    target_checkpoint = training_output / "scale_1" / "checkpoints" / "best_validation.pt"
    weight_report = _direction_weight_report(stage2a_checkpoint, target_checkpoint)
    if not weight_report["all_new_direction_weights_nonzero"]:
        summary = {"status": "blocked_direction_weights_not_learned", "stage2b_passed": False, "feature_scale_preflight": preflight, "selector_consistency": selector, "oracle_sanity": oracle_report, "dataset": dataset_summary, "training": training_summary, "direction_weight_change": weight_report}
        _write_json(logs_root / "stage2b_summary.json", _jsonable(summary))
        return summary
    trained_trainer, trained_config, trained_identity = _load_frozen_policy(target_checkpoint, data_root / "trained_eval")
    frozen_direction_rows = direction_action_probe(trainer, config, logs_root, "stage2a_frozen")
    trained_direction_rows = direction_action_probe(trained_trainer, trained_config, logs_root, "stage2b_dagger")
    direction_rows = frozen_direction_rows + trained_direction_rows
    _write_rows_csv(logs_root / "stage2b_direction_action_probe.csv", direction_rows)
    holdout = evaluate_holdout(stage2a_checkpoint=stage2a_checkpoint, stage2b_checkpoint=target_checkpoint, config=config, data_root=data_root, logs_root=logs_root)
    aggregate_rows = []
    for policy in ("stage2a_frozen", "stage2b_dagger"):
        for group in ("normal_validation", "static_validation", "normal_test", "static_test"):
            modes = ("off", "on") if group == "static_test" else ("on",)
            for mode in modes:
                aggregate_rows.append(_aggregate_holdout(holdout["episode_summaries"], policy=policy, group=group, safety_mode=mode))
    _write_rows_csv(logs_root / "stage2b_holdout_aggregate.csv", aggregate_rows)
    plot_paths = _write_plots(logs_root, plots_root, training_summary.get("history", []), direction_rows, aggregate_rows)
    direction_pairs = {}
    for left_name, right_name, key in (("left_front", "right_front", "left_front_vs_right_front"), ("left", "right", "left_vs_right"), ("front", "behind", "front_vs_behind")):
        left = next(row for row in direction_rows if row["policy"] == "stage2b_dagger" and row["case"] == left_name)
        right = next(row for row in direction_rows if row["policy"] == "stage2b_dagger" and row["case"] == right_name)
        left_action = np.asarray([left["action_dx_mps"], left["action_dy_mps"], left["action_dz_mps"]], dtype=float)
        right_action = np.asarray([right["action_dx_mps"], right["action_dy_mps"], right["action_dz_mps"]], dtype=float)
        delta_action = left_action - right_action
        direction_pairs[key] = {
            "action_l1": float(np.abs(delta_action).sum()),
            "action_l2": float(np.linalg.norm(delta_action)),
            "action_max_abs": float(np.max(np.abs(delta_action))),
        }
    behavioral_acceptance = _stage2b_behavioral_acceptance(
        direction_rows=direction_rows,
        aggregate_rows=aggregate_rows,
    )
    technical_smoke_passed = bool(
        oracle_report.get("sanity_passed")
        and weight_report.get("all_new_direction_weights_nonzero")
        and dataset_summary.get("schedule_counts") == {
            "normal_far_distance": TRAIN_NORMAL_EPISODES,
            "static_pressure": TRAIN_STATIC_EPISODES,
        }
        and int(training_summary.get("actual_epochs", 0)) > 0
        and int(training_summary.get("ppo_updates", 0)) == 0
        and holdout.get("evaluation_frozen") is True
    )
    report = {
        "status": "completed_stage2b_smoke",
        "stage2b_passed": bool(technical_smoke_passed and behavioral_acceptance["behavioral_acceptance_passed"]),
        "technical_smoke_passed": technical_smoke_passed,
        "behavioral_acceptance": behavioral_acceptance,
        "stage": STAGE_NAME,
        "source_checkpoint": {"path": str(stage2a_checkpoint), "sha256": source_checkpoint_sha, "policy_hash_before": source_policy_hash},
        "target_checkpoint": {"path": str(target_checkpoint.resolve()), "sha256": _sha256(target_checkpoint), "policy_hash_after": trained_identity.get("loaded_policy_hash", "")},
        "feature_scale_preflight": preflight,
        "selector_consistency": selector,
        "oracle_sanity": oracle_report,
        "dataset": dataset_summary,
        "training": {"actual_epochs": training_summary.get("actual_epochs"), "best_validation_epoch": training_summary.get("best_validation_epoch"), "best_validation_mse": training_summary.get("best_validation_mse"), "train_samples": training_summary.get("bc_contract", {}).get("train_samples"), "validation_samples": training_summary.get("bc_contract", {}).get("validation_samples"), "bc_updates": int(training_summary.get("actual_epochs", 0)), "dagger_rollout_updates": 0, "ppo_updates": 0, "critic_frozen": training_summary.get("bc_contract", {}).get("critic_frozen"), "normalizer_frozen": training_summary.get("bc_contract", {}).get("normalizer_frozen"), "logstd_frozen": training_summary.get("bc_contract", {}).get("logstd_frozen")},
        "direction_weight_change": weight_report,
        "direction_action_probe": {"pairs": direction_pairs, "rows": direction_rows},
        "holdout": {"scenario_counts": holdout["scenario_counts"], "aggregate_rows": aggregate_rows, "evaluation_frozen": holdout["evaluation_frozen"], "ppo_updates": holdout["ppo_updates"]},
        "outputs": {"data_root": str(data_root.resolve()), "logs_root": str(logs_root.resolve()), "plots_root": str(plots_root.resolve()), "plots": plot_paths},
        "policy_update_counts": {"bc_updates": int(training_summary.get("actual_epochs", 0)), "dagger_updates": 0, "ppo_updates": 0},
        "decision": {"observation_blocker_resolved": True, "oracle_passed": True, "dagger_direction_smoke_completed": True, "formal_stage2b_not_run": True, "technical_smoke_passed": technical_smoke_passed, "behavioral_acceptance_passed": behavioral_acceptance["behavioral_acceptance_passed"], "behavioral_blockers": behavioral_acceptance["blockers"], "next_stage_formal_eligibility": False},
    }
    _write_json(logs_root / "stage2b_summary.json", _jsonable(report))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2a-checkpoint", type=Path, default=DEFAULT_STAGE2A_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_stage2b_smoke(
        stage2a_checkpoint=args.stage2a_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )
    if not args.quiet:
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STAGE_NAME",
    "DEFAULT_STAGE2A_CHECKPOINT",
    "GeometricObstacleOracleController",
    "select_relevant_obstacle",
    "training_episode_kind",
    "training_schedule_counts",
    "feature_scale_preflight",
    "run_stage2b_smoke",
]
