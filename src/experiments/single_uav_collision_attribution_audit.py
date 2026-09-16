"""Paired safety-on/safety-off collision attribution audit for one UAV.

This module is an evaluation-only diagnostic.  It does not train a policy,
change the reward, or alter the existing experiment entry points.  It runs
the frozen baseline RL, DAgger round-1, and hierarchical waypoint policies
on the same scenario/seed pairs with the scene safety projection enabled and
disabled.  A second raw-action replay isolates the execution layer from the
closed-loop policy response.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..simulation.obstacle_model import (
    collision_event,
    is_physical_obstacle,
    moving_obstacle_swept_segment_intersects,
    point_obstacle_distance,
    segment_intersects_obstacle,
)
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _make_environment,
    _scenario_for,
    _scale_seed,
)
from .pilot import _assert_action_feasible, _assert_observation_finite, _seed_everything
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_hierarchical_waypoint import FrozenDaggerLowLevelPolicy


BASELINE_RL = "baseline_rl"
DAGGER_R1 = "dagger_round1"
HIERARCHICAL_WAYPOINT = "hierarchical_waypoint"
POLICY_NAMES: tuple[str, ...] = (
    BASELINE_RL,
    DAGGER_R1,
    HIERARCHICAL_WAYPOINT,
)
POLICY_LABELS = {
    BASELINE_RL: "coef=0.25 baseline RL",
    DAGGER_R1: "DAgger round-1 Actor",
    HIERARCHICAL_WAYPOINT: "hierarchical waypoint policy",
}
SAFETY_MODES: tuple[str, ...] = ("safety_on", "safety_off")
SCENE_NAMES: tuple[str, ...] = (
    "normal_far_distance",
    "static_obstacle_near",
    "dynamic_obstacle_crossing",
    "boundary_pressure",
)


@dataclass(frozen=True)
class CollisionAttributionConfig:
    """Configuration for the independent paired execution audit."""

    baseline_checkpoint: Path = Path(
        "data/synthetic/single_uav_braking_progress025_seed2026/"
        "scale_1/checkpoints/final.pt"
    )
    dagger_checkpoint: Path = Path(
        "data/synthetic/single_uav_dagger_actor_seed2026/"
        "scale_1/checkpoints/best_validation.pt"
    )
    hierarchical_checkpoint: Path = Path(
        "data/synthetic/single_uav_dagger_actor_seed2026/"
        "scale_1/checkpoints/best_validation.pt"
    )
    output_dir: Path = Path(
        "data/synthetic/single_uav_collision_attribution_audit_seed2026"
    )
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    episodes: int = 20
    max_steps: int | None = None
    seed: int = 2026
    include_raw_action_replay: bool = True
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if int(self.episodes) < 1:
            raise ValueError("episodes must be positive")
        if self.max_steps is not None and int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive when supplied")


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean(values: Iterable[Any]) -> float:
    numbers = [
        number
        for number in (_number(value) for value in values)
        if np.isfinite(number)
    ]
    return float(np.mean(numbers)) if numbers else float("nan")


def _minimum(values: Iterable[Any]) -> float:
    numbers = [
        number
        for number in (_number(value) for value in values)
        if np.isfinite(number)
    ]
    return float(np.min(numbers)) if numbers else float("nan")


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else np.zeros(3, dtype=float)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = json.dumps(
                        _jsonable(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                output[column] = value
            writer.writerow(output)


def _prepare_root(path: Path, *, force: bool) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()) and not force:
        raise FileExistsError(
            f"collision attribution output is not empty: {resolved}; "
            "use --force only for this exact diagnostic root"
        )
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    # Do not remove an existing directory here.  The audit is intentionally
    # recoverable; callers should choose a new root instead of deleting data.
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    import torch

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {resolved}")
    return dict(payload), _sha256(resolved)


def _evaluation_config(
    config: FormalExperimentConfig,
    *,
    output_dir: Path,
    max_steps: int | None,
) -> FormalExperimentConfig:
    """Freeze normal evaluation semantics for the audit."""

    return replace(
        config,
        output_dir=Path(output_dir),
        max_steps=int(max_steps if max_steps is not None else config.max_steps),
        scenario_path=None,
        scenario_pool_root=None,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        dynamic_obstacle_swept_projection_enabled=False,
        safety_execution_mode="legacy",
        force=True,
        progress=False,
        command_line="single-UAV collision attribution audit",
    )


def _normal_scenario(
    config: FormalExperimentConfig,
    *,
    seed: int,
    episode_index: int,
) -> dict[str, Any]:
    return _scenario_for(
        _evaluation_config(config, output_dir=config.output_dir, max_steps=config.max_steps),
        seed,
        split="test",
        episode_index=episode_index,
        phase="test",
    )


def _controlled_pressure_scenario(scene: str, *, episode_index: int, seed: int) -> dict[str, Any]:
    """Build deterministic one-UAV pressure scenes without changing dynamics."""

    if scene not in SCENE_NAMES[1:]:
        raise ValueError(f"unknown pressure scene: {scene}")
    lateral_offset = float((int(episode_index) % 5) - 2) * 2.0
    y = 100.0 + lateral_offset
    start = np.asarray((35.0, y, 20.0), dtype=float)
    goal = np.asarray((65.0, y, 20.0), dtype=float)
    velocity = np.zeros(3, dtype=float)
    specs: list[dict[str, Any]] = []
    if scene == "static_obstacle_near":
        specs.append(
            {
                "obstacle_id": "audit_static_wall",
                "obstacle_type": "building",
                "geometry_type": "box",
                "center_m": [50.0, y, 20.0],
                "size_m": [4.0, 24.0, 30.0],
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        )
    elif scene == "dynamic_obstacle_crossing":
        specs.append(
            {
                "obstacle_id": "audit_crossing_vehicle",
                "obstacle_type": "dynamic_vehicle",
                "geometry_type": "sphere",
                "center_m": [50.0, y - 7.0, 20.0],
                "radius_m": 3.0,
                "velocity_mps": [0.0, 2.5, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        )
    else:
        # A maximum-speed incoming state makes the boundary executor visibly
        # relevant while leaving collision geometry empty by design.
        start = np.asarray((1.2, y, 20.0), dtype=float)
        goal = np.asarray((0.0, y, 20.0), dtype=float)
        velocity = np.asarray((-12.0, 0.0, 0.0), dtype=float)

    return {
        "scenario_version": "single-uav-collision-attribution-v1",
        "seed": int(seed),
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": start.tolist()},
        "initial_velocities": {"uav_00": velocity.tolist()},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": goal.tolist()},
        "obstacle_specs": specs,
        "hierarchical_waypoint": goal.tolist(),
        "collision_attribution_scene": {
            "scene": scene,
            "pressure_scene": True,
            "geometry_seed": int(seed),
            "episode_index": int(episode_index),
            "direct_path_crosses_static_wall": scene == "static_obstacle_near",
            "dynamic_crossing_enabled": scene == "dynamic_obstacle_crossing",
            "boundary_pressure_enabled": scene == "boundary_pressure",
        },
        "generation_rule": "deterministic_single_uav_collision_attribution_pressure_scene",
    }


def _build_scenarios(
    config: FormalExperimentConfig,
    *,
    episodes: int,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    normal_config = _evaluation_config(
        config,
        output_dir=config.output_dir,
        max_steps=config.max_steps,
    )
    for index in range(int(episodes)):
        seed = _scale_seed(config.seed, 1, "test", index)
        normal = _scenario_for(
            normal_config,
            seed,
            split="test",
            episode_index=index,
            phase="test",
        )
        scenarios.append(
            {
                "scene": "normal_far_distance",
                "episode_index": index + 1,
                "seed": int(seed),
                "scenario": normal,
                "scenario_hash": _canonical_hash(normal),
            }
        )
        for pressure_index, scene in enumerate(SCENE_NAMES[1:], start=1):
            pressure_seed = int(seed + 10_000 * pressure_index)
            pressure = _controlled_pressure_scenario(
                scene,
                episode_index=index,
                seed=pressure_seed,
            )
            scenarios.append(
                {
                    "scene": scene,
                    "episode_index": index + 1,
                    "seed": int(seed),
                    "scenario": pressure,
                    "scenario_hash": _canonical_hash(pressure),
                }
            )
    return scenarios


class _FrozenPolicy:
    """Common frozen policy adapter for the three requested policy roles."""

    def __init__(
        self,
        role: str,
        checkpoint: Path,
        *,
        output_dir: Path,
        max_steps: int | None,
    ) -> None:
        if role not in POLICY_NAMES:
            raise ValueError(f"unknown policy role: {role}")
        self.role = role
        self.checkpoint = Path(checkpoint).resolve()
        self.payload, self.checkpoint_sha256 = _load_checkpoint(self.checkpoint)
        self.config = _config_from_checkpoint(
            self.payload,
            output_dir=output_dir,
            progress=False,
        )
        self.config = _evaluation_config(
            self.config,
            output_dir=output_dir,
            max_steps=max_steps,
        )
        self.trainer = None
        self.low_level = None
        if role == BASELINE_RL:
            self.trainer, policy_hashes, _ = _load_frozen_trainer(
                self.payload,
                self.config,
                metric_scales=self.config.metric_scales,
            )
            self.trainer.policy.eval()
            for parameter in self.trainer.policy.parameters():
                parameter.requires_grad_(False)
            self.policy_hash = _state_dict_hash(self.trainer.policy.state_dict())
            self.normalizer_hash = _canonical_hash(self.trainer.normalizer.state_dict())
            self.constructed_policy_hash = policy_hashes.get("constructed_policy_hash", "")
        else:
            self.low_level = FrozenDaggerLowLevelPolicy(
                self.checkpoint,
                output_dir=output_dir,
            )
            self.policy_hash = self.low_level.policy_hash_before
            self.normalizer_hash = self.low_level.normalizer_hash
            self.constructed_policy_hash = self.policy_hash

    def act(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        info: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.role == BASELINE_RL:
            assert self.trainer is not None
            decision = self.trainer.act(
                observations,
                info["central_state"],
                deterministic=True,
            )
            return dict(decision.actions), {
                "policy_action_source": "frozen_mappo_actor_mean",
                "truth_state_read_for_control": False,
            }
        assert self.low_level is not None
        delta, control = self.low_level.action(observations["uav_00"])
        return {
            "uav_00": {
                "delta_v": np.asarray(delta, dtype=float).reshape(3).tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }, {
            **dict(control),
            "policy_action_source": (
                "frozen_dagger_actor"
                if self.role == DAGGER_R1
                else "frozen_dagger_actor_with_temporary_waypoint"
            ),
            "truth_state_read_for_control": False,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": POLICY_LABELS[self.role],
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_hash": self.policy_hash,
            "constructed_policy_hash": self.constructed_policy_hash,
            "normalizer_hash": self.normalizer_hash,
            "max_steps": int(self.config.max_steps),
            "uav_count": int(self.config.uav_count),
            "frozen": True,
            "training_performed": False,
            "policy_updates_performed": False,
        }


def _action_vector(action: Mapping[str, Any]) -> np.ndarray:
    vector = np.asarray(action.get("delta_v", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("policy action contains non-finite delta_v")
    return vector


def _pre_candidate_diagnostics(
    environment: Any,
    *,
    position: np.ndarray,
    raw_delta_v: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    nominal_state, nominal_execution = apply_velocity_increment(
        environment.states["uav_00"],
        raw_delta_v,
        environment.config.motion,
        safety_projector=None,
    )
    nominal_position = np.asarray(nominal_state.position_m, dtype=float).copy()
    safety_distance = float(environment.config.motion.safety_distance_m)
    endpoint_collision, endpoint_clearance = collision_event(
        {"uav_00": nominal_position},
        environment.obstacles,
        safety_distance,
    )
    static_conflicts: list[str] = []
    moving_conflicts: list[str] = []
    physical_swept_contacts: list[str] = []
    for obstacle in environment.obstacles:
        if not obstacle.active or not is_physical_obstacle(obstacle):
            continue
        obstacle_id = str(obstacle.obstacle_id)
        if segment_intersects_obstacle(position, nominal_position, obstacle):
            static_conflicts.append(obstacle_id)
            physical_swept_contacts.append(obstacle_id)
        if moving_obstacle_swept_segment_intersects(
            position,
            nominal_position,
            obstacle,
            environment.config.motion.dt_s,
            np.asarray(environment.config.motion.bounds_min_m, dtype=float),
            np.asarray(environment.config.motion.bounds_max_m, dtype=float),
            safety_distance,
        ):
            moving_conflicts.append(obstacle_id)
        if moving_obstacle_swept_segment_intersects(
            position,
            nominal_position,
            obstacle,
            environment.config.motion.dt_s,
            np.asarray(environment.config.motion.bounds_min_m, dtype=float),
            np.asarray(environment.config.motion.bounds_max_m, dtype=float),
            0.0,
        ):
            physical_swept_contacts.append(obstacle_id)
    unsafe = bool(endpoint_collision or static_conflicts or moving_conflicts)
    return nominal_position, {
        "pre_candidate_endpoint_collision": int(endpoint_collision),
        "pre_candidate_endpoint_clearance_m": float(endpoint_clearance),
        "pre_candidate_static_swept_conflict": int(bool(static_conflicts)),
        "pre_candidate_dynamic_swept_conflict": int(bool(moving_conflicts)),
        "pre_candidate_unsafe": int(unsafe),
        "pre_candidate_physical_swept_contact": int(bool(physical_swept_contacts)),
        "pre_candidate_static_conflict_objects": static_conflicts,
        "pre_candidate_dynamic_conflict_objects": moving_conflicts,
        "pre_candidate_physical_contact_objects": sorted(set(physical_swept_contacts)),
        "pre_candidate_boundary_projected": int(
            bool(nominal_execution.get("boundary_projected", False))
        ),
        "pre_candidate_nominal_position_m": nominal_position.tolist(),
    }


def _step_row(
    *,
    scene: str,
    policy: str,
    evaluation_type: str,
    safety_mode: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
    step: int,
    environment: Any,
    position_before: np.ndarray,
    velocity_before: np.ndarray,
    raw_action: Mapping[str, Any],
    nominal_position: np.ndarray,
    pre_candidate: Mapping[str, Any],
    next_info: Mapping[str, Any],
    execution: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    uid = "uav_00"
    position_after = np.asarray(environment.states[uid].position_m, dtype=float)
    velocity_after = np.asarray(environment.states[uid].velocity_mps, dtype=float)
    goal = np.asarray(environment.goals[uid], dtype=float)
    direction = _unit(goal - position_before)
    distance_before = float(np.linalg.norm(goal - position_before))
    distance_after = float(np.linalg.norm(goal - position_after))
    raw_delta = _action_vector(raw_action[uid])
    effective_action = next_info.get("actions", {}).get(uid, raw_action[uid])
    effective_delta = _action_vector(effective_action)
    applied_delta = np.asarray(
        execution.get("delta_v_applied", effective_delta),
        dtype=float,
    ).reshape(3)
    post_conflicts = list(
        next_info.get("post_projection_obstacle_conflicts", [])
    ) + list(next_info.get("post_projection_uav_pair_conflicts", []))
    pre_conflicts = list(
        next_info.get("pre_projection_obstacle_conflicts", [])
    ) + list(next_info.get("pre_projection_uav_pair_conflicts", []))
    learning = next_info.get("learning_reward_components", {})
    if not isinstance(learning, Mapping):
        learning = {}
    return {
        "scene": scene,
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "evaluation_type": evaluation_type,
        "safety_mode": safety_mode,
        "episode_index": int(episode_index),
        "seed": int(seed),
        "scenario_hash": scenario_hash,
        "step": int(step),
        "timestamp_s": float((step + 1) * environment.config.motion.dt_s),
        "position_before_x": float(position_before[0]),
        "position_before_y": float(position_before[1]),
        "position_before_z": float(position_before[2]),
        "position_after_x": float(position_after[0]),
        "position_after_y": float(position_after[1]),
        "position_after_z": float(position_after[2]),
        "nominal_position_x": float(nominal_position[0]),
        "nominal_position_y": float(nominal_position[1]),
        "nominal_position_z": float(nominal_position[2]),
        "velocity_before_mps": float(np.linalg.norm(velocity_before)),
        "velocity_after_mps": float(np.linalg.norm(velocity_after)),
        "goal_distance_before_m": distance_before,
        "goal_distance_after_m": distance_after,
        "radial_velocity_before_mps": float(np.dot(velocity_before, direction)),
        "radial_velocity_after_mps": float(np.dot(velocity_after, _unit(goal - position_after))),
        "raw_delta_v_x": float(raw_delta[0]),
        "raw_delta_v_y": float(raw_delta[1]),
        "raw_delta_v_z": float(raw_delta[2]),
        "effective_delta_v_x": float(effective_delta[0]),
        "effective_delta_v_y": float(effective_delta[1]),
        "effective_delta_v_z": float(effective_delta[2]),
        "applied_delta_v_x": float(applied_delta[0]),
        "applied_delta_v_y": float(applied_delta[1]),
        "applied_delta_v_z": float(applied_delta[2]),
        "raw_action_norm_mps": float(np.linalg.norm(raw_delta)),
        "effective_action_norm_mps": float(np.linalg.norm(effective_delta)),
        "applied_action_norm_mps": float(np.linalg.norm(applied_delta)),
        "raw_applied_difference_norm_mps": float(np.linalg.norm(raw_delta - applied_delta)),
        "effective_applied_difference_norm_mps": float(np.linalg.norm(effective_delta - applied_delta)),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "safety_projected": int(bool(execution.get("safety_projected", False))),
        "boundary_projected": int(bool(execution.get("boundary_projected", False))),
        "braking_safety_projected": int(bool(execution.get("braking_safety_projected", False))),
        "pre_projection_collision": int(next_info.get("pre_projection_collision", 0)),
        "pre_projection_minimum_clearance_m": float(
            next_info.get("pre_projection_minimum_clearance", float("nan"))
        ),
        "pre_projection_conflict_count": len(pre_conflicts),
        "post_projection_conflict_count": len(post_conflicts),
        "minimum_clearance_m": float(next_info.get("minimum_clearance", float("nan"))),
        "collision": int(next_info.get("collision", 0)),
        "physical_collision": int(bool(next_info.get("physical_collision", False))),
        "clearance_violation": int(bool(next_info.get("clearance_violation", False))),
        "pre_candidate_endpoint_collision": int(pre_candidate["pre_candidate_endpoint_collision"]),
        "pre_candidate_endpoint_clearance_m": float(pre_candidate["pre_candidate_endpoint_clearance_m"]),
        "pre_candidate_static_swept_conflict": int(pre_candidate["pre_candidate_static_swept_conflict"]),
        "pre_candidate_dynamic_swept_conflict": int(pre_candidate["pre_candidate_dynamic_swept_conflict"]),
        "pre_candidate_unsafe": int(pre_candidate["pre_candidate_unsafe"]),
        "pre_candidate_physical_swept_contact": int(pre_candidate["pre_candidate_physical_swept_contact"]),
        "pre_candidate_boundary_projected": int(pre_candidate["pre_candidate_boundary_projected"]),
        "pre_candidate_static_conflict_objects": pre_candidate["pre_candidate_static_conflict_objects"],
        "pre_candidate_dynamic_conflict_objects": pre_candidate["pre_candidate_dynamic_conflict_objects"],
        "post_conflict_types": sorted(
            str(conflict.get("conflict_type", "unknown"))
            for conflict in post_conflicts
            if isinstance(conflict, Mapping)
        ),
        "safety_projection_reason": str(execution.get("safety_projection_reason", "")),
        "termination_reason": str(next_info.get("termination_reason", "running")),
        "learning_collision_penalty": float(
            learning.get("learning_collision_penalty", 0.0)
        ),
        "policy_action_source": str(control.get("policy_action_source", evaluation_type)),
    }


def _episode_summary(
    *,
    scene: str,
    policy: str,
    evaluation_type: str,
    safety_mode: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
    scenario: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(rows)
    last_reason = str(rows[-1].get("termination_reason", "running")) if rows else "empty"
    def count_rows(key: str) -> int:
        return int(sum(int(row.get(key, 0) or 0) for row in rows))

    def any_row(key: str) -> int:
        return int(any(bool(row.get(key, 0)) for row in rows))

    def minimum_key(key: str) -> float:
        return _minimum(row.get(key) for row in rows)

    positions = scenario.get("initial_positions", {})
    goals = scenario.get("goals", {})
    start_distance = float("nan")
    if isinstance(positions, Mapping) and isinstance(goals, Mapping):
        try:
            start_distance = float(
                np.linalg.norm(
                    np.asarray(goals["uav_00"], dtype=float)
                    - np.asarray(positions["uav_00"], dtype=float)
                )
            )
        except (KeyError, TypeError, ValueError):
            pass
    pre_unsafe_steps = count_rows("pre_candidate_unsafe")
    collision_steps = count_rows("collision")
    return {
        "scene": scene,
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "evaluation_type": evaluation_type,
        "safety_mode": safety_mode,
        "episode_index": int(episode_index),
        "seed": int(seed),
        "scenario_hash": scenario_hash,
        "steps": int(count),
        "start_distance_m": start_distance,
        "collision_occurred": int(bool(collision_steps)),
        "collision_steps": collision_steps,
        "physical_collision": any_row("physical_collision"),
        "clearance_violation": any_row("clearance_violation"),
        "pre_projection_collision_steps": count_rows("pre_projection_collision"),
        "pre_candidate_unsafe_steps": pre_unsafe_steps,
        "pre_candidate_static_swept_conflict_steps": count_rows("pre_candidate_static_swept_conflict"),
        "pre_candidate_dynamic_swept_conflict_steps": count_rows("pre_candidate_dynamic_swept_conflict"),
        "pre_candidate_physical_swept_contact_steps": count_rows("pre_candidate_physical_swept_contact"),
        "safety_projection_count": count_rows("safety_projected"),
        "boundary_projection_count": count_rows("boundary_projected"),
        "braking_safety_projection_count": count_rows("braking_safety_projected"),
        "action_clipped_steps": count_rows("action_clipped"),
        "action_clipping_fraction": _rate(count_rows("action_clipped"), count),
        "raw_applied_changed_steps": int(
            sum(
                _number(row.get("raw_applied_difference_norm_mps"), 0.0) > 1.0e-9
                for row in rows
            )
        ),
        "raw_applied_changed_fraction": _rate(
            sum(
                _number(row.get("raw_applied_difference_norm_mps"), 0.0) > 1.0e-9
                for row in rows
            ),
            count,
        ),
        "max_raw_applied_difference_norm_mps": max(
            (_number(row.get("raw_applied_difference_norm_mps"), 0.0) for row in rows),
            default=0.0,
        ),
        "minimum_pre_clearance_m": minimum_key("pre_projection_minimum_clearance_m"),
        "minimum_post_clearance_m": minimum_key("minimum_clearance_m"),
        "mean_pre_clearance_m": _mean(row.get("pre_projection_minimum_clearance_m") for row in rows),
        "mean_post_clearance_m": _mean(row.get("minimum_clearance_m") for row in rows),
        "mean_raw_action_norm_mps": _mean(row.get("raw_action_norm_mps") for row in rows),
        "mean_applied_action_norm_mps": _mean(row.get("applied_action_norm_mps") for row in rows),
        "mean_speed_mps": _mean(row.get("velocity_after_mps") for row in rows),
        "mean_radial_velocity_mps": _mean(row.get("radial_velocity_after_mps") for row in rows),
        "goal_reached": int(last_reason == "goal_reached"),
        "stable_goal": int(last_reason == "goal_reached"),
        "timeout": int(last_reason == "time_limit"),
        "termination_reason": last_reason,
    }


def _run_episode(
    *,
    policy: _FrozenPolicy,
    scenario_item: Mapping[str, Any],
    safety_mode: str,
    evaluation_type: str,
    action_trace: Sequence[Mapping[str, Any]] | None = None,
    progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if safety_mode not in SAFETY_MODES:
        raise ValueError(f"unknown safety mode: {safety_mode}")
    scene = str(scenario_item["scene"])
    episode_index = int(scenario_item["episode_index"])
    seed = int(scenario_item["seed"])
    scenario_hash = str(scenario_item["scenario_hash"])
    scenario = copy.deepcopy(dict(scenario_item["scenario"]))
    _seed_everything(seed)
    environment = _make_environment(
        policy.config,
        seed,
        metric_scales=policy.config.metric_scales,
        safety_projection_enabled=(safety_mode == "safety_on"),
    )
    observations, info = environment.reset(
        seed=seed,
        scenario_config=copy.deepcopy(scenario),
    )
    if policy.role == HIERARCHICAL_WAYPOINT:
        waypoint = np.asarray(
            scenario.get("hierarchical_waypoint", environment.goals["uav_00"]),
            dtype=float,
        ).reshape(3)
        environment.set_temporary_goal("uav_00", waypoint)
        observations = environment._observations()
    rows: list[dict[str, Any]] = []
    recorded_actions: list[dict[str, Any]] = []
    max_steps = int(policy.config.max_steps)
    for step in range(max_steps):
        _assert_observation_finite(observations)
        position_before = np.asarray(environment.states["uav_00"].position_m, dtype=float).copy()
        velocity_before = np.asarray(environment.states["uav_00"].velocity_mps, dtype=float).copy()
        if action_trace is None:
            action, control = policy.act(observations, info)
            recorded_actions.append(copy.deepcopy(action))
        else:
            if step >= len(action_trace):
                break
            action = copy.deepcopy(dict(action_trace[step]))
            control = {"policy_action_source": "raw_action_replay"}
        _assert_action_feasible(environment, action)
        raw_delta = _action_vector(action["uav_00"])
        nominal_position, pre_candidate = _pre_candidate_diagnostics(
            environment,
            position=position_before,
            raw_delta_v=raw_delta,
        )
        next_observations, _rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        row = _step_row(
            scene=scene,
            policy=policy.role,
            evaluation_type=evaluation_type,
            safety_mode=safety_mode,
            episode_index=episode_index,
            seed=seed,
            scenario_hash=scenario_hash,
            step=step,
            environment=environment,
            position_before=position_before,
            velocity_before=velocity_before,
            raw_action=action,
            nominal_position=nominal_position,
            pre_candidate=pre_candidate,
            next_info=next_info,
            execution=execution,
            control=control,
        )
        rows.append(row)
        observations, info = next_observations, next_info
        if progress and (step == 0 or (step + 1) % max(1, max_steps // 5) == 0):
            print(
                f"[{evaluation_type}/{safety_mode}] {policy.role} "
                f"{scene} episode {episode_index}/{len(action_trace) if action_trace is not None else '?'} "
                f"step {step + 1}/{max_steps}",
                flush=True,
            )
        if terminated or truncated:
            break
    summary = _episode_summary(
        scene=scene,
        policy=policy.role,
        evaluation_type=evaluation_type,
        safety_mode=safety_mode,
        episode_index=episode_index,
        seed=seed,
        scenario_hash=scenario_hash,
        scenario=scenario,
        rows=rows,
    )
    summary["action_trace_steps"] = len(recorded_actions)
    return summary, rows, recorded_actions


def _paired_contrasts(
    episode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], dict[str, Mapping[str, Any]]] = {}
    for row in episode_rows:
        key = (
            str(row.get("evaluation_type", "")),
            str(row.get("policy", "")),
            str(row.get("scene", "")),
            int(row.get("episode_index", 0)),
            str(row.get("scenario_hash", "")),
        )
        grouped.setdefault(key, {})[str(row.get("safety_mode", ""))] = row
    output: list[dict[str, Any]] = []
    for key, modes in sorted(grouped.items()):
        on = modes.get("safety_on")
        off = modes.get("safety_off")
        if on is None or off is None:
            continue
        row = {
            "evaluation_type": key[0],
            "policy": key[1],
            "policy_label": POLICY_LABELS.get(key[1], key[1]),
            "scene": key[2],
            "episode_index": key[3],
            "scenario_hash": key[4],
            "safety_on_collision": int(on.get("collision_occurred", 0)),
            "safety_off_collision": int(off.get("collision_occurred", 0)),
            "safety_on_pre_candidate_unsafe_steps": int(on.get("pre_candidate_unsafe_steps", 0)),
            "safety_off_pre_candidate_unsafe_steps": int(off.get("pre_candidate_unsafe_steps", 0)),
            "safety_on_projection_count": int(on.get("safety_projection_count", 0)),
            "safety_on_boundary_projection_count": int(on.get("boundary_projection_count", 0)),
            "safety_off_boundary_projection_count": int(off.get("boundary_projection_count", 0)),
            "safety_on_action_clipped_steps": int(on.get("action_clipped_steps", 0)),
            "safety_off_action_clipped_steps": int(off.get("action_clipped_steps", 0)),
        }
        row["closed_loop_safety_saved_collision"] = int(
            row["safety_off_collision"]
            and not row["safety_on_collision"]
            and row["safety_on_projection_count"] > 0
        )
        row["raw_action_replay_safety_saved_collision"] = int(
            key[0] == "raw_action_replay"
            and row["safety_off_collision"]
            and not row["safety_on_collision"]
            and row["safety_on_projection_count"] > 0
        )
        row["pre_candidate_risk_present"] = int(
            row["safety_on_pre_candidate_unsafe_steps"] > 0
            or row["safety_off_pre_candidate_unsafe_steps"] > 0
        )
        output.append(row)
    return output


def _aggregate(
    episode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in episode_rows:
        key = (
            str(row.get("evaluation_type", "")),
            str(row.get("policy", "")),
            str(row.get("scene", "")),
            str(row.get("safety_mode", "")),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (evaluation_type, policy, scene, safety_mode), rows in sorted(groups.items()):
        n = len(rows)
        def total(key: str) -> int:
            return int(sum(int(row.get(key, 0) or 0) for row in rows))
        def episodes_with(key: str) -> int:
            return int(sum(bool(row.get(key, 0)) for row in rows))
        collision_eps = episodes_with("collision_occurred")
        pre_risk_eps = int(sum(int(row.get("pre_candidate_unsafe_steps", 0)) > 0 for row in rows))
        output.append(
            {
                "evaluation_type": evaluation_type,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(policy, policy),
                "scene": scene,
                "safety_mode": safety_mode,
                "episodes": n,
                "collision_episodes": collision_eps,
                "collision_rate": _rate(collision_eps, n),
                "physical_collision_episodes": episodes_with("physical_collision"),
                "clearance_violation_episodes": episodes_with("clearance_violation"),
                "pre_projection_collision_steps": total("pre_projection_collision_steps"),
                "pre_candidate_risk_episodes": pre_risk_eps,
                "pre_candidate_unsafe_steps": total("pre_candidate_unsafe_steps"),
                "static_swept_conflict_steps": total("pre_candidate_static_swept_conflict_steps"),
                "dynamic_swept_conflict_steps": total("pre_candidate_dynamic_swept_conflict_steps"),
                "safety_projection_count": total("safety_projection_count"),
                "boundary_projection_count": total("boundary_projection_count"),
                "action_clipped_steps": total("action_clipped_steps"),
                "action_clipping_fraction": _mean(row.get("action_clipping_fraction") for row in rows),
                "raw_applied_changed_steps": total("raw_applied_changed_steps"),
                "raw_applied_changed_fraction": _mean(row.get("raw_applied_changed_fraction") for row in rows),
                "minimum_pre_clearance_m": _minimum(row.get("minimum_pre_clearance_m") for row in rows),
                "minimum_post_clearance_m": _minimum(row.get("minimum_post_clearance_m") for row in rows),
                "mean_pre_clearance_m": _mean(row.get("mean_pre_clearance_m") for row in rows),
                "mean_post_clearance_m": _mean(row.get("mean_post_clearance_m") for row in rows),
                "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in rows),
                "mean_raw_action_norm_mps": _mean(row.get("mean_raw_action_norm_mps") for row in rows),
                "mean_applied_action_norm_mps": _mean(row.get("mean_applied_action_norm_mps") for row in rows),
                "goal_reached_episodes": episodes_with("goal_reached"),
                "stable_goal_episodes": episodes_with("stable_goal"),
                "timeout_episodes": episodes_with("timeout"),
                "termination_reasons": ",".join(
                    f"{reason}:{sum(str(row.get('termination_reason')) == reason for row in rows)}"
                    for reason in sorted({str(row.get("termination_reason")) for row in rows})
                ),
            }
        )
    return output


def _classify(aggregates: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pressure = [row for row in aggregates if str(row.get("scene")) != "normal_far_distance"]
    pressure_pre_risk = int(sum(int(row.get("pre_candidate_risk_episodes", 0)) for row in pressure))
    replay_rescued = int(
        sum(
            int(row.get("raw_action_replay_safety_saved_collision", 0))
            for row in contrasts
        )
    )
    closed_rescued = int(
        sum(int(row.get("closed_loop_safety_saved_collision", 0)) for row in contrasts)
    )
    all_pressure_collision_free = bool(pressure) and all(
        int(row.get("collision_episodes", 0)) == 0 for row in pressure
    )
    if replay_rescued > 0 or closed_rescued > 0:
        primary = "collision=0 主要由安全执行机制保证"
    elif pressure_pre_risk == 0 and all_pressure_collision_free:
        primary = "当前场景没有触发足够的碰撞风险，暂时无法判断"
    else:
        primary = "已有证据支持策略主动避障"
    return {
        "primary_class": primary,
        "pressure_aggregate_groups": len(pressure),
        "pressure_pre_candidate_risk_episode_count": pressure_pre_risk,
        "raw_action_replay_safety_saved_collision_count": replay_rescued,
        "closed_loop_safety_saved_collision_count": closed_rescued,
        "interpretation_scope": "episode/scenario; step rows are execution diagnostics only",
    }


def _plot_diagnostics(
    *,
    aggregates: Sequence[Mapping[str, Any]],
    plots_root: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pressure_scenes = list(SCENE_NAMES[1:])
    policy_colors = {
        BASELINE_RL: "#2563eb",
        DAGGER_R1: "#16a34a",
        HIERARCHICAL_WAYPOINT: "#dc2626",
    }
    plots_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    width = 0.12
    positions = np.arange(len(pressure_scenes), dtype=float)
    for policy_index, policy in enumerate(POLICY_NAMES):
        for mode_index, mode in enumerate(SAFETY_MODES):
            values = []
            for scene in pressure_scenes:
                selected = next(
                    (
                        row
                        for row in aggregates
                        if row.get("policy") == policy
                        and row.get("scene") == scene
                        and row.get("evaluation_type") == "closed_loop"
                        and row.get("safety_mode") == mode
                    ),
                    None,
                )
                values.append(float(selected.get("collision_rate", 0.0)) if selected else 0.0)
            offset = (policy_index * 2 + mode_index - 2.5) * width
            axes[0].bar(
                positions + offset,
                values,
                width=width,
                color=policy_colors[policy],
                alpha=0.95 if mode == "safety_on" else 0.45,
                label=f"{POLICY_LABELS[policy]} {mode}",
            )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(pressure_scenes, rotation=20, ha="right")
    axes[0].set_ylabel("collision episode rate")
    axes[0].set_title("Closed-loop collision rate")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)

    for policy_index, policy in enumerate(POLICY_NAMES):
        values = []
        for scene in pressure_scenes:
            selected = [
                row
                for row in aggregates
                if row.get("policy") == policy
                and row.get("scene") == scene
                and row.get("evaluation_type") == "raw_action_replay"
                and row.get("safety_mode") == "safety_on"
            ]
            values.append(
                float(sum(row.get("safety_projection_count", 0) for row in selected))
                if selected
                else 0.0
            )
        axes[1].bar(
            positions + (policy_index - 1) * 0.22,
            values,
            width=0.20,
            color=policy_colors[policy],
            label=POLICY_LABELS[policy],
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(pressure_scenes, rotation=20, ha="right")
    axes[1].set_ylabel("safety projection steps")
    axes[1].set_title("Raw-action replay safety intervention")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    path = plots_root / "collision_attribution_overview.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for policy in POLICY_NAMES:
        for mode, linestyle in (("safety_on", "-"), ("safety_off", "--")):
            selected = [
                row
                for row in aggregates
                if row.get("policy") == policy
                and row.get("evaluation_type") == "raw_action_replay"
                and row.get("safety_mode") == mode
            ]
            means = []
            for scene in pressure_scenes:
                scene_rows = [row for row in selected if row.get("scene") == scene]
                means.append(
                    _mean(row.get("mean_post_clearance_m") for row in scene_rows)
                )
            axis.plot(
                pressure_scenes,
                means,
                marker="o",
                linestyle=linestyle,
                color=policy_colors[policy],
                label=f"{POLICY_LABELS[policy]} {mode}",
            )
    axis.axhline(2.0, color="#111827", linestyle=":", linewidth=1.2, label="safety distance 2 m")
    axis.set_ylabel("mean post-execution clearance (m)")
    axis.set_title("Raw-action replay post-execution clearance")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    path2 = plots_root / "collision_attribution_clearance.png"
    figure.savefig(path2, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "overview": str(path.resolve()),
        "clearance": str(path2.resolve()),
    }


def run_collision_attribution_audit(
    *,
    baseline_checkpoint: Path,
    dagger_checkpoint: Path,
    hierarchical_checkpoint: Path,
    output_dir: Path,
    logs_dir: Path | None = None,
    plots_dir: Path | None = None,
    episodes: int = 20,
    max_steps: int | None = None,
    seed: int = 2026,
    include_raw_action_replay: bool = True,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the paired safety attribution audit and write independent outputs."""

    config = CollisionAttributionConfig(
        baseline_checkpoint=Path(baseline_checkpoint),
        dagger_checkpoint=Path(dagger_checkpoint),
        hierarchical_checkpoint=Path(hierarchical_checkpoint),
        output_dir=Path(output_dir),
        logs_dir=Path(logs_dir) if logs_dir is not None else None,
        plots_dir=Path(plots_dir) if plots_dir is not None else None,
        episodes=int(episodes),
        max_steps=max_steps,
        seed=int(seed),
        include_raw_action_replay=bool(include_raw_action_replay),
        force=bool(force),
        progress=bool(progress),
    )
    data_root = _prepare_root(config.output_dir, force=config.force)
    logs_root = _prepare_root(
        config.logs_dir
        or Path("logs/preliminary") / data_root.name,
        force=config.force,
    )
    plots_root = _prepare_root(
        config.plots_dir
        or Path("plots/preliminary") / data_root.name,
        force=config.force,
    )

    baseline_payload, _baseline_sha = _load_checkpoint(config.baseline_checkpoint)
    baseline_config = _config_from_checkpoint(
        baseline_payload,
        output_dir=data_root / "baseline_rl",
        progress=False,
    )
    if baseline_config.uav_count != 1:
        raise ValueError("collision attribution audit requires a one-UAV baseline checkpoint")
    baseline_config = replace(baseline_config, seed=int(config.seed))
    scenarios = _build_scenarios(baseline_config, episodes=config.episodes)
    scenario_manifest_rows: list[dict[str, Any]] = []
    for item in scenarios:
        scene_dir = data_root / "scenarios" / str(item["scene"])
        scenario_path = scene_dir / f"episode_{int(item['episode_index']):06d}.json"
        _write_json(scenario_path, _jsonable(item["scenario"]))
        scenario_manifest_rows.append(
            {
                "scene": item["scene"],
                "episode_index": item["episode_index"],
                "seed": item["seed"],
                "scenario_hash": item["scenario_hash"],
                "scenario_path": str(scenario_path.resolve()),
            }
        )
    _write_rows_csv(data_root / "scenario_manifest.csv", scenario_manifest_rows)

    policies = {
        BASELINE_RL: _FrozenPolicy(
            BASELINE_RL,
            config.baseline_checkpoint,
            output_dir=data_root / BASELINE_RL,
            max_steps=config.max_steps,
        ),
        DAGGER_R1: _FrozenPolicy(
            DAGGER_R1,
            config.dagger_checkpoint,
            output_dir=data_root / DAGGER_R1,
            max_steps=config.max_steps,
        ),
        HIERARCHICAL_WAYPOINT: _FrozenPolicy(
            HIERARCHICAL_WAYPOINT,
            config.hierarchical_checkpoint,
            output_dir=data_root / HIERARCHICAL_WAYPOINT,
            max_steps=config.max_steps,
        ),
    }
    all_episode_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    replay_episode_rows: list[dict[str, Any]] = []
    replay_step_rows: list[dict[str, Any]] = []
    source_action_traces: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}

    for item in scenarios:
        for policy_name in POLICY_NAMES:
            policy = policies[policy_name]
            off_summary, off_steps, off_actions = _run_episode(
                policy=policy,
                scenario_item=item,
                safety_mode="safety_off",
                evaluation_type="closed_loop",
                progress=False,
            )
            all_episode_rows.append(off_summary)
            all_step_rows.extend(off_steps)
            source_action_traces[(policy_name, str(item["scene"]), int(item["episode_index"]), str(item["scenario_hash"]))] = off_actions
            on_summary, on_steps, _on_actions = _run_episode(
                policy=policy,
                scenario_item=item,
                safety_mode="safety_on",
                evaluation_type="closed_loop",
                progress=False,
            )
            all_episode_rows.append(on_summary)
            all_step_rows.extend(on_steps)
            if config.include_raw_action_replay:
                for mode in SAFETY_MODES:
                    replay_summary, replay_steps, _unused = _run_episode(
                        policy=policy,
                        scenario_item=item,
                        safety_mode=mode,
                        evaluation_type="raw_action_replay",
                        action_trace=off_actions,
                        progress=False,
                    )
                    replay_episode_rows.append(replay_summary)
                    replay_step_rows.extend(replay_steps)
        if config.progress:
            print(
                f"[collision-attribution] completed scene={item['scene']} "
                f"episode={item['episode_index']}/{config.episodes}",
                flush=True,
            )

    closed_aggregates = _aggregate(all_episode_rows)
    replay_aggregates = _aggregate(replay_episode_rows)
    aggregates = closed_aggregates + replay_aggregates
    closed_contrasts = _paired_contrasts(all_episode_rows)
    replay_contrasts = _paired_contrasts(replay_episode_rows)
    contrasts = closed_contrasts + replay_contrasts
    classification = _classify(aggregates, contrasts)
    plot_paths = _plot_diagnostics(aggregates=aggregates, plots_root=plots_root)

    _write_rows_csv(logs_root / "episode_metrics_closed_loop.csv", all_episode_rows)
    _write_rows_csv(logs_root / "step_metrics_closed_loop.csv", all_step_rows)
    if config.include_raw_action_replay:
        _write_rows_csv(logs_root / "episode_metrics_raw_action_replay.csv", replay_episode_rows)
        _write_rows_csv(logs_root / "step_metrics_raw_action_replay.csv", replay_step_rows)
    _write_rows_csv(logs_root / "collision_attribution_aggregates.csv", aggregates)
    _write_rows_csv(logs_root / "paired_on_off_contrasts.csv", contrasts)

    checkpoint_manifest = {
        "baseline_rl": policies[BASELINE_RL].manifest(),
        "dagger_round1": policies[DAGGER_R1].manifest(),
        "hierarchical_waypoint": policies[HIERARCHICAL_WAYPOINT].manifest(),
    }
    _write_json(data_root / "checkpoint_manifest.json", _jsonable(checkpoint_manifest))
    _write_json(data_root / "scenario_manifest.json", _jsonable(scenario_manifest_rows))
    summary = {
        "dataset_id": "single_uav_collision_attribution_audit",
        "protocol": "safety_on_off_paired_execution_v1",
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "episodes_per_scene": int(config.episodes),
        "scenes": list(SCENE_NAMES),
        "policies": [POLICY_LABELS[name] for name in POLICY_NAMES],
        "safety_modes": list(SAFETY_MODES),
        "safety_execution_mode": "legacy",
        "dynamic_obstacle_swept_projection_enabled": False,
        "raw_action_replay_enabled": bool(config.include_raw_action_replay),
        "episode_unit": "scenario/episode",
        "step_rows_are_diagnostics_only": True,
        "checkpoint_manifest": str((data_root / "checkpoint_manifest.json").resolve()),
        "scenario_manifest": str((data_root / "scenario_manifest.json").resolve()),
        "logs_root": str(logs_root),
        "plots_root": str(plots_root),
        "aggregates": _jsonable(aggregates),
        "classification": classification,
        "plot_paths": plot_paths,
        "audit_ok": bool(
            len(all_episode_rows) == len(SCENE_NAMES) * config.episodes * len(POLICY_NAMES) * 2
            and (
                not config.include_raw_action_replay
                or len(replay_episode_rows)
                == len(SCENE_NAMES) * config.episodes * len(POLICY_NAMES) * 2
            )
        ),
    }
    _write_json(data_root / "summary.json", _jsonable(summary))
    _write_json(logs_root / "summary.json", _jsonable(summary))
    _write_json(data_root / "manifest.json", _jsonable({
        "dataset_id": summary["dataset_id"],
        "protocol": summary["protocol"],
        "summary": str((data_root / "summary.json").resolve()),
        "checkpoint_manifest": summary["checkpoint_manifest"],
        "scenario_manifest": summary["scenario_manifest"],
        "evaluation_only": True,
        "training_performed": False,
    }))
    return summary


__all__ = [
    "CollisionAttributionConfig",
    "run_collision_attribution_audit",
    "BASELINE_RL",
    "DAGGER_R1",
    "HIERARCHICAL_WAYPOINT",
]
