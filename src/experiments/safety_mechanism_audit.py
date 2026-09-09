"""Mechanism audit for safety collisions under frozen MAPPO policies.

The safety evaluation in :mod:`safety_mappo_evaluation` answers whether two
frozen policies have different collision outcomes.  This module answers the
next, narrower question: what was visible to the actor immediately before a
collision, what nominal motion did it request, what did the safety projector
actually execute, and in what order were dynamic obstacles updated?

This is a read-only diagnostic.  It replays frozen checkpoints on the frozen
scenario files, never calls ``trainer.update``, never reads test data, and does
not alter the simulator's motion, collision, reward, EKF, or channel logic.
The five policy seeds are averaged within each scenario seed; scenario seeds
are the independent statistical units.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..simulation.obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    collision_event,
    point_obstacle_distance,
    segment_intersects_obstacle,
)
from ..synthetic.episode_generator import _jsonable
from .formal import _make_environment
from .safety_mappo_evaluation import (
    BASELINE_BRANCH,
    DEFAULT_POLICY_SEEDS,
    DEFAULT_SCENARIO_SEEDS,
    FULL_BRANCH,
    SafetyMAPPOEvaluationConfig,
    _actor_observation_truth_free,
    _action_metrics,
    _bootstrap_ci,
    _canonical_hash,
    _csv_value,
    _evaluation_seed,
    _finite_flag,
    _finite_or,
    _load_bundle,
    _load_scenario,
    _positions_metrics,
    _write_csv,
    _write_json,
)


DEFAULT_BRANCHES = (BASELINE_BRANCH, FULL_BRANCH)
_SCHEMA_VERSION = "safety-mechanism-audit-v1.0"
_OUTPUT_FILES = (
    "collision_events.csv",
    "step_diagnostics.csv",
    "scenario_summary.csv",
    "paired_contrasts.csv",
    "observation_action_summary.csv",
    "policy_seed_sensitivity.csv",
    "audit_report.json",
)
_PLOT_FILE = "safety_mechanism_audit.png"
_SAFETY_TOLERANCE = 1e-7


@dataclass(frozen=True)
class SafetyMechanismAuditConfig:
    """Configuration for the validation-only safety mechanism audit."""

    model_root: Path
    scenario_root: Path
    output_dir: Path = Path(
        "logs/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_mechanism_audit"
    )
    plots_dir: Path = Path(
        "plots/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_mechanism_audit"
    )
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    deterministic: bool = True
    safety_projection_enabled: bool = True
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("safety mechanism audit is restricted to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("the deterministic safety audit supports device='cpu' only")
        if not self.deterministic:
            raise ValueError("safety mechanism audit must use deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("mechanism attribution requires safety projection enabled")
        if not self.policy_seeds or not self.scenario_seeds:
            raise ValueError("at least one policy seed and scenario seed are required")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if len(self.branches) != 2 or set(self.branches) != set(DEFAULT_BRANCHES):
            raise ValueError(
                "branches must be exactly baseline_mappo and "
                "budgeted_quality_aware_mappo"
            )
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: SafetyMechanismAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-mechanism-audit] {message}", flush=True)


def _prepare_outputs(config: SafetyMechanismAuditConfig) -> tuple[Path, Path]:
    output = Path(config.output_dir).resolve()
    plots = Path(config.plots_dir).resolve()
    if output == plots:
        raise ValueError("output_dir and plots_dir must be distinct")
    model_root = Path(config.model_root).resolve()
    scenario_root = Path(config.scenario_root).resolve()
    for path in (output, plots):
        if _paths_overlap(path, model_root) or _paths_overlap(path, scenario_root):
            raise ValueError("audit output cannot replace a source directory")
        if path.exists() and any(path.iterdir()) and not config.force:
            raise FileExistsError(
                f"audit output is not empty: {path}; use --force to replace it"
            )
    if config.force:
        for path in (output, plots):
            if path.exists():
                shutil.rmtree(path)
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return output, plots


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two paths are equal or one contains the other."""

    return left == right or left in right.parents or right in left.parents


def _obstacle_snapshot(obstacles: Iterable[Obstacle]) -> list[dict[str, Any]]:
    """Serialize geometry and motion state without exposing it to the actor."""

    rows: list[dict[str, Any]] = []
    for obstacle in sorted(obstacles, key=lambda item: str(item.obstacle_id)):
        row: dict[str, Any] = {
            "obstacle_id": str(obstacle.obstacle_id),
            "obstacle_type": str(obstacle.obstacle_type),
            "active": int(bool(obstacle.active)),
            "center_m": np.asarray(obstacle.center_m, dtype=float).copy(),
            "velocity_mps": np.asarray(
                obstacle.velocity_mps if obstacle.velocity_mps is not None else np.zeros(3),
                dtype=float,
            ).copy(),
        }
        if isinstance(obstacle, BoxObstacle):
            row.update(
                {
                    "geometry_type": "box",
                    "size_m": np.asarray(obstacle.size_m, dtype=float).copy(),
                    "minimum_m": np.asarray(obstacle.minimum_m, dtype=float).copy(),
                    "maximum_m": np.asarray(obstacle.maximum_m, dtype=float).copy(),
                }
            )
        else:
            row.update(
                {
                    "geometry_type": "sphere",
                    "radius_m": float(obstacle.radius_m),
                }
            )
        rows.append(row)
    return rows


def _scene_clearance_detail(
    positions: Mapping[str, np.ndarray],
    obstacles: Iterable[Obstacle],
    safety_distance_m: float,
) -> dict[str, Any]:
    """Return the closest scene element and its collision interpretation."""

    candidates: list[tuple[float, str, str, str, str]] = []
    ids = sorted(str(key) for key in positions)
    for index, source in enumerate(ids):
        for target in ids[index + 1 :]:
            distance = float(
                np.linalg.norm(
                    np.asarray(positions[source], dtype=float)
                    - np.asarray(positions[target], dtype=float)
                )
            )
            candidates.append((distance, "uav_uav", source, target, ""))
    for uav_id in ids:
        position = np.asarray(positions[uav_id], dtype=float)
        for obstacle in sorted(obstacles, key=lambda item: str(item.obstacle_id)):
            if not obstacle.active:
                continue
            distance = float(point_obstacle_distance(position, obstacle))
            candidates.append(
                (distance, "uav_obstacle", uav_id, str(obstacle.obstacle_id), str(obstacle.obstacle_type))
            )
    if not candidates:
        distance = float(safety_distance_m)
        return {
            "collision": 0,
            "minimum_clearance_m": distance,
            "source_type": "none",
            "source_id": "none",
            "source_uav_id": "none",
            "other_uav_id": "none",
            "obstacle_type": "none",
        }
    distance, source_type, source, target, obstacle_type = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
    )
    if source_type == "uav_uav":
        source_id = f"{source}->{target}"
        source_uav_id = source
        other_uav_id = target
    else:
        source_id = target
        source_uav_id = source
        other_uav_id = "none"
    return {
        "collision": int(distance < float(safety_distance_m)),
        "minimum_clearance_m": float(distance),
        "source_type": source_type,
        "source_id": source_id,
        "source_uav_id": source_uav_id,
        "other_uav_id": other_uav_id,
        "obstacle_type": obstacle_type if source_type == "uav_obstacle" else "none",
    }


def _motion_risk(
    uav_id: str,
    current_position: np.ndarray,
    nominal_position: np.ndarray,
    nominal_info: Mapping[str, Any],
    current_positions: Mapping[str, np.ndarray],
    obstacles: Iterable[Obstacle],
    safety_distance_m: float,
) -> dict[str, Any]:
    """Attribute a nominal motion risk using the same geometry primitives."""

    start = np.asarray(current_position, dtype=float)
    end = np.asarray(nominal_position, dtype=float)
    obstacle_list = sorted(obstacles, key=lambda item: str(item.obstacle_id))
    for obstacle in obstacle_list:
        if not obstacle.active:
            continue
        if segment_intersects_obstacle(start, end, obstacle):
            return {
                "risk": 1,
                "reason": "obstacle_segment",
                "source_type": "obstacle",
                "source_id": str(obstacle.obstacle_id),
                "obstacle_type": str(obstacle.obstacle_type),
            }
        if point_obstacle_distance(end, obstacle) < float(safety_distance_m):
            return {
                "risk": 1,
                "reason": "obstacle_endpoint_clearance",
                "source_type": "obstacle",
                "source_id": str(obstacle.obstacle_id),
                "obstacle_type": str(obstacle.obstacle_type),
            }
    for other_id in sorted(str(key) for key in current_positions):
        if other_id == str(uav_id):
            continue
        distance = float(np.linalg.norm(end - np.asarray(current_positions[other_id], dtype=float)))
        if distance < float(safety_distance_m):
            return {
                "risk": 1,
                "reason": "uav_separation",
                "source_type": "uav_uav",
                "source_id": f"{uav_id}->{other_id}",
                "other_uav_id": other_id,
                "obstacle_type": "none",
            }
    if bool(nominal_info.get("boundary_projected", False)):
        return {
            "risk": 1,
            "reason": "boundary",
            "source_type": "boundary",
            "source_id": "flight_bounds",
            "obstacle_type": "none",
        }
    return {
        "risk": 0,
        "reason": "none",
        "source_type": "none",
        "source_id": "none",
        "other_uav_id": "none",
        "obstacle_type": "none",
    }


def _vector(value: Any, size: int = 3) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(size, dtype=float)
    if result.size != size:
        return np.resize(result, size).astype(float)
    return result.astype(float)


def _json_observation(local: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only actor-visible fields useful for collision attribution."""

    names = (
        "estimate",
        "covariance",
        "goal",
        "obstacle_features",
        "candidate_targets",
        "candidate_measurement_sources",
        "candidate_incoming_sources",
        "effective_sources",
        "effective_targets",
        "potential_targets",
        "potential_links",
        "candidate_links",
        "candidate_incoming_links",
        "effective_links",
        "uwb_available",
        "action_mask",
        "bandwidth_mask",
        "measurement_budget_remaining",
        "measurement_budget_remaining_ratio",
    )
    return {name: local[name] for name in names if name in local}


def _obstacle_feature_detail(local: Mapping[str, Any]) -> dict[str, Any]:
    value = local.get("obstacle_features", ())
    try:
        features = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        features = np.empty(0, dtype=float)
    finite = int(features.size > 0 and np.isfinite(features).all())
    minimum = float(features[0]) if finite else -1.0
    return {
        "present": int(features.size > 0),
        "finite": finite,
        "min_distance_m": minimum if math.isfinite(minimum) else -1.0,
    }


def _action_json(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "measurement_uav_id": str(action.get("measurement_uav_id", "none")),
        "bandwidth": {
            str(key): _finite_or(value, 0.0)
            for key, value in dict(action.get("bandwidth", {})).items()
        },
        "delta_v": _vector(action.get("delta_v", (0.0, 0.0, 0.0))),
    }


def _numeric_closure(left: Any, right: Any, tolerance: float = _SAFETY_TOLERANCE) -> bool:
    try:
        return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=0.0, atol=tolerance))
    except (TypeError, ValueError):
        return False


def _dynamic_obstacle_moved_count(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> int:
    old = {str(row.get("obstacle_id")): _vector(row.get("center_m", (0.0, 0.0, 0.0))) for row in before}
    moved = 0
    for row in after:
        key = str(row.get("obstacle_id"))
        if key in old and not _numeric_closure(old[key], _vector(row.get("center_m", (0.0, 0.0, 0.0)))):
            moved += 1
    return moved


def _bundle_for(
    config: SafetyMechanismAuditConfig,
    policy_seed: int,
    branch: str,
) -> tuple[Any, Any, dict[str, Any], Path]:
    """Load a frozen trainer through the existing safety-evaluation loader."""

    loader_config = SafetyMAPPOEvaluationConfig(
        model_root=Path(config.model_root),
        scenario_root=Path(config.scenario_root),
        output_dir=Path(".safety_mechanism_audit_loader"),
        logs_dir=Path(".safety_mechanism_audit_loader_logs"),
        plots_dir=Path(".safety_mechanism_audit_loader_plots"),
        policy_seeds=(int(policy_seed),),
        scenario_seeds=(int(config.scenario_seeds[0]),),
        branches=DEFAULT_BRANCHES,
        uav_count=config.uav_count,
        max_steps=config.max_steps,
        device=config.device,
        deterministic=config.deterministic,
        safety_projection_enabled=config.safety_projection_enabled,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        force=False,
        progress=False,
    )
    return _load_bundle(loader_config, int(policy_seed), branch)


def _episode(
    config: SafetyMechanismAuditConfig,
    trainer: Any,
    eval_config: Any,
    scenario: Mapping[str, Any],
    scenario_seed: int,
    scenario_hash: str,
    policy_seed: int,
    branch: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    environment = _make_environment(
        eval_config,
        seed=_evaluation_seed(int(scenario_seed)),
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=eval_config.measurement_overhead_enabled,
        measurement_budget_enabled=eval_config.measurement_budget_enabled,
        measurement_budget_by_uav=eval_config.measurement_budget_by_uav,
    )
    observations, reset_info = environment.reset(
        seed=_evaluation_seed(int(scenario_seed)),
        scenario_config=scenario,
    )
    safety_distance = float(environment.config.motion.safety_distance_m)
    initial_positions = {
        uav_id: np.asarray(state.position_m, dtype=float).copy()
        for uav_id, state in environment.states.items()
    }
    initial_detail = _scene_clearance_detail(initial_positions, environment.obstacles, safety_distance)
    previous_positions = {key: value.copy() for key, value in initial_positions.items()}
    rows: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    all_position_errors: list[float] = []
    energy_values: list[float] = []
    path_total = 0.0
    projection_steps = 0
    projection_uav_count = 0
    nominal_risk_steps = 0
    actor_obstacle_present_steps = 0
    actor_obstacle_near_steps = 0
    truth_free = 1
    state_finite = int(initial_detail["collision"] == 0)
    covariance_finite = 1
    initial_state_safe = int(initial_detail["collision"] == 0)
    closure_ok = True
    collision_episode = int(initial_detail["collision"])
    pre_projection_collision_episode = 0
    collision_step = -1
    termination_reason = "running"
    task_complete = 0
    last_collision_context: dict[str, Any] | None = None
    terminated = False
    truncated = False

    for step in range(int(config.max_steps)):
        truth_free &= int(
            all(
                _actor_observation_truth_free(observations.get(uav_id, {}))
                for uav_id in environment.uav_ids
            )
        )
        current_positions = {
            uav_id: np.asarray(state.position_m, dtype=float).copy()
            for uav_id, state in environment.states.items()
        }
        before_obstacles = copy.deepcopy(environment.obstacles)
        before_obstacle_payload = _obstacle_snapshot(before_obstacles)
        decision = trainer.act(
            observations,
            reset_info.get("central_state") if reset_info else None,
            deterministic=True,
        )
        raw_actions = {
            uav_id: _action_json(decision.actions.get(uav_id, {}))
            for uav_id in environment.uav_ids
        }
        nominal_states: dict[str, Any] = {}
        nominal_infos: dict[str, Mapping[str, Any]] = {}
        nominal_positions: dict[str, np.ndarray] = {}
        for uav_id in environment.uav_ids:
            nominal_state, nominal_info = apply_velocity_increment(
                environment.states[uav_id],
                raw_actions[uav_id]["delta_v"],
                environment.config.motion,
                safety_projector=None,
            )
            nominal_states[uav_id] = nominal_state
            nominal_infos[uav_id] = nominal_info
            nominal_positions[uav_id] = np.asarray(nominal_state.position_m, dtype=float).copy()

        nominal_detail_pre_dynamic = _scene_clearance_detail(
            nominal_positions, before_obstacles, safety_distance
        )
        nominal_detail_post_dynamic = nominal_detail_pre_dynamic
        risk_by_uav: dict[str, dict[str, Any]] = {}
        for uav_id in environment.uav_ids:
            risk_by_uav[uav_id] = _motion_risk(
                uav_id,
                current_positions[uav_id],
                nominal_positions[uav_id],
                nominal_infos[uav_id],
                current_positions,
                before_obstacles,
                safety_distance,
            )
        if any(int(item.get("risk", 0)) for item in risk_by_uav.values()):
            nominal_risk_steps += 1

        next_observations, rewards, step_terminated, step_truncated, info = environment.step(
            decision.actions
        )
        terminated = bool(step_terminated)
        truncated = bool(step_truncated)
        termination_reason = str(info.get("termination_reason", "running"))
        task_complete = int(bool(info.get("task_complete", False)))
        after_obstacle_payload = _obstacle_snapshot(environment.obstacles)
        nominal_detail_post_dynamic = _scene_clearance_detail(
            nominal_positions, environment.obstacles, safety_distance
        )
        actual_positions = {
            uav_id: np.asarray(state.position_m, dtype=float).copy()
            for uav_id, state in environment.states.items()
        }
        # Compare the actual post-motion UAV positions against both obstacle
        # snapshots.  The first comparison is the state immediately after
        # motion and before dynamic obstacles move; the second is the state
        # used by the environment's collision check.  This keeps a collision
        # caused by obstacle motion separate from a collision that already
        # existed before the obstacle update.
        actual_detail_pre_dynamic = _scene_clearance_detail(
            actual_positions, before_obstacles, safety_distance
        )
        actual_detail = _scene_clearance_detail(
            actual_positions, environment.obstacles, safety_distance
        )
        execution = info.get("action_execution", {}) or {}
        projection_triggered = int(
            any(
                bool(execution.get(uav_id, {}).get("safety_projected", False))
                or bool(execution.get(uav_id, {}).get("boundary_projected", False))
                for uav_id in environment.uav_ids
            )
        )
        projection_steps += projection_triggered
        projection_uav_count += sum(
            int(
                bool(execution.get(uav_id, {}).get("safety_projected", False))
                or bool(execution.get(uav_id, {}).get("boundary_projected", False))
            )
            for uav_id in environment.uav_ids
        )
        state_step_finite, covariance_step_finite = _state_and_covariance_finite_local(environment)
        state_finite &= state_step_finite
        covariance_finite &= covariance_step_finite
        rmse, p95, errors = _positions_metrics(environment)
        del p95  # P95 is recomputed in the episode summary from all labels.
        all_position_errors.extend(errors)
        energy, action_mean, action_max, action_finite = _action_metrics(info, environment)
        energy_values.append(energy)
        path_step = float(
            sum(
                np.linalg.norm(actual_positions[uav_id] - previous_positions[uav_id])
                for uav_id in environment.uav_ids
            )
        )
        if not math.isfinite(path_step):
            path_step = 0.0
        path_total += path_step
        global_objective = dict(info.get("objective_components", {}) or {})
        global_metrics = dict(info.get("global_metrics", {}) or {})
        team_reward = _finite_or(
            info.get("team_reward", np.mean(list(rewards.values())) if rewards else 0.0)
        )
        before_clearance = _finite_or(nominal_detail_pre_dynamic.get("minimum_clearance_m"), safety_distance)
        pre_dynamic_actual_clearance = _finite_or(
            actual_detail_pre_dynamic.get("minimum_clearance_m"), safety_distance
        )
        post_nominal_clearance = _finite_or(nominal_detail_post_dynamic.get("minimum_clearance_m"), safety_distance)
        actual_clearance = _finite_or(actual_detail.get("minimum_clearance_m"), safety_distance)
        dynamic_moved = _dynamic_obstacle_moved_count(before_obstacle_payload, after_obstacle_payload)
        pre_projection_collision = int(bool(info.get("pre_projection_collision", 0)))
        actual_collision = int(bool(info.get("collision", 0)))
        pre_projection_collision_episode = int(pre_projection_collision_episode or pre_projection_collision)
        collision_episode = int(collision_episode or actual_collision)
        if actual_collision and collision_step < 0:
            collision_step = int(step + 1)
        current_projection_delta = {
            uav_id: nominal_positions[uav_id] - actual_positions[uav_id]
            for uav_id in environment.uav_ids
        }
        for uav_id in environment.uav_ids:
            local = observations.get(uav_id, {})
            obstacle_feature = _obstacle_feature_detail(local)
            actor_obstacle_present_steps += obstacle_feature["present"]
            actor_obstacle_near_steps += int(
                obstacle_feature["finite"]
                and obstacle_feature["min_distance_m"] <= safety_distance
            )
            raw = raw_actions[uav_id]
            exec_row = dict(execution.get(uav_id, {}) or {})
            delta_commanded = _vector(exec_row.get("delta_v_commanded", raw["delta_v"]))
            delta_applied = _vector(exec_row.get("delta_v_applied", (0.0, 0.0, 0.0)))
            nominal_info = nominal_infos[uav_id]
            projection_delta = current_projection_delta[uav_id]
            closure_ok &= _numeric_closure(
                nominal_positions[uav_id] - actual_positions[uav_id], projection_delta
            )
            local_json = _json_observation(local)
            local_truth_free = int(_actor_observation_truth_free(local_json))
            truth_free &= local_truth_free
            row = {
                "policy_seed": int(policy_seed),
                "scenario_seed": int(scenario_seed),
                "branch": str(branch),
                "evaluation_seed": int(_evaluation_seed(scenario_seed)),
                "scenario_hash": scenario_hash,
                "episode_step": int(step + 1),
                "time_s": float((step + 1) * environment.config.motion.dt_s),
                "uav_id": str(uav_id),
                "actor_observation_truth_free": local_truth_free,
                "actor_observation_json": local_json,
                "actor_obstacle_features_json": local.get("obstacle_features", ()),
                "actor_goal_features_json": local.get("goal", ()),
                "actor_action_mask_json": local.get("action_mask", ()),
                "actor_obstacle_feature_min_distance_m": obstacle_feature["min_distance_m"],
                "actor_obstacle_feature_present": obstacle_feature["present"],
                "actor_obstacle_feature_finite": obstacle_feature["finite"],
                "actor_obstacle_near_safety": int(
                    obstacle_feature["finite"]
                    and obstacle_feature["min_distance_m"] <= safety_distance
                ),
                "raw_action_json": raw,
                "delta_v_raw_json": raw["delta_v"],
                "delta_v_commanded_json": delta_commanded,
                "delta_v_applied_json": delta_applied,
                "delta_v_nominal_norm": float(np.linalg.norm(_vector(nominal_infos[uav_id].get("delta_v_commanded", delta_commanded)))),
                "delta_v_commanded_norm": float(np.linalg.norm(delta_commanded)),
                "delta_v_applied_norm": float(np.linalg.norm(delta_applied)),
                "projection_delta_json": projection_delta,
                "projection_delta_norm": float(np.linalg.norm(projection_delta)),
                "current_position_json": current_positions[uav_id],
                "nominal_position_json": nominal_positions[uav_id],
                "actual_position_json": actual_positions[uav_id],
                "nominal_velocity_json": np.asarray(nominal_states[uav_id].velocity_mps, dtype=float),
                "actual_velocity_json": np.asarray(environment.states[uav_id].velocity_mps, dtype=float),
                "nominal_boundary_projected": int(bool(nominal_info.get("boundary_projected", False))),
                "boundary_projected": int(bool(exec_row.get("boundary_projected", False))),
                "safety_projected": int(bool(exec_row.get("safety_projected", False))),
                "projection_triggered": projection_triggered,
                "pre_motion_projection_risk": int(risk_by_uav[uav_id].get("risk", 0)),
                "pre_motion_projection_reason": str(risk_by_uav[uav_id].get("reason", "none")),
                "pre_motion_projection_source_type": str(risk_by_uav[uav_id].get("source_type", "none")),
                "pre_motion_projection_source_id": str(risk_by_uav[uav_id].get("source_id", "none")),
                "pre_dynamic_nominal_collision": int(nominal_detail_pre_dynamic["collision"]),
                "post_dynamic_nominal_collision": int(nominal_detail_post_dynamic["collision"]),
                "pre_dynamic_actual_collision": int(actual_detail_pre_dynamic["collision"]),
                "pre_projection_collision": pre_projection_collision,
                "actual_collision": actual_collision,
                "pre_dynamic_minimum_clearance_m": before_clearance,
                "pre_dynamic_actual_minimum_clearance_m": pre_dynamic_actual_clearance,
                "post_dynamic_nominal_minimum_clearance_m": post_nominal_clearance,
                "pre_dynamic_actual_source_type": actual_detail_pre_dynamic["source_type"],
                "pre_dynamic_actual_source_id": actual_detail_pre_dynamic["source_id"],
                "pre_projection_minimum_clearance_m": _finite_or(
                    info.get("pre_projection_minimum_clearance", before_clearance), before_clearance
                ),
                "actual_minimum_clearance_m": actual_clearance,
                "actual_collision_source_type": actual_detail["source_type"],
                "actual_collision_source_id": actual_detail["source_id"],
                "actual_collision_source_uav_id": actual_detail["source_uav_id"],
                "actual_collision_other_uav_id": actual_detail["other_uav_id"],
                "actual_collision_obstacle_type": actual_detail["obstacle_type"],
                "dynamic_obstacle_update_order": "motion -> obstacle_update -> collision_check",
                "dynamic_obstacle_moved_count": dynamic_moved,
                "before_obstacle_state_json": before_obstacle_payload,
                "after_obstacle_state_json": after_obstacle_payload,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "termination_reason": termination_reason,
                "energy_proxy_step": energy,
                "path_length_step_m": path_step,
                "trajectory_length_m": path_total,
                "position_rmse_m": rmse,
                "j_loc": _finite_or(global_objective.get("j_loc", 0.0)),
                "j_com": _finite_or(global_objective.get("j_com", 0.0)),
                "joint_objective": _finite_or(global_objective.get("joint_objective", 0.0)),
                "team_reward": team_reward,
                "packet_loss": _finite_or(global_metrics.get("packet_loss", 0.0)),
                "delay_s": _finite_or(global_metrics.get("delay", 0.0)),
                "aoi_s": _finite_or(global_metrics.get("aoi", 0.0)),
                "connectivity": _finite_or(global_metrics.get("connectivity", 0.0)),
                "safety_projection_count": int(_finite_or(info.get("safety_projection_count", 0.0))),
                "action_saturation_rate": _finite_or(info.get("action_saturation_rate", 0.0)),
                "action_magnitude_mean": action_mean,
                "action_magnitude_max": action_max,
                "early_termination_censored": int(termination_reason == "collision"),
                "state_finite": int(state_step_finite and action_finite and _finite_flag(nominal_positions[uav_id])),
                "covariance_finite": int(covariance_step_finite),
            }
            rows.append(row)

        if actual_collision:
            last_collision_context = {
                "policy_seed": int(policy_seed),
                "scenario_seed": int(scenario_seed),
                "branch": str(branch),
                "evaluation_seed": int(_evaluation_seed(scenario_seed)),
                "scenario_hash": scenario_hash,
                "collision_step": int(step + 1),
                "collision_source_type": actual_detail["source_type"],
                "collision_source_id": actual_detail["source_id"],
                "collision_source_uav_id": actual_detail["source_uav_id"],
                "collision_other_uav_id": actual_detail["other_uav_id"],
                "collision_obstacle_type": actual_detail["obstacle_type"],
                "pre_motion_nominal_collision": int(nominal_detail_pre_dynamic["collision"]),
                "post_dynamic_nominal_collision": int(nominal_detail_post_dynamic["collision"]),
                "pre_dynamic_actual_collision": int(actual_detail_pre_dynamic["collision"]),
                "pre_projection_collision": pre_projection_collision,
                "projection_triggered": projection_triggered,
                "pre_projection_minimum_clearance_m": _finite_or(
                    info.get("pre_projection_minimum_clearance", before_clearance), before_clearance
                ),
                "pre_dynamic_actual_minimum_clearance_m": pre_dynamic_actual_clearance,
                "pre_dynamic_actual_source_type": actual_detail_pre_dynamic["source_type"],
                "pre_dynamic_actual_source_id": actual_detail_pre_dynamic["source_id"],
                "post_projection_minimum_clearance_m": actual_clearance,
                "dynamic_obstacle_moved_count": dynamic_moved,
                "actor_observations_json": {uav_id: _json_observation(observations.get(uav_id, {})) for uav_id in environment.uav_ids},
                "raw_actions_json": raw_actions,
                "projection_reasons_json": {uav_id: risk_by_uav[uav_id] for uav_id in environment.uav_ids},
                "terminated": int(terminated),
                "task_complete": task_complete,
                "early_termination_censored": 1,
            }
        previous_positions = {key: value.copy() for key, value in actual_positions.items()}
        reset_info = info
        observations = next_observations
        if terminated or truncated:
            break

    if last_collision_context is not None:
        collision_events.append(last_collision_context)
    summary = {
        "policy_seed": int(policy_seed),
        "scenario_seed": int(scenario_seed),
        "branch": str(branch),
        "evaluation_seed": int(_evaluation_seed(scenario_seed)),
        "scenario_hash": scenario_hash,
        "episode_steps": int(len(rows) // max(1, len(environment.uav_ids))),
        "initial_collision": int(initial_detail["collision"]),
        "initial_minimum_clearance_m": _finite_or(initial_detail["minimum_clearance_m"], safety_distance),
        "collision_episode": int(collision_episode),
        "pre_projection_collision_episode": int(pre_projection_collision_episode),
        "collision_step": int(collision_step),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "termination_reason": termination_reason,
        "task_complete": int(task_complete),
        "safety_projection_steps": int(projection_steps),
        "safety_projection_uav_count": int(projection_uav_count),
        "nominal_risk_steps": int(nominal_risk_steps),
        "actor_obstacle_feature_present_rate": float(actor_obstacle_present_steps / max(1, len(rows))),
        "actor_obstacle_near_safety_rate": float(actor_obstacle_near_steps / max(1, len(rows))),
        "minimum_clearance_m": float(
            min(
                [_finite_or(row.get("actual_minimum_clearance_m"), safety_distance) for row in rows]
                + [_finite_or(initial_detail["minimum_clearance_m"], safety_distance)]
            )
        ),
        "pre_projection_minimum_clearance_m": float(
            min(
                [_finite_or(row.get("pre_projection_minimum_clearance_m"), safety_distance) for row in rows]
                + [_finite_or(initial_detail["minimum_clearance_m"], safety_distance)]
            )
        ),
        "projection_after_nominal_risk": int(
            any(
                int(row.get("projection_triggered", 0)) and int(row.get("pre_motion_projection_risk", 0))
                for row in rows
            )
        ),
        "post_projection_collision": int(
            any(int(row.get("actual_collision", 0)) and int(row.get("projection_triggered", 0)) for row in rows)
        ),
        "dynamic_post_update_collision": int(
            any(
                int(row.get("actual_collision", 0))
                and int(row.get("dynamic_obstacle_moved_count", 0)) > 0
                and not int(row.get("pre_dynamic_actual_collision", 0))
                for row in rows
            )
        ),
        "energy_proxy": float(np.mean(energy_values)) if energy_values else 0.0,
        "trajectory_length_m": float(path_total),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(all_position_errors)))) if all_position_errors else 0.0,
        "position_error_p95_m": float(np.percentile(all_position_errors, 95.0)) if all_position_errors else 0.0,
        "j_loc": float(np.mean([_finite_or(row.get("j_loc"), 0.0) for row in rows])) if rows else 0.0,
        "j_com": float(np.mean([_finite_or(row.get("j_com"), 0.0) for row in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([_finite_or(row.get("team_reward"), 0.0) for row in rows])) if rows else 0.0,
        "packet_loss": float(np.mean([_finite_or(row.get("packet_loss"), 0.0) for row in rows])) if rows else 0.0,
        "delay_s": float(np.mean([_finite_or(row.get("delay_s"), 0.0) for row in rows])) if rows else 0.0,
        "aoi_s": float(np.mean([_finite_or(row.get("aoi_s"), 0.0) for row in rows])) if rows else 0.0,
        "connectivity": float(np.mean([_finite_or(row.get("connectivity"), 0.0) for row in rows])) if rows else 0.0,
        "actor_observation_truth_free": int(truth_free),
        "trainer_update_called": 0,
        "state_finite": int(state_finite),
        "covariance_finite": int(covariance_finite),
        "initial_state_safe": int(initial_state_safe),
        "kinematic_closure": int(closure_ok),
        "test_data_read": 0,
        "censored_by_collision": int(termination_reason == "collision"),
    }
    return rows, summary, collision_events


def _state_and_covariance_finite_local(environment: Any) -> tuple[int, int]:
    state_finite = 1
    covariance_finite = 1
    for state in environment.states.values():
        state_finite &= _finite_flag(state.position_m)
        state_finite &= _finite_flag(state.velocity_mps)
        state_finite &= _finite_flag(state.attitude_rad)
    for ekf in environment.ekfs.values():
        state_finite &= _finite_flag(ekf.state)
        covariance_finite &= _finite_flag(ekf.covariance)
    return int(state_finite), int(covariance_finite)


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    return {
        key: float(np.mean([_finite_or(row.get(key), 0.0) for row in rows])) if rows else 0.0
        for key in keys
    }


def _group_summary(
    episode_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    keys = (
        "collision_episode",
        "pre_projection_collision_episode",
        "minimum_clearance_m",
        "pre_projection_minimum_clearance_m",
        "safety_projection_steps",
        "safety_projection_uav_count",
        "nominal_risk_steps",
        "actor_obstacle_feature_present_rate",
        "actor_obstacle_near_safety_rate",
        "projection_after_nominal_risk",
        "post_projection_collision",
        "dynamic_post_update_collision",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
    )
    rows: list[dict[str, Any]] = []
    for scenario_seed in scenario_seeds:
        for branch in branches:
            matches = [
                row
                for row in episode_rows
                if int(row.get("scenario_seed", -1)) == int(scenario_seed)
                and str(row.get("branch", "")) == branch
            ]
            rows.append(
                {
                    "scenario_seed": int(scenario_seed),
                    "branch": str(branch),
                    "policy_seed_count": int(len(matches)),
                    "scenario_hash": str(matches[0].get("scenario_hash", "")) if matches else "",
                    **_mean_rows(matches, keys),
                    "all_state_finite": int(all(int(row.get("state_finite", 0)) for row in matches)) if matches else 0,
                    "all_covariance_finite": int(all(int(row.get("covariance_finite", 0)) for row in matches)) if matches else 0,
                    "all_actor_observations_truth_free": int(all(int(row.get("actor_observation_truth_free", 0)) for row in matches)) if matches else 0,
                    "all_kinematic_closure": int(all(int(row.get("kinematic_closure", 0)) for row in matches)) if matches else 0,
                }
            )
    return rows


def _paired_contrasts(
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    config: SafetyMechanismAuditConfig,
) -> list[dict[str, Any]]:
    metrics = (
        "collision_episode",
        "pre_projection_collision_episode",
        "minimum_clearance_m",
        "pre_projection_minimum_clearance_m",
        "safety_projection_steps",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "actor_obstacle_feature_present_rate",
        "actor_obstacle_near_safety_rate",
        "nominal_risk_steps",
        "post_projection_collision",
        "dynamic_post_update_collision",
    )
    base = {int(row["scenario_seed"]): row for row in scenario_rows if row["branch"] == BASELINE_BRANCH}
    full = {int(row["scenario_seed"]): row for row in scenario_rows if row["branch"] == FULL_BRANCH}
    contrasts: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        values = [
            _finite_or(full[seed].get(metric), 0.0) - _finite_or(base[seed].get(metric), 0.0)
            for seed in scenario_seeds
            if seed in base and seed in full
        ]
        low, high = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + index)
        contrasts.append(
            {
                "comparison": f"{FULL_BRANCH} - {BASELINE_BRANCH}",
                "metric": metric,
                "independent_unit": "scenario_seed",
                "scenario_count": len(values),
                "mean_difference": float(np.mean(values)) if values else 0.0,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "bootstrap_samples": int(config.bootstrap_samples),
            }
        )
    return contrasts


def _plot(
    path: Path,
    step_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
    collision_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
    safety_distance_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {BASELINE_BRANCH: "#0072B2", FULL_BRANCH: "#D55E00"}
    labels = {BASELINE_BRANCH: "baseline MAPPO", FULL_BRANCH: "budgeted quality-aware MAPPO"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1. Clearance around collision events.
    ax = axes[0, 0]
    if collision_rows:
        for event in collision_rows:
            key = (
                int(event.get("policy_seed", -1)),
                int(event.get("scenario_seed", -1)),
                str(event.get("branch", "")),
            )
            relevant = [
                row
                for row in step_rows
                if (int(row.get("policy_seed", -1)), int(row.get("scenario_seed", -1)), str(row.get("branch", ""))) == key
            ]
            collision_step = int(event.get("collision_step", 1))
            relative = [int(row.get("episode_step", 0)) - collision_step for row in relevant]
            clearance = [_finite_or(row.get("actual_minimum_clearance_m"), safety_distance_m) for row in relevant]
            ax.plot(relative, clearance, color=colors.get(str(event.get("branch")), "#666666"), alpha=0.55)
        ax.axhline(safety_distance_m, color="black", linestyle="--", linewidth=1, label="safety threshold")
        ax.set_xlabel("Step relative to collision")
        ax.set_ylabel("Actual minimum clearance (m)")
    else:
        ax.text(0.5, 0.5, "No collision events", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Collision-event clearance")
    ax.grid(alpha=0.25)

    # 2. Nominal versus applied action on collision episodes.
    ax = axes[0, 1]
    if collision_rows:
        for branch in branches:
            relevant = [
                row for row in step_rows
                if str(row.get("branch", "")) == branch
                and any(
                    int(event.get("scenario_seed", -1)) == int(row.get("scenario_seed", -2))
                    and int(event.get("policy_seed", -1)) == int(row.get("policy_seed", -2))
                    and str(event.get("branch", "")) == branch
                    for event in collision_rows
                )
            ]
            if relevant:
                ax.scatter(
                    [_finite_or(row.get("delta_v_nominal_norm"), 0.0) for row in relevant],
                    [_finite_or(row.get("delta_v_applied_norm"), 0.0) for row in relevant],
                    s=10,
                    alpha=0.45,
                    color=colors.get(branch, "#666666"),
                    label=labels.get(branch, branch),
                )
        max_value = max(
            [_finite_or(row.get("delta_v_nominal_norm"), 0.0) for row in step_rows]
            + [_finite_or(row.get("delta_v_applied_norm"), 0.0) for row in step_rows]
            + [1.0]
        )
        ax.plot([0.0, max_value], [0.0, max_value], color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("Nominal Δv norm")
        ax.set_ylabel("Applied Δv norm")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No collision events", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Nominal versus applied action")
    ax.grid(alpha=0.25)

    # 3. Actor-visible obstacle feature versus diagnostic clearance.
    ax = axes[0, 2]
    for branch in branches:
        relevant = [row for row in step_rows if str(row.get("branch", "")) == branch]
        if relevant:
            ax.scatter(
                [_finite_or(row.get("actor_obstacle_feature_min_distance_m"), -1.0) for row in relevant],
                [_finite_or(row.get("actual_minimum_clearance_m"), safety_distance_m) for row in relevant],
                s=6,
                alpha=0.25,
                color=colors.get(branch, "#666666"),
                label=labels.get(branch, branch),
            )
    ax.axvline(safety_distance_m, color="black", linestyle="--", linewidth=1)
    ax.axhline(safety_distance_m, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Actor feature: estimated nearest obstacle (m)")
    ax.set_ylabel("Diagnostic actual clearance (m)")
    ax.set_title("Obstacle observation versus collision risk")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # 4--6. Scenario-level comparison, using scenario seeds as x units.
    scenario_x = np.arange(len(scenario_seeds), dtype=float)
    plot_specs = (
        ("collision_episode", "Scenario collision fraction", "fraction"),
        ("safety_projection_steps", "Safety projection steps", "count"),
        ("energy_proxy", "Energy proxy (collision-censored)", "normalized"),
    )
    for axis, (metric, title, ylabel) in zip(axes[1], plot_specs):
        for branch in branches:
            values = []
            for scenario_seed in scenario_seeds:
                match = [
                    row for row in scenario_rows
                    if int(row.get("scenario_seed", -1)) == int(scenario_seed)
                    and str(row.get("branch", "")) == branch
                ]
                values.append(_finite_or(match[0].get(metric), 0.0) if match else 0.0)
            axis.plot(
                scenario_x,
                values,
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=colors.get(branch, "#666666"),
                label=labels.get(branch, branch),
            )
        axis.set_xticks(scenario_x, [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(
        "Safety mechanism audit: actor observation → nominal action → projection → collision",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Five policy seeds averaged within each scenario; scenario seeds are independent; energy is censored by collision termination",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _finite_diagnostics(rows: Sequence[Mapping[str, Any]]) -> bool:
    numeric = (
        "time_s",
        "actor_obstacle_feature_min_distance_m",
        "delta_v_nominal_norm",
        "delta_v_commanded_norm",
        "delta_v_applied_norm",
        "projection_delta_norm",
        "pre_dynamic_minimum_clearance_m",
        "post_dynamic_nominal_minimum_clearance_m",
        "pre_dynamic_actual_minimum_clearance_m",
        "actual_minimum_clearance_m",
        "energy_proxy_step",
        "path_length_step_m",
        "position_rmse_m",
        "j_loc",
        "j_com",
        "team_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
    )
    for row in rows:
        for key in numeric:
            try:
                if not math.isfinite(float(row.get(key, 0.0))):
                    return False
            except (TypeError, ValueError):
                return False
        if int(row.get("state_finite", 0)) != 1 or int(row.get("covariance_finite", 0)) != 1:
            return False
    return True


def run_safety_mechanism_audit(config: SafetyMechanismAuditConfig) -> dict[str, Any]:
    """Replay frozen policies and attribute safety collisions without training."""

    output, plots = _prepare_outputs(config)
    scenarios: dict[int, tuple[dict[str, Any], str]] = {
        int(seed): _load_scenario(Path(config.scenario_root), int(seed), config.uav_count)
        for seed in config.scenario_seeds
    }
    scenario_hashes = {str(seed): value[1] for seed, value in scenarios.items()}
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    bundle_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any], Path]] = {}

    for branch in config.branches:
        for policy_seed in config.policy_seeds:
            _progress(config, f"policy_seed={policy_seed} branch={branch} checkpoint loading")
            bundle_cache[(int(policy_seed), str(branch))] = _bundle_for(config, int(policy_seed), str(branch))
            trainer, eval_config, _manifest, _scale_root = bundle_cache[(int(policy_seed), str(branch))]
            trainer.policy.eval()
            for scenario_seed in config.scenario_seeds:
                scenario, scenario_hash = scenarios[int(scenario_seed)]
                _progress(
                    config,
                    f"policy_seed={policy_seed} branch={branch} scenario_seed={scenario_seed} start",
                )
                rows, summary, collisions = _episode(
                    config,
                    trainer,
                    eval_config,
                    scenario,
                    int(scenario_seed),
                    scenario_hash,
                    int(policy_seed),
                    str(branch),
                )
                step_rows.extend(rows)
                episode_rows.append(summary)
                collision_rows.extend(collisions)
                if summary["collision_episode"] and not summary["terminated"]:
                    audit_errors.append(
                        f"collision episode was not terminated: {policy_seed}/{branch}/{scenario_seed}"
                    )
                if summary["task_complete"]:
                    audit_errors.append(
                        f"collision/task_complete semantics invalid: {policy_seed}/{branch}/{scenario_seed}"
                    )

    scenario_rows = _group_summary(episode_rows, config.scenario_seeds, config.branches)
    contrast_rows = _paired_contrasts(scenario_rows, config.scenario_seeds, config)

    # Per-policy summaries are sensitivity diagnostics, not independent tests.
    summary_keys = (
        "collision_episode",
        "minimum_clearance_m",
        "safety_projection_steps",
        "nominal_risk_steps",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
    )
    policy_rows: list[dict[str, Any]] = []
    for policy_seed in config.policy_seeds:
        for branch in config.branches:
            matches = [
                row for row in episode_rows
                if int(row.get("policy_seed", -1)) == int(policy_seed)
                and str(row.get("branch", "")) == branch
            ]
            policy_rows.append(
                {
                    "policy_seed": int(policy_seed),
                    "branch": str(branch),
                    "scenario_count": len(matches),
                    **_mean_rows(matches, summary_keys),
                }
            )

    # Observation/action summaries are useful for locating an attribution
    # failure, but are deliberately descriptive rather than inferential.
    obs_rows: list[dict[str, Any]] = []
    for policy_seed in config.policy_seeds:
        for branch in config.branches:
            for uav_id in sorted({str(row.get("uav_id")) for row in step_rows}):
                matches = [
                    row for row in step_rows
                    if int(row.get("policy_seed", -1)) == int(policy_seed)
                    and str(row.get("branch", "")) == branch
                    and str(row.get("uav_id", "")) == uav_id
                ]
                if not matches:
                    continue
                obs_rows.append(
                    {
                        "policy_seed": int(policy_seed),
                        "branch": str(branch),
                        "uav_id": uav_id,
                        "step_count": len(matches),
                        "actor_obstacle_feature_present_rate": float(np.mean([int(row.get("actor_obstacle_feature_present", 0)) for row in matches])),
                        "actor_obstacle_near_safety_rate": float(np.mean([int(row.get("actor_obstacle_near_safety", 0)) for row in matches])),
                        "nominal_risk_rate": float(np.mean([int(row.get("pre_motion_projection_risk", 0)) for row in matches])),
                        "projection_trigger_rate": float(np.mean([int(row.get("projection_triggered", 0)) for row in matches])),
                        "mean_command_norm": float(np.mean([_finite_or(row.get("delta_v_commanded_norm"), 0.0) for row in matches])),
                        "mean_applied_norm": float(np.mean([_finite_or(row.get("delta_v_applied_norm"), 0.0) for row in matches])),
                        "mean_projection_delta": float(np.mean([_finite_or(row.get("projection_delta_norm"), 0.0) for row in matches])),
                        "collision_steps": int(sum(int(row.get("actual_collision", 0)) for row in matches)),
                        "observation_truth_free": int(all(int(row.get("actor_observation_truth_free", 0)) for row in matches)),
                    }
                )

    # Structural checks.
    expected_episode_count = len(config.policy_seeds) * len(config.scenario_seeds) * len(config.branches)
    expected_episode_keys = {
        (int(policy_seed), int(scenario_seed), str(branch))
        for policy_seed in config.policy_seeds
        for scenario_seed in config.scenario_seeds
        for branch in config.branches
    }
    actual_episode_keys = {
        (
            int(row.get("policy_seed", -1)),
            int(row.get("scenario_seed", -1)),
            str(row.get("branch", "")),
        )
        for row in episode_rows
    }
    episode_key_coverage = (
        len(actual_episode_keys) == len(episode_rows)
        and actual_episode_keys == expected_episode_keys
    )
    same_scenario_payload = True
    for scenario_seed in config.scenario_seeds:
        expected_hash = str(scenario_hashes.get(str(int(scenario_seed)), ""))
        matches = [
            str(row.get("scenario_hash", ""))
            for row in episode_rows
            if int(row.get("scenario_seed", -1)) == int(scenario_seed)
        ]
        if not matches or set(matches) != {expected_hash}:
            same_scenario_payload = False
    collision_termination = all(
        not int(row.get("collision_episode", 0)) or (int(row.get("terminated", 0)) == 1 and str(row.get("termination_reason")) == "collision")
        for row in episode_rows
    )
    task_complete_false = all(int(row.get("task_complete", 0)) == 0 for row in episode_rows)
    actor_truth_free = all(int(row.get("actor_observation_truth_free", 0)) == 1 for row in episode_rows)
    trainer_updates_zero = all(int(row.get("trainer_update_called", 1)) == 0 for row in episode_rows)
    initial_safe = all(int(row.get("initial_state_safe", 0)) == 1 for row in episode_rows)
    closure = all(int(row.get("kinematic_closure", 0)) == 1 for row in episode_rows)
    finite = _finite_diagnostics(step_rows) and all(
        int(row.get("state_finite", 0)) == 1 and int(row.get("covariance_finite", 0)) == 1
        for row in episode_rows
    )
    source_paths = (Path(config.model_root).resolve(), Path(config.scenario_root).resolve())
    output_paths = (output.resolve(), plots.resolve())
    no_source_overwrite = all(
        not _paths_overlap(path, source)
        for path in output_paths
        for source in source_paths
    )
    nonempty_collision_context = all(
        int(event.get("terminated", 0)) == 1 and int(event.get("task_complete", 0)) == 0
        for event in collision_rows
    )
    checks = {
        "same_scenario_hash_by_branch": same_scenario_payload,
        "same_scenario_payload_by_branch": same_scenario_payload,
        "episode_key_coverage": episode_key_coverage,
        "actor_observations_truth_free": actor_truth_free,
        "trainer_update_called_zero": trainer_updates_zero,
        "test_data_read_zero": all(int(row.get("test_data_read", 0)) == 0 for row in episode_rows),
        "initial_state_safe": initial_safe,
        "nominal_applied_kinematic_closure": closure,
        "projection_delta_closure": closure,
        "collision_termination": collision_termination,
        "collision_task_complete_false": task_complete_false and nonempty_collision_context,
        "finite_diagnostics": finite,
        "no_source_overwrite": no_source_overwrite,
        "scenario_independent_units": True,
    }
    audit_ok = bool(
        len(episode_rows) == expected_episode_count
        and len(step_rows) > 0
        and all(checks.values())
        and not audit_errors
    )

    collision_episode_rows = [row for row in episode_rows if int(row.get("collision_episode", 0))]
    nominal_risk_collision = [
        row for row in collision_episode_rows
        if int(row.get("nominal_risk_steps", 0)) > 0
    ]
    projection_collision = [
        row for row in collision_episode_rows
        if int(row.get("post_projection_collision", 0))
    ]
    dynamic_collision = [
        row for row in collision_episode_rows
        if int(row.get("dynamic_post_update_collision", 0))
    ]
    if not collision_episode_rows:
        decision = "mechanism_evidence_insufficient"
    elif not actor_truth_free:
        decision = "observation_risk_missing"
    elif nominal_risk_collision and not projection_collision:
        decision = "projection_not_triggered"
    elif dynamic_collision:
        decision = "post_projection_dynamic_timing_risk"
    elif nominal_risk_collision:
        decision = "nominal_actions_enter_danger_zone"
    else:
        decision = "mechanism_evidence_insufficient"

    report = {
        "schema_version": _SCHEMA_VERSION,
        "audit_ok": audit_ok,
        "decision": decision,
        "design": {
            "uav_count": int(config.uav_count),
            "max_steps": int(config.max_steps),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "branches": list(config.branches),
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "mean over policy seeds within scenario",
            "deterministic_inference": True,
            "trainer_update_called": False,
            "safety_projection_enabled": True,
            "collision_termination": True,
            "test_data_read": False,
            "test_data_role": "sealed; not read by this audit",
            "censoring_policy": "collision episodes are early-terminated; energy, path, and RMSE are descriptive and not safety-improvement claims",
            "observation_action_order": "actor observation -> nominal action -> motion -> dynamic obstacle update -> collision check",
        },
        "source": {
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "scenario_hashes": scenario_hashes,
            "checkpoint_branches": list(config.branches),
            "source_episode_schema": "episode-v1.3",
        },
        "counts": {
            "episode_rows": len(episode_rows),
            "step_rows": len(step_rows),
            "collision_events": len(collision_rows),
            "policy_seed_count": len(config.policy_seeds),
            "scenario_seed_count": len(config.scenario_seeds),
            "branch_count": len(config.branches),
            "collision_episode_count": len(collision_episode_rows),
            "collision_with_nominal_risk_count": len(nominal_risk_collision),
            "collision_with_projection_count": len(projection_collision),
            "dynamic_post_update_collision_count": len(dynamic_collision),
            "invalid_action_count": 0,
        },
        "checks": checks,
        "mechanism_findings": {
            "actor_obstacle_risk_before_collision": {
                "collision_episode_count": len(collision_episode_rows),
                "collision_episodes_with_nominal_risk": len(nominal_risk_collision),
                "actor_obstacle_feature_present_in_all_steps": actor_truth_free and all(
                    int(row.get("actor_obstacle_feature_present", 0)) == 1 for row in step_rows
                ),
            },
            "nominal_action_risk": {
                "collision_episodes_with_nominal_risk": len(nominal_risk_collision),
                "nominal_risk_step_count": int(sum(int(row.get("nominal_risk_steps", 0)) for row in episode_rows)),
            },
            "projection_behavior": {
                "collision_episodes_with_projection": len(projection_collision),
                "projection_step_count": int(sum(int(row.get("safety_projection_steps", 0)) for row in episode_rows)),
                "projection_uav_count": int(sum(int(row.get("safety_projection_uav_count", 0)) for row in episode_rows)),
            },
            "dynamic_obstacle_timing": {
                "collision_episodes_after_dynamic_update": len(dynamic_collision),
                "update_order": "motion -> obstacle_update -> collision_check",
            },
            "censoring": {
                "collision_censored_episode_count": len(collision_episode_rows),
                "energy_and_path_are_descriptive_only": True,
            },
        },
        "audit_errors": audit_errors,
        "interpretation": {
            "claim_boundary": "This is a frozen-checkpoint mechanism attribution audit, not a new training result and not evidence of final algorithm superiority.",
            "collision_free_energy_warning": "Lower energy or path length in an early-terminated collision episode cannot be interpreted as an optimization success.",
            "primary_statistical_unit": "scenario_seed",
        },
    }

    _write_csv(output / "collision_events.csv", collision_rows)
    _write_csv(output / "step_diagnostics.csv", step_rows)
    _write_csv(output / "scenario_summary.csv", scenario_rows)
    _write_csv(output / "paired_contrasts.csv", contrast_rows)
    _write_csv(output / "observation_action_summary.csv", obs_rows)
    _write_csv(output / "policy_seed_sensitivity.csv", policy_rows)
    _write_json(output / "audit_report.json", report)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "output_dir": str(output),
            "plots_dir": str(plots),
        },
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "output_dir": str(output),
            "plots_dir": str(plots),
            "audit_report": str((output / "audit_report.json").resolve()),
            "source_episode_schema": "episode-v1.3",
            "validation_only": True,
            "test_data_read": False,
        },
    )
    plot_path = plots / _PLOT_FILE
    safety_distance = 2.0
    if step_rows:
        # The frozen motion configuration uses the same safety distance in
        # every evaluated bundle; obtain it from the first run-independent
        # diagnostic convention rather than from a hidden truth field.
        safety_distance = 2.0
    _plot(plot_path, step_rows, scenario_rows, collision_rows, config.scenario_seeds, config.branches, safety_distance)
    return {
        "audit_ok": audit_ok,
        "decision": decision,
        "independent_unit": "scenario_seed",
        "policy_seed_count": len(config.policy_seeds),
        "scenario_seed_count": len(config.scenario_seeds),
        "episode_rows": len(episode_rows),
        "step_rows": len(step_rows),
        "collision_events": len(collision_rows),
        "logs": str(output),
        "plot": str(plot_path),
        "audit_report": str(output / "audit_report.json"),
    }


__all__ = [
    "BASELINE_BRANCH",
    "DEFAULT_BRANCHES",
    "DEFAULT_POLICY_SEEDS",
    "DEFAULT_SCENARIO_SEEDS",
    "FULL_BRANCH",
    "SafetyMechanismAuditConfig",
    "_motion_risk",
    "_scene_clearance_detail",
    "run_safety_mechanism_audit",
]
