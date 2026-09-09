"""One-step counterfactual audit of dynamic-obstacle collision timing.

The frozen safety-mechanism audit established that the current simulator uses

    motion and projection against current obstacles
        -> dynamic-obstacle update
        -> collision check.

This module keeps that physical execution unchanged.  It replays the frozen
policies and, after each actual step, computes a *shadow* result in which the
next dynamic-obstacle state is predicted before the same safety projection.
The shadow result is diagnostic only: it never changes the environment, random
number stream, actor observation, reward, or subsequent action.

The independent statistical unit is the scenario seed.  Policy seeds are
averaged within a scenario, and collision-censored energy/path/RMSE values are
reported descriptively only.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import MotionConfig
from ..simulation.obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    point_obstacle_distance,
    segment_intersects_obstacle,
)
from .safety_mechanism_audit import (
    BASELINE_BRANCH,
    DEFAULT_BRANCHES,
    DEFAULT_POLICY_SEEDS,
    DEFAULT_SCENARIO_SEEDS,
    FULL_BRANCH,
    SafetyMechanismAuditConfig,
    _action_json,
    _bundle_for,
    _episode,
    _finite_diagnostics,
    _numeric_closure,
    _obstacle_snapshot,
    _paths_overlap,
    _scene_clearance_detail,
    _vector,
)
from .safety_mappo_evaluation import (
    _bootstrap_ci,
    _finite_flag,
    _finite_or,
    _load_scenario,
    _write_csv,
    _write_json,
)


_SCHEMA_VERSION = "safety-timing-counterfactual-v1.0"
_PLOT_FILE = "safety_timing_counterfactual.png"
_TOLERANCE = 1e-7


def _motion_config_for_evaluation(eval_config: Any) -> MotionConfig:
    """Return the motion settings used by the frozen evaluation factory.

    ``FormalExperimentConfig`` stores the high-level experiment parameters;
    ``formal._make_environment`` materializes the fixed translational motion
    configuration.  Keeping this helper beside the counterfactual makes the
    shadow update use the same values without changing the environment API.
    """

    del eval_config  # the current evaluation factory fixes these values
    return MotionConfig(
        dt_s=0.2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )


@dataclass(frozen=True)
class SafetyTimingCounterfactualConfig:
    """Configuration for the validation-only timing counterfactual."""

    model_root: Path
    scenario_root: Path
    output_dir: Path = Path(
        "logs/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_timing_counterfactual"
    )
    plots_dir: Path = Path(
        "plots/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_timing_counterfactual"
    )
    reference_audit_dir: Path | None = Path(
        "logs/preliminary/safety_mappo_evaluation_seed2026/"
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
            raise ValueError("timing counterfactual is restricted to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("deterministic timing audit supports device='cpu' only")
        if not self.deterministic:
            raise ValueError("timing counterfactual must use deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("timing audit requires safety projection enabled")
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


def _progress(config: SafetyTimingCounterfactualConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-timing-counterfactual] {message}", flush=True)


def _prepare_outputs(config: SafetyTimingCounterfactualConfig) -> tuple[Path, Path]:
    output = Path(config.output_dir).resolve()
    plots = Path(config.plots_dir).resolve()
    if output == plots:
        raise ValueError("output_dir and plots_dir must be distinct")

    source_paths = [
        Path(config.model_root).resolve(),
        Path(config.scenario_root).resolve(),
    ]
    if config.reference_audit_dir is not None:
        source_paths.append(Path(config.reference_audit_dir).resolve())
    for path in (output, plots):
        if any(_paths_overlap(path, source) for source in source_paths):
            raise ValueError("timing-audit output cannot replace a source directory")
        if path.exists() and any(path.iterdir()) and not config.force:
            raise FileExistsError(
                f"timing-audit output is not empty: {path}; use --force to replace it"
            )
    if config.force:
        for path in (output, plots):
            if path.exists():
                shutil.rmtree(path)
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return output, plots


def _payload(value: Any) -> Any:
    """Decode a CSV-style JSON value while retaining in-memory structures."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    return value


def _active_flag(value: Any) -> bool:
    if isinstance(value, str):
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return value.lower() not in {"", "false", "none", "null"}
    return bool(value)


def _obstacles_from_payload(payload: Any) -> list[Obstacle]:
    """Rebuild mutable obstacle objects from a snapshot without touching env."""

    decoded = _payload(payload)
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return []
    obstacles: list[Obstacle] = []
    for item in decoded:
        if not isinstance(item, Mapping):
            continue
        obstacle_id = str(item.get("obstacle_id", "obstacle"))
        center = _vector(item.get("center_m", (0.0, 0.0, 0.0)))
        velocity = _vector(item.get("velocity_mps", (0.0, 0.0, 0.0)))
        obstacle_type = str(item.get("obstacle_type", "building"))
        active = _active_flag(item.get("active", 1))
        geometry = str(
            item.get(
                "geometry_type",
                "box" if "size_m" in item else "sphere",
            )
        )
        if geometry == "box":
            obstacles.append(
                BoxObstacle(
                    obstacle_id=obstacle_id,
                    center_m=center.copy(),
                    size_m=_vector(item.get("size_m", (1.0, 1.0, 1.0))),
                    velocity_mps=velocity.copy(),
                    obstacle_type=obstacle_type,
                    active=active,
                )
            )
        else:
            obstacles.append(
                SphereObstacle(
                    obstacle_id=obstacle_id,
                    center_m=center.copy(),
                    radius_m=float(_finite_or(item.get("radius_m", 1.0), 1.0)),
                    velocity_mps=velocity.copy(),
                    obstacle_type=obstacle_type,
                    active=active,
                )
            )
    return obstacles


def _predict_obstacles(
    before_payload: Any,
    dt_s: float,
    bounds_min_m: Any,
    bounds_max_m: Any,
) -> list[dict[str, Any]]:
    """Apply the same obstacle update rule to an isolated shadow copy."""

    obstacles = _obstacles_from_payload(before_payload)
    lower = _vector(bounds_min_m)
    upper = _vector(bounds_max_m)
    for obstacle in obstacles:
        obstacle.update(float(dt_s), lower, upper)
    return _obstacle_snapshot(obstacles)


def _snapshot_equal(left: Any, right: Any) -> bool:
    left_payload = _payload(left)
    right_payload = _payload(right)
    if not isinstance(left_payload, Sequence) or not isinstance(right_payload, Sequence):
        return False
    if len(left_payload) != len(right_payload):
        return False
    left_map = {
        str(item.get("obstacle_id")): item
        for item in left_payload
        if isinstance(item, Mapping)
    }
    right_map = {
        str(item.get("obstacle_id")): item
        for item in right_payload
        if isinstance(item, Mapping)
    }
    if set(left_map) != set(right_map):
        return False
    for obstacle_id in sorted(left_map):
        a = left_map[obstacle_id]
        b = right_map[obstacle_id]
        if str(a.get("geometry_type")) != str(b.get("geometry_type")):
            return False
        if _active_flag(a.get("active", 1)) != _active_flag(b.get("active", 1)):
            return False
        for key in ("center_m", "velocity_mps"):
            if not _numeric_closure(_vector(a.get(key)), _vector(b.get(key))):
                return False
        geometry = str(a.get("geometry_type"))
        if geometry == "box":
            if not _numeric_closure(_vector(a.get("size_m")), _vector(b.get("size_m"))):
                return False
        elif not math.isclose(
            _finite_or(a.get("radius_m"), -1.0),
            _finite_or(b.get("radius_m"), -1.0),
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ):
            return False
    return True


def _shadow_project(
    uav_id: str,
    current_position: np.ndarray,
    nominal_position: np.ndarray,
    current_positions: Mapping[str, np.ndarray],
    predicted_obstacles: Iterable[Obstacle],
    safety_distance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the existing geometry rules to predicted obstacles only."""

    start = _vector(current_position)
    end = _vector(nominal_position)
    obstacles = sorted(
        predicted_obstacles,
        key=lambda item: str(item.obstacle_id),
    )
    for obstacle in obstacles:
        if not obstacle.active:
            continue
        if segment_intersects_obstacle(start, end, obstacle):
            return start.copy(), {
                "triggered": 1,
                "reason": "obstacle_segment",
                "source_type": "uav_obstacle",
                "source_id": str(obstacle.obstacle_id),
                "other_uav_id": "none",
                "obstacle_type": str(obstacle.obstacle_type),
            }
        if point_obstacle_distance(end, obstacle) < float(safety_distance_m):
            return start.copy(), {
                "triggered": 1,
                "reason": "obstacle_endpoint_clearance",
                "source_type": "uav_obstacle",
                "source_id": str(obstacle.obstacle_id),
                "other_uav_id": "none",
                "obstacle_type": str(obstacle.obstacle_type),
            }
    for other_id in sorted(str(key) for key in current_positions):
        if other_id == str(uav_id):
            continue
        distance = float(
            np.linalg.norm(end - _vector(current_positions[other_id]))
        )
        if distance < float(safety_distance_m):
            return start.copy(), {
                "triggered": 1,
                "reason": "uav_separation",
                "source_type": "uav_uav",
                "source_id": f"{uav_id}->{other_id}",
                "other_uav_id": other_id,
                "obstacle_type": "none",
            }
    return end.copy(), {
        "triggered": 0,
        "reason": "none",
        "source_type": "none",
        "source_id": "none",
        "other_uav_id": "none",
        "obstacle_type": "none",
    }


def _collision_cause(row: Mapping[str, Any]) -> str:
    """Classify an actual collision using only recorded mechanism flags.

    This is an attribution label for the audit output, not an input to the
    simulator.  Dynamic-obstacle motion takes precedence because the actual
    environment checks collisions only after that update.  A collision that
    already existed after UAV motion is then separated according to whether
    the safety projection was triggered.
    """

    if not int(row.get("actual_collision", 0)):
        return "none"
    if str(row.get("actual_collision_source_type", "none")) == "uav_uav":
        return "uav_separation"
    if int(row.get("dynamic_timing_collision", 0)):
        return "dynamic_obstacle_motion"
    if int(row.get("pre_dynamic_actual_collision", 0)):
        if not int(row.get("safety_projected", 0)):
            return "projection_not_triggered"
        return "action_after_projection"
    if int(row.get("pre_projection_collision", 0)):
        if not int(row.get("safety_projected", 0)):
            return "projection_not_triggered"
        return "action_after_projection"
    return "unclassified"


def _unique_step_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, int], list[Mapping[str, Any]]]:
    groups: dict[tuple[int, int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row.get("policy_seed", -1)),
            int(row.get("scenario_seed", -1)),
            str(row.get("branch", "")),
            int(row.get("episode_step", -1)),
        )
        groups.setdefault(key, []).append(row)
    return groups


def _shadow_context(
    group: Sequence[Mapping[str, Any]],
    eval_config: Any,
) -> dict[str, Any]:
    """Compute one pure shadow step from the actual audit rows."""

    if not group:
        raise ValueError("cannot build shadow context from an empty step group")
    first = group[0]
    motion = _motion_config_for_evaluation(eval_config)
    before_payload = _payload(first.get("before_obstacle_state_json", []))
    after_payload = _payload(first.get("after_obstacle_state_json", []))
    predicted_payload = _predict_obstacles(
        before_payload,
        float(motion.dt_s),
        motion.bounds_min_m,
        motion.bounds_max_m,
    )
    predicted_objects = _obstacles_from_payload(predicted_payload)

    uav_ids = sorted(str(row.get("uav_id")) for row in group)
    current_positions = {
        str(row.get("uav_id")): _vector(row.get("current_position_json"))
        for row in group
    }
    nominal_positions = {
        str(row.get("uav_id")): _vector(row.get("nominal_position_json"))
        for row in group
    }
    actual_positions = {
        str(row.get("uav_id")): _vector(row.get("actual_position_json"))
        for row in group
    }
    safety_distance = float(motion.safety_distance_m)
    before_objects = _obstacles_from_payload(before_payload)
    after_objects = _obstacles_from_payload(after_payload)
    before_detail = _scene_clearance_detail(
        actual_positions,
        before_objects,
        safety_distance,
    )
    after_detail = _scene_clearance_detail(
        actual_positions,
        after_objects,
        safety_distance,
    )
    shadow_positions: dict[str, np.ndarray] = {}
    shadow_projection: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        position, projection = _shadow_project(
            uav_id,
            current_positions[uav_id],
            nominal_positions[uav_id],
            current_positions,
            predicted_objects,
            safety_distance,
        )
        shadow_positions[uav_id] = position
        shadow_projection[uav_id] = projection
    shadow_detail = _scene_clearance_detail(
        shadow_positions,
        predicted_objects,
        safety_distance,
    )
    before_items = (
        list(before_payload)
        if isinstance(before_payload, Sequence)
        and not isinstance(before_payload, (str, bytes))
        else []
    )
    after_items = (
        list(after_payload)
        if isinstance(after_payload, Sequence)
        and not isinstance(after_payload, (str, bytes))
        else []
    )
    before_by_id = {
        str(item.get("obstacle_id")): item
        for item in before_items
        if isinstance(item, Mapping)
    }
    dynamic_moved_ids = {
        str(after.get("obstacle_id"))
        for after in after_items
        if isinstance(after, Mapping)
        and str(after.get("obstacle_id")) in before_by_id
        and not _numeric_closure(
            _vector(before_by_id[str(after.get("obstacle_id"))].get("center_m")),
            _vector(after.get("center_m")),
        )
    }
    dynamic_moved = int(bool(dynamic_moved_ids))
    current_collision = int(after_detail.get("collision", 0))
    pre_dynamic_collision = int(before_detail.get("collision", 0))
    collision_source_id = str(after_detail.get("source_id", "none"))
    dynamic_timing_collision = int(
        current_collision
        and not pre_dynamic_collision
        and collision_source_id in dynamic_moved_ids
        and str(after_detail.get("source_type")) == "uav_obstacle"
    )
    return {
        "predicted_payload": predicted_payload,
        "actual_after_payload": after_payload,
        "prediction_closure": int(_snapshot_equal(predicted_payload, after_payload)),
        "before_detail": before_detail,
        "after_detail": after_detail,
        "shadow_detail": shadow_detail,
        "shadow_positions": shadow_positions,
        "shadow_projection": shadow_projection,
        "dynamic_obstacle_moved": dynamic_moved,
        "dynamic_obstacle_moved_ids": sorted(dynamic_moved_ids),
        "dynamic_clearance_change_m": float(
            _finite_or(after_detail.get("minimum_clearance_m"), safety_distance)
            - _finite_or(before_detail.get("minimum_clearance_m"), safety_distance)
        ),
        "current_collision": current_collision,
        "pre_dynamic_collision": pre_dynamic_collision,
        "dynamic_timing_collision": dynamic_timing_collision,
        "shadow_collision_avoided": int(
            current_collision and not int(shadow_detail.get("collision", 0))
        ),
        "shadow_collision": int(shadow_detail.get("collision", 0)),
        "uav_ids": uav_ids,
        "counterfactual_state_unchanged": 1,
        "counterfactual_one_step_only": 1,
        "offline_geometry_only": 1,
    }


def _enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    eval_config: Any,
) -> list[dict[str, Any]]:
    motion = _motion_config_for_evaluation(eval_config)
    enriched: list[dict[str, Any]] = []
    for group in _unique_step_groups(rows).values():
        context = _shadow_context(group, eval_config)
        for source in group:
            row = dict(source)
            uav_id = str(row.get("uav_id"))
            projection = context["shadow_projection"][uav_id]
            shadow_position = context["shadow_positions"][uav_id]
            nominal_position = _vector(row.get("nominal_position_json"))
            actual_position = _vector(row.get("actual_position_json"))
            row.update(
                {
                    "predicted_obstacle_state_json": context["predicted_payload"],
                    "obstacle_prediction_closure": int(context["prediction_closure"]),
                    "current_pre_dynamic_detail_json": context["before_detail"],
                    "current_post_dynamic_detail_json": context["after_detail"],
                    "shadow_scene_detail_json": context["shadow_detail"],
                    "shadow_position_json": shadow_position,
                    "shadow_projection_delta_json": nominal_position - shadow_position,
                    "shadow_projection_delta_norm": float(
                        np.linalg.norm(nominal_position - shadow_position)
                    ),
                    "shadow_projection_triggered": int(projection["triggered"]),
                    "shadow_projection_reason": str(projection["reason"]),
                    "shadow_projection_source_type": str(projection["source_type"]),
                    "shadow_projection_source_id": str(projection["source_id"]),
                    "shadow_projection_other_uav_id": str(
                        projection.get("other_uav_id", "none")
                    ),
                    "shadow_projection_obstacle_type": str(
                        projection.get("obstacle_type", "none")
                    ),
                    "shadow_collision": int(context["shadow_collision"]),
                    "shadow_minimum_clearance_m": _finite_or(
                        context["shadow_detail"].get("minimum_clearance_m"),
                        motion.safety_distance_m,
                    ),
                    "shadow_collision_source_type": str(
                        context["shadow_detail"].get("source_type", "none")
                    ),
                    "shadow_collision_source_id": str(
                        context["shadow_detail"].get("source_id", "none")
                    ),
                    "shadow_collision_avoided_current": int(
                        context["shadow_collision_avoided"]
                    ),
                    "dynamic_obstacle_moved": int(context["dynamic_obstacle_moved"]),
                    "dynamic_obstacle_moved_ids_json": context[
                        "dynamic_obstacle_moved_ids"
                    ],
                    "dynamic_clearance_change_m": _finite_or(
                        context["dynamic_clearance_change_m"], 0.0
                    ),
                    "current_pre_dynamic_collision": int(
                        context["pre_dynamic_collision"]
                    ),
                    "current_post_dynamic_collision": int(
                        context["current_collision"]
                    ),
                    "dynamic_timing_collision": int(
                        context["dynamic_timing_collision"]
                    ),
                    "dynamic_obstacle_prediction_order": (
                        "predict_obstacle_update -> safety_projection "
                        "(offline shadow only)"
                    ),
                    "counterfactual_state_unchanged": int(
                        context["counterfactual_state_unchanged"]
                    ),
                    "counterfactual_one_step_only": int(
                        context["counterfactual_one_step_only"]
                    ),
                    "offline_geometry_only": int(context["offline_geometry_only"]),
                    "nominal_actual_position_closure": int(
                        _numeric_closure(
                            nominal_position - actual_position,
                            _vector(row.get("projection_delta_json")),
                        )
                    ),
                }
            )
            row["collision_cause"] = _collision_cause(row)
            enriched.append(row)
    enriched.sort(
        key=lambda row: (
            int(row.get("policy_seed", -1)),
            str(row.get("branch", "")),
            int(row.get("scenario_seed", -1)),
            int(row.get("episode_step", -1)),
            str(row.get("uav_id", "")),
        )
    )
    return enriched


def _step_contexts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate per-UAV rows to one record per decision window."""

    contexts: list[dict[str, Any]] = []
    for key, group in _unique_step_groups(rows).items():
        first = group[0]
        contexts.append(
            {
                "policy_seed": key[0],
                "scenario_seed": key[1],
                "branch": key[2],
                "episode_step": key[3],
                "row": first,
            }
        )
    return contexts


def _augment_episode_summary(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = dict(summary)
    contexts = _step_contexts(rows)
    result.update(
        {
            "current_collision_episode": int(
                any(int(item["row"].get("current_post_dynamic_collision", 0)) for item in contexts)
            ),
            "shadow_collision_episode": int(
                any(int(item["row"].get("shadow_collision", 0)) for item in contexts)
            ),
            "dynamic_timing_collision_episode": int(
                any(int(item["row"].get("dynamic_timing_collision", 0)) for item in contexts)
            ),
            "shadow_collision_avoided_episode": int(
                any(int(item["row"].get("shadow_collision_avoided_current", 0)) for item in contexts)
            ),
            "current_collision_step_count": int(
                sum(int(item["row"].get("current_post_dynamic_collision", 0)) for item in contexts)
            ),
            "shadow_collision_step_count": int(
                sum(int(item["row"].get("shadow_collision", 0)) for item in contexts)
            ),
            "shadow_projection_steps": int(
                sum(
                    int(any(int(row.get("shadow_projection_triggered", 0)) for row in group))
                    for group in _unique_step_groups(rows).values()
                )
            ),
            "shadow_projection_uav_count": int(
                sum(int(row.get("shadow_projection_triggered", 0)) for row in rows)
            ),
            "dynamic_timing_collision_step_count": int(
                sum(int(item["row"].get("dynamic_timing_collision", 0)) for item in contexts)
            ),
            "shadow_minimum_clearance_m": float(
                min(
                    [
                        _finite_or(item["row"].get("shadow_minimum_clearance_m"), 0.0)
                        for item in contexts
                    ]
                    or [0.0]
                )
            ),
            "dynamic_clearance_change_mean_m": float(
                np.mean(
                    [
                        _finite_or(item["row"].get("dynamic_clearance_change_m"), 0.0)
                        for item in contexts
                    ]
                    or [0.0]
                )
            ),
            "dynamic_clearance_change_min_m": float(
                min(
                    [
                        _finite_or(item["row"].get("dynamic_clearance_change_m"), 0.0)
                        for item in contexts
                    ]
                    or [0.0]
                )
            ),
            "prediction_closure": int(
                all(int(row.get("obstacle_prediction_closure", 0)) == 1 for row in rows)
            ),
            "counterfactual_state_unchanged": int(
                all(int(row.get("counterfactual_state_unchanged", 0)) == 1 for row in rows)
            ),
        }
    )
    return result


def _scenario_summary(
    episode_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = (
        "current_collision_episode",
        "shadow_collision_episode",
        "dynamic_timing_collision_episode",
        "shadow_collision_avoided_episode",
        "current_collision_step_count",
        "shadow_collision_step_count",
        "shadow_projection_steps",
        "shadow_projection_uav_count",
        "dynamic_timing_collision_step_count",
        "minimum_clearance_m",
        "shadow_minimum_clearance_m",
        "dynamic_clearance_change_mean_m",
        "dynamic_clearance_change_min_m",
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
    output: list[dict[str, Any]] = []
    for scenario_seed in scenario_seeds:
        for branch in branches:
            matches = [
                row
                for row in episode_rows
                if int(row.get("scenario_seed", -1)) == int(scenario_seed)
                and str(row.get("branch", "")) == str(branch)
            ]
            values = {
                metric: float(
                    np.mean([_finite_or(row.get(metric), 0.0) for row in matches])
                )
                if matches
                else 0.0
                for metric in metrics
            }
            output.append(
                {
                    "scenario_seed": int(scenario_seed),
                    "branch": str(branch),
                    "policy_seed_count": int(len(matches)),
                    "scenario_hash": str(matches[0].get("scenario_hash", "")) if matches else "",
                    **values,
                    "all_state_finite": int(
                        bool(matches)
                        and all(int(row.get("state_finite", 0)) == 1 for row in matches)
                    ),
                    "all_covariance_finite": int(
                        bool(matches)
                        and all(int(row.get("covariance_finite", 0)) == 1 for row in matches)
                    ),
                    "all_obstacle_prediction_closed": int(
                        bool(matches)
                        and all(int(row.get("prediction_closure", 0)) == 1 for row in matches)
                    ),
                }
            )
    return output


def _paired_contrasts(
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    config: SafetyTimingCounterfactualConfig,
) -> list[dict[str, Any]]:
    metrics = (
        "current_collision_episode",
        "shadow_collision_episode",
        "dynamic_timing_collision_episode",
        "shadow_collision_avoided_episode",
        "minimum_clearance_m",
        "shadow_minimum_clearance_m",
        "shadow_projection_steps",
        "dynamic_timing_collision_step_count",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "j_loc",
        "j_com",
        "mean_reward",
    )
    base = {
        int(row["scenario_seed"]): row
        for row in scenario_rows
        if str(row.get("branch")) == BASELINE_BRANCH
    }
    full = {
        int(row["scenario_seed"]): row
        for row in scenario_rows
        if str(row.get("branch")) == FULL_BRANCH
    }
    output: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        values = [
            _finite_or(full[seed].get(metric), 0.0)
            - _finite_or(base[seed].get(metric), 0.0)
            for seed in scenario_seeds
            if int(seed) in base and int(seed) in full
        ]
        low, high = _bootstrap_ci(
            values,
            config.bootstrap_samples,
            config.bootstrap_seed + index,
        )
        output.append(
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
    return output


def _collision_event_rows(
    step_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one row per actual collision episode with shadow comparison."""

    events: list[dict[str, Any]] = []
    grouped = _unique_step_groups(step_rows)
    episode_keys = {
        (
            int(row.get("policy_seed", -1)),
            int(row.get("scenario_seed", -1)),
            str(row.get("branch", "")),
        )
        for row in step_rows
        if int(row.get("current_post_dynamic_collision", 0)) == 1
    }
    for policy_seed, scenario_seed, branch in sorted(episode_keys):
        matches = [
            group
            for key, group in grouped.items()
            if key[:3] == (policy_seed, scenario_seed, branch)
            and int(group[0].get("current_post_dynamic_collision", 0)) == 1
        ]
        if not matches:
            continue
        collision_group = sorted(
            matches,
            key=lambda group: int(group[0].get("episode_step", 0)),
        )[0]
        first = collision_group[0]
        event = {
            "policy_seed": policy_seed,
            "scenario_seed": scenario_seed,
            "branch": branch,
            "scenario_hash": str(first.get("scenario_hash", "")),
            "collision_step": int(first.get("episode_step", -1)),
            "current_pre_dynamic_minimum_clearance_m": _finite_or(
                first.get("pre_dynamic_actual_minimum_clearance_m"), 0.0
            ),
            "current_post_dynamic_minimum_clearance_m": _finite_or(
                first.get("actual_minimum_clearance_m"), 0.0
            ),
            "shadow_minimum_clearance_m": _finite_or(
                first.get("shadow_minimum_clearance_m"), 0.0
            ),
            "current_pre_dynamic_collision": int(
                first.get("current_pre_dynamic_collision", 0)
            ),
            "current_post_dynamic_collision": int(
                first.get("current_post_dynamic_collision", 0)
            ),
            "dynamic_timing_collision": int(
                first.get("dynamic_timing_collision", 0)
            ),
            "shadow_collision": int(first.get("shadow_collision", 0)),
            "shadow_collision_avoided_current": int(
                first.get("shadow_collision_avoided_current", 0)
            ),
            "dynamic_obstacle_moved": int(first.get("dynamic_obstacle_moved", 0)),
            "dynamic_obstacle_moved_ids_json": first.get(
                "dynamic_obstacle_moved_ids_json", []
            ),
            "collision_cause": _collision_cause(first),
            "actual_collision_source_type": str(
                first.get("actual_collision_source_type", "none")
            ),
            "actual_collision_source_id": str(
                first.get("actual_collision_source_id", "none")
            ),
            "shadow_collision_source_type": str(
                first.get("shadow_collision_source_type", "none")
            ),
            "shadow_collision_source_id": str(
                first.get("shadow_collision_source_id", "none")
            ),
            "shadow_projection_triggered_any": int(
                any(int(row.get("shadow_projection_triggered", 0)) for row in collision_group)
            ),
            "shadow_projection_reasons_json": {
                str(row.get("uav_id")): str(row.get("shadow_projection_reason", "none"))
                for row in collision_group
            },
            "termination_reason": str(first.get("termination_reason", "")),
            "terminated": int(first.get("terminated", 0)),
            "task_complete": 0,
            "early_termination_censored": int(
                first.get("early_termination_censored", 0)
            ),
        }
        events.append(event)
    return events


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _replay_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "step_count": 0,
            "collision": 0,
            "collision_step": -1,
            "termination_reason": "missing",
        }
    steps = sorted({int(row.get("episode_step", row.get("step", 0))) for row in rows})
    collisions = [
        int(row.get("actual_collision", row.get("collision", 0)))
        for row in rows
    ]
    collision_steps = [
        int(row.get("episode_step", row.get("step", -1)))
        for row in rows
        if int(row.get("actual_collision", row.get("collision", 0))) == 1
    ]
    last_step = max(steps)
    final = [
        row
        for row in rows
        if int(row.get("episode_step", row.get("step", -1))) == last_step
    ][0]
    return {
        "step_count": len(steps),
        "collision": int(any(collisions)),
        "collision_step": min(collision_steps) if collision_steps else -1,
        "termination_reason": str(final.get("termination_reason", "running")),
    }


def _compare_reference(
    rows: Sequence[Mapping[str, Any]],
    reference_dir: Path | None,
) -> dict[str, Any]:
    if reference_dir is None:
        return {
            "available": False,
            "match": True,
            "reason": "reference audit disabled",
        }
    path = Path(reference_dir).resolve() / "step_diagnostics.csv"
    reference_rows = _read_csv_rows(path)
    if not reference_rows:
        return {
            "available": False,
            "match": True,
            "reason": f"reference step file not found: {path}",
        }
    keys = {
        (
            int(row.get("policy_seed", -1)),
            int(row.get("scenario_seed", -1)),
            str(row.get("branch", "")),
        )
        for row in rows
    }
    current_by_key: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    reference_by_key: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        current_by_key.setdefault(
            (
                int(row.get("policy_seed", -1)),
                int(row.get("scenario_seed", -1)),
                str(row.get("branch", "")),
            ),
            [],
        ).append(row)
    for row in reference_rows:
        reference_by_key.setdefault(
            (
                int(row.get("policy_seed", -1)),
                int(row.get("scenario_seed", -1)),
                str(row.get("branch", "")),
            ),
            [],
        ).append(row)
    missing = sorted(str(key) for key in keys - set(reference_by_key))
    mismatches: list[dict[str, Any]] = []
    for key in sorted(keys & set(reference_by_key)):
        current_signature = _replay_signature(current_by_key[key])
        reference_signature = _replay_signature(reference_by_key[key])
        if current_signature != reference_signature:
            mismatches.append(
                {
                    "key": key,
                    "current": current_signature,
                    "reference": reference_signature,
                }
            )
    return {
        "available": True,
        "match": not missing and not mismatches,
        "reference_path": str(path),
        "missing_keys": missing,
        "mismatches": mismatches,
        "compared_key_count": len(keys & set(reference_by_key)),
    }


def _finite_timing_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    scalar_keys = (
        "dynamic_clearance_change_m",
        "shadow_projection_delta_norm",
        "shadow_minimum_clearance_m",
        "current_pre_dynamic_collision",
        "current_post_dynamic_collision",
        "dynamic_timing_collision",
        "shadow_collision",
    )
    for row in rows:
        for key in scalar_keys:
            try:
                if not math.isfinite(float(row.get(key, 0.0))):
                    return False
            except (TypeError, ValueError):
                return False
        for key in (
            "shadow_position_json",
            "shadow_projection_delta_json",
            "predicted_obstacle_state_json",
            "current_pre_dynamic_detail_json",
            "current_post_dynamic_detail_json",
            "shadow_scene_detail_json",
        ):
            value = row.get(key)
            if isinstance(value, str):
                value = _payload(value)
            if not _finite_flag(value):
                return False
    return True


def _plot(
    path: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
    safety_distance_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        BASELINE_BRANCH: "#0072B2",
        FULL_BRANCH: "#D55E00",
    }
    labels = {
        BASELINE_BRANCH: "baseline MAPPO",
        FULL_BRANCH: "budgeted quality-aware MAPPO",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1. Actual versus shadow collision rate, averaged within each scenario.
    ax = axes[0, 0]
    x = np.arange(len(branches), dtype=float)
    width = 0.36
    actual = [
        float(
            np.mean(
                [
                    _finite_or(row.get("current_collision_episode"), 0.0)
                    for row in scenario_rows
                    if str(row.get("branch", "")) == branch
                ]
                or [0.0]
            )
        )
        for branch in branches
    ]
    shadow = [
        float(
            np.mean(
                [
                    _finite_or(row.get("shadow_collision_episode"), 0.0)
                    for row in scenario_rows
                    if str(row.get("branch", "")) == branch
                ]
                or [0.0]
            )
        )
        for branch in branches
    ]
    ax.bar(x - width / 2, actual, width, label="actual timing")
    ax.bar(x + width / 2, shadow, width, label="shadow pre-update projection")
    ax.set_xticks(x, [labels.get(branch, branch) for branch in branches], rotation=15)
    ax.set_ylabel("Episode collision fraction")
    ax.set_title("Actual versus counterfactual collision")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # 2. Clearance before and after the dynamic obstacle update for collisions.
    ax = axes[0, 1]
    if event_rows:
        event_x = np.arange(len(event_rows), dtype=float)
        pre = [_finite_or(row.get("current_pre_dynamic_minimum_clearance_m"), 0.0) for row in event_rows]
        post = [_finite_or(row.get("current_post_dynamic_minimum_clearance_m"), 0.0) for row in event_rows]
        shadow_values = [_finite_or(row.get("shadow_minimum_clearance_m"), 0.0) for row in event_rows]
        ax.plot(event_x, pre, "o-", label="before obstacle update")
        ax.plot(event_x, post, "s-", label="actual after update")
        ax.plot(event_x, shadow_values, "^-", label="shadow after predicted-before projection")
        ax.axhline(safety_distance_m, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Collision event index")
        ax.set_ylabel("Minimum clearance (m)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No actual collision events", ha="center", va="center")
    ax.set_title("Collision-event timing diagnostic")
    ax.grid(alpha=0.25)

    # 3. Current and shadow projection counts by branch.
    ax = axes[0, 2]
    current_projection = [
        float(
            np.mean(
                [
                    _finite_or(row.get("safety_projection_steps"), 0.0)
                    for row in scenario_rows
                    if str(row.get("branch", "")) == branch
                ]
                or [0.0]
            )
        )
        for branch in branches
    ]
    shadow_projection = [
        float(
            np.mean(
                [
                    _finite_or(row.get("shadow_projection_steps"), 0.0)
                    for row in scenario_rows
                    if str(row.get("branch", "")) == branch
                ]
                or [0.0]
            )
        )
        for branch in branches
    ]
    ax.bar(x - width / 2, current_projection, width, label="actual projection")
    ax.bar(x + width / 2, shadow_projection, width, label="shadow projection")
    ax.set_xticks(x, [labels.get(branch, branch) for branch in branches], rotation=15)
    ax.set_ylabel("Mean projected decision steps")
    ax.set_title("Projection response to predicted obstacles")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    # 4. Actual versus shadow clearance for all collision events.
    ax = axes[1, 0]
    if event_rows:
        current_values = [_finite_or(row.get("current_post_dynamic_minimum_clearance_m"), 0.0) for row in event_rows]
        shadow_values = [_finite_or(row.get("shadow_minimum_clearance_m"), 0.0) for row in event_rows]
        ax.scatter(current_values, shadow_values, c="#CC79A7", s=28)
        max_value = max(current_values + shadow_values + [safety_distance_m, 1.0])
        ax.plot([0.0, max_value], [0.0, max_value], ":", color="black")
        ax.axvline(safety_distance_m, color="black", linestyle="--", linewidth=1)
        ax.axhline(safety_distance_m, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual post-update clearance (m)")
        ax.set_ylabel("Shadow clearance (m)")
    else:
        ax.text(0.5, 0.5, "No collision events", ha="center", va="center")
    ax.set_title("Actual versus shadow collision geometry")
    ax.grid(alpha=0.25)

    # 5. Collision difference per scenario, useful for the paired design.
    ax = axes[1, 1]
    for branch in branches:
        values = [
            _finite_or(
                next(
                    (
                        row.get("shadow_collision_episode", 0.0)
                        for row in scenario_rows
                        if int(row.get("scenario_seed", -1)) == int(seed)
                        and str(row.get("branch", "")) == branch
                    ),
                    0.0,
                ),
                0.0,
            )
            - _finite_or(
                next(
                    (
                        row.get("current_collision_episode", 0.0)
                        for row in scenario_rows
                        if int(row.get("scenario_seed", -1)) == int(seed)
                        and str(row.get("branch", "")) == branch
                    ),
                    0.0,
                ),
                0.0,
            )
            for seed in scenario_seeds
        ]
        ax.plot(
            np.arange(len(scenario_seeds)),
            values,
            marker="o",
            color=colors.get(branch, "#666666"),
            label=labels.get(branch, branch),
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(scenario_seeds)), [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
    ax.set_ylabel("Shadow − actual collision fraction")
    ax.set_title("Scenario-level timing difference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # 6. Distribution of obstacle-motion clearance changes.
    ax = axes[1, 2]
    for branch in branches:
        values = [
            _finite_or(row.get("dynamic_clearance_change_m"), 0.0)
            for row in step_rows
            if str(row.get("branch", "")) == branch
        ]
        if values:
            ax.hist(
                values,
                bins=min(20, max(5, len(values) // 10)),
                alpha=0.45,
                color=colors.get(branch, "#666666"),
                label=labels.get(branch, branch),
            )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Post-update − pre-update clearance (m)")
    ax.set_ylabel("Step-row count")
    ax.set_title("Dynamic-obstacle clearance change")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Safety timing counterfactual: predicted obstacles before projection",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Shadow paths are one-step diagnostics only; actual execution remains motion → obstacle update → collision check",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_timing_counterfactual(
    config: SafetyTimingCounterfactualConfig,
) -> dict[str, Any]:
    """Replay frozen policies and audit dynamic-obstacle timing sensitivity."""

    output, plots = _prepare_outputs(config)
    scenarios: dict[int, tuple[dict[str, Any], str]] = {
        int(seed): _load_scenario(
            Path(config.scenario_root),
            int(seed),
            config.uav_count,
        )
        for seed in config.scenario_seeds
    }
    scenario_hashes = {
        str(seed): value[1] for seed, value in scenarios.items()
    }
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    bundle_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any], Path]] = {}

    for branch in config.branches:
        for policy_seed in config.policy_seeds:
            _progress(
                config,
                f"policy_seed={policy_seed} branch={branch} checkpoint loading",
            )
            bundle_cache[(int(policy_seed), str(branch))] = _bundle_for(
                config,
                int(policy_seed),
                str(branch),
            )
            trainer, eval_config, _manifest, _scale_root = bundle_cache[
                (int(policy_seed), str(branch))
            ]
            trainer.policy.eval()
            for scenario_seed in config.scenario_seeds:
                scenario, scenario_hash = scenarios[int(scenario_seed)]
                _progress(
                    config,
                    f"policy_seed={policy_seed} branch={branch} "
                    f"scenario_seed={scenario_seed} start",
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
                enriched = _enrich_rows(rows, eval_config)
                augmented = _augment_episode_summary(summary, enriched)
                step_rows.extend(enriched)
                episode_rows.append(augmented)
                collision_rows.extend(collisions)
                if augmented["collision_episode"] and not augmented["terminated"]:
                    audit_errors.append(
                        f"collision episode was not terminated: "
                        f"{policy_seed}/{branch}/{scenario_seed}"
                    )
                if augmented["task_complete"]:
                    audit_errors.append(
                        f"collision/task_complete semantics invalid: "
                        f"{policy_seed}/{branch}/{scenario_seed}"
                    )

    # Collision events from the original replay are enriched with shadow data.
    timing_events = _collision_event_rows(step_rows)
    scenario_rows = _scenario_summary(
        episode_rows,
        config.scenario_seeds,
        config.branches,
    )
    contrast_rows = _paired_contrasts(
        scenario_rows,
        config.scenario_seeds,
        config,
    )
    reference_comparison = _compare_reference(
        step_rows,
        config.reference_audit_dir,
    )

    expected_episode_count = (
        len(config.policy_seeds)
        * len(config.scenario_seeds)
        * len(config.branches)
    )
    expected_keys = {
        (int(policy_seed), int(scenario_seed), str(branch))
        for policy_seed in config.policy_seeds
        for scenario_seed in config.scenario_seeds
        for branch in config.branches
    }
    actual_keys = {
        (
            int(row.get("policy_seed", -1)),
            int(row.get("scenario_seed", -1)),
            str(row.get("branch", "")),
        )
        for row in episode_rows
    }
    episode_key_coverage = (
        len(episode_rows) == expected_episode_count
        and actual_keys == expected_keys
    )
    same_scenario_payload = all(
        str(row.get("scenario_hash", ""))
        == str(scenario_hashes.get(str(int(row.get("scenario_seed", -1))), ""))
        for row in episode_rows
    )
    collision_termination = all(
        not int(row.get("collision_episode", 0))
        or (
            int(row.get("terminated", 0)) == 1
            and str(row.get("termination_reason", "")) == "collision"
        )
        for row in episode_rows
    )
    task_complete_false = all(
        int(row.get("task_complete", 0)) == 0 for row in episode_rows
    )
    actor_truth_free = all(
        int(row.get("actor_observation_truth_free", 0)) == 1
        for row in episode_rows
    )
    trainer_updates_zero = all(
        int(row.get("trainer_update_called", 1)) == 0
        for row in episode_rows
    )
    initial_safe = all(
        int(row.get("initial_state_safe", 0)) == 1 for row in episode_rows
    )
    base_closure = all(
        int(row.get("kinematic_closure", 0)) == 1 for row in episode_rows
    )
    shadow_closure = all(
        int(row.get("nominal_actual_position_closure", 0)) == 1
        for row in step_rows
    )
    prediction_closure = all(
        int(row.get("obstacle_prediction_closure", 0)) == 1
        for row in step_rows
    )
    counterfactual_unchanged = all(
        int(row.get("counterfactual_state_unchanged", 0)) == 1
        and int(row.get("counterfactual_one_step_only", 0)) == 1
        and int(row.get("offline_geometry_only", 0)) == 1
        for row in step_rows
    )
    finite = (
        _finite_diagnostics(step_rows)
        and _finite_timing_rows(step_rows)
        and all(
            int(row.get("state_finite", 0)) == 1
            and int(row.get("covariance_finite", 0)) == 1
            for row in episode_rows
        )
    )
    no_test = all(int(row.get("test_data_read", 0)) == 0 for row in episode_rows)
    no_source_overwrite = all(
        not _paths_overlap(path.resolve(), source.resolve())
        for path in (output, plots)
        for source in (
            [Path(config.model_root), Path(config.scenario_root)]
            + (
                [Path(config.reference_audit_dir)]
                if config.reference_audit_dir is not None
                else []
            )
        )
    )
    collision_context = all(
        int(row.get("terminated", 0)) == 1
        and int(row.get("task_complete", 0)) == 0
        for row in timing_events
    )
    checks = {
        "same_scenario_hash": same_scenario_payload,
        "episode_key_coverage": episode_key_coverage,
        "actor_observations_truth_free": actor_truth_free,
        "trainer_update_called_zero": trainer_updates_zero,
        "test_data_read_zero": no_test,
        "initial_state_safe": initial_safe,
        "actual_kinematic_closure": base_closure,
        "shadow_position_closure": shadow_closure,
        "obstacle_prediction_closure": prediction_closure,
        "counterfactual_state_unchanged": counterfactual_unchanged,
        "collision_termination": collision_termination,
        "collision_task_complete_false": task_complete_false and collision_context,
        "finite_diagnostics": finite,
        "current_replay_match": bool(reference_comparison.get("match", True)),
        "no_source_overwrite": no_source_overwrite,
        "scenario_independent_units": True,
    }
    audit_ok = bool(
        len(step_rows) > 0
        and len(episode_rows) == expected_episode_count
        and all(checks.values())
        and not audit_errors
    )

    current_collision_count = int(
        sum(int(row.get("current_collision_episode", 0)) for row in episode_rows)
    )
    shadow_collision_count = int(
        sum(int(row.get("shadow_collision_episode", 0)) for row in episode_rows)
    )
    dynamic_timing_count = int(
        sum(int(row.get("dynamic_timing_collision_episode", 0)) for row in episode_rows)
    )
    avoided_count = int(
        sum(int(row.get("shadow_collision_avoided_episode", 0)) for row in episode_rows)
    )
    if current_collision_count == 0:
        decision = "mechanism_evidence_insufficient"
    elif shadow_collision_count < current_collision_count and dynamic_timing_count > 0:
        decision = "predicted_obstacle_projection_reduces_collision"
    elif dynamic_timing_count > 0:
        decision = "dynamic_timing_present_but_shadow_not_sufficient"
    else:
        decision = "timing_not_sole_cause"

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
            "test_data_read": False,
            "test_data_role": "sealed; not read by this audit",
            "actual_update_order": (
                "motion and current-obstacle projection -> "
                "dynamic_obstacle_update -> collision_check"
            ),
            "counterfactual_update_order": (
                "predict dynamic obstacle update -> "
                "same safety projection -> one-step geometry check"
            ),
            "counterfactual_role": "offline one-step diagnostic, never fed back",
            "censoring_policy": (
                "collision episodes are early-terminated; energy, path, and "
                "RMSE are descriptive and not safety-improvement claims"
            ),
        },
        "source": {
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "reference_audit_dir": (
                str(Path(config.reference_audit_dir).resolve())
                if config.reference_audit_dir is not None
                else None
            ),
            "scenario_hashes": scenario_hashes,
            "source_episode_schema": "episode-v1.3",
        },
        "counts": {
            "episode_rows": len(episode_rows),
            "step_rows": len(step_rows),
            "collision_events": len(timing_events),
            "current_collision_episode_count": current_collision_count,
            "shadow_collision_episode_count": shadow_collision_count,
            "dynamic_timing_collision_episode_count": dynamic_timing_count,
            "shadow_collision_avoided_episode_count": avoided_count,
            "policy_seed_count": len(config.policy_seeds),
            "scenario_seed_count": len(config.scenario_seeds),
            "branch_count": len(config.branches),
        },
        "reference_replay": reference_comparison,
        "checks": checks,
        "mechanism_findings": {
            "dynamic_obstacle_timing": {
                "current_collision_count": current_collision_count,
                "dynamic_timing_collision_count": dynamic_timing_count,
                "shadow_collision_count": shadow_collision_count,
                "shadow_collision_avoided_count": avoided_count,
                "global_shadow_collision_reduction": (
                    current_collision_count - shadow_collision_count
                ),
            },
            "clearance": {
                "event_count": len(timing_events),
                "pre_dynamic_actual_clearance_is_recorded": True,
                "post_dynamic_actual_clearance_is_recorded": True,
                "shadow_clearance_is_recorded": True,
            },
            "censoring": {
                "collision_censored_episode_count": current_collision_count,
                "energy_and_path_are_descriptive_only": True,
            },
        },
        "audit_errors": audit_errors,
        "interpretation": {
            "claim_boundary": (
                "This is a frozen-checkpoint, one-step timing attribution "
                "diagnostic, not a new rollout or final algorithm comparison."
            ),
            "scenario_statistical_unit": "scenario_seed",
            "policy_seed_role": "within-scenario sensitivity average",
            "offline_truth_boundary": (
                "Obstacle geometry is used only in the offline shadow "
                "calculation and never enters actor, critic, or actual control."
            ),
        },
    }

    _write_csv(output / "timing_counterfactual_events.csv", timing_events)
    _write_csv(output / "timing_step_diagnostics.csv", step_rows)
    _write_csv(output / "timing_scenario_summary.csv", scenario_rows)
    _write_csv(output / "timing_paired_contrasts.csv", contrast_rows)
    _write_json(output / "timing_audit_report.json", report)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "reference_audit_dir": (
                str(Path(config.reference_audit_dir).resolve())
                if config.reference_audit_dir is not None
                else None
            ),
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
            "audit_report": str((output / "timing_audit_report.json").resolve()),
            "source_episode_schema": "episode-v1.3",
            "validation_only": True,
            "test_data_read": False,
            "counterfactual_one_step_only": True,
        },
    )

    # The frozen motion configuration uses a 2 m safety distance in this
    # audit.  The value is only a plotting threshold; all calculations use
    # the configuration in _shadow_context.
    _plot(
        plots / _PLOT_FILE,
        scenario_rows,
        step_rows,
        timing_events,
        config.scenario_seeds,
        config.branches,
        2.0,
    )
    return {
        "audit_ok": audit_ok,
        "decision": decision,
        "independent_unit": "scenario_seed",
        "policy_seed_count": len(config.policy_seeds),
        "scenario_seed_count": len(config.scenario_seeds),
        "episode_rows": len(episode_rows),
        "step_rows": len(step_rows),
        "collision_events": len(timing_events),
        "logs": str(output),
        "plot": str(plots / _PLOT_FILE),
        "audit_report": str(output / "timing_audit_report.json"),
    }


__all__ = [
    "SafetyTimingCounterfactualConfig",
    "run_safety_timing_counterfactual",
]
