"""Seeded, serializable scene generation for train/validation/test splits."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np


def _inside(point: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float = 2.0) -> bool:
    return bool(np.all(point >= lower + margin) and np.all(point <= upper - margin))


def _clear_of_box(point: np.ndarray, center: np.ndarray, size: np.ndarray, margin: float) -> bool:
    lower = center - size / 2.0 - margin
    upper = center + size / 2.0 + margin
    return bool(np.any(point < lower) or np.any(point > upper))


def generate_scenario(
    seed: int,
    uav_count: int,
    *,
    bounds_min_m: tuple[float, float, float] = (0.0, 0.0, 2.0),
    bounds_max_m: tuple[float, float, float] = (200.0, 200.0, 60.0),
) -> dict[str, Any]:
    """Generate one deterministic scenario from a phase-specific seed.

    The generation rule is shared by all UAV scales.  The seed controls the
    initial formation, goals, buildings, and dynamic obstacle trajectory;
    none of these values are exposed through actor observations.
    """

    if uav_count < 1:
        raise ValueError("uav_count must be at least 1")
    rng = np.random.default_rng(int(seed))
    lower = np.asarray(bounds_min_m, dtype=float)
    upper = np.asarray(bounds_max_m, dtype=float)
    span = upper - lower

    center = lower + np.array(
        [rng.uniform(0.40, 0.60) * span[0], rng.uniform(0.40, 0.60) * span[1], rng.uniform(0.22, 0.38) * span[2]],
        dtype=float,
    )
    radius = float(min(22.0, 8.0 + 2.0 * np.sqrt(uav_count)))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    initial_positions: dict[str, list[float]] = {}
    initial_velocities: dict[str, list[float]] = {}
    goals: dict[str, list[float]] = {}
    for index in range(uav_count):
        angle = phase + 2.0 * np.pi * index / uav_count
        jitter = rng.normal(0.0, 1.2, size=3)
        jitter[2] *= 0.35
        position = center + np.array(
            [radius * np.cos(angle), radius * np.sin(angle), 0.0], dtype=float
        ) + jitter
        position[2] = np.clip(center[2] + rng.uniform(-2.5, 2.5), lower[2] + 4.0, upper[2] - 4.0)
        position = np.clip(position, lower + 4.0, upper - 4.0)
        goal_offset = np.array(
            [rng.uniform(-36.0, 36.0), rng.uniform(-36.0, 36.0), rng.uniform(-5.0, 8.0)],
            dtype=float,
        )
        goal = np.clip(position + goal_offset, lower + 4.0, upper - 4.0)
        initial_positions[f"uav_{index:02d}"] = position.tolist()
        initial_velocities[f"uav_{index:02d}"] = np.zeros(3, dtype=float).tolist()
        goals[f"uav_{index:02d}"] = goal.tolist()

    positions = np.asarray(list(initial_positions.values()), dtype=float)
    goal_points = np.asarray(list(goals.values()), dtype=float)
    obstacle_specs: list[dict[str, Any]] = []
    for obstacle_index in range(2):
        for _attempt in range(100):
            size = np.array(
                [rng.uniform(18.0, 30.0), rng.uniform(18.0, 34.0), rng.uniform(14.0, 30.0)],
                dtype=float,
            )
            obstacle_center = np.array(
                [
                    rng.uniform(lower[0] + size[0] / 2.0 + 8.0, upper[0] - size[0] / 2.0 - 8.0),
                    rng.uniform(lower[1] + size[1] / 2.0 + 8.0, upper[1] - size[1] / 2.0 - 8.0),
                    rng.uniform(lower[2] + size[2] / 2.0, upper[2] - size[2] / 2.0),
                ],
                dtype=float,
            )
            if all(_clear_of_box(point, obstacle_center, size, 8.0) for point in positions) and all(
                _clear_of_box(point, obstacle_center, size, 5.0) for point in goal_points
            ):
                break
        obstacle_specs.append(
            {
                "obstacle_id": f"building_{obstacle_index}",
                "obstacle_type": "building",
                "geometry_type": "box",
                "center_m": obstacle_center.tolist(),
                "size_m": size.tolist(),
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
            }
        )

    for _attempt in range(100):
        radius_m = float(rng.uniform(2.0, 4.0))
        dynamic_center = np.array(
            [
                rng.uniform(lower[0] + radius_m + 8.0, upper[0] - radius_m - 8.0),
                rng.uniform(lower[1] + radius_m + 8.0, upper[1] - radius_m - 8.0),
                rng.uniform(lower[2] + radius_m, min(upper[2] - radius_m, 12.0)),
            ],
            dtype=float,
        )
        if all(np.linalg.norm(point - dynamic_center) > radius_m + 10.0 for point in positions):
            break
    dynamic_velocity = rng.uniform(-2.5, 2.5, size=3)
    dynamic_velocity[2] = 0.0
    obstacle_specs.append(
        {
            "obstacle_id": "dynamic_vehicle_0",
            "obstacle_type": "dynamic_vehicle",
            "geometry_type": "sphere",
            "center_m": dynamic_center.tolist(),
            "radius_m": radius_m,
            "velocity_mps": dynamic_velocity.tolist(),
            "active": 1,
        }
    )
    return {
        "scenario_version": "seeded-python-scene-v1",
        "seed": int(seed),
        "uav_count": int(uav_count),
        "bounds_min_m": lower.tolist(),
        "bounds_max_m": upper.tolist(),
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
        "goals": goals,
        "obstacle_specs": obstacle_specs,
        "generation_rule": "common_rule_seeded_cluster_goals_two_boxes_one_dynamic_sphere",
    }


def normalize_communication_occluder_semantics(
    scenario: dict[str, Any] | Mapping[str, Any],
    *,
    physical_collision_enabled: bool,
) -> dict[str, Any]:
    """Set the semantic role of an existing LOS screen without changing geometry.

    Old scenario files do not have the additive flag and therefore default to
    physical geometry in the environment.  A versioned pure-communication
    experiment can use this normalizer when it replays such a file, while
    preserving the original file on disk.
    """

    result = json.loads(json.dumps(dict(scenario)))
    specs = result.get("obstacle_specs", [])
    found = False
    if isinstance(specs, list):
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("obstacle_id", "")) != "screening_moving_nlos_screen":
                continue
            spec["physical_collision_enabled"] = bool(physical_collision_enabled)
            spec["semantic_role"] = "communication_occluder"
            spec["affects_los"] = True
            found = True
    if found:
        screening = result.setdefault("screening_geometry", {})
        if isinstance(screening, dict):
            screening.update(
                {
                    "semantic_role": "communication_occluder",
                    "physical_collision_enabled": bool(physical_collision_enabled),
                    "affects_los": True,
                }
            )
    return result


def add_moving_los_screen(
    scenario: dict[str, Any] | Mapping[str, Any],
    *,
    physical_collision_enabled: bool = True,
) -> dict[str, Any]:
    """Add moving geometry for LOS/NLOS screening.

    The historical default keeps the screen physical.  New mainline
    communication-occluder protocols pass ``False``: the geometry remains a
    LOS/NLOS blocker but is excluded from collision, safety projection, and
    physical-obstacle observations.

    The screen is constructed from the initial positions of UAV 0 and UAV 1,
    then translated perpendicular to their horizontal separation.  It changes
    visibility through the 3-D obstacle intersection test; it never modifies
    packet loss directly as a function of time.
    """

    result = json.loads(json.dumps(dict(scenario)))
    positions = result.get("initial_positions", {})
    if "uav_00" not in positions or "uav_01" not in positions:
        return result
    first = np.asarray(positions["uav_00"], dtype=float)
    second = np.asarray(positions["uav_01"], dtype=float)
    midpoint = 0.5 * (first + second)
    horizontal = second[:2] - first[:2]
    if abs(horizontal[0]) >= abs(horizontal[1]):
        size = np.array([3.0, 10.0, 54.0], dtype=float)
        velocity = np.array([0.0, 1.8, 0.0], dtype=float)
    else:
        size = np.array([10.0, 3.0, 54.0], dtype=float)
        velocity = np.array([1.8, 0.0, 0.0], dtype=float)
    center = np.array([midpoint[0], midpoint[1], 31.0], dtype=float)
    result.setdefault("obstacle_specs", []).append(
        {
            "obstacle_id": "screening_moving_nlos_screen",
            "obstacle_type": "screening_dynamic_blocker",
            "geometry_type": "box",
            "center_m": center.tolist(),
            "size_m": size.tolist(),
            "velocity_mps": velocity.tolist(),
            "active": 1,
            "physical_collision_enabled": bool(physical_collision_enabled),
            "semantic_role": "communication_occluder",
            "affects_los": True,
        }
    )
    result["screening_geometry"] = {
        "type": "moving_3d_los_screen",
        "construction": "pair_00_01_midpoint_then_perpendicular_translation",
        "packet_loss_modified_directly": False,
        "semantic_role": "communication_occluder",
        "physical_collision_enabled": bool(physical_collision_enabled),
        "affects_los": True,
    }
    return result


def generate_safety_risk_scenario(
    seed: int,
    uav_count: int = 3,
    *,
    bounds_min_m: tuple[float, float, float] = (0.0, 0.0, 2.0),
    bounds_max_m: tuple[float, float, float] = (200.0, 200.0, 60.0),
) -> dict[str, Any]:
    """Generate a deterministic, non-forced collision-risk scene.

    The first three UAVs fly on separated horizontal corridors at the same
    altitude.  A dynamic spherical obstacle crosses only the first corridor
    after the episode starts.  Its longitudinal offset is seeded, so a fixed
    collection of scenarios contains both near-miss and collision cases.
    This geometry is intentionally independent of MAPPO and does not expose
    truth to the controller.
    """

    if uav_count != 3:
        raise ValueError("the safety-risk screening is restricted to three UAVs")
    lower = np.asarray(bounds_min_m, dtype=float).reshape(3)
    upper = np.asarray(bounds_max_m, dtype=float).reshape(3)
    if np.any(lower >= upper):
        raise ValueError("bounds_min_m must be strictly below bounds_max_m")
    altitude = float(np.clip(20.0, lower[2] + 4.0, upper[2] - 4.0))
    rng = np.random.default_rng(int(seed))

    initial_positions = {
        "uav_00": [45.0, 85.0, altitude],
        "uav_01": [45.0, 135.0, altitude],
        "uav_02": [155.0, 35.0, altitude],
    }
    goals = {
        "uav_00": [155.0, 85.0, altitude],
        "uav_01": [155.0, 135.0, altitude],
        "uav_02": [45.0, 35.0, altitude],
    }
    initial_velocities = {uav_id: [0.0, 0.0, 0.0] for uav_id in initial_positions}

    # The weak controller reaches x ~= 60--65 m while the obstacle crosses
    # y=85 m.  A broad seeded x offset creates a mixture of near misses and
    # genuine pre-projection risks rather than forcing every episode to hit.
    obstacle_x = float(rng.uniform(38.0, 86.0))
    obstacle_y = float(rng.uniform(67.0, 71.0))
    obstacle_speed = float(rng.uniform(2.35, 2.75))
    obstacle_radius = float(rng.uniform(2.4, 2.8))
    obstacle_specs: list[dict[str, Any]] = [
        {
            "obstacle_id": "safety_screening_building_0",
            "obstacle_type": "building",
            "geometry_type": "box",
            "center_m": [170.0, 180.0, 10.0],
            "size_m": [20.0, 20.0, 12.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
        },
        {
            "obstacle_id": "safety_screening_building_1",
            "obstacle_type": "building",
            "geometry_type": "box",
            "center_m": [175.0, 25.0, 10.0],
            "size_m": [20.0, 20.0, 12.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
        },
        {
            "obstacle_id": "safety_screening_crossing_vehicle",
            "obstacle_type": "dynamic_vehicle",
            "geometry_type": "sphere",
            "center_m": [obstacle_x, obstacle_y, altitude],
            "radius_m": obstacle_radius,
            "velocity_mps": [0.0, obstacle_speed, 0.0],
            "active": 1,
        },
    ]
    return {
        "scenario_version": "safety-risk-screening-v1",
        "seed": int(seed),
        "uav_count": 3,
        "bounds_min_m": lower.tolist(),
        "bounds_max_m": upper.tolist(),
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
        "goals": goals,
        "obstacle_specs": obstacle_specs,
        "generation_rule": (
            "separated_same_altitude_corridors_with_seeded_dynamic_sphere_crossing_uav_00"
        ),
        "safety_risk_design": {
            "flight_altitude_m": altitude,
            "dynamic_obstacle_altitude_overlap": True,
            "initial_state_is_safe_by_construction": True,
            "collision_is_not_forced": True,
            "controller": "independent_fixed_goal_tracking_without_neighbor_measurements",
            "obstacle_crossing_axis": "y",
            "obstacle_crossing_speed_mps": obstacle_speed,
            "obstacle_radius_m": obstacle_radius,
            "obstacle_x_m": obstacle_x,
        },
    }


__all__ = [
    "add_moving_los_screen",
    "normalize_communication_occluder_semantics",
    "generate_safety_risk_scenario",
    "generate_scenario",
]
