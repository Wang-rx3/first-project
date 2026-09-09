"""Contract tests for communication-only LOS screening geometry."""

from __future__ import annotations

import numpy as np

from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.obstacle_model import (
    BoxObstacle,
    collision_event,
    joint_motion_collision_diagnostics,
    line_of_sight,
)
from src.simulation.scenario import add_moving_los_screen


def _screen(*, physical: bool) -> BoxObstacle:
    return BoxObstacle(
        "screening_moving_nlos_screen",
        np.asarray([5.0, 0.0, 0.0]),
        np.asarray([2.0, 2.0, 2.0]),
        np.zeros(3),
        "screening_dynamic_blocker",
        True,
        physical,
    )


def test_screen_remains_a_los_occluder_but_not_a_collision_obstacle() -> None:
    screen = _screen(physical=False)
    assert line_of_sight(np.zeros(3), np.asarray([10.0, 0.0, 0.0]), [screen]) is False
    collision, clearance = collision_event(
        {"uav_00": np.asarray([5.0, 0.0, 0.0])}, [screen], 2.0
    )
    assert collision == 0
    assert clearance == 2.0


def test_joint_safety_ignores_communication_only_geometry_and_reports_it() -> None:
    screen = _screen(physical=False)
    result = joint_motion_collision_diagnostics(
        {"uav_00": np.asarray([0.0, 0.0, 0.0])},
        {"uav_00": np.asarray([10.0, 0.0, 0.0])},
        [screen],
        [screen],
        dt_s=0.2,
        bounds_min=np.asarray([-20.0, -20.0, -20.0]),
        bounds_max=np.asarray([20.0, 20.0, 20.0]),
        safety_distance_m=2.0,
    )
    assert result["collision"] is False
    assert result["physical_contact"] is False
    assert result["obstacle_conflicts"] == []
    assert result["ignored_nonphysical_obstacle_ids"] == [
        "screening_moving_nlos_screen"
    ]


def test_generated_screen_semantics_are_versioned_without_changing_geometry() -> None:
    scenario = {
        "initial_positions": {
            "uav_00": [20.0, 20.0, 20.0],
            "uav_01": [80.0, 20.0, 20.0],
        },
        "obstacle_specs": [],
    }
    historical = add_moving_los_screen(scenario)
    communication_only = add_moving_los_screen(
        scenario, physical_collision_enabled=False
    )
    old = historical["obstacle_specs"][-1]
    new = communication_only["obstacle_specs"][-1]
    assert old["physical_collision_enabled"] is True
    assert new["physical_collision_enabled"] is False
    assert new["semantic_role"] == "communication_occluder"
    assert new["affects_los"] is True
    assert new["center_m"] == old["center_m"]
    assert new["size_m"] == old["size_m"]
    assert new["velocity_mps"] == old["velocity_mps"]


def test_actor_physical_obstacle_features_exclude_communication_screen() -> None:
    environment = MultiUAVJointEnv(SimulationConfig(uav_count=3, seed=42))
    scenario = {
        "initial_positions": {
            "uav_00": [20.0, 20.0, 20.0],
            "uav_01": [80.0, 20.0, 20.0],
            "uav_02": [20.0, 80.0, 20.0],
        },
        "obstacle_specs": [
            {
                "obstacle_id": "screening_moving_nlos_screen",
                "obstacle_type": "screening_dynamic_blocker",
                "geometry_type": "box",
                "center_m": [50.0, 20.0, 20.0],
                "size_m": [2.0, 2.0, 2.0],
                "velocity_mps": [0.0, 1.0, 0.0],
                "active": 1,
                "physical_collision_enabled": False,
            }
        ],
    }
    environment.reset(seed=42, scenario_config=scenario)
    features = environment._local_obstacle_features(np.asarray([20.0, 20.0, 20.0]))
    assert features.shape == (4,)
    assert features[1] == 0.0
    assert features[2] == 0.0
