import numpy as np

from src.experiments.single_uav_collision_attribution_audit import (
    SCENE_NAMES,
    _controlled_pressure_scenario,
)


def test_pressure_scenarios_are_single_uav_and_deterministic():
    assert len(SCENE_NAMES) == 4
    for scene in SCENE_NAMES[1:]:
        first = _controlled_pressure_scenario(scene, episode_index=2, seed=12026)
        second = _controlled_pressure_scenario(scene, episode_index=2, seed=12026)
        assert first == second
        assert first["uav_count"] == 1
        assert set(first["initial_positions"]) == {"uav_00"}
        assert set(first["goals"]) == {"uav_00"}
        assert np.isfinite(np.asarray(first["initial_positions"]["uav_00"], dtype=float)).all()


def test_static_pressure_scene_contains_physical_wall_on_direct_route():
    scenario = _controlled_pressure_scenario(
        "static_obstacle_near", episode_index=0, seed=2026
    )
    obstacle = scenario["obstacle_specs"][0]
    assert obstacle["physical_collision_enabled"] is True
    assert obstacle["obstacle_id"] == "audit_static_wall"
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)
    goal = np.asarray(scenario["goals"]["uav_00"], dtype=float)
    center = np.asarray(obstacle["center_m"], dtype=float)
    assert start[0] < center[0] < goal[0]


def test_boundary_pressure_scene_uses_in_bounds_state_and_nonzero_velocity():
    scenario = _controlled_pressure_scenario(
        "boundary_pressure", episode_index=0, seed=2026
    )
    position = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)
    velocity = np.asarray(scenario["initial_velocities"]["uav_00"], dtype=float)
    assert np.all(position >= np.asarray(scenario["bounds_min_m"], dtype=float))
    assert np.all(position <= np.asarray(scenario["bounds_max_m"], dtype=float))
    assert velocity[0] < 0.0
    assert scenario["obstacle_specs"] == []
