import numpy as np
import pytest

from src.experiments.formal import FormalExperimentConfig, _make_environment
from src.experiments.single_uav_hierarchical_waypoint import (
    WAYPOINT_DISTANCE_BANDS,
    _make_waypoint_scenarios,
)


def _config() -> FormalExperimentConfig:
    return FormalExperimentConfig(
        output_dir="data/synthetic/test_hierarchical_waypoint",
        uav_count=1,
        max_steps=20,
        single_uav_learning_mode="loc_goal_v2",
        single_uav_speed_control_mode="braking_distance",
        single_uav_stable_hold_steps=5,
    )


def test_waypoint_design_covers_distance_bands_and_smoke_directions():
    cases = _make_waypoint_scenarios(_config(), smoke=True)
    assert len(cases) == 12
    assert {case["waypoint_distance_band"] for case in cases} == {
        band[0] for band in WAYPOINT_DISTANCE_BANDS
    }
    assert {case["direction"] for case in cases} == {"front", "back", "left", "right"}
    assert len({case["geometry_seed"] for case in cases}) == len(cases)
    assert len({case["noise_seed"] for case in cases}) == len(cases)
    assert all(case["scenario"]["hierarchical_waypoint_diagnostic"]["curriculum_used"] is False for case in cases)


def test_temporary_goal_rebuilds_actor_goal_delta_and_restores_mission_goal():
    config = _config()
    environment = _make_environment(config, seed=702026, metric_scales=config.metric_scales)
    observations, _ = environment.reset(seed=702026, scenario_config={})
    mission_goal = environment.goals["uav_00"].copy()
    estimate = np.asarray(observations["uav_00"]["estimate"][:3], dtype=float)
    temporary_goal = estimate + np.asarray((3.0, 1.0, 0.5), dtype=float)

    installed = environment.set_temporary_goal("uav_00", temporary_goal)
    updated = environment._observations()["uav_00"]
    own_index = list(updated["node_ids"]).index("uav_00")
    goal_delta = np.asarray(updated["node_features"], dtype=float)[own_index, 30:33]
    assert installed == pytest.approx(temporary_goal)
    assert goal_delta == pytest.approx(temporary_goal - np.asarray(updated["estimate"][:3], dtype=float))
    assert environment.temporary_goals["uav_00"] == pytest.approx(temporary_goal)

    restored = environment.clear_temporary_goal("uav_00")
    assert restored == pytest.approx(mission_goal)
    assert environment.temporary_goals == {}
    restored_observation = environment._observations()["uav_00"]
    restored_index = list(restored_observation["node_ids"]).index("uav_00")
    assert np.asarray(restored_observation["node_features"], dtype=float)[restored_index, 30:33] == pytest.approx(
        mission_goal - np.asarray(restored_observation["estimate"][:3], dtype=float)
    )


def test_temporary_goal_rejects_invalid_inputs():
    config = _config()
    environment = _make_environment(config, seed=702026, metric_scales=config.metric_scales)
    environment.reset(seed=702026, scenario_config={})
    with pytest.raises(ValueError, match="finite 3-D"):
        environment.set_temporary_goal("uav_00", (1.0, 2.0))
    with pytest.raises(KeyError, match="unknown UAV"):
        environment.set_temporary_goal("uav_99", (1.0, 2.0, 3.0))
