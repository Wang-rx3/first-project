from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.formal import FormalExperimentConfig
from src.experiments.single_uav_static_obstacle_avoidance import (
    StaticObstacleAvoidanceRewardConfig,
    _static_pressure_scenario,
    static_obstacle_reward,
    training_schedule_counts,
)
from src.simulation.obstacle_model import BoxObstacle, segment_obstacle_clearance


def test_stage1_schedule_is_70_30_for_thirty_episodes() -> None:
    assert training_schedule_counts(30) == {
        "normal_far_distance": 21,
        "static_pressure": 9,
    }


def test_static_reward_is_zero_by_default_and_uses_pre_projection_clearance() -> None:
    diagnostics = {
        "pre_candidate_static_swept_minimum_clearance_m": 0.0,
    }
    assert static_obstacle_reward(
        diagnostics, StaticObstacleAvoidanceRewardConfig()
    )["reward"] == 0.0

    active = StaticObstacleAvoidanceRewardConfig(
        clearance_weight=0.5,
        risk_weight=1.0,
        warning_distance_m=8.0,
        safe_distance_m=2.0,
    )
    result = static_obstacle_reward(diagnostics, active)
    assert result["active"] is True
    assert result["reward"] < 0.0


def test_static_pressure_direct_path_has_swept_conflict() -> None:
    config = FormalExperimentConfig(output_dir=Path("."), uav_count=1)
    scenario = _static_pressure_scenario(
        config, seed=2026, split="test", episode_index=0
    )
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)
    goal = np.asarray(scenario["goals"]["uav_00"], dtype=float)
    spec = scenario["obstacle_specs"][0]
    obstacle = BoxObstacle(
        obstacle_id=str(spec["obstacle_id"]),
        center_m=np.asarray(spec["center_m"], dtype=float),
        size_m=np.asarray(spec["size_m"], dtype=float),
        velocity_mps=np.asarray(spec["velocity_mps"], dtype=float),
        obstacle_type=str(spec["obstacle_type"]),
        active=True,
        physical_collision_enabled=True,
    )
    assert segment_obstacle_clearance(start, goal, obstacle) == 0.0
