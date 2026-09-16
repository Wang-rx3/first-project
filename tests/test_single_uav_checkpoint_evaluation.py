from types import SimpleNamespace

import numpy as np

from src.experiments.single_uav_checkpoint_evaluation import (
    _diagnostic_scenario,
    _distance_grid,
    _group_summary,
)


def _braking_config():
    return SimpleNamespace(
        single_uav_goal_threshold_m=1.0,
        single_uav_braking_margin_m=2.0,
        single_uav_goal_speed_threshold_mps=0.5,
        single_uav_braking_acceleration_mps2=2.0,
    )


def test_distance_grid_has_expected_count_and_endpoints():
    values = _distance_grid(1.5, 3.0, 20)

    assert len(values) == 20
    assert np.isclose(values[0], 1.5)
    assert np.isclose(values[-1], 3.0)
    assert all(left < right for left, right in zip(values, values[1:]))


def test_controlled_diagnostic_start_is_nonzero_and_on_goal_ray():
    scenario = {
        "initial_positions": {"uav_00": [0.0, 0.0, 10.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [0.0, 0.0, 20.0]},
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
    }
    group = {
        "name": "terminal_3_5m",
        "label": "3-5 m",
        "low_m": 3.0,
        "high_m": 5.0,
    }

    controlled, actual_distance, initial_speed = _diagnostic_scenario(
        scenario, _braking_config(), group, 4.0
    )

    position = np.asarray(controlled["initial_positions"]["uav_00"], dtype=float)
    velocity = np.asarray(controlled["initial_velocities"]["uav_00"], dtype=float)
    goal = np.asarray(controlled["goals"]["uav_00"], dtype=float)
    assert np.isclose(np.linalg.norm(goal - position), 4.0)
    assert np.isclose(actual_distance, 4.0)
    assert initial_speed > 0.0
    assert np.dot(goal - position, velocity) > 0.0
    assert controlled["evaluation_diagnostic"]["evaluation_only"] is True
    assert controlled["evaluation_diagnostic"]["curriculum_used"] is False


def test_group_summary_reports_hold_threshold_counts_and_rates():
    group = {
        "name": "terminal_1p5_3m",
        "label": "1.5-3 m",
        "low_m": 1.5,
        "high_m": 3.0,
    }
    rows = [
        {
            "group": "terminal_1p5_3m",
            "actual_start_distance_m": 1.5,
            "minimum_goal_distance_m": 0.7,
            "mean_speed_mps": 1.0,
            "entered_8m": 1,
            "entered_1m": 1,
            "stable_goal": 0,
            "maximum_consecutive_goal_hold_steps": 3,
            "hold_steps_ge_1": 1,
            "hold_steps_ge_3": 1,
            "hold_steps_ge_5": 0,
        },
        {
            "group": "terminal_1p5_3m",
            "actual_start_distance_m": 3.0,
            "minimum_goal_distance_m": 1.2,
            "mean_speed_mps": 2.0,
            "entered_8m": 1,
            "entered_1m": 0,
            "stable_goal": 0,
            "maximum_consecutive_goal_hold_steps": 0,
            "hold_steps_ge_1": 0,
            "hold_steps_ge_3": 0,
            "hold_steps_ge_5": 0,
        },
    ]

    summary = _group_summary(group, rows)

    assert summary["episodes"] == 2
    assert summary["entered_8m_episodes"] == 2
    assert summary["entered_1m_episodes"] == 1
    assert summary["episodes_with_hold_steps_ge_1"] == 1
    assert summary["episodes_with_hold_steps_ge_3_rate"] == 0.5
    assert summary["episodes_with_hold_steps_ge_5"] == 0
