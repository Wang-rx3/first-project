"""Contracts for action-linked, delayed EKF update feedback."""

from __future__ import annotations

import numpy as np

from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig


def _actions(env: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
    result = {}
    for uav_id in env.uav_ids:
        result[uav_id] = {
            "measurement_uav_id": "uav_01" if uav_id == "uav_00" else "none",
            "bandwidth": {},
            "delta_v": [0.0, 0.0, 0.0],
        }
    return result


def _execution(env: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
    return {
        uav_id: {"delta_v_applied": [0.0, 0.0, 0.0]}
        for uav_id in env.uav_ids
    }


def test_accepted_update_feedback_is_delayed_and_does_not_change_report_reward() -> None:
    env = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=3,
            max_steps=10,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_reward_mode="report_plus_feedback",
                learning_feedback_mode="accepted_update_progress_v1",
                accepted_update_feedback_coef=0.1,
                accepted_update_feedback_delay_steps=1,
                learning_potential_coef=0.0,
            ),
        )
    )
    env.reset(seed=2026)
    actions = _actions(env)
    execution = _execution(env)
    env.relative_range_update_events = [
        {
            "target_uav_id": "uav_00",
            "source_uav_id": "uav_01",
            "accepted": True,
            "position_covariance_trace_before_m2": 3.0,
            "position_covariance_trace_after_linear_m2": 2.0,
            "position_covariance_trace_after_final_m2": 3.0,
            "local_direction_variance_m2": 1.0,
            "effective_measurement_variance_m2": 3.0,
            "nis": 0.0,
        }
    ]

    report_first, info_first = env._rewards(0, actions, execution)
    first_components = info_first["learning_reward_components"]
    assert float(first_components["accepted_update_feedback_reward"]) == 0.0
    assert len(first_components["accepted_update_feedback_records"]) == 1
    assert 0.0 < float(
        first_components["accepted_update_feedback_records"][0][
            "directional_information_fraction"
        ]
    ) < 1.0
    first_report = float(next(iter(report_first.values())))
    first_learning = float(next(iter(info_first["learning_rewards"].values())))
    assert np.isclose(first_learning, first_report)

    # Move to the next decision window without changing the physical state;
    # the pending record is now released.  No true position or RMSE is used.
    env.timestep = 1
    env.relative_range_update_events = []
    report_second, info_second = env._rewards(0, actions, execution)
    second_components = info_second["learning_reward_components"]
    second_report = float(next(iter(report_second.values())))
    second_learning = float(next(iter(info_second["learning_rewards"].values())))
    assert np.isclose(second_report, first_report)
    assert float(second_components["accepted_update_feedback_reward"]) > 0.0
    assert len(second_components["released_update_feedback_records"]) == 1
    assert second_learning > second_report
