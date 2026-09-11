from __future__ import annotations

import numpy as np

from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig


def _statistics() -> dict[str, float]:
    return {
        "u_center": 0.0,
        "u_scale": 1.0,
        "c_center": 0.0,
        "c_scale": 1.0,
        "delta_u_scale": 0.5,
        "delta_goal_scale": 0.5,
    }


def _action(delta_v: list[float]) -> dict[str, dict[str, object]]:
    return {
        "uav_00": {
            "measurement_uav_id": "none",
            "bandwidth": {},
            "delta_v": list(delta_v),
        }
    }


def _environment(*, enabled: bool) -> MultiUAVJointEnv:
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=20,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                single_uav_speed_control_mode="braking_distance",
                single_uav_goal_threshold_m=1.0,
                single_uav_goal_speed_threshold_mps=0.5,
                single_uav_braking_acceleration_mps2=2.0,
                single_uav_braking_margin_m=2.0,
                single_uav_target_speed_far_mps=6.0,
                single_uav_braking_safety_projection_enabled=enabled,
                potential_statistics=_statistics(),
            ),
        )
    )


def _reset_near_goal(environment: MultiUAVJointEnv) -> None:
    goal = np.asarray([10.0, 10.0, 10.0])
    environment.reset(
        seed=42,
        scenario_config={
            "initial_positions": {"uav_00": [6.0, 10.0, 10.0]},
            "initial_velocities": {"uav_00": [4.0, 0.0, 0.0]},
            "goals": {"uav_00": goal.tolist()},
            "obstacles": [],
        },
    )


def test_braking_safety_projection_overrides_unsafe_command_and_logs_reason() -> None:
    environment = _environment(enabled=True)
    _reset_near_goal(environment)
    requested = _action([0.6, 0.0, 0.0])

    _observations, _rewards, _terminated, _truncated, info = environment.step(requested)

    execution = info["action_execution"]["uav_00"]
    effective = info["actions"]["uav_00"]
    assert requested["uav_00"]["delta_v"] == [0.6, 0.0, 0.0]
    assert execution["braking_safety_projection_enabled"] is True
    assert execution["braking_safety_projected"] is True
    assert execution["braking_safety_projection_reason"] == "stopping_distance_overspeed"
    assert float(effective["delta_v"][0]) < 0.0
    assert float(execution["applied_longitudinal_delta_v_mps"]) < 0.0
    assert float(execution["post_speed_mps"]) < float(execution["pre_speed_mps"])
    assert info["braking_safety_projection_count"] == 1


def test_braking_safety_projection_is_opt_in_and_disabled_preserves_command() -> None:
    environment = _environment(enabled=False)
    _reset_near_goal(environment)
    requested = _action([0.6, 0.0, 0.0])

    _observations, _rewards, _terminated, _truncated, info = environment.step(requested)

    execution = info["action_execution"]["uav_00"]
    effective = info["actions"]["uav_00"]
    assert execution["braking_safety_projection_enabled"] is False
    assert execution["braking_safety_projected"] is False
    assert info["braking_safety_projection_count"] == 0
    assert np.allclose(effective["delta_v"], [0.6, 0.0, 0.0])


def test_braking_safety_projection_holds_a_uav_inside_goal_radius() -> None:
    environment = _environment(enabled=True)
    goal = np.asarray([10.0, 10.0, 10.0])
    environment.reset(
        seed=42,
        scenario_config={
            "initial_positions": {"uav_00": [10.5, 10.0, 10.0]},
            "initial_velocities": {"uav_00": [0.4, 0.0, 0.0]},
            "goals": {"uav_00": goal.tolist()},
            "obstacles": [],
        },
    )

    _observations, _rewards, _terminated, _truncated, info = environment.step(
        _action([0.6, 0.0, 0.0])
    )

    execution = info["action_execution"]["uav_00"]
    assert execution["braking_safety_projected"] is True
    assert execution["braking_safety_projection_reason"] == "goal_radius_hold"
    assert float(execution["post_speed_mps"]) < float(execution["pre_speed_mps"])
    assert np.allclose(info["actions"]["uav_00"]["delta_v"], [-0.4, 0.0, 0.0])
