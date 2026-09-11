from __future__ import annotations

import numpy as np
import pytest

from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from src.simulation.motion_model import MotionConfig


def _statistics() -> dict[str, float]:
    return {
        "u_center": 0.0,
        "u_scale": 1.0,
        "c_center": 0.0,
        "c_scale": 1.0,
        "delta_u_scale": 0.5,
        "delta_goal_scale": 0.5,
    }


def _actions(environment: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
    return {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {
                target: 1.0 / max(1, len(environment.uav_ids) - 1)
                for target in environment.uav_ids
                if target != uav_id
            },
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in environment.uav_ids
    }


def _boundary_scenario(uav_count: int = 1) -> dict[str, object]:
    positions = {
        "uav_00": [9.9, 5.0, 5.0],
        "uav_01": [5.0, 5.0, 5.0],
    }
    velocities = {
        "uav_00": [3.0, 0.0, 0.0],
        "uav_01": [0.0, 0.0, 0.0],
    }
    goals = {
        "uav_00": [1.0, 1.0, 5.0],
        "uav_01": [1.0, 9.0, 5.0],
    }
    return {
        "initial_positions": {
            uav_id: positions[uav_id] for uav_id in list(positions)[:uav_count]
        },
        "initial_velocities": {
            uav_id: velocities[uav_id] for uav_id in list(velocities)[:uav_count]
        },
        "goals": {uav_id: goals[uav_id] for uav_id in list(goals)[:uav_count]},
        "obstacles": [],
    }


def _environment(
    *, uav_count: int = 1, boundary_penalty_weight: float = 0.5
) -> MultiUAVJointEnv:
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=uav_count,
            max_steps=3,
            seed=2026,
            gnss_mode="denied",
            motion=MotionConfig(
                bounds_min_m=(0.0, 0.0, 2.0),
                bounds_max_m=(10.0, 10.0, 10.0),
            ),
            reward=RewardConfig(
                boundary_penalty_weight=boundary_penalty_weight,
                learning_reward_clip=10.0,
            ),
        )
    )


def test_reward_config_boundary_weight_is_finite_and_non_negative() -> None:
    assert RewardConfig().boundary_penalty_weight == 0.0
    with pytest.raises(ValueError, match="boundary_penalty_weight"):
        RewardConfig(boundary_penalty_weight=-0.1)
    with pytest.raises(ValueError, match="boundary_penalty_weight"):
        RewardConfig(boundary_penalty_weight=float("nan"))


def test_no_boundary_has_zero_local_boundary_reward() -> None:
    environment = _environment()
    environment.reset(
        seed=2026,
        scenario_config={
            "initial_positions": {"uav_00": [5.0, 5.0, 5.0]},
            "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
            "goals": {"uav_00": [1.0, 1.0, 5.0]},
            "obstacles": [],
        },
    )
    _observations, _rewards, _terminated, _truncated, info = environment.step(
        _actions(environment)
    )

    assert info["boundary_projected_by_uav"]["uav_00"] is False
    assert info["boundary_reward_by_uav"]["uav_00"] == 0.0
    assert info["boundary_projection_count"] == 0
    assert info["boundary_penalty_sum"] == 0.0


def test_single_axis_boundary_projection_gets_one_local_penalty_and_stays_in_bounds() -> None:
    environment = _environment()
    environment.reset(seed=2026, scenario_config=_boundary_scenario())
    _observations, _rewards, terminated, _truncated, info = environment.step(
        _actions(environment)
    )

    execution = info["action_execution"]["uav_00"]
    assert execution["boundary_projected"] is True
    assert info["boundary_reward_by_uav"]["uav_00"] == -0.5
    assert info["boundary_projection_count"] == 1
    assert info["boundary_penalty_sum"] == -0.5
    assert np.all(environment.states["uav_00"].position_m <= 10.0)
    assert np.all(environment.states["uav_00"].position_m >= 0.0)
    assert terminated is False


def test_multiple_axes_in_one_step_still_count_as_one_boundary_event() -> None:
    environment = _environment()
    environment.reset(
        seed=2026,
        scenario_config={
            "initial_positions": {"uav_00": [9.9, 9.9, 9.9]},
            "initial_velocities": {"uav_00": [3.0, 3.0, 3.0]},
            "goals": {"uav_00": [1.0, 1.0, 5.0]},
            "obstacles": [],
        },
    )
    _observations, _rewards, _terminated, _truncated, info = environment.step(
        _actions(environment)
    )

    assert info["action_execution"]["uav_00"]["boundary_projected"] is True
    assert info["boundary_projection_count"] == 1
    assert info["boundary_reward_by_uav"]["uav_00"] == -0.5
    assert info["boundary_penalty_sum"] == -0.5


def test_boundary_is_not_collision_or_safety_projection_and_episode_continues() -> None:
    environment = _environment()
    environment.reset(seed=2026, scenario_config=_boundary_scenario())
    _observations, _rewards, terminated, truncated, info = environment.step(
        _actions(environment)
    )

    execution = info["action_execution"]["uav_00"]
    assert execution["boundary_projected"] is True
    assert execution["safety_projected"] is False
    assert info["collision"] == 0
    assert info["safety_projection_count"] == 0
    assert terminated is False
    assert truncated is False


def test_boundary_penalty_reaches_learning_reward_before_existing_clip() -> None:
    def evaluate(boundary_projected: bool) -> tuple[float, dict[str, object]]:
        environment = _environment()
        environment.reset(seed=2026, scenario_config=_boundary_scenario())
        actions = _actions(environment)
        execution = {
            "uav_00": {
                "delta_v_applied": [0.0, 0.0, 0.0],
                "boundary_projected": boundary_projected,
                "safety_projected": False,
            }
        }
        _reporting, info = environment._rewards(0, actions, execution)
        return float(info["learning_rewards"]["uav_00"]), info

    no_boundary_reward, no_boundary_info = evaluate(False)
    boundary_reward, boundary_info = evaluate(True)

    assert no_boundary_info["boundary_reward"] == 0.0
    assert boundary_info["boundary_reward"] == -0.5
    assert boundary_info["learning_reward_components"]["base_learning_reward"] == pytest.approx(
        no_boundary_info["learning_reward_components"]["base_learning_reward"]
    )
    assert boundary_info["learning_reward_components"]["learning_reward_preclip"] == pytest.approx(
        no_boundary_info["learning_reward_components"]["learning_reward_preclip"] - 0.5
    )
    assert boundary_reward == pytest.approx(no_boundary_reward - 0.5)


def test_boundary_penalty_is_independent_per_uav() -> None:
    environment = _environment(uav_count=2)
    environment.reset(seed=2026, scenario_config=_boundary_scenario(uav_count=2))
    _observations, _rewards, _terminated, _truncated, info = environment.step(
        _actions(environment)
    )

    assert info["boundary_projected_by_uav"] == {
        "uav_00": True,
        "uav_01": False,
    }
    assert info["boundary_reward_by_uav"] == {
        "uav_00": -0.5,
        "uav_01": 0.0,
    }
    assert info["learning_rewards"]["uav_00"] == pytest.approx(
        info["learning_rewards"]["uav_01"] - 0.5
    )
    assert info["boundary_projection_count"] == 1


def test_zero_boundary_weight_preserves_learning_and_reporting_rewards() -> None:
    baseline = _environment(boundary_penalty_weight=0.0)
    explicit_zero = _environment(boundary_penalty_weight=0.0)
    scenario = _boundary_scenario()
    baseline.reset(seed=2026, scenario_config=scenario)
    explicit_zero.reset(seed=2026, scenario_config=scenario)
    actions_baseline = _actions(baseline)
    actions_explicit_zero = _actions(explicit_zero)
    baseline_report, baseline_info = baseline._rewards(
        0,
        actions_baseline,
        {"uav_00": {"delta_v_applied": [0.0, 0.0, 0.0], "boundary_projected": True}},
    )
    explicit_report, explicit_info = explicit_zero._rewards(
        0,
        actions_explicit_zero,
        {"uav_00": {"delta_v_applied": [0.0, 0.0, 0.0], "boundary_projected": True}},
    )

    assert explicit_report == baseline_report
    assert explicit_info["learning_rewards"] == baseline_info["learning_rewards"]
    assert explicit_info["boundary_reward_by_uav"]["uav_00"] == 0.0

