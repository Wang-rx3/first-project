"""Acceptance tests for the localization-first single-UAV learning channel."""

from __future__ import annotations

import numpy as np

from src.simulation.environment import (
    MultiUAVJointEnv,
    RewardConfig,
    SimulationConfig,
    _distance_target_speed,
)


def _statistics() -> dict[str, float]:
    return {
        "u_center": 0.0,
        "u_scale": 1.0,
        "c_center": 0.0,
        "c_scale": 1.0,
        "delta_u_scale": 0.5,
        "delta_goal_scale": 0.5,
    }


def _actions(environment: MultiUAVJointEnv, *, delta_v: list[float] | None = None):
    return {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {},
            "delta_v": list(delta_v or [0.0, 0.0, 0.0]),
        }
        for uav_id in environment.uav_ids
    }


def _execution(environment: MultiUAVJointEnv, *, delta_v: list[float] | None = None):
    return {
        uav_id: {"delta_v_applied": list(delta_v or [0.0, 0.0, 0.0])}
        for uav_id in environment.uav_ids
    }


def _single_environment() -> MultiUAVJointEnv:
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v1",
                potential_statistics=_statistics(),
            ),
        )
    )


def test_single_uav_progress_signs_and_reporting_reward_separation() -> None:
    environment = _single_environment()
    environment.reset(seed=42)
    current_uncertainty = environment._position_logdet_uncertainty()
    current_goal_distance = environment.estimated_goal_distance()
    environment._previous_position_logdet = current_uncertainty + 1.0
    environment._previous_estimated_goal_distance = current_goal_distance + 1.0

    reporting, info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    components = info["learning_reward_components"]
    assert info["communication_applicable"] is False
    assert components["communication_reward_in_learning"] is False
    assert float(components["localization_progress_score"]) > 0.0
    assert float(components["goal_progress_score"]) > 0.0
    assert float(next(iter(reporting.values()))) < 0.0
    assert float(next(iter(info["learning_rewards"].values()))) > 0.0
    assert np.isfinite(float(next(iter(info["learning_rewards"].values()))))


def test_single_uav_navigation_v2_adds_goal_progress_and_stable_one_shot_bonus() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                potential_statistics=_statistics(),
                single_uav_goal_bonus=0.75,
                single_uav_goal_threshold_m=1.0,
            ),
        )
    )
    environment.reset(seed=42)
    environment._previous_position_logdet = environment._position_logdet_uncertainty()
    environment._previous_estimated_goal_distance = (
        environment.estimated_goal_distance() + 0.5
    )
    _reporting, info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    components = info["learning_reward_components"]
    assert info["single_uav_learning_active"] is True
    assert float(components["single_uav_goal_progress_reward"]) > 0.0
    assert np.isclose(
        float(components["single_uav_goal_progress_reward"]),
        float(components["goal_progress_delta"]),
    )
    assert float(components["single_uav_localization_cost"]) > 0.0
    assert float(components["single_uav_goal_bonus"]) == 0.0
    assert float(next(iter(info["learning_rewards"].values()))) > -1.0

    environment.reset(seed=42)
    environment.ekfs["uav_00"].state[:3] = environment.goals["uav_00"]
    environment.states["uav_00"].velocity_mps[:] = [1.0, 0.0, 0.0]
    environment._previous_estimated_goal_distance = 2.0
    _reporting, moving_goal_info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    moving_components = moving_goal_info["learning_reward_components"]
    assert moving_components["single_uav_goal_in_radius"] is True
    assert moving_components["single_uav_goal_speed_ok"] is False
    assert moving_components["single_uav_goal_stable"] is False
    assert moving_components["single_uav_goal_bonus_triggered"] is False

    environment.reset(seed=42)
    environment.ekfs["uav_00"].state[:3] = environment.goals["uav_00"]
    environment.states["uav_00"].velocity_mps[:] = 0.0
    environment._previous_estimated_goal_distance = 2.0
    _reporting, first_goal_info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    first_components = first_goal_info["learning_reward_components"]
    assert first_components["single_uav_goal_in_radius"] is True
    assert first_components["single_uav_goal_speed_ok"] is True
    assert first_components["single_uav_goal_stable"] is True
    assert first_components["single_uav_goal_bonus_triggered"] is True
    assert np.isclose(float(first_components["single_uav_goal_bonus"]), 0.75)

    _reporting, second_goal_info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    second_components = second_goal_info["learning_reward_components"]
    assert second_components["single_uav_goal_bonus_triggered"] is False
    assert np.isclose(float(second_components["single_uav_goal_bonus"]), 0.0)


def test_single_uav_navigation_v2_stable_goal_terminates_episode() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=5,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                potential_statistics=_statistics(),
            ),
        )
    )
    target = [170.0, 170.0, 10.0]
    environment.reset(
        seed=42,
        scenario_config={
            "initial_positions": {"uav_00": target},
            "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
            "goals": {"uav_00": target},
            "obstacles": [],
        },
    )

    _observations, _rewards, terminated, truncated, info = environment.step(
        _actions(environment)
    )

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "goal_reached"
    assert info["task_complete"] is True
    assert info["single_uav_goal_stable"] is True


def test_single_uav_distance_target_speed_penalizes_only_overspeed() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                single_uav_speed_control_mode="distance_target_speed",
                single_uav_target_speed_far_mps=6.0,
                single_uav_near_goal_radius_m=8.0,
                single_uav_near_goal_speed_penalty_weight=0.4,
                potential_statistics=_statistics(),
            ),
        )
    )
    environment.reset(seed=42)
    environment.ekfs["uav_00"].state[:3] = (
        environment.goals["uav_00"] + np.asarray([4.5, 0.0, 0.0])
    )
    environment.states["uav_00"].velocity_mps[:] = [4.5, 0.0, 0.0]
    current_distance = environment.estimated_goal_distance()
    environment._previous_estimated_goal_distance = current_distance + 1.0
    environment._previous_position_logdet = environment._position_logdet_uncertainty()

    _reporting, info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    components = info["learning_reward_components"]
    expected_target = _distance_target_speed(
        current_distance,
        goal_threshold_m=1.0,
        near_goal_radius_m=8.0,
        stable_speed_mps=0.5,
        far_target_speed_mps=6.0,
    )
    expected_excess = max(0.0, 4.5 - expected_target)
    assert components["single_uav_speed_control_mode"] == "distance_target_speed"
    assert np.isclose(
        float(components["single_uav_target_speed_mps"]), expected_target
    )
    assert np.isclose(
        float(components["single_uav_speed_excess_mps"]), expected_excess
    )
    assert np.isclose(
        float(components["single_uav_near_goal_speed_cost"]), 0.4 * expected_excess
    )

    environment.states["uav_00"].velocity_mps[:] = [0.5, 0.0, 0.0]
    _reporting, low_speed_info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    low_speed_components = low_speed_info["learning_reward_components"]
    assert np.isclose(float(low_speed_components["single_uav_speed_excess_mps"]), 0.0)
    assert np.isclose(
        float(low_speed_components["single_uav_near_goal_speed_cost"]), 0.0
    )


def test_single_uav_braking_distance_penalizes_overspeed_before_near_goal() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                single_uav_speed_control_mode="braking_distance",
                single_uav_target_speed_far_mps=6.0,
                single_uav_near_goal_radius_m=8.0,
                single_uav_near_goal_speed_penalty_weight=0.4,
                single_uav_braking_acceleration_mps2=2.0,
                single_uav_braking_margin_m=2.0,
                potential_statistics=_statistics(),
            ),
        )
    )
    observations, _info = environment.reset(seed=42)
    assert observations["uav_00"]["node_features"].shape[-1] == 39
    environment.ekfs["uav_00"].state[:3] = (
        environment.goals["uav_00"] + np.asarray([15.0, 0.0, 0.0])
    )
    environment.ekfs["uav_00"].state[3:6] = [8.0, 0.0, 0.0]
    environment.states["uav_00"].velocity_mps[:] = [8.0, 0.0, 0.0]
    current_distance = environment.estimated_goal_distance()
    environment._previous_estimated_goal_distance = current_distance + 1.0
    environment._previous_position_logdet = environment._position_logdet_uncertainty()

    _reporting, info = environment._rewards(
        0,
        _actions(environment),
        _execution(environment),
    )
    components = info["learning_reward_components"]
    assert components["single_uav_speed_control_mode"] == "braking_distance"
    assert np.isclose(float(components["single_uav_target_speed_mps"]), 6.0)
    assert np.isclose(float(components["single_uav_speed_excess_mps"]), 2.0)
    assert float(components["single_uav_brake_margin_m"]) < 0.0

    _reporting, braking_info = environment._rewards(
        0,
        _actions(environment),
        {
            "uav_00": {
                "delta_v_applied": [-0.4, 0.0, 0.0],
                "pre_speed_mps": 8.0,
                "post_speed_mps": 7.6,
                "applied_longitudinal_delta_v_mps": -0.4,
            }
        },
    )
    braking_components = braking_info["learning_reward_components"]
    assert float(braking_components["single_uav_braking_need_signal"]) > 0.0
    assert float(braking_components["single_uav_braking_action_signal"]) > 0.0
    assert float(braking_components["single_uav_braking_action_reward"]) > 0.0

    _reporting, accelerating_info = environment._rewards(
        0,
        _actions(environment),
        {
            "uav_00": {
                "delta_v_applied": [0.4, 0.0, 0.0],
                "pre_speed_mps": 8.0,
                "post_speed_mps": 8.4,
                "applied_longitudinal_delta_v_mps": 0.4,
            }
        },
    )
    accelerating_components = accelerating_info["learning_reward_components"]
    assert float(accelerating_components["single_uav_braking_action_reward"]) < 0.0


def test_single_uav_target_velocity_action_shaping_prefers_braking_delta() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                single_uav_speed_control_mode="braking_distance",
                single_uav_near_goal_radius_m=8.0,
                single_uav_braking_acceleration_mps2=2.0,
                single_uav_braking_margin_m=2.0,
                single_uav_braking_action_reward_coef=0.0,
                single_uav_target_velocity_action_reward_coef=1.0,
                potential_statistics=_statistics(),
            ),
        )
    )
    environment.reset(seed=42)
    goal = environment.goals["uav_00"].copy()
    position = goal + np.asarray([4.0, 0.0, 0.0])
    environment.ekfs["uav_00"].state[:3] = position
    environment.states["uav_00"].position_m[:] = position
    environment.states["uav_00"].velocity_mps[:] = [3.4, 0.0, 0.0]
    current_distance = environment.estimated_goal_distance()
    environment._previous_estimated_goal_distance = current_distance + 1.0
    environment._previous_position_logdet = environment._position_logdet_uncertainty()

    _reporting, braking_info = environment._rewards(
        0,
        _actions(environment),
        {
            "uav_00": {
                "delta_v_applied": [-0.6, 0.0, 0.0],
                "pre_speed_mps": 4.0,
                "post_speed_mps": 3.4,
            }
        },
    )
    braking_components = braking_info["learning_reward_components"]
    assert np.isclose(
        float(braking_components["single_uav_target_velocity_action_error"]), 0.0
    )
    assert float(braking_components["single_uav_target_velocity_action_reward"]) > 0.9

    environment.states["uav_00"].velocity_mps[:] = [4.6, 0.0, 0.0]
    _reporting, accelerating_info = environment._rewards(
        0,
        _actions(environment),
        {
            "uav_00": {
                "delta_v_applied": [0.6, 0.0, 0.0],
                "pre_speed_mps": 4.0,
                "post_speed_mps": 4.6,
            }
        },
    )
    accelerating_components = accelerating_info["learning_reward_components"]
    assert float(
        accelerating_components["single_uav_target_velocity_action_reward"]
    ) < -0.9


def test_single_uav_stable_arrival_requires_consecutive_hold_steps() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=5,
            gnss_mode="denied",
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v2",
                single_uav_stable_hold_steps=3,
                single_uav_goal_hold_reward=0.2,
                potential_statistics=_statistics(),
            ),
        )
    )
    target = [170.0, 170.0, 10.0]
    environment.reset(
        seed=42,
        scenario_config={
            "initial_positions": {"uav_00": target},
            "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
            "goals": {"uav_00": target},
            "obstacles": [],
        },
    )
    environment._previous_estimated_goal_distance = 1.0
    environment._previous_position_logdet = environment._position_logdet_uncertainty()

    results = []
    for _ in range(3):
        _reporting, info = environment._rewards(
            0,
            _actions(environment),
            _execution(environment),
        )
        results.append(info["learning_reward_components"])
    assert [int(item["single_uav_stable_steps"]) for item in results] == [1, 2, 3]
    assert [bool(item["single_uav_goal_stable"]) for item in results] == [False, False, True]
    assert float(results[0]["single_uav_goal_hold_reward"]) == 0.2
    assert bool(results[2]["single_uav_goal_bonus_triggered"]) is True


def test_default_single_uav_mode_preserves_legacy_reward_path() -> None:
    baseline = MultiUAVJointEnv(
        SimulationConfig(uav_count=1, max_steps=2, gnss_mode="denied")
    )
    explicit_off = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            max_steps=2,
            gnss_mode="denied",
            reward=RewardConfig(single_uav_learning_mode="off"),
        )
    )
    baseline.reset(seed=2026)
    explicit_off.reset(seed=2026)

    baseline_reporting, baseline_info = baseline._rewards(
        0, _actions(baseline), _execution(baseline)
    )
    explicit_reporting, explicit_info = explicit_off._rewards(
        0, _actions(explicit_off), _execution(explicit_off)
    )

    assert explicit_reporting == baseline_reporting
    assert explicit_info["learning_rewards"] == baseline_info["learning_rewards"]
    assert explicit_info["communication_applicable"] is False
    assert explicit_info["single_uav_learning_mode"] == "off"


def test_single_uav_energy_and_collision_are_negative_and_bounded() -> None:
    environment = _single_environment()
    environment.reset(seed=42)
    current_uncertainty = environment._position_logdet_uncertainty()
    current_goal_distance = environment.estimated_goal_distance()
    environment._previous_position_logdet = current_uncertainty
    environment._previous_estimated_goal_distance = current_goal_distance

    _reporting, energy_info = environment._rewards(
        0,
        _actions(environment, delta_v=[1.0, 0.0, 0.0]),
        _execution(environment, delta_v=[1.0, 0.0, 0.0]),
    )
    assert float(next(iter(energy_info["learning_rewards"].values()))) < 0.0
    assert float(energy_info["learning_reward_components"]["energy_cost"]) > 0.0

    environment.reset(seed=42)
    _reporting, collision_info = environment._rewards(
        1,
        _actions(environment),
        _execution(environment),
    )
    collision_value = float(next(iter(collision_info["learning_rewards"].values())))
    assert collision_value < 0.0
    assert -1.0 <= collision_value <= 1.0
    assert float(
        collision_info["learning_reward_components"]["single_uav_learning_collision_cost"]
    ) > 0.0


def test_single_uav_masks_have_only_none_and_no_candidate_or_self_link() -> None:
    environment = _single_environment()
    observations, _info = environment.reset(seed=42)
    assert observations["uav_00"]["action_mask"].tolist() == [False, True]
    assert observations["uav_00"]["bandwidth_mask"].tolist() == [False]
    assert environment.snapshot.candidate_edges == set()
    assert environment.snapshot.potential_edges == set()
    assert all(source != target for source, target in environment.snapshot.candidate_edges)
    assert all(source != target for source, target in environment.snapshot.potential_edges)


def test_single_uav_mode_does_not_change_multi_uav_communication_learning_path() -> None:
    common = {
        "uav_count": 3,
        "max_steps": 2,
        "gnss_mode": "denied",
    }
    legacy = MultiUAVJointEnv(SimulationConfig(**common))
    opt_in = MultiUAVJointEnv(
        SimulationConfig(
            **common,
            reward=RewardConfig(
                single_uav_learning_mode="loc_goal_v1",
                potential_statistics=_statistics(),
            ),
        )
    )
    legacy.reset(seed=2026)
    opt_in.reset(seed=2026)
    actions_legacy = _actions(legacy)
    actions_opt_in = _actions(opt_in)
    report_legacy, info_legacy = legacy._rewards(
        0, actions_legacy, _execution(legacy)
    )
    report_opt_in, info_opt_in = opt_in._rewards(
        0, actions_opt_in, _execution(opt_in)
    )
    assert info_opt_in["communication_applicable"] is True
    assert info_opt_in["learning_reward_components"]["single_uav_learning_active"] is False
    assert report_opt_in == report_legacy
    assert info_opt_in["learning_rewards"] == info_legacy["learning_rewards"]
    assert all(source != target for source, target in opt_in.snapshot.candidate_edges)
