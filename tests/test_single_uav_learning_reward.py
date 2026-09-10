"""Acceptance tests for the localization-first single-UAV learning channel."""

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
