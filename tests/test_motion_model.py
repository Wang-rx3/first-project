from __future__ import annotations

import numpy as np
import pytest

from src.simulation.motion_model import (
    BRAKING_DISTANCE_SPEED_PROFILE_EARLY_SMOOTH,
    BRAKING_DISTANCE_SPEED_PROFILE_LEGACY,
    MotionConfig,
    UAVKinematicState,
    apply_velocity_increment,
    braking_distance_target_speed,
    stopping_distance_m,
)


def _state(velocity: tuple[float, float, float]) -> UAVKinematicState:
    return UAVKinematicState(
        position_m=np.asarray([10.0, 10.0, 10.0], dtype=float),
        velocity_mps=np.asarray(velocity, dtype=float),
        attitude_rad=np.zeros(3, dtype=float),
        acceleration_mps2=np.zeros(3, dtype=float),
        angular_velocity_rps=np.zeros(3, dtype=float),
    )


def test_motion_action_contract_explicitly_allows_braking() -> None:
    contract = MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0).action_contract()

    assert contract["type"] == "continuous_3d_velocity_increment"
    assert contract["negative_delta_v_allowed"] is True
    assert contract["braking_supported"] is True
    assert np.isclose(float(contract["max_norm_mps"]), 0.6)


def test_negative_delta_v_reduces_speed_and_is_logged_as_effective_braking() -> None:
    state = _state((4.0, 0.0, 0.0))
    next_state, info = apply_velocity_increment(
        state,
        (-0.6, 0.0, 0.0),
        MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0),
    )

    assert np.isclose(float(np.linalg.norm(next_state.velocity_mps)), 3.4)
    assert np.isclose(float(info["applied_longitudinal_delta_v_mps"]), -0.6)
    assert np.isclose(float(info["braking_delta_v_mps"]), 0.6)
    assert np.isclose(float(info["speed_reduction_mps"]), 0.6)
    assert int(info["braking_action"]) == 1
    assert int(info["brake_effective"]) == 1
    assert int(info["brake_blocked"]) == 0


def test_negative_delta_v_at_rest_is_not_mislabeled_as_braking() -> None:
    _next_state, info = apply_velocity_increment(
        _state((0.0, 0.0, 0.0)),
        (-0.6, 0.0, 0.0),
        MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0),
    )

    assert int(info["braking_action"]) == 0
    assert int(info["brake_effective"]) == 0


def test_stopping_distance_and_target_speed_are_bounded_and_monotone() -> None:
    assert np.isclose(stopping_distance_m(12.0, 0.5, 2.0), 35.9375)
    slow_target = braking_distance_target_speed(
        5.0,
        goal_threshold_m=1.0,
        stable_speed_mps=0.5,
        braking_acceleration_mps2=2.0,
        braking_margin_m=2.0,
        far_target_speed_mps=6.0,
        max_speed_mps=12.0,
    )
    far_target = braking_distance_target_speed(
        15.0,
        goal_threshold_m=1.0,
        stable_speed_mps=0.5,
        braking_acceleration_mps2=2.0,
        braking_margin_m=2.0,
        far_target_speed_mps=6.0,
        max_speed_mps=12.0,
    )
    assert 0.5 <= slow_target < far_target <= 6.0


def test_early_smooth_braking_profile_is_continuous_and_preserves_endpoints() -> None:
    kwargs = {
        "goal_threshold_m": 1.0,
        "near_goal_radius_m": 8.0,
        "stable_speed_mps": 0.5,
        "braking_acceleration_mps2": 2.0,
        "braking_margin_m": 2.0,
        "far_target_speed_mps": 6.0,
        "max_speed_mps": 12.0,
        "speed_profile": BRAKING_DISTANCE_SPEED_PROFILE_EARLY_SMOOTH,
    }
    distances = np.linspace(0.0, 8.0, 801)
    speeds = np.asarray(
        [braking_distance_target_speed(float(distance), **kwargs) for distance in distances]
    )
    assert np.isclose(speeds[0], 0.5)
    assert np.isclose(speeds[-1], 6.0)
    assert np.all(np.diff(speeds) >= -1e-10)
    assert np.isclose(
        braking_distance_target_speed(3.0, **kwargs),
        braking_distance_target_speed(3.0 - 1e-8, **kwargs),
        atol=1e-6,
    )
    assert np.isclose(
        braking_distance_target_speed(3.0, **kwargs),
        braking_distance_target_speed(3.0 + 1e-8, **kwargs),
        atol=1e-6,
    )
    assert braking_distance_target_speed(3.0, **kwargs) > 0.5
    assert braking_distance_target_speed(5.0, **kwargs) < braking_distance_target_speed(
        5.0,
        **{**kwargs, "speed_profile": BRAKING_DISTANCE_SPEED_PROFILE_LEGACY},
    )


def test_unknown_braking_distance_speed_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="speed_profile"):
        braking_distance_target_speed(
            3.0,
            goal_threshold_m=1.0,
            stable_speed_mps=0.5,
            braking_acceleration_mps2=2.0,
            braking_margin_m=2.0,
            far_target_speed_mps=6.0,
            max_speed_mps=12.0,
            speed_profile="unknown",
        )
