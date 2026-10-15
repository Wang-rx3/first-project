"""Pure, environment-free tests for the fixed navigation controller."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.control.fixed_navigation import (
    FixedNavigationConfig,
    FixedWaypointNavigator,
    fixed_navigation_command,
    local_observation_command,
)


def test_default_config_is_frozen_and_has_expected_component_limit() -> None:
    config = FixedNavigationConfig()
    assert config.position_gain_s_inv == 0.6
    assert config.velocity_gain_s_inv == 1.6
    assert config.cruise_speed_mps == 2.5
    assert config.waypoint_radius_m == 0.8
    assert config.max_delta_v_norm_mps == 0.6
    assert config.dt_s == 0.2
    assert config.component_limit_mps == pytest.approx(0.6 / np.sqrt(3.0))
    with pytest.raises(FrozenInstanceError):
        config.dt_s = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("field", [
    "position_gain_s_inv",
    "velocity_gain_s_inv",
    "cruise_speed_mps",
    "waypoint_radius_m",
    "max_delta_v_norm_mps",
    "dt_s",
])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_nonpositive_or_nonfinite_values(field: str, bad_value: float) -> None:
    with pytest.raises(ValueError, match=field):
        FixedNavigationConfig(**{field: bad_value})


def test_command_uses_the_declared_formula_and_limits_each_axis_and_norm() -> None:
    config = FixedNavigationConfig()
    estimate = np.array([0.0, 0.0, 0.0, -2.0, 0.5, 0.0])
    reference = np.array([20.0, 10.0, 0.0])
    result = fixed_navigation_command(estimate, reference, config)

    desired = np.array([2.5 * 20.0 / np.sqrt(500.0), 2.5 * 10.0 / np.sqrt(500.0), 0.0])
    expected_acceleration = 1.6 * (desired - estimate[3:6])
    expected_raw = expected_acceleration * 0.2
    expected_axis = np.clip(expected_raw, -config.component_limit_mps, config.component_limit_mps)
    expected_command = expected_axis.copy()

    np.testing.assert_array_equal(result["estimated_position_m"], estimate[:3])
    np.testing.assert_array_equal(result["estimated_velocity_mps"], estimate[3:6])
    np.testing.assert_array_equal(result["reference_position_m"], reference)
    np.testing.assert_allclose(result["desired_velocity_mps"], desired)
    np.testing.assert_allclose(result["requested_acceleration_mps2"], expected_acceleration)
    np.testing.assert_allclose(result["raw_delta_v_mps"], expected_raw)
    np.testing.assert_allclose(result["axis_limited_delta_v_mps"], expected_axis)
    np.testing.assert_allclose(result["command_delta_v_mps"], expected_command)
    assert result["command_norm"] == pytest.approx(np.linalg.norm(expected_axis))
    assert result["controller_limited"] is True
    assert np.max(np.abs(result["command_delta_v_mps"])) <= config.component_limit_mps + 1e-12
    assert np.linalg.norm(result["command_delta_v_mps"]) <= config.max_delta_v_norm_mps + 1e-12


def test_command_supports_three_dimensions_and_brakes_moving_vehicle_near_goal() -> None:
    estimate = np.array([9.5, -2.0, 3.0, 2.0, 0.0, 0.0])
    reference = np.array([10.0, -2.0, 3.0])
    result = fixed_navigation_command(estimate, reference, FixedNavigationConfig())
    command = np.asarray(result["command_delta_v_mps"])
    assert command.shape == (3,)
    assert command[0] < 0.0
    assert np.linalg.norm(command) <= 0.6 + 1e-12


def test_zero_position_and_velocity_error_produces_zero_action() -> None:
    estimate = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    result = fixed_navigation_command(estimate, estimate[:3], FixedNavigationConfig())
    np.testing.assert_array_equal(result["command_delta_v_mps"], np.zeros(3))
    assert result["command_norm"] == 0.0
    assert result["controller_limited"] is False


def test_local_observation_ignores_poisoned_non_controller_fields_bitwise() -> None:
    estimate = np.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3])
    goal = np.array([8.0, -1.0, 5.0])
    clean = {"estimate": estimate.copy(), "goal": goal.copy()}
    poisoned = {
        **clean,
        "truth": {"position": np.array([np.nan, np.inf, -np.inf])},
        "graph": object(),
        "noise": np.nan,
        "reward": float("inf"),
        "checkpoint": object(),
    }
    first = local_observation_command(clean)
    second = local_observation_command(poisoned)
    assert first.keys() == second.keys()
    for key in first:
        first_value = first[key]
        second_value = second[key]
        if isinstance(first_value, np.ndarray):
            np.testing.assert_array_equal(first_value, second_value)
        else:
            assert first_value == second_value


def test_local_observation_explicit_reference_ignores_observation_goal() -> None:
    estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    local = {"estimate": estimate, "goal": np.array([100.0, 0.0, 0.0]), "reward": 99.0}
    explicit = np.array([0.0, 4.0, 0.0])
    result = local_observation_command(local, reference=explicit)
    np.testing.assert_array_equal(result["reference_position_m"], explicit)


def test_changing_unsaturated_estimate_changes_command() -> None:
    config = FixedNavigationConfig()
    first = fixed_navigation_command(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], config
    )
    second = fixed_navigation_command(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0], config
    )
    assert not np.array_equal(first["command_delta_v_mps"], second["command_delta_v_mps"])


def test_waypoint_switching_uses_only_estimated_position_and_final_is_sticky() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    navigator = FixedWaypointNavigator(points)
    estimate = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = navigator.command(estimate)
    assert first["reference_index"] == 1
    assert first["reference_count"] == 3
    assert first["waypoint_switch_count"] == 1

    at_second = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    final = navigator.command(at_second)
    assert final["reference_index"] == 2
    assert final["waypoint_switch_count"] == 2
    at_final = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    again = navigator.command(at_final)
    assert again["reference_index"] == 2
    assert again["waypoint_switch_count"] == 2


def test_waypoint_progress_ignores_truth_graph_noise_and_reward_fields() -> None:
    navigator = FixedWaypointNavigator([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    estimate = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    # ``command`` accepts exactly the estimate vector; fields outside the
    # estimate cannot influence waypoint selection.
    first = navigator.command(estimate)
    assert first["reference_index"] == 1
    assert navigator.index == 1


def test_waypoints_are_defensively_copied_and_outputs_cannot_mutate_goal() -> None:
    points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    navigator = FixedWaypointNavigator(points)
    points[1] = [99.0, 99.0, 99.0]
    command = navigator.command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    command["reference_position_m"][0] = -100.0
    exposed = navigator.references
    exposed[1][0] = -200.0
    next_command = navigator.command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(next_command["reference_position_m"], [5.0, 0.0, 0.0])


def test_navigator_reset_starts_a_fresh_case() -> None:
    navigator = FixedWaypointNavigator([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    navigator.command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert navigator.index == 1
    navigator.reset()
    assert navigator.index == 0
    assert navigator.waypoint_switch_count == 0


@pytest.mark.parametrize("bad_estimate", [[0.0] * 5, [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]])
def test_inputs_require_finite_estimate_with_six_values(bad_estimate: list[float]) -> None:
    with pytest.raises(ValueError):
        fixed_navigation_command(bad_estimate, [0.0, 0.0, 0.0], FixedNavigationConfig())


def test_reference_must_be_a_finite_three_vector() -> None:
    with pytest.raises(ValueError):
        fixed_navigation_command([0.0] * 6, [0.0, 0.0], FixedNavigationConfig())
    with pytest.raises(ValueError):
        fixed_navigation_command([0.0] * 6, [0.0, 0.0, float("inf")], FixedNavigationConfig())
