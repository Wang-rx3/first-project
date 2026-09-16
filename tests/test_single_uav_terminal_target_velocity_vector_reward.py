import numpy as np

from src.simulation.environment import (
    terminal_target_velocity_vector_reward_components,
)


def _components(**overrides):
    values = {
        "previous_position_m": np.array([0.0, 0.0, 0.0]),
        "goal_position_m": np.array([2.0, 0.0, 0.0]),
        "previous_velocity_mps": np.array([3.0, 0.0, 0.0]),
        "current_velocity_mps": np.array([2.0, 0.0, 0.0]),
        "current_goal_distance_m": 2.0,
        "target_speed_mps": 1.0,
        "reward_coef": 0.5,
    }
    values.update(overrides)
    return terminal_target_velocity_vector_reward_components(**values)


def test_terminal_vector_reward_is_zero_outside_band() -> None:
    assert _components(current_goal_distance_m=4.0)["reward"] == 0.0
    assert _components(current_goal_distance_m=1.0)["reward"] == 0.0


def test_terminal_vector_reward_credits_error_reduction() -> None:
    result = _components()
    assert result["active"] is True
    assert result["error_pre_mps"] == 2.0
    assert result["error_post_mps"] == 1.0
    assert result["error_reduction_mps"] == 1.0
    assert result["signal"] == 1.0
    assert result["reward"] == 0.5


def test_terminal_vector_reward_is_zero_without_improvement_or_when_worse() -> None:
    assert _components(
        current_velocity_mps=np.array([3.0, 0.0, 0.0])
    )["reward"] == 0.0
    assert _components(
        current_velocity_mps=np.array([4.0, 0.0, 0.0])
    )["reward"] == 0.0


def test_terminal_vector_reward_credits_direction_and_overspeed_correction() -> None:
    direction = _components(
        previous_velocity_mps=np.array([0.0, 2.0, 0.0]),
        current_velocity_mps=np.array([1.0, 0.0, 0.0]),
    )
    assert direction["reward"] > 0.0
    braking = _components(
        previous_velocity_mps=np.array([4.0, 0.0, 0.0]),
        current_velocity_mps=np.array([2.0, 0.0, 0.0]),
    )
    assert braking["reward"] == 0.5


def test_terminal_vector_reward_is_positive_only_and_bounded() -> None:
    result = _components(
        previous_velocity_mps=np.array([10.0, 0.0, 0.0]),
        current_velocity_mps=np.array([0.0, 0.0, 0.0]),
    )
    assert 0.0 <= result["reward"] <= 0.5


def test_radial_gate_uses_pre_action_distance_and_preserves_frozen_target() -> None:
    gated = _components(
        current_goal_distance_m=4.0,
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([0.0, 0.0, 0.0]),
        current_velocity_mps=np.array([0.5, 0.0, 0.0]),
        radial_gate_enabled=True,
    )
    assert gated["active"] is True
    assert gated["activation_distance_m"] == 2.0
    assert gated["post_radial_velocity_mps"] == 0.5
    assert gated["radial_gate"] == 0.5
    assert gated["reward"] == 0.25

    historical = _components(
        current_goal_distance_m=4.0,
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([0.0, 0.0, 0.0]),
        current_velocity_mps=np.array([0.5, 0.0, 0.0]),
        radial_gate_enabled=False,
    )
    assert historical["active"] is False
    assert historical["reward"] == 0.0


def test_radial_gate_suppresses_nonpositive_radial_reward() -> None:
    for velocity in (
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.0, 0.0]),
    ):
        result = _components(
            previous_goal_distance_m=2.0,
            previous_velocity_mps=np.array([2.0, 0.0, 0.0]),
            current_velocity_mps=velocity,
            radial_gate_enabled=True,
        )
        assert result["active"] is True
        assert result["post_radial_velocity_mps"] <= 0.0
        assert result["radial_gate"] == 0.0
        assert result["reward"] == 0.0


def test_radial_gate_rewards_stronger_positive_alignment_more() -> None:
    weak = _components(
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([0.0, 0.0, 0.0]),
        current_velocity_mps=np.array([0.2, 0.0, 0.0]),
        radial_gate_enabled=True,
    )
    strong = _components(
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([0.0, 0.0, 0.0]),
        current_velocity_mps=np.array([0.8, 0.0, 0.0]),
        radial_gate_enabled=True,
    )
    assert 0.0 < weak["radial_gate"] < strong["radial_gate"] <= 1.0
    assert 0.0 < weak["reward"] < strong["reward"] <= 0.5


def test_radial_gate_is_clipped_to_one() -> None:
    result = _components(
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([2.0, 0.0, 0.0]),
        current_velocity_mps=np.array([1.5, 0.0, 0.0]),
        radial_gate_enabled=True,
    )
    assert result["radial_gate"] == 1.0
    assert result["reward"] == 0.5


def test_gated_vector_reward_requires_pre_distance_and_positive_reduction() -> None:
    outside = _components(
        previous_goal_distance_m=3.1,
        current_goal_distance_m=2.0,
        radial_gate_enabled=True,
    )
    assert outside["active"] is False
    assert outside["reward"] == 0.0

    no_reduction = _components(
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([1.0, 0.0, 0.0]),
        current_velocity_mps=np.array([1.0, 0.0, 0.0]),
        radial_gate_enabled=True,
    )
    assert no_reduction["active"] is True
    assert no_reduction["error_reduction_mps"] == 0.0
    assert no_reduction["reward"] == 0.0


def test_radial_gate_default_keeps_historical_reward_contract() -> None:
    result = _components(
        previous_goal_distance_m=2.0,
        previous_velocity_mps=np.array([3.0, 0.0, 0.0]),
        current_velocity_mps=np.array([2.0, 0.0, 0.0]),
    )
    assert result["active"] is True
    assert result["radial_gate_enabled"] is False
    assert result["radial_gate"] == 1.0
    assert result["reward"] == 0.5
