from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.single_uav_rl_equivalent_oracle_audit import (
    ESTIMATED_ORACLE,
    EstimatedStateOracleController,
    _angle_deg,
    _distance_bin,
    _extract_rl_visible_state,
)
from src.simulation.environment import RewardConfig
from src.simulation.motion_model import MotionConfig


def _controller_environment() -> SimpleNamespace:
    return SimpleNamespace(
        uav_ids=["uav_00"],
        config=SimpleNamespace(
            motion=MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0),
            reward=RewardConfig(
                single_uav_goal_threshold_m=1.0,
                single_uav_goal_speed_threshold_mps=0.5,
                single_uav_near_goal_radius_m=8.0,
                single_uav_target_speed_far_mps=6.0,
                single_uav_braking_acceleration_mps2=2.0,
            ),
        ),
    )


def _observation(
    *,
    position=(0.0, 0.0, 2.0),
    velocity=(0.0, 0.0, 0.0),
    goal_delta=(5.0, 0.0, 0.0),
) -> dict[str, object]:
    node = np.zeros((1, 39), dtype=float)
    node[0, :3] = position
    node[0, 3:6] = velocity
    node[0, 30:33] = goal_delta
    return {"node_features": node, "node_ids": ["uav_00"]}


def test_estimated_oracle_uses_only_actor_visible_node_features() -> None:
    controller = EstimatedStateOracleController(_controller_environment())

    action, diagnostics = controller.action({"uav_00": _observation()})
    delta = np.asarray(action["uav_00"]["delta_v"], dtype=float)

    assert diagnostics["state_source"] == "actor_visible_node_features_only"
    assert diagnostics["truth_state_read_for_control"] is False
    assert delta[0] > 0.0
    assert np.linalg.norm(delta) <= 0.6 + 1.0e-12
    assert action["uav_00"]["measurement_uav_id"] == "none"


def test_estimated_oracle_does_not_fallback_to_truth() -> None:
    controller = EstimatedStateOracleController(_controller_environment())

    with pytest.raises(ValueError, match="node_features"):
        controller.action({"uav_00": {}})


def test_visible_state_extracts_estimate_and_goal_delta_without_convenience_fields() -> None:
    visible = _extract_rl_visible_state(_observation(velocity=(1.0, 0.0, 0.0)))

    assert visible["state_source"] == "actor_visible_node_features_only"
    assert visible["distance_m"] == pytest.approx(5.0)
    assert visible["radial_velocity_mps"] == pytest.approx(1.0)


def test_distance_bins_and_angular_error_are_explicit() -> None:
    assert _distance_bin(9.0) == "gt_8m"
    assert _distance_bin(8.0) == "8_5m"
    assert _distance_bin(5.0) == "5_3m"
    assert _distance_bin(3.0) == "3_1m"
    assert _distance_bin(1.0) == "le_1m"
    assert _angle_deg(np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))) == pytest.approx(90.0)


def test_policy_name_is_kept_independent_from_legacy_feasibility_module() -> None:
    assert ESTIMATED_ORACLE == "estimated_state_oracle"
