from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.single_uav_actor_constrained_oracle import (
    ACTOR_CONSTRAINED_ORACLE,
    ActorConstrainedEstimatedStateOracleController,
    _action_amplitude_distance_summary,
    actor_action_transform,
    actor_component_limit,
    actor_reachable_radial_norm,
)
from src.experiments.single_uav_rl_equivalent_oracle_audit import _extract_rl_visible_state
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


def test_actor_component_box_is_stricter_than_environment_norm_ball() -> None:
    limit = actor_component_limit(0.6)
    transformed = actor_action_transform((0.6, -0.6, 0.6), 0.6)
    action = np.asarray(transformed["constrained_delta_v_mps"], dtype=float)

    assert limit == pytest.approx(0.6 / np.sqrt(3.0))
    assert np.max(np.abs(action)) <= limit + 1.0e-12
    assert np.linalg.norm(action) == pytest.approx(0.6)
    assert transformed["actor_constraint_affected"] == 1
    assert transformed["actor_any_axis_saturated"] == 1


def test_actor_transform_preserves_legal_action_when_inside_box() -> None:
    transformed = actor_action_transform((0.1, -0.2, 0.05), 0.6)
    action = np.asarray(transformed["constrained_delta_v_mps"], dtype=float)

    assert action == pytest.approx((0.1, -0.2, 0.05))
    assert transformed["actor_constraint_affected"] == 0
    assert transformed["actor_any_axis_saturated"] == 0


def test_radial_reachability_matches_axis_and_diagonal_directions() -> None:
    assert actor_reachable_radial_norm((1.0, 0.0, 0.0), 0.6) == pytest.approx(0.6 / np.sqrt(3.0))
    assert actor_reachable_radial_norm((1.0, 1.0, 1.0), 0.6) == pytest.approx(0.6)


def test_actor_constrained_controller_uses_only_visible_features_and_actor_box() -> None:
    environment = _controller_environment()
    controller = ActorConstrainedEstimatedStateOracleController(environment)
    action, diagnostics = controller.action(
        {"uav_00": _observation(velocity=(2.0, 0.0, 0.0), goal_delta=(5.0, 0.0, 0.0))}
    )
    delta = np.asarray(action["uav_00"]["delta_v"], dtype=float)
    visible = _extract_rl_visible_state(_observation(velocity=(2.0, 0.0, 0.0)))

    assert diagnostics["state_source"] == "actor_visible_node_features_only"
    assert diagnostics["truth_state_read_for_control"] is False
    assert diagnostics["action_parameterization"] == "normal_tanh_component_box"
    assert np.max(np.abs(delta)) <= actor_component_limit(0.6) + 1.0e-12
    assert np.linalg.norm(delta) <= 0.6 + 1.0e-12
    assert visible["state_source"] == "actor_visible_node_features_only"
    assert action["uav_00"]["measurement_uav_id"] == "none"


def test_actor_constrained_role_name_is_stable() -> None:
    assert ACTOR_CONSTRAINED_ORACLE == "actor_constrained_estimated_state_oracle"


def test_action_amplitude_summary_reports_rl_to_successful_oracle_ratio() -> None:
    rows = [
        {
            "policy": ACTOR_CONSTRAINED_ORACLE,
            "distance_bin": "3_1m",
            "action_norm_mps": 0.4,
            "action_projection_toward_goal_mps": -0.3,
            "braking_projection_mps": 0.3,
            "tangential_action_mps": 0.1,
            "radial_velocity_toward_mps": 1.0,
            "total_speed_mps": 1.2,
            "target_speed_mps": 1.0,
        },
        {
            "policy": "baseline_rl",
            "distance_bin": "3_1m",
            "action_norm_mps": 0.1,
            "action_projection_toward_goal_mps": 0.02,
            "braking_projection_mps": 0.0,
            "tangential_action_mps": 0.08,
            "radial_velocity_toward_mps": -0.1,
            "total_speed_mps": 1.8,
            "target_speed_mps": 0.5,
        },
    ]
    summary = _action_amplitude_distance_summary(rows)
    baseline = next(
        row
        for row in summary
        if row["policy"] == "baseline_rl" and row["distance_bin"] == "3_1m"
    )

    assert baseline["mean_action_norm_ratio_vs_actor_constrained_oracle"] == pytest.approx(0.25)
