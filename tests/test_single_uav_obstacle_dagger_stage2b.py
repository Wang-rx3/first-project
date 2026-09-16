from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.experiments.single_uav_obstacle_dagger_stage2b import (
    GeometricObstacleOracleController,
    _candidate_from_estimate,
    _query_gate,
    _stage2b_behavioral_acceptance,
    _world_to_body,
    select_relevant_obstacle,
    training_episode_kind,
    training_schedule_counts,
)
from src.experiments.single_uav_obstacle_observation_audit import probe_scenarios
from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from src.simulation.motion_model import MotionConfig
from src.simulation.obstacle_model import SphereObstacle


def test_stage2b_training_schedule_is_exactly_70_30_for_smoke() -> None:
    assert training_schedule_counts(30) == {
        "normal_far_distance": 21,
        "static_pressure": 9,
    }
    assert training_episode_kind(1) == "normal_far_distance"
    assert training_episode_kind(7) == "normal_far_distance"
    assert training_episode_kind(8) == "static_pressure"
    assert training_episode_kind(10) == "static_pressure"


def test_relevant_obstacle_uses_clearance_then_id_tie_break() -> None:
    position = np.zeros(3)
    farther = SphereObstacle("b", np.asarray([5.0, 0.0, 0.0]), 1.0, velocity_mps=np.zeros(3))
    nearer = SphereObstacle("a", np.asarray([3.0, 0.0, 0.0]), 1.0, velocity_mps=np.zeros(3))
    assert select_relevant_obstacle(position, [farther, nearer]).obstacle_id == "a"

    tied_a = SphereObstacle("a", np.asarray([3.0, 0.0, 0.0]), 1.0, velocity_mps=np.zeros(3))
    tied_b = SphereObstacle("b", np.asarray([3.0, 0.0, 0.0]), 1.0, velocity_mps=np.zeros(3))
    assert select_relevant_obstacle(position, [tied_b, tied_a]).obstacle_id == "a"
    assert select_relevant_obstacle(position, []) is None


def test_direction_vector_body_frame_preserves_forward_left_signs() -> None:
    assert np.allclose(_world_to_body(np.asarray([2.0, 0.0, 1.0]), 0.0), [2.0, 0.0, 1.0])
    assert np.allclose(_world_to_body(np.asarray([0.0, 2.0, 0.0]), 0.0), [0.0, 2.0, 0.0])
    assert np.allclose(_world_to_body(np.asarray([0.0, -2.0, 0.0]), 0.0), [0.0, -2.0, 0.0])
    assert np.allclose(_world_to_body(np.asarray([0.0, 2.0, 0.0]), np.pi / 2.0), [2.0, 0.0, 0.0])


def test_estimated_candidate_respects_delta_v_and_speed_limits() -> None:
    motion = MotionConfig(dt_s=0.2, max_speed_mps=12.0, max_acceleration_mps2=3.0)
    visible = {
        "position_m": np.asarray([10.0, 20.0, 5.0]),
        "velocity_mps": np.asarray([4.0, 0.0, 0.0]),
    }
    candidate, next_velocity = _candidate_from_estimate(
        visible,
        np.asarray([10.0, 0.0, 0.0]),
        motion,
    )
    assert np.linalg.norm(next_velocity - visible["velocity_mps"]) <= 0.6 + 1.0e-9
    assert np.linalg.norm(next_velocity) <= motion.max_speed_mps + 1.0e-9
    assert np.allclose(candidate, [10.86, 20.0, 5.0])


def test_oracle_side_choice_maps_left_obstacle_to_right_bypass() -> None:
    oracle = object.__new__(GeometricObstacleOracleController)
    obstacle = SphereObstacle("o", np.asarray([5.0, 0.0, 5.0]), 1.0, velocity_mps=np.zeros(3))
    goal_direction = np.asarray([1.0, 0.0, 0.0])
    left_visible = {"obstacle_direction_body_m": np.asarray([5.0, 2.0, 0.0])}
    right_visible = {"obstacle_direction_body_m": np.asarray([5.0, -2.0, 0.0])}
    assert oracle._choose_side(left_visible, obstacle, goal_direction, 0.0) == -1.0
    assert oracle._choose_side(right_visible, obstacle, goal_direction, 0.0) == 1.0


def test_oracle_front_bypass_uses_legal_finite_action_and_normal_step() -> None:
    environment = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            seed=2026,
            reward=RewardConfig(single_uav_speed_control_mode="braking_distance"),
            single_uav_obstacle_direction_features_enabled=True,
        )
    )
    observations, info = environment.reset(
        seed=2026,
        scenario_config=probe_scenarios()["front"],
    )
    before = environment.states["uav_00"].position_m.copy()
    oracle = GeometricObstacleOracleController(environment)
    action, control = oracle.action(observations)
    delta_v = np.asarray(action["uav_00"]["delta_v"], dtype=float)
    max_delta_v = environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s
    assert np.isfinite(delta_v).all()
    assert np.linalg.norm(delta_v) <= max_delta_v + 1.0e-9
    assert control["truth_state_read_for_control"] is False
    assert control["route_phase"] in {"bypass_pre", "bypass_post", "goal_after_bypass"}
    next_observations, _rewards, terminated, truncated, _next_info = environment.step(action)
    after = environment.states["uav_00"].position_m
    assert np.isfinite(np.asarray(next_observations["uav_00"]["node_features"], dtype=float)).all()
    assert np.linalg.norm(after - before) > 0.0
    assert not (terminated or truncated)


def test_query_gate_does_not_query_oracle_without_physical_obstacle() -> None:
    visible = {"position_m": np.zeros(3)}
    environment = SimpleNamespace(obstacles=[])
    gate, info = _query_gate(
        visible,
        environment,
        np.zeros(3),
        np.zeros(3),
        object(),
    )
    assert gate is False
    assert info["reason"] == "no_physical_obstacle"
    assert info["obstacle"] is None


def test_behavioral_acceptance_is_not_implied_by_technical_completion() -> None:
    direction_rows = [
        {"policy": "stage2b_dagger", "case": "left", "lateral_action_projection_mps": -0.2},
        {"policy": "stage2b_dagger", "case": "right", "lateral_action_projection_mps": -0.1},
    ]
    aggregate_rows = [
        {"policy": "stage2a_frozen", "group": "normal_test", "safety_mode": "on", "entered_8m": 5},
        {"policy": "stage2b_dagger", "group": "normal_test", "safety_mode": "on", "entered_8m": 5},
        {"policy": "stage2a_frozen", "group": "static_test", "safety_mode": "on", "entered_8m": 0, "entered_3m": 0, "passed_obstacle_episodes": 0},
        {"policy": "stage2b_dagger", "group": "static_test", "safety_mode": "on", "entered_8m": 0, "entered_3m": 0, "passed_obstacle_episodes": 0},
    ]
    result = _stage2b_behavioral_acceptance(
        direction_rows=direction_rows,
        aggregate_rows=aggregate_rows,
    )
    assert result["normal_navigation_preserved"] is True
    assert result["static_holdout_progress"] is False
    assert result["lateral_bypass_signs_correct"] is False
    assert result["behavioral_acceptance_passed"] is False
    assert "static_holdout_no_learned_progress" in result["blockers"]
    assert "left_right_bypass_action_sign_not_learned" in result["blockers"]
