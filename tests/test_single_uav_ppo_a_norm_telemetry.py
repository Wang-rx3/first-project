from types import SimpleNamespace

import numpy as np

from src.experiments.single_uav_ppo_a_norm_telemetry import motion_candidate
from src.simulation.motion_model import MotionConfig, UAVKinematicState, apply_velocity_increment
from src.analysis.single_uav_ppo_a_norm_history import single_node_observation
from src.learning.mappo import MAPPOConfig, MAPPOTrainer


def test_candidate_matches_unprojected_velocity_increment():
    motion = MotionConfig(dt_s=.2, max_speed_mps=12, max_acceleration_mps2=3)
    state = UAVKinematicState(np.array([30., 50., 20.]), np.array([.3, -.1, .2]), np.zeros(3), np.zeros(3), np.zeros(3))
    env = SimpleNamespace(uav_ids=['uav_00'], states={'uav_00': state}, config=SimpleNamespace(motion=motion))
    delta = [.12, .04, -.1]
    candidate = motion_candidate(env, delta)
    actual, info = apply_velocity_increment(state, delta, motion)
    np.testing.assert_allclose(candidate['candidate_position_m'], actual.position_m)
    np.testing.assert_allclose(candidate['candidate_velocity_mps'], actual.velocity_mps)
    assert not any(candidate['boundary_attempt_axes'])
    assert not info['boundary_projected']


def test_boundary_reconciles_tangential_velocity_to_displacement():
    motion = MotionConfig(dt_s=.2, max_speed_mps=12, max_acceleration_mps2=3)
    state = UAVKinematicState(np.array([199.95, 50., 20.]), np.array([1.2, .15, 0.]), np.zeros(3), np.zeros(3), np.zeros(3))
    actual, info = apply_velocity_increment(state, [.2, .1, 0.], motion)
    assert info['boundary_projected']
    np.testing.assert_allclose(actual.velocity_mps, [.25, .20, 0.], atol=1e-8)
    np.testing.assert_allclose(actual.velocity_mps, (actual.position_m-state.position_m)/.2)
    assert np.linalg.norm(actual.acceleration_mps2) > 3
    # The projection contract preserves displacement consistency; it is not
    # an actuator-only acceleration bound on the reset impulse.


def test_replay_empty_graph_matches_policy_input_api():
    trainer = MAPPOTrainer(MAPPOConfig(uav_count=1, node_feature_dim=42, max_obstacles=8), seed=2026)
    local = single_node_observation(np.zeros(42))
    context, *rest = trainer.policy._local_context(local)
    assert context.numel() == trainer.config.gat_output_dim + 47
    assert tuple(local['effective_edge_features'].shape) == (0, 11)
