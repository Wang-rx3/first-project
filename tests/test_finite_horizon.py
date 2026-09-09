"""Contracts for raw reward training and finite-horizon critic inputs."""
import numpy as np
import pytest

from src.simulation.environment import MultiUAVJointEnv, SimulationConfig, RewardConfig
from src.learning.mappo import MAPPOConfig, central_state_to_vector


def test_direct_reward_bypasses_collision_and_total_clipping():
    env = MultiUAVJointEnv(SimulationConfig(reward=RewardConfig(
        learning_reward_mode="report_direct", learning_potential_coef=0.0)))
    env.reset(seed=42)
    actions = {u: {"measurement_uav_id": "none", "bandwidth": {}, "delta_v": [0, 0, 0]} for u in env.uav_ids}
    execution = {u: {"delta_v_applied": [0, 0, 0]} for u in env.uav_ids}
    rewards, info = env._rewards(1, actions, execution)
    assert min(rewards.values()) < -1
    assert rewards == info["learning_rewards"] == info["reporting_rewards"]


def test_direct_reward_rejects_feedback():
    with pytest.raises(ValueError, match="forbids"):
        RewardConfig(learning_reward_mode="report_direct", learning_potential_coef=0.25)


def test_warmup_quantile_uncertainty_normalization_uses_frozen_bounds():
    env = MultiUAVJointEnv(SimulationConfig(reward=RewardConfig(
        uncertainty_normalization="warmup_q05_q95",
        metric_scales={
            "uncertainty": 100.0,
            "uncertainty_q05": 0.1,
            "uncertainty_q95": 0.3,
        }
    )))
    env.reset(seed=42)
    actions = {
        u: {"measurement_uav_id": "none", "bandwidth": {}, "delta_v": [0, 0, 0]}
        for u in env.uav_ids
    }
    execution = {u: {"delta_v_applied": [0, 0, 0]} for u in env.uav_ids}
    _rewards, info = env._rewards(0, actions, execution)
    raw = float(info["global_metrics"]["uncertainty"])
    expected = np.clip((raw - 0.1) / 0.2, 0.0, 1.0)
    assert info["global_metrics"]["uncertainty_normalized"] == pytest.approx(expected)
    assert info["global_metrics"]["uncertainty_normalization"] == "warmup_q05_q95"


def test_mappo_max_grad_norm_is_explicit_and_positive():
    assert MAPPOConfig(max_grad_norm=1.0).max_grad_norm == pytest.approx(1.0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        MAPPOConfig(max_grad_norm=0.0)


def test_finite_horizon_contract_and_environment_clock():
    with pytest.raises(ValueError, match="finite_horizon"):
        MAPPOConfig(episode_objective="finite_horizon", include_remaining_time=True,
                    bootstrap_time_limits=True)
    config = MAPPOConfig(episode_objective="finite_horizon", include_remaining_time=True)
    env = MultiUAVJointEnv(SimulationConfig(max_steps=50))
    obs, info = env.reset(seed=42)
    assert all(o["remaining_time_fraction"] == 1 for o in obs.values())
    vector = central_state_to_vector(info["central_state"], uav_count=3, include_remaining_time=True)
    assert len(vector) == config.critic_input_dim
    assert vector[-1] == 1
    assert np.isfinite(vector).all()


def test_remaining_time_is_saved_and_terminal_return_does_not_bootstrap():
    from src.learning.mappo import MAPPOTrainer
    env = MultiUAVJointEnv(SimulationConfig(max_steps=50))
    trainer = MAPPOTrainer.from_environment(env, seed=42,
        episode_objective="finite_horizon", include_remaining_time=True,
        bootstrap_time_limits=False)
    observations, info = env.reset(seed=42)
    decision = trainer.act(observations, info["central_state"], deterministic=True)
    first = next(iter(observations))
    observations[first]["remaining_time_fraction"] = 0.25
    assert decision.observations[first]["remaining_time_fraction"] == 1.0
    for terminal, reward in [(False, -2.0), (True, -10.0)]:
        trainer.add_transition(decision, {u: reward for u in env.uav_ids}, info["central_state"],
            done=True, terminated=terminal, truncated=not terminal)
    records = trainer._training_records()
    assert all(abs(r["return"] + 2.0) < 1e-6 for r in records[:3])
    assert all(abs(r["return"] + 10.0) < 1e-6 for r in records[3:])
