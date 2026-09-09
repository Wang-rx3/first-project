"""Regression tests for the opt-in stabilized-v3 training path."""

from __future__ import annotations

import numpy as np


def _zero_actions(environment):
    return {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {},
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in environment.uav_ids
    }


def test_bounded_learning_reward_keeps_reporting_reward_unchanged():
    from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(
            max_steps=1,
            gnss_mode="denied",
            reward=RewardConfig(learning_reward_mode="bounded_potential"),
        )
    )
    environment.reset(seed=42)
    _observations, reporting, _terminated, _truncated, info = environment.step(
        _zero_actions(environment)
    )

    assert info["reporting_rewards"] == reporting
    assert info["learning_reward_mode"] == "bounded_potential"
    learning = info["learning_rewards"]
    assert set(learning) == set(reporting)
    assert all(np.isfinite(float(value)) and abs(float(value)) <= 1.0 for value in learning.values())
    components = info["learning_reward_components"]
    assert np.isfinite(float(components["potential_shaping"]))
    assert float(components["collision_cost_learning"]) <= 1.0 + 1e-12


def test_time_limit_bootstrap_does_not_connect_gae_across_episodes():
    from src.learning.mappo import MAPPOTrainer
    from src.simulation.environment import MultiUAVJointEnv, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(max_steps=1, gnss_mode="denied")
    )
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=42,
        policy_epochs=1,
        minibatch_size=4,
        bootstrap_time_limits=False,
    )
    observations, info = environment.reset(seed=42)
    decision = trainer.act(observations, info["central_state"], deterministic=True)
    rewards = [{uav_id: 1.0 for uav_id in environment.uav_ids},
               {uav_id: 100.0 for uav_id in environment.uav_ids}]
    for reward in rewards:
        trainer.add_transition(
            decision,
            reward,
            info["central_state"],
            done=True,
            terminated=False,
            truncated=True,
        )

    records = trainer._training_records()
    assert all(abs(float(row["return"]) - 1.0) < 1e-6 for row in records[:3])
    assert all(abs(float(row["return"]) - 100.0) < 1e-6 for row in records[3:])


def test_stabilized_ppo_update_is_finite():
    from src.learning.mappo import MAPPOTrainer
    from src.simulation.environment import MultiUAVJointEnv, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(max_steps=2, gnss_mode="denied")
    )
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=42,
        gat_hidden_dim=8,
        gat_output_dim=8,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        policy_epochs=1,
        minibatch_size=4,
        critic_loss_type="huber",
        value_clip_enabled=True,
        value_clip_epsilon=0.2,
        target_kl=1.0,
        bootstrap_time_limits=True,
    )
    trainer.collect_rollout(environment, horizon=2, seed=42)
    stats = trainer.update()
    assert all(np.isfinite(float(value)) for value in stats.values())
    assert stats["value_clip_enabled"] == 1.0
    assert stats["bootstrap_time_limits"] == 1.0

