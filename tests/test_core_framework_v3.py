from __future__ import annotations

import copy

import numpy as np

from src.experiments.mainline_baseline_comparison import (
    MainlineBaselineComparisonConfig,
    PROPOSED_METHOD,
)
from src.experiments.mainline_corrected_stages import (
    _learners_for,
    _methods_for,
    _template,
)
from src.learning.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    PolicyDecision,
)


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _observation() -> dict[str, object]:
    return {
        "node_ids": ["uav_00", "uav_01", "candidate:uav_01", "candidate:uav_02"],
        "node_features": np.zeros((4, 37), dtype=np.float32),
        "effective_edge_index": np.asarray(
            [[1, 2, 3], [0, 0, 0]], dtype=np.int64
        ),
        "effective_edge_features": np.zeros((3, 11), dtype=np.float32),
        "candidate_edge_features": np.zeros((2, 11), dtype=np.float32),
        "candidate_incoming_edge_features": np.zeros((2, 11), dtype=np.float32),
        "potential_edge_features": np.zeros((2, 11), dtype=np.float32),
        "action_mask": np.asarray([True, True, True, True], dtype=bool),
        "bandwidth_mask": np.asarray([True, True, True], dtype=bool),
    }


def _small_config(*, mode: str, actor_scale: float = 1.0) -> MAPPOConfig:
    return MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        gat_hidden_dim=8,
        gat_output_dim=8,
        gat_heads=1,
        gat_layers=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        parameter_initialization_seed=42,
        measurement_head_mode="action_conditioned_v2",
        ppo_action_logprob_mode=mode,
        actor_learning_rate_scale=actor_scale,
    )


def test_core_framework_v3_is_proposed_only_and_uses_sampled_mode(tmp_path) -> None:
    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v3",
        stage="smoke",
        output_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        plots_dir=tmp_path / "plots",
        figure_profile="compact-proposed-v3",
    )
    assert _methods_for(config) == (
        "local_only_fixed",
        "random_cooperative",
        "budgeted_greedy_ig",
        PROPOSED_METHOD,
    )
    assert _learners_for(config) == (PROPOSED_METHOD,)
    formal = _template(config, PROPOSED_METHOD, tmp_path / "data" / "seed_42")
    assert formal.ppo_action_logprob_mode == "sampled"
    assert formal.rollout_episodes_per_update == 4
    assert formal.gradient_clip_scope == "actor_critic"
    assert formal.measurement_head_mode == "action_conditioned_v2"
    assert formal.learning_reward_mode == "progress_potential_v2"


def test_sampled_log_probability_survives_execution_projection() -> None:
    observation = _observation()
    sampled_trainer = MAPPOTrainer(_small_config(mode="sampled"), uav_ids=UAV_IDS, seed=42)
    action, sample = sampled_trainer.policy.act(observation, deterministic=False)
    decision = PolicyDecision(
        actions={"uav_00": action},
        samples={"uav_00": sample},
        observations={"uav_00": copy.deepcopy(observation)},
        central_vector=np.zeros(sampled_trainer.policy.config.critic_input_dim, dtype=np.float32),
        value=0.0,
    )
    old_log_prob = sample.old_log_prob
    sampled_trainer.reconcile_action_execution(
        decision,
        {
            "actions": {
                "uav_00": {
                    "measurement_uav_id": "none",
                    "bandwidth": {},
                    "delta_v": [0.0, 0.0, 0.0],
                }
            },
            "applied_bandwidth": {"uav_00": {}},
            "action_execution": {
                "uav_00": {"delta_v_applied": [0.0, 0.0, 0.0]}
            },
        },
    )
    assert sample.old_log_prob == old_log_prob
    assert decision.measurement_diagnostics["uav_00"]["action_execution_changed"] == 1
    assert sample.executed_measurement_index == 3


def test_executed_mode_remains_compatible_and_scaled_optimizer_has_two_groups() -> None:
    observation = _observation()
    legacy_trainer = MAPPOTrainer(_small_config(mode="executed"), uav_ids=UAV_IDS, seed=42)
    action, sample = legacy_trainer.policy.act(observation, deterministic=False)
    decision = PolicyDecision(
        actions={"uav_00": action},
        samples={"uav_00": sample},
        observations={"uav_00": copy.deepcopy(observation)},
        central_vector=np.zeros(legacy_trainer.policy.config.critic_input_dim, dtype=np.float32),
        value=0.0,
    )
    old_log_prob = sample.old_log_prob
    legacy_trainer.reconcile_action_execution(
        decision,
        {
            "actions": {
                "uav_00": {
                    "measurement_uav_id": "none",
                    "bandwidth": {},
                    "delta_v": [0.0, 0.0, 0.0],
                }
            },
            "applied_bandwidth": {"uav_00": {}},
            "action_execution": {
                "uav_00": {"delta_v_applied": [0.0, 0.0, 0.0]}
            },
        },
    )
    assert sample.old_log_prob != old_log_prob or sample.measurement_index == 3
    assert len(legacy_trainer.optimizer.param_groups) == 1
    scaled_trainer = MAPPOTrainer(
        _small_config(mode="sampled", actor_scale=0.5), uav_ids=UAV_IDS, seed=42
    )
    assert len(scaled_trainer.optimizer.param_groups) == 2
    actor_ids = {
        id(parameter)
        for parameter in scaled_trainer.optimizer.param_groups[0]["params"]
    }
    critic_ids = {
        id(parameter)
        for parameter in scaled_trainer.optimizer.param_groups[1]["params"]
    }
    assert actor_ids.isdisjoint(critic_ids)
