"""Contracts for the v9 candidate-action and accepted-update diagnostics."""

from __future__ import annotations

import numpy as np

from src.experiments.mainline_baseline_comparison import MainlineBaselineComparisonConfig
from src.experiments.mainline_corrected_stages import _template
from src.learning.mappo import MAPPOConfig, MAPPOPolicy
from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _observation() -> dict[str, object]:
    return {
        "node_ids": ["uav_00", "uav_01", "candidate:uav_01", "candidate:uav_02"],
        "node_features": np.zeros((4, 37), dtype=np.float32),
        "effective_edge_index": np.asarray([[1, 2, 3], [0, 0, 0]], dtype=np.int64),
        "effective_edge_features": np.zeros((3, 11), dtype=np.float32),
        "candidate_edge_features": np.zeros((2, 11), dtype=np.float32),
        "candidate_incoming_edge_features": {
            "uav_01": np.zeros(11, dtype=np.float32),
            "uav_02": np.zeros(11, dtype=np.float32),
        },
        "candidate_incoming_sources": ["uav_01", "uav_02"],
        "effective_sources": ["uav_01"],
        "action_mask": np.asarray([True, True, True, True], dtype=bool),
        "bandwidth_mask": np.asarray([True, True, True], dtype=bool),
    }


def test_v9_template_separates_control_head_and_feedback_arms(tmp_path) -> None:
    for arm, head, reward_mode, feedback_mode in (
        ("action-selection-control", "action_conditioned_v2", "report_direct", "v3"),
        ("action-head-v3", "action_conditioned_v3", "report_direct", "v3"),
        (
            "action-linked-feedback-v2",
            "action_conditioned_v3",
            "report_plus_feedback",
            "accepted_update_progress_v2",
        ),
    ):
        config = MainlineBaselineComparisonConfig(
            protocol="core-framework-v9-action-selection",
            stage="smoke",
            experiment_arm=arm,
        )
        formal = _template(config, "baseline_mappo", tmp_path / arm)
        assert formal.measurement_head_mode == head
        assert formal.learning_reward_mode == reward_mode
        assert formal.learning_feedback_mode == feedback_mode
        assert formal.ppo_action_logprob_mode == "sampled"
        assert formal.rollout_episodes_per_update == 4


def test_action_conditioned_v3_copies_candidate_trace_into_each_policy_sample() -> None:
    config = MAPPOConfig(
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
        measurement_head_mode="action_conditioned_v3",
        ppo_action_logprob_mode="sampled",
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    _action, sample = policy.act(_observation(), deterministic=True)
    assert sample.candidate_source_ids == list(UAV_IDS)
    assert sample.candidate_visible.shape == (3,)
    assert sample.candidate_action_mask.shape == (4,)
    assert sample.candidate_action_features.shape == (3, 12)
    assert sample.candidate_quality_scores.shape == (3,)
    assert sample.none_context_summary.shape == (4,)
    assert np.isfinite(sample.candidate_action_features).all()
    assert np.isfinite(sample.candidate_quality_scores).all()


def test_accepted_update_progress_v2_is_relative_to_warmup_none_baseline() -> None:
    env = MultiUAVJointEnv(
        SimulationConfig(
            uav_count=3,
            max_steps=10,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_reward_mode="report_plus_feedback",
                learning_feedback_mode="accepted_update_progress_v2",
                accepted_update_feedback_coef=0.1,
                accepted_update_feedback_delay_steps=0,
                accepted_update_feedback_location=0.2,
                accepted_update_feedback_scale=0.1,
                accepted_update_feedback_none_baseline=0.1,
                learning_potential_coef=0.0,
            ),
        )
    )
    env.reset(seed=2026)
    actions = {
        uav_id: {
            "measurement_uav_id": "uav_01" if uav_id == "uav_00" else "none",
            "bandwidth": {},
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in env.uav_ids
    }
    execution = {
        uav_id: {"delta_v_applied": [0.0, 0.0, 0.0]}
        for uav_id in env.uav_ids
    }
    env.relative_range_update_events = [
        {
            "target_uav_id": "uav_00",
            "source_uav_id": "uav_01",
            "accepted": True,
            "position_covariance_trace_before_m2": 3.0,
            "position_covariance_trace_after_linear_m2": 2.0,
            "local_direction_variance_m2": 1.0,
            "effective_measurement_variance_m2": 3.0,
            "nis": 0.0,
        }
    ]
    _report, info = env._rewards(0, actions, execution)
    records = info["learning_reward_components"]["accepted_update_feedback_records"]
    assert len(records) == 1
    record = records[0]
    assert np.isclose(float(record["none_baseline_quality"]), 0.1)
    assert np.isclose(float(record["feedback_difference_raw"]), float(record["quality"]) - 0.1)
    assert float(record["feedback_scale"]) == 0.1
    assert np.isfinite(float(record["feedback_value"]))
