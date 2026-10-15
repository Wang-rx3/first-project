from __future__ import annotations

from dataclasses import replace

import pytest

from src.experiments.single_uav_ppo_a_support import (
    SOURCE_A_DEFAULT,
    SHARED_STATISTICS_DEFAULT,
    capture_policy_distribution,
    cosine_alignment_utility,
    grouped_parameter_deltas,
    initialize_a_foundation,
    load_shared_a_statistics,
    normal_release_gate,
    parameter_snapshot,
    summarize_foundation_episodes,
)
from src.experiments.single_uav_ppo_baseline import (
    create_single_uav_ppo_environment,
    create_single_uav_ppo_trainer,
    load_single_uav_ppo_config,
)
from src.learning.mappo import MAPPOConfig, MAPPOTrainer


def _trainer():
    normalizer, central, _potential, _provenance = load_shared_a_statistics()
    config = MAPPOConfig(
        uav_count=1,
        node_feature_dim=42,
        edge_feature_dim=11,
        gat_hidden_dim=32,
        gat_output_dim=32,
        gat_heads=1,
        gat_layers=2,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        max_obstacles=8,
        max_delta_v=0.6,
        ppo_action_logprob_mode="sampled",
        parameter_initialization_seed=2026,
    )
    return MAPPOTrainer(
        config,
        uav_ids=["uav_00"],
        normalizer=normalizer,
        central_normalizer=central,
        seed=2026,
    )


def test_load_shared_statistics_is_frozen_and_complete():
    normalizer, central, potential, provenance = load_shared_a_statistics(SHARED_STATISTICS_DEFAULT)
    assert normalizer.fitted and central.fitted
    assert normalizer.normalizers["node_features"].feature_dim == 42
    assert central.feature_dim == 174
    assert potential["fit_split"] == "train"
    assert provenance["fit_split"] == "train"
    assert provenance["node_count"] == 306
    assert provenance["feature_index_map"]["obstacle_body_direction"] == [39, 42]
    with pytest.raises(RuntimeError):
        normalizer.fit([])


def test_a_foundation_g_and_p_keep_expected_parameter_objects():
    trainer_g = _trainer()
    ids_g = {name: id(value) for name, value in trainer_g.policy.named_parameters()}
    report_g = initialize_a_foundation(trainer_g, SOURCE_A_DEFAULT, "G")
    assert report_g["loaded_parameter_count"] == 18
    assert report_g["fresh_constructor_parameter_count"] == 11
    assert report_g["first_gat_fresh"]
    assert report_g["source_optimizer_ignored"]
    assert report_g["target_optimizer_state_entries"] == 0
    assert all(id(value) == ids_g[name] for name, value in trainer_g.policy.named_parameters())

    trainer_p = _trainer()
    report_p = initialize_a_foundation(trainer_p, SOURCE_A_DEFAULT, "P")
    assert report_p["loaded_parameter_count"] == 23
    assert report_p["fresh_constructor_parameter_count"] == 6
    assert report_p["first_gat_source_loaded"]
    assert all(value.requires_grad for value in trainer_p.policy.parameters())


def test_a_foundation_real_baseline_factory_wrappers_and_distribution(tmp_path):
    normalizer, central, potential, provenance = load_shared_a_statistics()
    base = load_single_uav_ppo_config("data/configs/single_uav_ppo_baseline_v1.json")

    def build(arm: str):
        config = replace(
            base,
            train_potential_statistics=potential,
            train_normalizer_path=provenance["artifact_path"],
            train_normalizer_provenance=provenance,
            action_log_path=str(tmp_path / f"{arm}.csv"),
        )
        environment = create_single_uav_ppo_environment(config)
        trainer = create_single_uav_ppo_trainer(
            environment,
            config,
            normalizer=normalizer,
            central_normalizer=central,
            normalizer_provenance=provenance,
            action_log_path=config.action_log_path,
        )
        report = initialize_a_foundation(trainer, SOURCE_A_DEFAULT, arm)
        decision = trainer.act(environment._observations(), None)
        capture = capture_policy_distribution(trainer, decision)
        return report, capture, trainer

    report_g, capture_g, trainer_g = build("G")
    report_p, capture_p, trainer_p = build("P")
    assert report_g["loaded_parameter_count"] == 18
    assert report_p["loaded_parameter_count"] == 23
    assert capture_g["uavs"]["uav_00"]["raw_std"]
    assert capture_p["uavs"]["uav_00"]["bounded_mean_units"] == "m/s"
    assert len(trainer_g.rollout) == 0 and len(trainer_p.rollout) == 0


def test_foundation_rejects_rollout_and_optimizer_state():
    trainer = _trainer()
    trainer.rollout.append(object())
    with pytest.raises(RuntimeError, match="non-empty rollout"):
        initialize_a_foundation(trainer, SOURCE_A_DEFAULT, "G")


def test_parameter_groups_and_cosine_are_finite():
    trainer = _trainer()
    before = parameter_snapshot(trainer.policy)
    deltas = grouped_parameter_deltas(before, trainer.policy)
    assert deltas["total_actor"]["tensor_count"] == 23
    assert deltas["critic"]["tensor_count"] == 6
    assert cosine_alignment_utility([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine_alignment_utility([0, 0], [1, 0]) == 0.0


def test_summary_and_release_gate_include_partial_episode_counts():
    rows = []
    for index in range(8):
        rows.append(
            {
                "kind": "normal",
                "steps": 120,
                "task_success": index < 5,
                "collision": False,
                "physical_contact_steps": 0,
                "timeout_failure": False,
                "rollout_window_cut": index == 7,
                "boundary_attempt_steps": 0,
                "boundary_projection_steps": 0,
                "safety_projection_steps": 0,
                "raw_action_danger_steps": 0,
                "final_actual_goal_distance_m": 1.0,
            }
        )
    metrics = summarize_foundation_episodes(rows)
    assert metrics["all"]["success_stable_count"] == 5
    assert metrics["all"]["rollout_window_cut_count"] == 1
    gate = normal_release_gate({**metrics["all"], "episode_count": 6, "success_stable_count": 4}, metrics["all"])
    assert gate["checks"]["validation_success_at_least_5_of_8"]
    assert gate["validation_selection_performed"] is False


def test_summary_uses_single_full_physical_bypass_and_none_for_normal():
    rows = [
        {
            "kind": "normal",
            "steps": 120,
            "task_success": True,
            "collision": False,
            "physical_bypass": None,
            "raw_action_physical_denominator": 0,
            "rollout_window_cut": False,
            "truncated": True,
            "final_actual_goal_distance_m": 0.4,
        },
        {
            "kind": "single",
            "steps": 120,
            "task_success": True,
            "collision": False,
            "physical_bypass": True,
            "raw_action_danger_steps": 2,
            "raw_action_physical_denominator": 120,
            "rollout_window_cut": False,
            "truncated": True,
            "final_actual_goal_distance_m": 0.8,
        },
        {
            "kind": "full",
            "steps": 120,
            "task_success": False,
            "collision": False,
            "physical_bypass": False,
            "raw_action_danger_steps": 1,
            "raw_action_physical_denominator": 120,
            "rollout_window_cut": False,
            "truncated": True,
            "final_actual_goal_distance_m": 3.0,
        },
    ]
    metrics = summarize_foundation_episodes(rows)
    assert metrics["normal"]["physical_bypass_rate"] is None
    assert metrics["normal"]["raw_action_danger_fraction"] is None
    assert metrics["all"]["physical_bypass_episodes"] == 2
    assert metrics["all"]["physical_bypasses"] == 1
    assert metrics["all"]["physical_bypass_rate"] == pytest.approx(0.5)
    assert metrics["all"]["complete_episode_count"] == 3
