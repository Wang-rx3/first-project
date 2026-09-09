from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.experiments.mainline_baseline_comparison as mainline_comparison
from src.experiments.mainline_baseline_comparison import (
    ACTIVE_LEARNED_METHODS,
    ACTIVE_MAINLINE_METHODS,
    DEPRECATED_METHODS,
    LEARNED_METHODS,
    METHODS,
    MainlineBaselineComparisonConfig,
    PAPER_BRANCH_LABELS,
    PROPOSED_METHOD,
    _aggregate_reward_decomposition_by_episode,
    _check_reward_closure,
)
from src.learning.mappo import MAPPOConfig, MAPPOPolicy


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _observation() -> dict[str, object]:
    return {
        "node_ids": ["uav_00", "uav_01", "candidate:uav_01", "candidate:uav_02"],
        "node_features": np.zeros((4, 37), dtype=np.float32),
        "effective_edge_index": np.asarray([[1, 2, 3], [0, 0, 0]], dtype=np.int64),
        "effective_edge_features": np.zeros((3, 11), dtype=np.float32),
        "candidate_edge_features": np.zeros((2, 11), dtype=np.float32),
        "candidate_incoming_edge_features": np.zeros((2, 11), dtype=np.float32),
        "potential_edge_features": np.zeros((2, 11), dtype=np.float32),
        "action_mask": np.asarray([True, True, True, True], dtype=bool),
        "bandwidth_mask": np.asarray([True, True, True], dtype=bool),
    }


def test_seed42_stage_defaults_and_method_registry() -> None:
    smoke = MainlineBaselineComparisonConfig(stage="smoke")
    development = MainlineBaselineComparisonConfig(stage="development")
    assert smoke.counts == (2, 5, 5, 5)
    assert development.counts == (20, 100, 20, 20)
    assert smoke.resolved_output_dir.name == "mainline_baseline_comparison_smoke_seed42"
    assert development.resolved_output_dir.name == "mainline_baseline_comparison_dev_seed42"
    assert len(METHODS) == 6
    assert set(LEARNED_METHODS) == {
        "baseline_mappo",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        "communication_constrained_proxy_guided_mappo",
        "mappo_mlp",
    }


def test_core_framework_scope_uses_local_ekf_gat_mappo_as_proposed_method() -> None:
    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v1",
        stage="smoke",
    )
    assert PROPOSED_METHOD == "baseline_mappo"
    assert PAPER_BRANCH_LABELS[PROPOSED_METHOD] == "Local-EKF-GAT-MAPPO (Proposed Method)"
    assert "communication_constrained_proxy_guided_mappo" in DEPRECATED_METHODS
    assert "communication_constrained_proxy_guided_mappo" not in ACTIVE_MAINLINE_METHODS
    assert ACTIVE_LEARNED_METHODS == ("baseline_mappo", "mappo_mlp")
    assert config.resolved_output_dir.parent.parent.name == "mainline_core_framework_v1_seed42"
    with pytest.raises(ValueError, match="historical versioned"):
        MainlineBaselineComparisonConfig(protocol="core-framework-v1", stage="replay-audit")


def test_core_framework_v2_uses_candidate_head_and_stable_training_contract() -> None:
    from src.experiments.mainline_corrected_stages import _template

    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v2",
        stage="smoke",
    )
    assert config.resolved_output_dir.parent.parent.name == "mainline_core_framework_v2_seed42"
    formal = _template(
        config,
        "baseline_mappo",
        config.resolved_output_dir / "seed_42" / "baseline_mappo",
    )
    assert formal.measurement_head_mode == "action_conditioned_v2"
    assert formal.learning_reward_mode == "progress_potential_v2"
    assert formal.rollout_episodes_per_update == 4
    assert formal.critic_loss_type == "huber"
    assert formal.value_clip_enabled is True
    assert formal.bootstrap_time_limits is True
    assert formal.gradient_clip_scope == "actor_critic"


def test_action_conditioned_measurement_head_depends_on_candidate_edge_and_keeps_none() -> None:
    import copy
    import torch

    from src.learning.mappo import MAPPOConfig, MAPPOPolicy

    observation = _observation()
    observation.update(
        {
            "candidate_incoming_edge_features": {
                "uav_01": np.zeros(11, dtype=np.float32),
                "uav_02": np.zeros(11, dtype=np.float32),
            },
            "candidate_incoming_sources": ["uav_01", "uav_02"],
            "effective_sources": ["uav_01"],
        }
    )
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
        measurement_head_mode="action_conditioned_v2",
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    with torch.no_grad():
        first = policy._actor_distributions(observation)[0].probs.detach().cpu().numpy()
    changed = copy.deepcopy(observation)
    changed["candidate_incoming_edge_features"]["uav_02"][0] = 4.0
    with torch.no_grad():
        second = policy._actor_distributions(changed)[0].probs.detach().cpu().numpy()
    assert not np.allclose(first, second)

    masked = copy.deepcopy(observation)
    masked["action_mask"] = np.asarray([False, False, False, True], dtype=bool)
    action, _sample = policy.act(masked, deterministic=True)
    assert action["measurement_uav_id"] == "none"


def test_stabilized_v4_is_versioned_and_persistent_only_for_proposed_branch() -> None:
    from src.experiments.mainline_corrected_stages import _template

    config = MainlineBaselineComparisonConfig(
        protocol="stabilized-v4",
        stage="smoke",
    )
    assert config.resolved_output_dir.parent.parent.name == "mainline_baseline_comparison_v4_seed42"
    assert config.resolved_logs_dir.parent.parent.name == "mainline_baseline_comparison_v4_seed42"

    model = {"persistent_ig_q05": 0.0, "persistent_ig_q95": 1.0, "scales": {}}
    proposed = _template(
        config,
        "communication_constrained_proxy_guided_mappo",
        config.resolved_output_dir / "seed_42" / "communication_constrained_proxy_guided_mappo",
        persistent_model=model,
    )
    baseline = _template(
        config,
        "baseline_mappo",
        config.resolved_output_dir / "seed_42" / "baseline_mappo",
    )
    assert proposed.measurement_persistent_proxy_features_enabled is True
    assert proposed.measurement_persistent_proxy_guidance_enabled is True
    assert proposed.measurement_proxy_features_enabled is False
    assert proposed.measurement_proxy_full_packet_cost is True
    assert proposed.measurement_persistent_proxy_horizon == 3
    assert baseline.measurement_persistent_proxy_features_enabled is False
    assert baseline.measurement_proxy_features_enabled is False


def test_stabilized_v4_corrected_has_a_new_root_and_greedy_requires_proxy(tmp_path, monkeypatch) -> None:
    config = MainlineBaselineComparisonConfig(
        protocol="stabilized-v4-corrected",
        stage="smoke",
    )
    assert config.resolved_output_dir.parent.parent.name == "mainline_baseline_comparison_v4_corrected_seed42"

    monkeypatch.setattr(
        mainline_comparison,
        "_fit_measurement_budgets",
        lambda _baseline, uav_count: {f"uav_{index:02d}": 1 for index in range(uav_count)},
    )
    with pytest.raises(RuntimeError, match="requires the baseline-train execution proxy"):
        mainline_comparison._run_rule_method(
            config,
            tmp_path,
            "budgeted_greedy_ig",
            baseline_source=tmp_path / "baseline",
        )


def test_mlp_flat_encoder_preserves_roles_and_is_finite() -> None:
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
        encoder_type="mlp_flat",
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    action, sample, trace = policy.act_with_trace(_observation(), deterministic=True)
    assert action["measurement_uav_id"] in {*UAV_IDS, "none"}
    assert np.isfinite(sample.delta_v).all()
    assert trace["encoder_type"] == "mlp_flat"
    assert trace["layer_count"] == 0
    assert len(trace["flat_presence_mask"]) == 2 * len(UAV_IDS)
    assert trace["attention_maps"] == []


def test_reward_decomposition_closure_is_explicit() -> None:
    rows = [
        {
            "total": -0.25,
            "localization": -0.30,
            "communication": 0.10,
            "collision": -0.05,
            "energy": 0.0,
            "closure_error": 0.0,
        }
    ]
    report = _check_reward_closure(rows)
    assert report["within_tolerance"] is True
    assert report["max_abs_error"] <= 1e-6


def test_reward_decomposition_episode_view_keeps_mean_and_cumulative_scales() -> None:
    rows = [
        {
            "method": "baseline_mappo",
            "episode_id": "episode_000002",
            "total": -0.4,
            "localization": -0.3,
            "communication": 0.1,
            "collision": 0.0,
            "energy": -0.2,
            "closure_error": 0.0,
        },
        {
            "method": "baseline_mappo",
            "episode_id": "episode_000002",
            "total": -0.6,
            "localization": -0.3,
            "communication": 0.1,
            "collision": -0.2,
            "energy": -0.2,
            "closure_error": 0.0,
        },
    ]
    result = _aggregate_reward_decomposition_by_episode(rows)
    assert len(result) == 1
    assert result[0]["base_seed"] == 42
    assert result[0]["window_count"] == 2
    assert result[0]["total_mean_per_window"] == -0.5
    assert result[0]["total_cumulative"] == -1.0
    assert result[0]["collision_cumulative"] == -0.2
