"""Unit checks for the read-only two-group smoke initializer."""

from pathlib import Path

import pytest
import torch

from src.experiments.single_uav_ppo_smoke_initialization import (
    b_pretraining_cost,
    initialize_smoke_trainer,
)
from src.learning.mappo import MAPPOConfig, MAPPOTrainer
from src.learning.normalization import FeatureNormalizer, ObservationNormalizer


def _trainer() -> MAPPOTrainer:
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
    )
    return MAPPOTrainer(
        config,
        uav_ids=["uav_00"],
        normalizer=ObservationNormalizer(),
        central_normalizer=FeatureNormalizer(),
    )


def _checkpoint(path: Path, *, bc: bool = False, mismatch_actor: bool = False) -> None:
    source = _trainer()
    state = {name: value.detach().cpu().clone() for name, value in source.policy.state_dict().items()}
    for name in state:
        if name.startswith("actor_body."):
            state[name].fill_(0.25)
    if mismatch_actor:
        state["actor_body.0.weight"] = torch.zeros((1, 1), dtype=torch.float32)
    payload = {
        "uav_count": 1,
        "update_count": 100,
        "trainer_config": {
            "uav_count": 1,
            "node_feature_dim": 42,
            "edge_feature_dim": 11,
            "ppo_action_logprob_mode": "executed",
        },
        "policy_state_dict": state,
        "optimizer_state_dict": {"state": {"stale": torch.tensor(1)}, "param_groups": []},
        "normalization_stats": {
            "normalizers": {"node_features": {"mean": [0.0] * 42, "count": [0] * 42}}
        },
        "central_normalization_stats": {"mean": [0.0] * 174},
    }
    if bc:
        payload["bc_training"] = {
            "critic_frozen": True,
            "batch_size": 32,
            "actual_epochs": 45,
            "train_samples": 1728,
            "validation_samples": 384,
        }
    torch.save(payload, path)


def test_a_loads_actor_only_and_keeps_fresh_first_gat_and_critic(tmp_path: Path):
    source = tmp_path / "a.pt"
    _checkpoint(source)
    trainer = _trainer()
    fresh = {name: value.detach().clone() for name, value in trainer.policy.state_dict().items()}
    report = initialize_smoke_trainer(trainer, source, "A")
    assert report["initialized_from_checkpoint"]
    assert report["actor_body_load_verified"]
    assert report["source_optimizer_ignored"]
    assert report["teacher_online_samples"] == 0
    assert report["normalization"]["normalizer_reused"] is False
    assert report["parameters"]["fresh_constructor_verification"] == {
        "gat.layers.0": True,
        "critic": True,
    }
    for name, value in trainer.policy.state_dict().items():
        if name.startswith("gat.layers.0.") or name.startswith("critic."):
            assert torch.equal(value, fresh[name])
    assert len(trainer.rollout) == 0
    assert len(trainer.optimizer.state) == 0


def test_b_requires_bc_lineage_and_reports_derived_cost(tmp_path: Path):
    source = tmp_path / "b.pt"
    _checkpoint(source, bc=True)
    trainer = _trainer()
    report = initialize_smoke_trainer(trainer, source, "B")
    assert report["source"]["lineage"] == "behavior_cloning_dagger"
    cost = b_pretraining_cost(source)
    assert cost["derived"]["train_example_presentations"] == 77760
    assert cost["derived"]["train_batches"] == 2430
    assert cost["teacher_rollout_total"] is None


def test_incompatible_actor_module_fails_before_partial_load(tmp_path: Path):
    source = tmp_path / "bad.pt"
    _checkpoint(source, mismatch_actor=True)
    trainer = _trainer()
    before = {name: value.detach().clone() for name, value in trainer.policy.state_dict().items()}
    with pytest.raises(ValueError, match="actor_body"):
        initialize_smoke_trainer(trainer, source, "A")
    for name, value in trainer.policy.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonempty_target_optimizer_is_rejected(tmp_path: Path):
    source = tmp_path / "a.pt"
    _checkpoint(source)
    trainer = _trainer()
    parameter = next(trainer.policy.parameters())
    trainer.optimizer.state[parameter] = {"step": torch.tensor(1)}
    with pytest.raises(RuntimeError, match="optimizer state"):
        initialize_smoke_trainer(trainer, source, "A")
