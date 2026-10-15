"""Pure contract checks for fixed-navigation communication MAPPO mode."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.control.fixed_navigation_execution import compose_fixed_navigation_actions
from src.learning.mappo import MAPPOConfig, MAPPOPolicy, MAPPOTrainer, PolicyDecision


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _observation(uid: str = "uav_00") -> dict[str, object]:
    return {
        "uav_id": uid,
        "estimate": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "goal": np.asarray([1.0, 0.0, 0.0]),
        "node_ids": [uid, "candidate:uav_01", "candidate:uav_02"],
        "node_features": np.zeros((3, 37), dtype=np.float32),
        "effective_edge_index": np.empty((2, 0), dtype=np.int64),
        "effective_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_incoming_edge_features": {},
        "potential_edge_features": np.empty((0, 11), dtype=np.float32),
        "action_mask": np.asarray([True, True, True, True]),
        "bandwidth_mask": np.asarray([True, True, True]),
        "potential_targets": ["uav_01", "uav_02"],
    }


def _config(**kwargs: object) -> MAPPOConfig:
    values = dict(
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
        motion_action_mode="fixed_navigation_comm_v1",
        ppo_action_logprob_mode="sampled",
    )
    values.update(kwargs)
    return MAPPOConfig(**values)


def test_composer_uses_only_raw_estimate_and_goal_and_rejects_motion_override() -> None:
    raw = {uid: _observation(uid) for uid in UAV_IDS}
    communication = {
        uid: {"measurement_uav_id": "none", "bandwidth": {"uav_01": 0.0}}
        for uid in UAV_IDS
    }
    actions, diagnostics = compose_fixed_navigation_actions(raw, communication)
    assert tuple(actions) == UAV_IDS
    assert np.allclose(actions["uav_00"]["delta_v"], [0.192, 0.0, 0.0])
    assert np.allclose(diagnostics["uav_00"]["requested_delta_v"], [0.192, 0.0, 0.0])
    with pytest.raises(ValueError, match="must not contain learned delta_v"):
        compose_fixed_navigation_actions(
            raw,
            {uid: {"measurement_uav_id": "none", "bandwidth": {}, "delta_v": [0, 0, 0]} for uid in UAV_IDS},
        )


def test_fixed_policy_masks_self_and_evaluates_stored_simplex_without_motion() -> None:
    observation = _observation()
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    action, sample = policy.act(observation, deterministic=True)
    assert "delta_v" not in action
    assert sample.delta_raw.size == 0 and sample.delta_v.size == 0
    assert sample.link_selection.shape == (3,)
    assert not bool(sample.link_selection[0])
    log_prob, entropy = policy.evaluate_actions(observation, sample)
    assert np.isfinite(float(log_prob.detach()))
    assert np.isfinite(float(entropy.detach()))
    assert np.isclose(float(log_prob.detach()), sample.old_log_prob, atol=1e-5)


def test_fixed_config_requires_sampled_mode_and_frozen_motion_limit() -> None:
    with pytest.raises(ValueError, match="sampled"):
        _config(ppo_action_logprob_mode="executed")
    with pytest.raises(ValueError, match="max_delta_v"):
        _config(max_delta_v=0.5)


def test_fixed_reconciliation_preserves_policy_sample_and_separates_navigation_projection() -> None:
    observations = {uid: _observation(uid) for uid in UAV_IDS}
    trainer = MAPPOTrainer(_config(), uav_ids=UAV_IDS, seed=42)
    decision = trainer.act(observations, None, deterministic=True)
    original = copy.deepcopy(decision.samples["uav_00"])
    trainer.reconcile_action_execution(
        decision,
        {
            "actions": {
                "uav_00": {
                    "measurement_uav_id": "none",
                    "bandwidth": {},
                    "delta_v": [0.0, 0.0, 0.0],
                }
            },
            "applied_bandwidth": {"uav_00": {"uav_01": 0.25}},
            "action_execution": {"uav_00": {"delta_v_applied": [0.0, 0.0, 0.0]}},
        },
    )
    sample = decision.samples["uav_00"]
    assert sample.old_log_prob == original.old_log_prob
    assert sample.delta_v.size == 0
    assert decision.measurement_diagnostics["uav_00"]["action_execution_changed"] == 1
    assert decision.measurement_diagnostics["uav_00"]["nav_projection_changed"] == 1


@pytest.mark.parametrize("active_count", [0, 1, 2])
def test_active_link_simplex_and_probability_edge_cases(active_count) -> None:
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    with torch.no_grad():
        policy.link_selection_head.weight.zero_()
        policy.link_selection_head.bias.copy_(torch.tensor([-20., 20. if active_count >= 1 else -20., 20. if active_count >= 2 else -20.]))
    action, sample = policy.act(_observation(), deterministic=True)
    assert sum(sample.link_selection) == active_count
    assert not sample.link_selection[0] and sample.sampled_link_probabilities[0] == 0
    assert "uav_00" not in action['bandwidth']
    assert np.sum(sample.bandwidth) == pytest.approx(float(active_count > 0))
    assert np.all(sample.bandwidth[~sample.link_selection] == 0)
    lp, entropy = policy.evaluate_actions(_observation(), sample)
    assert float(lp.detach()) == pytest.approx(sample.old_log_prob, abs=2e-6)
    assert torch.isfinite(entropy)
    original = sample.bandwidth.copy()
    policy.evaluate_actions(_observation(), sample)
    assert np.array_equal(original, sample.bandwidth)
    if active_count == 1:
        corrupted = copy.deepcopy(sample)
        corrupted.bandwidth[2] = .2
        with pytest.raises(ValueError, match='inactive'):
            policy.evaluate_actions(_observation(), corrupted)


def test_measurement_incoming_mask_is_not_an_outgoing_recipient_mask() -> None:
    local = _observation()
    local['action_mask'] = np.array([False, True, False, True])
    local['potential_targets'] = ['uav_02']
    local['bandwidth_mask'] = np.array([False, False, True])
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    measurement, _, motion, _ = policy._actor_distributions(local)
    assert float(measurement.probs[1].detach()) > 0 and float(measurement.probs[0].detach()) == 0 and float(measurement.probs[2].detach()) == 0
    assert motion is None


def test_empty_legal_links_none_measurement_and_normal_not_constructed(monkeypatch) -> None:
    import src.learning.mappo as module
    def no_normal(*args, **kwargs):
        raise AssertionError('A fixed navigation policy must not build a Normal')
    monkeypatch.setattr(module, 'Normal', no_normal)
    local = _observation()
    local['action_mask'] = np.zeros(4, dtype=bool)
    local['bandwidth_mask'] = np.zeros(3, dtype=bool)
    local['potential_targets'] = []
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    action, sample = policy.act(local)
    assert action['measurement_uav_id'] == 'none' and action['bandwidth'] == {}
    assert not sample.link_selection.any() and np.all(sample.bandwidth == 0)
    lp, entropy = policy.evaluate_actions(local, sample)
    assert float(lp) == pytest.approx(0) and float(entropy) == pytest.approx(0)
    sample.delta_v = np.zeros(3)
    with pytest.raises(ValueError, match='learned motion'):
        policy.evaluate_actions(local, sample)


def test_legacy_checkpoint_schema_cannot_silently_load_new_link_head() -> None:
    legacy = MAPPOPolicy(_config(motion_action_mode='joint'), uav_ids=UAV_IDS)
    fixed = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    assert set(fixed.state_dict()) - set(legacy.state_dict()) == {'link_selection_head.weight', 'link_selection_head.bias'}
    with pytest.raises(RuntimeError, match='link_selection_head'):
        fixed.load_state_dict(legacy.state_dict(), strict=True)


def test_offline_ppo_gradient_routes_to_communication_and_gat_without_weight_update() -> None:
    local = _observation()
    local['node_ids'] = list(UAV_IDS)
    local['node_features'] = np.arange(111, dtype=np.float32).reshape(3, 37) / 500
    local['effective_edge_index'] = np.array([[1, 2], [0, 0]], dtype=np.int64)
    local['effective_edge_features'] = np.ones((2, 11), dtype=np.float32) * .01
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    _, sample = policy.act(local)
    lp, entropy = policy.evaluate_actions(local, sample)
    ratio = torch.exp(lp - torch.as_tensor(sample.old_log_prob))
    loss = -torch.minimum(ratio * .7, ratio.clamp(.8, 1.2) * .7) - .005 * entropy
    policy.zero_grad(set_to_none=True)
    loss.backward()
    for prefix in ('gat.', 'measurement_head.', 'link_selection_head.'):
        assert sum(float((parameter.grad.detach() ** 2).sum()) for name, parameter in policy.named_parameters()
                   if name.startswith(prefix) and parameter.grad is not None) > 0
    assert all(parameter.grad is None and not parameter.requires_grad for name, parameter in policy.named_parameters()
               if name.startswith('delta_mean_head.') or name == 'delta_logstd')
    assert all(torch.equal(before[name], tensor) for name, tensor in policy.state_dict().items())
