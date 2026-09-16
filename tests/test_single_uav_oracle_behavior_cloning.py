import numpy as np
import torch

from src.experiments.single_uav_oracle_behavior_cloning import (
    actor_mean_trainable_parameter_names,
    actor_physical_mean,
    _restore_observation,
)


class _FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = torch.nn.Linear(1, 1)
        self.actor_body = torch.nn.Linear(1, 1)
        self.delta_mean_head = torch.nn.Linear(1, 3)
        self.measurement_head = torch.nn.Linear(1, 1)
        self.critic = torch.nn.Linear(1, 1)
        self.delta_logstd = torch.nn.Parameter(torch.zeros(3))

    def _actor_distributions(self, observation):
        del observation
        base = type("Base", (), {"mean": torch.tensor([0.0, 1.0, -1.0]), "scale": torch.ones(3)})()
        distribution = type("Distribution", (), {"base_dist": base})()
        return None, None, distribution, None

    def _delta_component_limit(self):
        return 0.6 / np.sqrt(3.0)


def test_actor_mean_trainability_excludes_critic_heads_and_logstd():
    policy = _FakePolicy()
    names = actor_mean_trainable_parameter_names(policy)
    assert names
    assert policy.critic.weight.requires_grad is False
    assert policy.measurement_head.weight.requires_grad is False
    assert policy.delta_logstd.requires_grad is False
    assert policy.gat.weight.requires_grad is True
    assert policy.actor_body.weight.requires_grad is True
    assert any(name.startswith("delta_mean_head.") for name in names)
    assert not policy.delta_logstd.requires_grad
    assert not policy.critic.weight.requires_grad


def test_physical_mean_uses_post_tanh_scaled_action_space():
    policy = _FakePolicy()
    result = actor_physical_mean(policy, {"node_features": np.zeros((1, 39), dtype=np.float32)})
    expected = torch.tanh(torch.tensor([0.0, 1.0, -1.0])) * (0.6 / np.sqrt(3.0))
    assert torch.allclose(result, expected)
    assert float(torch.linalg.vector_norm(result)) <= 0.6 + 1.0e-7


def test_restore_observation_restores_feature_arrays_without_changing_metadata():
    restored = _restore_observation(
        {
            "node_features": [[1.0, 2.0]],
            "effective_edge_features": [],
            "node_ids": ["uav_00"],
            "uav_id": "uav_00",
        }
    )
    assert isinstance(restored["node_features"], np.ndarray)
    assert restored["node_features"].dtype == np.float32
    assert restored["node_ids"] == ["uav_00"]
    assert restored["uav_id"] == "uav_00"
