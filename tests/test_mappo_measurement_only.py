"""Pure contracts for the measurement-only fixed-navigation MAPPO scope.

These tests deliberately stay above the environment and optimizer lifecycle:
they exercise the policy distributions, central-state encoding, gradients, and
offline GAE records only.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import src.learning.mappo as mappo_module
from src.learning.mappo import (
    MAPPOConfig,
    MAPPOPolicy,
    MAPPOTrainer,
    PolicyDecision,
    PolicySample,
    RolloutTransition,
    central_state_to_vector,
    central_state_vector_dim,
)


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _config(**overrides: object) -> MAPPOConfig:
    values: dict[str, object] = {
        "uav_count": 3,
        "node_feature_dim": 37,
        "edge_feature_dim": 11,
        "gat_hidden_dim": 8,
        "gat_output_dim": 8,
        "gat_heads": 1,
        "gat_layers": 1,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
        "parameter_initialization_seed": 2026116200,
        "motion_action_mode": "fixed_navigation_comm_v1",
        "ppo_action_logprob_mode": "sampled",
        "communication_learning_scope": "measurement_only",
    }
    values.update(overrides)
    return MAPPOConfig(**values)


def _observation(uid: str = "uav_00", *, v3: bool = False) -> dict[str, object]:
    other_ids = [other for other in UAV_IDS if other != uid]
    node_ids = [uid, *other_ids] if v3 else [uid, *(f"candidate:{other}" for other in other_ids)]
    node_features = np.zeros((len(node_ids), 37), dtype=np.float32)
    covariance = np.eye(6, dtype=np.float64)
    messages = {
        other: {"estimate": np.zeros(6), "covariance": (2.0 + index) * covariance}
        for index, other in enumerate(other_ids)
    }
    observation: dict[str, object] = {
        "uav_id": uid,
        "estimate": np.zeros(6, dtype=np.float64),
        "covariance": covariance.copy(),
        "goal": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "node_ids": node_ids,
        "node_features": node_features,
        "effective_edge_index": np.empty((2, 0), dtype=np.int64),
        "effective_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_edge_features": {
            other: np.zeros(11, dtype=np.float32) for other in other_ids
        },
        "candidate_incoming_edge_features": {
            other: np.zeros(11, dtype=np.float32) for other in other_ids
        },
        "potential_edge_features": {
            other: np.zeros(11, dtype=np.float32) for other in other_ids
        },
        "candidate_incoming_sources": other_ids,
        "effective_sources": other_ids,
        "neighbor_messages": messages,
        "candidate_incoming_links": {
            other: {
                "packet_loss": 0.1,
                "delay_s": 0.1,
                "aoi_s": 1.0,
                "queue_length_packets": 1.0,
                "rssi_dbm": -50.0,
                "snr_db": 20.0,
            }
            for other in other_ids
        },
        "action_mask": np.ones(4, dtype=bool),
        "bandwidth_mask": np.ones(3, dtype=bool),
        "potential_targets": other_ids,
    }
    if v3:
        observation["candidate_incoming_links"] = {
            other: dict(value)
            for other, value in observation["candidate_incoming_links"].items()
        }
    return observation


def _central_state(*, remaining_time: float | None = None) -> dict[str, object]:
    state: dict[str, object] = {}
    if remaining_time is not None:
        state["remaining_time_fraction"] = float(remaining_time)
    return state


def test_measurement_only_scope_guards_and_categorical_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="communication_learning_scope"):
        MAPPOConfig(communication_learning_scope="invalid")
    with pytest.raises(ValueError, match="measurement_only"):
        MAPPOConfig(communication_learning_scope="measurement_only")
    with pytest.raises(ValueError, match="sampled"):
        MAPPOConfig(
            motion_action_mode="fixed_navigation_comm_v1",
            ppo_action_logprob_mode="executed",
            communication_learning_scope="measurement_only",
        )

    def forbidden_distribution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("measurement-only policy constructed an unused distribution")

    monkeypatch.setattr(mappo_module, "Bernoulli", forbidden_distribution)
    monkeypatch.setattr(mappo_module, "Dirichlet", forbidden_distribution)
    monkeypatch.setattr(mappo_module, "Normal", forbidden_distribution)
    observation = _observation()
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    action, sample = policy.act(observation, deterministic=True)

    assert action["measurement_uav_id"] in {"uav_01", "uav_02", "none"}
    assert action["bandwidth"] == {"uav_01": 0.5, "uav_02": 0.5}
    assert sample.bandwidth.size == 0
    assert sample.delta_raw.size == 0 and sample.delta_v.size == 0
    assert sample.link_selection.size == 0
    assert sample.sampled_link_probabilities.size == 0
    assert sample.measurement_probabilities.shape == (4,)
    assert sample.measurement_probabilities[0] == pytest.approx(0.0)
    assert sample.measurement_probabilities[3] > 0.0
    log_prob, entropy = policy.evaluate_actions(observation, sample)
    assert float(log_prob.detach()) == pytest.approx(sample.old_log_prob, abs=1e-6)
    assert float(entropy.detach()) >= 0.0


@pytest.mark.parametrize("field", ["bandwidth", "delta_raw", "delta_v", "link_selection"])
def test_measurement_only_evaluate_rejects_learned_fields(field: str) -> None:
    observation = _observation()
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    _action, sample = policy.act(observation, deterministic=True)
    corrupted = copy.deepcopy(sample)
    if field == "bandwidth":
        corrupted.bandwidth = np.zeros(3, dtype=np.float32)
    elif field == "delta_raw":
        corrupted.delta_raw = np.zeros(3, dtype=np.float32)
    elif field == "delta_v":
        corrupted.delta_v = np.zeros(3, dtype=np.float32)
    else:
        corrupted.link_selection = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError, match="learned"):
        policy.evaluate_actions(observation, corrupted)


def test_measurement_only_freezes_delegated_heads_and_filters_optimizer() -> None:
    config = _config()
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    for name, parameter in policy.named_parameters():
        if (
            name.startswith("bandwidth_head.")
            or name.startswith("link_selection_head.")
            or name.startswith("delta_mean_head.")
            or name == "delta_logstd"
        ):
            assert not parameter.requires_grad, name
        elif (
            name.startswith("gat.")
            or name.startswith("actor_body.")
            or name.startswith("measurement_head.")
            or name.startswith("critic.")
        ):
            assert parameter.requires_grad, name

    trainer = MAPPOTrainer(config, uav_ids=UAV_IDS, seed=42)
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter)
        for parameter in trainer.policy.parameters()
        if parameter.requires_grad
    }


def test_measurement_only_surrogate_and_critic_backward_have_active_gradients() -> None:
    observation = _observation()
    observation["node_ids"] = list(UAV_IDS)
    observation["node_features"] = np.arange(111, dtype=np.float32).reshape(3, 37) / 500.0
    observation["effective_edge_index"] = np.asarray([[1, 2], [0, 0]], dtype=np.int64)
    observation["effective_edge_features"] = np.ones((2, 11), dtype=np.float32) * 0.01
    policy = MAPPOPolicy(_config(), uav_ids=UAV_IDS)
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    _action, sample = policy.act(observation)
    log_prob, entropy = policy.evaluate_actions(observation, sample)
    value = policy.value(np.zeros(policy.config.critic_input_dim, dtype=np.float32))
    loss = -torch.exp(log_prob - torch.as_tensor(sample.old_log_prob)) - 0.005 * entropy
    loss = loss + torch.nn.functional.mse_loss(value, torch.ones_like(value))
    policy.zero_grad(set_to_none=True)
    loss.backward()

    for prefix in ("gat.", "actor_body.", "measurement_head.", "critic."):
        active = [
            parameter
            for name, parameter in policy.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        assert active and sum(float(parameter.grad.detach().square().sum()) for parameter in active) > 0.0
    for name, parameter in policy.named_parameters():
        if (
            name.startswith("bandwidth_head.")
            or name.startswith("link_selection_head.")
            or name.startswith("delta_mean_head.")
            or name == "delta_logstd"
        ):
            assert parameter.grad is None
    assert all(torch.equal(before[name], tensor) for name, tensor in policy.state_dict().items())


def test_measurement_only_reconcile_uses_action_bandwidth_and_preserves_lp() -> None:
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
    diagnostics = decision.measurement_diagnostics["uav_00"]
    assert sample.old_log_prob == original.old_log_prob
    assert sample.bandwidth.size == 0 and sample.link_selection.size == 0
    assert np.array_equal(diagnostics["sampled_bandwidth"], np.asarray([0.0, 0.5, 0.5], dtype=np.float32))
    assert np.array_equal(diagnostics["executed_bandwidth"], np.asarray([0.0, 0.25, 0.0], dtype=np.float32))
    assert np.array_equal(diagnostics["sampled_link_selection"], np.asarray([False, True, True]))
    assert np.array_equal(diagnostics["executed_link_selection"], np.asarray([False, True, False]))
    assert diagnostics["action_execution_changed"] == 1
    assert diagnostics["nav_projection_changed"] == 1


def test_critic_measurement_budget_dimensions_and_uid_validation() -> None:
    assert central_state_vector_dim(3, 37, 11, 0) == 435
    assert central_state_vector_dim(3, 37, 11, 0, True) == 436
    assert central_state_vector_dim(3, 37, 11, 0, True, True) == 439
    config = MAPPOConfig(
        max_obstacles=0,
        include_remaining_time=True,
        critic_measurement_budget_features=True,
    )
    assert config.critic_input_dim == 439
    state = {
        "remaining_time_fraction": 0.5,
        "measurement_budget_remaining_ratios": {
            "uav_00": 0.1,
            "uav_01": 0.2,
            "uav_02": 0.3,
        },
    }
    vector = central_state_to_vector(
        state,
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        max_obstacles=0,
        include_remaining_time=True,
        critic_measurement_budget_features=True,
    )
    assert vector.shape == (439,)
    assert np.array_equal(vector[-3:], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    with pytest.raises(ValueError, match="required"):
        central_state_to_vector(
            {"remaining_time_fraction": 0.5},
            uav_count=3,
            max_obstacles=0,
            include_remaining_time=True,
            critic_measurement_budget_features=True,
        )
    for bad_ratios in (
        {"uav_00": 0.1, "uav_01": 0.2},
        {"uav_00": 0.1, "uav_01": 0.2, "uav_02": np.nan},
        {"uav_00": 0.1, "uav_01": 0.2, "uav_02": 1.1},
    ):
        with pytest.raises(ValueError):
            central_state_to_vector(
                {
                    "remaining_time_fraction": 0.5,
                    "measurement_budget_remaining_ratios": bad_ratios,
                },
                uav_count=3,
                max_obstacles=0,
                include_remaining_time=True,
                critic_measurement_budget_features=True,
            )


def test_measurement_only_v3_confidence_uses_raw_covariance_and_fails_closed() -> None:
    observation = _observation(v3=True)
    policy = MAPPOPolicy(
        _config(measurement_head_mode="action_conditioned_v3"),
        uav_ids=UAV_IDS,
    )
    policy._actor_distributions(observation)
    baseline = policy._last_measurement_head_trace["candidate_action_features"].copy()

    poisoned_nodes = copy.deepcopy(observation)
    poisoned_nodes["node_features"][:, 12:15] = 10000.0
    policy._actor_distributions(poisoned_nodes)
    poisoned = policy._last_measurement_head_trace["candidate_action_features"].copy()
    assert np.array_equal(baseline[:, 4:6], poisoned[:, 4:6])

    changed_covariance = copy.deepcopy(observation)
    changed_covariance["covariance"] = 3.0 * np.eye(6)
    changed_covariance["neighbor_messages"]["uav_01"]["covariance"] = 4.0 * np.eye(6)
    policy._actor_distributions(changed_covariance)
    changed = policy._last_measurement_head_trace["candidate_action_features"]
    assert changed[0, 4] != pytest.approx(baseline[0, 4])
    assert changed[1, 5] != pytest.approx(baseline[1, 5])

    invalid = copy.deepcopy(observation)
    invalid["covariance"] = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    invalid["neighbor_messages"]["uav_01"]["covariance"] = np.asarray(
        [[1.0, 0.2, 0.0, 0, 0, 0], [0.0, 1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
        dtype=np.float64,
    )
    policy._actor_distributions(invalid)
    failed_closed = policy._last_measurement_head_trace["candidate_action_features"]
    assert failed_closed[0, 4] == pytest.approx(0.0)
    assert failed_closed[1, 5] == pytest.approx(0.0)


def test_finite_horizon_gae_records_match_independent_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(
        episode_objective="finite_horizon",
        include_remaining_time=True,
        bootstrap_time_limits=False,
    )
    trainer = MAPPOTrainer(config, uav_ids=UAV_IDS, seed=42)
    monkeypatch.setattr(
        trainer.policy,
        "value",
        lambda _vector: torch.as_tensor([0.25], dtype=torch.float32),
    )
    transitions: list[RolloutTransition] = []
    values = [1.0, 0.5, 0.2]
    rewards = [1.0, 2.0, 3.0]
    for index, (value, reward) in enumerate(zip(values, rewards)):
        samples = {
            uid: PolicySample(
                measurement_index=3,
                bandwidth=np.empty(0, dtype=np.float32),
                delta_raw=np.empty(0, dtype=np.float32),
                delta_v=np.empty(0, dtype=np.float32),
                old_log_prob=0.0,
            )
            for uid in UAV_IDS
        }
        decision = PolicyDecision(
            actions={uid: {} for uid in UAV_IDS},
            samples=samples,
            observations={uid: {} for uid in UAV_IDS},
            central_vector=np.zeros(config.critic_input_dim, dtype=np.float32),
            value=value,
        )
        transitions.append(
            RolloutTransition(
                decision=decision,
                rewards={uid: reward for uid in UAV_IDS},
                next_central_state=_central_state(remaining_time=0.5),
                done=index == 2,
                terminated=index == 2,
                truncated=False,
            )
        )
    trainer.rollout = transitions
    records = trainer._training_records()

    expected_advantages = [0.0, 0.0, 0.0]
    next_advantage = 0.0
    for index in range(2, -1, -1):
        next_value = 0.0 if index == 2 else 0.25
        delta = rewards[index] + config.gamma * next_value - values[index]
        expected_advantages[index] = delta + config.gamma * config.gae_lambda * (
            0.0 if index == 2 else next_advantage
        )
        next_advantage = expected_advantages[index]
    for index, expected in enumerate(expected_advantages):
        returns = [float(record["return"]) for record in records[index * 3 : (index + 1) * 3]]
        assert returns == pytest.approx([expected + values[index]] * 3)
