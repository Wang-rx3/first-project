from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.learning.mappo import MAPPOConfig, MAPPOPolicy, MAPPOTrainer
from src.learning.measurement_proxy import (
    add_persistent_information_gain_features,
    persistent_information_gain_features,
)


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _persistent_model() -> dict[str, object]:
    return {
        "age_variance_coefficient": 1.0,
        "persistent_horizon": 3,
        "decision_dt_s": 0.2,
        "process_position_std_m": 0.08,
        "process_velocity_std_mps": 0.15,
        "persistent_ig_q05": 0.0,
        "persistent_ig_q95": 1.0,
        "scales": {
            "measurement_std_median": 0.5,
            "neighbor_covariance_trace_q95_m2": 3.0,
            "delay_q75_s": 1.0,
            "age_q75_s": 1.0,
            "queue_q75_packets": 1.0,
            "rssi_q05_dbm": -100.0,
            "rssi_q95_dbm": -40.0,
            "snr_q05_db": 0.0,
            "snr_q95_db": 20.0,
            "ig_q05": 0.0,
            "ig_q95": 1.0,
            "persistent_ig_q05": 0.0,
            "persistent_ig_q95": 1.0,
        },
    }


def _observation() -> dict[str, object]:
    nodes = np.zeros((3, 37), dtype=np.float32)
    nodes[0, :3] = (0.0, 0.0, 10.0)
    nodes[1, :3] = (10.0, 0.0, 10.0)
    nodes[2, :3] = (0.0, 10.0, 10.0)
    nodes[:, 15:21] = np.asarray(
        [
            [1.0, 1.0, 1.0, 0.2, 0.2, 0.2],
            [0.8, 0.8, 0.8, 0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    link = {
        "rssi_dbm": -55.0,
        "snr_db": 15.0,
        "packet_loss": 0.05,
        "delay_s": 0.1,
        "aoi_s": 0.1,
        "queue_length_packets": 0.1,
        "link_state": "connected",
    }
    return {
        "node_ids": list(UAV_IDS),
        "node_features": nodes,
        "effective_edge_index": np.asarray([[1, 2], [0, 0]], dtype=np.int64),
        "effective_edge_features": np.asarray(
            [
                [10.0, 0.0, 0.0] + [0.0] * 8,
                [0.0, 10.0, 0.0] + [0.0] * 8,
            ],
            dtype=np.float32,
        ),
        "candidate_incoming_links": {
            "uav_01": dict(link),
            "uav_02": {**link, "rssi_dbm": -70.0, "snr_db": 8.0},
        },
        "candidate_incoming_edge_features": {
            "uav_01": [10.0] + [0.0] * 10,
            "uav_02": [10.0] + [0.0] * 10,
        },
        "candidate_edge_features": {
            "uav_01": [10.0] + [0.0] * 10,
            "uav_02": [10.0] + [0.0] * 10,
        },
        "potential_edge_features": {
            "uav_01": [10.0] + [0.0] * 10,
            "uav_02": [10.0] + [0.0] * 10,
        },
        "effective_edge_features_by_source": {},
        "neighbor_messages": {
            "uav_01": {
                "estimate": np.asarray([10.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                "covariance": np.diag([0.8, 0.8, 0.8, 0.1, 0.1, 0.1]),
            },
            "uav_02": {
                "estimate": np.asarray([0.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
                "covariance": np.diag([0.6, 0.6, 0.6, 0.1, 0.1, 0.1]),
            },
        },
        "uwb_available": {"uav_01": True, "uav_02": True},
        "action_mask": np.asarray([True, True, True, True], dtype=bool),
        "bandwidth_mask": np.asarray([True, True, True], dtype=bool),
    }


def test_persistent_proxy_is_finite_nonnegative_and_truth_independent() -> None:
    observation = _observation()
    model = _persistent_model()
    original = persistent_information_gain_features(observation, UAV_IDS, model, horizon=3)
    altered = dict(observation)
    altered.update(
        {
            "true_position": np.asarray([999.0, 999.0, 999.0]),
            "true_range": 999.0,
            "physical_los_nlos": "nlos",
            "future_state": {"position": [999.0, 999.0, 999.0]},
            "future_packet_outcome": "dropped",
        }
    )
    changed = persistent_information_gain_features(altered, UAV_IDS, model, horizon=3)
    for before, after in zip(original, changed):
        assert np.isfinite(before).all()
        assert np.array_equal(before, after)
    gain, cost, valid = original
    assert gain.shape == cost.shape == valid.shape == (3,)
    assert np.all((gain >= 0.0) & (gain <= 1.0))
    assert np.all((cost >= 0.0) & (cost <= 1.0))
    assert np.all((valid == 0.0) | (valid == 1.0))
    assert np.any(valid > 0.5)


def test_persistent_proxy_separates_gain_from_visible_link_cost() -> None:
    observation = _observation()
    model = _persistent_model()
    gain, cost, valid = persistent_information_gain_features(observation, UAV_IDS, model)
    assert valid[1] == 1.0 and valid[2] == 1.0
    assert cost[2] > cost[1]
    # The fixed-horizon gain is a covariance calculation; changing only the
    # visible radio fields must not silently multiply or otherwise alter it.
    worse_link = dict(observation)
    worse_links = dict(observation["candidate_incoming_links"])
    worse_links["uav_01"] = {
        **worse_links["uav_01"],
        "rssi_dbm": -95.0,
        "snr_db": 1.0,
        "packet_loss": 0.8,
        "link_state": "degraded",
    }
    worse_link["candidate_incoming_links"] = worse_links
    worse_gain, worse_cost, _ = persistent_information_gain_features(worse_link, UAV_IDS, model)
    assert np.isclose(gain[1], worse_gain[1])
    assert worse_cost[1] > cost[1]


def test_persistent_policy_reads_only_persistent_measurement_block() -> None:
    torch.manual_seed(2026)
    model = _persistent_model()
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_persistent_proxy_features=True,
        measurement_persistent_proxy_guidance_enabled=True,
        measurement_persistent_proxy_horizon=3,
        measurement_persistent_proxy_model=model,
        measurement_proxy_teacher_coef=0.25,
        measurement_proxy_temperature=0.25,
        measurement_proxy_communication_cost_weight=1.0,
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    observation = add_persistent_information_gain_features(
        _observation(), UAV_IDS, model, horizon=3
    )
    distribution, _bandwidth, _delta, _mask = policy._actor_distributions(observation)
    assert distribution.probs.shape == (4,)
    assert torch.isfinite(distribution.probs).all()
    loss, diagnostics = policy.measurement_proxy_teacher_loss(observation)
    assert torch.isfinite(loss)
    assert diagnostics["proxy_teacher_valid"] == 1.0
    assert diagnostics["proxy_teacher_persistent"] == 1.0
    policy.zero_grad(set_to_none=True)
    loss.backward()
    measurement_grads = [
        parameter.grad
        for name, parameter in policy.named_parameters()
        if name.startswith("measurement_head")
    ]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and float(gradient.abs().sum()) > 0.0
        for gradient in measurement_grads
    )
    for name, parameter in policy.named_parameters():
        if not name.startswith("measurement_head"):
            assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name


def test_persistent_teacher_can_charge_only_target_relative_excess_cost() -> None:
    torch.manual_seed(2026)
    model = _persistent_model()
    targets = {uav_id: 0.5 for uav_id in UAV_IDS}
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_persistent_proxy_features=True,
        measurement_persistent_proxy_guidance_enabled=True,
        measurement_persistent_proxy_model=model,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav=targets,
        measurement_persistent_proxy_target_relative_cost=True,
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    observation = add_persistent_information_gain_features(
        {**_observation(), "uav_id": "uav_00"}, UAV_IDS, model, horizon=3
    )
    loss, diagnostics = policy.measurement_proxy_teacher_loss(
        observation, uav_id="uav_00"
    )
    assert torch.isfinite(loss)
    assert diagnostics["proxy_teacher_target_relative_cost"] == 1.0
    assert diagnostics["proxy_teacher_target_cost"] == 0.5


def test_target_relative_persistent_teacher_requires_targets() -> None:
    with pytest.raises(ValueError, match="targets"):
        MAPPOConfig(
            use_measurement_persistent_proxy_features=True,
            measurement_persistent_proxy_guidance_enabled=True,
            measurement_persistent_proxy_model=_persistent_model(),
            measurement_persistent_proxy_target_relative_cost=True,
        )


def test_baseline_ignores_persistent_features() -> None:
    torch.manual_seed(2026)
    policy = MAPPOPolicy(
        MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11),
        uav_ids=UAV_IDS,
    )
    base = _observation()
    altered = add_persistent_information_gain_features(base, UAV_IDS, _persistent_model())
    first = policy._actor_distributions(base)[0].probs
    second = policy._actor_distributions(altered)[0].probs
    assert torch.allclose(first, second)


def test_persistent_config_requires_model_and_positive_horizon() -> None:
    with pytest.raises(ValueError, match="model"):
        MAPPOConfig(
            use_measurement_persistent_proxy_features=True,
            measurement_persistent_proxy_guidance_enabled=True,
        )
    with pytest.raises(ValueError, match="horizon"):
        MAPPOConfig(measurement_persistent_proxy_horizon=0)


def test_trainer_from_environment_propagates_persistent_model_into_config() -> None:
    environment = SimpleNamespace(
        uav_ids=UAV_IDS,
        config=SimpleNamespace(
            motion=SimpleNamespace(max_acceleration_mps2=3.0, dt_s=0.2)
        ),
    )
    model = _persistent_model()
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=2026,
        use_measurement_persistent_proxy_features=True,
        measurement_persistent_proxy_guidance_enabled=True,
        measurement_persistent_proxy_model=model,
    )
    assert trainer.config.use_measurement_persistent_proxy_features is True
    assert trainer.config.measurement_persistent_proxy_model == model
    assert trainer.measurement_persistent_proxy_model == model
