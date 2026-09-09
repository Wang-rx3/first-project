from __future__ import annotations

import numpy as np
import pytest
import torch

from src.learning.mappo import MAPPOConfig, MAPPOPolicy
from src.learning.measurement_proxy import information_gain_features


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _proxy_model() -> dict[str, object]:
    return {
        "age_variance_coefficient": 1.0,
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
        },
    }


def _observation() -> dict[str, object]:
    return {
        "node_ids": list(UAV_IDS),
        "node_features": np.zeros((3, 37), dtype=np.float32),
        "effective_edge_index": np.asarray([[1, 2], [0, 0]], dtype=np.int64),
        "effective_edge_features": np.asarray(
            [[10.0, 0.0, 0.0] + [0.0] * 8, [0.0, 10.0, 0.0] + [0.0] * 8],
            dtype=np.float32,
        ),
        "action_mask": np.asarray([True, True, True, True], dtype=bool),
        "bandwidth_mask": np.asarray([True, True, True], dtype=bool),
        "candidate_incoming_links": {
            "uav_01": {
                "rssi_dbm": -55.0,
                "snr_db": 15.0,
                "packet_loss": 0.05,
                "delay_s": 0.1,
                "aoi_s": 0.1,
                "queue_length_packets": 0.1,
                "link_state": "connected",
            },
            "uav_02": {
                "rssi_dbm": -55.0,
                "snr_db": 15.0,
                "packet_loss": 0.05,
                "delay_s": 0.1,
                "aoi_s": 0.1,
                "queue_length_packets": 0.1,
                "link_state": "connected",
            },
        },
        "neighbor_messages": {
            "uav_01": {"covariance": np.eye(6, dtype=np.float32)},
            "uav_02": {"covariance": np.eye(6, dtype=np.float32)},
        },
        "uwb_available": {"uav_01": True, "uav_02": True},
    }


def test_proxy_is_finite_id_aligned_and_truth_independent() -> None:
    observation = _observation()
    base = information_gain_features(observation, UAV_IDS, _proxy_model())
    altered = dict(observation)
    altered.update(
        {
            "true_position": np.asarray([999.0, 999.0, 999.0]),
            "true_range": 999.0,
            "physical_los_nlos": "nlos",
            "future_state": {"position": [999.0, 999.0, 999.0]},
        }
    )
    changed = information_gain_features(altered, UAV_IDS, _proxy_model())
    for before, after in zip(base, changed):
        assert np.isfinite(before).all()
        assert np.array_equal(before, after)
    gain, cost, valid = base
    assert gain.shape == cost.shape == valid.shape == (3,)
    assert np.all((0.0 <= gain) & (gain <= 1.0))
    assert np.all((0.0 <= cost) & (cost <= 1.0))
    assert np.all((0.0 <= valid) & (valid <= 1.0))


def test_worse_visible_link_increases_predicted_cost() -> None:
    good = _observation()
    poor = _observation()
    poor["candidate_incoming_links"] = dict(good["candidate_incoming_links"])
    poor["candidate_incoming_links"]["uav_01"] = {
        **poor["candidate_incoming_links"]["uav_01"],
        "rssi_dbm": -95.0,
        "snr_db": 1.0,
        "packet_loss": 0.8,
        "delay_s": 3.0,
        "aoi_s": 3.0,
        "queue_length_packets": 3.0,
        "link_state": "degraded",
    }
    _, good_cost, _ = information_gain_features(good, UAV_IDS, _proxy_model())
    _, poor_cost, _ = information_gain_features(poor, UAV_IDS, _proxy_model())
    assert poor_cost[1] > good_cost[1]


def test_proxy_policy_auxiliary_gradient_is_measurement_head_only() -> None:
    torch.manual_seed(2026)
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_proxy_features=True,
        measurement_proxy_guidance_enabled=True,
        measurement_proxy_teacher_coef=0.25,
        measurement_proxy_temperature=0.25,
        measurement_proxy_communication_cost_weight=1.0,
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    observation = _observation()
    observation.update(
        {
            "expected_information_gain_norm": np.asarray([0.0, 0.9, 0.3], dtype=np.float32),
            "predicted_communication_cost_norm": np.asarray([1.0, 0.1, 0.8], dtype=np.float32),
            "proxy_valid": np.ones(3, dtype=np.float32),
        }
    )
    loss, diagnostics = policy.measurement_proxy_teacher_loss(observation)
    assert torch.isfinite(loss)
    assert diagnostics["proxy_teacher_valid"] == 1.0
    policy.zero_grad(set_to_none=True)
    loss.backward()
    measurement_grads = [
        parameter.grad for name, parameter in policy.named_parameters() if name.startswith("measurement_head")
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0.0 for gradient in measurement_grads)
    for name, parameter in policy.named_parameters():
        if not name.startswith("measurement_head"):
            assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name


def test_communication_proxy_teacher_charges_only_cost_above_target() -> None:
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_proxy_features=True,
        measurement_proxy_guidance_enabled=True,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav={
            "uav_00": 0.5,
            "uav_01": 0.5,
            "uav_02": 0.5,
        },
        measurement_proxy_target_relative_cost=True,
    )
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    observation = _observation()
    observation.update(
        {
            "uav_id": "uav_00",
            "expected_information_gain_norm": np.asarray([0.0, 0.9, 0.3], dtype=np.float32),
            "predicted_communication_cost_norm": np.asarray([1.0, 0.1, 0.8], dtype=np.float32),
            "proxy_valid": np.ones(3, dtype=np.float32),
        }
    )
    policy._actor_distributions(observation)
    utility = policy._last_measurement_head_trace["proxy_utility"]
    assert np.allclose(utility, np.asarray([-0.5, 0.9, 0.0], dtype=np.float32))
    loss, diagnostics = policy.measurement_proxy_teacher_loss(
        observation, uav_id="uav_00"
    )
    assert torch.isfinite(loss)
    assert diagnostics["proxy_teacher_target_relative_cost"] == 1.0
    assert diagnostics["proxy_teacher_target_cost"] == 0.5


def test_baseline_policy_ignores_proxy_arrays() -> None:
    config = MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11)
    policy = MAPPOPolicy(config, uav_ids=UAV_IDS)
    base = _observation()
    altered = dict(base)
    altered.update(
        {
            "expected_information_gain_norm": np.ones(3, dtype=np.float32),
            "predicted_communication_cost_norm": np.zeros(3, dtype=np.float32),
            "proxy_valid": np.ones(3, dtype=np.float32),
        }
    )
    first = policy._actor_distributions(base)[0].probs
    second = policy._actor_distributions(altered)[0].probs
    assert torch.allclose(first, second)


def test_proxy_config_rejects_invalid_teacher_settings() -> None:
    with pytest.raises(ValueError, match="temperature"):
        MAPPOConfig(
            use_measurement_proxy_features=True,
            measurement_proxy_temperature=0.0,
        )
    with pytest.raises(ValueError, match="requires"):
        MAPPOConfig(measurement_proxy_guidance_enabled=True)
