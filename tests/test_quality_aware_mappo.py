from __future__ import annotations

import numpy as np
import pytest
import torch

from src.learning.measurement_quality import fit_warmup_quality_model, quality_features
from src.learning.mappo import MAPPOConfig, MAPPOPolicy


def _empty_observation() -> dict[str, object]:
    return {
        "node_features": np.zeros((1, 37), dtype=np.float32),
        "effective_edge_index": np.empty((2, 0), dtype=np.int64),
        "effective_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_edge_features": np.empty((0, 11), dtype=np.float32),
        "candidate_incoming_edge_features": np.empty((0, 11), dtype=np.float32),
        "potential_edge_features": np.empty((0, 11), dtype=np.float32),
        "action_mask": np.asarray([True, True, False, True], dtype=bool),
        "bandwidth_mask": np.asarray([False, False, False], dtype=bool),
        "candidate_incoming_links": {},
        "neighbor_messages": {},
    }


def test_quality_model_is_actor_visible_and_finite() -> None:
    observation = _empty_observation()
    model = fit_warmup_quality_model(
        [{"uav_00": observation}], ("uav_00", "uav_01", "uav_02")
    )
    risk, valid = quality_features(
        observation, ("uav_00", "uav_01", "uav_02"), model
    )
    assert risk.shape == (3,)
    assert valid.shape == (3,)
    assert np.isfinite(risk).all()
    assert np.isfinite(valid).all()
    assert set(("true_position", "true_error", "physical_los_nlos", "future_state")) <= set(
        model["forbidden_inputs"]
    )
    assert model["source_split"] == "warmup"


def test_quality_features_change_only_measurement_head_input() -> None:
    torch.manual_seed(2026)
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_quality_features=True,
    )
    policy = MAPPOPolicy(config, uav_ids=("uav_00", "uav_01", "uav_02"))
    base = _empty_observation()
    enriched = dict(base)
    enriched["measurement_quality_risk"] = np.asarray([0.05, 0.80, 1.0], dtype=np.float32)
    enriched["measurement_quality_valid"] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)

    with torch.no_grad():
        # Make the test deterministic with respect to the newly added input
        # block while leaving the other actor heads untouched.
        policy.measurement_head.weight[:, -6:] = torch.arange(
            policy.measurement_head.weight[:, -6:].numel(), dtype=torch.float32
        ).reshape_as(policy.measurement_head.weight[:, -6:]) / 100.0

    measurement_base, bandwidth_base, delta_base, _ = policy._actor_distributions(base)
    measurement_enriched, bandwidth_enriched, delta_enriched, _ = policy._actor_distributions(enriched)

    assert not torch.allclose(measurement_base.probs, measurement_enriched.probs)
    assert torch.allclose(bandwidth_base, bandwidth_enriched)
    assert torch.allclose(delta_base.base_dist.loc, delta_enriched.base_dist.loc)


def test_baseline_policy_has_no_quality_input_dimension() -> None:
    baseline = MAPPOPolicy(
        MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11),
        uav_ids=("uav_00", "uav_01", "uav_02"),
    )
    quality = MAPPOPolicy(
        MAPPOConfig(
            uav_count=3,
            node_feature_dim=37,
            edge_feature_dim=11,
            use_measurement_quality_features=True,
        ),
        uav_ids=("uav_00", "uav_01", "uav_02"),
    )
    assert baseline.measurement_head.in_features + 6 == quality.measurement_head.in_features
    assert baseline.bandwidth_head.in_features == quality.bandwidth_head.in_features
    assert baseline.delta_mean_head.in_features == quality.delta_mean_head.in_features


def test_quality_logit_penalty_makes_candidate_probability_nonincreasing() -> None:
    torch.manual_seed(2026)
    config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        use_measurement_quality_features=True,
        measurement_quality_logit_penalty=1.0,
    )
    policy = MAPPOPolicy(config, uav_ids=("uav_00", "uav_01", "uav_02"))
    with torch.no_grad():
        # Isolate the fixed monotone term from any learned direct risk
        # response; the production head still receives the full quality block.
        policy.measurement_head.weight[:, -6:] = 0.0
    base = _empty_observation()
    low = dict(
        base,
        measurement_quality_risk=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        measurement_quality_valid=np.ones(3, dtype=np.float32),
    )
    high = dict(
        base,
        measurement_quality_risk=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        measurement_quality_valid=np.ones(3, dtype=np.float32),
    )
    low_measurement, low_bandwidth, low_delta, _ = policy._actor_distributions(low)
    high_measurement, high_bandwidth, high_delta, _ = policy._actor_distributions(high)

    assert float(high_measurement.probs[1]) <= float(low_measurement.probs[1]) + 1e-7
    assert float(high_measurement.probs[3]) >= float(low_measurement.probs[3]) - 1e-7
    assert torch.allclose(low_bandwidth, high_bandwidth)
    assert torch.allclose(low_delta.base_dist.loc, high_delta.base_dist.loc)


def test_baseline_logit_penalty_and_quality_input_do_not_change_baseline() -> None:
    torch.manual_seed(2026)
    base = _empty_observation()
    low = dict(
        base,
        measurement_quality_risk=np.zeros(3, dtype=np.float32),
        measurement_quality_valid=np.ones(3, dtype=np.float32),
    )
    high = dict(
        base,
        measurement_quality_risk=np.ones(3, dtype=np.float32),
        measurement_quality_valid=np.ones(3, dtype=np.float32),
    )
    policy = MAPPOPolicy(
        MAPPOConfig(
            uav_count=3,
            node_feature_dim=37,
            edge_feature_dim=11,
            measurement_quality_logit_penalty=1.0,
        ),
        uav_ids=("uav_00", "uav_01", "uav_02"),
    )
    low_measurement, low_bandwidth, low_delta, _ = policy._actor_distributions(low)
    high_measurement, high_bandwidth, high_delta, _ = policy._actor_distributions(high)
    assert torch.allclose(low_measurement.probs, high_measurement.probs)
    assert torch.allclose(low_bandwidth, high_bandwidth)
    assert torch.allclose(low_delta.base_dist.loc, high_delta.base_dist.loc)
    assert float(policy._last_measurement_head_trace["coefficient"]) == 0.0


def test_quality_logit_penalty_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="measurement_quality_logit_penalty"):
        MAPPOConfig(measurement_quality_logit_penalty=-1.0)
    with pytest.raises(ValueError, match="measurement_quality_logit_penalty"):
        MAPPOConfig(measurement_quality_logit_penalty=float("nan"))
