"""Unit tests for the validation-only quality-feature sensitivity audit."""

from __future__ import annotations

import copy

import numpy as np

from src.experiments.quality_feature_sensitivity_audit import (
    QualityFeatureSensitivityAuditConfig,
    _counterfactual_summary,
    _risk_semantic_rows,
)
from src.learning.measurement_quality import RISK_FIELDS, quality_features
from src.learning.mappo import MAPPOConfig, MAPPOPolicy


def _model() -> dict[str, object]:
    return {
        "source_split": "warmup",
        "quantiles": {field: {"q25": 0.0, "q75": 1.0} for field in RISK_FIELDS},
    }


def _observation() -> dict[str, object]:
    return {
        "node_features": np.zeros((1, 2), dtype=np.float32),
        "effective_edge_index": np.empty((2, 0), dtype=np.int64),
        "effective_edge_features": np.empty((0, 2), dtype=np.float32),
        "candidate_edge_features": {},
        "candidate_incoming_edge_features": {},
        "potential_edge_features": {},
        "action_mask": np.asarray([1, 1, 0, 1], dtype=np.float32),
        "bandwidth_mask": np.asarray([0, 0, 0], dtype=np.float32),
        "candidate_incoming_links": {
            "uav_01": {
                "rssi_dbm": -70.0,
                "snr_db": 20.0,
                "packet_loss": 0.1,
                "delay_s": 0.01,
                "aoi_s": 0.1,
                "queue_length_packets": 0.5,
                "link_state": "connected",
            }
        },
        "neighbor_messages": {
            "uav_01": {"covariance": np.eye(3, dtype=np.float32)},
        },
    }


def test_risk_semantics_are_monotone_for_all_registered_fields() -> None:
    rows = _risk_semantic_rows(2026, "quality_aware_mappo", _model())

    assert [row["field"] for row in rows] == list(RISK_FIELDS)
    assert all(row["finite_ok"] == 1 for row in rows)
    assert all(row["monotonic_ok"] == 1 for row in rows)


def test_missing_neighbor_covariance_is_finite_and_not_better_than_complete() -> None:
    uav_ids = ("uav_00", "uav_01", "uav_02")
    observation = _observation()
    full_risk, full_valid = quality_features(observation, uav_ids, _model())
    missing = copy.deepcopy(observation)
    missing["neighbor_messages"] = {"uav_01": {}}
    missing_risk, missing_valid = quality_features(missing, uav_ids, _model())

    index = uav_ids.index("uav_01")
    assert np.isfinite(missing_risk).all()
    assert missing_valid[index] == 1.0
    assert missing_risk[index] >= full_risk[index]
    assert full_valid[index] == 1.0


def test_baseline_policy_ignores_injected_quality_and_quality_policy_reads_it() -> None:
    base = _observation()
    low = dict(base, measurement_quality_risk=np.zeros(3), measurement_quality_valid=np.ones(3))
    high = dict(base, measurement_quality_risk=np.ones(3), measurement_quality_valid=np.ones(3))
    common = dict(
        uav_count=3,
        node_feature_dim=2,
        edge_feature_dim=2,
        gat_hidden_dim=4,
        gat_output_dim=4,
        gat_heads=1,
        gat_layers=1,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        max_obstacles=1,
        device="cpu",
    )
    baseline = MAPPOPolicy(MAPPOConfig(**common, use_measurement_quality_features=False))
    quality = MAPPOPolicy(MAPPOConfig(**common, use_measurement_quality_features=True))

    baseline_low = baseline._actor_distributions(low)[0].probs.detach().numpy()
    baseline_high = baseline._actor_distributions(high)[0].probs.detach().numpy()
    quality_low_features = quality._measurement_quality_features(low).detach().numpy()
    quality_high_features = quality._measurement_quality_features(high).detach().numpy()

    assert np.allclose(baseline_low, baseline_high)
    assert not np.allclose(quality_low_features, quality_high_features)


def test_counterfactual_summary_is_deterministic_and_expected_direction_is_explicit() -> None:
    rows = []
    for risk, probability in zip((0.0, 0.25, 0.5, 0.75, 1.0), (0.8, 0.7, 0.6, 0.4, 0.2)):
        rows.append({
            "base_seed": 2026,
            "branch": "quality_aware_mappo",
            "episode_id": "episode_400001",
            "step": 0,
            "uav_id": "uav_00",
            "candidate_uav_id": "uav_01",
            "risk_value": risk,
            "candidate_probability": probability,
            "none_probability": 1.0 - probability,
        })

    summary = _counterfactual_summary(rows)

    assert len(summary) == 1
    assert summary[0]["monotonic_expected_direction"] == 1
    assert summary[0]["endpoint_delta_candidate_probability"] < 0.0


def test_audit_is_validation_only_and_cpu_only() -> None:
    try:
        QualityFeatureSensitivityAuditConfig(reference_root="data", split="test")
    except ValueError as exc:
        assert "validation-only" in str(exc)
    else:
        raise AssertionError("test split must be rejected")

    try:
        QualityFeatureSensitivityAuditConfig(reference_root="data", device="cuda")
    except ValueError as exc:
        assert "cpu" in str(exc)
    else:
        raise AssertionError("non-CPU device must be rejected")
