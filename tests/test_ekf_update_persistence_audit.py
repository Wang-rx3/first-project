"""Tests for the validation-only EKF update persistence audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.ekf_update_persistence_audit import (
    EKFUpdatePersistenceAuditConfig,
    _event_row,
    classify_update_effect,
)


def test_persistence_audit_is_validation_only_and_rejects_duplicate_inputs() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        EKFUpdatePersistenceAuditConfig(
            reference_root="reference",
            split="test",
        )
    with pytest.raises(ValueError, match="unique"):
        EKFUpdatePersistenceAuditConfig(
            reference_root="reference",
            seeds=(12136, 12136),
        )
    with pytest.raises(ValueError, match="positive"):
        EKFUpdatePersistenceAuditConfig(
            reference_root="reference",
            horizons=(0,),
        )


def test_update_effect_classification_uses_fixed_one_millimetre_threshold() -> None:
    assert classify_update_effect(-0.001001) == "beneficial"
    assert classify_update_effect(0.001001) == "harmful"
    assert classify_update_effect(-0.001) == "approximately_neutral"


def test_event_row_records_future_censoring_and_persistence_without_truth_input() -> None:
    event = {
        "accepted": True,
        "episode_id": "episode_400001",
        "step": 0,
        "measurement_time_s": 0.2,
        "source_uav_id": "uav_01",
        "target_uav_id": "uav_00",
        "los": True,
        "true_position_error_before_m": 1.0,
        "true_position_error_after_m": 0.8,
        "j_loc_before": 0.3,
        "j_loc_after": 0.3,
        "position_covariance_trace_before_m2": 12.0,
        "position_covariance_trace_after_linear_m2": 11.0,
        "position_covariance_trace_after_final_m2": 12.0,
        "effective_measurement_variance_m2": 4.0,
    }
    sequences = {
        "uav_00": [
            {"timestamp": 0.2, "position_error_m": 0.8, "position_nees": 1.0, "covariance_trace_m2": 12.0},
            {"timestamp": 0.4, "position_error_m": 0.9, "position_nees": 1.1, "covariance_trace_m2": 12.1},
        ]
    }
    rewards = {
        ("uav_00", 0.4): {"j_loc": 0.31, "j_com": 0.1, "mean_reward": -0.4},
    }
    row = _event_row(
        event=event,
        seed=12136,
        branch="baseline_mappo",
        episode_root=Path("episode"),
        update_index=0,
        horizons=(1, 2),
        sequences=sequences,
        rewards=rewards,
    )

    assert row is not None
    assert row["immediate_effect_class"] == "beneficial"
    assert row["h1_observed"] == 1
    assert row["h1_benefit_retained"] == 1
    assert row["h2_observed"] == 0
    assert row["h2_future_error_m"] == -1.0
    assert row["truth_used_offline_only"] == 1
    assert row["linear_cov_reduction_m2"] == 1.0
