"""Focused tests for the validation-only EKF sensor interaction audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.ekf_sensor_interaction_audit import (
    EKFSensorInteractionAuditConfig,
    _event_row,
)


def test_sensor_interaction_audit_is_validation_only_and_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        EKFSensorInteractionAuditConfig(reference_root=Path("reference"), split="test")
    with pytest.raises(ValueError, match="unique"):
        EKFSensorInteractionAuditConfig(reference_root=Path("reference"), seeds=(1, 1))
    with pytest.raises(ValueError, match="positive"):
        EKFSensorInteractionAuditConfig(reference_root=Path("reference"), horizons=(0,))


def test_event_pairs_next_visual_odometry_and_horizon_effects() -> None:
    event = {
        "accepted": True,
        "episode_id": "episode_400001",
        "scenario_id": "scenario_1",
        "step": 0,
        "measurement_time_s": 0.2,
        "source_uav_id": "uav_01",
        "target_uav_id": "uav_00",
        "los": True,
        "message_age_s": 0.1,
        "message_delay_s": 0.02,
        "nis": 1.1,
        "measurement_variance_m2": 0.2,
        "effective_measurement_variance_m2": 0.4,
        "local_direction_variance_m2": 0.1,
        "neighbor_direction_variance_m2": 0.1,
        "unknown_cross_bound_m2": 0.2,
        "unknown_cross_bound_fraction": 0.5,
        "kalman_gain_norm": 0.3,
        "position_correction_norm_m": 0.1,
        "state_before": [1.0, 2.0, 3.0],
        "state_after": [1.1, 2.0, 3.0],
        "state_correction": [0.1, 0.0, 0.0],
        "position_covariance_trace_before_m2": 3.0,
        "position_covariance_trace_after_linear_m2": 2.0,
        "position_covariance_trace_after_final_m2": 3.0,
        "covariance_restored": True,
        "j_loc_before": 0.3,
        "j_loc_after": 0.3,
        "true_position_error_before_m": 1.0,
        "true_position_error_after_m": 0.8,
    }
    sequences = {
        "uav_00": [
            {
                "timestamp": 0.2,
                "position_error_m": 0.8,
                "position_nees": 1.0,
                "covariance_trace_m2": 3.0,
                "vo_update_attempted": False,
                "vo_update_accepted": False,
                "vo_update_rejected": False,
                "uwb_update_attempted": False,
                "uwb_update_accepted": False,
                "uwb_update_rejected": False,
                "gnss_update_attempted": False,
                "gnss_update_accepted": False,
                "gnss_update_rejected": False,
            },
            {
                "timestamp": 0.4,
                "position_error_m": 0.7,
                "position_nees": 1.2,
                "covariance_trace_m2": 3.1,
                "nis_vo": 1.4,
                "vo_update_attempted": True,
                "vo_update_accepted": True,
                "vo_update_rejected": False,
                "uwb_update_attempted": False,
                "uwb_update_accepted": False,
                "uwb_update_rejected": False,
                "gnss_update_attempted": False,
                "gnss_update_accepted": False,
                "gnss_update_rejected": False,
            },
        ]
    }
    rewards = {
        ("uav_00", 0.4): {"j_loc": 0.31, "j_com": 0.1, "mean_reward": -0.4}
    }
    row = _event_row(
        event=event,
        seed=2026,
        branch="baseline_mappo",
        episode_root=Path("episode_400001"),
        update_index=0,
        horizons=(1, 2),
        sequences=sequences,
        rewards=rewards,
        cooperative_times={0.2},
    )

    assert row is not None
    assert row["next_window_transition_type"] == "visual_odometry_update"
    assert row["next_vo_observed"] == 1
    assert row["next_vo_nis"] == 1.4
    assert row["next_vo_accepted"] == 1
    assert row["h1_observed"] == 1
    assert row["h1_benefit_retained"] == 1
    assert row["h2_observed"] == 0
    assert row["state_correction_residual_norm_m"] == pytest.approx(0.0, abs=1.0e-15)
    assert row["linear_cov_reduction_m2"] == 1.0
    assert row["final_cov_reduction_m2"] == 0.0
