"""Unit tests for the read-only measurement-policy stability audit."""

from __future__ import annotations

import math

from src.experiments.measurement_policy_stability_audit import (
    MeasurementPolicyStabilityAuditConfig,
    _comparison_rows,
    _rate_variation,
    summarize_training_rows,
)


def test_training_summary_is_descriptive_and_finite() -> None:
    rows = [
        {
            "phase": "train",
            "update": 1,
            "actor_loss": -0.1,
            "critic_loss": 4.0,
            "loss": 2.0,
            "entropy": 1.0,
            "gradient_norm": 3.0,
            "clip_fraction": 0.1,
            "mean_reward": -0.4,
            "mean_action_saturation": 0.0,
            "max_action_saturation": 0.0,
        },
        {
            "phase": "train",
            "update": 2,
            "actor_loss": -0.2,
            "critic_loss": 3.0,
            "loss": 1.5,
            "entropy": 0.4,
            "gradient_norm": 2.0,
            "clip_fraction": 0.3,
            "mean_reward": -0.3,
            "mean_action_saturation": 0.1,
            "max_action_saturation": 1.0,
        },
    ]

    summary = summarize_training_rows(rows, seed=8080, branch="budgeted_quality_aware_mappo")

    assert summary["update_count"] == 2
    assert summary["finite_all"] == 1
    assert summary["entropy_ratio"] == 0.4
    assert summary["action_saturation_observed"] == 1
    assert summary["heuristic_training_dynamics_flag"] == 1


def test_measurement_rate_variation_flags_seed_sensitive_policy() -> None:
    rows = [
        {"base_seed": 7071, "branch": "budgeted_quality_aware_mappo", "measurement_selection_rate": 0.01},
        {"base_seed": 8080, "branch": "budgeted_quality_aware_mappo", "measurement_selection_rate": 0.44},
        {"base_seed": 9099, "branch": "budgeted_quality_aware_mappo", "measurement_selection_rate": 0.02},
    ]

    result = _rate_variation(rows, ("budgeted_quality_aware_mappo",))

    assert result["extreme_measurement_variation"] == 1
    assert result["by_branch"]["budgeted_quality_aware_mappo"]["range"] == 0.43


def test_seed_comparison_is_validation_only_and_seed_level() -> None:
    rows = [
        {
            "base_seed": 7071,
            "branch": "baseline_mappo",
            "measurement_selection_rate": 0.8,
            "position_rmse_m": 0.3,
            "j_com": 0.1,
        },
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "measurement_selection_rate": 0.4,
            "position_rmse_m": 0.2,
            "j_com": 0.12,
        },
    ]

    result = _comparison_rows(
        rows,
        (7071,),
        ("baseline_mappo", "budgeted_quality_aware_mappo"),
    )

    assert len(result) == 1
    assert result[0]["split"] == "validation"
    assert result[0]["delta_measurement_selection_rate"] == -0.4
    assert math.isclose(result[0]["delta_position_rmse_m"], -0.1)


def test_stability_audit_rejects_test_and_duplicate_seeds() -> None:
    try:
        MeasurementPolicyStabilityAuditConfig(reference_root="data", split="test")
    except ValueError as exc:
        assert "validation-only" in str(exc)
    else:
        raise AssertionError("test split must be rejected")

    try:
        MeasurementPolicyStabilityAuditConfig(reference_root="data", seeds=(7071, 7071))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate seeds must be rejected")
