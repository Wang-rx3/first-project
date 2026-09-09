"""Tests for the read-only 3-UAV curve-cause audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.final_mainline_audit import BASELINE_BRANCH, PROPOSED_BRANCH
from src.experiments.mainline_curve_cause_audit import (
    MainlineCurveCauseAuditConfig,
    _attributions,
    _collision_summaries,
    _expected_actual_rows,
    _is_validation_seed_row,
    _trailing_mean,
)


def test_trailing_mean_preserves_prefix_and_uses_declared_window() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    result = _trailing_mean(values, window=5)

    assert result[:2] == [1.0, 1.5]
    assert np.isclose(result[4], 3.0)
    assert np.isclose(result[5], 4.0)
    assert values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_collision_summary_separates_collision_and_noncollision_updates() -> None:
    rows = []
    for update in range(1, 6):
        collision = update in {2, 5}
        rows.append(
            {
                "base_seed": 101,
                "branch": BASELINE_BRANCH,
                "update": update,
                "mean_reward_raw": -0.9 if collision else -0.4,
                "collision_occurred": int(collision),
                "loss": 8.0 if collision else 2.0,
                "critic_loss": 16.0 if collision else 4.0,
            }
        )

    summary = _collision_summaries(rows, (101,), (BASELINE_BRANCH,))[0]

    assert summary["collision_update_count"] == 2
    assert summary["low_reward_update_count"] == 2
    assert summary["collision_fraction_among_low_reward"] == 1.0
    assert np.isclose(summary["critic_collision_to_noncollision_ratio"], 4.0)


def test_expected_actual_rows_marks_deterministic_argmax_gap() -> None:
    seed_rows = [
        {
            "base_seed": 101,
            "branch": branch,
            "split": "validation",
            "measurement_count": 10,
            "j_com": 0.1,
        }
        for branch in (BASELINE_BRANCH, PROPOSED_BRANCH)
    ]
    cost_rows = [
        {
            "base_seed": 101,
            "treatment": branch,
            "split": "validation",
            "uav_id": f"uav_{index:02d}",
            "target": 0.5,
            "expected_communication_cost": 0.4,
            "actual_communication_cost": 0.6,
        }
        for branch in (BASELINE_BRANCH, PROPOSED_BRANCH)
        for index in range(3)
    ]

    rows = _expected_actual_rows(
        seed_rows, cost_rows, (101,), (BASELINE_BRANCH, PROPOSED_BRANCH)
    )

    assert len(rows) == 2
    assert all(row["validation_execution_mode"] == "deterministic_argmax" for row in rows)
    assert all(np.isclose(float(row["actual_minus_expected"]), 0.2) for row in rows)
    assert all(row["expected_within_target_plus_0_05"] == 1 for row in rows)
    assert all(row["actual_within_target_plus_0_05"] == 0 for row in rows)


def test_attribution_rules_distinguish_confirmed_and_ruled_out() -> None:
    collision_rows = [
        {
            "low_reward_update_count": 10,
            "low_reward_collision_count": 10,
            "mean_reward_collision": -0.85,
            "mean_reward_noncollision": -0.4,
            "loss_critic_correlation": 0.999,
            "critic_collision_to_noncollision_ratio": 3.0,
            "noncollision_reward_std": 0.06,
        }
    ]
    shift_rows = [{"shift_over_0_10": 1} for _ in range(3)]
    gap_rows = [
        {
            "branch": PROPOSED_BRANCH,
            "actual_minus_expected": 0.15,
        }
        for _ in range(3)
    ]
    quality_rows = [
        {
            "branch": PROPOSED_BRANCH,
            "h5_benefit_retained_fraction": 0.4,
            "update_coverage_any": 1.0,
        },
        {"branch": BASELINE_BRANCH, "update_coverage_any": 1.0},
    ]
    jloc_rows = [
        {
            "base_seed": 1,
            "branch": BASELINE_BRANCH,
            "j_loc": 0.35,
            "j_loc_saturation_fraction": 0.2,
        },
        {
            "base_seed": 1,
            "branch": PROPOSED_BRANCH,
            "j_loc": 0.35,
            "j_loc_saturation_fraction": 0.2,
        },
    ]
    outlier_rows = [
        {
            "delta_position_rmse_norm_m": -0.02,
            "delta_actual_communication_cost": 0.1,
        },
        {
            "delta_position_rmse_norm_m": -0.01,
            "delta_actual_communication_cost": 0.1,
        },
    ]
    source_report = {
        "audit_ok": True,
        "checks": {"shared_initialization_hashes_equal": True},
    }
    sensor_report = {
        "jloc_objective_break": True,
        "decision": {"persistence_primary_cause": "visual_odometry_interaction"},
    }

    result = _attributions(
        collision_rows,
        shift_rows,
        gap_rows,
        quality_rows,
        jloc_rows,
        outlier_rows,
        source_report,
        sensor_report,
    )

    assert result["collision_mixture_dominates_reward_curve"]["status"] == "confirmed"
    assert result["critic_target_variance_dominates_loss"]["status"] == "confirmed"
    assert result["ekf_benefit_not_persistent"]["status"] == "confirmed"
    assert result["shared_initialization_confound"]["status"] == "ruled_out"


def test_config_rejects_test_split(tmp_path: Path) -> None:
    try:
        MainlineCurveCauseAuditConfig(
            reference_root=tmp_path,
            split="test",
        )
    except ValueError as exc:
        assert "validation-only" in str(exc)
    else:
        raise AssertionError("test split was accepted by a validation-only audit")


def test_validation_seed_row_accepts_legacy_nan_split() -> None:
    assert _is_validation_seed_row({"split": "validation"})
    assert _is_validation_seed_row({"split": "nan"})
    assert _is_validation_seed_row({"split": ""})
    assert not _is_validation_seed_row({"split": "test"})
