"""Tests for leakage-safe visual-odometry quality covariance calibration."""

from __future__ import annotations

import numpy as np

from src.learning.visual_odometry_quality import (
    covariance_for_quality,
    covariance_scale_for_quality,
    fit_visual_odometry_quality_model,
)


def _warmup_records() -> list[dict[str, object]]:
    return [
        {"uav_id": "uav_00", "valid": 1, "tracking_quality": 0.20, "nis_vo": 4.0},
        {"uav_id": "uav_00", "valid": 1, "tracking_quality": 0.45, "nis_vo": 2.0},
        {"uav_id": "uav_00", "valid": 1, "tracking_quality": 0.70, "nis_vo": 1.2},
        {"uav_id": "uav_00", "valid": 1, "tracking_quality": 0.90, "nis_vo": 0.8},
    ]


def test_quality_covariance_is_monotone_and_finite() -> None:
    model = fit_visual_odometry_quality_model(
        _warmup_records(), base_position_std_m=0.03
    )
    low_scale, low_valid, _ = covariance_scale_for_quality(0.10, model)
    high_scale, high_valid, _ = covariance_scale_for_quality(0.95, model)
    assert low_valid and high_valid
    assert np.isfinite([low_scale, high_scale]).all()
    assert low_scale >= high_scale >= 1.0

    low_covariance, *_ = covariance_for_quality(0.03, 0.10, model)
    high_covariance, *_ = covariance_for_quality(0.03, 0.95, model)
    assert np.isfinite(low_covariance).all()
    assert np.isfinite(high_covariance).all()
    assert float(np.trace(low_covariance)) >= float(np.trace(high_covariance))


def test_missing_quality_uses_conservative_finite_fallback() -> None:
    model = fit_visual_odometry_quality_model(
        _warmup_records(), base_position_std_m=0.03
    )
    covariance, scale, valid, bin_index = covariance_for_quality(0.03, None, model)
    assert valid is False
    assert bin_index == -1
    assert scale == 4.0
    assert np.isfinite(covariance).all()
    assert float(np.trace(covariance)) > 0.0


def test_fit_ignores_truth_and_split_leakage_fields() -> None:
    records = _warmup_records()
    altered = [
        {
            **record,
            "true_position": [999.0, 999.0, 999.0],
            "true_error": 999.0,
            "physical_los_nlos": "nlos",
            "future_state": {"position": [999.0, 999.0, 999.0]},
            "split": "test",
        }
        for record in records
    ]
    base = fit_visual_odometry_quality_model(records, base_position_std_m=0.03)
    changed = fit_visual_odometry_quality_model(altered, base_position_std_m=0.03)
    assert base == changed
    assert base["source_split"] == "warmup"
    assert base["validation_used_for_fitting"] is False
    assert base["test_used_for_fitting"] is False


def test_invalid_records_do_not_make_the_model_nonfinite() -> None:
    model = fit_visual_odometry_quality_model(
        [
            {"valid": 0, "tracking_quality": 0.1, "nis_vo": 10.0},
            {"valid": 1, "tracking_quality": None, "nis_vo": None},
            {"valid": 1, "tracking_quality": 0.8, "nis_vo": float("nan")},
        ],
        base_position_std_m=0.03,
    )
    assert model["sample_count"] == 1
    assert all(np.isfinite(float(value)) for value in model["covariance_scale_by_bin"])
    assert all(
        value is None or np.isfinite(float(value))
        for value in model["mean_nis_by_bin"]
    )
