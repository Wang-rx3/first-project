from types import SimpleNamespace

import pytest

from src.experiments import training_trend_diagnostic_v2 as diagnostic


def test_action_space_mapping_is_explicit_and_round_trips():
    assert diagnostic._action_index("uav_00", 3) == 0
    assert diagnostic._action_index("uav_02", 3) == 2
    assert diagnostic._action_index("none", 3) == 3
    assert diagnostic._action_label(0, 3) == "uav_00"
    assert diagnostic._action_label(3, 3) == "none"
    with pytest.raises(ValueError):
        diagnostic._action_index("uav_03", 3)


def test_applied_action_replaces_invalid_or_sanitized_request_for_accounting():
    applied, index = diagnostic._resolve_applied_measurement_action(
        {"uav_00": {"measurement_uav_id": "none"}},
        "uav_00",
        "uav_02",
        3,
    )
    assert (applied, index) == ("none", 3)

    applied, index = diagnostic._resolve_applied_measurement_action(
        {},
        "uav_01",
        "uav_01",
        3,
    )
    assert (applied, index) == ("uav_01", 1)


def test_quality_classes_use_validation_label_threshold_and_close():
    rows = [
        {"accepted": True, "true_position_error_before_m": 1.0, "true_position_error_after_m": 0.998},
        {"accepted": True, "true_position_error_before_m": 1.0, "true_position_error_after_m": 1.0015},
        {"accepted": True, "true_position_error_before_m": 1.0, "true_position_error_after_m": 1.0005},
        {"accepted": False, "true_position_error_before_m": 1.0, "true_position_error_after_m": 0.0},
    ]
    assert diagnostic._quality_counts(rows) == {
        "accepted": 3,
        "beneficial": 1,
        "neutral": 1,
        "harmful": 1,
    }


def test_retention_censors_missing_future_windows():
    events = [
        {
            "accepted": True,
            "target_uav_id": "uav_00",
            "step": 0,
            "true_position_error_before_m": 1.0,
            "true_position_error_after_m": 0.998,
        }
    ]
    windows = [
        {"step": 0, "position_error_uav_00": 0.998},
        {"step": 1, "position_error_uav_00": 0.997},
    ]
    result = diagnostic._retention(events, windows, ["uav_00"])
    assert result["h1_observed_count"] == 1
    assert result["h1_retained_fraction"] == pytest.approx(1.0)
    assert result["h5_censored_count"] == 1
    assert result["h5_retained_fraction"] != result["h5_retained_fraction"]


def test_synthetic_positive_constraint_reaches_measurement_head_without_update():
    result = diagnostic._synthetic_positive_gradient_case()
    assert result["residual"] > 0.0
    assert result["measurement_head_gradient_norm"] > 0.0
    assert result["parameters_unchanged"] is True
    assert result["optimizer_step_called"] == 0


def test_v2_run_rejects_non_seed42_and_non_audit_stage():
    with pytest.raises(ValueError):
        diagnostic.run("frozen-audit", 2026)
    with pytest.raises(ValueError):
        diagnostic.run("smoke", 42)
