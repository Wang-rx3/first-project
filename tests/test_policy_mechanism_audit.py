"""Tests for the validation-only policy mechanism audit."""

from __future__ import annotations

import numpy as np

from src.experiments.policy_mechanism_audit import (
    PolicyMechanismAuditConfig,
    _finite_trace_value,
    _trace_action_summary,
    _trace_budget_summary,
    _trace_overhead_summary,
    _trace_transition_aggregates,
    _trace_transition_summary,
    compute_risk_probability_correlation,
)


def _trace_row(*, selected: str, none_probability: float, risk: float, probability: float) -> dict[str, object]:
    return {
        "base_seed": 2026,
        "branch": "quality_aware_mappo",
        "selected_measurement_uav_id": selected,
        "none_probability": none_probability,
        "quality_risk": [risk, risk, risk],
        "quality_valid": [1.0, 1.0, 1.0],
        "measurement_probabilities": [probability, probability, probability, none_probability],
        "selected_quality_risk": risk if selected != "none" else -1.0,
        "selected_probability": probability if selected != "none" else none_probability,
        "overhead_event_present": 1 if selected != "none" else 0,
        "load_packets_requested": 1.0 if selected != "none" else 0.0,
        "load_packets_applied": 1.0 if selected != "none" else 0.0,
        "queue_before_packets": 0.0,
        "queue_with_measurement_packets": 1.0 if selected != "none" else 0.0,
        "queue_next_packets": 1.0 if selected != "none" else 0.0,
        "rate_bps": 1000.0,
        "packet_loss": 0.1,
        "delay_s": 0.01,
        "aoi_s": 0.1,
    }


def test_action_summary_distinguishes_none_from_measurement_rate() -> None:
    rows = [
        _trace_row(selected="none", none_probability=0.8, risk=0.2, probability=0.05),
        _trace_row(selected="uav_01", none_probability=0.2, risk=0.1, probability=0.6),
    ]

    summary = _trace_action_summary(rows)

    assert len(summary) == 1
    assert summary[0]["none_selection_rate"] == 0.5
    assert summary[0]["measurement_selection_rate"] == 0.5


def test_overhead_summary_has_one_packet_per_selected_measurement() -> None:
    rows = [
        _trace_row(selected="none", none_probability=0.8, risk=0.2, probability=0.05),
        _trace_row(selected="uav_01", none_probability=0.2, risk=0.1, probability=0.6),
    ]

    summary = _trace_overhead_summary(rows)

    assert summary[0]["measurement_count"] == 1
    assert summary[0]["overhead_event_count"] == 1
    assert summary[0]["load_packets_requested"] == 1.0
    assert summary[0]["load_packets_applied"] == 1.0


def test_quality_probability_correlation_is_negative_for_lower_risk() -> None:
    correlation = compute_risk_probability_correlation(
        [0.1, 0.5, 0.9], [0.8, 0.4, 0.1], [1.0, 1.0, 1.0]
    )

    assert correlation < -0.9


def test_audit_rejects_non_cpu_and_non_validation_configuration() -> None:
    try:
        PolicyMechanismAuditConfig(reference_root="data", split="test")
    except ValueError as exc:
        assert "validation-only" in str(exc)
    else:
        raise AssertionError("test split must be rejected")

    try:
        PolicyMechanismAuditConfig(reference_root="data", device="cuda")
    except ValueError as exc:
        assert "cpu" in str(exc)
    else:
        raise AssertionError("non-CPU device must be rejected")


def test_audit_accepts_three_branch_configuration() -> None:
    config = PolicyMechanismAuditConfig(
        reference_root="data",
        branches=(
            "baseline_mappo",
            "quality_aware_mappo",
            "budgeted_quality_aware_mappo",
        ),
    )

    assert config.branches[-1] == "budgeted_quality_aware_mappo"


def test_transition_summary_separates_submission_and_motion_invalidation() -> None:
    rows = [
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "uav_id": "uav_00",
            "submitted_measurement_uav_id": "uav_01",
            "invalid_measurement_action": 0,
            "measurement_transition_invalidated": 1,
            "post_motion_potential_edge": 0,
            "effective_measurement": 0,
        },
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "uav_id": "uav_00",
            "submitted_measurement_uav_id": "none",
            "invalid_measurement_action": 0,
            "measurement_transition_invalidated": 0,
            "post_motion_potential_edge": 0,
            "effective_measurement": 0,
        },
    ]

    summary = _trace_transition_summary(rows)[0]

    assert summary["submitted_non_none_count"] == 1
    assert summary["invalid_measurement_action_count"] == 0
    assert summary["transition_invalidation_count"] == 1
    assert summary["transition_invalidation_denominator"] == 1
    assert summary["small_denominator"] == 1
    assert summary["actionable_transition_invalidation"] == 0
    assert summary["transition_timing_status"] == "insufficient_denominator"
    assert summary["post_motion_edge_loss_count"] == 1


def test_transition_small_denominator_does_not_trigger_global_timing_alert() -> None:
    rows = [
        {
            "base_seed": 9099,
            "branch": "budgeted_quality_aware_mappo",
            "submitted_measurement_uav_id": "uav_01",
            "invalid_measurement_action": 0,
            "measurement_transition_invalidated": int(index == 0),
            "post_motion_potential_edge": int(index != 0),
            "effective_measurement": int(index != 0),
        }
        for index in range(5)
    ]

    aggregates = _trace_transition_aggregates(rows)
    assert aggregates["global"]["transition_invalidation_count"] == 1
    assert aggregates["global"]["transition_invalidation_denominator"] == 5
    assert aggregates["global"]["small_denominator"] == 1
    assert aggregates["global"]["actionable_transition_invalidation"] == 0
    assert aggregates["global"]["transition_timing_status"] == "insufficient_denominator"


def test_transition_rate_above_five_percent_triggers_only_with_sufficient_denominator() -> None:
    rows = [
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "submitted_measurement_uav_id": "uav_01",
            "invalid_measurement_action": 0,
            "measurement_transition_invalidated": int(index < 3),
            "post_motion_potential_edge": int(index >= 3),
            "effective_measurement": int(index >= 3),
        }
        for index in range(40)
    ]

    summary = _trace_transition_summary(rows)[0]
    assert summary["transition_invalidation_denominator"] == 40
    assert summary["transition_invalidation_rate"] == 0.075
    assert summary["small_denominator"] == 0
    assert summary["actionable_transition_invalidation"] == 1
    assert summary["transition_timing_status"] == "actionable_post_motion_invalidation"


def test_budget_summary_sums_one_limit_per_uav() -> None:
    rows = [
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "uav_id": "uav_00",
            "measurement_budget_enabled": 1,
            "budget_limit": 2,
            "budget_consumed_step": 1,
            "budget_overrun": 0,
            "budget_used_after": 1,
            "budget_remaining_after": 1,
            "budget_exhausted_before": 0,
        },
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "uav_id": "uav_00",
            "measurement_budget_enabled": 1,
            "budget_limit": 2,
            "budget_consumed_step": 0,
            "budget_overrun": 0,
            "budget_used_after": 1,
            "budget_remaining_after": 1,
            "budget_exhausted_before": 0,
        },
        {
            "base_seed": 7071,
            "branch": "budgeted_quality_aware_mappo",
            "uav_id": "uav_01",
            "measurement_budget_enabled": 1,
            "budget_limit": 3,
            "budget_consumed_step": 1,
            "budget_overrun": 0,
            "budget_used_after": 1,
            "budget_remaining_after": 2,
            "budget_exhausted_before": 0,
        },
    ]

    summary = _trace_budget_summary(rows)[0]

    assert summary["budget_limit_total"] == 5
    assert summary["budget_consumed_count"] == 2
    assert summary["budget_overrun_count"] == 0


def test_trace_finiteness_handles_arrays_and_rejects_nan() -> None:
    assert _finite_trace_value({"values": np.asarray([0.0, 1.0])})
    assert not _finite_trace_value({"values": np.asarray([0.0, np.nan])})
