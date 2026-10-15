"""Focused tests for the bounded offline learnability audit."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.analysis.three_uav_measurement_learnability_audit import (
    _counterfactual_row,
    _fresh_source,
    _replay_selected,
    expected_linear_credit,
    mask_loop_audit,
    native_measurement_mask,
    scalar_covariance_counterexample,
    run_audit,
)
from src.simulation.ekf import LocalEKF


def _message(position=(1.0, 0.0, 0.0)):
    return {"estimate": list(position) + [0.0] * 12, "covariance": np.eye(15).tolist()}


def _observation():
    return {
        "uav_id": "uav_00",
        "estimate": [0.0] * 15,
        "covariance": np.eye(15).tolist(),
        "candidate_measurement_sources": ["uav_01", "uav_02"],
        "effective_sources": ["uav_01", "uav_02"],
        "uwb_available": {"uav_01": True, "uav_02": True},
        "neighbor_messages": {"uav_01": _message(), "uav_02": _message((0.0, 1.0, 0.0))},
        "measurement_update_eligibility": [False, True, True],
        "measurement_budget_remaining": 2.0,
        "action_mask": [False, True, True, True],
    }


def test_native_mask_validates_physical_intersection_and_self_exclusion():
    observation = _observation()
    assert native_measurement_mask(observation) == [False, True, True, True]
    observation["uwb_available"]["uav_02"] = False
    assert native_measurement_mask(observation) == [False, True, False, True]
    observation["measurement_budget_remaining"] = 0.0
    assert native_measurement_mask(observation) == [False, False, False, True]


def test_mask_loop_and_credit_formula_are_deterministic():
    observation = _observation()
    report = mask_loop_audit([{"scene_id": "synthetic", "step": 0, "observations": {"uav_00": observation}}])
    assert report["checked"] == 1
    assert report["mismatch_count"] == 0
    assert expected_linear_credit(observation, "uav_01") > 0.0


def test_selected_replay_and_counterfactual_do_not_pollute_inputs():
    observation = _observation()
    state_before = np.zeros(15)
    ekf = LocalEKF(state_before)
    ekf.covariance = np.eye(15)
    assert ekf.update_relative_range(1.1, np.asarray(observation["neighbor_messages"]["uav_01"]["estimate"])[:3], np.eye(15), measurement_variance=0.0064, timestamp=0.2)
    event = {
        "source_uav_id": "uav_01",
        "target_uav_id": "uav_00",
        "state_before": state_before.tolist(),
        "state_after": ekf.state.tolist(),
        "measured_range_m": 1.1,
        "measurement_variance_m2": 0.0064,
        "nlos_bias_m": 0.0,
        "measurement_time_s": 0.2,
        "position_covariance_trace_after_final_m2": float(np.trace(ekf.covariance[:3, :3])),
        "covariance_restored": True,
        "ci_weight": 1.0,
    }
    before = copy.deepcopy(observation)
    replay = _replay_selected(event, observation, np.zeros(3))
    replay_repeat = _replay_selected(event, observation, np.zeros(3))
    assert replay["state_max_abs_error"] == 0.0
    assert replay == replay_repeat
    row = _counterfactual_row({**observation, "_scene_id": "synthetic", "_step": 2, "_uav_id": "uav_00"}, event, np.zeros(3), np.asarray([1.0, 0.0, 0.0]), "B", 0.0)
    assert row["cost_packet"] == 1
    assert observation == before


def test_period_audit_ignores_failed_uwb_gate_and_preserves_unknown_packet_time():
    gates = [
        {"scene_id": "s", "source_uav_id": "uav_01", "target_uav_id": "uav_00", "seed": "1", "step": "4", "ekf_update_attempted": "True"},
        {"scene_id": "s", "source_uav_id": "uav_01", "target_uav_id": "uav_00", "seed": "1", "step": "8", "ekf_update_attempted": "False"},
    ]
    result = _fresh_source([], gates, "s", 8, "uav_01", "uav_00", "1", post_message_available=True)
    assert result["period_passed"] is True
    assert result["message_age_s"] is None
    assert result["new_send_time"] is None
    assert result["timestamp_known"] is False


def test_scalar_counterexample_matches_locked_witness():
    witness = scalar_covariance_counterexample()
    assert np.isclose(witness["true_posterior_variance"], 1.2344713493, atol=1e-10)
    assert witness["restored_is_smaller"] is True
    assert np.allclose(witness["joint_covariance_eigenvalues"], [0.0, 5.0])


def test_output_scope_and_frozen_source_are_enforced(tmp_path):
    with pytest.raises(ValueError, match='Unsafe run-id'):
        run_audit(tmp_path, '../old_results')
    with pytest.raises(ValueError, match='locked to the v3'):
        run_audit(tmp_path, 'fresh', 'another_run')

