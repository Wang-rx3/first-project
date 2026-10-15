"""Pure synthetic contract tests for the v11 counterfactual backend."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.analysis import partial_gnss_replay_v9 as replay_v9
from src.analysis.neighbor_selection_counterfactual_v11 import (
    apply_branch,
    covariance_intersection,
    evaluate_point,
    freeze_filter,
    thaw_filter,
)
from src.experiments import partial_gnss_baseline_v7 as v7


def _sim():
    return v7.simulation()


def _filter(sim=None):
    sim = sim or _sim()
    state = np.zeros(15, dtype=float)
    state[:3] = [1.0, 2.0, 3.0]
    return replay_v9.StartupMeasurementCI(
        state,
        sim.ekf,
        sensor_config=sim.sensor,
        startup_steps=5,
        vo_motion_limit=16.27,
    )


def _snapshot(*, budget=3.0, valid=True, effective=True, state_time=1.2, last_update=None, cached_send=None):
    sim = _sim()
    estimator = _filter(sim)
    peer_state = np.zeros(15, dtype=float)
    peer_state[:3] = [11.0, 2.0, 3.0]
    peer_covariance = np.eye(15, dtype=float)
    return sim, {
        "step": 6,
        "time_s": 1.2,
        "own_filter": freeze_filter(estimator),
        "budget_remaining_before": budget,
        "peers": {
            "uav_00": {
                "state": peer_state,
                "covariance": peer_covariance,
                "send_time_s": 1.0,
                "state_time_s": state_time,
                "cache_mean": None,
                "cache_covariance": None,
                "cache_time_s": None,
                "cache_message_send_s": cached_send,
                "last_update_time_s": last_update,
            }
        },
        "ranges": {
            "uav_00": {
                "present": True,
                "valid": valid,
                "measurement": 10.0,
                "variance": 0.01,
                "nlos_bias": 0.0,
                "effective_edge": effective,
            }
        },
    }


def test_filter_round_trip_preserves_temporal_context_and_next_local_vo() -> None:
    sim = _sim()
    original = _filter(sim)
    imu = {"accel": np.array([0.01, -0.02, 0.03]), "gyro": np.array([0.0, 0.01, -0.01])}
    for _ in range(5):
        original.predict(imu, 0.2)
    snapshot = freeze_filter(original)
    json.dumps(snapshot, allow_nan=False)
    clone = thaw_filter(snapshot, sim)
    assert clone.config is sim.ekf
    assert clone.sensor_config is sim.sensor

    before_state = original.state.copy()
    before_covariance = original.covariance.copy()
    transition = original.predict(imu, 0.2)
    clone_transition = clone.predict(imu, 0.2)
    np.testing.assert_array_equal(original.state, clone.state)
    np.testing.assert_array_equal(original.covariance, clone.covariance)
    np.testing.assert_array_equal(transition, clone_transition)
    kwargs = {
        "previous_state": before_state,
        "previous_covariance": before_covariance,
        "transition_matrix": transition,
        "dt_s": 0.2,
        "previous_time_s": 1.0,
        "current_time_s": 1.2,
    }
    assert original.update_visual_delta(np.array([0.01, 0.0, 0.0]), **kwargs) == clone.update_visual_delta(
        np.array([0.01, 0.0, 0.0]), **kwargs
    )
    np.testing.assert_array_equal(original.state, clone.state)
    np.testing.assert_array_equal(original.covariance, clone.covariance)


def test_none_and_invalid_requests_have_explicit_cost_and_no_state_change() -> None:
    sim, snapshot = _snapshot(valid=False)
    before = thaw_filter(snapshot["own_filter"], sim)
    none_filter, none_event = apply_branch(snapshot, None, sim)
    np.testing.assert_array_equal(none_filter.state, before.state)
    np.testing.assert_array_equal(none_filter.covariance, before.covariance)
    assert none_event["request_cost"] == 0
    assert not none_event["attempted"]

    invalid_filter, invalid_event = apply_branch(snapshot, "uav_00", sim)
    np.testing.assert_array_equal(invalid_filter.state, before.state)
    np.testing.assert_array_equal(invalid_filter.covariance, before.covariance)
    assert invalid_event["request_cost"] == 1
    assert not invalid_event["attempted"]
    assert invalid_event["reason"] == "uwb_invalid"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"time_s": 2.0}, "message_stale"),
        ({"peers": {"uav_00": {"state": np.zeros(15), "covariance": np.eye(15), "send_time_s": 1.0, "state_time_s": 1.2, "cache_mean": None, "cache_covariance": None, "cache_time_s": None, "cache_message_send_s": None, "last_update_time_s": 1.0}}}, "update_period_not_elapsed"),
        ({"peers": {"uav_00": {"state": np.zeros(15), "covariance": np.eye(15), "send_time_s": 1.0, "state_time_s": 1.2, "cache_mean": None, "cache_covariance": None, "cache_time_s": None, "cache_message_send_s": 1.0, "last_update_time_s": None}}}, "reused_neighbor_packet"),
    ],
)
def test_ttl_period_and_reused_packet_gates_are_ordered(changes, reason) -> None:
    sim, snapshot = _snapshot()
    altered = copy.deepcopy(snapshot)
    altered.update(changes)
    _, event = apply_branch(altered, "uav_00", sim)
    assert event["reason"] == reason
    assert event["request_cost"] == 1
    assert not event["attempted"]


def test_old_send_is_stale_even_when_propagated_state_time_is_current() -> None:
    sim, snapshot = _snapshot()
    snapshot["peers"]["uav_00"]["send_time_s"] = 0.4
    snapshot["peers"]["uav_00"]["state_time_s"] = snapshot["time_s"]
    _, event = apply_branch(snapshot, "uav_00", sim)
    assert event["reason"] == "message_stale"
    assert event["request_cost"] == 1


def test_cache_ci_matches_native_51_point_helper_and_kernel_receives_cache_value() -> None:
    sim, snapshot = _snapshot()
    peer = snapshot["peers"]["uav_00"]
    peer["cache_mean"] = np.array([10.5, 2.0, 3.0])
    peer["cache_covariance"] = np.eye(3) * 0.5
    peer["cache_time_s"] = 1.0
    peer["cache_message_send_s"] = 0.8
    expected_mean, expected_covariance, expected_weight = covariance_intersection(
        peer["cache_mean"], peer["cache_covariance"], peer["state"][:3], peer["covariance"][:3, :3]
    )
    _, event = apply_branch(snapshot, "uav_00", sim)
    np.testing.assert_allclose(event["reference_position"], expected_mean)
    np.testing.assert_allclose(event["reference_covariance"], expected_covariance)
    assert event["cache_weight"] == pytest.approx(expected_weight)
    np.testing.assert_allclose(snapshot["peers"]["uav_00"]["cache_mean"], [10.5, 2.0, 3.0])


def _frame(step: int) -> dict:
    packet = {
        "imu": {"accel": np.zeros(3), "gyro": np.zeros(3)},
        "vo": {"valid": False, "delta_position": None, "covariance": np.eye(3) * 0.08**2},
        "gnss": {"valid": False, "position": None, "std": None},
    }
    return {
        "step": step,
        "observable": {"sensor_packets": {"uav_02": packet}, "pre_legal_sources": ["uav_00", "uav_01"]},
        "evaluation_truth": {"position": {"uav_02": np.zeros(3)}},
        "audit": {"selection": "none"},
    }


def test_evaluate_point_replays_three_future_local_packets_and_retains_invalid_outcome() -> None:
    sim, snapshot = _snapshot(valid=False)
    # The point is step six; frames[6:9] are exactly the three future packets.
    record = {"frames": [_frame(step) for step in range(1, 10)]}
    score_row = {
        "step": 6,
        "legal_sources": ["uav_00", "uav_01"],
        "candidates": {
            "uav_00": {"valid": True, "score": 0.7},
            "uav_01": {"valid": True, "score": 0.4},
        },
    }
    result = evaluate_point(record, snapshot, score_row, {"simulation": sim})
    assert result["local_states"] == 12
    assert result["relative_ci_calls"] == 0
    assert result["dual_eligible"] is True
    assert result["branches"]["uav_00"]["request_cost"] == 1
    assert result["branches"]["uav_00"]["reason"] == "uwb_invalid"
    assert result["branches"]["uav_01"]["reason"] in {"selected_pair_not_observed", "message_not_arrived"}
    assert result["physical_steps"] == 0
    assert result["optimizer_steps"] == 0


def test_invalid_actual_selected_peer_still_checks_none_like_identity() -> None:
    sim, snapshot = _snapshot(valid=False)
    base = thaw_filter(snapshot["own_filter"], sim)
    record = {"frames": [_frame(step) for step in range(1, 10)]}
    current = record["frames"][5]
    current["audit"]["selection"] = "uav_00"
    current["audit"]["states"] = {"uav_02": base.state.copy()}
    current["audit"]["covariances"] = {"uav_02": base.covariance.copy()}
    current["audit"]["gate_events"] = [
        {
            "source_uav_id": "uav_00",
            "target_uav_id": "uav_02",
            "selected_non_none": True,
            "uwb_valid": False,
            "ekf_update_attempted": False,
            "ekf_update_accepted": False,
        }
    ]
    scores = {"candidates": {"uav_00": {"valid": True, "score": 0.7}, "uav_01": {"valid": True, "score": 0.4}}}
    result = evaluate_point(record, snapshot, scores, {"simulation": sim})
    assert result["identity_matches"] is True
    assert result["identity_error_max"] == pytest.approx(0.0)


def test_preferred_peer_is_utility_winner_when_scores_disagree() -> None:
    sim, snapshot = _snapshot(valid=True)
    snapshot["peers"]["uav_01"] = copy.deepcopy(snapshot["peers"]["uav_00"])
    snapshot["peers"]["uav_01"]["state"] = np.r_[[-9.0, 2.0, 3.0], np.zeros(12)]
    snapshot["peers"]["uav_01"]["send_time_s"] = 0.9
    snapshot["ranges"]["uav_01"] = copy.deepcopy(snapshot["ranges"]["uav_00"])
    snapshot["ranges"]["uav_00"]["measurement"] = 9.0
    record = {"frames": [_frame(step) for step in range(1, 10)]}
    for frame in record["frames"]:
        frame["evaluation_truth"]["position"]["uav_02"] = np.array([10.0, 2.0, 3.0])
    result = evaluate_point(
        record,
        snapshot,
        {"candidates": {"uav_00": {"valid": True, "score": 0.1}, "uav_01": {"valid": True, "score": 0.9}}},
        {"simulation": sim},
    )
    assert result["scored_peer"] == "uav_01"
    assert result["preferred_peer"] == "uav_00"
    assert result["ranking_correct"] == 0.0

