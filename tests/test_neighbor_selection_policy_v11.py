"""Synthetic contract tests for the isolated v11 neighbor selector."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.analysis.neighbor_selection_policy_v11 import (
    PEERS,
    TARGET,
    choose_greedy,
    peer_reference,
    score_context,
)
from src.analysis.partial_gnss_fusion_v7 import MeasurementCIEKF
from src.simulation.ekf import covariance_intersection
from src.simulation.environment import SimulationConfig
from src.simulation.random_streams import indexed_rng


def _packet(
    position: tuple[float, float, float],
    *,
    covariance: np.ndarray | None = None,
    send_time: float | None = 1.0,
    state_time: float | None = 1.0,
    rssi: float | None = -50.0,
    aoi: float | None = 0.1,
    cache_mean: np.ndarray | None = None,
    cache_covariance: np.ndarray | None = None,
    cache_time: float | None = None,
    cache_send_time: float | None = None,
) -> dict[str, object]:
    state = np.zeros(15)
    state[:3] = position
    full_covariance = np.eye(15) * 0.1 if covariance is None else np.asarray(covariance, dtype=float)
    return {
        "state": state,
        "covariance": full_covariance,
        "send_time_s": send_time,
        "state_time_s": state_time,
        "cache_mean": cache_mean,
        "cache_covariance": cache_covariance,
        "cache_time_s": cache_time,
        "cache_message_send_s": cache_send_time,
        "last_update_time_s": None,
        "rssi_dbm": rssi,
        "aoi_s": aoi,
    }


def _context(*, peer0: dict[str, object] | None = None, peer1: dict[str, object] | None = None, **updates: object) -> dict[str, object]:
    identity = np.eye(15)
    context: dict[str, object] = {
        "step": 4,
        "seed": 20261013,
        "time_s": 1.0,
        "own_state": np.zeros(15),
        "own_covariance": np.diag([4.0, 4.0, 4.0] + [1.0] * 12),
        "legal_sources": list(PEERS),
        "peers": {
            "uav_00": _packet((2.0, 0.0, 0.0), covariance=0.01 * identity),
            "uav_01": _packet((0.0, 2.0, 0.0), covariance=0.01 * identity),
        },
        "measurement_budget_remaining": 1,
    }
    if peer0 is not None:
        context["peers"]["uav_00"] = peer0  # type: ignore[index]
    if peer1 is not None:
        context["peers"]["uav_01"] = peer1  # type: ignore[index]
    context.update(updates)
    return context


def test_score_context_is_pure_and_json_normalizable() -> None:
    context = _context()
    before = copy.deepcopy(context)
    result = score_context(context, SimulationConfig())
    assert json.loads(json.dumps(result))["clone_calls"] == 2
    np.testing.assert_array_equal(context["own_state"], before["own_state"])  # type: ignore[arg-type]
    np.testing.assert_array_equal(
        context["peers"]["uav_00"]["state"],  # type: ignore[index]
        before["peers"]["uav_00"]["state"],  # type: ignore[index]
    )


@pytest.mark.parametrize(
    ("field", "reason"),
    [("rssi_dbm", "missing_rssi_dbm"), ("send_time_s", "missing_send_time_s")],
)
def test_missing_required_peer_fields_are_nullable(field: str, reason: str) -> None:
    packet = _packet((2.0, 0.0, 0.0))
    packet[field] = None
    result = score_context(_context(peer0=packet), SimulationConfig())
    candidate = result["candidates"]["uav_00"]
    assert candidate["valid"] is False
    assert candidate["score"] is None
    assert candidate["reason"] == reason
    assert candidate["predicted_accepted"] is None


def test_missing_message_is_nullable_and_current_age_uses_state_time() -> None:
    context = _context()
    context["peers"]["uav_00"] = None  # type: ignore[index]
    result = score_context(context, SimulationConfig())
    assert result["candidates"]["uav_00"]["reason"] == "message_not_arrived"

    delayed = _packet((2.0, 0.0, 0.0), send_time=0.8, state_time=1.0)
    candidate = score_context(_context(peer0=delayed), SimulationConfig())["candidates"]["uav_00"]
    assert candidate["valid"] is True


def test_aoi_and_staleness_use_send_time_and_state_time_is_current() -> None:
    delayed = _packet((2.0, 0.0, 0.0), send_time=0.6, state_time=1.0, aoi=99.0)
    result = score_context(_context(peer0=delayed), SimulationConfig())
    candidate = result["candidates"]["uav_00"]
    assert candidate["valid"] is True
    assert result["features"]["aoi0_s"] == pytest.approx(0.4)
    assert candidate["aoi_s"] == pytest.approx(0.4)
    expected_cost = 0.5 * (1.0 - ((-50.0 + 100.0) / 110.0) + 0.4 / 0.6)
    assert candidate["communication_cost"] == pytest.approx(expected_cost)

    old_send = _packet((2.0, 0.0, 0.0), send_time=0.2, state_time=1.0)
    old_candidate = score_context(_context(peer0=old_send), SimulationConfig())["candidates"]["uav_00"]
    assert old_candidate["valid"] is False
    assert old_candidate["reason"] == "stale_or_future_message"

    mismatched_state_time = _packet((2.0, 0.0, 0.0), send_time=0.8, state_time=0.9)
    mismatched = score_context(
        _context(peer0=mismatched_state_time), SimulationConfig()
    )["candidates"]["uav_00"]
    assert mismatched["valid"] is False
    assert mismatched["reason"] == "state_time_mismatch"


def test_duplicate_packet_is_skipped_but_new_packet_is_scored() -> None:
    duplicate = _packet(
        (2.0, 0.0, 0.0),
        covariance=0.01 * np.eye(15),
        cache_mean=np.array([2.0, 0.0, 0.0]),
        cache_covariance=0.01 * np.eye(3),
        cache_time=1.0,
        cache_send_time=1.0,
    )
    duplicate_candidate = score_context(_context(peer0=duplicate), SimulationConfig())["candidates"]["uav_00"]
    assert duplicate_candidate["valid"] is False
    assert duplicate_candidate["reason"] == "duplicate_packet"
    new_packet = copy.deepcopy(duplicate)
    new_packet["send_time_s"] = 1.1
    new_packet["state_time_s"] = 1.1
    new_candidate = score_context(
        _context(peer0=new_packet, time_s=1.1), SimulationConfig()
    )["candidates"]["uav_00"]
    assert new_candidate["valid"] is True
    assert new_candidate["predicted_ci_weight"] != 1.0


def test_peer_reference_matches_native_ci_and_skips_stale_cache() -> None:
    current = _packet((4.0, 1.0, 0.0), covariance=0.1 * np.eye(15), send_time=2.0, state_time=2.0)
    current["cache_mean"] = np.array([1.0, 0.0, 0.0])
    current["cache_covariance"] = 0.2 * np.eye(3)
    current["cache_time_s"] = 1.8
    current["cache_message_send_s"] = 1.0
    mean, covariance, weight = peer_reference(current, 2.0, 0.6)
    expected_mean, expected_covariance, expected_weight = covariance_intersection(
        current["cache_mean"],  # type: ignore[arg-type]
        current["cache_covariance"],  # type: ignore[arg-type]
        current["state"][:3],  # type: ignore[index]
        current["covariance"][:3, :3],  # type: ignore[index]
    )
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(covariance, expected_covariance)
    assert weight == expected_weight

    current["cache_time_s"] = 1.0
    stale_mean, stale_covariance, stale_weight = peer_reference(current, 2.0, 0.6)
    np.testing.assert_array_equal(stale_mean, current["state"][:3])  # type: ignore[index]
    np.testing.assert_array_equal(stale_covariance, current["covariance"][:3, :3])  # type: ignore[index]
    assert stale_weight == 1.0


def test_predicted_relative_ci_weight_is_separate_from_cache_ci_weight() -> None:
    packet = _packet(
        (2.0, 0.0, 0.0),
        covariance=0.01 * np.eye(15),
        send_time=1.1,
        state_time=1.1,
        cache_mean=np.array([1.0, 0.0, 0.0]),
        cache_covariance=0.2 * np.eye(3),
        cache_time=0.8,
        cache_send_time=1.0,
    )
    context = _context(peer0=packet, time_s=1.1)
    sim = SimulationConfig()
    scoring = score_context(context, sim)
    candidate = scoring["candidates"]["uav_00"]
    reference_mean, reference_covariance, cache_weight = peer_reference(packet, 1.1, 0.6)
    clone = MeasurementCIEKF(context["own_state"].copy(), sim.ekf)  # type: ignore[union-attr]
    clone.covariance = context["own_covariance"].copy()  # type: ignore[union-attr]
    assert clone.update_relative_range(
        float(np.linalg.norm(context["own_state"][:3] - reference_mean)),  # type: ignore[index]
        reference_mean,
        reference_covariance,
        measurement_variance=sim.sensor.uwb_los_std_m**2,
        nlos_bias_m=0.0,
        timestamp=1.1,
    )
    assert candidate["predicted_cache_ci_weight"] == pytest.approx(cache_weight)
    assert candidate["predicted_ci_weight"] == pytest.approx(
        clone.last_relative_range_update["weight"]
    )


def test_nonisotropic_geometry_uses_directional_full_3d_covariance() -> None:
    own_covariance = np.diag([1.0, 2.0, 30.0] + [1.0] * 12)
    context = _context()
    context["own_covariance"] = own_covariance
    context["peers"]["uav_00"]["state"][:3] = [0.0, 0.0, 3.0]  # type: ignore[index]
    result = score_context(context, SimulationConfig())
    candidate = result["candidates"]["uav_00"]
    assert candidate["direction"] == [0.0, 0.0, 1.0]
    assert candidate["geometry"] == pytest.approx(30.0 / 33.0)


def test_nominal_clone_is_psd_and_information_matches_trace_change() -> None:
    result = score_context(_context(), SimulationConfig())
    candidate = result["candidates"]["uav_00"]
    own_trace = result["features"]["self_position_trace_m2"]
    predicted_trace = candidate["predicted_position_trace_m2"]
    assert candidate["predicted_accepted"] is True
    assert candidate["trace_gain_m2"] == pytest.approx(own_trace - predicted_trace)
    assert candidate["information_raw"] == pytest.approx(
        (own_trace - predicted_trace) / own_trace
    )
    assert candidate["information_fraction"] == pytest.approx(
        np.clip(candidate["information_raw"], 0.0, 1.0)
    )


def test_zero_distance_or_zero_trace_is_guarded() -> None:
    zero_distance = _packet((0.0, 0.0, 0.0), covariance=0.01 * np.eye(15))
    candidate = score_context(_context(peer0=zero_distance), SimulationConfig())["candidates"]["uav_00"]
    assert candidate["valid"] is False
    assert candidate["reason"] == "degenerate_distance"
    zero_trace_context = _context()
    zero_trace_context["own_covariance"] = np.diag([0.0, 0.0, 0.0] + [1.0] * 12)
    with pytest.raises(RuntimeError):
        score_context(zero_trace_context, SimulationConfig())


def test_reliability_q_is_constant_for_same_peer_covariance() -> None:
    first = score_context(_context(), SimulationConfig())["candidates"]["uav_00"]
    second_context = _context()
    second_context["peers"]["uav_00"]["state"][:3] = [0.0, 3.0, 0.0]  # type: ignore[index]
    second = score_context(second_context, SimulationConfig())["candidates"]["uav_00"]
    assert first["reliability"] == pytest.approx(second["reliability"])


def test_negative_score_is_preserved_for_a_valid_but_unhelpful_candidate() -> None:
    packet = _packet(
        (2.0, 0.0, 0.0),
        covariance=100.0 * np.eye(15),
        send_time=0.4,
        state_time=1.0,
        rssi=-100.0,
    )
    candidate = score_context(_context(peer0=packet), SimulationConfig())["candidates"]["uav_00"]
    assert candidate["valid"] is True
    assert candidate["score"] < 0.0
    assert choose_greedy({"seed": 1, "step": 1, "candidates": {"uav_00": candidate}}) == "none"


def test_none_and_tie_selection_are_deterministic() -> None:
    assert choose_greedy({"seed": 1, "step": 2, "candidates": {}}) == "none"
    scoring = {
        "seed": 20261013,
        "step": 4,
        "candidates": {
            "uav_00": {"valid": True, "score": 0.5},
            "uav_01": {"valid": True, "score": 0.5 + 5.0e-13},
        },
    }
    expected = str(indexed_rng(scoring["seed"], scoring["step"], "v11_greedy_tie", TARGET).choice(list(PEERS)))
    assert choose_greedy(scoring) == expected
    assert choose_greedy(scoring) == expected
