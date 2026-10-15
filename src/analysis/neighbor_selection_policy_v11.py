"""Pure, auditable greedy neighbor-selection policy for the v11 contract.

The policy scores the two legal peer-state packets available to ``uav_02``.
It intentionally operates on a caller-supplied snapshot: it does not read an
environment, advance a filter, sample physical randomness, or mutate the
snapshot.  The only estimator execution is a bounded clone of the frozen v7
measurement-CI EKF used to predict the information fraction of each packet.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.analysis.partial_gnss_fusion_v7 import MeasurementCIEKF
from src.simulation.ekf import STATE_DIM, covariance_intersection
from src.simulation.random_streams import indexed_rng


PEERS = ("uav_00", "uav_01")
TARGET = "uav_02"
_POSITION_DIM = 3
_PSD_TOL = 1.0e-8
_TIME_TOL = 1.0e-8
_SEND_TOL = 1.0e-12
_TRACE_EPS = 1.0e-12
_INFORMATION_SEMANTICS = "predicted_ci_covariance_proxy_not_truth_benefit"

_FEATURE_KEYS = (
    "self_position_trace_m2",
    "peer0_position_trace_m2",
    "peer1_position_trace_m2",
    "distance20_m",
    "distance21_m",
    "angle01_rad",
    "rssi0_dbm",
    "rssi1_dbm",
    "aoi0_s",
    "aoi1_s",
)

_COMPONENT_KEYS = (
    "score",
    "information_fraction",
    "information_raw",
    "trace_gain_m2",
    "geometry",
    "reliability",
    "communication_cost",
    "valid",
    "reason",
    "direction",
    "distance_m",
    "peer_trace_m2",
    "rssi_dbm",
    "aoi_s",
    "predicted_ci_weight",
    "predicted_cache_ci_weight",
    "predicted_accepted",
    "predicted_position_trace_m2",
)


def _as_finite_scalar(value: Any, name: str) -> float | None:
    """Return a finite scalar without leaking numpy scalars to the result."""

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    result = float(array.reshape(-1)[0])
    return result if math.isfinite(result) else None


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array.copy()


def _json_array(value: np.ndarray | Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        return None
    return [float(item) for item in array.reshape(-1)]


def _symmetric_psd(array: np.ndarray, *, name: str, positive_definite: bool = False) -> None:
    if not np.allclose(array, array.T, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(f"{name} must be symmetric")
    minimum = float(np.min(np.linalg.eigvalsh(0.5 * (array + array.T))))
    if not math.isfinite(minimum):
        raise RuntimeError(f"{name} has a non-finite eigenvalue")
    if minimum < -_PSD_TOL or (positive_definite and minimum <= 0.0):
        raise RuntimeError(f"{name} is materially invalid")


def _missing_candidate(reason: str) -> dict[str, Any]:
    result = {key: None for key in _COMPONENT_KEYS}
    result["valid"] = False
    result["reason"] = reason
    result["information_semantics"] = _INFORMATION_SEMANTICS
    return result


def peer_reference(
    peer_payload: Mapping[str, Any],
    now: float,
    stale_limit: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the current packet or one native v7 covariance-intersection blend.

    A cache is eligible only when its position estimate and covariance are
    present, its timestamp is fresh relative to the current packet, and the
    current packet has a distinct send timestamp.  The returned arrays are
    always copies, so callers cannot mutate the payload through this helper.
    """

    if not isinstance(peer_payload, Mapping):
        raise TypeError("peer_payload must be a mapping")
    current_mean = _array(peer_payload.get("state"), (STATE_DIM,))
    current_covariance = _array(peer_payload.get("covariance"), (STATE_DIM, STATE_DIM))
    if current_mean is None or current_covariance is None:
        raise ValueError("current peer state and covariance must have finite v7 shapes")
    current_mean = current_mean[:_POSITION_DIM].copy()
    current_covariance = current_covariance[:_POSITION_DIM, :_POSITION_DIM].copy()

    now_value = _as_finite_scalar(now, "now")
    limit_value = _as_finite_scalar(stale_limit, "stale_limit")
    state_time = _as_finite_scalar(peer_payload.get("state_time_s"), "state_time_s")
    cache_mean = _array(peer_payload.get("cache_mean"), (_POSITION_DIM,))
    cache_covariance = _array(
        peer_payload.get("cache_covariance"), (_POSITION_DIM, _POSITION_DIM)
    )
    cache_time = _as_finite_scalar(peer_payload.get("cache_time_s"), "cache_time_s")
    send_time = _as_finite_scalar(peer_payload.get("send_time_s"), "send_time_s")
    cache_send_time = _as_finite_scalar(
        peer_payload.get("cache_message_send_s"), "cache_message_send_s"
    )

    history_exists = (
        cache_mean is not None
        and cache_covariance is not None
        and cache_time is not None
        and cache_send_time is not None
    )
    cache_fresh = (
        history_exists
        and state_time is not None
        and now_value is not None
        and limit_value is not None
        and state_time - cache_time <= limit_value
    )
    new_packet = (
        cache_send_time is not None
        and send_time is not None
        and abs(send_time - cache_send_time) >= _SEND_TOL
    )
    if cache_fresh and new_packet:
        _symmetric_psd(cache_covariance, name="cache covariance")
        try:
            blended_mean, blended_covariance, weight = covariance_intersection(
                cache_mean,
                cache_covariance,
                current_mean,
                current_covariance,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise RuntimeError("native covariance intersection failed") from exc
        blended_mean = np.asarray(blended_mean, dtype=float).reshape(_POSITION_DIM)
        blended_covariance = np.asarray(blended_covariance, dtype=float).reshape(
            _POSITION_DIM, _POSITION_DIM
        )
        if not np.isfinite(blended_mean).all() or not np.isfinite(blended_covariance).all():
            raise RuntimeError("native covariance intersection returned non-finite values")
        _symmetric_psd(blended_covariance, name="native CI covariance")
        weight_value = float(weight)
        if not math.isfinite(weight_value):
            raise RuntimeError("native covariance intersection returned a non-finite weight")
        return blended_mean.copy(), blended_covariance.copy(), weight_value

    return current_mean.copy(), current_covariance.copy(), 1.0


def _peer_feature_payload(payload: Any) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
    if not isinstance(payload, Mapping):
        return None, None, None, None
    state = _array(payload.get("state"), (STATE_DIM,))
    covariance = _array(payload.get("covariance"), (STATE_DIM, STATE_DIM))
    position = None if state is None else state[:_POSITION_DIM].copy()
    trace = None
    if covariance is not None:
        block = covariance[:_POSITION_DIM, :_POSITION_DIM]
        value = float(np.trace(block))
        if math.isfinite(value):
            trace = value
    rssi = _as_finite_scalar(payload.get("rssi_dbm"), "rssi_dbm")
    return position, covariance, trace, rssi


def _angle_between(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < _TRACE_EPS or second_norm < _TRACE_EPS:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    if not math.isfinite(cosine):
        return None
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _feature_value(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _score_candidate(
    peer: str,
    payload: Any,
    *,
    own_state: np.ndarray,
    own_covariance: np.ndarray,
    now: float,
    stale_limit: float,
    legal: set[str],
    budget: float,
    sim: Any,
) -> dict[str, Any]:
    """Score one peer; exceptions from material filter failures are deliberate."""

    if peer not in legal:
        return _missing_candidate("not_legal")
    if not isinstance(payload, Mapping):
        return _missing_candidate("message_not_arrived")

    state = _array(payload.get("state"), (STATE_DIM,))
    covariance = _array(payload.get("covariance"), (STATE_DIM, STATE_DIM))
    if state is None:
        return _missing_candidate("message_not_arrived")
    if covariance is None:
        return _missing_candidate("invalid_covariance")
    try:
        _symmetric_psd(covariance, name=f"{peer} covariance")
    except (RuntimeError, np.linalg.LinAlgError):
        return _missing_candidate("invalid_covariance")

    rssi = _as_finite_scalar(payload.get("rssi_dbm"), "rssi_dbm")
    if rssi is None:
        return _missing_candidate("missing_rssi_dbm")
    send_time = _as_finite_scalar(payload.get("send_time_s"), "send_time_s")
    if send_time is None:
        return _missing_candidate("missing_send_time_s")
    state_time = _as_finite_scalar(payload.get("state_time_s"), "state_time_s")
    if state_time is None:
        return _missing_candidate("missing_state_time_s")
    if abs(state_time - now) > _TIME_TOL:
        return _missing_candidate("state_time_mismatch")
    age = now - send_time
    if age < -_TIME_TOL or age > stale_limit + _TIME_TOL:
        return _missing_candidate("stale_or_future_message")

    cached_send = _as_finite_scalar(payload.get("cache_message_send_s"), "cache_message_send_s")
    if cached_send is not None and abs(send_time - cached_send) < _SEND_TOL:
        return _missing_candidate("duplicate_packet")
    if budget <= 0.0 or not math.isfinite(budget):
        return _missing_candidate("budget_exhausted")

    own_position = own_state[:_POSITION_DIM]
    peer_position = state[:_POSITION_DIM]
    raw_difference = peer_position - own_position
    distance = float(np.linalg.norm(raw_difference))
    if not math.isfinite(distance) or distance < 1.0e-8:
        return _missing_candidate("degenerate_distance")

    own_block = own_covariance[:_POSITION_DIM, :_POSITION_DIM]
    peer_block = covariance[:_POSITION_DIM, :_POSITION_DIM]
    own_trace = float(np.trace(own_block))
    peer_trace = float(np.trace(peer_block))
    if (
        not math.isfinite(own_trace)
        or own_trace <= _TRACE_EPS
        or not math.isfinite(peer_trace)
        or peer_trace <= _TRACE_EPS
    ):
        return _missing_candidate("unscorable_position_trace")

    direction = raw_difference / distance
    geometry = float(np.clip(float(direction @ own_block @ direction) / own_trace, 0.0, 1.0))
    qscale = 3.0 * float(sim.sensor.gnss_nominal_std_m) ** 2
    if not math.isfinite(qscale) or qscale <= 0.0:
        raise RuntimeError("invalid GNSS nominal covariance scale")
    reliability = float(qscale / (qscale + peer_trace))
    numerator = rssi - float(sim.channel.min_rssi_dbm)
    denominator = float(sim.channel.tx_power_dbm) - float(sim.channel.min_rssi_dbm)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("invalid channel RSSI range")
    rssi_fraction = float(np.clip(numerator / denominator, 0.0, 1.0))
    communication_cost = float(
        0.5 * (1.0 - rssi_fraction + float(np.clip(age / stale_limit, 0.0, 1.0)))
    )

    try:
        reference_mean, reference_covariance, cache_ci_weight = peer_reference(
            payload, now, stale_limit
        )
        predicted_range = float(np.linalg.norm(own_position - reference_mean))
        if not math.isfinite(predicted_range) or predicted_range < 1.0e-8:
            return _missing_candidate("degenerate_predicted_distance")
        clone = MeasurementCIEKF(own_state.copy(), sim.ekf)
        clone.covariance = own_covariance.copy()
        clone_calls = 1
        accepted = bool(
            clone.update_relative_range(
                predicted_range,
                reference_mean,
                reference_covariance,
                measurement_variance=float(sim.sensor.uwb_los_std_m) ** 2,
                nlos_bias_m=0.0,
                timestamp=state_time,
            )
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"v7 clone failed for {peer}") from exc

    if clone_calls > 2:
        raise RuntimeError("v11 clone-call bound exceeded")
    predicted_state = np.asarray(clone.state, dtype=float)
    predicted_covariance = np.asarray(clone.covariance, dtype=float)
    if (
        predicted_state.shape != (STATE_DIM,)
        or predicted_covariance.shape != (STATE_DIM, STATE_DIM)
        or not np.isfinite(predicted_state).all()
        or not np.isfinite(predicted_covariance).all()
    ):
        raise RuntimeError("v7 clone returned non-finite values")
    _symmetric_psd(predicted_covariance, name=f"{peer} clone covariance")
    relative_ci_weight = _as_finite_scalar(
        clone.last_relative_range_update.get("weight"), "relative CI weight"
    )
    if relative_ci_weight is None:
        raise RuntimeError("v7 clone did not expose a finite relative CI weight")
    predicted_trace = float(np.trace(predicted_covariance[:_POSITION_DIM, :_POSITION_DIM]))
    trace_gain = own_trace - predicted_trace
    information_raw = trace_gain / own_trace
    if not math.isfinite(information_raw):
        raise RuntimeError("information fraction became non-finite")
    if information_raw < -1.0e-8 or information_raw > 1.0 + 1.0e-8:
        raise RuntimeError(f"information fraction outside integrity bounds: {information_raw}")
    information_fraction = float(np.clip(information_raw, 0.0, 1.0))
    score_raw = 0.25 * (
        information_fraction + geometry + reliability - communication_cost
    )
    if not math.isfinite(score_raw):
        raise RuntimeError("candidate score became non-finite")
    score = float(score_raw)
    aoi = float(age)
    return {
        "score": score,
        "information_fraction": information_fraction,
        "information_raw": float(information_raw),
        "trace_gain_m2": float(trace_gain),
        "geometry": geometry,
        "reliability": reliability,
        "communication_cost": communication_cost,
        "valid": True,
        "reason": "none" if accepted else "predicted_update_rejected",
        "direction": _json_array(direction),
        "distance_m": distance,
        "peer_trace_m2": peer_trace,
        "rssi_dbm": rssi,
        "aoi_s": aoi,
        "predicted_ci_weight": float(relative_ci_weight),
        "predicted_cache_ci_weight": float(cache_ci_weight),
        "predicted_accepted": accepted,
        "predicted_position_trace_m2": predicted_trace,
        "information_semantics": _INFORMATION_SEMANTICS,
    }


def score_context(context: Mapping[str, Any], sim: Any) -> dict[str, Any]:
    """Score the v11 snapshot and return a JSON-normalizable audit record."""

    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    started = time.perf_counter()
    own_state = _array(context.get("own_state"), (STATE_DIM,))
    own_covariance = _array(context.get("own_covariance"), (STATE_DIM, STATE_DIM))
    if own_state is None:
        raise RuntimeError("own_state must be a finite 15-vector")
    if own_covariance is None:
        raise RuntimeError("own_covariance must be a finite 15x15 matrix")
    _symmetric_psd(own_covariance, name="own covariance", positive_definite=True)
    now = _as_finite_scalar(context.get("time_s"), "time_s")
    if now is None:
        raise RuntimeError("time_s must be finite")
    stale_limit = _as_finite_scalar(getattr(sim.ekf, "stale_message_limit_s", None), "stale_limit")
    if stale_limit is None or stale_limit <= 0.0:
        raise RuntimeError("sim.ekf.stale_message_limit_s must be positive and finite")
    peers = context.get("peers", {})
    if not isinstance(peers, Mapping):
        raise TypeError("context peers must be a mapping")
    legal_sources_raw = context.get("legal_sources", ())
    if isinstance(legal_sources_raw, (str, bytes)):
        legal_sources = [str(legal_sources_raw)]
    elif isinstance(legal_sources_raw, Sequence):
        legal_sources = [str(value) for value in legal_sources_raw]
    else:
        legal_sources = []
    legal = set(legal_sources)
    budget_value = _as_finite_scalar(context.get("measurement_budget_remaining"), "budget")
    budget = 0.0 if budget_value is None else budget_value

    positions: dict[str, np.ndarray | None] = {}
    traces: dict[str, float | None] = {}
    rssis: dict[str, float | None] = {}
    aois: dict[str, float | None] = {}
    for peer in PEERS:
        position, _covariance, trace, rssi = _peer_feature_payload(peers.get(peer))
        positions[peer] = position
        traces[peer] = _feature_value(trace)
        rssis[peer] = _feature_value(rssi)
        payload = peers.get(peer)
        send_time = (
            _as_finite_scalar(payload.get("send_time_s"), "send_time_s")
            if isinstance(payload, Mapping)
            else None
        )
        age = None if send_time is None else float(now - send_time)
        aois[peer] = _feature_value(age)
    own_trace = float(np.trace(own_covariance[:_POSITION_DIM, :_POSITION_DIM]))
    own_trace_value = _feature_value(own_trace)
    own_position = own_state[:_POSITION_DIM]
    distances: dict[str, float | None] = {}
    for peer in PEERS:
        if positions[peer] is None:
            distances[peer] = None
        else:
            distance = float(np.linalg.norm(positions[peer] - own_position))
            distances[peer] = _feature_value(distance)
    angle = _angle_between(
        None if positions[PEERS[0]] is None else positions[PEERS[0]] - own_position,
        None if positions[PEERS[1]] is None else positions[PEERS[1]] - own_position,
    )
    features: dict[str, Any] = {
        "self_position_trace_m2": own_trace_value,
        "peer0_position_trace_m2": traces[PEERS[0]],
        "peer1_position_trace_m2": traces[PEERS[1]],
        "distance20_m": distances[PEERS[0]],
        "distance21_m": distances[PEERS[1]],
        "angle01_rad": _feature_value(angle),
        "rssi0_dbm": rssis[PEERS[0]],
        "rssi1_dbm": rssis[PEERS[1]],
        "aoi0_s": aois[PEERS[0]],
        "aoi1_s": aois[PEERS[1]],
    }

    candidates: dict[str, dict[str, Any]] = {}
    clone_calls = 0
    for peer in PEERS:
        candidate = _score_candidate(
            peer,
            peers.get(peer),
            own_state=own_state,
            own_covariance=own_covariance,
            now=now,
            stale_limit=stale_limit,
            legal=legal,
            budget=budget,
            sim=sim,
        )
        candidates[peer] = candidate
        if candidate.get("predicted_accepted") is not None or candidate.get("predicted_ci_weight") is not None:
            clone_calls += 1
    directions: dict[str, list[float] | None] = {}
    for peer in PEERS:
        position = positions[peer]
        if position is None:
            directions[peer] = None
            continue
        difference = position - own_position
        norm = float(np.linalg.norm(difference))
        directions[peer] = _json_array(difference / norm) if norm >= 1.0e-8 else None

    result: dict[str, Any] = {
        "step": int(context.get("step", 0)),
        "time_s": float(now),
        "seed": int(context.get("seed", 0)),
        "legal_sources": list(legal_sources),
        "features": features,
        "candidates": candidates,
        "extra_inputs": {
            "own_covariance": _json_array(own_covariance),
            "directions": directions,
        },
        "score_compute_ms": float((time.perf_counter() - started) * 1000.0),
        "clone_calls": int(clone_calls),
    }
    return result


def choose_greedy(scoring: Mapping[str, Any]) -> str:
    """Choose the best valid positive score with indexed tie randomness."""

    if not isinstance(scoring, Mapping):
        raise TypeError("scoring must be a mapping returned by score_context")
    candidates = scoring.get("candidates", {})
    if not isinstance(candidates, Mapping):
        return "none"
    eligible: list[tuple[str, float]] = []
    for peer in PEERS:
        candidate = candidates.get(peer)
        if not isinstance(candidate, Mapping) or candidate.get("valid") is not True:
            continue
        score = _as_finite_scalar(candidate.get("score"), "candidate score")
        if score is not None and score > 0.0:
            eligible.append((peer, score))
    if not eligible:
        return "none"
    best_score = max(score for _peer, score in eligible)
    tied = [peer for peer, score in eligible if abs(score - best_score) <= _SEND_TOL]
    if len(tied) == 1:
        return tied[0]
    seed = int(scoring.get("seed", 0))
    step = int(scoring.get("step", 0))
    rng = indexed_rng(seed, step, "v11_greedy_tie", TARGET)
    return str(rng.choice(tied))


__all__ = ["PEERS", "TARGET", "choose_greedy", "peer_reference", "score_context"]
