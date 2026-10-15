"""Offline, frozen-runtime counterfactuals for the v11 neighbour selector.

This module is deliberately small at the boundary with the live experiment.
The live side supplies a post-local-update snapshot.  The functions here
clone that snapshot, apply at most one native v7 measurement-CI update per
candidate, and replay three already-recorded local sensor packets.  No
simulator state, truth state, random stream, or control action is consulted
while constructing a counterfactual.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..experiments import partial_gnss_baseline_v7 as baseline_v7
from ..experiments import three_uav_fusion_validation_v6 as prior
from ..simulation.ekf import STATE_DIM, covariance_intersection
from .partial_gnss_fusion_v7 import MeasurementCIEKF
from .partial_gnss_replay_v9 import StartupMeasurementCI


TARGET = baseline_v7.TARGET
PEERS = tuple(baseline_v7.PEERS)
NONE = "none"
DT_S = 0.2
PAIR_PERIOD_S = 0.8
SEND_TOLERANCE_S = 1.0e-12
IDENTITY_TOLERANCE = 1.0e-8
MEANINGFUL_UTILITY_M2 = 0.01

# These are the fields present in LocalEKF plus the fixed five-step startup
# adapter.  A TimedStartupOriginalCI may add the timing field after its first
# range call; it is harmless state and is therefore explicitly admitted.
_FILTER_FIELDS = frozenset(
    {
        "state",
        "covariance",
        "valid_observation_count",
        "valid_neighbor_count",
        "last_neighbor_timestamp",
        "last_nis",
        "rejected_observation_count",
        "last_prediction_context",
        "last_visual_update",
        "last_linear_update",
        "last_relative_range_update",
        "_step_diagnostics",
        "startup_steps",
        "vo_motion_limit",
        "_prediction_call_count",
        "_last_predict_was_startup",
        "startup_valid_vo_count",
        "startup_bias_update_count",
        "valid_startup_vo_count",
        "valid_bias_update_count",
        "last_startup_update",
        "last_ci_compute_ms",
    }
)
_FILTER_EXCLUDED = frozenset({"config", "sensor_config"})
_RESERVED_TAGS = frozenset({"__ndarray__", "__tuple__", "__float__"})


def _encode(value: Any, *, path: str = "value") -> Any:
    """Encode only the JSON-shaped state types used by the EKF runtime."""

    if isinstance(value, np.ndarray):
        # Recurse over tolist() so allow_nan=False JSON encoders never see a
        # native NaN/Inf.  The outer tag restores an ndarray on thaw.
        return {"__ndarray__": _encode(value.tolist(), path=path)}
    if isinstance(value, np.generic):
        return _encode(value.item(), path=path)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        label = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__float__": label}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, path=f"{path}[]") for item in value]}
    if isinstance(value, list):
        return [_encode(item, path=f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported non-string key at {path}")
            result[key] = _encode(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"unsupported filter state at {path}: {type(value).__name__}")


def _decode(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, list):
        return [_decode(item, path=f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        keys = set(value)
        if "__float__" in keys:
            if keys != {"__float__"} or value["__float__"] not in {"nan", "inf", "-inf"}:
                raise ValueError(f"invalid non-finite scalar tag at {path}")
            return float(value["__float__"] if value["__float__"] != "-inf" else "-inf")
        if "__ndarray__" in keys:
            if keys != {"__ndarray__"}:
                raise ValueError(f"invalid ndarray tag at {path}")
            raw = _decode(value["__ndarray__"], path=f"{path}.__ndarray__")
            try:
                return np.asarray(raw, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid ndarray payload at {path}") from exc
        if "__tuple__" in keys:
            if keys != {"__tuple__"} or not isinstance(value["__tuple__"], list):
                raise ValueError(f"invalid tuple tag at {path}")
            return tuple(_decode(item, path=f"{path}.__tuple__[]") for item in value["__tuple__"])
        if keys & _RESERVED_TAGS:
            raise ValueError(f"reserved filter tag used as ordinary mapping at {path}")
        return {str(key): _decode(item, path=f"{path}.{key}") for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported thaw payload at {path}: {type(value).__name__}")


def freeze_filter(ekf: Any) -> dict[str, Any]:
    """Freeze the complete mutable filter state while excluding configuration.

    Configuration objects contain dataclass/runtime references and are restored
    from the supplied simulation by :func:`thaw_filter`.  Every other known
    ``__dict__`` field is copied recursively; this includes temporal
    prediction context required for an exact next-predict and startup VO
    replay.
    """

    state = getattr(ekf, "__dict__", None)
    if not isinstance(state, dict):
        raise TypeError("ekf must expose a mutable __dict__")
    unknown = set(state) - _FILTER_FIELDS - _FILTER_EXCLUDED
    if unknown:
        raise ValueError(f"unsupported EKF runtime fields: {sorted(unknown)!r}")
    missing = {"state", "covariance"} - set(state)
    if missing:
        raise ValueError(f"EKF runtime fields missing: {sorted(missing)!r}")
    return {
        key: _encode(copy.deepcopy(value), path=f"filter.{key}")
        for key, value in state.items()
        if key not in _FILTER_EXCLUDED
    }


def thaw_filter(payload: Mapping[str, Any], sim: Any) -> StartupMeasurementCI:
    """Rebuild a startup measurement-CI filter with simulation-owned config."""

    if not isinstance(payload, Mapping):
        raise TypeError("frozen filter payload must be a mapping")
    unknown = set(payload) - _FILTER_FIELDS
    if unknown:
        raise ValueError(f"unsupported frozen EKF fields: {sorted(unknown)!r}")
    if "state" not in payload or "covariance" not in payload:
        raise ValueError("frozen filter requires state and covariance")
    decoded = {str(key): _decode(value, path=f"filter.{key}") for key, value in payload.items()}
    state = np.asarray(decoded["state"], dtype=float)
    covariance = np.asarray(decoded["covariance"], dtype=float)
    if state.shape != (STATE_DIM,) or covariance.shape != (STATE_DIM, STATE_DIM):
        raise ValueError("frozen filter state/covariance has invalid shape")
    # Non-finite values are deliberately representable in the serializer for
    # diagnostics.  The constructor itself remains responsible for rejecting
    # unusable estimator state when a prediction/update is attempted.
    ekf_config = getattr(sim, "ekf", None)
    sensor_config = getattr(sim, "sensor", None)
    if ekf_config is None:
        raise ValueError("simulation must expose ekf configuration")
    clone = StartupMeasurementCI(
        state.copy(),
        ekf_config,
        sensor_config=sensor_config,
        startup_steps=5,
        vo_motion_limit=16.27,
    )
    # Replace all runtime state only after construction.  Never restore the
    # excluded configuration fields, even if a hostile payload contains them.
    for key, value in decoded.items():
        if key in {"state", "covariance"}:
            continue
        setattr(clone, key, copy.deepcopy(value))
    clone.state = state.copy()
    clone.covariance = covariance.copy()
    clone.config = ekf_config
    clone.sensor_config = sensor_config if sensor_config is not None else clone.sensor_config
    return clone


def _finite_scalar(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    scalar = float(array.reshape(-1)[0])
    return scalar if math.isfinite(scalar) else None


def _array(value: Any, shape: tuple[int, ...], name: str, *, finite: bool = True) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != shape or (finite and not np.isfinite(array).all()):
        return None
    return array.copy()


def _normal(value: Any) -> Any:
    """Use the project normalizer while retaining explicit diagnostic NaNs."""

    return baseline_v7.old.normalize(copy.deepcopy(value))


def _base_event(estimator: StartupMeasurementCI, peer: str | None, request_cost: int) -> dict[str, Any]:
    return {
        "peer": peer,
        "request_cost": int(request_cost),
        "attempted": False,
        "accepted": False,
        "reason": "none" if peer is None else "not_evaluated",
        "kernel": {},
        "cache_weight": None,
        "ci_weight": None,
        "x_initial": np.asarray(estimator.state, dtype=float).copy(),
        "P_initial": np.asarray(estimator.covariance, dtype=float).copy(),
        "x_post": np.asarray(estimator.state, dtype=float).copy(),
        "P_post": np.asarray(estimator.covariance, dtype=float).copy(),
    }


def _finish_event(event: dict[str, Any], estimator: StartupMeasurementCI) -> dict[str, Any]:
    event["x_post"] = np.asarray(estimator.state, dtype=float).copy()
    event["P_post"] = np.asarray(estimator.covariance, dtype=float).copy()
    # Compact aliases are useful to CSV consumers while x/P remain the
    # canonical state names used by the v11 contract.
    event["state_initial"] = event["x_initial"].copy()
    event["covariance_initial"] = event["P_initial"].copy()
    event["state_post"] = event["x_post"].copy()
    event["covariance_post"] = event["P_post"].copy()
    # ``apply_branch`` is also used by pure numerical tests, so retain arrays
    # here as independent copies.  The public ``evaluate_point`` result is
    # normalized at its boundary before it is written to JSON/CSV.
    return event


def _gate_failure(event: dict[str, Any], estimator: StartupMeasurementCI, reason: str) -> tuple[StartupMeasurementCI, dict[str, Any]]:
    event["reason"] = str(reason)
    return estimator, _finish_event(event, estimator)


def apply_branch(snapshot: Mapping[str, Any], peer: str | None, sim: Any) -> tuple[StartupMeasurementCI, dict[str, Any]]:
    """Apply one candidate's frozen range update to an isolated filter clone."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    if peer not in (None, NONE, *PEERS):
        raise ValueError(f"unknown counterfactual peer {peer!r}")
    estimator = thaw_filter(snapshot.get("own_filter"), sim)
    selected_peer = None if peer in (None, NONE) else str(peer)
    event = _base_event(estimator, selected_peer, 0 if selected_peer is None else 1)
    if selected_peer is None:
        return estimator, _finish_event(event, estimator)

    # Gate order mirrors the native environment exactly.  A peer request is
    # charged once even when its frozen UWB or packet gate fails.
    budget = _finite_scalar(snapshot.get("budget_remaining_before"))
    if budget is None or budget <= 0.0:
        return _gate_failure(event, estimator, "budget_exhausted")
    ranges = snapshot.get("ranges", {})
    if not isinstance(ranges, Mapping):
        return _gate_failure(event, estimator, "selected_pair_not_observed")
    range_payload = ranges.get(selected_peer)
    if not isinstance(range_payload, Mapping) or not bool(range_payload.get("present", False)):
        return _gate_failure(event, estimator, "selected_pair_not_observed")
    if not bool(range_payload.get("valid", False)):
        return _gate_failure(event, estimator, "uwb_invalid")
    if not bool(range_payload.get("effective_edge", False)):
        return _gate_failure(event, estimator, "communication_edge_ineffective")

    peers = snapshot.get("peers", {})
    payload = peers.get(selected_peer) if isinstance(peers, Mapping) else None
    if not isinstance(payload, Mapping):
        return _gate_failure(event, estimator, "message_not_arrived")
    state = _array(payload.get("state"), (STATE_DIM,), "message state")
    covariance = _array(payload.get("covariance"), (STATE_DIM, STATE_DIM), "message covariance")
    send_time = _finite_scalar(payload.get("send_time_s"))
    state_time = _finite_scalar(payload.get("state_time_s"))
    now = _finite_scalar(snapshot.get("time_s"))
    if state is None or covariance is None or send_time is None or state_time is None:
        return _gate_failure(event, estimator, "message_invalid")
    stale_limit = _finite_scalar(getattr(getattr(sim, "ekf", None), "stale_message_limit_s", None))
    # The native arrival envelope's age is measured from send time.  The
    # propagated estimator state itself must represent this decision instant;
    # these are separate checks because a delayed packet can carry a current
    # propagated state while an old send timestamp still violates TTL.
    if now is None or stale_limit is None or stale_limit < 0.0:
        return _gate_failure(event, estimator, "message_stale")
    if send_time > now + 1.0e-9 or now - send_time > stale_limit + 1.0e-9:
        return _gate_failure(event, estimator, "message_stale")
    if abs(now - state_time) > 1.0e-8:
        return _gate_failure(event, estimator, "message_time_mismatch")

    last_update = _finite_scalar(payload.get("last_update_time_s"))
    if last_update is not None and now - last_update < PAIR_PERIOD_S - 1.0e-9:
        return _gate_failure(event, estimator, "update_period_not_elapsed")
    cached_send = _finite_scalar(payload.get("cache_message_send_s"))
    if cached_send is not None and abs(send_time - cached_send) < SEND_TOLERANCE_S:
        return _gate_failure(event, estimator, "reused_neighbor_packet")

    measured = _finite_scalar(range_payload.get("measurement"))
    variance = _finite_scalar(range_payload.get("variance"))
    nlos_bias = _finite_scalar(range_payload.get("nlos_bias"))
    if measured is None or variance is None or variance <= 0.0 or nlos_bias is None or measured < 0.0:
        return _gate_failure(event, estimator, "measurement_invalid")

    # Reproduce native cache CI: only a present, fresh previous belief paired
    # with a genuinely new packet is fused.  The incoming message is otherwise
    # used directly.  CI itself remains the simulation helper's 51-point grid.
    current_mean = state[:3].copy()
    current_covariance = covariance[:3, :3].copy()
    cache_mean = _array(payload.get("cache_mean"), (3,), "cache mean")
    cache_covariance = _array(payload.get("cache_covariance"), (3, 3), "cache covariance")
    cache_time = _finite_scalar(payload.get("cache_time_s"))
    cache_eligible = (
        cache_mean is not None
        and cache_covariance is not None
        and cache_time is not None
        and state_time - cache_time <= stale_limit
        and cached_send is not None
        and abs(send_time - cached_send) >= SEND_TOLERANCE_S
    )
    cache_weight = 1.0
    reference_mean = current_mean
    reference_covariance = current_covariance
    if cache_eligible:
        try:
            reference_mean, reference_covariance, cache_weight = covariance_intersection(
                cache_mean,
                cache_covariance,
                current_mean,
                current_covariance,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise RuntimeError("native covariance intersection failed") from exc
        reference_mean = np.asarray(reference_mean, dtype=float).reshape(3).copy()
        reference_covariance = np.asarray(reference_covariance, dtype=float).reshape(3, 3).copy()
        cache_weight = float(cache_weight)
    if not np.isfinite(reference_mean).all() or not np.isfinite(reference_covariance).all():
        raise RuntimeError("native covariance intersection returned non-finite values")
    event["cache_weight"] = float(cache_weight)
    event["reference_position"] = reference_mean.copy()
    event["reference_covariance"] = reference_covariance.copy()
    accepted = estimator.update_relative_range(
        measured,
        reference_mean,
        reference_covariance,
        measurement_variance=variance,
        nlos_bias_m=nlos_bias,
        timestamp=state_time,
    )
    event["attempted"] = True
    event["accepted"] = bool(accepted)
    event["kernel"] = copy.deepcopy(estimator.last_relative_range_update)
    event["ci_weight"] = event["kernel"].get("ci_weight", event["kernel"].get("weight"))
    event["reason"] = str(event["kernel"].get("rejection_reason", "none" if accepted else "innovation_rejected"))
    return estimator, _finish_event(event, estimator)


def _frame_for_step(record: Mapping[str, Any], step: int) -> Mapping[str, Any]:
    frames = record.get("frames")
    if not isinstance(frames, Sequence) or step < 1 or step > len(frames):
        raise ValueError(f"record lacks frame for step {step}")
    frame = frames[step - 1]
    if not isinstance(frame, Mapping):
        raise ValueError(f"record frame {step} is not a mapping")
    return frame


def _truth_position(frame: Mapping[str, Any]) -> np.ndarray:
    try:
        return np.asarray(frame["evaluation_truth"]["position"][TARGET], dtype=float).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluation frame lacks target truth position") from exc


def _branch_summary(
    estimator: StartupMeasurementCI,
    event: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    step: int,
) -> dict[str, Any]:
    states = [np.asarray(estimator.state, dtype=float).copy()]
    covariances = [np.asarray(estimator.covariance, dtype=float).copy()]
    for offset, frame in enumerate(frames, start=1):
        packets = frame.get("observable", {}).get("sensor_packets", {})
        packet = packets.get(TARGET) if isinstance(packets, Mapping) else None
        if not isinstance(packet, Mapping):
            raise ValueError("future frame lacks sanitized target sensor packet")
        # Each branch receives a copied packet and mapping.  The native helper
        # owns the prediction/VO/GNSS ordering and mutates only this estimator.
        prior.local_sensor_update(
            {TARGET: estimator},
            {TARGET: copy.deepcopy(packet)},
            None,
            step + offset,
        )
        states.append(np.asarray(estimator.state, dtype=float).copy())
        covariances.append(np.asarray(estimator.covariance, dtype=float).copy())
    return {"states": states, "covariances": covariances}


def _native_gate_hint(frame: Mapping[str, Any], peer: str) -> Mapping[str, Any] | None:
    audit = frame.get("audit", {})
    if not isinstance(audit, Mapping):
        return None
    for key in ("gate_events", "native_gate_events"):
        gates = audit.get(key)
        if isinstance(gates, Mapping):
            for value in gates.values():
                if isinstance(value, Mapping) and str(value.get("source_uav_id")) == peer and str(value.get("target_uav_id")) == TARGET:
                    return value
        elif isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
            for value in gates:
                if isinstance(value, Mapping) and str(value.get("source_uav_id")) == peer and str(value.get("target_uav_id")) == TARGET:
                    return value
    return None


def evaluate_point(record: Mapping[str, Any], snapshot: Mapping[str, Any], score_row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate none and both peers at one frozen decision point."""

    if not isinstance(record, Mapping) or not isinstance(snapshot, Mapping) or not isinstance(score_row, Mapping):
        raise TypeError("record, snapshot, and score_row must be mappings")
    step = int(snapshot.get("step", 0))
    current_frame = _frame_for_step(record, step)
    frames = record.get("frames")
    assert isinstance(frames, Sequence)
    # Python slice [step:step+3] is intentional: step n's current frame is
    # frames[n-1], while exactly the next three recorded packets are n+1..n+3.
    future_frames = list(frames[step : step + 3])
    if len(future_frames) != 3:
        raise ValueError("counterfactual point lacks three future frames")
    if any(not isinstance(frame, Mapping) for frame in future_frames):
        raise ValueError("future counterfactual frame is not a mapping")

    branches: dict[str, dict[str, Any]] = {}
    branch_estimators: dict[str, StartupMeasurementCI] = {}
    for candidate in (NONE, *PEERS):
        simulation = spec.get("simulation") if isinstance(spec, Mapping) else None
        if simulation is None and isinstance(spec, Mapping):
            simulation = spec.get("sim")
        if simulation is None or not hasattr(simulation, "ekf"):
            simulation = baseline_v7.simulation()
        estimator, event = apply_branch(snapshot, candidate, simulation)
        branch_estimators[candidate] = estimator
        summary = _branch_summary(estimator, event, future_frames, step)
        summary.update(
            {
                "accepted": bool(event["accepted"]),
                "attempted": bool(event["attempted"]),
                "reason": event["reason"],
                "ci_weight": event["kernel"].get("ci_weight") if isinstance(event.get("kernel"), Mapping) else None,
                "cache_weight": event.get("cache_weight"),
                "request_cost": int(event["request_cost"]),
                "event": event,
                "state": np.asarray(event["x_post"], dtype=float).copy(),
                "covariance": np.asarray(event["P_post"], dtype=float).copy(),
            }
        )
        branches[candidate] = summary

    truths = [_truth_position(current_frame)] + [_truth_position(frame) for frame in future_frames]
    for candidate, branch in branches.items():
        errors = [np.asarray(state, dtype=float)[:3] - truth for state, truth in zip(branch["states"], truths)]
        sse = [float(error @ error) for error in errors]
        branch["sse_m2"] = sse
        branch["utility"] = float(-np.mean(sse))
        branch["instantaneous_sse_improvement_m2"] = float(branches[NONE]["sse_m2"][0] - sse[0]) if candidate != NONE else 0.0
        branch["four_state_utility_difference_m2"] = float(branch["utility"] - branches[NONE]["utility"]) if candidate != NONE else 0.0
        branch["local_states"] = int(len(branch["states"]))
        branch["relative_ci_calls"] = int(bool(branch["attempted"]))

    candidates = score_row.get("candidates", {})
    c0 = candidates.get(PEERS[0], {}) if isinstance(candidates, Mapping) else {}
    c1 = candidates.get(PEERS[1], {}) if isinstance(candidates, Mapping) else {}
    score0 = _finite_scalar(c0.get("score")) if isinstance(c0, Mapping) else None
    score1 = _finite_scalar(c1.get("score")) if isinstance(c1, Mapping) else None
    legal = current_frame.get("observable", {}).get("pre_legal_sources", [])
    if not isinstance(legal, Sequence) or isinstance(legal, (str, bytes)):
        legal = []
    dual_eligible = bool(
        all(peer in [str(item) for item in legal] for peer in PEERS)
        and isinstance(c0, Mapping) and isinstance(c1, Mapping)
        and c0.get("valid") is True and c1.get("valid") is True
        and score0 is not None and score1 is not None
        and len(future_frames) == 3
        and isinstance(snapshot.get("own_filter"), Mapping)
    )
    score_difference = None if score0 is None or score1 is None else float(score0 - score1)
    utility_difference = float(branches[PEERS[0]]["utility"] - branches[PEERS[1]]["utility"])
    meaningful = bool(abs(utility_difference) >= MEANINGFUL_UTILITY_M2)
    if score_difference is None or abs(score_difference) <= SEND_TOLERANCE_S:
        scored_peer = "tie" if dual_eligible else None
    elif dual_eligible:
        scored_peer = PEERS[0] if score_difference > 0 else PEERS[1]
    else:
        scored_peer = None
    if not dual_eligible:
        preferred_peer = None
    elif utility_difference > SEND_TOLERANCE_S:
        preferred_peer = PEERS[0]
    elif utility_difference < -SEND_TOLERANCE_S:
        preferred_peer = PEERS[1]
    else:
        preferred_peer = "tie"
    if not dual_eligible or not meaningful or score_difference is None:
        ranking_correct = None
    elif abs(score_difference) <= SEND_TOLERANCE_S or abs(utility_difference) <= SEND_TOLERANCE_S:
        ranking_correct = 0.5
    else:
        ranking_correct = 1.0 if score_difference * utility_difference > 0.0 else 0.0

    actual_selection = str(current_frame.get("audit", {}).get("selection", NONE)) if isinstance(current_frame.get("audit", {}), Mapping) else NONE
    identity_matches: bool | None = None
    identity_error_max: float | None = None
    identity_peer = actual_selection if actual_selection in PEERS else NONE if actual_selection == NONE else None
    if identity_peer is not None:
        gate_hint = _native_gate_hint(current_frame, identity_peer) if identity_peer in PEERS else None
        audit = current_frame.get("audit", {})
        expected_state = _array(audit.get("states", {}).get(TARGET), (STATE_DIM,), "identity state") if isinstance(audit, Mapping) and isinstance(audit.get("states"), Mapping) else None
        expected_cov = _array(audit.get("covariances", {}).get(TARGET), (STATE_DIM, STATE_DIM), "identity covariance") if isinstance(audit, Mapping) and isinstance(audit.get("covariances"), Mapping) else None
        if expected_state is not None and expected_cov is not None:
            actual_state = branches[identity_peer]["states"][0]
            actual_cov = branches[identity_peer]["covariances"][0]
            identity_error_max = float(max(np.max(np.abs(actual_state - expected_state)), np.max(np.abs(actual_cov - expected_cov))))
            identity_matches = bool(identity_error_max <= IDENTITY_TOLERANCE)
            if gate_hint is not None and bool(gate_hint.get("ekf_update_attempted", False)) != bool(branches[identity_peer]["attempted"]):
                identity_matches = False
            if gate_hint is not None and bool(gate_hint.get("ekf_update_accepted", False)) != bool(branches[identity_peer]["accepted"]):
                identity_matches = False
            if not identity_matches:
                raise RuntimeError("v11 conditional identity mismatch")

    result = {
        "step": step,
        "dual_eligible": dual_eligible,
        "score_difference": score_difference,
        "score_delta": score_difference,
        "scoreB": score0,
        "scoreC": score1,
        "utility_difference": utility_difference,
        "utility_delta": utility_difference,
        "meaningful": meaningful,
        "ranking_correct": ranking_correct,
        "preferred_peer": preferred_peer,
        "scored_peer": scored_peer,
        "identity_matches": identity_matches,
        "identity_error_max": identity_error_max,
        "actual_selection": actual_selection,
        "branches": branches,
        "local_states": int(sum(len(branch["states"]) for branch in branches.values())),
        "relative_ci_calls": int(sum(branch["relative_ci_calls"] for branch in branches.values())),
        "optimizer_steps": 0,
        "physical_steps": 0,
        "context_steps": 1,
        "request_cost": int(sum(branch["request_cost"] for branch in branches.values())),
    }
    return _normal(result)


__all__ = ["apply_branch", "evaluate_point", "freeze_filter", "thaw_filter"]
