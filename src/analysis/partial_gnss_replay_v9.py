"""Truth-free replay helpers for the partial-GNSS v9 revalidation.

The producer in :mod:`src.experiments.partial_gnss_revalidation_v9` is the
only component which owns an environment.  This module consumes its frozen
sensor tape and reconstructs the three estimator arms offline.  In
particular, requests are decided from the common ``None`` tape before the
post-transition queue state is inspected; queue failures therefore remain
visible in the communication event table.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..experiments import partial_gnss_baseline_v7 as v7
from ..experiments import three_uav_fusion_validation_v6 as prior
from ..simulation.ekf import LocalEKF, POSITION_SLICE, STATE_DIM
from .cooperative_fusion_estimators_v6 import UIDS
from .local_ekf_startup_filter_v8 import StartupLocalEKF
from .partial_gnss_fusion_v7 import MeasurementCIEKF, PartialGNSSJointEKF


TARGET = v7.TARGET
PEERS = tuple(v7.PEERS)
DEFAULT_DT = 0.2
DEFAULT_TTL = 0.6
DEFAULT_PERIOD = 0.8
DEFAULT_BUDGET = 30
REQUEST_BYTES = 512
RECONSTRUCTION_TOL = 1.0e-8


class StartupMeasurementCI(StartupLocalEKF, MeasurementCIEKF):
    """Startup-local filter with the v7 measurement-CI range update.

    The order is intentional.  ``StartupLocalEKF.predict`` owns the first
    five stationary predictions and ``MeasurementCIEKF.update_relative_range``
    remains the range kernel after startup.  No method is copied or wrapped,
    so the v8/v7 implementations stay the single source of truth.
    """


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _copy_state_map(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {uid: np.asarray(values[uid], dtype=float).copy() for uid in UIDS}


def _copy_cov_map(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {uid: np.asarray(values[uid], dtype=float).copy() for uid in UIDS}


def _config(spec: Mapping[str, Any]):
    """Build the frozen v7 EKF configuration used by the producer."""

    # ``v7.simulation`` reads the locked baseline configuration and only
    # changes the cooperative flag.  Replay never enables the environment.
    return v7.simulation().ekf


def _sensor_config(spec: Mapping[str, Any]):
    return v7.simulation().sensor


def _limits(spec: Mapping[str, Any]) -> tuple[float, float, int]:
    dt = float(spec.get("dt_s", DEFAULT_DT))
    ttl = float(spec.get("message_ttl_s", DEFAULT_TTL))
    budget = int(spec.get("measurement_budget", DEFAULT_BUDGET))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not np.isfinite(ttl) or ttl < 0.0:
        raise ValueError("message_ttl_s must be finite and non-negative")
    if budget < 0:
        raise ValueError("measurement_budget must be non-negative")
    return dt, float(spec.get("pair_period_s", DEFAULT_PERIOD)), budget


def _initial_filters(record: Mapping[str, Any], method: str, spec: Mapping[str, Any]):
    sim = v7.simulation()
    startup_steps = int(spec.get("startup_steps", 5))
    motion_limit = float(spec.get("motion_vo_chi_square_limit", 16.27))
    filters: dict[str, StartupLocalEKF] = {}
    for uid in UIDS:
        klass = StartupMeasurementCI if uid == TARGET and method == "measurement_ci" else StartupLocalEKF
        initial = record["initial"][uid]
        estimator = klass(
            _array(initial["state"], (STATE_DIM,), f"initial state {uid}"),
            sim.ekf,
            sensor_config=sim.sensor,
            startup_steps=startup_steps,
            vo_motion_limit=motion_limit,
        )
        # The native producer's P is part of the frozen input.  In particular,
        # do not let LocalEKF's constructor silently replace it with a default
        # diagonal covariance.
        estimator.covariance = _array(initial["covariance"], (STATE_DIM, STATE_DIM), f"initial covariance {uid}")
        filters[uid] = estimator
    return filters


def _joint_from_filters(filters: Mapping[str, StartupLocalEKF], sim) -> PartialGNSSJointEKF:
    states = {uid: filters[uid].state.copy() for uid in UIDS}
    # The constructor supplies the correct shape and bookkeeping.  Replace
    # its covariance with the *actual* post-startup local blocks and retain
    # zero cross blocks as required by the v9 reference.
    joint = PartialGNSSJointEKF(states, sim.ekf, initial_cross_position=0.0)
    joint.state = np.concatenate([states[uid] for uid in UIDS]).astype(float, copy=True)
    joint.covariance = np.zeros((len(UIDS) * STATE_DIM, len(UIDS) * STATE_DIM), dtype=float)
    for i, uid in enumerate(UIDS):
        start = i * STATE_DIM
        joint.covariance[start : start + STATE_DIM, start : start + STATE_DIM] = filters[uid].covariance.copy()
    joint.covariance = 0.5 * (joint.covariance + joint.covariance.T)
    joint.begin_step()
    return joint


def _joint_maps(joint: PartialGNSSJointEKF) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {uid: joint.local_state(uid) for uid in UIDS},
        {uid: joint.local_covariance(uid) for uid in UIDS},
    )


def _watch_covariances(states: Mapping[str, np.ndarray], covariances: Mapping[str, np.ndarray], joint: PartialGNSSJointEKF | None) -> tuple[float, float]:
    minimum = float("inf")
    maximum_trace = 0.0
    for uid in UIDS:
        state = np.asarray(states[uid], dtype=float)
        covariance = np.asarray(covariances[uid], dtype=float)
        if state.shape != (STATE_DIM,) or covariance.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("invalid estimator state/covariance shape")
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise FloatingPointError(f"non-finite estimator block for {uid}")
        covariance = 0.5 * (covariance + covariance.T)
        eig = np.linalg.eigvalsh(covariance)
        if float(np.min(eig)) < -1.0e-8:
            raise FloatingPointError(f"non-PSD estimator covariance for {uid}")
        trace = float(np.trace(covariance))
        if trace >= 1.0e4:
            raise FloatingPointError(f"estimator covariance trace limit for {uid}")
        minimum = min(minimum, float(np.min(eig)))
        maximum_trace = max(maximum_trace, trace)
    if joint is not None:
        full = np.asarray(joint.covariance, dtype=float)
        if full.shape != (len(UIDS) * STATE_DIM, len(UIDS) * STATE_DIM) or not np.isfinite(full).all():
            raise FloatingPointError("non-finite joint covariance")
        full = 0.5 * (full + full.T)
        full_eig = np.linalg.eigvalsh(full)
        if float(np.min(full_eig)) < -1.0e-8:
            raise FloatingPointError("non-PSD joint covariance")
        full_trace = float(np.trace(full))
        if full_trace >= 1.0e4:
            raise FloatingPointError("joint covariance trace limit")
        minimum = min(minimum, float(np.min(full_eig)))
        maximum_trace = max(maximum_trace, full_trace)
    return minimum, maximum_trace


def _position_nees(error: np.ndarray, covariance: np.ndarray) -> float:
    return float(error @ np.linalg.pinv(covariance) @ error)


def _metrics(
    states: Mapping[str, np.ndarray],
    covariances: Mapping[str, np.ndarray],
    frame: Mapping[str, Any],
    joint: PartialGNSSJointEKF | None,
    *,
    local_diagnostics: Mapping[str, Any] | None = None,
    gnss_counts: list[int] | None = None,
    legal_count: int = 0,
    window: int | None = None,
    budget_used: int = 0,
    cumulative_accepted: int = 0,
) -> dict[str, Any]:
    truth = frame["evaluation_truth"]["position"]
    errors = {uid: np.asarray(states[uid], dtype=float)[:3] - np.asarray(truth[uid], dtype=float) for uid in UIDS}
    covariance = np.asarray(covariances[TARGET], dtype=float)
    minimum, maximum_trace = _watch_covariances(states, covariances, joint)
    pair_sse = []
    bound_nees = []
    actual_nees = []
    for left, right in itertools.combinations(UIDS, 2):
        error = errors[left] - errors[right]
        pair_sse.append(float(error @ error))
        bound = 2.0 * (np.asarray(covariances[left])[:3, :3] + np.asarray(covariances[right])[:3, :3])
        bound_nees.append(_position_nees(error, bound))
        if joint is not None:
            actual = joint.relative_covariance(left, right)
            actual_nees.append(_position_nees(error, actual))
    row = v7.old.normalize({
        "target_sse_m2": float(errors[TARGET] @ errors[TARGET]),
        "relative_sse_mean_m2": float(np.mean(pair_sse)),
        "target_position_nees": _position_nees(errors[TARGET], covariance[:3, :3]),
        "target_covariance_trace_m2": float(np.trace(covariance[:3, :3])),
        "target_position_eigenvalues_m2": np.linalg.eigvalsh(covariance[:3, :3]),
        "target_position_error_m": errors[TARGET],
        "states": states,
        "covariances": covariances,
        "gnss_accepted_0_1_2": list(gnss_counts or [0, 0, 0]),
        "local_diagnostics": dict(local_diagnostics or {}),
        "legal_count": int(legal_count),
        "window": window,
        "decision_window": window is not None,
        "budget_used": int(budget_used),
        "cumulative_accepted": int(cumulative_accepted),
        "minPfull_eig": float(minimum),
        "maxtrace": float(maximum_trace),
        "relative_NEESbound2": float(np.mean(bound_nees)),
        "relative_nees_mean": float(np.mean(actual_nees)) if actual_nees else float(np.mean(bound_nees)),
        "relative_nees_kind": "maintained_cross" if joint is not None else "unknown_cross_upper_bound",
    })
    if joint is not None:
        row["joint_position_covariance_9"] = v7.old.normalize(joint.position_covariance_9())
        row["joint_covariance_full_trace"] = float(np.trace(joint.covariance))
    else:
        row["joint_position_covariance_9"] = None
        row["joint_covariance_full_trace"] = None
    return row


def _window_number(step: int, spec: Mapping[str, Any]) -> int | None:
    decision_steps = tuple(int(s) for s in spec.get("decision_steps", range(6, 119, 4)))
    try:
        return decision_steps.index(int(step))
    except ValueError:
        return None


def _post_conditions(frame: Mapping[str, Any], source: str) -> dict[str, bool]:
    conditions = frame.get("observable", {}).get("post_conditions", {})
    raw = conditions.get(source, {}) if isinstance(conditions, Mapping) else {}
    bank = frame.get("observable", {}).get("range_bank", {})
    # Old/adversarial fixtures may omit the new post-transition audit.  A
    # range-bank record is still evidence of all successful post conditions.
    if not raw and source in bank:
        raw = {"uwb_valid": True, "effective_edge": True, "message_arrived": True, "potential_edge": True}
    return {
        "uwb_valid": bool(raw.get("uwb_valid", False)),
        "effective_edge": bool(raw.get("effective_edge", False)),
        "message_arrived": bool(raw.get("message_arrived", False)),
        "potential_edge": bool(raw.get("potential_edge", False)),
    }


def _packet_event_data(measurement: Mapping[str, Any], now: float) -> tuple[dict[str, Any], float]:
    envelope = measurement.get("message_envelope", {})
    if not isinstance(envelope, Mapping):
        raise ValueError("message_envelope must be a mapping")
    if "send_time_s" not in envelope:
        raise ValueError("message envelope lacks send_time_s")
    age = float(now) - float(envelope["send_time_s"])
    return dict(envelope), age


def _event_base(step: int, source: str, selected: bool, message_arrived: bool, age: float | None) -> dict[str, Any]:
    return {
        "step": int(step),
        "source": str(source),
        "selected": bool(selected),
        "request_cost_byte_equivalent512": int(REQUEST_BYTES if selected else 0),
        "message_arrived": bool(message_arrived),
        "uwb_valid": False,
        "effective_edge": False,
        "potential_edge": False,
        "post_conditions": {"uwb_valid": False, "effective_edge": False, "message_arrived": bool(message_arrived), "potential_edge": False},
        "attempted": False,
        "accepted": False,
        "rejection_reason": "not_selected" if not selected else "unavailable",
        "packet_age_s": age,
        "event_kernel": {},
        "event": {},
        "evaluation_sse_gain_m2": None,
    }


def replay(record: Mapping[str, Any], method: str, policy: str, spec: Mapping[str, Any]):
    """Replay one frozen producer record.

    The return value is ``(rows, events, audit)``.  ``events`` contains one
    row for every selected request, including requests that fail at a
    queue/measurement gate.  Unselected decision windows remain represented
    by their step rows and zero request cost.
    """

    if method not in tuple(v7.METHODS):
        raise ValueError(f"unknown method {method!r}")
    if policy not in tuple(v7.POLICIES):
        raise ValueError(f"unknown policy {policy!r}")
    if record.get("status", "complete") != "complete":
        raise ValueError("cannot replay a censored producer record")
    dt, pair_period, budget_limit = _limits(spec)
    if abs(dt - DEFAULT_DT) > 1.0e-12:
        raise ValueError("v9 packet reconstruction requires dt_s=0.2")
    sim = v7.simulation()
    filters = _initial_filters(record, method, spec)
    joint: PartialGNSSJointEKF | None = None
    history: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    pair_times: dict[str, float] = {}
    seen_packet: dict[str, float] = {}
    last_attempt_propagated: dict[str, float] = {}
    cache_age_checks: list[dict[str, Any]] = []
    cumulative_accepted = 0
    source_reconstruction_max = 0.0
    packet_reconstruction_max = 0.0
    none_driver_reconstruction_max = 0.0
    joint_none_reconstruction_max = 0.0
    source_reconstruction_P_max = 0.0
    packet_reconstruction_P_max = 0.0
    startup_evidence_checked = False

    for frame in record["frames"]:
        step = int(frame["step"])
        now = step * dt
        # Rows expose per-window accepted GNSS counts.  JointEKF resets its
        # event lists in begin_step(), so only this frame's events belong here.
        gnss_counts = [0, 0, 0]
        # This snapshot is deliberately before local sensor updates: packet
        # send times identify this source history index.
        history.append({uid: (filters[uid].state.copy(), filters[uid].covariance.copy()) for uid in UIDS})
        packets = frame["observable"]["sensor_packets"]
        prior.local_sensor_update(filters, packets, None, step)
        if step <= 5:
            if frame["observable"].get("pre_legal_sources"):
                raise AssertionError("startup requests are forbidden")
            if step == 5:
                for estimator in filters.values():
                    estimator.check_startup_evidence()
                startup_evidence_checked = True

        expected_states = _copy_state_map(frame["audit"]["states"])
        expected_covariances = _copy_cov_map(frame["audit"]["covariances"])
        compare_uids = UIDS if policy == "none" else PEERS
        for uid in compare_uids:
            state_diff = float(np.max(np.abs(filters[uid].state - expected_states[uid])))
            covariance_diff = float(np.max(np.abs(filters[uid].covariance - expected_covariances[uid])))
            source_reconstruction_max = max(source_reconstruction_max, state_diff)
            source_reconstruction_P_max = max(source_reconstruction_P_max, covariance_diff)
            if policy == "none":
                none_driver_reconstruction_max = max(none_driver_reconstruction_max, state_diff, covariance_diff)
        if policy == "none" and none_driver_reconstruction_max > RECONSTRUCTION_TOL:
            raise AssertionError(f"None driver reconstruction mismatch: {none_driver_reconstruction_max:g}")

        # The joint arm starts from the completed five-step local filters.  It
        # is created at the end of step five, before step six's prediction,
        # so step six is predicted exactly once.
        if method == "joint_reference" and step == 5:
            if not startup_evidence_checked:
                raise AssertionError("startup evidence was not checked before joint arm")
            joint = _joint_from_filters(filters, sim)
        use_joint = joint is not None and step > 5
        if use_joint:
            joint.predict_and_local_updates(packets, dt)
            joint_states, joint_covariances = _joint_maps(joint)
            # With no selected range, the zero-cross joint reference should
            # reconstruct the independent local arm.  Keep this as an audit,
            # rather than replacing the joint state with the local state.
            if policy == "none":
                for uid in UIDS:
                    joint_none_reconstruction_max = max(
                        joint_none_reconstruction_max,
                        float(np.max(np.abs(joint_states[uid] - filters[uid].state))),
                        float(np.max(np.abs(joint_covariances[uid] - filters[uid].covariance))),
                    )
            active_states, active_covariances = joint_states, joint_covariances
            current_local_diagnostics = {"joint": joint.diagnostics_snapshot(), **{uid: filters[uid].diagnostics_snapshot() for uid in UIDS}}
            for event in joint.gnss_update_events:
                if bool(event.get("accepted", False)):
                    gnss_counts[UIDS.index(str(event.get("uid")))] += 1
        else:
            active_states = {uid: filters[uid].state.copy() for uid in UIDS}
            active_covariances = {uid: filters[uid].covariance.copy() for uid in UIDS}
            current_local_diagnostics = {uid: filters[uid].diagnostics_snapshot() for uid in UIDS}
            for uid in UIDS:
                if current_local_diagnostics[uid].get("gnss_update_accepted", 0):
                    gnss_counts[UIDS.index(uid)] += int(current_local_diagnostics[uid]["gnss_update_accepted"])

        pre_legal = [str(source) for source in frame["observable"].get("pre_legal_sources", []) if str(source) in PEERS]
        decision_window = _window_number(step, spec)
        legal = [] if step <= 5 or decision_window is None else [source for source in pre_legal if source not in pair_times or now - pair_times[source] >= pair_period - 1.0e-9]
        if budget_limit <= 0 or sum(1 for event in events if event.get("selected")) >= budget_limit:
            legal = []
        source = v7.choose(policy, legal, int(record["case"]["seed"]), step)
        selected = source != "none"
        budget_used = sum(1 for event in events if event.get("selected")) + int(selected)
        event = _event_base(step, source, selected, False, None)
        event["legal_count"] = int(len(legal))
        event["window"] = decision_window
        event["request_cost_byte_equivalent"] = event["request_cost_byte_equivalent512"]
        if selected:
            conditions = _post_conditions(frame, source)
            event["message_arrived"] = conditions["message_arrived"]
            event["uwb_valid"] = conditions["uwb_valid"]
            event["effective_edge"] = conditions["effective_edge"]
            event["potential_edge"] = conditions["potential_edge"]
            event["post_conditions"] = conditions
            bank = frame["observable"].get("range_bank", {})
            measurement = bank.get(source) if isinstance(bank, Mapping) else None
            age = None
            envelope = None
            if measurement is not None:
                envelope, age = _packet_event_data(measurement, now)
                event["packet_age_s"] = age
            if not conditions["message_arrived"]:
                event["rejection_reason"] = "message_unavailable"
            elif not conditions["uwb_valid"]:
                event["rejection_reason"] = "uwb_invalid"
            elif not conditions["potential_edge"]:
                event["rejection_reason"] = "not_potential_edge"
            elif not conditions["effective_edge"]:
                event["rejection_reason"] = "ineffective_edge"
            elif measurement is None:
                event["rejection_reason"] = "range_measurement_unavailable"
            elif age is None or age > float(spec.get("message_ttl_s", DEFAULT_TTL)) + 1.0e-9:
                event["rejection_reason"] = "packet_expired"
            elif source in seen_packet and float(envelope["send_time_s"]) <= seen_packet[source] + 1.0e-12:
                event["rejection_reason"] = "duplicate_packet"
            else:
                packet_state = _array(measurement["packet_state"], (STATE_DIM,), "packet_state")
                packet_covariance = _array(measurement["packet_covariance"], (STATE_DIM, STATE_DIM), "packet_covariance")
                send_step = int(round(float(envelope["send_time_s"]) / dt))
                if send_step < 0 or send_step >= len(history):
                    raise ValueError("packet points to future or missing source history")
                reconstructed_state, reconstructed_covariance = prior.propagate_packet(
                    *history[send_step][source], envelope, float(envelope.get("last_propagated_time_s", now)), sim.ekf
                )
                state_diff = float(np.max(np.abs(reconstructed_state - packet_state)))
                cov_diff = float(np.max(np.abs(reconstructed_covariance - packet_covariance)))
                packet_reconstruction_max = max(packet_reconstruction_max, state_diff, cov_diff)
                packet_reconstruction_P_max = max(packet_reconstruction_P_max, cov_diff)
                if state_diff > RECONSTRUCTION_TOL or cov_diff > RECONSTRUCTION_TOL:
                    raise AssertionError(f"packet reconstruction mismatch: {state_diff:g}, {cov_diff:g}")
                if source in last_attempt_propagated:
                    cache_gap = now - last_attempt_propagated[source]
                    cache_age_checks.append({"source": source, "step": step, "gap_s": cache_gap, "ttl_s": float(spec.get("message_ttl_s", DEFAULT_TTL))})
                    if cache_gap <= float(spec.get("message_ttl_s", DEFAULT_TTL)) + 1.0e-9:
                        raise AssertionError("previous neighbor belief remained fresh across a pair-period")
                # The period timestamp is advanced before entering either
                # kernel so a rejected or w=1 update still consumes the pair
                # period.  The send-time identity is likewise consumed.
                pair_times[source] = now
                seen_packet[source] = float(envelope["send_time_s"])
                last_attempt_propagated[source] = float(envelope.get("last_propagated_time_s", now))
                before_target = active_states[TARGET].copy()
                truth = np.asarray(frame["evaluation_truth"]["position"][TARGET], dtype=float)
                if not use_joint:
                    neighbor_position_covariance = packet_covariance[POSITION_SLICE, POSITION_SLICE].copy()
                    target = filters[TARGET]
                    timestamp = float(envelope.get("last_propagated_time_s", now))
                    accepted = target.update_relative_range(
                        measurement["measurement"],
                        reconstructed_state[POSITION_SLICE],
                        neighbor_position_covariance,
                        measurement_variance=measurement["variance"],
                        nlos_bias_m=measurement.get("nlos_bias", 0.0),
                        timestamp=timestamp,
                    )
                    kernel = copy.deepcopy(target.last_relative_range_update)
                    active_states[TARGET] = target.state.copy()
                    active_covariances[TARGET] = target.covariance.copy()
                    current_local_diagnostics[TARGET] = target.diagnostics_snapshot()
                else:
                    accepted = joint.update_range(
                        TARGET,
                        source,
                        measurement["measurement"],
                        measurement["variance"],
                        nlos_bias=measurement.get("nlos_bias", 0.0),
                        measurement_id=f"{record['case']['scene_id']}_{record['case']['seed']}:{step}:{source}",
                    )
                    kernel = copy.deepcopy(joint.events[-1])
                    active_states, active_covariances = _joint_maps(joint)
                    current_local_diagnostics["joint"] = joint.diagnostics_snapshot()
                event.update({
                    "attempted": True,
                    "accepted": bool(accepted),
                    "rejection_reason": kernel.get("rejection_reason", "none" if accepted else "rejected"),
                    "event_kernel": kernel,
                    "event": kernel,
                    "evaluation_sse_gain_m2": float(np.sum((before_target[:3] - truth) ** 2) - np.sum((active_states[TARGET][:3] - truth) ** 2)),
                    "kernel_weight": kernel.get("weight"),
                })
                if accepted:
                    cumulative_accepted += 1
        # Communication events are decision-window records.  A selected but
        # unavailable packet is retained here as an unattempted request;
        # ordinary sensor windows with no request opportunity do not create a
        # fictitious zero-cost communication event.
        if decision_window is not None and selected:
            events.append(v7.old.normalize(event))
        window_value = decision_window
        row = _metrics(
            active_states,
            active_covariances,
            frame,
            joint if use_joint else None,
            local_diagnostics=current_local_diagnostics,
            gnss_counts=gnss_counts,
            legal_count=len(legal),
            window=window_value,
            budget_used=budget_used,
            cumulative_accepted=cumulative_accepted,
        )
        row.update({"step": step, "source": source, "selected": selected})
        rows.append(v7.old.normalize(row))

    if not startup_evidence_checked:
        raise AssertionError("replay did not reach startup step five")
    if max(source_reconstruction_max, source_reconstruction_P_max) > RECONSTRUCTION_TOL:
        raise AssertionError("reference source reconstruction mismatch")
    if method == "joint_reference" and policy == "none" and joint_none_reconstruction_max > RECONSTRUCTION_TOL:
        raise AssertionError(f"joint None reconstruction mismatch: {joint_none_reconstruction_max:g}")
    audit = {
        "status": "complete",
        "source_reconstruction_max": float(source_reconstruction_max),
        "source_reconstruction_P_max": float(source_reconstruction_P_max),
        "packet_reconstruction_max": float(packet_reconstruction_max),
        "packet_reconstruction_P_max": float(packet_reconstruction_P_max),
        "none_driver_reconstruction_max": float(none_driver_reconstruction_max),
        "joint_none_reconstruction_max": float(joint_none_reconstruction_max),
        "reconstruction_tolerance": RECONSTRUCTION_TOL,
        "startup_evidence_checked": startup_evidence_checked,
        "no_startup_requests_or_ranges": True,
        "cache_ci_replay": False,
        "cache_age_checks": cache_age_checks,
        "pair_period_s": pair_period,
        "message_ttl_s": float(spec.get("message_ttl_s", DEFAULT_TTL)),
        "measurement_budget": budget_limit,
        "method": method,
        "policy": policy,
    }
    return rows, events, v7.old.normalize(audit)


def physical_rows(record: Mapping[str, Any], spec: Mapping[str, Any]):
    """Extract rows/events from a live candidate record's actual frame audit."""

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    cumulative_budget = 0
    cumulative_accepted = 0
    for frame in record.get("frames", []):
        step = int(frame["step"])
        audit_frame = frame.get("audit", {})
        if not isinstance(audit_frame, Mapping) or "states" not in audit_frame or "covariances" not in audit_frame:
            # A censored producer may retain a terminal failure marker without
            # an estimator snapshot.  Preserve the record and skip that
            # incomplete transition in the physical table.
            continue
        states = _copy_state_map(audit_frame.get("states", {}))
        covariances = _copy_cov_map(audit_frame.get("covariances", {}))
        gnss_counts = [0, 0, 0]
        source = str(audit_frame.get("selection", "none"))
        selected = source != "none"
        if selected:
            cumulative_budget += 1
        kernel = copy.deepcopy(audit_frame.get("fusion_event") or {})
        gates = audit_frame.get("gate_events", []) or []
        attempted = any(bool(g.get("ekf_update_attempted", False)) for g in gates if isinstance(g, Mapping))
        accepted = bool(kernel.get("accepted", False)) if attempted else False
        if accepted:
            cumulative_accepted += 1
        conditions = _post_conditions(frame, source) if selected else {"message_arrived": False, "uwb_valid": False, "effective_edge": False, "potential_edge": False}
        measurement = frame.get("observable", {}).get("range_bank", {}).get(source) if selected else None
        age = None
        if measurement is not None:
            try:
                _, age = _packet_event_data(measurement, step * float(spec.get("dt_s", DEFAULT_DT)))
            except (TypeError, ValueError):
                age = None
        event = _event_base(step, source, selected, conditions["message_arrived"], age)
        event.update({
            "attempted": attempted,
            "accepted": accepted,
            "rejection_reason": kernel.get("rejection_reason", "none" if accepted else ("not_selected" if not selected else "rejected")),
            "event_kernel": kernel if attempted else None,
            "event": kernel if attempted else None,
            "evaluation_sse_gain_m2": None,
            "kernel_weight": kernel.get("weight") if attempted else None,
            "physical": True,
            "uwb_valid": conditions["uwb_valid"],
            "effective_edge": conditions["effective_edge"],
            "potential_edge": conditions["potential_edge"],
            "post_conditions": conditions,
        })
        if _window_number(step, spec) is not None and selected:
            events.append(v7.old.normalize(event))
        diagnostics = audit_frame.get("diagnostics", {})
        for i, uid in enumerate(UIDS):
            value = diagnostics.get(uid, {}) if isinstance(diagnostics, Mapping) else {}
            gnss_counts[i] += int(value.get("gnss_update_accepted", 0) or 0)
        row = _metrics(
            states,
            covariances,
            frame,
            None,
            local_diagnostics=diagnostics,
            gnss_counts=gnss_counts,
            legal_count=len(frame.get("observable", {}).get("pre_legal_sources", [])),
            window=_window_number(step, spec),
            budget_used=cumulative_budget,
            cumulative_accepted=cumulative_accepted,
        )
        row.update({"step": step, "source": source, "selected": selected, "phase": "closed_loop"})
        rows.append(v7.old.normalize(row))
    audit = {
        "status": str(record.get("status", "complete")),
        "phase": "closed_loop",
        "physical": True,
        "kernel_source": "live_frame_audit",
        "cache_ci_replay": False,
        "measurement_weight_source": "audit.fusion_event.weight",
        "overhead_source": "audit.overhead",
        "live_gate_events": [copy.deepcopy(frame.get("audit", {}).get("gate_events", [])) for frame in record.get("frames", [])],
        "live_overhead": [copy.deepcopy(frame.get("audit", {}).get("overhead", [])) for frame in record.get("frames", [])],
        "live_cache_ci_weights": [frame.get("audit", {}).get("cache_ci_weight") for frame in record.get("frames", [])],
        "failure": copy.deepcopy(record.get("failure")),
    }
    return rows, events, v7.old.normalize(audit)


__all__ = ["StartupMeasurementCI", "replay", "physical_rows"]
