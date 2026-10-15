"""Truth-free v10 execution adapters for the partial-GNSS CI comparison.

The v10 execution layer deliberately reuses the frozen v9 replay and episode
code.  It selects one of the two scoped CI classes supplied by
``partial_gnss_ci_search_v10`` and records the native queue state around each
physical transition.  No simulator, estimator, or navigation implementation
is copied here.
"""

from __future__ import annotations

import copy
import threading
from time import perf_counter
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from ..analysis import partial_gnss_replay_v9 as replay_v9
from ..analysis.partial_gnss_ci_search_v10 import METHOD_CLASSES
from ..control.fixed_navigation import fixed_navigation_command
from . import partial_gnss_baseline_v7 as v7
from . import partial_gnss_revalidation_v9 as v9


METHODS = ("original_ci", "refined_ci")
POLICIES = tuple(v7.POLICIES)
UIDS = tuple(v7.UIDS)
TARGET = v7.TARGET
PEERS = tuple(v7.PEERS)
DT_S = 0.2
MESSAGE_TTL_S = 0.6
PAIR_PERIOD_S = 0.8
MEASUREMENT_BUDGET = 30
DECISION_STEPS = tuple(range(6, 119, 4))
RECONSTRUCTION_TOLERANCE = 1.0e-8
NAVIGATION_TOLERANCE = 1.0e-10

# The v10 contract is a main-thread diagnostic.  The lock protects the
# temporary class patches and also makes nested callers deterministic.
_API_LOCK = threading.RLock()
_CAPTURED_ENVS: list["AuditEnv"] = []


def _main_thread_guard() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("v10 execution must run on the main thread")


def _method_classes() -> Mapping[str, type]:
    if set(METHOD_CLASSES) != set(METHODS):
        raise ValueError("v10 METHOD_CLASSES must contain original_ci and refined_ci")
    return METHOD_CLASSES


def _validate_method_policy(method: str, policy: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown v10 method {method!r}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")


def _execution_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping")
    result = dict(spec)
    result.setdefault("dt_s", DT_S)
    result.setdefault("message_ttl_s", MESSAGE_TTL_S)
    result.setdefault("pair_period_s", PAIR_PERIOD_S)
    result.setdefault("measurement_budget", MEASUREMENT_BUDGET)
    result.setdefault("decision_steps", list(DECISION_STEPS))
    result.setdefault("startup_steps", 5)
    result.setdefault("motion_vo_chi_square_limit", 16.27)
    return result


@contextmanager
def _patched_method(method: str) -> Iterator[type]:
    """Temporarily route the v9 startup filter to one v10 class."""

    klass = _method_classes()[method]
    # v9.episode imports this symbol inside the function, while replay uses
    # it in _initial_filters.  Patching the module attribute covers both and
    # is automatically rolled back if the kernel raises.
    with patch.object(replay_v9, "StartupMeasurementCI", klass):
        yield klass


def _finite_timing(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for event in events:
        kernel = event.get("event", {}) if isinstance(event, Mapping) else {}
        value = kernel.get("compute_time_ms") if isinstance(kernel, Mapping) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0.0:
            values.append(value)
    return {
        "timed_event_count": len(values),
        "compute_time_ms_total": float(np.sum(values)) if values else 0.0,
        "compute_time_ms_mean": float(np.mean(values)) if values else None,
        "compute_time_ms_min": float(np.min(values)) if values else None,
        "compute_time_ms_max": float(np.max(values)) if values else None,
        "compute_time_ms": values,
        "timing_source": "event.compute_time_ms",
    }


def _conditional_bound_audit(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    residuals: list[float] = []
    matches = 0
    observed = 0
    for event in events:
        kernel = event.get("event", {}) if isinstance(event, Mapping) else {}
        if not isinstance(kernel, Mapping) or not event.get("attempted"):
            continue
        if kernel.get("conditional_linear_error_bound") is not None:
            observed += 1
        if bool(kernel.get("conditional_bound_matches_posterior", False)):
            matches += 1
        try:
            residual = float(kernel.get("conditional_bound_residual_norm"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(residual):
            residuals.append(residual)
    return {
        "attempted_count": sum(bool(e.get("attempted")) for e in events),
        "conditional_bound_count": observed,
        "conditional_bound_match_count": matches,
        "conditional_bound_residual_max": float(max(residuals)) if residuals else None,
        "conditional_bound_residual_mean": float(np.mean(residuals)) if residuals else None,
        "scope": "fixed_H_linear_unbiased_prior_pseudo_observation",
        "nonlinear_range_guarantee": False,
    }


def _augment_replay_audit(
    audit: Mapping[str, Any],
    events: list[dict[str, Any]],
    method: str,
    policy: str,
    klass: type,
) -> dict[str, Any]:
    result = dict(audit)
    for event in events:
        event["method"] = method
        event["policy"] = policy
    result.update(
        {
            "method": method,
            "policy": policy,
            "method_class": f"{klass.__module__}.{klass.__qualname__}",
            "method_mro": [base.__name__ for base in klass.__mro__],
            "timing": _finite_timing(events),
            "conditional_bounds": _conditional_bound_audit(events),
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "decision_steps": list(DECISION_STEPS),
            "dt_s": DT_S,
            "message_ttl_s": MESSAGE_TTL_S,
            "pair_period_s": PAIR_PERIOD_S,
            "measurement_budget": MEASUREMENT_BUDGET,
            "first_five_requests": 0,
            "full_horizon_steps": 120,
            "source_reconstruction_ok": float(result.get("source_reconstruction_max", 0.0))
            <= RECONSTRUCTION_TOLERANCE,
            "packet_reconstruction_ok": float(result.get("packet_reconstruction_max", 0.0))
            <= RECONSTRUCTION_TOLERANCE,
        }
    )
    # Preserve the v9 audit values and use its normalizer for arrays and
    # numpy scalars, while retaining the exact event/row cardinalities.
    return v7.old.normalize(result)


def replay(
    record: Mapping[str, Any],
    method: str,
    policy: str,
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay a frozen v9 tape with one scoped v10 CI class.

    The v9 kernel owns source/packet reconstruction, startup handling, packet
    age, budget, and per-step row/event cardinality.  This wrapper only adds
    v10 method/timing metadata.
    """

    _main_thread_guard()
    _validate_method_policy(method, policy)
    execution_spec = _execution_spec(spec)
    with _API_LOCK:
        _main_thread_guard()
        with _patched_method(method) as klass:
            original_update = replay_v9.prior.local_sensor_update
            def bounded_update(*args, **kwargs):
                if perf_counter() >= execution_spec.get("phase_deadline", float("inf")):
                    raise RuntimeError("v10 frozen phase wall limit")
                return original_update(*args, **kwargs)
            with patch.object(replay_v9.prior, "local_sensor_update", bounded_update):
                rows, events, audit = replay_v9.replay(record, "measurement_ci", policy, execution_spec)
            return rows, events, _augment_replay_audit(audit, events, method, policy, klass)


def _copy_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


def _normal(value: Any) -> Any:
    return v7.old.normalize(copy.deepcopy(value))


def _queue_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return _normal(dict(value))


def _message_envelope(message: Any) -> dict[str, Any] | None:
    if message is None:
        return None
    fields = (
        "source",
        "target",
        "send_time_s",
        "arrival_step",
        "arrival_time_s",
        "delay_s",
        "last_propagated_time_s",
    )
    return _normal({name: getattr(message, name, None) for name in fields})


def _message_buffer(messages: Any) -> dict[str, Any]:
    if isinstance(messages, Mapping):
        return {
            str(key): _message_envelope(value)
            for key, value in messages.items()
        }
    if isinstance(messages, (list, tuple)):
        return {str(index): _message_envelope(value) for index, value in enumerate(messages)}
    return {}


def _payload_snapshot(env: Any) -> dict[str, Any]:
    """Copy only estimated 15-state payloads recorded by DiagnosticEnv."""

    history = getattr(env, "transmitted_payload_history", [])
    if not history:
        return {}
    latest = history[-1]
    result: dict[str, Any] = {}
    if isinstance(latest, Mapping):
        for uid, payload in latest.items():
            if not isinstance(payload, Mapping):
                continue
            state = _copy_array(payload.get("state"), (15,))
            covariance = _copy_array(payload.get("covariance"), (15, 15))
            result[str(uid)] = {"state": state, "covariance": covariance}
    return _normal(result)


def _navigation_check(
    step: int,
    estimates: Mapping[str, Any],
    goals: Mapping[str, Any],
    actions: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    max_error = 0.0
    first_five_zero = True
    for uid in UIDS:
        estimate = _copy_array(estimates[uid], (15,))
        goal = _copy_array(goals[uid], (3,))
        action = actions.get(uid, {}) if isinstance(actions, Mapping) else {}
        actual = _copy_array(action.get("delta_v", np.zeros(3)), (3,))
        expected = np.zeros(3) if step <= 5 else np.asarray(
            fixed_navigation_command(estimate[:6], goal)["command_delta_v_mps"], dtype=float
        )
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        if step <= 5 and np.max(np.abs(actual)) > NAVIGATION_TOLERANCE:
            first_five_zero = False
        checks[uid] = {
            "pre_estimate": estimate,
            "goal": goal,
            "actual_delta_v": actual,
            "expected_delta_v": expected,
            "max_abs_error": error,
            "matches": bool(error <= NAVIGATION_TOLERANCE),
            "used_truth": False,
        }
    return {
        "step": int(step),
        "per_uav": checks,
        "max_abs_error": float(max_error),
        "first_five_zero": bool(first_five_zero),
        "used_truth": False,
        "source": "fixed_navigation_command(pre_estimate[:6], goal)",
    }


def _selected_message(env: Any, actions: Mapping[str, Any]) -> dict[str, Any]:
    target_action = actions.get(TARGET, {}) if isinstance(actions, Mapping) else {}
    source = str(target_action.get("measurement_uav_id", "none"))
    message = None if source == "none" else getattr(env, "arrived_messages", {}).get((source, TARGET))
    payload = {
        "source": source,
        "target": TARGET,
        "available": bool(message is not None),
        "envelope": _message_envelope(message),
        "timestamp_s": None,
        "state": None,
        "covariance": None,
        "P_estimate": None,
    }
    if message is not None:
        timestamp = message.last_propagated_time_s
        if timestamp is None:
            timestamp = message.arrival_time_s
        payload["timestamp_s"] = None if timestamp is None else float(timestamp)
        state = _copy_array(message.state, (15,))
        covariance = _copy_array(message.covariance, (15, 15))
        payload["state"] = state
        payload["covariance"] = covariance
        payload["P_estimate"] = covariance.copy()
    return _normal(payload)


class AuditEnv(v7.PartialGNSSEnv):
    """v7 noisy-GNSS adapter with a read-only native queue audit."""

    def __init__(self, *args: Any, phase_deadline=float("inf"), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v10_phase_deadline = phase_deadline
        self.v10_live_audits: list[dict[str, Any]] = []
        _CAPTURED_ENVS.append(self)

    def step(self, actions: Mapping[str, Any]):
        if perf_counter() >= self.v10_phase_deadline:
            raise RuntimeError("v10 frozen phase wall limit")
        step = int(getattr(self, "timestep", 0)) + 1
        pre_estimates = {uid: _copy_array(self.ekfs[uid].state, (15,)) for uid in UIDS}
        pre_payloads = {
            uid: {"state": value.copy(), "covariance": _copy_array(self.ekfs[uid].covariance, (15, 15))}
            for uid, value in pre_estimates.items()
        }
        goals = {uid: _copy_array(self.goals[uid], (3,)) for uid in UIDS}
        queue_before = _queue_snapshot(self.queue_state)
        pending_before = _message_buffer(self.pending_messages)
        arrived_before = _message_buffer(self.arrived_messages)
        passed_actions = _normal(actions)
        navigation = _navigation_check(step, pre_estimates, goals, actions)
        base = {
            "step": step,
            "pre_own_estimates": pre_estimates,
            "pre_payloads": pre_payloads,
            "goals": goals,
            "passed_actions": passed_actions,
            "navigation_check": navigation,
            "queue_state_before": queue_before,
            "pending_messages_before": pending_before,
            "arrived_messages_before": arrived_before,
        }
        try:
            result = super().step(actions)
        except BaseException as exc:
            base.update({"complete": False, "exception": {"type": type(exc).__name__, "reason": str(exc)}})
            self.v10_live_audits.append(_normal(base))
            raise

        info = result[-1] if isinstance(result, tuple) and result else {}
        overhead = copy.deepcopy(info.get("measurement_overhead_events", [])) if isinstance(info, Mapping) else []
        packet_load = 0.0
        for row in overhead:
            if not isinstance(row, Mapping):
                continue
            try:
                load = float(row.get("load_packets_applied", row.get("measurement_packet_count", 0.0)))
            except (TypeError, ValueError):
                load = 0.0
            packet_load += max(0.0, load)
        queue_after = _queue_snapshot(self.queue_state)
        pending_after = _message_buffer(self.pending_messages)
        arrived_after = _message_buffer(self.arrived_messages)
        overhead_rows = []
        for row in overhead:
            item = dict(row) if isinstance(row, Mapping) else {"value": row}
            item["queue_state_before"] = queue_before
            item["queue_state_after"] = queue_after
            item["measurement_overhead_bytes_equivalent512"] = float(
                max(0.0, float(item.get("load_packets_applied", item.get("measurement_packet_count", 0.0)) or 0.0))
                * 512.0
            )
            overhead_rows.append(item)
        raw_uwb = getattr(getattr(self, "snapshot", None), "uwb_measurements", {})
        selected = _selected_message(self, actions)
        transmitted_payloads = _payload_snapshot(self)
        sender_payload_state_error = 0.0
        sender_payload_covariance_error = 0.0
        if transmitted_payloads:
            for uid, payload in transmitted_payloads.items():
                if not isinstance(payload, Mapping) or uid not in pre_estimates:
                    continue
                sender_payload_state_error = max(
                    sender_payload_state_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(payload.get("state"), dtype=float)
                                - pre_estimates[uid]
                            )
                        )
                    ),
                )
                if uid in pre_payloads:
                    sender_payload_covariance_error = max(
                        sender_payload_covariance_error,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(payload.get("covariance"), dtype=float)
                                    - np.asarray(pre_payloads[uid]["covariance"], dtype=float)
                                )
                            )
                        ),
                    )
        post_physical = {
            uid: {
                "position_m": _copy_array(self.states[uid].position_m, (3,)),
                "velocity_mps": _copy_array(self.states[uid].velocity_mps, (3,)),
            }
            for uid in UIDS
        }
        budget_events = copy.deepcopy(info.get("measurement_budget_events", [])) if isinstance(info, Mapping) else []
        gate_events = copy.deepcopy(info.get("measurement_gate_events", [])) if isinstance(info, Mapping) else []
        message_events = copy.deepcopy(
            info.get("message_buffer", {}).get("events", [])
            if isinstance(info, Mapping) and isinstance(info.get("message_buffer"), Mapping)
            else []
        )
        base.update(
            {
                "complete": True,
                "queue_state_after": queue_after,
                "pending_messages_after": pending_after,
                "arrived_messages_after": arrived_after,
                "message_events": message_events,
                "selected_peer_message": selected,
                "measurement_overhead_events": _normal(overhead_rows),
                "measurement_overhead_packet_load": float(packet_load),
                "measurement_overhead_bytes_equivalent512": float(packet_load * 512.0),
                "measurement_budget_step_events": _normal(budget_events),
                "native_gate_events": _normal(gate_events),
                "raw_uwb_measurements": _normal(raw_uwb),
                "actual_next_physical": _normal(post_physical),
                "transmitted_payloads": transmitted_payloads,
                "sender_payload_state_error_max": float(sender_payload_state_error),
                "sender_payload_covariance_error_max": float(sender_payload_covariance_error),
                "controller_command_error_max": float(navigation["max_abs_error"]),
                "own_estimator_mro": {
                    uid: [base.__name__ for base in type(self.ekfs[uid]).__mro__]
                    for uid in UIDS
                },
                "native_info": {
                    "measurement_budget_remaining": _normal(info.get("measurement_budget_remaining", {})),
                    "measurement_budget_used": _normal(info.get("measurement_budget_used", {})),
                    "safety_projection_count": int(info.get("safety_projection_count", 0)),
                    "boundary_projection_count": int(info.get("boundary_projection_count", 0)),
                },
            }
        )
        self.v10_live_audits.append(_normal(base))
        return result


def _safety_watch(
    env: Any,
    info: Mapping[str, Any],
    previous_positions: Mapping[str, Any],
    previous_velocities: Mapping[str, Any],
    original_check: Any,
):
    """Apply v9 safety checks plus conservative estimator-state watching."""

    result = original_check(env, info, previous_positions, previous_velocities)
    if int(info.get("boundary_projection_count", 0) or 0) != 0:
        raise RuntimeError("v10 conservative safety watch: boundary projection")
    if int(info.get("safety_projection_count", 0) or 0) != 0:
        raise RuntimeError("v10 conservative safety watch: safety projection")
    for uid, estimator in getattr(env, "ekfs", {}).items():
        state = np.asarray(estimator.state, dtype=float)
        if state.shape != (15,) or not np.isfinite(state).all():
            raise RuntimeError(f"v10 conservative safety watch: non-finite state {uid}")
        if np.any(np.abs(state) >= 1.0e3):
            raise RuntimeError(f"v10 conservative safety watch: estimator state magnitude {uid}")
        covariance = np.asarray(estimator.covariance, dtype=float)
        if not np.isfinite(covariance).all():
            raise RuntimeError(f"v10 conservative safety watch: non-finite covariance {uid}")
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-8:
            raise RuntimeError(f"v10 conservative safety watch: non-PSD covariance {uid}")
        if float(np.trace(covariance)) >= 1.0e4:
            raise RuntimeError(f"v10 conservative safety watch: covariance trace limit {uid}")
    return result


def _feedback_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    frames = {int(frame.get("step")): frame for frame in record.get("frames", []) if isinstance(frame, Mapping)}
    pairs: list[dict[str, Any]] = []
    max_input_error = 0.0
    counterfactual_command_delta_max = 0.0
    command_change_count = 0
    for step, frame in sorted(frames.items()):
        audit = frame.get("audit", {})
        event = audit.get("fusion_event", {}) if isinstance(audit, Mapping) else {}
        if not isinstance(event, Mapping) or not bool(event.get("accepted", False)):
            continue
        next_frame = frames.get(step + 1)
        if next_frame is None:
            continue
        next_audit = next_frame.get("audit", {})
        next_live = next_audit.get("live_queue", {}) if isinstance(next_audit, Mapping) else {}
        pre = next_live.get("pre_own_estimates", {}).get(TARGET) if isinstance(next_live, Mapping) else None
        post = audit.get("states", {}).get(TARGET) if isinstance(audit, Mapping) else None
        if pre is None or post is None:
            continue
        input_error = float(np.sqrt(np.mean((np.asarray(pre, dtype=float) - np.asarray(post, dtype=float)) ** 2)))
        max_input_error = max(max_input_error, input_error)
        next_action = next_audit.get("actions", {}).get(TARGET, {}) if isinstance(next_audit, Mapping) else {}
        actual = np.asarray(next_action.get("delta_v", np.zeros(3)), dtype=float)
        goal = np.asarray(getattr(record, "goals", {}).get(TARGET, np.zeros(3)) if hasattr(record, "goals") else np.zeros(3), dtype=float)
        # The frame audit stores the goal through live_queue; no truth field is
        # consulted.  Fall back to that observable controller input if needed.
        live_next = next_audit.get("live_queue", {}) if isinstance(next_audit, Mapping) else {}
        if isinstance(live_next, Mapping):
            goal = np.asarray(live_next.get("goals", {}).get(TARGET, goal), dtype=float)
        before_state = np.asarray(event.get("state_before", post), dtype=float)
        pd_before = np.asarray(fixed_navigation_command(before_state[:6], goal)["command_delta_v_mps"], dtype=float)
        command_error = float(np.max(np.abs(actual - pd_before)))
        counterfactual_command_delta_max = max(counterfactual_command_delta_max, command_error)
        command_changed = bool(command_error > NAVIGATION_TOLERANCE)
        command_change_count += int(command_changed)
        pairs.append(
            {
                "step": step,
                "next_step": step + 1,
                "accepted": True,
                "controller_input_rms_error": input_error,
                "actual_next_delta_v": actual,
                "pd_delta_v_from_ci_before_state": pd_before,
                "counterfactual_command_delta_error": command_error,
                "command_error": command_error,
                "command_changed_after_update": command_changed,
                "counterfactual_delta_v": actual - pd_before,
                "truth_used": False,
            }
        )
    return {
        "feedback_update_pairs": len(pairs),
        "feedback_command_change_pairs": command_change_count,
        "max_controller_input_error": float(max_input_error),
        "counterfactual_command_delta_max": float(counterfactual_command_delta_max),
        # Compatibility alias; safety admission uses controller_command_error_max,
        # which checks the actual controller input against its fixed PD command.
        "max_command_error": float(counterfactual_command_delta_max),
        "pairs": pairs,
        "scope": "accepted CI update with adjacent native navigation frame; no truth control input",
    }


def _attach_live_audits(record: dict[str, Any], env: AuditEnv, method: str, policy: str) -> None:
    captures = {int(item.get("step")): item for item in env.v10_live_audits}
    for frame in record.get("frames", []):
        step = int(frame.get("step", -1))
        capture = captures.get(step)
        if capture is None or not capture.get("complete", False):
            continue
        audit = frame.setdefault("audit", {})
        # CI's actual event remains the immutable kernel record.  Preserve the
        # existing cache field for historical record compatibility; v10 never
        # uses it as an estimator input.
        audit["live_queue"] = copy.deepcopy(capture)
    execution = record.setdefault("execution_audit", {})
    live = [frame.get("audit", {}).get("live_queue", {}) for frame in record.get("frames", [])]
    arrivals = [event for item in live for event in item.get("message_events", []) if isinstance(event, Mapping) and event.get("status") == "arrived"]
    selected_available = [item for item in live if item.get("selected_peer_message", {}).get("available", False)]
    overhead_load = float(sum(float(item.get("measurement_overhead_packet_load", 0.0) or 0.0) for item in live))
    max_nav_error = float(max((item.get("navigation_check", {}).get("max_abs_error", 0.0) for item in live), default=0.0))
    source_constraints_ok = True
    window_legal = True
    for frame in record.get("frames", []):
        audit = frame.get("audit", {})
        selection = str(audit.get("selection", "none"))
        step = int(frame.get("step", 0))
        live_queue = audit.get("live_queue", {}) if isinstance(audit, Mapping) else {}
        passed_actions = live_queue.get("passed_actions", {}) if isinstance(live_queue, Mapping) else {}
        for uid in PEERS:
            peer_action = passed_actions.get(uid, {}) if isinstance(passed_actions, Mapping) else {}
            if str(peer_action.get("measurement_uav_id", "none")) != "none":
                source_constraints_ok = False
        if step <= 5 and selection != "none":
            source_constraints_ok = window_legal = False
        if selection != "none":
            legal = frame.get("observable", {}).get("pre_legal_sources", [])
            window_legal &= step in DECISION_STEPS and selection in legal and selection in PEERS
    feedback = _feedback_audit(record)
    execution.update(
        {
            "method": method,
            "policy": policy,
            "live_steps": len(live),
            "actual_message_arrival_count": len(arrivals),
            "request_overhead_packet_load": overhead_load,
            "request_overhead_bytes_equivalent512": overhead_load * 512.0,
            "selected_message_availability_count": len(selected_available),
            "selected_message_availability_scope": "native arrived message; never request-response packet",
            "selected_message_availability_not_request_response_packets": True,
            "navigation_checks_max_error": max_nav_error,
            "controller_command_error_max": max_nav_error,
            "sender_payload_state_error_max": float(
                max(
                    (
                        item.get("sender_payload_state_error_max", 0.0)
                        for item in live
                        if isinstance(item, Mapping)
                    ),
                    default=0.0,
                )
            ),
            "sender_payload_covariance_error_max": float(
                max(
                    (
                        item.get("sender_payload_covariance_error_max", 0.0)
                        for item in live
                        if isinstance(item, Mapping)
                    ),
                    default=0.0,
                )
            ),
            "native_live_queue": True,
            "navigation_truth_used": False,
            "feedback_update_pairs": feedback["feedback_update_pairs"],
            "feedback_command_change_pairs": feedback["feedback_command_change_pairs"],
            "max_controller_input_error": feedback["max_controller_input_error"],
            "counterfactual_command_delta_max": feedback["counterfactual_command_delta_max"],
            "max_command_error": feedback["max_command_error"],
            "feedback": feedback,
            "source_payload_estimates_only": True,
            "no_fixed_queue": True,
            "source_constraints_uid0_uid1_none": source_constraints_ok,
            "selection_window_legal": window_legal,
            "native_arrival_statuses_only": True,
        }
    )


def physical(
    case: Mapping[str, Any],
    spec: Mapping[str, Any],
    method: str,
    policy: str,
) -> dict[str, Any]:
    """Run one native v9 episode with v10 live queue auditing."""

    _main_thread_guard()
    _validate_method_policy(method, policy)
    execution_spec = _execution_spec(spec)
    with _API_LOCK:
        _main_thread_guard()
        _CAPTURED_ENVS.clear()
        original_check = v7.check_safety

        def guarded_check(env, info, previous_positions, previous_velocities):
            return _safety_watch(env, info, previous_positions, previous_velocities, original_check)

        def environment_factory(*args, **kwargs):
            return AuditEnv(*args, phase_deadline=execution_spec.get("phase_deadline", float("inf")), **kwargs)
        with patch.object(v7, "PartialGNSSEnv", environment_factory), patch.object(v7, "check_safety", guarded_check):
            with _patched_method(method):
                # v9 owns case preparation, fixed navigation, noisy GNSS, and
                # failure/status semantics.  Let its exceptions propagate so
                # the root runner can record them without reinterpretation.
                record = v9.episode(case, execution_spec, policy)
        if not _CAPTURED_ENVS:
            raise RuntimeError("v10 physical episode did not construct AuditEnv")
        env = _CAPTURED_ENVS[-1]
        result = copy.deepcopy(record)
        result["kind"] = f"actual_live_queue_{method}"
        result["method"] = method
        result["policy"] = policy
        result.setdefault("phase", "physical")
        _attach_live_audits(result, env, method, policy)
        result["execution_audit"] = v7.old.normalize(result["execution_audit"])
        return v7.old.normalize(result)


__all__ = [
    "AuditEnv",
    "DECISION_STEPS",
    "METHODS",
    "POLICIES",
    "physical",
    "replay",
    "_safety_watch",
]
