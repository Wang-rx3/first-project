"""Read-only pre-pass reorientation audit over frozen single-UAV trajectories.

This module imports only the previous pure audit by file path for its source
identity checks and frame projection. It never loads an environment, controller,
policy, simulator, or training module and never performs a rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path("logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026")
PRIOR_ROOT = Path("logs/preliminary/single_uav_post_lateral_exit_reorientation_audit_seed2026")
OUTPUT_ROOT = Path("logs/preliminary/single_uav_prepass_reorientation_reference_audit_seed2026")
PRIOR_MODULE_PATH = Path(__file__).with_name("single_uav_post_lateral_exit_reorientation_audit.py")
EXPECTED_PREREGISTRATION_HASH = "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2"
EXPECTED_PREVIOUS_STATUS = "PASS"
EXIT_THRESHOLD_MPS = 0.5
POSITIVE_EPSILON = 1.0e-8
PASS_CLEARANCE_M = 2.5
PREVIOUS_FAIL_FAST_REASON = "first pass transition row could already have Goal Recovery active by evaluator timing"

SUCCESS_IDS = tuple(f"test_r99_episode_{suffix}" for suffix in (
    "0001", "0003", "0007", "0008", "0009", "0010", "0015", "0016", "0017", "0018", "0019"
))
FAILURE_IDS = tuple(f"test_r99_episode_{suffix}" for suffix in (
    "0002", "0004", "0005", "0006", "0011", "0012", "0020"
))
SPECIAL_IDS = {
    "test_r99_episode_0013": "SAFETY_DOMINATED_SPECIAL_CASE",
    "test_r99_episode_0014": "BOUNDARY_DOMINATED_SPECIAL_CASE",
}
ALL_IDS = tuple(f"test_r99_episode_{index:04d}" for index in range(1, 21))
OUTPUT_FILES = (
    "prepass_reorientation_summary.json",
    "prepass_success_reference.csv",
    "prepass_failure_vs_reference.csv",
    "prepass_failure_segmentation.csv",
    "prepass_episode_diagnostics.csv",
    "prepass_reorientation_report.md",
)

DIAGNOSIS_BY_FLAG = (
    ("radial_intent_not_sustained", "RADIAL_INTENT_NOT_SUSTAINED"),
    ("radial_propulsion_insufficient", "RADIAL_PROPULSION_INSUFFICIENT"),
    ("radial_motion_response_weak", "RADIAL_MOTION_RESPONSE_WEAK"),
    ("lateral_reacceleration", "LATERAL_REACCELERATION"),
    ("pass_progress_regression", "PASS_PROGRESS_REGRESSION"),
    ("pipeline_radial_attenuation", "PIPELINE_RADIAL_ATTENUATION"),
)

# All metric records have value, finite sample count, and an explicit reason
# when undefined. Fractions additionally carry numerator and denominator.
METRIC_NAMES = (
    "actor_goalward_fraction", "final_goalward_fraction", "actual_goalward_fraction",
    "actor_goalward_radial_step_fraction_prepass", "final_goalward_radial_step_fraction_prepass",
    "actual_goalward_radial_step_fraction_prepass",
    "actor_goalward_sustained_window_fraction", "final_goalward_sustained_window_fraction",
    "actual_goalward_sustained_3step_fraction",
    "actor_radial_mean_mps", "final_radial_mean_mps", "estimated_radial_mean_mps", "actual_radial_mean_mps",
    "actor_radial_positive_mean_mps", "final_radial_positive_mean_mps",
    "estimated_radial_positive_mean_mps", "actual_radial_positive_mean_mps",
    "estimated_radial_velocity_mean_mps", "actual_radial_velocity_mean_mps",
    "actor_lateral_mean_mps", "final_lateral_mean_mps", "estimated_lateral_mean_mps", "actual_lateral_mean_mps",
    "actor_mean_abs_lateral_mps", "final_mean_abs_lateral_mps",
    "actor_to_final_radial_retention", "actor_radial_to_abs_lateral_ratio", "final_radial_to_abs_lateral_ratio",
    "goal_distance_ols_slope_m_per_step", "goal_distance_net_reduction_m",
    "margin_start_m", "margin_final_m", "margin_best_m", "margin_best_step",
    "margin_best_minus_start_m", "margin_final_minus_start_m", "margin_best_minus_final_m",
    "margin_ols_slope_m_per_step",
    "lateral_reacceleration_run_count", "lateral_reacceleration_duration_steps", "lateral_reacceleration_peak_abs_mps",
    "actor_original_side_fraction", "actor_counter_side_fraction", "actor_lateral_sign_change_count",
)

REFERENCE_METRICS = METRIC_NAMES
COMPARISON_METRICS = METRIC_NAMES


class AuditError(RuntimeError):
    """Raised when a frozen-input or output contract fails closed."""


class PrepassContaminationError(AuditError):
    """Includes the exact pre-pass rows and flags that contaminated a reference."""

    def __init__(
        self,
        episode_id: str,
        row: Mapping[str, Any],
        fields: Sequence[str],
        *,
        local_exit_step: int | None = None,
        first_pass_step: int | None = None,
    ):
        self.details = {
            "episode_id": episode_id,
            "local_exit_step": local_exit_step,
            "first_pass_step": first_pass_step,
            "first_offending_step": integer(row.get("step")),
            "goal_recovery_reason": row.get("goal_recovery_correction_reason") or None,
            "contaminating_fields": list(fields),
        }
        super().__init__(
            "PREPASS_REFERENCE_CONTROLLER_CONTAMINATION "
            f"episode_id={episode_id} local_exit_step={local_exit_step} "
            f"first_pass_step={first_pass_step} first_offending_step={self.details['first_offending_step']} "
            f"goal_recovery_reason={self.details['goal_recovery_reason']} fields={','.join(fields)}"
        )


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def integer(value: Any) -> int | None:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def type7_quantile(values: Sequence[float], probability: float) -> float | None:
    """Hyndman-Fan type-7 linear sample quantile."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    weight = h - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_record(
    value: Any,
    sample_count: int,
    *,
    reason: str | None = None,
    numerator: Any = None,
    denominator: Any = None,
) -> dict[str, Any]:
    number = finite_float(value)
    if number is None and reason is None:
        reason = "UNDEFINED_FROM_FINITE_SAMPLES"
    record: dict[str, Any] = {"value": number, "sample_count": int(sample_count), "unavailable_reason": reason if number is None else None}
    if numerator is not None or denominator is not None:
        record["numerator"] = finite_float(numerator)
        record["denominator"] = finite_float(denominator)
    return record


def null_metric(reason: str, sample_count: int = 0) -> dict[str, Any]:
    return metric_record(None, sample_count, reason=reason)


def _metric_value(metrics: Mapping[str, Any], name: str) -> float | None:
    entry = metrics.get(name)
    if isinstance(entry, Mapping):
        return finite_float(entry.get("value"))
    return finite_float(entry)


def _metric_count(metrics: Mapping[str, Any], name: str) -> int:
    entry = metrics.get(name)
    return integer(entry.get("sample_count")) or 0 if isinstance(entry, Mapping) else 0


def _finite_field(rows: Sequence[Mapping[str, Any]], *keys: str) -> list[float | None]:
    result: list[float | None] = []
    for row in rows:
        value = None
        for key in keys:
            value = finite_float(row.get(key))
            if value is not None:
                break
        result.append(value)
    return result


def _mean(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [value for value in values if value is not None]
    return metric_record(sum(finite) / len(finite), len(finite)) if finite else null_metric("NO_FINITE_SAMPLES")


def _positive_mean(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [value for value in values if value is not None]
    return metric_record(sum(max(0.0, value) for value in finite) / len(finite), len(finite)) if finite else null_metric("NO_FINITE_SAMPLES")


def _fraction(numerator: int, denominator: int, *, reason: str = "NO_FINITE_SAMPLES") -> dict[str, Any]:
    if denominator <= 0:
        return metric_record(None, 0, reason=reason, numerator=numerator, denominator=denominator)
    return metric_record(numerator / denominator, denominator, numerator=numerator, denominator=denominator)


def _sustained_fraction(rows: Sequence[Mapping[str, Any]], values: Sequence[float | None], *, window: int, minimum: int) -> dict[str, Any]:
    numerator = 0
    denominator = 0
    steps = [integer(row.get("step")) for row in rows]
    for start in range(0, max(0, len(rows) - window + 1)):
        indices = range(start, start + window)
        if any(steps[index] is None for index in indices):
            continue
        if [steps[index] for index in indices] != list(range(steps[start], steps[start] + window)):
            continue
        section = [values[index] for index in indices]
        if any(value is None for value in section):
            continue
        denominator += 1
        numerator += int(sum(value > POSITIVE_EPSILON for value in section if value is not None) >= minimum)
    return _fraction(numerator, denominator, reason="NO_COMPLETE_FINITE_WINDOWS")


def _ols_slope(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, Any]:
    pairs = [(integer(row.get("step")), finite_float(row.get(value_key))) for row in rows]
    points = [(float(step), value) for step, value in pairs if step is not None and value is not None]
    if len(points) < 2:
        return null_metric("FEWER_THAN_TWO_FINITE_POINTS", len(points))
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0.0:
        return null_metric("NO_DISTINCT_FINITE_STEPS", len(points))
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return metric_record(slope, len(points))


def _record_first_last_reduction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points = [(integer(row.get("step")), finite_float(row.get("goal_distance_m"))) for row in rows]
    finite = [(step, value) for step, value in points if step is not None and value is not None]
    if len(finite) < 2:
        return null_metric("FEWER_THAN_TWO_FINITE_GOAL_DISTANCES", len(finite))
    return metric_record(finite[0][1] - finite[-1][1], len(finite))


def _margin_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    points = [(integer(row.get("step")), finite_float(row.get("passed_margin_m"))) for row in rows]
    finite = [(step, value) for step, value in points if step is not None and value is not None]
    names = ("margin_start_m", "margin_final_m", "margin_best_m", "margin_best_step", "margin_best_minus_start_m", "margin_final_minus_start_m", "margin_best_minus_final_m")
    if not finite:
        return {name: null_metric("NO_FINITE_MARGIN_SAMPLES") for name in names} | {"margin_ols_slope_m_per_step": null_metric("NO_FINITE_MARGIN_SAMPLES")}
    start = finite[0][1]
    final = finite[-1][1]
    best = max(value for _, value in finite)
    best_step = min(step for step, value in finite if value == best)
    count = len(finite)
    return {
        "margin_start_m": metric_record(start, count),
        "margin_final_m": metric_record(final, count),
        "margin_best_m": metric_record(best, count),
        "margin_best_step": metric_record(best_step, count),
        "margin_best_minus_start_m": metric_record(best - start, count),
        "margin_final_minus_start_m": metric_record(final - start, count),
        "margin_best_minus_final_m": metric_record(best - final, count),
        "margin_ols_slope_m_per_step": _ols_slope(rows, "passed_margin_m"),
    }


def _reacceleration_metrics(rows: Sequence[Mapping[str, Any]], actual_lateral: Sequence[float | None]) -> dict[str, dict[str, Any]]:
    valid = sum(value is not None for value in actual_lateral)
    if not valid:
        return {
            "lateral_reacceleration_run_count": null_metric("NO_FINITE_ACTUAL_LATERAL_SAMPLES"),
            "lateral_reacceleration_duration_steps": null_metric("NO_FINITE_ACTUAL_LATERAL_SAMPLES"),
            "lateral_reacceleration_peak_abs_mps": null_metric("NO_FINITE_ACTUAL_LATERAL_SAMPLES"),
        }
    steps = [integer(row.get("step")) for row in rows]
    qualifying: list[list[float]] = []
    run: list[float] = []
    previous_step: int | None = None
    for step, speed in zip(steps, actual_lateral):
        continuing = step is not None and previous_step is not None and step == previous_step + 1
        if speed is not None and abs(speed) > EXIT_THRESHOLD_MPS and (not run or continuing):
            run.append(abs(speed))
        else:
            if len(run) >= 3:
                qualifying.append(run)
            run = [abs(speed)] if speed is not None and abs(speed) > EXIT_THRESHOLD_MPS else []
        previous_step = step
    if len(run) >= 3:
        qualifying.append(run)
    count = len(qualifying)
    duration = sum(len(group) for group in qualifying)
    peak = max((max(group) for group in qualifying), default=0.0)
    return {
        "lateral_reacceleration_run_count": metric_record(count, valid),
        "lateral_reacceleration_duration_steps": metric_record(duration, valid),
        "lateral_reacceleration_peak_abs_mps": metric_record(peak, valid),
    }


def compute_metrics(rows: Sequence[Mapping[str, Any]], *, initial_lateral_sign: int = 0) -> dict[str, dict[str, Any]]:
    """Compute finite-only metrics on one explicitly bounded window."""
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    actor = _finite_field(ordered, "audit_actor_radial_delta_v_mps", "actor_raw_radial_delta_v_mps", "actor_raw_delta_v_mps_radial")
    final = _finite_field(ordered, "audit_final_radial_delta_v_mps", "final_applied_radial_delta_v_mps", "final_applied_delta_v_mps_radial")
    estimated = _finite_field(ordered, "audit_estimated_radial_velocity_mps", "estimated_radial_velocity_mps")
    actual = _finite_field(ordered, "audit_actual_radial_velocity_mps", "actual_radial_velocity_mps")
    actor_lat = _finite_field(ordered, "audit_actor_lateral_delta_v_mps", "actor_raw_lateral_delta_v_mps", "actor_raw_delta_v_mps_lateral")
    final_lat = _finite_field(ordered, "audit_final_lateral_delta_v_mps", "final_applied_lateral_delta_v_mps", "final_applied_delta_v_mps_lateral")
    estimated_lat = _finite_field(ordered, "audit_estimated_lateral_velocity_mps", "estimated_lateral_velocity_mps")
    actual_lat = _finite_field(ordered, "audit_actual_lateral_velocity_mps", "actual_lateral_velocity_mps")
    result: dict[str, dict[str, Any]] = {}

    for name, values in (("actor", actor), ("final", final), ("actual", actual)):
        numerator = sum(value is not None and value > POSITIVE_EPSILON for value in values)
        result[f"{name}_goalward_fraction"] = _fraction(numerator, sum(value is not None for value in values))
    # Preserve the compact internal names used by the diagnosis rules while
    # exposing the preregistered, explicitly pre-pass field names verbatim.
    result["actor_goalward_radial_step_fraction_prepass"] = dict(result["actor_goalward_fraction"])
    result["final_goalward_radial_step_fraction_prepass"] = dict(result["final_goalward_fraction"])
    result["actual_goalward_radial_step_fraction_prepass"] = dict(result["actual_goalward_fraction"])
    result["actor_goalward_sustained_window_fraction"] = _sustained_fraction(ordered, actor, window=5, minimum=4)
    result["final_goalward_sustained_window_fraction"] = _sustained_fraction(ordered, final, window=5, minimum=4)
    result["actual_goalward_sustained_3step_fraction"] = _sustained_fraction(ordered, actual, window=3, minimum=3)

    for name, values in (("actor", actor), ("final", final), ("estimated", estimated), ("actual", actual)):
        result[f"{name}_radial_mean_mps"] = _mean(values)
        result[f"{name}_radial_positive_mean_mps"] = _positive_mean(values)
    result["estimated_radial_velocity_mean_mps"] = result["estimated_radial_mean_mps"]
    result["actual_radial_velocity_mean_mps"] = result["actual_radial_mean_mps"]
    for name, values in (("actor", actor_lat), ("final", final_lat), ("estimated", estimated_lat), ("actual", actual_lat)):
        result[f"{name}_lateral_mean_mps"] = _mean(values)
    for name, values in (("actor", actor_lat), ("final", final_lat)):
        finite = [abs(value) for value in values if value is not None]
        result[f"{name}_mean_abs_lateral_mps"] = metric_record(sum(finite) / len(finite), len(finite)) if finite else null_metric("NO_FINITE_LATERAL_SAMPLES")

    pairs = [(a, f) for a, f in zip(actor, final) if a is not None and f is not None]
    actor_positive_sum = sum(max(0.0, a) for a, _ in pairs)
    final_positive_sum = sum(max(0.0, f) for _, f in pairs)
    if not pairs:
        result["actor_to_final_radial_retention"] = metric_record(None, 0, reason="NO_PAIRED_FINITE_ACTOR_FINAL_SAMPLES", numerator=final_positive_sum, denominator=actor_positive_sum)
    elif actor_positive_sum <= POSITIVE_EPSILON:
        result["actor_to_final_radial_retention"] = metric_record(None, len(pairs), reason="ACTOR_POSITIVE_RADIAL_SUM_NOT_ABOVE_EPSILON", numerator=final_positive_sum, denominator=actor_positive_sum)
    else:
        result["actor_to_final_radial_retention"] = metric_record(final_positive_sum / actor_positive_sum, len(pairs), numerator=final_positive_sum, denominator=actor_positive_sum)
    for name, radial, lateral in (("actor", actor, actor_lat), ("final", final, final_lat)):
        radial_positive = _metric_value(result, f"{name}_radial_positive_mean_mps")
        lateral_finite = [abs(value) for value in lateral if value is not None]
        lateral_mean = sum(lateral_finite) / len(lateral_finite) if lateral_finite else None
        if radial_positive is None:
            result[f"{name}_radial_to_abs_lateral_ratio"] = null_metric("RADIAL_POSITIVE_MEAN_UNAVAILABLE")
        elif lateral_mean is None:
            result[f"{name}_radial_to_abs_lateral_ratio"] = null_metric("NO_FINITE_LATERAL_SAMPLES")
        elif lateral_mean <= POSITIVE_EPSILON:
            result[f"{name}_radial_to_abs_lateral_ratio"] = null_metric("MEAN_ABS_LATERAL_NOT_ABOVE_EPSILON", len(lateral_finite))
        else:
            result[f"{name}_radial_to_abs_lateral_ratio"] = metric_record(radial_positive / lateral_mean, min(_metric_count(result, f"{name}_radial_positive_mean_mps"), len(lateral_finite)))

    result["goal_distance_ols_slope_m_per_step"] = _ols_slope(ordered, "goal_distance_m")
    result["goal_distance_net_reduction_m"] = _record_first_last_reduction(ordered)
    result.update(_margin_metrics(ordered))
    result.update(_reacceleration_metrics(ordered, actual_lat))

    actor_nonzero = [value for value in actor_lat if value is not None and abs(value) > POSITIVE_EPSILON]
    if initial_lateral_sign == 0:
        result["actor_original_side_fraction"] = null_metric("INITIAL_ENCOUNTER_LATERAL_SIGN_UNAVAILABLE", len(actor_nonzero))
        result["actor_counter_side_fraction"] = null_metric("INITIAL_ENCOUNTER_LATERAL_SIGN_UNAVAILABLE", len(actor_nonzero))
    elif not actor_nonzero:
        result["actor_original_side_fraction"] = null_metric("NO_NONZERO_FINITE_ACTOR_LATERAL_SAMPLES")
        result["actor_counter_side_fraction"] = null_metric("NO_NONZERO_FINITE_ACTOR_LATERAL_SAMPLES")
    else:
        same = sum((1 if value > 0 else -1) == initial_lateral_sign for value in actor_nonzero)
        result["actor_original_side_fraction"] = _fraction(same, len(actor_nonzero))
        result["actor_counter_side_fraction"] = _fraction(len(actor_nonzero) - same, len(actor_nonzero))
    signs = [1 if value > 0 else -1 for value in actor_nonzero]
    if not signs:
        result["actor_lateral_sign_change_count"] = null_metric("NO_NONZERO_FINITE_ACTOR_LATERAL_SAMPLES")
    else:
        result["actor_lateral_sign_change_count"] = metric_record(sum(a != b for a, b in zip(signs, signs[1:])), len(signs))

    for name in METRIC_NAMES:
        result.setdefault(name, null_metric("METRIC_NOT_COMPUTABLE"))
    return result


def first_local_exit_step(rows: Sequence[Mapping[str, Any]], search_anchor_step: int, *, threshold_mps: float = EXIT_THRESHOLD_MPS) -> int | None:
    ordered = sorted((row for row in rows if integer(row.get("step")) is not None and integer(row.get("step")) >= search_anchor_step), key=lambda row: integer(row.get("step")) or 0)
    for index in range(max(0, len(ordered) - 2)):
        trio = ordered[index:index + 3]
        steps = [integer(row.get("step")) for row in trio]
        speeds = [finite_float(row.get("audit_actual_lateral_velocity_mps", row.get("actual_lateral_velocity_mps"))) for row in trio]
        if steps[0] is not None and steps == list(range(steps[0], steps[0] + 3)) and all(speed is not None and abs(speed) <= threshold_mps for speed in speeds):
            return steps[0]
    return None


def exact_pass_gate(row: Mapping[str, Any]) -> bool:
    margin = finite_float(row.get("passed_margin_m"))
    clearance = finite_float(row.get("relevant_obstacle_clearance_m", row.get("signed_obstacle_clearance_m")))
    return truthy(row.get("passed_obstacle")) and margin is not None and clearance is not None and margin > 0.0 and clearance >= PASS_CLEARANCE_M


def success_prepass_window(
    rows: Sequence[Mapping[str, Any]], local_exit_step: int | None,
) -> tuple[list[Mapping[str, Any]], int | None, Mapping[str, Any] | None, str | None]:
    """Return [local exit, first pass), with the pass-transition row separate."""
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    transition = next((row for row in ordered if integer(row.get("step")) is not None and exact_pass_gate(row)), None)
    first_pass = integer(transition.get("step")) if transition is not None else None
    if local_exit_step is None:
        return [], first_pass, transition, "LOCAL_EXIT_NOT_OBSERVED"
    if first_pass is None:
        return [], None, None, "FIRST_PASS_NOT_OBSERVED"
    if first_pass <= local_exit_step:
        return [], first_pass, transition, "NO_PREPASS_REFERENCE_WINDOW"
    window = [row for row in ordered if integer(row.get("step")) is not None and local_exit_step <= integer(row.get("step")) < first_pass]
    if not window:
        return [], first_pass, transition, "NO_PREPASS_REFERENCE_WINDOW"
    return window, first_pass, transition, None


def pass_transition_metadata(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the exact gate-transition row into explicit non-reference fields."""
    if row is None:
        return {
            "transition_passed_obstacle": None,
            "transition_post_bypass_recovery": None,
            "transition_goal_recovery_eligible": None,
            "transition_goal_recovery_active": None,
            "transition_goal_recovery_reason": None,
            "transition_passed_margin": None,
            "transition_clearance": None,
    }
    reason = row.get("goal_recovery_correction_reason")
    clearance = finite_float(row.get("relevant_obstacle_clearance_m"))
    if clearance is None:
        clearance = finite_float(row.get("signed_obstacle_clearance_m"))
    return {
        "transition_passed_obstacle": truthy(row.get("passed_obstacle")),
        "transition_post_bypass_recovery": truthy(row.get("post_bypass_recovery")),
        "transition_goal_recovery_eligible": truthy(row.get("goal_recovery_correction_eligible")),
        "transition_goal_recovery_active": truthy(row.get("goal_recovery_correction_active")),
        "transition_goal_recovery_reason": reason if reason not in (None, "") else None,
        "transition_passed_margin": finite_float(row.get("passed_margin_m")),
        "transition_clearance": clearance,
    }


def validate_success_window_contamination(
    windows: Mapping[str, Sequence[Mapping[str, Any]]],
    local_exit_steps: Mapping[str, int | None] | None = None,
    first_pass_steps: Mapping[str, int | None] | None = None,
) -> None:
    for episode_id in SUCCESS_IDS:
        for row in windows.get(episode_id, ()):
            active_fields = [
                name for name in ("goal_recovery_correction_active", "post_bypass_recovery", "passed_obstacle")
                if truthy(row.get(name))
            ]
            if active_fields:
                raise PrepassContaminationError(
                    episode_id,
                    row,
                    active_fields,
                    local_exit_step=(local_exit_steps or {}).get(episode_id),
                    first_pass_step=(first_pass_steps or {}).get(episode_id),
                )


def make_segments(rows: Sequence[Mapping[str, Any]], exit_step: int | None) -> dict[str, dict[str, Any]]:
    if exit_step is None:
        unavailable = {"rows": [], "start_step": None, "end_step": None, "start_inclusive": None, "end_inclusive": None, "structural_reason": "LOCAL_EXIT_NOT_OBSERVED"}
        return {"WHOLE": dict(unavailable), "PRE_BEST": dict(unavailable), "POST_BEST": dict(unavailable)}
    ordered = sorted((row for row in rows if integer(row.get("step")) is not None and integer(row.get("step")) >= exit_step), key=lambda row: integer(row.get("step")) or 0)
    if not ordered:
        unavailable = {"rows": [], "start_step": exit_step, "end_step": None, "start_inclusive": True, "end_inclusive": True, "structural_reason": "NO_ROWS_AFTER_LOCAL_EXIT"}
        return {"WHOLE": dict(unavailable), "PRE_BEST": dict(unavailable), "POST_BEST": dict(unavailable)}
    margins = [(integer(row.get("step")), finite_float(row.get("passed_margin_m"))) for row in ordered]
    finite = [(step, margin) for step, margin in margins if step is not None and margin is not None]
    if not finite:
        whole = {"rows": ordered, "start_step": exit_step, "end_step": integer(ordered[-1].get("step")), "start_inclusive": True, "end_inclusive": True, "structural_reason": None}
        empty = {"rows": [], "start_step": None, "end_step": None, "start_inclusive": None, "end_inclusive": None, "structural_reason": "BEST_MARGIN_UNAVAILABLE"}
        return {"WHOLE": whole, "PRE_BEST": dict(empty), "POST_BEST": dict(empty)}
    best_margin = max(value for _, value in finite)
    best_step = min(step for step, value in finite if value == best_margin)
    end_step = integer(ordered[-1].get("step"))
    pre = [row for row in ordered if integer(row.get("step")) is not None and integer(row.get("step")) <= best_step]
    post = [row for row in ordered if integer(row.get("step")) is not None and integer(row.get("step")) > best_step]
    whole = {"rows": ordered, "start_step": exit_step, "end_step": end_step, "start_inclusive": True, "end_inclusive": True, "structural_reason": None}
    before = {"rows": pre, "start_step": exit_step, "end_step": best_step, "start_inclusive": True, "end_inclusive": True, "structural_reason": None}
    after = {"rows": post, "start_step": best_step, "end_step": end_step, "start_inclusive": False, "end_inclusive": True, "structural_reason": "EMPTY_POST_BEST_AT_WINDOW_END" if not post else None}
    return {"WHOLE": whole, "PRE_BEST": before, "POST_BEST": after}


def build_success_references(success_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for name in REFERENCE_METRICS:
        values = [_metric_value(success_metrics[episode_id], name) for episode_id in SUCCESS_IDS if episode_id in success_metrics]
        finite = [value for value in values if value is not None]
        if not finite:
            references[name] = {
                "eligible_episode_count": len(SUCCESS_IDS), "n": 0, "median": None, "q1": None, "q3": None,
                "iqr": None, "minimum": None, "maximum": None, "quantile_method": "numpy_linear_type7",
                "unavailable_reason": "NO_FINITE_REFERENCE_VALUES",
            }
            continue
        q1 = type7_quantile(finite, 0.25)
        median = type7_quantile(finite, 0.5)
        q3 = type7_quantile(finite, 0.75)
        references[name] = {
            "eligible_episode_count": len(SUCCESS_IDS), "n": len(finite), "median": median, "q1": q1, "q3": q3,
            "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
            "minimum": min(finite), "maximum": max(finite), "quantile_method": "numpy_linear_type7",
        }
    return references


def compare_iqr(value: Any, reference: Mapping[str, Any]) -> str | None:
    number = finite_float(value)
    q1, q3 = finite_float(reference.get("q1")), finite_float(reference.get("q3"))
    if number is None or q1 is None or q3 is None:
        return None
    if number < q1:
        return "BELOW_IQR"
    if number > q3:
        return "ABOVE_IQR"
    return "WITHIN_IQR"


def _class(metrics: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]], name: str) -> str | None:
    return compare_iqr(_metric_value(metrics, name), references.get(name, {}))


def diagnose_failure(metrics: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]], *, passed: bool) -> dict[str, Any]:
    actor_fraction = _class(metrics, references, "actor_goalward_fraction")
    actor_sustained = _class(metrics, references, "actor_goalward_sustained_window_fraction")
    actor_persistence_assessable = actor_fraction is not None and actor_sustained is not None
    intent_below = actor_persistence_assessable and (actor_fraction == "BELOW_IQR" or actor_sustained == "BELOW_IQR")

    final_fraction = _class(metrics, references, "final_goalward_fraction")
    retention = _class(metrics, references, "actor_to_final_radial_retention")
    pipeline_assessable = actor_fraction is not None and final_fraction is not None and retention is not None
    pipeline_active = None if not pipeline_assessable else actor_fraction != "BELOW_IQR" and (final_fraction == "BELOW_IQR" or retention == "BELOW_IQR")

    actor_positive = _class(metrics, references, "actor_radial_positive_mean_mps")
    propulsion = None if not actor_persistence_assessable or actor_positive is None else actor_positive == "BELOW_IQR"

    actual_fraction = _class(metrics, references, "actual_goalward_fraction")
    actual_mean = _class(metrics, references, "actual_radial_mean_mps")
    motion = None if actor_fraction is None or actual_fraction is None or actual_mean is None else actor_fraction != "BELOW_IQR" and (actual_fraction == "BELOW_IQR" or actual_mean == "BELOW_IQR")

    event_count = _metric_value(metrics, "lateral_reacceleration_run_count")
    duration = _class(metrics, references, "lateral_reacceleration_duration_steps")
    peak = _class(metrics, references, "lateral_reacceleration_peak_abs_mps")
    reacceleration = None if event_count is None or duration is None or peak is None else event_count > 0 and (duration == "ABOVE_IQR" or peak == "ABOVE_IQR")

    regression_value = _metric_value(metrics, "margin_best_minus_final_m")
    regression_class = _class(metrics, references, "margin_best_minus_final_m")
    regression = None if regression_value is None or regression_class is None else regression_value > 0.0 and regression_class == "ABOVE_IQR"

    flags: dict[str, bool | None] = {
        "radial_intent_not_sustained": bool(intent_below) if actor_persistence_assessable else None,
        "pipeline_radial_attenuation": pipeline_active,
        "radial_propulsion_insufficient": propulsion,
        "radial_motion_response_weak": motion,
        "lateral_reacceleration": reacceleration,
        "pass_progress_regression": regression,
    }
    core = actor_persistence_assessable and actor_positive is not None and pipeline_assessable and actual_fraction is not None and actual_mean is not None and regression_class is not None and passed is False
    active = [diagnosis for key, diagnosis in DIAGNOSIS_BY_FLAG if flags.get(key) is True]
    if not core:
        primary = "NOT_ASSESSABLE"
    elif len(active) > 1:
        primary = "MIXED"
    elif len(active) == 1:
        primary = active[0]
    else:
        primary = "PASS_GEOMETRY_NOT_REACHED"
    return {
        "primary_diagnosis": primary,
        "evidence_flags": flags,
        "evidence_comparisons": {
            "actor_goalward_fraction": actor_fraction,
            "actor_goalward_sustained_window_fraction": actor_sustained,
            "final_goalward_fraction": final_fraction,
            "actor_to_final_radial_retention": retention,
            "actor_radial_positive_mean_mps": actor_positive,
            "actual_goalward_fraction": actual_fraction,
            "actual_radial_mean_mps": actual_mean,
            "lateral_reacceleration_duration_steps": duration,
            "lateral_reacceleration_peak_abs_mps": peak,
            "margin_best_minus_final_m": regression_class,
        },
        "core_assessable": bool(core),
    }


def evidence_change_status(old_value: Any, new_value: Any) -> str:
    old = old_value if isinstance(old_value, bool) else (truthy(old_value) if old_value not in (None, "") else None)
    new = new_value if isinstance(new_value, bool) else (truthy(new_value) if new_value not in (None, "") else None)
    if old is None or new is None:
        return "NOT_ASSESSABLE"
    if old and new:
        return "RETAINED_AFTER_PREPASS_CORRECTION"
    if old and not new:
        return "WITHDRAWN_AFTER_PREPASS_CORRECTION"
    return "NOT_PREVIOUSLY_ACTIVE"


def recommendation_decision(failure_metrics: Mapping[str, Mapping[str, Any]], references: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, dict[str, Any]] = {}
    intent_below_ids: list[str] = []
    actor_all_assessable = True
    pipeline_every_intent_below = True
    pipeline_active_count = 0
    intent_count_known = True
    for episode_id in FAILURE_IDS:
        metrics = failure_metrics.get(episode_id, {})
        actor_fraction = _class(metrics, references, "actor_goalward_fraction")
        actor_sustained = _class(metrics, references, "actor_goalward_sustained_window_fraction")
        assessable = actor_fraction is not None and actor_sustained is not None
        actor_all_assessable = actor_all_assessable and assessable
        below = assessable and (actor_fraction == "BELOW_IQR" or actor_sustained == "BELOW_IQR")
        if below:
            intent_below_ids.append(episode_id)
            pipeline = diagnose_failure(metrics, references, passed=False)["evidence_flags"]["pipeline_radial_attenuation"]
            if pipeline is None:
                pipeline_every_intent_below = False
            elif pipeline:
                pipeline_active_count += 1
        elif not assessable:
            intent_count_known = False
        statuses[episode_id] = {
            "actor_persistence_assessable": assessable,
            "actor_goalward_fraction_comparison": actor_fraction,
            "actor_sustained_window_comparison": actor_sustained,
            "intent_below_reference": bool(below) if assessable else None,
        }
    if actor_all_assessable:
        intent_below_count: int | None = len(intent_below_ids)
        pipeline_assessable = pipeline_every_intent_below
        active_count: int | None = pipeline_active_count
        authorized = len(intent_below_ids) >= 4 and pipeline_every_intent_below and pipeline_active_count < len(intent_below_ids)
    else:
        intent_below_count = None
        pipeline_assessable = None if not intent_below_ids else False
        active_count = None
        authorized = None
    suggestion = (
        "PREPARE_AUTHORIZED_MINIMAL_RADIAL_INTENT_PERSISTENCE_OFF_ON_CAUSAL_DIAGNOSTIC"
        if authorized is True else "NO_INTERVENTION_CONTINUE_OFFLINE_MECHANISM_AUDIT_ONLY"
    )
    return {
        "per_failure_actor_persistence": statuses,
        "all_seven_actor_persistence_comparisons_assessable": actor_all_assessable,
        "intent_below_reference_count": intent_below_count,
        "intent_below_reference_episode_ids": intent_below_ids if intent_count_known or actor_all_assessable else None,
        "pipeline_assessable_for_every_intent_below_episode": pipeline_assessable,
        "pipeline_active_count_among_intent_below": active_count,
        "authorized_minimal_radial_intent_persistence_diagnostic_preparation": authorized,
        "single_suggestion": suggestion,
        "intervention_implemented": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))


def _load_previous_module(path: Path):
    name = "_codex_previous_post_lateral_exit_pure_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous pure audit module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _frozen_file_map(source_root: Path, prior_root: Path, previous_module: Any) -> dict[str, Path]:
    files = {f"source/{name}": source_root / name for name in previous_module.SOURCE_FILES}
    checkpoint = Path(previous_module.CHECKPOINT_RELATIVE_SUFFIX)
    files[f"source/{checkpoint.as_posix()}"] = checkpoint
    files.update({f"prior/{name}": prior_root / name for name in previous_module.OUTPUT_FILES})
    return files


def _hash_frozen_files(paths: Mapping[str, Path]) -> dict[str, str]:
    absent = [name for name, path in paths.items() if not path.is_file()]
    if absent:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: frozen input missing: " + ", ".join(absent))
    return {name: sha256_file(path) for name, path in paths.items()}


def _assert_exact_prior_outputs(prior_root: Path, previous_module: Any) -> None:
    expected = set(previous_module.OUTPUT_FILES)
    actual = {path.name for path in prior_root.iterdir()} if prior_root.is_dir() else set()
    if actual != expected or any(not (prior_root / name).is_file() for name in expected):
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous audit output contract is not exactly eight files")


def _validate_prior_and_sources(
    source_root: Path,
    prior_root: Path,
    previous_module: Any,
    frozen_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any], list[dict[str, str]]]:
    previous_summary = _read_json(prior_root / "post_lateral_exit_audit_summary.json")
    if previous_summary.get("status") != EXPECTED_PREVIOUS_STATUS:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous summary status is not PASS")
    identity = previous_summary.get("source_identity", {})
    if identity.get("active_preregistration", {}).get("computed_active_hash") != EXPECTED_PREREGISTRATION_HASH:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: active preregistration hash mismatch")
    if tuple(previous_summary.get("success_reference_episode_ids", ())) != SUCCESS_IDS or tuple(previous_summary.get("main_failure_episode_ids", ())) != FAILURE_IDS:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous cohort identity mismatch")

    expected_source_hashes: dict[str, str] = {}
    for name in previous_module.SOURCE_FILES:
        expected_source_hashes[name] = frozen_hashes[f"source/{name}"]
    checkpoint = Path(previous_module.CHECKPOINT_RELATIVE_SUFFIX).resolve()
    expected_source_hashes[str(checkpoint)] = frozen_hashes[f"source/{Path(previous_module.CHECKPOINT_RELATIVE_SUFFIX).as_posix()}"]
    declared = previous_summary.get("source_hashes", {})
    if set(declared) != set(expected_source_hashes):
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous summary source hash key set mismatch")
    for name, digest in expected_source_hashes.items():
        record = declared.get(name, {})
        if record.get("sha256_before") != digest or record.get("sha256_after") != digest or record.get("unchanged") is not True:
            raise AuditError(f"PREPASS_REFERENCE_IDENTITY_FAIL: previous source hash mismatch: {name}")

    paths = {name: source_root / name for name in previous_module.SOURCE_FILES}
    prereg = previous_module.read_json(paths["preregistered_endpoints.json"])
    paired = previous_module.read_json(paths["lateral_exit_braking_assistance_summary.json"])
    off_summary = previous_module.read_json(paths["off/summary.json"])
    on_summary = previous_module.read_json(paths["on/summary.json"])
    off_episodes = previous_module.read_csv(paths["off/episodes.csv"], max_rows=25)
    on_episodes = previous_module.read_csv(paths["on/episodes.csv"], max_rows=25)
    off_steps = previous_module.read_csv(paths["off/steps.csv"], max_rows=2500)
    on_steps = previous_module.read_csv(paths["on/steps.csv"], max_rows=2500)
    pair_rows = previous_module.read_csv(paths["lateral_exit_braking_assistance_pairs.csv"], max_rows=25)
    checkpoint_stat = checkpoint.stat()
    source_identity = previous_module.validate_source_identity(
        prereg, paired, off_summary, on_summary, off_episodes, on_episodes,
        off_steps, on_steps, pair_rows,
        checkpoint_sha256=frozen_hashes[f"source/{Path(previous_module.CHECKPOINT_RELATIVE_SUFFIX).as_posix()}"],
        checkpoint_size_bytes=checkpoint_stat.st_size,
    )
    if source_identity.get("active_preregistration", {}).get("computed_active_hash") != EXPECTED_PREREGISTRATION_HASH:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: validated preregistration hash mismatch")

    previous_diagnostics = previous_module.read_csv(prior_root / "post_lateral_exit_episode_diagnostics.csv", max_rows=25)
    if [str(row.get("episode_id", "")) for row in previous_diagnostics] != list(ALL_IDS):
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: previous diagnostics are not exact ordered 20 episodes")
    if [str(row.get("episode_id", "")) for row in off_episodes] != list(ALL_IDS) or [str(row.get("episode_id", "")) for row in on_episodes] != list(ALL_IDS):
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: OFF/ON identity order mismatch")
    pair_by_id = {str(row.get("episode_id", "")): row for row in pair_rows}
    for index, row in enumerate(previous_diagnostics, start=1):
        episode_id = ALL_IDS[index - 1]
        off_row, on_row, pair = off_episodes[index - 1], on_episodes[index - 1], pair_by_id[episode_id]
        if not (
            integer(row.get("episode_index")) == index
            and str(row.get("scenario_seed")) == str(pair.get("scenario_seed"))
            and str(row.get("noise_seed")) == str(pair.get("noise_seed"))
            and str(row.get("scenario_hash")) == str(off_row.get("scenario_hash")) == str(on_row.get("scenario_hash"))
            and integer(off_row.get("episode_index")) == integer(on_row.get("episode_index")) == index
        ):
            raise AuditError(f"PREPASS_REFERENCE_IDENTITY_FAIL: prior OFF/ON ordered identity mismatch: {episode_id}")
    return previous_summary, off_episodes, on_episodes, on_steps, source_identity, previous_diagnostics


def _metric_flatten(metrics: Mapping[str, Mapping[str, Any]], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in METRIC_NAMES:
        record = metrics.get(name, null_metric("METRIC_NOT_COMPUTABLE"))
        stem = f"{prefix}{name}"
        result[stem] = record.get("value")
        result[f"{stem}_sample_count"] = record.get("sample_count")
        result[f"{stem}_unavailable_reason"] = record.get("unavailable_reason")
        if "numerator" in record or "denominator" in record:
            result[f"{stem}_numerator"] = record.get("numerator")
            result[f"{stem}_denominator"] = record.get("denominator")
    return result


def goal_recovery_activity_counts(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    successes = [row for row in diagnostics if row.get("cohort") == "SUCCESS_REFERENCE"]
    before_counts = [int(row.get("goal_recovery_active_before_pass_row_count") or 0) for row in successes]
    active_transition_count = sum(row.get("transition_goal_recovery_active") is True for row in successes)
    return {
        "active_before_transition_row_count": sum(before_counts),
        "active_before_transition_episode_count": sum(value > 0 for value in before_counts),
        "active_on_transition_row_count": active_transition_count,
        "active_on_transition_episode_count": active_transition_count,
    }


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        clean = {key: ("" if value is None else value) for key, value in row.items()}
        writer.writerow(clean)
    return output.getvalue()


def _build_diagnosis_comparison_rows(
    failure_segments: Mapping[str, Mapping[str, Mapping[str, Any]]],
    references: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id in FAILURE_IDS:
        for window_name in ("WHOLE", "PRE_BEST", "POST_BEST"):
            segment = failure_segments[episode_id][window_name]
            metrics = segment["metrics"]
            for metric_name in COMPARISON_METRICS:
                value = _metric_value(metrics, metric_name)
                reference = references[metric_name]
                category = compare_iqr(value, reference)
                if value is None:
                    reason = metrics[metric_name].get("unavailable_reason") or "FAILURE_METRIC_UNAVAILABLE"
                elif reference.get("n", 0) <= 0:
                    reason = reference.get("unavailable_reason") or "REFERENCE_METRIC_UNAVAILABLE"
                else:
                    reason = None
                rows.append({
                    "episode_id": episode_id,
                    "window": window_name,
                    "metric": metric_name,
                    "value": value,
                    "sample_count": metrics[metric_name].get("sample_count"),
                    "comparison": category,
                    "comparison_unavailable_reason": reason,
                    "reference_n": reference.get("n"),
                    "reference_median": reference.get("median"),
                    "reference_q1": reference.get("q1"),
                    "reference_q3": reference.get("q3"),
                    "reference_iqr": reference.get("iqr"),
                })
    return rows


def _build_segmentation_rows(failure_segments: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id in FAILURE_IDS:
        for window_name in ("WHOLE", "PRE_BEST", "POST_BEST"):
            segment = failure_segments[episode_id][window_name]
            segment_rows = segment["rows"]
            observed_steps = [integer(row.get("step")) for row in segment_rows]
            expected_count = 0 if not segment_rows else len(segment_rows)
            valid_count = sum(step is not None for step in observed_steps)
            metrics = compute_metrics(segment_rows, initial_lateral_sign=failure_segments[episode_id]["initial_lateral_sign"])
            row = {
                "episode_id": episode_id,
                "segment": window_name,
                "start_step": segment.get("start_step"),
                "end_step": segment.get("end_step"),
                "start_inclusive": segment.get("start_inclusive"),
                "end_inclusive": segment.get("end_inclusive"),
                "observed_row_count": len(segment_rows),
                "expected_step_count": expected_count,
                "finite_step_count": valid_count,
                "coverage_fraction": valid_count / expected_count if expected_count else None,
                "structural_reason": segment.get("structural_reason"),
            }
            row.update(_metric_flatten(metrics))
            rows.append(row)
    return rows


def _build_success_reference_rows(references: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"reference": "PRE_PASS_SUCCESS_REFERENCE", "metric": name, **references[name]} for name in REFERENCE_METRICS]


def _build_report(summary: Mapping[str, Any]) -> str:
    diagnosis_counts = summary["diagnosis_counts"]
    changes = summary["evidence_change_counts"]
    decision = summary["recommendation_decision"]
    failures = summary["failure_diagnoses"]
    lines = [
        "# Pre-pass reorientation reference audit",
        "",
        "This is a pure offline audit of frozen ON trajectories. It uses no post-pass rows, performs no rollout, and does not implement an intervention.",
        "",
        "## Frozen identity and windows",
        "",
        f"Active preregistration SHA-256: `{summary['active_preregistration_hash']}`. Previous audit status: `{summary['previous_audit_status']}`. The previous post-exit reference is `{summary['old_reference_status']}` and is excluded from corrected diagnoses.",
        "",
        f"Previous fail-fast reason: `{summary['previous_fail_fast_reason']}`.",
        "",
        "Local exit is the first start of three consecutive frozen steps with absolute actual lateral velocity at most 0.5 m/s, searched from the previous validated anchor. Each fixed ON success contributes the strict pre-pass interval `[local_exit_step, first_pass_step)`; the exact pass-transition row is stored separately and excluded from reference metrics. The seven fixed main failures use the entire local-exit-to-end window. Every metric uses finite samples only; unavailable values are null with a reason and sample count.",
        "",
        "The exact pass gate reuses the frozen `passed_obstacle` release latch together with `passed_margin_m > 0` and same-step relevant obstacle clearance at least 2.5 m. Type-7 linear quartiles are computed only from the 11 corrected pre-pass success windows. Failure segments are whole, `[local_exit, earliest_best_margin]`, and `(earliest_best_margin, end]`.",
        "",
        "## Q1. Corrected pre-pass success reference",
        "",
        f"The corrected reference is labeled `PRE_PASS_SUCCESS_REFERENCE`. Its pre-pass endpoint is `first_pass_step - 1`, and `pass_transition_row_excluded_from_reference=true`. Success transitions with Goal Recovery active: {summary['goal_recovery_activity_counts']['active_on_transition_episode_count']}; active rows before transition: {summary['goal_recovery_activity_counts']['active_before_transition_row_count']}.",
        f"Success episodes without an assessable pre-pass window are retained as `NOT_ASSESSABLE/NO_PREPASS_REFERENCE_WINDOW`: {', '.join(summary['prepass_reference_unavailable_episode_ids']) or 'none'}.",
        "",
        "## Q2. Main failure diagnoses",
        "",
        "| Episode | Corrected primary diagnosis | Core assessable |",
        "|---|---|---:|",
    ]
    for episode_id in FAILURE_IDS:
        item = failures[episode_id]
        lines.append(f"| {episode_id[-4:]} | {item['primary_diagnosis']} | {item['core_assessable']} |")
    lines += [
        "",
        "Primary diagnosis counts: " + ", ".join(f"{name}={count}" for name, count in sorted(diagnosis_counts.items())),
        "",
        "## Q3. Before/after-best segmentation",
        "",
        "`prepass_failure_segmentation.csv` contains 21 rows: whole, pre-best, and post-best for each main failure. The pre and post windows are disjoint and cover the failure window exactly; an empty post segment is preserved as a structural row.",
        "",
        "## Q4. Prior evidence flags",
        "",
        "Evidence is compared flag by flag against the previous episode diagnostics. Counts: " + ", ".join(f"{name}={count}" for name, count in sorted(changes.items())),
        "",
        "## Q5. One next-step suggestion",
        "",
        f"All seven actor-persistence comparisons assessable: `{decision['all_seven_actor_persistence_comparisons_assessable']}`; intent-below count: `{decision['intent_below_reference_count']}`; pipeline assessable for every intent-below episode: `{decision['pipeline_assessable_for_every_intent_below_episode']}`; pipeline-active count: `{decision['pipeline_active_count_among_intent_below']}`.",
        "",
        f"Suggestion: `{decision['single_suggestion']}`. Intervention implemented: `false`.",
        "",
        "## Q6. Special cases and limitations",
        "",
        "Episode 0008 remains in the success reference with case annotation `DOWNSTREAM_ADVERSE_SPECIAL_CASE` and primary `NOT_ASSESSABLE`. Episodes 0013 and 0014 remain excluded from primary diagnosis as safety-dominated and boundary-dominated special cases. The seven fixed main failures and 11 fixed successes are not reselected from outcomes.",
        "",
        "This is a descriptive comparison against a fixed small success cohort. IQR labels use inclusive Q1/Q3 bounds; they are not population uncertainty intervals and do not establish causality. The prior diagnosis system is preserved only for evidence-change accounting.",
        "",
        f"Exactly six files are emitted under `{OUTPUT_ROOT.as_posix()}`.",
        "",
    ]
    return "\n".join(lines)


def _build_summary(
    *,
    identity: Mapping[str, Any],
    frozen_hashes: Mapping[str, str],
    previous_summary: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    failure_diagnoses: Mapping[str, Mapping[str, Any]],
    evidence_change_counts: Mapping[str, int],
    decision: Mapping[str, Any],
    contamination: Mapping[str, Any],
    segmentation_count: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in diagnostics:
        diagnosis = str(row.get("primary_diagnosis") or "NOT_ASSESSABLE")
        counts[diagnosis] = counts.get(diagnosis, 0) + 1
    activity_counts = goal_recovery_activity_counts(diagnostics)
    success_diagnostics = [row for row in diagnostics if row.get("cohort") == "SUCCESS_REFERENCE"]
    transition_records = [
        {
            "episode_id": row.get("episode_id"),
            "first_pass_step": row.get("first_pass_step"),
            "transition_passed_obstacle": row.get("transition_passed_obstacle"),
            "transition_post_bypass_recovery": row.get("transition_post_bypass_recovery"),
            "transition_goal_recovery_eligible": row.get("transition_goal_recovery_eligible"),
            "transition_goal_recovery_active": row.get("transition_goal_recovery_active"),
            "transition_goal_recovery_reason": row.get("transition_goal_recovery_reason"),
            "transition_passed_margin": row.get("transition_passed_margin"),
            "transition_clearance": row.get("transition_clearance"),
        }
        for row in success_diagnostics
    ]
    previous_contract = previous_summary.get("output_contract", [])
    previous_files = previous_contract.get("files", []) if isinstance(previous_contract, Mapping) else previous_contract
    if not isinstance(previous_files, Sequence) or isinstance(previous_files, (str, bytes)):
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: malformed previous output contract")
    return {
        "audit": "single_uav_prepass_reorientation_reference_audit",
        "status": "PASS",
        "active_preregistration_hash": EXPECTED_PREREGISTRATION_HASH,
        "previous_audit_status": previous_summary.get("status"),
        "previous_source_identity": identity,
        "frozen_input_hashes_before_and_after": {name: {"sha256_before": digest, "sha256_after": digest, "unchanged": True} for name, digest in frozen_hashes.items()},
        "previous_output_contract": {"exact_file_count": 8, "files": list(sorted(previous_files))},
        "output_contract": {"exact_file_count": 6, "files": list(OUTPUT_FILES)},
        "cohorts": {"success_reference_episode_ids": list(SUCCESS_IDS), "main_failure_episode_ids": list(FAILURE_IDS), "special_excluded_episode_ids": list(SPECIAL_IDS)},
        "window_definitions": {
            "local_exit": "first start of three consecutive frozen steps with abs(actual lateral velocity)<=0.5 m/s, searched from max(encounter step, prior validated OFF-defined anchor when present)",
            "success": "strict pre-pass [local_exit_step, first_pass_step); pass-transition row is separately recorded and excluded",
            "prepass_window_end": "first_pass_step_minus_1",
            "pass_transition_row_excluded_from_reference": True,
            "failure_whole": "inclusive [local_exit_step, final logged step]",
            "failure_pre_best": "inclusive [local_exit_step, earliest step attaining maximum finite pass margin]",
            "failure_post_best": "exclusive at best-margin step, inclusive through final logged step; empty structural row retained",
            "exact_pass_gate": "frozen passed_obstacle release latch is true, passed_margin_m>0, and same-step relevant obstacle clearance>=2.5 m",
        },
        "old_reference_status": "SUPERSEDED_FOR_UPSTREAM_DIAGNOSIS",
        "old_reference_use": "EVIDENCE_CHANGE_ACCOUNTING_ONLY",
        "previous_fail_fast_reason": PREVIOUS_FAIL_FAST_REASON,
        "success_reference_name": "PRE_PASS_SUCCESS_REFERENCE",
        "pass_transition_row_excluded_from_reference": True,
        "prepass_window_end": "first_pass_step_minus_1",
        "prepass_reference_unavailable_episode_ids": [
            row.get("episode_id") for row in diagnostics
            if row.get("cohort") == "SUCCESS_REFERENCE" and row.get("window_unavailable_reason") == "NO_PREPASS_REFERENCE_WINDOW"
        ],
        "success_pass_transition_rows": transition_records,
        "goal_recovery_activity_counts": activity_counts,
        "success_reference_quantiles": {"method": "numpy_linear_type7", "eligible_episode_count": len(SUCCESS_IDS), "statistics": dict(references)},
        "contamination_check": dict(contamination),
        "diagnosis_counts": counts,
        "failure_diagnoses": dict(failure_diagnoses),
        "evidence_change_statuses": [row.get("evidence_changes", {}) for row in diagnostics],
        "evidence_change_counts": dict(evidence_change_counts),
        "segmentation": {"row_count": segmentation_count, "window_order": ["WHOLE", "PRE_BEST", "POST_BEST"], "pre_post_disjoint_exact_union": True},
        "recommendation_decision": dict(decision),
        "episode_diagnostic_count": len(diagnostics),
        "limitations": [
            "The fixed 11-episode type-7 IQR is descriptive and not a population uncertainty interval.",
            "No post-pass sample contributes to success reference metrics.",
            "Episode 0008 is a downstream adverse special case and remains in the fixed success reference.",
            "Episodes 0013 and 0014 are special cases excluded from primary diagnosis.",
            "The recommended causal diagnostic is not implemented by this audit.",
        ],
    }


def run_audit(
    source_root: Path = SOURCE_ROOT,
    prior_root: Path = PRIOR_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root, prior_root, output_root = Path(source_root), Path(prior_root), Path(output_root)
    project_root = Path(__file__).resolve().parents[2]
    expected_source = (project_root / SOURCE_ROOT).resolve()
    expected_prior = (project_root / PRIOR_ROOT).resolve()
    expected_output = (project_root / OUTPUT_ROOT).resolve()
    if source_root.resolve() != expected_source or prior_root.resolve() != expected_prior or output_root.resolve() != expected_output:
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: audit roots are fixed and read-only except the declared output root")
    if output_root.exists():
        raise AuditError(f"PREPASS_REFERENCE_OUTPUT_EXISTS: refusing overwrite: {output_root}")
    previous_module = _load_previous_module(PRIOR_MODULE_PATH)
    _assert_exact_prior_outputs(prior_root, previous_module)

    # Freeze every declared source, the checkpoint, and all eight prior outputs
    # before parsing or interpreting any artifact.
    frozen_paths = _frozen_file_map(source_root, prior_root, previous_module)
    hashes_before = _hash_frozen_files(frozen_paths)
    previous_summary, off_episodes, on_episodes, on_steps, source_identity, previous_diagnostics = _validate_prior_and_sources(
        source_root, prior_root, previous_module, hashes_before,
    )
    off_steps = previous_module.read_csv(source_root / "off/steps.csv", max_rows=2500)
    pair_rows = previous_module.read_csv(source_root / "lateral_exit_braking_assistance_pairs.csv", max_rows=25)
    on_grouped = previous_module.validate_step_continuity(on_steps, condition="on")
    off_grouped = previous_module.validate_step_continuity(off_steps, condition="off")
    frame_checks = {
        "OFF": previous_module.validate_frame_and_decomposition(off_grouped, off_episodes),
        "ON": previous_module.validate_frame_and_decomposition(on_grouped, on_episodes),
    }
    pass_checks = {
        "OFF": previous_module.validate_pass_latches(off_grouped, off_episodes),
        "ON": previous_module.validate_pass_latches(on_grouped, on_episodes),
    }
    if frame_checks["ON"].get("status") != "PASS" or pass_checks["ON"].get("status") != "PASS":
        raise AuditError("PREPASS_REFERENCE_IDENTITY_FAIL: prior frame/pass checks failed")

    prior_by_id = {str(row["episode_id"]): row for row in previous_diagnostics}
    pair_by_id = {str(row["episode_id"]): row for row in pair_rows}
    on_episode_by_id = {str(row["episode_id"]): row for row in on_episodes}
    local_exit_by_id: dict[str, int | None] = {}
    initial_sign_by_id: dict[str, int] = {}
    search_anchor_by_id: dict[str, int | None] = {}
    for episode_id in ALL_IDS:
        prior_row = prior_by_id[episode_id]
        encounter = integer(prior_row.get("encounter_step"))
        off_anchor = integer(prior_row.get("off_defined_anchor_step"))
        if encounter is None:
            raise AuditError(f"PREPASS_REFERENCE_IDENTITY_FAIL: missing prior encounter anchor: {episode_id}")
        search_anchor = max(encounter, off_anchor) if off_anchor is not None else encounter
        search_anchor_by_id[episode_id] = search_anchor
        rows = on_grouped[episode_id]
        exit_step = first_local_exit_step(rows, search_anchor)
        expected_exit = integer(prior_row.get("on_local_exit_step"))
        if exit_step != expected_exit:
            raise AuditError(f"PREPASS_REFERENCE_IDENTITY_FAIL: prior local-exit anchor validation differs: {episode_id}")
        local_exit_by_id[episode_id] = exit_step
        initial_sign_by_id[episode_id] = integer(prior_row.get("on_metrics_initial_encounter_actual_lateral_sign")) or 0

    success_windows: dict[str, list[Mapping[str, Any]]] = {}
    pass_step_by_id: dict[str, int | None] = {}
    transition_by_id: dict[str, Mapping[str, Any] | None] = {}
    success_window_reason_by_id: dict[str, str | None] = {}
    for episode_id in SUCCESS_IDS:
        window, first_pass, transition, unavailable_reason = success_prepass_window(
            on_grouped[episode_id], local_exit_by_id[episode_id],
        )
        if first_pass is None:
            raise AuditError(f"PREPASS_REFERENCE_IDENTITY_FAIL: fixed success lacks exact pass gate: {episode_id}")
        success_windows[episode_id] = window
        pass_step_by_id[episode_id] = first_pass
        transition_by_id[episode_id] = transition
        success_window_reason_by_id[episode_id] = unavailable_reason

    # This check completes for all 11 windows before metrics, tables, or output
    # staging can be constructed.
    validate_success_window_contamination(success_windows, local_exit_by_id, pass_step_by_id)

    success_metrics = {
        episode_id: (
            {name: null_metric(success_window_reason_by_id[episode_id]) for name in METRIC_NAMES}
            if success_window_reason_by_id[episode_id]
            else compute_metrics(success_windows[episode_id], initial_lateral_sign=initial_sign_by_id[episode_id])
        )
        for episode_id in SUCCESS_IDS
    }
    references = build_success_references(success_metrics)

    failure_segments: dict[str, dict[str, Any]] = {}
    failure_metrics: dict[str, dict[str, Any]] = {}
    failure_diagnoses: dict[str, dict[str, Any]] = {}
    for episode_id in FAILURE_IDS:
        segments = make_segments(on_grouped[episode_id], local_exit_by_id[episode_id])
        for segment in segments.values():
            segment["metrics"] = compute_metrics(segment["rows"], initial_lateral_sign=initial_sign_by_id[episode_id])
        segments["initial_lateral_sign"] = initial_sign_by_id[episode_id]
        failure_segments[episode_id] = segments
        failure_metrics[episode_id] = segments["WHOLE"]["metrics"]
        failure_diagnoses[episode_id] = diagnose_failure(failure_metrics[episode_id], references, passed=False)

    old_by_id = prior_by_id
    diagnostics: list[dict[str, Any]] = []
    evidence_change_counts: dict[str, int] = {}
    for episode_id in ALL_IDS:
        prior_row = old_by_id[episode_id]
        episode = on_episode_by_id[episode_id]
        special = SPECIAL_IDS.get(episode_id)
        annotation = "DOWNSTREAM_ADVERSE_SPECIAL_CASE" if episode_id == "test_r99_episode_0008" else None
        if episode_id in SUCCESS_IDS:
            metrics = success_metrics[episode_id]
            diagnosis = {"primary_diagnosis": "NOT_ASSESSABLE", "evidence_flags": {}}
            window_kind = "SUCCESS_PRE_PASS"
            local_window = success_windows[episode_id]
            transition = transition_by_id[episode_id]
            active_before_count = sum(truthy(item.get("goal_recovery_correction_active")) for item in local_window)
            transition_values = pass_transition_metadata(transition)
        elif episode_id in FAILURE_IDS:
            metrics = failure_metrics[episode_id]
            diagnosis = failure_diagnoses[episode_id]
            window_kind = "FAILURE_LOCAL_EXIT_TO_END"
            local_window = failure_segments[episode_id]["WHOLE"]["rows"]
            transition = None
            active_before_count = None
            transition_values = {
                "transition_passed_obstacle": None,
                "transition_post_bypass_recovery": None,
                "transition_goal_recovery_eligible": None,
                "transition_goal_recovery_active": None,
                "transition_goal_recovery_reason": None,
                "transition_passed_margin": None,
                "transition_clearance": None,
            }
        else:
            metrics = {name: null_metric("SPECIAL_CASE_EXCLUDED_FROM_PRIMARY_DIAGNOSIS") for name in METRIC_NAMES}
            diagnosis = {"primary_diagnosis": "NOT_ASSESSABLE", "evidence_flags": {}}
            window_kind = "SPECIAL_CASE_EXCLUDED"
            local_window = []
            transition = None
            active_before_count = None
            transition_values = {
                "transition_passed_obstacle": None,
                "transition_post_bypass_recovery": None,
                "transition_goal_recovery_eligible": None,
                "transition_goal_recovery_active": None,
                "transition_goal_recovery_reason": None,
                "transition_passed_margin": None,
                "transition_clearance": None,
            }
        old_flags: dict[str, Any] = {}
        corrected_flags: dict[str, Any] = {}
        changes: dict[str, str] = {}
        for flag_name, _ in DIAGNOSIS_BY_FLAG:
            old_raw = prior_row.get(f"evidence_{flag_name}")
            old_bool = None if old_raw in (None, "") else truthy(old_raw)
            corrected = diagnosis.get("evidence_flags", {}).get(flag_name)
            old_flags[flag_name] = old_bool
            corrected_flags[flag_name] = corrected
            status = evidence_change_status(old_bool, corrected)
            changes[flag_name] = status
            evidence_change_counts[status] = evidence_change_counts.get(status, 0) + 1
        record = {
            "episode_index": integer(episode.get("episode_index")),
            "episode_id": episode_id,
            "scenario_seed": integer(episode.get("scenario_seed")),
            "noise_seed": integer(episode.get("noise_seed")),
            "scenario_hash": episode.get("scenario_hash"),
            "cohort": "SUCCESS_REFERENCE" if episode_id in SUCCESS_IDS else "MAIN_FAILURE" if episode_id in FAILURE_IDS else "SPECIAL_EXCLUDED",
            "window_kind": window_kind,
            "search_anchor_step": search_anchor_by_id[episode_id],
            "local_exit_step": local_exit_by_id[episode_id],
            "first_pass_step": pass_step_by_id.get(episode_id),
            "first_exact_pass_step": pass_step_by_id.get(episode_id),
            "prepass_window_end_step": pass_step_by_id[episode_id] - 1 if episode_id in SUCCESS_IDS and pass_step_by_id[episode_id] is not None else None,
            "window_start_step": integer(local_window[0].get("step")) if local_window else None,
            "window_end_step": integer(local_window[-1].get("step")) if local_window else None,
            "window_row_count": len(local_window),
            "window_unavailable_reason": success_window_reason_by_id.get(episode_id) if episode_id in SUCCESS_IDS else None,
            "diagnosis_unavailable_reason": success_window_reason_by_id.get(episode_id) if episode_id in SUCCESS_IDS else None,
            "goal_recovery_active_before_pass_row_count": active_before_count,
            "passed_obstacle": truthy(episode.get("passed_obstacle")),
            "primary_diagnosis": special and "NOT_ASSESSABLE" or diagnosis["primary_diagnosis"],
            "special_case_diagnosis": special,
            "case_annotation": annotation,
            "old_evidence_flags": old_flags,
            "corrected_evidence_flags": corrected_flags,
            "evidence_changes": changes,
            "metrics": metrics,
            "evidence_comparisons": diagnosis.get("evidence_comparisons", {}),
        }
        record.update(transition_values)
        diagnostics.append(record)

    decision = recommendation_decision(failure_metrics, references)
    segmentation_rows = _build_segmentation_rows(failure_segments)
    comparison_rows = _build_diagnosis_comparison_rows(failure_segments, references)
    reference_rows = _build_success_reference_rows(references)
    episode_csv_rows: list[dict[str, Any]] = []
    for row in diagnostics:
        flat = {key: value for key, value in row.items() if key not in {"old_evidence_flags", "corrected_evidence_flags", "evidence_changes", "metrics", "evidence_comparisons"}}
        flat.update({f"old_evidence_{key}": value for key, value in row["old_evidence_flags"].items()})
        flat.update({f"corrected_evidence_{key}": value for key, value in row["corrected_evidence_flags"].items()})
        flat.update({f"evidence_change_{key}": value for key, value in row["evidence_changes"].items()})
        flat.update(_metric_flatten(row["metrics"]))
        for key, value in row["evidence_comparisons"].items():
            flat[f"comparison_{key}"] = value
        episode_csv_rows.append(flat)

    contamination = {
        "status": "PASS",
        "checked_success_episode_count": len(SUCCESS_IDS),
        "checked_success_window_row_count": sum(len(rows) for rows in success_windows.values()),
        "controller_contamination_rows": 0,
        "fields": ["goal_recovery_correction_active", "post_bypass_recovery", "passed_obstacle"],
        "window_stop": "first_pass_step_exclusive",
        "transition_rows_excluded": sum(transition_by_id[episode_id] is not None for episode_id in SUCCESS_IDS),
    }
    summary = _build_summary(
        identity={"active_preregistration": source_identity.get("active_preregistration"), "ordered_episode_count": source_identity.get("ordered_episode_count"), "off_fingerprint": source_identity.get("off_fingerprint"), "frame_checks": frame_checks, "pass_checks": pass_checks},
        frozen_hashes=hashes_before,
        previous_summary=previous_summary,
        references=references,
        diagnostics=diagnostics,
        failure_diagnoses=failure_diagnoses,
        evidence_change_counts=evidence_change_counts,
        decision=decision,
        contamination=contamination,
        segmentation_count=len(segmentation_rows),
    )
    report = _build_report(summary)
    success_csv = _csv_text(reference_rows, ("reference", "metric", "eligible_episode_count", "n", "median", "q1", "q3", "iqr", "minimum", "maximum", "quantile_method", "unavailable_reason"))
    comparison_csv = _csv_text(comparison_rows, ("episode_id", "window", "metric", "value", "sample_count", "comparison", "comparison_unavailable_reason", "reference_n", "reference_median", "reference_q1", "reference_q3", "reference_iqr"))
    segmentation_fields = list(segmentation_rows[0].keys()) if segmentation_rows else ["episode_id", "segment"]
    segmentation_csv = _csv_text(segmentation_rows, segmentation_fields)
    episode_fields = (
        list(dict.fromkeys(key for row in episode_csv_rows for key in row))
        if episode_csv_rows else ["episode_id"]
    )
    episode_csv = _csv_text(episode_csv_rows, episode_fields)
    summary_text = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    contents = {
        OUTPUT_FILES[0]: summary_text,
        OUTPUT_FILES[1]: success_csv,
        OUTPUT_FILES[2]: comparison_csv,
        OUTPUT_FILES[3]: segmentation_csv,
        OUTPUT_FILES[4]: episode_csv,
        OUTPUT_FILES[5]: report,
    }
    if tuple(contents) != OUTPUT_FILES:
        raise AuditError("PREPASS_REFERENCE_OUTPUT_CONTRACT_FAIL: file order mismatch")

    # Hash again after all computations and serialization, before any write.
    hashes_after = _hash_frozen_files(frozen_paths)
    if dict(hashes_before) != hashes_after:
        changed = sorted(name for name in set(hashes_before) | set(hashes_after) if hashes_before.get(name) != hashes_after.get(name))
        raise AuditError("PREPASS_REFERENCE_SOURCE_CHANGED_DURING_AUDIT: " + ", ".join(changed))
    if output_root.exists():
        raise AuditError(f"PREPASS_REFERENCE_OUTPUT_EXISTS: refusing overwrite: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".prepass_reorientation_staging_", dir=output_root.parent))
    bundle = stage_parent / "bundle"
    committed = False
    try:
        bundle.mkdir()
        for name in OUTPUT_FILES:
            (bundle / name).write_text(contents[name], encoding="utf-8", newline="\n")
        actual = {path.name for path in bundle.iterdir() if path.is_file()}
        if actual != set(OUTPUT_FILES) or len(actual) != 6:
            raise AuditError("PREPASS_REFERENCE_OUTPUT_CONTRACT_FAIL: staged output must contain exactly six files")
        if output_root.exists():
            raise AuditError(f"PREPASS_REFERENCE_OUTPUT_EXISTS: refusing overwrite: {output_root}")
        os.replace(bundle, output_root)
        committed = True
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "output_files": list(OUTPUT_FILES),
        "frozen_input_hashes_unchanged": True,
        "frozen_input_count": len(hashes_before),
        "episode_diagnostic_count": len(diagnostics),
        "failure_diagnoses": {episode_id: failure_diagnoses[episode_id]["primary_diagnosis"] for episode_id in FAILURE_IDS},
        "recommendation": decision["single_suggestion"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.source_root, args.prior_root, args.output_root)
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
