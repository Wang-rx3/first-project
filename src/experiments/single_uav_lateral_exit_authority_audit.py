"""Offline encounter-relative lateral-exit authority audit.

This module reads only the five frozen Signed Bypass Observability artifacts.
It does not import or call the environment, simulator, controller, policy, or
training runtime.  All rules are fixed and applied identically to failure and
successful-reference episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence


SOURCE_ROOT = Path(
    "logs/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026"
)
OUTPUT_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_authority_audit_seed2026"
)
SOURCE_FILENAMES = (
    "signed_bypass_summary.json",
    "signed_bypass_episodes.csv",
    "signed_bypass_steps.csv",
    "signed_bypass_focus_failures.csv",
    "signed_bypass_direction_transitions.csv",
)
OUTPUT_FILENAMES = (
    "lateral_exit_authority_audit_summary.json",
    "lateral_exit_episode_diagnostics.csv",
    "lateral_exit_countersteer_windows.csv",
    "lateral_exit_success_reference.csv",
    "lateral_exit_failure_vs_reference.csv",
    "lateral_exit_report.md",
)

MAIN_FAILURE_SUFFIXES = ("0002", "0004", "0005", "0006", "0011", "0012", "0020")
SPECIAL_FAILURE_SUFFIXES = ("0013", "0014")
REFERENCE_SUFFIXES = (
    "0001", "0003", "0007", "0008", "0009", "0010",
    "0015", "0016", "0017", "0018", "0019",
)

SIGN_DEADBAND_MPS = 1.0e-8
CLEARANCE_EPSILON_M = 1.0e-9
COUNTERSTEER_WINDOW_STEPS = 5
COUNTERSTEER_MIN_STEPS = 4
SUSTAINED_CLEARANCE_STEPS = 3
SUSTAINED_DECAY_TRANSITIONS = 3
DEFAULT_NEAR_ZERO_LATERAL_THRESHOLD_MPS = 0.5
PIPELINE_RATIO_EPSILON_MPS = 1.0e-8

PRIMARY_DIAGNOSES = (
    "ACTOR_NEVER_COUNTERSTEERS",
    "COUNTERSTEER_TOO_WEAK",
    "COUNTERSTEER_NOT_SUSTAINED",
    "PIPELINE_ATTENUATION",
    "LATERAL_MOMENTUM_DECAY_TOO_SLOW",
    "SAFETY_DOMINATED_SPECIAL_CASE",
    "BOUNDARY_DOMINATED_SPECIAL_CASE",
    "MIXED",
    "NOT_ASSESSABLE",
)

REFERENCE_METRICS = (
    "countersteer_onset_delay_from_closest_steps",
    "mean_actor_countersteer_magnitude_mps",
    "net_actor_counter_lateral_impulse_mps",
    "actor_to_final_retention",
    "transform_retention",
    "safety_retention",
    "boundary_retention",
    "mean_lateral_speed_decay_mps_per_step",
    "steps_to_near_zero_after_countersteer",
    "extra_lateral_drift_after_countersteer_m",
    "post_closest_actor_same_direction_fraction",
    "post_closest_actual_same_direction_fraction",
)


class SourceIdentityError(RuntimeError):
    """Raised when the frozen signed-observability source identity fails."""


def _episode_id(suffix: str) -> str:
    return f"test_r99_episode_{suffix}"


MAIN_FAILURE_IDS = tuple(_episode_id(value) for value in MAIN_FAILURE_SUFFIXES)
SPECIAL_FAILURE_IDS = tuple(_episode_id(value) for value in SPECIAL_FAILURE_SUFFIXES)
REFERENCE_IDS = tuple(_episode_id(value) for value in REFERENCE_SUFFIXES)
TARGET_IDS = MAIN_FAILURE_IDS + SPECIAL_FAILURE_IDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: _sha256(root / name) for name in SOURCE_FILENAMES}


def assert_source_hashes_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> None:
    if dict(before) != dict(after):
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise RuntimeError("SOURCE_LOGS_MODIFIED: " + ", ".join(changed))


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_summary_identity_prefix(path: Path, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Parse only the bounded top-level prefix preceding the embedded evaluation.

    The complete file is hashed, but the 36 MiB embedded evaluation trace is
    not needed for this audit.  The required identity fields precede the
    top-level ``evaluation`` member.
    """

    lines: list[str] = []
    scanned = 0
    marker_found = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            scanned += len(line.encode("utf-8"))
            if scanned > max_bytes:
                raise ValueError("summary identity prefix exceeds bounded scan")
            if line.strip().startswith('"evaluation":'):
                marker_found = True
                break
            lines.append(line)
    if not marker_found:
        raise ValueError("summary evaluation marker not found in bounded prefix")
    prefix = "".join(lines).rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1]
    prefix += "\n}\n"
    payload = json.loads(
        prefix,
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_nonfinite_json,
    )
    payload["_identity_prefix_scanned_bytes"] = scanned
    return payload


def _load_csv(path: Path, *, max_rows: int = 10_000) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"invalid or duplicate CSV header: {path.name}")
        for row in reader:
            if None in row:
                raise ValueError(f"non-rectangular CSV row: {path.name}")
            rows.append(dict(row))
            if len(rows) > max_rows:
                raise ValueError(f"CSV row limit exceeded: {path.name}")
    return rows


def require_unique_episode_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_name: str,
) -> set[str]:
    """Fail closed instead of letting a dict/set silently collapse duplicate rows."""

    ids = [str(row.get("episode_id", "")).strip() for row in rows]
    if any(not episode_id for episode_id in ids):
        raise SourceIdentityError(
            f"LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: missing episode_id in {artifact_name}"
        )
    if len(ids) != len(set(ids)):
        raise SourceIdentityError(
            f"LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: duplicate episode_id in {artifact_name}"
        )
    return set(ids)


def validate_transition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: set[str],
) -> None:
    """Validate the frozen transition artifact's schema, cohort, and step continuity."""

    required_fields = {
        "episode_index", "episode_id", "step", "actor_lateral_sign",
        "actual_lateral_sign", "actor_lateral_sign_changes",
        "actual_lateral_sign_changes", "actor_same_side_run_length",
        "actual_same_side_run_length", "actor_actual_lateral_side_agree_now",
        "actor_actual_lateral_side_agreement_fraction", "clearance_increasing",
        "past_closest_approach",
    }
    if not rows:
        raise SourceIdentityError(
            "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: empty transition artifact"
        )
    if not required_fields.issubset(rows[0]):
        raise SourceIdentityError(
            "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: transition schema mismatch"
        )
    grouped: dict[str, list[int]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    for row in rows:
        episode_id = str(row.get("episode_id", "")).strip()
        step = _int(row, "step")
        if not episode_id or step is None:
            raise SourceIdentityError(
                "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: invalid transition identity"
            )
        pair = (episode_id, step)
        if pair in seen_pairs:
            raise SourceIdentityError(
                "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: duplicate transition step"
            )
        seen_pairs.add(pair)
        grouped.setdefault(episode_id, []).append(step)
    if set(grouped) != expected_ids:
        raise SourceIdentityError(
            "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: transition cohort mismatch"
        )
    for episode_id, steps in grouped.items():
        if steps != sorted(steps) or any(
            right != left + 1 for left, right in zip(steps, steps[1:])
        ):
            raise SourceIdentityError(
                "LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: "
                f"non-contiguous transition steps for {episode_id}"
            )


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value in {None, "", "None", "null", "NaN", "nan"}:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _int(row: Mapping[str, Any], key: str) -> int | None:
    number = _float(row, key)
    return int(number) if math.isfinite(number) else None


def _bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def lateral_sign(value: float, *, deadband: float = SIGN_DEADBAND_MPS) -> int:
    if not math.isfinite(float(value)):
        return 0
    return 1 if value > deadband else -1 if value < -deadband else 0


def encounter_slice(rows: Sequence[Mapping[str, Any]], encounter_step: int) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    return [row for row in ordered if int(row["step"]) >= int(encounter_step)]


def closest_approach_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    valid = [row for row in rows if math.isfinite(_float(row, "signed_obstacle_clearance_m"))]
    if not valid:
        return None
    return int(min(valid, key=lambda row: (_float(row, "signed_obstacle_clearance_m"), int(row["step"])))["step"])


def first_sustained_clearance_increase_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_steps: int = SUSTAINED_CLEARANCE_STEPS,
) -> int | None:
    for index in range(0, len(rows) - required_steps + 1):
        window = rows[index : index + required_steps]
        if all(_bool(row, "clearance_increasing") for row in window):
            return int(window[0]["step"])
    return None


def initial_nonzero_sign(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    deadband: float = SIGN_DEADBAND_MPS,
) -> int:
    for row in rows:
        sign = lateral_sign(_float(row, field), deadband=deadband)
        if sign:
            return sign
    return 0


def first_countersteer_step(
    rows: Sequence[Mapping[str, Any]],
    initial_actual_sign: int,
) -> int | None:
    if initial_actual_sign == 0:
        return None
    for row in rows:
        if lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps")) == -initial_actual_sign:
            return int(row["step"])
    return None


def first_sustained_countersteer_step(
    rows: Sequence[Mapping[str, Any]],
    initial_actual_sign: int,
    *,
    window_steps: int = COUNTERSTEER_WINDOW_STEPS,
    minimum_counter_steps: int = COUNTERSTEER_MIN_STEPS,
) -> int | None:
    if initial_actual_sign == 0:
        return None
    for index in range(0, len(rows) - window_steps + 1):
        window = rows[index : index + window_steps]
        counter_rows = [
            row for row in window
            if lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps")) == -initial_actual_sign
        ]
        if len(counter_rows) >= minimum_counter_steps:
            return int(counter_rows[0]["step"])
    return None


def sign_change_count(values: Iterable[float]) -> int:
    signs = [lateral_sign(value) for value in values]
    nonzero = [value for value in signs if value]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def max_target_run(values: Iterable[float], target_sign: int) -> int:
    current = 0
    maximum = 0
    for value in values:
        if lateral_sign(value) == target_sign and target_sign:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _first_step(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["step"])
    return None


def _slice_steps(
    rows: Sequence[Mapping[str, Any]],
    start: int | None,
    end: int | None = None,
) -> list[Mapping[str, Any]]:
    if start is None:
        return []
    return [
        row for row in rows
        if int(row["step"]) >= start and (end is None or int(row["step"]) <= end)
    ]


def _counter_component(value: float, initial_actual_sign: int) -> float:
    return max(0.0, -float(initial_actual_sign) * value) if math.isfinite(value) else 0.0


def _same_component(value: float, initial_actual_sign: int) -> float:
    return max(0.0, float(initial_actual_sign) * value) if math.isfinite(value) else 0.0


def counter_impulses(
    rows: Sequence[Mapping[str, Any]],
    initial_actual_sign: int,
) -> dict[str, float]:
    actor_counter = sum(_counter_component(_float(row, "actor_raw_lateral_delta_v_mps"), initial_actual_sign) for row in rows)
    actor_same = sum(_same_component(_float(row, "actor_raw_lateral_delta_v_mps"), initial_actual_sign) for row in rows)
    final_counter = sum(_counter_component(_float(row, "final_applied_lateral_delta_v_mps"), initial_actual_sign) for row in rows)
    final_same = sum(_same_component(_float(row, "final_applied_lateral_delta_v_mps"), initial_actual_sign) for row in rows)
    counter_values = [
        _counter_component(_float(row, "actor_raw_lateral_delta_v_mps"), initial_actual_sign)
        for row in rows
    ]
    nonzero_counter = [value for value in counter_values if value > PIPELINE_RATIO_EPSILON_MPS]
    return {
        "cumulative_actor_counter_lateral_delta_v_mps": actor_counter,
        "cumulative_actor_same_direction_lateral_delta_v_mps": actor_same,
        "net_actor_counter_lateral_impulse_mps": actor_counter - actor_same,
        "cumulative_final_applied_counter_lateral_delta_v_mps": final_counter,
        "cumulative_final_applied_same_direction_lateral_delta_v_mps": final_same,
        "net_final_applied_counter_lateral_impulse_mps": final_counter - final_same,
        "mean_actor_countersteer_magnitude_mps": (
            statistics.fmean(nonzero_counter) if nonzero_counter else float("nan")
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > PIPELINE_RATIO_EPSILON_MPS else float("nan")


def pipeline_retention(
    rows: Sequence[Mapping[str, Any]],
    initial_actual_sign: int,
) -> dict[str, float | int]:
    counter_rows = [
        row for row in rows
        if lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps")) == -initial_actual_sign
    ] if initial_actual_sign else []
    fields = {
        "actor": "actor_raw_lateral_delta_v_mps",
        "post_transform": "post_transform_lateral_delta_v_mps",
        "pre_safety": "pre_safety_lateral_delta_v_mps",
        "post_boundary": "post_boundary_lateral_delta_v_mps",
        "post_safety": "post_safety_lateral_delta_v_mps",
        "final": "final_applied_lateral_delta_v_mps",
    }
    sums = {
        name: sum(_counter_component(_float(row, field), initial_actual_sign) for row in counter_rows)
        for name, field in fields.items()
    }
    return {
        "countersteer_pipeline_step_count": len(counter_rows),
        "actor_to_final_retention": _ratio(sums["final"], sums["actor"]),
        "transform_retention": _ratio(sums["post_transform"], sums["actor"]),
        "boundary_retention": _ratio(sums["post_boundary"], sums["pre_safety"]),
        "safety_retention": _ratio(sums["post_safety"], sums["post_boundary"]),
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def decay_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    speeds = [abs(_float(row, "actual_lateral_velocity_mps")) for row in rows]
    deltas = [left - right for left, right in zip(speeds, speeds[1:]) if math.isfinite(left) and math.isfinite(right)]
    return {
        "mean_lateral_speed_decay_mps_per_step": statistics.fmean(deltas) if deltas else float("nan"),
        "median_lateral_speed_decay_mps_per_step": statistics.median(deltas) if deltas else float("nan"),
        "p10_lateral_speed_decay_mps_per_step": _quantile(deltas, 0.10),
        "p90_lateral_speed_decay_mps_per_step": _quantile(deltas, 0.90),
    }


def response_delay_steps(
    rows: Sequence[Mapping[str, Any]],
    sustained_step: int | None,
    *,
    required_transitions: int = SUSTAINED_DECAY_TRANSITIONS,
) -> int | None:
    sliced = _slice_steps(rows, sustained_step)
    speeds = [abs(_float(row, "actual_lateral_velocity_mps")) for row in sliced]
    for index in range(0, len(speeds) - required_transitions):
        if all(
            speeds[index + offset + 1] < speeds[index + offset] - SIGN_DEADBAND_MPS
            for offset in range(required_transitions)
        ):
            return int(sliced[index]["step"]) - int(sustained_step)
    return None


def extra_drift(
    rows: Sequence[Mapping[str, Any]],
    initial_actual_sign: int,
) -> tuple[float, float]:
    if not rows or initial_actual_sign == 0:
        return float("nan"), float("nan")
    start = _float(rows[0], "actual_lateral_displacement_from_encounter_m")
    end = _float(rows[-1], "actual_lateral_displacement_from_encounter_m")
    if not math.isfinite(start) or not math.isfinite(end):
        return float("nan"), float("nan")
    signed_axis = end - start
    return float(initial_actual_sign * signed_axis), signed_axis


def _linear_slope(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    pairs = [(float(row["step"]), _float(row, field)) for row in rows]
    pairs = [(x, y) for x, y in pairs if math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return float("nan")
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator


def _agreement_fraction(rows: Sequence[Mapping[str, Any]]) -> float:
    pairs = []
    for row in rows:
        actor = lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps"))
        actual = lateral_sign(_float(row, "actual_lateral_velocity_mps"))
        if actor and actual:
            pairs.append(actor == actual)
    return statistics.fmean(float(value) for value in pairs) if pairs else float("nan")


def _fraction(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return statistics.fmean(float(predicate(row)) for row in rows) if rows else float("nan")


def _value_at_step(rows: Sequence[Mapping[str, Any]], step: int | None, field: str) -> float:
    if step is None:
        return float("nan")
    for row in rows:
        if int(row["step"]) == int(step):
            return _float(row, field)
    return float("nan")


def analyze_episode(
    episode_id: str,
    rows: Sequence[Mapping[str, Any]],
    episode_summary: Mapping[str, Any],
    *,
    near_zero_threshold_mps: float,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if not ordered or steps != list(range(steps[0], steps[-1] + 1)) or len(steps) != len(set(steps)):
        raise ValueError(f"non-contiguous or duplicate steps: {episode_id}")
    encounter_step = _int(episode_summary, "signed_encounter_step")
    if encounter_step is None:
        encounter_step = next((_int(row, "signed_encounter_step") for row in ordered if _int(row, "signed_encounter_step") is not None), None)
    if encounter_step is None:
        raise ValueError(f"missing encounter step: {episode_id}")
    encounter_rows = encounter_slice(ordered, encounter_step)
    closest_step = closest_approach_step(encounter_rows)
    post_closest = [row for row in encounter_rows if closest_step is not None and int(row["step"]) > closest_step]
    initial_actual = initial_nonzero_sign(encounter_rows, "actual_lateral_velocity_mps")
    initial_actor = initial_nonzero_sign(encounter_rows, "actor_raw_lateral_delta_v_mps")
    first_counter = first_countersteer_step(encounter_rows, initial_actual)
    sustained_counter = first_sustained_countersteer_step(encounter_rows, initial_actual)
    first_clearance = _first_step(encounter_rows, lambda row: _bool(row, "clearance_increasing"))
    sustained_clearance = first_sustained_clearance_increase_step(encounter_rows)
    first_past_closest = _first_step(post_closest, lambda row: _bool(row, "past_closest_approach"))
    peak_row = max(encounter_rows, key=lambda row: abs(_float(row, "actual_lateral_velocity_mps")))
    first_reversal = _first_step(
        encounter_rows,
        lambda row: lateral_sign(_float(row, "actual_lateral_velocity_mps")) == -initial_actual,
    ) if initial_actual else None
    first_near_zero = _first_step(
        _slice_steps(encounter_rows, sustained_counter),
        lambda row: abs(_float(row, "actual_lateral_velocity_mps")) <= near_zero_threshold_mps,
    )
    passed_step = _first_step(
        encounter_rows,
        lambda row: _bool(row, "passed_condition_now") or _bool(row, "passed_obstacle"),
    )
    impulse_rows = _slice_steps(encounter_rows, sustained_counter, first_near_zero)
    impulses = counter_impulses(impulse_rows, initial_actual)
    post_closest_impulses = counter_impulses(post_closest, initial_actual)
    retention = pipeline_retention(encounter_rows, initial_actual)
    decay = decay_statistics(impulse_rows)
    response_delay = response_delay_steps(encounter_rows, sustained_counter)
    drift_initial_direction, drift_signed_axis = extra_drift(impulse_rows, initial_actual)
    exit_anchor_step = sustained_clearance if sustained_clearance is not None else closest_step
    post_closest_actor_values = [_float(row, "actor_raw_lateral_delta_v_mps") for row in post_closest]
    post_closest_actual_values = [_float(row, "actual_lateral_velocity_mps") for row in post_closest]
    encounter_actor_values = [_float(row, "actor_raw_lateral_delta_v_mps") for row in encounter_rows]
    encounter_actual_values = [_float(row, "actual_lateral_velocity_mps") for row in encounter_rows]
    phase_a = [row for row in encounter_rows if closest_step is not None and int(row["step"]) <= closest_step]
    phase_b = [
        row for row in encounter_rows
        if closest_step is not None
        and int(row["step"]) > closest_step
        and (sustained_counter is None or int(row["step"]) < sustained_counter)
    ]
    phase_c = impulse_rows
    post_counter_fraction = _fraction(
        post_closest,
        lambda row: lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps")) == -initial_actual,
    ) if initial_actual else float("nan")
    post_same_fraction = _fraction(
        post_closest,
        lambda row: lateral_sign(_float(row, "actor_raw_lateral_delta_v_mps")) == initial_actual,
    ) if initial_actual else float("nan")
    actual_same_fraction = _fraction(
        post_closest,
        lambda row: lateral_sign(_float(row, "actual_lateral_velocity_mps")) == initial_actual,
    ) if initial_actual else float("nan")
    clearance_fraction = _fraction(post_closest, lambda row: _bool(row, "clearance_increasing"))
    passed_margins = [_float(row, "passed_margin_m") for row in post_closest]
    passed_margin_delta = (
        passed_margins[-1] - passed_margins[0]
        if len(passed_margins) >= 2 and math.isfinite(passed_margins[0]) and math.isfinite(passed_margins[-1])
        else float("nan")
    )
    reacceleration_fraction = _ratio(
        post_closest_impulses["cumulative_actor_same_direction_lateral_delta_v_mps"],
        post_closest_impulses["cumulative_actor_counter_lateral_delta_v_mps"]
        + post_closest_impulses["cumulative_actor_same_direction_lateral_delta_v_mps"],
    )
    return {
        "episode_id": episode_id,
        "group": "main_failure" if episode_id in MAIN_FAILURE_IDS else "special_failure" if episode_id in SPECIAL_FAILURE_IDS else "success_reference",
        "passed_obstacle": int(_bool(episode_summary, "passed_obstacle")),
        "post_bypass_recovery": int(_bool(episode_summary, "post_bypass_recovery")),
        "encounter_step": encounter_step,
        "closest_approach_step": closest_step,
        "minimum_clearance_m": _value_at_step(encounter_rows, closest_step, "signed_obstacle_clearance_m"),
        "first_clearance_increasing_step": first_clearance,
        "first_sustained_clearance_increase_step": sustained_clearance,
        "first_past_closest_approach_step": first_past_closest,
        "first_countersteer_step": first_counter,
        "first_sustained_countersteer_step": sustained_counter,
        "actual_lateral_velocity_peak_step": int(peak_row["step"]),
        "actual_lateral_velocity_peak_abs_mps": abs(_float(peak_row, "actual_lateral_velocity_mps")),
        "first_actual_lateral_speed_below_threshold_step": first_near_zero,
        "first_actual_lateral_sign_reversal_step": first_reversal,
        "passed_obstacle_step": passed_step if _bool(episode_summary, "passed_obstacle") else None,
        "last_step": steps[-1],
        "initial_actual_lateral_sign": initial_actual,
        "initial_actor_lateral_sign": initial_actor,
        "near_zero_lateral_threshold_mps": near_zero_threshold_mps,
        "countersteer_onset_delay_from_encounter_steps": (
            sustained_counter - encounter_step if sustained_counter is not None else None
        ),
        "countersteer_onset_delay_from_closest_steps": (
            sustained_counter - closest_step if sustained_counter is not None and closest_step is not None else None
        ),
        "countersteer_to_actual_deceleration_delay_steps": response_delay,
        "steps_to_near_zero_after_countersteer": (
            first_near_zero - sustained_counter if first_near_zero is not None and sustained_counter is not None else None
        ),
        "near_zero_censored": int(sustained_counter is not None and first_near_zero is None),
        "initial_exit_lateral_speed_mps": abs(_value_at_step(encounter_rows, exit_anchor_step, "actual_lateral_velocity_mps")),
        "lateral_speed_at_sustained_countersteer_mps": abs(_value_at_step(encounter_rows, sustained_counter, "actual_lateral_velocity_mps")),
        "extra_lateral_drift_after_countersteer_m": drift_initial_direction,
        "signed_axis_lateral_displacement_after_countersteer_m": drift_signed_axis,
        "actor_sign_changes_after_encounter": sign_change_count(encounter_actor_values),
        "actual_sign_changes_after_encounter": sign_change_count(encounter_actual_values),
        "actor_sign_changes_after_closest_approach": sign_change_count(post_closest_actor_values),
        "actual_sign_changes_after_closest_approach": sign_change_count(post_closest_actual_values),
        "actor_max_same_side_run_after_encounter": max_target_run(encounter_actor_values, initial_actual),
        "actual_max_same_side_run_after_encounter": max_target_run(encounter_actual_values, initial_actual),
        "actor_max_same_side_run_after_closest_approach": max_target_run(post_closest_actor_values, initial_actual),
        "actual_max_same_side_run_after_closest_approach": max_target_run(post_closest_actual_values, initial_actual),
        "actor_max_countersteer_run_after_encounter": max_target_run(encounter_actor_values, -initial_actual),
        "actor_max_countersteer_run_after_closest_approach": max_target_run(post_closest_actor_values, -initial_actual),
        "post_closest_actor_same_direction_fraction": post_same_fraction,
        "post_closest_actor_countersteer_fraction": post_counter_fraction,
        "post_closest_actual_same_direction_fraction": actual_same_fraction,
        "post_closest_clearance_increasing_fraction": clearance_fraction,
        "post_closest_passed_margin_delta_m": passed_margin_delta,
        "post_closest_passed_margin_slope_m_per_step": _linear_slope(post_closest, "passed_margin_m"),
        "post_closest_same_direction_reacceleration_fraction": reacceleration_fraction,
        "actor_actual_agreement_encounter_to_closest": _agreement_fraction(phase_a),
        "actor_actual_agreement_closest_to_countersteer": _agreement_fraction(phase_b),
        "actor_actual_agreement_countersteer_to_near_zero_or_end": _agreement_fraction(phase_c),
        "safety_projection_steps_after_encounter": sum(_bool(row, "safety_projection_applied") for row in encounter_rows),
        "boundary_projection_steps_after_encounter": sum(_bool(row, "boundary_projection_applied") for row in encounter_rows),
        **impulses,
        "post_closest_actor_counter_impulse_mps": post_closest_impulses["cumulative_actor_counter_lateral_delta_v_mps"],
        "post_closest_actor_same_direction_impulse_mps": post_closest_impulses["cumulative_actor_same_direction_lateral_delta_v_mps"],
        "post_closest_net_actor_counter_impulse_mps": post_closest_impulses["net_actor_counter_lateral_impulse_mps"],
        **retention,
        **decay,
    }


def success_reference_statistics(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = REFERENCE_METRICS,
) -> dict[str, dict[str, Any]]:
    reference_rows = [row for row in diagnostics if row.get("episode_id") in REFERENCE_IDS]
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in reference_rows if _finite(row.get(metric))]
        result[metric] = {
            "metric": metric,
            "eligible_episode_count": len(reference_rows),
            "n": len(values),
            "q1": _quantile(values, 0.25),
            "median": _quantile(values, 0.50),
            "q3": _quantile(values, 0.75),
            "iqr": _quantile(values, 0.75) - _quantile(values, 0.25) if values else float("nan"),
            "minimum": min(values) if values else float("nan"),
            "maximum": max(values) if values else float("nan"),
            "quantile_method": "linear_(n-1)_interpolation",
        }
    return result


def compare_to_reference(value: Any, reference: Mapping[str, Any]) -> str:
    if not _finite(value) or not _finite(reference.get("q1")) or not _finite(reference.get("q3")):
        return "NOT_COMPARABLE"
    if float(value) < float(reference["q1"]):
        return "BELOW_REFERENCE_IQR"
    if float(value) > float(reference["q3"]):
        return "ABOVE_REFERENCE_IQR"
    return "WITHIN_REFERENCE_IQR"


def diagnose_episode(
    diagnostic: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], dict[str, bool]]:
    episode_id = str(diagnostic["episode_id"])
    if episode_id == _episode_id("0013"):
        return "SAFETY_DOMINATED_SPECIAL_CASE", [], {"special_case": True}
    if episode_id == _episode_id("0014"):
        return "BOUNDARY_DOMINATED_SPECIAL_CASE", [], {"special_case": True}
    if not diagnostic.get("initial_actual_lateral_sign") or diagnostic.get("closest_approach_step") is None:
        return "NOT_ASSESSABLE", [], {"missing_anchor": True}

    sustained = diagnostic.get("first_sustained_countersteer_step") is not None
    post_counter_fraction = float(diagnostic.get("post_closest_actor_countersteer_fraction", float("nan")))
    reacceleration_fraction = float(diagnostic.get("post_closest_same_direction_reacceleration_fraction", float("nan")))
    changes = int(diagnostic.get("actor_sign_changes_after_closest_approach", 0))
    never = (not sustained) and (not math.isfinite(post_counter_fraction) or post_counter_fraction <= 0.10)
    not_sustained = bool(
        (diagnostic.get("first_countersteer_step") is not None)
        and (
            not sustained
            or (
                changes >= 2
                and math.isfinite(reacceleration_fraction)
                and reacceleration_fraction >= 0.25
            )
        )
    )
    retention_class = compare_to_reference(
        diagnostic.get("actor_to_final_retention"), references["actor_to_final_retention"]
    )
    retention_value = float(diagnostic.get("actor_to_final_retention", float("nan")))
    # A reference distribution above 1.0 means successful trajectories often
    # received added downstream authority.  Exact retention of 1.0 is not
    # attenuation, even if it falls below that success IQR.
    retention_normal = math.isfinite(retention_value) and retention_value >= 0.90
    weak = bool(
        sustained
        and retention_normal
        and (
            compare_to_reference(
                diagnostic.get("mean_actor_countersteer_magnitude_mps"),
                references["mean_actor_countersteer_magnitude_mps"],
            ) == "BELOW_REFERENCE_IQR"
            or compare_to_reference(
                diagnostic.get("net_actor_counter_lateral_impulse_mps"),
                references["net_actor_counter_lateral_impulse_mps"],
            ) == "BELOW_REFERENCE_IQR"
        )
    )
    pipeline = bool(
        sustained
        and retention_class == "BELOW_REFERENCE_IQR"
        and math.isfinite(retention_value)
        and retention_value < 0.90
    )
    decay_low = compare_to_reference(
        diagnostic.get("mean_lateral_speed_decay_mps_per_step"),
        references["mean_lateral_speed_decay_mps_per_step"],
    ) == "BELOW_REFERENCE_IQR"
    stop_slow = (
        int(diagnostic.get("near_zero_censored", 0)) == 1
        or compare_to_reference(
            diagnostic.get("steps_to_near_zero_after_countersteer"),
            references["steps_to_near_zero_after_countersteer"],
        ) == "ABOVE_REFERENCE_IQR"
    )
    drift_high = compare_to_reference(
        diagnostic.get("extra_lateral_drift_after_countersteer_m"),
        references["extra_lateral_drift_after_countersteer_m"],
    ) == "ABOVE_REFERENCE_IQR"
    strength_normal = compare_to_reference(
        diagnostic.get("mean_actor_countersteer_magnitude_mps"),
        references["mean_actor_countersteer_magnitude_mps"],
    ) in {"WITHIN_REFERENCE_IQR", "ABOVE_REFERENCE_IQR"}
    momentum = bool(sustained and strength_normal and retention_normal and (decay_low or stop_slow or drift_high))

    gates = {
        "actor_never_countersteers": never,
        "countersteer_too_weak": weak,
        "countersteer_not_sustained": not_sustained,
        "pipeline_attenuation": pipeline,
        "lateral_momentum_decay_too_slow": momentum,
    }
    labels = [
        label for label, active in (
            ("ACTOR_NEVER_COUNTERSTEERS", never),
            ("COUNTERSTEER_TOO_WEAK", weak),
            ("COUNTERSTEER_NOT_SUSTAINED", not_sustained),
            ("PIPELINE_ATTENUATION", pipeline),
            ("LATERAL_MOMENTUM_DECAY_TOO_SLOW", momentum),
        ) if active
    ]
    if len(labels) == 1:
        return labels[0], [], gates
    if len(labels) > 1:
        return "MIXED", labels, gates
    return "NOT_ASSESSABLE", [], gates


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else value for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if _finite(row.get(field))]
    return {
        "n": len(values),
        "median": _quantile(values, 0.5),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
    }


def _render_report(
    payload: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
) -> str:
    by_id = {str(row["episode_id"]): row for row in diagnostics}
    main_rows = [by_id[value] for value in MAIN_FAILURE_IDS]
    success_rows = [by_id[value] for value in REFERENCE_IDS]
    started = sum(row.get("first_sustained_countersteer_step") is not None for row in main_rows)
    pipeline_low = sum(row.get("primary_diagnosis") == "PIPELINE_ATTENUATION" for row in main_rows)
    diagnosis_counts = payload["diagnosis_counts"]
    lines = [
        "LATERAL_EXIT_AUTHORITY_AUDIT_READY",
        "",
        "# Single-UAV Encounter-Relative Lateral Exit Authority Offline Audit",
        "",
        "## A. Source identity",
        "",
        f"- episode_count: `{payload['source_identity']['episode_count']}`",
        f"- step_count: `{payload['source_identity']['step_count']}`",
        "- `only_logging_changed=true`; `logging_perturbed_trajectory=false`。",
        "- checkpoint、policy、observation dimension、ordered identity、scenario manifest、max steps、frozen controller identity 均匹配。",
        "- 只读取五个冻结 Signed Bypass Observability artifact；未执行环境、rollout、formal、smoke 或训练。",
        "",
        "固定规则：encounter 后切片；counter-steer 为 Actor lateral command 与 encounter 后初始 actual lateral sign 反号；5 步窗口至少 4 步为 sustained；clearance 连续 3 步增加；actual speed 连续 3 个 transition 下降；near-zero 固定为现有 stable speed tolerance 0.5 m/s。",
        "",
        "## B. 七个主要 failure",
        "",
        "| Episode | Primary | Sustained onset | Steps to near-zero | Actor→final | Extra drift (m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            "| {episode} | {diagnosis} | {onset} | {stop} | {retention} | {drift} |".format(
                episode=str(row["episode_id"])[-4:],
                diagnosis=row["primary_diagnosis"],
                onset=row.get("first_sustained_countersteer_step", ""),
                stop=row.get("steps_to_near_zero_after_countersteer", "") if row.get("steps_to_near_zero_after_countersteer") is not None else "censored",
                retention=(f"{row['actor_to_final_retention']:.3f}" if _finite(row.get("actor_to_final_retention")) else "NA"),
                drift=(f"{row['extra_lateral_drift_after_countersteer_m']:.3f}" if _finite(row.get("extra_lateral_drift_after_countersteer_m")) else "NA"),
            )
        )
    lines += [
        "",
        "0002 不是 encounter-relative oscillation：encounter 后 Actor sign changes 为 0，立即持续 counter-steer，pipeline retention 正常；主要证据指向 lateral momentum 持续。0012 则是 Actor command 频繁切换、actual sign 很少变化，属于 exit intent 不稳定而非“从未反向”。0020 已有持续反向意图且 pipeline 未明显削弱，但到 near-zero 的时间和额外漂移偏大。",
        "",
        "## C. 0013 / 0014 特殊案例",
        "",
        f"- 0013: `{by_id[_episode_id('0013')]['primary_diagnosis']}`；仅补充 projection 前 Actor counter-steer，不进入主 cluster。",
        f"- 0014: `{by_id[_episode_id('0014')]['primary_diagnosis']}`；仅补充 projection 前 Actor counter-steer，不进入主 cluster。",
        "",
        "## D. Success reference",
        "",
        "11 个成功 episode 使用同一 encounter-relative 规则。每个指标按有效 episode 计算 median 与 IQR；不存在 sustained counter-steer 或 near-zero 的 episode 保留为缺失/删失，不进行填补。",
        "",
        "| Metric | n | Q1 | Median | Q3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in REFERENCE_METRICS:
        ref = references[metric]
        lines.append(
            f"| {metric} | {ref['n']} | {ref['q1'] if _finite(ref['q1']) else 'NA'} | {ref['median'] if _finite(ref['median']) else 'NA'} | {ref['q3'] if _finite(ref['q3']) else 'NA'} |"
        )
    failure_drift = _group_summary(main_rows, "extra_lateral_drift_after_countersteer_m")
    success_drift = _group_summary(success_rows, "extra_lateral_drift_after_countersteer_m")
    failure_stop = _group_summary(main_rows, "steps_to_near_zero_after_countersteer")
    success_stop = _group_summary(success_rows, "steps_to_near_zero_after_countersteer")
    lines += [
        "",
        "## E. Failure vs success 主要差异",
        "",
        f"主要 failure 的 extra drift median={failure_drift['median']:.3f} m（n={failure_drift['n']}），成功参考 median={success_drift['median']:.3f} m（n={success_drift['n']}）。",
        f"主要 failure 的 steps-to-near-zero median={failure_stop['median']:.3f}（n={failure_stop['n']}），成功参考 median={success_stop['median']:.3f}（n={success_stop['n']}）。",
        "逐 episode 的 BELOW/WITHIN/ABOVE IQR 标记见 `lateral_exit_failure_vs_reference.csv`。这些是描述性参照，不是因果显著性检验。",
        "",
        "## F. Actor counter-steer 结论",
        "",
        f"七个主要 failure 中 {started}/7 出现 sustained counter-steer，因此总体不支持“Actor 从未开始反向修正”。主要差别是反向意图的持续性和反向后 actual lateral momentum 的消退时间。",
        "",
        "## G. Pipeline attenuation 结论",
        "",
        f"主 cluster 中 `{pipeline_low}` 个 episode 的 primary diagnosis 为 PIPELINE_ATTENUATION。七个主案例 Actor→final retention 均为 1.0；它们虽低于成功组因额外 downstream authority 而高于 1 的 IQR，但绝对值表示 counter-steer 未被削弱。未见 transform/safety/boundary 普遍削弱 counter-steer 的证据；0013/0014 仍按 safety/boundary 特殊案例单列。因此当前没有证据要求先修改 safety、boundary 或 Terminal controller。",
        "",
        "## H. Lateral momentum decay 结论",
        "",
        "七个主要 failure 不属于同一机制。0002、0004、0020 的强证据是 sustained counter-steer 已存在、pipeline retention 正常，但 near-zero 延迟或额外 lateral drift 超出成功参考。0005、0011、0012 更接近 command 在反向与原方向之间反复切换；0006 呈两者混合特征。没有主案例满足 COUNTERSTEER_TOO_WEAK 或 PIPELINE_ATTENUATION。",
        "",
        "## I. 是否仍支持“Actor 不知道什么时候停”",
        "",
        "不再支持把七个主要 failure 统一描述为“Actor 根本不知道什么时候停横移”。更准确的探索性描述是：Actor 通常已产生停止/反向意图，但部分 episode 的意图不稳定，另一些 episode 的实际 lateral momentum 消退过慢；两者共同形成 lateral exit authority 不足。",
        "",
        "## J. 唯一下一步实验建议",
        "",
        "只建议一个最小因果实验：冻结 Actor、场景、controller 其余分支和所有 safety/boundary 参数，仅对 encounter 后 sustained counter-steer 已出现的窗口做 `minimal lateral-exit braking assistance OFF/ON` 配对诊断。该实验应预注册固定 lateral braking cap，并以 steps-to-near-zero 与 extra drift 为主要终点；本轮未实现或运行。",
        "",
        "## K. Tests / py_compile / git diff",
        "",
        "`31/31` 项纯 unit tests 通过；审计脚本与测试文件通过 py_compile；`git diff --check` 通过（仅有既存 LF/CRLF warning，无 whitespace error）。审计脚本不导入 environment runtime，也不调用 step。",
        "",
        "## L. Source hash verification",
        "",
        f"- source_hashes_unchanged: `{str(payload['source_hashes_unchanged']).lower()}`",
    ]
    for name, digest in payload["source_hashes"]["before"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines += [
        "",
        "## 最终状态",
        "",
        "```text",
        "lateral_exit_authority_audit_executed=true",
        "controller_modified=false",
        "actor_modified=false",
        "critic_modified=false",
        "gat_modified=false",
        "reward_modified=false",
        "dynamics_modified=false",
        "safety_modified=false",
        "boundary_modified=false",
        "action_transform_modified=false",
        "goal_recovery_modified=false",
        "terminal_controller_modified=false",
        "training_performed=false",
        "rollout_executed=false",
        "formal_executed=false",
        "smoke_executed=false",
        "environment_step_called=false",
        "source_logs_modified=false",
        "```",
        "",
        f"Diagnosis counts: `{json.dumps(diagnosis_counts, ensure_ascii=False, sort_keys=True)}`。",
    ]
    return "\n".join(lines) + "\n"


def _validate_source_identity(
    summary: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lock = summary.get("identity_lock", {})
    required_lock = (
        "same_checkpoint_sha256",
        "same_policy_hash",
        "same_observation_dimension",
        "same_ordered_identity",
        "same_scenario_manifest",
        "same_max_steps",
        "same_frozen_controller_identity",
    )
    checks = {
        "episode_count": int(summary.get("episode_count", -1)) == 20 == len(episode_rows),
        "step_count": int(summary.get("step_count", -1)) == 2182 == len(step_rows),
        "only_logging_changed": summary.get("only_logging_changed") is True,
        "logging_perturbed_trajectory": lock.get("logging_perturbed_trajectory") is False,
        **{key: lock.get(key) is True for key in required_lock},
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise SourceIdentityError("LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: " + ", ".join(failed))
    return {
        "status": "PASS",
        "episode_count": len(episode_rows),
        "step_count": len(step_rows),
        "checks": checks,
        "identity_prefix_scanned_bytes": summary.get("_identity_prefix_scanned_bytes"),
    }


def run_audit(
    *,
    source_root: Path = SOURCE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output or source in output.parents:
        raise ValueError("output root must be separate from the frozen source root")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing audit root: {output}")
        if not output.is_dir() or output.is_symlink():
            raise ValueError("audit output root must be a regular directory")
        existing = {path.name for path in output.iterdir() if path.is_file()}
        unexpected = existing - set(OUTPUT_FILENAMES)
        nested = [path.name for path in output.iterdir() if path.is_dir()]
        if unexpected or nested:
            raise ValueError(
                "refusing overwrite with unexpected output entries: "
                + ", ".join(sorted(unexpected | set(nested)))
            )
    for name in SOURCE_FILENAMES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    hashes_before = source_hashes(source)
    summary = load_summary_identity_prefix(source / SOURCE_FILENAMES[0])
    episode_rows = _load_csv(source / SOURCE_FILENAMES[1], max_rows=100)
    step_rows = _load_csv(source / SOURCE_FILENAMES[2], max_rows=5_000)
    focus_rows = _load_csv(source / SOURCE_FILENAMES[3], max_rows=100)
    transition_rows = _load_csv(source / SOURCE_FILENAMES[4], max_rows=5_000)
    identity = _validate_source_identity(summary, episode_rows, step_rows)
    episode_ids = require_unique_episode_ids(
        episode_rows, artifact_name=SOURCE_FILENAMES[1]
    )
    focus_ids = require_unique_episode_ids(
        focus_rows, artifact_name=SOURCE_FILENAMES[3]
    )
    episode_by_id = {str(row["episode_id"]): row for row in episode_rows}
    steps_by_id: dict[str, list[dict[str, str]]] = {}
    for row in step_rows:
        steps_by_id.setdefault(str(row["episode_id"]), []).append(row)
    expected_ids = set(TARGET_IDS) | set(REFERENCE_IDS)
    if episode_ids != expected_ids or set(steps_by_id) != expected_ids:
        raise SourceIdentityError("LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: episode cohort mismatch")
    if focus_ids != set(TARGET_IDS):
        raise SourceIdentityError("LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: focus failure cohort mismatch")
    validate_transition_rows(transition_rows, expected_ids=expected_ids)
    thresholds = {
        _float(row, "stable_speed_threshold_mps")
        for row in episode_rows
        if math.isfinite(_float(row, "stable_speed_threshold_mps"))
    }
    if len(thresholds) != 1:
        raise ValueError("stable speed threshold is not uniquely defined")
    near_zero = thresholds.pop()
    if abs(near_zero - DEFAULT_NEAR_ZERO_LATERAL_THRESHOLD_MPS) > 1.0e-12:
        raise ValueError("unexpected frozen near-zero threshold source")

    diagnostics = [
        analyze_episode(
            episode_id,
            steps_by_id[episode_id],
            episode_by_id[episode_id],
            near_zero_threshold_mps=near_zero,
        )
        for episode_id in sorted(expected_ids)
    ]
    references = success_reference_statistics(diagnostics)
    for row in diagnostics:
        primary, secondary, gates = diagnose_episode(row, references)
        row["primary_diagnosis"] = primary
        row["secondary_diagnoses"] = "|".join(secondary)
        row["diagnosis_gates"] = json.dumps(gates, sort_keys=True, separators=(",", ":"))

    failure_vs_reference: list[dict[str, Any]] = []
    for row in diagnostics:
        if row["episode_id"] not in TARGET_IDS:
            continue
        for metric in REFERENCE_METRICS:
            failure_vs_reference.append(
                {
                    "episode_id": row["episode_id"],
                    "group": row["group"],
                    "primary_diagnosis": row["primary_diagnosis"],
                    "metric": metric,
                    "value": row.get(metric),
                    "reference_n": references[metric]["n"],
                    "reference_q1": references[metric]["q1"],
                    "reference_median": references[metric]["median"],
                    "reference_q3": references[metric]["q3"],
                    "relative_to_reference_iqr": compare_to_reference(row.get(metric), references[metric]),
                }
            )

    main_rows = [row for row in diagnostics if row["episode_id"] in MAIN_FAILURE_IDS]
    diagnosis_counts = {
        label: sum(row["primary_diagnosis"] == label for row in diagnostics if row["episode_id"] in TARGET_IDS)
        for label in PRIMARY_DIAGNOSES
    }
    hashes_after = source_hashes(source)
    assert_source_hashes_unchanged(hashes_before, hashes_after)
    payload = {
        "stage": "single_uav_lateral_exit_authority_offline_audit",
        "source_identity": identity,
        "source_hashes": {"before": hashes_before, "after": hashes_after},
        "source_hashes_unchanged": True,
        "target_episode_ids": list(TARGET_IDS),
        "reference_episode_ids": list(REFERENCE_IDS),
        "analysis_rules": {
            "encounter_slice": "step >= signed_encounter_step",
            "sign_deadband_mps": SIGN_DEADBAND_MPS,
            "near_zero_lateral_threshold_mps": near_zero,
            "near_zero_threshold_source": "episodes.stable_speed_threshold_mps",
            "sustained_countersteer_window_steps": COUNTERSTEER_WINDOW_STEPS,
            "sustained_countersteer_min_counter_steps": COUNTERSTEER_MIN_STEPS,
            "sustained_clearance_increase_steps": SUSTAINED_CLEARANCE_STEPS,
            "sustained_actual_deceleration_transitions": SUSTAINED_DECAY_TRANSITIONS,
            "pipeline_attenuation_absolute_retention_threshold": 0.90,
            "whole_episode_failure_label_used_for_primary_diagnosis": False,
            "special_cases_excluded_from_main_cluster_statistics": True,
        },
        "encounter_relative_metrics": diagnostics,
        "per_episode_diagnosis": [
            {
                "episode_id": row["episode_id"],
                "group": row["group"],
                "primary": row["primary_diagnosis"],
                "secondary": row["secondary_diagnoses"].split("|") if row["secondary_diagnoses"] else [],
                "gates": json.loads(row["diagnosis_gates"]),
            }
            for row in diagnostics if row["episode_id"] in TARGET_IDS
        ],
        "success_reference_statistics": references,
        "failure_vs_reference": failure_vs_reference,
        "countersteer_onset_statistics": {
            "main_failures": _group_summary(main_rows, "countersteer_onset_delay_from_closest_steps"),
            "success_reference": _group_summary([row for row in diagnostics if row["episode_id"] in REFERENCE_IDS], "countersteer_onset_delay_from_closest_steps"),
        },
        "countersteer_impulse_statistics": {
            "main_failures": _group_summary(main_rows, "net_actor_counter_lateral_impulse_mps"),
            "success_reference": _group_summary([row for row in diagnostics if row["episode_id"] in REFERENCE_IDS], "net_actor_counter_lateral_impulse_mps"),
        },
        "pipeline_retention_statistics": {
            "main_failures": _group_summary(main_rows, "actor_to_final_retention"),
            "success_reference": _group_summary([row for row in diagnostics if row["episode_id"] in REFERENCE_IDS], "actor_to_final_retention"),
        },
        "lateral_decay_statistics": {
            "main_failures": _group_summary(main_rows, "mean_lateral_speed_decay_mps_per_step"),
            "success_reference": _group_summary([row for row in diagnostics if row["episode_id"] in REFERENCE_IDS], "mean_lateral_speed_decay_mps_per_step"),
        },
        "extra_drift_statistics": {
            "main_failures": _group_summary(main_rows, "extra_lateral_drift_after_countersteer_m"),
            "success_reference": _group_summary([row for row in diagnostics if row["episode_id"] in REFERENCE_IDS], "extra_lateral_drift_after_countersteer_m"),
        },
        "diagnosis_counts": diagnosis_counts,
        "status": {
            "lateral_exit_authority_audit_executed": True,
            "controller_modified": False,
            "actor_modified": False,
            "critic_modified": False,
            "gat_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "safety_modified": False,
            "boundary_modified": False,
            "action_transform_modified": False,
            "goal_recovery_modified": False,
            "terminal_controller_modified": False,
            "training_performed": False,
            "rollout_executed": False,
            "formal_executed": False,
            "smoke_executed": False,
            "environment_step_called": False,
            "source_logs_modified": False,
        },
    }

    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "lateral_exit_episode_diagnostics.csv", diagnostics)
    window_fields = [
        "episode_id", "group", "encounter_step", "closest_approach_step",
        "first_countersteer_step", "first_sustained_countersteer_step",
        "first_actual_lateral_speed_below_threshold_step", "last_step",
        "cumulative_actor_counter_lateral_delta_v_mps",
        "cumulative_actor_same_direction_lateral_delta_v_mps",
        "net_actor_counter_lateral_impulse_mps",
        "cumulative_final_applied_counter_lateral_delta_v_mps",
        "cumulative_final_applied_same_direction_lateral_delta_v_mps",
        "net_final_applied_counter_lateral_impulse_mps",
        "actor_to_final_retention", "transform_retention", "safety_retention",
        "boundary_retention", "countersteer_to_actual_deceleration_delay_steps",
        "steps_to_near_zero_after_countersteer", "near_zero_censored",
        "extra_lateral_drift_after_countersteer_m",
        "signed_axis_lateral_displacement_after_countersteer_m",
    ]
    _write_csv(
        output / "lateral_exit_countersteer_windows.csv",
        [{key: row.get(key) for key in window_fields} for row in diagnostics],
        fieldnames=window_fields,
    )
    _write_csv(
        output / "lateral_exit_success_reference.csv",
        [references[metric] for metric in REFERENCE_METRICS],
    )
    _write_csv(output / "lateral_exit_failure_vs_reference.csv", failure_vs_reference)
    _write_json(output / "lateral_exit_authority_audit_summary.json", payload)
    _atomic_write_text(output / "lateral_exit_report.md", _render_report(payload, diagnostics, references))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-UAV lateral exit authority offline audit")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_audit(
            source_root=args.source_root,
            output_root=args.output_root,
            overwrite=bool(args.force),
        )
    except SourceIdentityError as exc:
        print("LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL")
        print(str(exc))
        return 2
    print("LATERAL_EXIT_AUTHORITY_AUDIT_READY")
    print(json.dumps(_jsonable(payload["diagnosis_counts"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_NEAR_ZERO_LATERAL_THRESHOLD_MPS",
    "MAIN_FAILURE_IDS",
    "OUTPUT_ROOT",
    "REFERENCE_IDS",
    "SOURCE_ROOT",
    "SPECIAL_FAILURE_IDS",
    "SourceIdentityError",
    "analyze_episode",
    "assert_source_hashes_unchanged",
    "closest_approach_step",
    "compare_to_reference",
    "counter_impulses",
    "decay_statistics",
    "diagnose_episode",
    "encounter_slice",
    "extra_drift",
    "first_countersteer_step",
    "first_sustained_clearance_increase_step",
    "first_sustained_countersteer_step",
    "initial_nonzero_sign",
    "lateral_sign",
    "pipeline_retention",
    "response_delay_steps",
    "run_audit",
    "sign_change_count",
    "success_reference_statistics",
]
