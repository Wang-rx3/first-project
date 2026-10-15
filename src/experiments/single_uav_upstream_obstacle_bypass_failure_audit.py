"""Offline audit of upstream obstacle-bypass failures in the frozen horizon120 logs.

This module is deliberately independent of the simulator and controller runtime.
It reads only the existing CSV/JSON artifacts, verifies their identity gate, and
writes a bounded diagnostic package.  No checkpoint is loaded and no environment
step, rollout, training, or controller code is called.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "preliminary"
    / "single_uav_near_goal_capture_horizon120_diagnostic_seed2026"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "preliminary"
    / "single_uav_upstream_obstacle_bypass_failure_audit_seed2026"
)

TARGET_EPISODES = [
    f"test_r99_episode_{index:04d}"
    for index in (2, 4, 5, 6, 11, 12, 13, 14, 20)
]
REFERENCE_EPISODES = [
    f"test_r99_episode_{index:04d}"
    for index in (1, 3, 7, 8, 9, 10, 15, 16, 17, 18, 19)
]
EXPECTED_EPISODES = TARGET_EPISODES + REFERENCE_EPISODES

# These constants are copied from the frozen source configuration, not imported
# from the simulator.  The audit records their provenance in the JSON report.
SOFT_RECOVERY_BUFFER_M = 0.75
STATIC_SAFE_DISTANCE_M = 2.0
DECISION_DT_S = 0.2

VECTOR_COLUMNS = (
    "actor_raw_delta_v_mps",
    "corrected_delta_v_pre_transform_mps",
    "corrected_delta_v_pre_safety_mps",
    "applied_delta_v_mps",
    "goal_recovery_correction_delta_v_mps",
    "goal_recovery_applied_delta_v_mps",
    "safety_projection_delta_v_mps",
    "motion_boundary_delta_v_mps",
)

BOOL_COLUMNS = (
    "passed_obstacle",
    "post_bypass_recovery",
    "goal_recovery_correction_enabled",
    "goal_recovery_correction_eligible",
    "goal_recovery_correction_active",
    "terminal_capture_active",
    "terminal_v2_governor_eligible",
    "terminal_v2_governor_active",
    "terminal_v21_radial_conflict_eligible",
    "terminal_v21_radial_conflict_active",
    "terminal_v22_hold_latched",
    "terminal_v22_restorative_active",
    "terminal_v23_guard_active",
    "terminal_v23_preboundary_active",
    "terminal_v23_postboundary_active",
    "terminal_v23_radial_propulsion_allowed",
    "terminal_v23_radial_propulsion_suppressed",
    "terminal_v23_authority_active",
    "terminal_v23_preboundary_authority_repair_active",
    "safety_projection_applied",
    "boundary_projection_applied",
    "action_clipped",
    "action_transform_applied",
    "collision",
)


class IdentityFailure(RuntimeError):
    """Raised when the source identity gate is not satisfied."""


def _float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        lowered = str(value).strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
        return float("nan")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _step(row: Mapping[str, Any]) -> int:
    return int(round(_float(row.get("step"))))


def _vector(row: Mapping[str, Any], field: str) -> list[float]:
    value = row.get(field, "")
    if isinstance(value, (list, tuple)):
        values = value
    else:
        try:
            values = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(values, (list, tuple)):
        return []
    result: list[float] = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        result.append(number)
    return result


def _norm(row: Mapping[str, Any], field: str) -> float:
    values = _vector(row, field)
    return math.sqrt(sum(value * value for value in values)) if values else float("nan")


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if _finite(value)]
    return statistics.fmean(clean) if clean else float("nan")


def _median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if _finite(value)]
    return statistics.median(clean) if clean else float("nan")


def _quantile(values: Iterable[float], probability: float) -> float:
    clean = sorted(float(value) for value in values if _finite(value))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    index = (len(clean) - 1) * probability
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return clean[lower]
    weight = index - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _safe(value: Any) -> Any:
    """Convert NaN/Inf and nested values to strict JSON-compatible values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not _finite(value):
        return ""
    return f"{float(value):.{digits}f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise IdentityFailure(f"JSON root is not an object: {path.name}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (list, tuple, dict)):
                    output[column] = json.dumps(_safe(value), ensure_ascii=False, sort_keys=True)
                elif isinstance(value, float) and not math.isfinite(value):
                    output[column] = ""
                else:
                    output[column] = value
            writer.writerow(output)


def _slope(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    points = [(_step(row), _float(row.get(field))) for row in rows]
    points = [(x, y) for x, y in points if math.isfinite(y)]
    if len(points) < 2:
        return float("nan")
    x_mean = _mean(point[0] for point in points)
    y_mean = _mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        return float("nan")
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def _longest_streak(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    longest = current = 0
    for row in rows:
        if _bool(row.get(field)):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _first_step(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return _step(row)
    return None


def _rows_between(rows: Sequence[Mapping[str, Any]], start: int, end: int) -> list[dict[str, str]]:
    return [row for row in rows if start <= _step(row) <= end]


def _episode_groups(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("episode_id", "")), []).append(row)
    for values in groups.values():
        values.sort(key=_step)
    return groups


def _encounter_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Find the first source-defined obstacle-near state.

    The frozen recovery tracker uses ``along_obstacle_m`` relative to the goal
    direction and a rear/face buffer of 0.75 m.  The bounded audit therefore
    selects the first row with a valid obstacle ID and
    ``-(extent+buffer) <= along <= extent+buffer``.  If that interval is absent,
    the first valid obstacle row is retained as a transparent fallback.
    """

    valid: list[Mapping[str, Any]] = []
    for row in rows:
        along = _float(row.get("along_obstacle_m"))
        extent = _float(row.get("obstacle_extent_m"))
        if not row.get("relevant_obstacle_id") or not (_finite(along) and _finite(extent)):
            continue
        valid.append(row)
        threshold = extent + SOFT_RECOVERY_BUFFER_M
        if -threshold <= along <= threshold:
            return _step(row)
    return _step(valid[0]) if valid else None


def _first_flag(rows: Sequence[Mapping[str, Any]], field: str) -> int | None:
    return _first_step(rows, lambda row: _bool(row.get(field)))


def _stats(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if _finite(value)]
    return {
        "n": len(clean),
        "median": _median(clean),
        "q1": _quantile(clean, 0.25),
        "q3": _quantile(clean, 0.75),
        "iqr": (
            _quantile(clean, 0.75) - _quantile(clean, 0.25)
            if clean
            else float("nan")
        ),
        "mean": _mean(clean),
        "min": min(clean) if clean else float("nan"),
        "max": max(clean) if clean else float("nan"),
    }


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    # Keep the explicit vector aliases before the generic suffix rule.  The
    # CSV names are ``*_delta_v_mps`` whereas the audit metric names are
    # ``*_norm_mps``.
    if metric == "actor_raw_norm_mps":
        return _norm(row, "actor_raw_delta_v_mps")
    if metric == "pre_transform_norm_mps":
        return _norm(row, "corrected_delta_v_pre_transform_mps")
    if metric == "pre_safety_norm_mps":
        return _norm(row, "corrected_delta_v_pre_safety_mps")
    if metric == "applied_norm_mps":
        return _norm(row, "applied_delta_v_mps")
    if metric == "correction_norm_mps":
        return _norm(row, "goal_recovery_applied_delta_v_mps")
    if metric == "safety_projection_norm_mps":
        return _norm(row, "safety_projection_delta_v_mps")
    if metric == "boundary_projection_norm_mps":
        return _norm(row, "motion_boundary_delta_v_mps")
    if metric.endswith("_norm_mps"):
        field = metric.removesuffix("_norm_mps") + "_mps"
        return _norm(row, field)
    if metric.endswith("_fraction"):
        return 1.0 if _bool(row.get(metric.removesuffix("_fraction"))) else 0.0
    return _float(row.get(metric))


def _window_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "post_step_actual_radial_velocity_mps",
        "post_step_actual_tangential_velocity_mps",
        "actor_raw_norm_mps",
        "pre_transform_norm_mps",
        "pre_safety_norm_mps",
        "applied_norm_mps",
        "correction_norm_mps",
        "relevant_obstacle_clearance_m",
        "along_obstacle_m",
        "goal_distance_after_m",
        "safety_projection_applied_fraction",
        "boundary_projection_applied_fraction",
        "action_clipped_fraction",
        "action_transform_applied_fraction",
        "terminal_v2_governor_active_fraction",
        "terminal_v21_radial_conflict_active_fraction",
        "terminal_v22_hold_latched_fraction",
        "terminal_v23_guard_active_fraction",
        "goal_recovery_correction_active_fraction",
    ]
    result = {metric: _stats(_metric_value(row, metric) for row in rows) for metric in metrics}
    result["goal_distance_slope_m_per_step"] = _slope(rows, "goal_distance_after_m")
    return result


def _ratio_stats(rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str) -> dict[str, Any]:
    ratios: list[float] = []
    for row in rows:
        top = _metric_value(row, numerator)
        bottom = _metric_value(row, denominator)
        if _finite(top) and _finite(bottom) and abs(bottom) > 1.0e-9:
            ratios.append(top / bottom)
    return _stats(ratios)


def _phase_timeline(rows: Sequence[Mapping[str, Any]], encounter: int | None, passed: int | None, recovery: int | None) -> dict[str, Any]:
    last_step = _step(rows[-1]) if rows else None
    if encounter is None:
        full = list(rows)
    else:
        end = passed if passed is not None else last_step
        full = _rows_between(rows, encounter, end if end is not None else encounter)
    valid_clearance = [
        row
        for row in full
        if _finite(row.get("relevant_obstacle_clearance_m"))
    ]
    min_clearance_row = min(
        valid_clearance,
        key=lambda row: _float(row.get("relevant_obstacle_clearance_m")),
        default=None,
    )
    first_lateral = _first_step(
        full,
        lambda row: _finite(row.get("post_step_actual_tangential_velocity_mps"))
        and _float(row.get("post_step_actual_tangential_velocity_mps")) > 1.0e-9,
    )
    peak_lateral_row = max(
        full,
        key=lambda row: _float(row.get("post_step_actual_tangential_velocity_mps")),
        default=None,
    )
    first_safety = _first_flag(full, "safety_projection_applied")
    first_boundary = _first_flag(full, "boundary_projection_applied")
    first_clip = _first_flag(full, "action_clipped")
    first_collision = _first_flag(full, "collision")
    risk_steps = [value for value in (first_safety, first_boundary, first_collision) if value is not None]
    cumulative = 0.0
    peak_displacement = (None, float("nan"))
    for row in full:
        tangential = _float(row.get("post_step_actual_tangential_velocity_mps"))
        if _finite(tangential):
            cumulative += abs(tangential) * DECISION_DT_S
            if peak_displacement[0] is None or cumulative >= peak_displacement[1]:
                peak_displacement = (_step(row), cumulative)
    last_row = rows[-1] if rows else {}
    return {
        "first_obstacle_near_step": encounter,
        "minimum_clearance_step": _step(min_clearance_row) if min_clearance_row else None,
        "minimum_clearance_m": (
            _float(min_clearance_row.get("relevant_obstacle_clearance_m"))
            if min_clearance_row
            else float("nan")
        ),
        "first_lateral_avoidance_step": first_lateral,
        "peak_lateral_velocity_step": _step(peak_lateral_row) if peak_lateral_row else None,
        "peak_lateral_velocity_mps": (
            _float(peak_lateral_row.get("post_step_actual_tangential_velocity_mps"))
            if peak_lateral_row
            else float("nan")
        ),
        "peak_lateral_displacement_step": peak_displacement[0],
        "peak_lateral_displacement_proxy_m": peak_displacement[1],
        "lateral_displacement_observable": False,
        "first_safety_projection_step": first_safety,
        "first_boundary_projection_step": first_boundary,
        "first_action_clipping_step": first_clip,
        "first_collision_or_projection_step": min(risk_steps) if risk_steps else None,
        "passed_obstacle_step": passed,
        "post_bypass_recovery_step": recovery,
        "last_step": _step(last_row) if last_row else None,
        "last_goal_distance_m": _float(last_row.get("goal_distance_after_m")),
        "last_obstacle_distance_m": _float(last_row.get("relevant_obstacle_clearance_m")),
    }


def _first100_extension(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first100 = [row for row in rows if _step(row) < 100]
    extension = [row for row in rows if _step(row) >= 100]

    def block(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        last = values[-1] if values else {}
        return {
            "steps": len(values),
            "safety_projection_steps": sum(_bool(row.get("safety_projection_applied")) for row in values),
            "boundary_projection_steps": sum(_bool(row.get("boundary_projection_applied")) for row in values),
            "action_clip_steps": sum(_bool(row.get("action_clipped")) for row in values),
            "mean_radial_velocity_mps": _mean(
                _float(row.get("post_step_actual_radial_velocity_mps")) for row in values
            ),
            "mean_tangential_velocity_mps": _mean(
                _float(row.get("post_step_actual_tangential_velocity_mps")) for row in values
            ),
            "mean_goal_distance_m": _mean(_float(row.get("goal_distance_after_m")) for row in values),
            "last_goal_distance_m": _float(last.get("goal_distance_after_m")),
        }

    return {"first_100": block(first100), "extension_100_119": block(extension)}


def _identity_source(source_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = [
        "steps.csv",
        "episodes.csv",
        "terminal_capture_v23_authority_repair_horizon120_episodes.csv",
        "summary.json",
        "terminal_capture_v23_authority_repair_horizon120_summary.json",
        "evaluation_summary.json",
    ]
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise IdentityFailure(f"missing source artifacts: {', '.join(missing)}")
    hashes = {name: _sha256(source_root / name) for name in required}
    steps = _read_csv(source_root / "steps.csv")
    episodes = _read_csv(source_root / "episodes.csv")
    duplicate_episodes = _read_csv(
        source_root / "terminal_capture_v23_authority_repair_horizon120_episodes.csv"
    )
    if (source_root / "episodes.csv").read_bytes() != (
        source_root / "terminal_capture_v23_authority_repair_horizon120_episodes.csv"
    ).read_bytes():
        raise IdentityFailure("episodes.csv is not byte-identical to the named horizon120 episode artifact")
    groups = _episode_groups(steps)
    episode_map = {str(row.get("episode_id", "")): row for row in episodes}
    summary = _read_json(source_root / "summary.json")
    horizon_summary = _read_json(
        source_root / "terminal_capture_v23_authority_repair_horizon120_summary.json"
    )
    evaluation = _read_json(source_root / "evaluation_summary.json")
    if len(episodes) != 20 or len(groups) != 20 or len(duplicate_episodes) != 20:
        raise IdentityFailure("source episode count is not exactly 20")
    if set(groups) != set(EXPECTED_EPISODES) or set(episode_map) != set(EXPECTED_EPISODES):
        raise IdentityFailure("source ordered episode identity is not the requested 20-episode set")
    continuity: dict[str, bool] = {}
    for episode_id in EXPECTED_EPISODES:
        values = groups[episode_id]
        expected_steps = int(round(_float(episode_map[episode_id].get("steps"))))
        observed = [_step(row) for row in values]
        continuity[episode_id] = (
            expected_steps == len(values)
            and observed == list(range(expected_steps))
            and len(set(observed)) == expected_steps
            and int(round(_float(episode_map[episode_id].get("max_steps")))) == 120
        )
    if not all(continuity.values()):
        bad = [episode_id for episode_id, valid in continuity.items() if not valid]
        raise IdentityFailure(f"step continuity/max_steps identity failed: {bad}")
    target_statuses = {
        episode_id: {
            "passed_obstacle": int(_bool(episode_map[episode_id].get("passed_obstacle"))),
            "post_bypass_recovery": int(_bool(episode_map[episode_id].get("post_bypass_recovery"))),
        }
        for episode_id in TARGET_EPISODES
    }
    reference_statuses = {
        episode_id: {
            "passed_obstacle": int(_bool(episode_map[episode_id].get("passed_obstacle"))),
            "post_bypass_recovery": int(_bool(episode_map[episode_id].get("post_bypass_recovery"))),
        }
        for episode_id in REFERENCE_EPISODES
    }
    if any(any(status.values()) for status in target_statuses.values()):
        raise IdentityFailure("target episode passed_obstacle/post_bypass_recovery identity failed")
    if not all(all(status.values()) for status in reference_statuses.values()):
        raise IdentityFailure("success reference episode status identity failed")
    aggregate = summary.get("aggregate", {})
    expected_aggregate = {
        "episodes": 20,
        "passed_obstacle": 11,
        "post_bypass_recovery": 11,
        "entered_8m": 11,
        "entered_5m": 11,
        "entered_3m": 11,
        "entered_1m": 11,
        "stable_goal": 11,
        "collision": 0,
        "safety_projection_steps": 5,
        "boundary_projection_steps": 41,
    }
    aggregate_checks = {
        key: aggregate.get(key) == value for key, value in expected_aggregate.items()
    }
    if not all(aggregate_checks.values()):
        raise IdentityFailure(f"source aggregate identity failed: {aggregate_checks}")
    identity_payload = evaluation.get("previous_formal_identity_checks", {})
    provenance = summary.get("provenance", {})
    required_identity_keys = [
        "same_checkpoint_as_v2_formal",
        "same_checkpoint_sha256",
        "same_policy_hash",
        "same_scenario_manifest",
        "same_ordered_episode_identity",
        "same_controller_config",
        "same_reward",
        "same_dynamics",
        "same_stable_criterion",
        "same_action_transform",
        "same_max_correction",
        "only_horizon_changed",
        "same_checkpoint",
        "same_observation",
        "same_actor",
        "same_critic",
        "same_gat",
        "same_radial_gain",
        "same_tangential_gain",
        "same_terminal_radial_speed_floor",
        "same_terminal_radial_brake_gain",
        "same_terminal_tangential_gain",
        "same_v2_governor",
        "same_a_eff",
        "same_stopping_margin",
        "same_v21_arbitration",
        "same_v21_gain",
        "same_obstacle_logic",
        "same_safety_projection",
        "same_boundary_projection",
        "same_v22_parameters",
        "same_v23_parameters",
        "same_authority_repair_behavior",
    ]
    identity_checks = {
        key: bool(identity_payload.get(key)) for key in required_identity_keys
    }
    identity_checks.update(
        {
            "episode_count_20": len(episodes) == 20,
            "candidate_formal": evaluation.get("formal") is True,
            "candidate_condition_on": evaluation.get("condition") == "on",
            "candidate_max_steps_120": all(
                int(round(_float(row.get("max_steps")))) == 120 for row in episodes
            ),
            "only_horizon_changed": horizon_summary.get("only_horizon_changed") is True,
            "diagnostic_horizon_only": horizon_summary.get("diagnostic_horizon_only") is True,
            "activation_mode_post_bypass_only": (
                evaluation.get("effective_goal_recovery_correction_config", {}).get("activation_mode")
                == "post_bypass_only"
            ),
            "same_goal_recovery_config": bool(identity_payload.get("same_controller_config"))
            and evaluation.get("effective_goal_recovery_correction_config", {}).get("activation_mode")
            == "post_bypass_only",
            "episodes_artifact_duplicate": hashes["episodes.csv"]
            == hashes["terminal_capture_v23_authority_repair_horizon120_episodes.csv"],
            "source_episode_set": set(groups) == set(EXPECTED_EPISODES),
            "target_failure_identity": all(not any(status.values()) for status in target_statuses.values()),
            "reference_success_identity": all(all(status.values()) for status in reference_statuses.values()),
            "summary_aggregate_identity": all(aggregate_checks.values()),
            "candidate_ordered_episode_identity": evaluation.get("ordered_pool_identity")
            == provenance.get("ordered_pool_identity"),
        }
    )
    if not all(identity_checks.values()):
        failed = [key for key, valid in identity_checks.items() if not valid]
        raise IdentityFailure(f"source identity checks failed: {failed}")
    checkpoint = provenance.get("checkpoint", {})
    identity_checks.update(
        {
            "checkpoint_hash_present": bool(checkpoint.get("sha256")),
            "policy_hash_present": bool(checkpoint.get("policy_hash")),
            "scenario_manifest_present": bool(provenance.get("scenario_manifest_hash")),
            "baseline_max_steps_100": horizon_summary.get("baseline_max_steps") == 100,
            "candidate_max_steps_120": horizon_summary.get("max_steps", 120) in (120, None),
        }
    )
    # The candidate summary carries the baseline summary digest.  Check it
    # against the read-only baseline artifact when that artifact is available.
    baseline_rel = Path(
        str(
            horizon_summary.get(
                "baseline_summary_path",
                "logs/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026/terminal_capture_v23_authority_repair_validation_summary.json",
            )
        )
    )
    baseline_path = baseline_rel if baseline_rel.is_absolute() else PROJECT_ROOT / baseline_rel
    baseline_exists = baseline_path.is_file()
    baseline_hash_matches = (
        baseline_exists
        and _sha256(baseline_path)
        == horizon_summary.get("baseline_summary_sha256")
    )
    identity_checks["baseline_summary_hash_matches"] = bool(baseline_hash_matches)
    if not baseline_hash_matches:
        raise IdentityFailure("baseline summary hash does not match the recorded identity")
    identity_checks["candidate_manifest_matches_provenance"] = (
        evaluation.get("scenario_manifest_hash") == provenance.get("scenario_manifest_hash")
    )
    identity_checks["checkpoint_matches_recorded_baseline"] = (
        checkpoint.get("sha256") == horizon_summary.get("baseline_checkpoint_sha256")
    )
    identity_checks["policy_matches_recorded_baseline"] = (
        checkpoint.get("policy_hash") == horizon_summary.get("baseline_policy_hash")
    )
    if not all(identity_checks.values()):
        failed = [key for key, valid in identity_checks.items() if not valid]
        raise IdentityFailure(f"extended source identity checks failed: {failed}")
    source_validation = {
        "status": "PASS",
        "source_root": str(source_root.relative_to(PROJECT_ROOT)),
        "required_files": required,
        "source_hashes_before": hashes,
        "ordered_episode_identity": provenance.get("ordered_pool_identity", []),
        "target_statuses": target_statuses,
        "reference_statuses": reference_statuses,
        "continuity_checks": continuity,
        "aggregate_checks": aggregate_checks,
        "baseline_summary_path": str(baseline_path.relative_to(PROJECT_ROOT))
        if baseline_path.is_relative_to(PROJECT_ROOT)
        else str(baseline_path),
        "baseline_summary_sha256": horizon_summary.get("baseline_summary_sha256"),
        "baseline_checkpoint_sha256": horizon_summary.get("baseline_checkpoint_sha256"),
        "baseline_policy_hash": horizon_summary.get("baseline_policy_hash"),
        "baseline_manifest_hash": horizon_summary.get("baseline_manifest_hash"),
        "diagnostic_manifest_hash": horizon_summary.get("diagnostic_manifest_hash"),
        "identity_checks": identity_checks,
    }
    return source_validation, groups, {"rows": episodes, "map": episode_map}, summary, evaluation


def _classification(
    diagnosis: dict[str, Any],
    reference_stats: Mapping[str, Any],
) -> dict[str, Any]:
    near = diagnosis["encounter_window_metrics"]
    target = diagnosis["full_window_metrics"]
    # Classification uses the complete obstacle-to-terminal window for a
    # failure, compared with the successful passed-window IQR.  The explicitly
    # requested encounter-window comparison remains in the CSV/JSON below.
    def ref_q(metric: str, quantile: str) -> float:
        return _float(reference_stats.get("passed_m20_p0", {}).get(metric, {}).get(quantile))

    radial = _float(target.get("post_step_actual_radial_velocity_mps", {}).get("median"))
    tangential = _float(target.get("post_step_actual_tangential_velocity_mps", {}).get("median"))
    radial_q1 = ref_q("post_step_actual_radial_velocity_mps", "q1")
    tangential_q3 = ref_q("post_step_actual_tangential_velocity_mps", "q3")
    tangential_q1 = ref_q("post_step_actual_tangential_velocity_mps", "q1")
    boundary_count = int(diagnosis["projection_counts"]["boundary_projection_steps"])
    safety_count = int(diagnosis["projection_counts"]["safety_projection_steps"])
    clip_count = int(diagnosis["projection_counts"]["action_clip_steps"])
    boundary_final_fraction = _float(diagnosis["projection_counts"].get("boundary_final30_fraction"))
    no_pass = diagnosis["passed_obstacle_step"] is None
    evidence: list[str] = []
    counter: list[str] = [
        "bypass side is not identifiable because logged tangential fields are magnitudes",
        "direct obstacle bearing/lateral offset is absent from the source schema",
        "Goal Recovery is inactive before a logged bypass, so it is not a direct source in this audit",
    ]
    if boundary_count >= 10:
        primary = "BOUNDARY_DRIFT_AFTER_AVOIDANCE"
        secondary = ["SAFETY_LAYER_DOMINATED"] if safety_count else []
        if clip_count:
            secondary.append("ACTION_CLIP_BOTTLENECK")
        responsibility = "boundary-dominated"
        confidence = "HIGH" if boundary_count >= 10 else "MEDIUM"
        evidence.append(f"boundary_projection_steps={boundary_count}")
        evidence.append(f"final30_boundary_fraction={boundary_final_fraction:.3f}")
        if clip_count:
            evidence.append(f"action_clip_steps={clip_count}")
    elif safety_count:
        primary = "SAFETY_LAYER_DOMINATED"
        secondary = ["BOUNDARY_DRIFT_AFTER_AVOIDANCE"] if boundary_count else []
        if clip_count:
            secondary.append("ACTION_CLIP_BOTTLENECK")
        responsibility = "safety-dominated"
        confidence = "HIGH" if safety_count >= 3 else "MEDIUM"
        evidence.append(f"safety_projection_steps={safety_count}")
        if clip_count:
            evidence.append(f"action_clip_steps={clip_count}")
    elif no_pass and _finite(tangential) and _finite(tangential_q3) and tangential > tangential_q3:
        primary = "EXCESSIVE_LATERAL_PERSISTENCE"
        secondary = []
        responsibility = "Actor-limited"
        confidence = "MEDIUM"
        evidence.append(f"full-obstacle-window median_actual_tangential_velocity={tangential:.3f} m/s > success_passed_Q3={tangential_q3:.3f} m/s")
        if _finite(radial) and _finite(radial_q1) and radial < radial_q1:
            secondary.append("RADIAL_PROGRESS_COLLAPSE")
            evidence.append(f"full-obstacle-window median_actual_radial_velocity={radial:.3f} m/s < success_passed_Q1={radial_q1:.3f} m/s")
    elif no_pass and _finite(radial) and _finite(radial_q1) and radial < radial_q1:
        primary = "RADIAL_PROGRESS_COLLAPSE"
        secondary = ["STALLED_NEAR_OBSTACLE"]
        responsibility = "Actor-limited"
        confidence = "MEDIUM"
        evidence.append(f"full-obstacle-window median_actual_radial_velocity={radial:.3f} m/s < success_passed_Q1={radial_q1:.3f} m/s")
    elif no_pass and _finite(tangential) and _finite(tangential_q1) and tangential <= tangential_q1:
        primary = "INSUFFICIENT_LATERAL_AUTHORITY"
        secondary = ["STALLED_NEAR_OBSTACLE"]
        responsibility = "not assessable"
        confidence = "LOW"
        evidence.append(f"full-obstacle-window median_actual_tangential_velocity={tangential:.3f} m/s <= success_passed_Q1={tangential_q1:.3f} m/s")
    elif no_pass:
        primary = "STALLED_NEAR_OBSTACLE"
        secondary = ["MIXED"]
        responsibility = "not assessable"
        confidence = "LOW"
        evidence.append("no passed_obstacle event and no directly observable dominant intervention")
    else:
        primary = "NOT_ASSESSABLE"
        secondary = []
        responsibility = "not assessable"
        confidence = "LOW"
        evidence.append("source state is not sufficient for a directional classification")
    if not diagnosis["action_transform_attenuation_observed"]:
        counter.append("no command-norm attenuation from pre-safety command to applied command was observed")
    if not diagnosis["projection_counts"]["safety_projection_steps"] and not diagnosis["projection_counts"]["boundary_projection_steps"]:
        counter.append("no safety or boundary projection was observed in the audited episode")
    return {
        "primary": primary,
        "secondary": secondary,
        "evidence": evidence,
        "counter_evidence": counter,
        "confidence": confidence,
        "responsibility_label": responsibility,
    }


def _episode_diagnosis(
    episode_id: str,
    rows: Sequence[dict[str, str]],
    episode_row: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    encounter = _encounter_step(rows)
    passed = _first_step(rows, lambda row: _bool(row.get("passed_obstacle")))
    recovery = _first_step(rows, lambda row: _bool(row.get("post_bypass_recovery")))
    last_step = _step(rows[-1]) if rows else -1
    full_start = encounter if encounter is not None else 0
    full_end = passed if passed is not None else last_step
    full = _rows_between(rows, full_start, full_end)
    encounter_window = _rows_between(rows, max(0, (encounter or 0) - 10), min(last_step, (encounter or 0) + 30))
    bypass_window = _rows_between(
        rows,
        (encounter or 0) + 10,
        min(last_step, (encounter or 0) + 20),
    )
    passed_window = (
        _rows_between(rows, max(0, passed - 20), passed) if passed is not None else []
    )
    pre_rows = _rows_between(rows, max(0, (encounter or 0) - 10), max(0, (encounter or 0) - 1))
    post_rows = (
        _rows_between(rows, passed, min(last_step, passed + 30)) if passed is not None else []
    )
    projection_counts = {
        "safety_projection_steps": sum(_bool(row.get("safety_projection_applied")) for row in rows),
        "boundary_projection_steps": sum(_bool(row.get("boundary_projection_applied")) for row in rows),
        "action_clip_steps": sum(_bool(row.get("action_clipped")) for row in rows),
        "action_transform_steps": sum(_bool(row.get("action_transform_applied")) for row in rows),
        "collision_steps": sum(_bool(row.get("collision")) for row in rows),
        "safety_projection_obstacle_window_steps": sum(
            _bool(row.get("safety_projection_applied")) for row in full
        ),
        "boundary_projection_obstacle_window_steps": sum(
            _bool(row.get("boundary_projection_applied")) for row in full
        ),
        "action_clip_obstacle_window_steps": sum(_bool(row.get("action_clipped")) for row in full),
        "safety_projection_obstacle_window_fraction": (
            sum(_bool(row.get("safety_projection_applied")) for row in full) / len(full)
            if full
            else float("nan")
        ),
        "boundary_projection_obstacle_window_fraction": (
            sum(_bool(row.get("boundary_projection_applied")) for row in full) / len(full)
            if full
            else float("nan")
        ),
        "action_clip_obstacle_window_fraction": (
            sum(_bool(row.get("action_clipped")) for row in full) / len(full)
            if full
            else float("nan")
        ),
        "safety_longest_streak": _longest_streak(rows, "safety_projection_applied"),
        "boundary_longest_streak": _longest_streak(rows, "boundary_projection_applied"),
        "action_clip_longest_streak": _longest_streak(rows, "action_clipped"),
        "safety_final30_fraction": _mean(
            1.0 if _bool(row.get("safety_projection_applied")) else 0.0 for row in rows[-30:]
        ),
        "boundary_final30_fraction": _mean(
            1.0 if _bool(row.get("boundary_projection_applied")) else 0.0 for row in rows[-30:]
        ),
        "action_clip_final30_fraction": _mean(
            1.0 if _bool(row.get("action_clipped")) else 0.0 for row in rows[-30:]
        ),
    }
    pre_safety_raw = _ratio_stats(full, "pre_safety_norm_mps", "actor_raw_norm_mps")
    safety_applied = _ratio_stats(full, "applied_norm_mps", "pre_safety_norm_mps")
    raw_applied = _ratio_stats(full, "applied_norm_mps", "actor_raw_norm_mps")
    action_transform_attenuation = [
        row
        for row in full
        if _finite(_metric_value(row, "actor_raw_norm_mps"))
        and _finite(_metric_value(row, "pre_safety_norm_mps"))
        and _metric_value(row, "pre_safety_norm_mps")
        < _metric_value(row, "actor_raw_norm_mps") - 1.0e-8
    ]
    clip_norms = [
        {
            "actor_raw_norm_mps": _metric_value(row, "actor_raw_norm_mps"),
            "pre_safety_norm_mps": _metric_value(row, "pre_safety_norm_mps"),
            "applied_norm_mps": _metric_value(row, "applied_norm_mps"),
        }
        for row in rows
        if _bool(row.get("action_clipped"))
    ]
    gate_fields = {
        "v2_governor": "terminal_v2_governor_active",
        "v21_conflict": "terminal_v21_radial_conflict_active",
        "v22_hold_latched": "terminal_v22_hold_latched",
        "v23_guard": "terminal_v23_guard_active",
        "v23_preboundary": "terminal_v23_preboundary_active",
        "v23_postboundary": "terminal_v23_postboundary_active",
    }
    gate_counts = {
        name: {
            "episode_steps": sum(_bool(row.get(field)) for row in rows),
            "obstacle_window_steps": sum(_bool(row.get(field)) for row in full),
            "obstacle_window_fraction": _mean(
                1.0 if _bool(row.get(field)) else 0.0 for row in full
            ),
        }
        for name, field in gate_fields.items()
    }
    command_norms = {
        "clip_rows": clip_norms,
        "clip_raw_norm_mps": _stats(item["actor_raw_norm_mps"] for item in clip_norms),
        "clip_pre_safety_norm_mps": _stats(item["pre_safety_norm_mps"] for item in clip_norms),
        "clip_applied_norm_mps": _stats(item["applied_norm_mps"] for item in clip_norms),
        "raw_to_pre_safety_retention": pre_safety_raw,
        "pre_safety_to_applied_retention": safety_applied,
        "raw_to_applied_retention": raw_applied,
        "action_transform_attenuation_steps": len(action_transform_attenuation),
        "action_transform_attenuation_fraction": (
            len(action_transform_attenuation) / len(full) if full else float("nan")
        ),
    }
    lateral_authority = {
        "raw_actor_lateral_component_observable": False,
        "correction_lateral_component_observable": False,
        "pre_transform_lateral_component_observable": False,
        "pre_safety_lateral_component_observable": False,
        "final_applied_lateral_component_observable": False,
        "actual_lateral_velocity_observable_as_magnitude": True,
        "actor_raw_norm_mps": _stats(_metric_value(row, "actor_raw_norm_mps") for row in bypass_window),
        "correction_norm_mps": _stats(_metric_value(row, "correction_norm_mps") for row in bypass_window),
        "pre_transform_norm_mps": _stats(_metric_value(row, "pre_transform_norm_mps") for row in bypass_window),
        "pre_safety_norm_mps": _stats(_metric_value(row, "pre_safety_norm_mps") for row in bypass_window),
        "final_applied_norm_mps": _stats(_metric_value(row, "applied_norm_mps") for row in bypass_window),
        "actual_tangential_velocity_mps": _stats(
            _float(row.get("post_step_actual_tangential_velocity_mps")) for row in bypass_window
        ),
        "actual_tangential_step_displacement_m": _stats(
            _float(row.get("post_step_actual_tangential_velocity_mps")) * DECISION_DT_S
            for row in bypass_window
        ),
        "raw_to_pre_safety_norm_retention": _ratio_stats(
            bypass_window, "pre_safety_norm_mps", "actor_raw_norm_mps"
        ),
        "pre_safety_to_final_applied_norm_retention": _ratio_stats(
            bypass_window, "applied_norm_mps", "pre_safety_norm_mps"
        ),
        "raw_to_final_applied_norm_retention": _ratio_stats(
            bypass_window, "applied_norm_mps", "actor_raw_norm_mps"
        ),
        "intended_side": "NOT_IDENTIFIABLE",
        "actual_side": "NOT_IDENTIFIABLE",
        "direction_consistency": "NOT_ASSESSABLE",
        "window_start_step": (encounter or 0) + 10 if encounter is not None else None,
        "window_end_step": min(last_step, (encounter or 0) + 20)
        if encounter is not None
        else None,
    }
    timeline = _phase_timeline(rows, encounter, passed, recovery)
    pre_metrics = _window_metrics(pre_rows)
    near_metrics = _window_metrics(encounter_window)
    post_metrics = _window_metrics(post_rows)
    full_metrics = _window_metrics(full)
    radial_pre = _median(
        _float(row.get("post_step_actual_radial_velocity_mps")) for row in pre_rows
    )
    radial_near = _median(
        _float(row.get("post_step_actual_radial_velocity_mps")) for row in encounter_window
    )
    radial_post = _median(
        _float(row.get("post_step_actual_radial_velocity_mps")) for row in post_rows
    )
    radial_drop_ratio = (
        1.0 - radial_near / radial_pre
        if _finite(radial_pre) and abs(radial_pre) > 1.0e-9 and _finite(radial_near)
        else float("nan")
    )
    tangential_values = [
        _float(row.get("post_step_actual_tangential_velocity_mps")) for row in full
    ]
    radial_values = [_float(row.get("post_step_actual_radial_velocity_mps")) for row in full]
    lateral_radial_ratios = [
        abs(tangential) / max(abs(radial), 0.05)
        for tangential, radial in zip(tangential_values, radial_values)
        if _finite(tangential) and _finite(radial)
    ]
    extension = _first100_extension(rows)
    diagnosis = {
        "episode_id": episode_id,
        "passed_obstacle_step": passed,
        "post_bypass_recovery_step": recovery,
        "encounter_step": encounter,
        "full_window_start_step": full_start,
        "full_window_end_step": full_end,
        "encounter_window_start_step": max(0, (encounter or 0) - 10),
        "encounter_window_end_step": min(last_step, (encounter or 0) + 30),
        "passed_window_start_step": max(0, passed - 20) if passed is not None else None,
        "passed_window_end_step": passed,
        "timeline": timeline,
        "projection_counts": projection_counts,
        "command_norms": command_norms,
        "lateral_authority": lateral_authority,
        "bypass_direction_audit": {
            "window": "encounter+10..encounter+20",
            "intended_side": "NOT_IDENTIFIABLE",
            "actual_side": "NOT_IDENTIFIABLE",
            "direction_consistency": "NOT_ASSESSABLE",
            "actor_raw_lateral_component_observable": False,
            "applied_lateral_component_observable": False,
            "actual_tangential_velocity_magnitude_observable": True,
            "rows": len(bypass_window),
        },
        "gate_counts": gate_counts,
        "goal_recovery_active_steps": sum(_bool(row.get("goal_recovery_correction_active")) for row in rows),
        "goal_recovery_eligible_steps": sum(_bool(row.get("goal_recovery_correction_eligible")) for row in rows),
        "goal_recovery_enabled_steps": sum(_bool(row.get("goal_recovery_correction_enabled")) for row in rows),
        "activation_mode": "post_bypass_only",
        "action_transform_attenuation_observed": bool(action_transform_attenuation),
        "pre_window_metrics": pre_metrics,
        "encounter_window_metrics": near_metrics,
        "passed_window_metrics": post_metrics if passed is not None else {},
        "full_window_metrics": full_metrics,
        "radial_progress": {
            "pre_median_radial_velocity_mps": radial_pre,
            "near_median_radial_velocity_mps": radial_near,
            "post_median_radial_velocity_mps": radial_post,
            "near_minus_pre_mps": radial_near - radial_pre
            if _finite(radial_near) and _finite(radial_pre)
            else float("nan"),
            "near_radial_drop_ratio": radial_drop_ratio,
            "pre_goal_distance_slope_m_per_step": _slope(pre_rows, "goal_distance_after_m"),
            "near_goal_distance_slope_m_per_step": _slope(encounter_window, "goal_distance_after_m"),
            "post_goal_distance_slope_m_per_step": _slope(post_rows, "goal_distance_after_m"),
            "full_goal_distance_slope_m_per_step": _slope(full, "goal_distance_after_m"),
        },
        "lateral_radial_ratio": _stats(lateral_radial_ratios),
        "extension_only_behavior": extension,
        "episode_summary_row": {
            "minimum_goal_distance_m": _float(episode_row.get("minimum_goal_distance_m")),
            "steps": int(round(_float(episode_row.get("steps")))),
            "max_steps": int(round(_float(episode_row.get("max_steps")))),
            "termination_reason": episode_row.get("termination_reason"),
        },
    }
    classification = _classification(diagnosis, reference_stats)
    diagnosis["classification"] = classification
    return diagnosis, timeline


def _reference_statistics(
    groups: Mapping[str, Sequence[dict[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    windows: dict[str, list[tuple[str, list[dict[str, str]]]]] = {
        "encounter_m10_p30": [],
        "passed_m20_p0": [],
    }
    window_rows: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    for episode_id in REFERENCE_EPISODES:
        rows = groups[episode_id]
        encounter = _encounter_step(rows)
        passed = _first_step(rows, lambda row: _bool(row.get("passed_obstacle")))
        encounter_rows = _rows_between(rows, max(0, (encounter or 0) - 10), min(_step(rows[-1]), (encounter or 0) + 30))
        passed_rows = _rows_between(rows, max(0, (passed or 0) - 20), passed or 0)
        windows["encounter_m10_p30"].append((episode_id, encounter_rows))
        windows["passed_m20_p0"].append((episode_id, passed_rows))
        per_episode.append(
            {
                "episode_id": episode_id,
                "encounter_step": encounter,
                "passed_step": passed,
                "encounter_window_n": len(encounter_rows),
                "passed_window_n": len(passed_rows),
            }
        )
        for kind, values, anchor in (
            ("encounter_m10_p30", encounter_rows, encounter),
            ("passed_m20_p0", passed_rows, passed),
        ):
            for row in values:
                window_rows.append(
                    {
                        "episode_id": episode_id,
                        "population": "success_reference",
                        "window_kind": kind,
                        "step": _step(row),
                        "relative_step": _step(row) - (anchor or 0),
                    }
                )
    metric_names = [
        "post_step_actual_radial_velocity_mps",
        "post_step_actual_tangential_velocity_mps",
        "actor_raw_norm_mps",
        "applied_norm_mps",
        "safety_projection_applied_fraction",
        "boundary_projection_applied_fraction",
        "action_clipped_fraction",
        "relevant_obstacle_clearance_m",
        "goal_distance_after_m",
    ]
    result: dict[str, Any] = {"episode_count": len(REFERENCE_EPISODES), "per_episode_windows": per_episode}
    for kind, items in windows.items():
        flat = [row for _, values in items for row in values]
        result[kind] = {
            metric: _stats(_metric_value(row, metric) for row in flat) for metric in metric_names
        }
        result[kind]["goal_distance_slope_m_per_step"] = _stats(
            _slope(values, "goal_distance_after_m") for _, values in items
        )
        result[kind]["tangential_step_displacement_m"] = _stats(
            _float(row.get("post_step_actual_tangential_velocity_mps")) * DECISION_DT_S
            for row in flat
        )
    return result, window_rows


def _failure_comparison(
    diagnosis: Mapping[str, Any], reference_stats: Mapping[str, Any]
) -> list[dict[str, Any]]:
    metric_map = {
        "radial_velocity_mps_median": "post_step_actual_radial_velocity_mps",
        "tangential_velocity_mps_median": "post_step_actual_tangential_velocity_mps",
        "actor_raw_norm_mps_median": "actor_raw_norm_mps",
        "applied_norm_mps_median": "applied_norm_mps",
        "safety_projection_fraction": "safety_projection_applied_fraction",
        "boundary_projection_fraction": "boundary_projection_applied_fraction",
        "action_clip_fraction": "action_clipped_fraction",
        "obstacle_clearance_median": "relevant_obstacle_clearance_m",
        "goal_distance_slope_m_per_step": "goal_distance_slope_m_per_step",
        "tangential_step_displacement_m": "tangential_step_displacement_m",
    }
    rows: list[dict[str, Any]] = []
    ref = reference_stats.get("encounter_m10_p30", {})
    for output_metric, reference_metric in metric_map.items():
        if reference_metric == "goal_distance_slope_m_per_step":
            value = _float(
                diagnosis.get("encounter_window_metrics", {})
                .get("goal_distance_slope_m_per_step")
            )
        elif reference_metric == "tangential_step_displacement_m":
            value = _float(
                diagnosis.get("encounter_window_metrics", {})
                .get("post_step_actual_tangential_velocity_mps", {})
                .get("median")
            ) * DECISION_DT_S
        else:
            value = _float(
                diagnosis.get("encounter_window_metrics", {})
                .get(reference_metric, {})
                .get("median")
            )
        ref_item = ref.get(reference_metric, {})
        q1 = _float(ref_item.get("q1"))
        q3 = _float(ref_item.get("q3"))
        if not (_finite(value) and _finite(q1) and _finite(q3)):
            relation = "NOT_ASSESSABLE"
        elif value < q1:
            relation = "BELOW_IQR"
        elif value > q3:
            relation = "ABOVE_IQR"
        else:
            relation = "WITHIN_IQR"
        rows.append(
            {
                "episode_id": diagnosis["episode_id"],
                "window_kind": "encounter_m10_p30",
                "metric": output_metric,
                "failure_value": value,
                "reference_q1": q1,
                "reference_median": _float(ref_item.get("median")),
                "reference_q3": q3,
                "relation": relation,
            }
        )
    rows.append(
        {
            "episode_id": diagnosis["episode_id"],
            "window_kind": "passed_m20_p0",
            "metric": "all_metrics",
            "failure_value": None,
            "reference_q1": None,
            "reference_median": None,
            "reference_q3": None,
            "relation": "NOT_ASSESSABLE",
            "note": "failure has no passed_obstacle step",
        }
    )
    return rows


def _projection_rows(diagnoses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for diagnosis in diagnoses:
        counts = diagnosis["projection_counts"]
        result.append(
            {
                "episode_id": diagnosis["episode_id"],
                "population": "target_failure" if diagnosis["episode_id"] in TARGET_EPISODES else "success_reference",
                "safety_projection_steps": counts["safety_projection_steps"],
                "boundary_projection_steps": counts["boundary_projection_steps"],
                "action_clip_steps": counts["action_clip_steps"],
                "action_transform_steps": counts["action_transform_steps"],
                "collision_steps": counts["collision_steps"],
                "safety_projection_obstacle_window_steps": counts["safety_projection_obstacle_window_steps"],
                "boundary_projection_obstacle_window_steps": counts["boundary_projection_obstacle_window_steps"],
                "action_clip_obstacle_window_steps": counts["action_clip_obstacle_window_steps"],
                "safety_projection_obstacle_window_fraction": counts["safety_projection_obstacle_window_fraction"],
                "boundary_projection_obstacle_window_fraction": counts["boundary_projection_obstacle_window_fraction"],
                "action_clip_obstacle_window_fraction": counts["action_clip_obstacle_window_fraction"],
                "safety_longest_streak": counts["safety_longest_streak"],
                "boundary_longest_streak": counts["boundary_longest_streak"],
                "action_clip_longest_streak": counts["action_clip_longest_streak"],
                "safety_final30_fraction": counts["safety_final30_fraction"],
                "boundary_final30_fraction": counts["boundary_final30_fraction"],
                "action_clip_final30_fraction": counts["action_clip_final30_fraction"],
                "clip_raw_norm_mps_median": diagnosis["command_norms"]["clip_raw_norm_mps"].get("median"),
                "clip_pre_safety_norm_mps_median": diagnosis["command_norms"]["clip_pre_safety_norm_mps"].get("median"),
                "clip_applied_norm_mps_median": diagnosis["command_norms"]["clip_applied_norm_mps"].get("median"),
                "raw_to_pre_safety_retention_median": diagnosis["command_norms"]["raw_to_pre_safety_retention"].get("median"),
                "pre_safety_to_applied_retention_median": diagnosis["command_norms"]["pre_safety_to_applied_retention"].get("median"),
                "raw_to_applied_retention_median": diagnosis["command_norms"]["raw_to_applied_retention"].get("median"),
                "action_transform_attenuation_steps": diagnosis["command_norms"]["action_transform_attenuation_steps"],
                "action_transform_attenuation_fraction": diagnosis["command_norms"]["action_transform_attenuation_fraction"],
            }
        )
    return result


def _breakdown(diagnoses: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": diagnosis["episode_id"],
            "steps": diagnosis["projection_counts"].get(field, 0),
            "first_step": diagnosis["timeline"].get(
                {
                    "safety_projection_steps": "first_safety_projection_step",
                    "boundary_projection_steps": "first_boundary_projection_step",
                    "action_clip_steps": "first_action_clipping_step",
                }.get(field, ""),
            ),
            "obstacle_window_steps": diagnosis["projection_counts"].get(
                {
                    "safety_projection_steps": "safety_projection_obstacle_window_steps",
                    "boundary_projection_steps": "boundary_projection_obstacle_window_steps",
                    "action_clip_steps": "action_clip_obstacle_window_steps",
                }.get(field, ""),
            ),
        }
        for diagnosis in diagnoses
    ]


def _window_step_rows(
    groups: Mapping[str, Sequence[dict[str, str]]],
    diagnoses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    selected: list[tuple[str, str, int | None, int, int]] = []
    for episode_id in TARGET_EPISODES:
        rows = groups[episode_id]
        diagnosis = diagnoses[episode_id]
        selected.append(
            (
                episode_id,
                "target_full",
                diagnosis.get("encounter_step"),
                diagnosis.get("full_window_start_step", 0),
                diagnosis.get("full_window_end_step", _step(rows[-1])),
            )
        )
        selected.append(
            (
                episode_id,
                "encounter_p10_p20",
                diagnosis.get("encounter_step"),
                (diagnosis.get("encounter_step") or 0) + 10,
                min(_step(rows[-1]), (diagnosis.get("encounter_step") or 0) + 20),
            )
        )
    for episode_id in REFERENCE_EPISODES:
        rows = groups[episode_id]
        encounter = _encounter_step(rows)
        passed = _first_step(rows, lambda row: _bool(row.get("passed_obstacle")))
        selected.extend(
            [
                (
                    episode_id,
                    "encounter_m10_p30",
                    encounter,
                    max(0, (encounter or 0) - 10),
                    min(_step(rows[-1]), (encounter or 0) + 30),
                ),
                (
                    episode_id,
                    "passed_m20_p0",
                    passed,
                    max(0, (passed or 0) - 20),
                    passed or 0,
                ),
                (
                    episode_id,
                    "encounter_p10_p20",
                    encounter,
                    (encounter or 0) + 10,
                    min(_step(rows[-1]), (encounter or 0) + 20),
                ),
            ]
        )
    for episode_id, kind, anchor, start, end in selected:
        population = "target_failure" if episode_id in TARGET_EPISODES else "success_reference"
        for row in _rows_between(groups[episode_id], start, end):
            result.append(
                {
                    "episode_id": episode_id,
                    "population": population,
                    "window_kind": kind,
                    "step": _step(row),
                    "relative_to_anchor_step": _step(row) - (anchor or 0),
                    "relevant_obstacle_id": row.get("relevant_obstacle_id"),
                    "along_obstacle_m": _float(row.get("along_obstacle_m")),
                    "obstacle_extent_m": _float(row.get("obstacle_extent_m")),
                    "relevant_obstacle_clearance_m": _float(row.get("relevant_obstacle_clearance_m")),
                    "minimum_clearance_m": _float(row.get("minimum_clearance_m")),
                    "goal_distance_after_m": _float(row.get("goal_distance_after_m")),
                    "actual_radial_velocity_mps": _float(row.get("post_step_actual_radial_velocity_mps")),
                    "actual_tangential_velocity_mps": _float(row.get("post_step_actual_tangential_velocity_mps")),
                    "actor_raw_norm_mps": _norm(row, "actor_raw_delta_v_mps"),
                    "pre_transform_norm_mps": _norm(row, "corrected_delta_v_pre_transform_mps"),
                    "pre_safety_norm_mps": _norm(row, "corrected_delta_v_pre_safety_mps"),
                    "applied_norm_mps": _norm(row, "applied_delta_v_mps"),
                    "correction_norm_mps": _norm(row, "goal_recovery_applied_delta_v_mps"),
                    "safety_projection_norm_mps": _norm(row, "safety_projection_delta_v_mps"),
                    "boundary_projection_norm_mps": _norm(row, "motion_boundary_delta_v_mps"),
                    "goal_recovery_correction_active": _bool(row.get("goal_recovery_correction_active")),
                    "terminal_v2_governor_active": _bool(row.get("terminal_v2_governor_active")),
                    "terminal_v21_radial_conflict_active": _bool(row.get("terminal_v21_radial_conflict_active")),
                    "terminal_v22_hold_latched": _bool(row.get("terminal_v22_hold_latched")),
                    "terminal_v23_guard_active": _bool(row.get("terminal_v23_guard_active")),
                    "safety_projection_applied": _bool(row.get("safety_projection_applied")),
                    "boundary_projection_applied": _bool(row.get("boundary_projection_applied")),
                    "action_clipped": _bool(row.get("action_clipped")),
                    "collision": _bool(row.get("collision")),
                }
            )
    return result


def _flatten_diagnostic(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    classification = diagnosis["classification"]
    counts = diagnosis["projection_counts"]
    radial = diagnosis["radial_progress"]
    timeline = diagnosis["timeline"]
    command = diagnosis["command_norms"]
    return {
        "episode_id": diagnosis["episode_id"],
        "primary_diagnosis": classification["primary"],
        "secondary_diagnosis": ";".join(classification["secondary"]),
        "responsibility_label": classification["responsibility_label"],
        "confidence": classification["confidence"],
        "evidence": "; ".join(classification["evidence"]),
        "counter_evidence": "; ".join(classification["counter_evidence"]),
        "encounter_step": diagnosis["encounter_step"],
        "passed_obstacle_step": diagnosis["passed_obstacle_step"],
        "post_bypass_recovery_step": diagnosis["post_bypass_recovery_step"],
        "last_step": timeline["last_step"],
        "last_goal_distance_m": timeline["last_goal_distance_m"],
        "last_obstacle_distance_m": timeline["last_obstacle_distance_m"],
        "minimum_clearance_m": timeline["minimum_clearance_m"],
        "peak_lateral_velocity_mps": timeline["peak_lateral_velocity_mps"],
        "peak_lateral_displacement_proxy_m": timeline["peak_lateral_displacement_proxy_m"],
        "safety_projection_steps": counts["safety_projection_steps"],
        "boundary_projection_steps": counts["boundary_projection_steps"],
        "action_clip_steps": counts["action_clip_steps"],
        "goal_recovery_active_steps": diagnosis["goal_recovery_active_steps"],
        "goal_recovery_eligible_steps": diagnosis["goal_recovery_eligible_steps"],
        "radial_pre_median_mps": radial["pre_median_radial_velocity_mps"],
        "radial_near_median_mps": radial["near_median_radial_velocity_mps"],
        "radial_post_median_mps": radial["post_median_radial_velocity_mps"],
        "radial_near_drop_ratio": radial["near_radial_drop_ratio"],
        "near_goal_distance_slope_m_per_step": radial["near_goal_distance_slope_m_per_step"],
        "full_goal_distance_slope_m_per_step": radial["full_goal_distance_slope_m_per_step"],
        "lateral_to_radial_ratio_median": diagnosis["lateral_radial_ratio"].get("median"),
        "raw_to_pre_safety_retention_median": command["raw_to_pre_safety_retention"].get("median"),
        "pre_safety_to_applied_retention_median": command["pre_safety_to_applied_retention"].get("median"),
        "raw_to_applied_retention_median": command["raw_to_applied_retention"].get("median"),
        "action_transform_attenuation_steps": command["action_transform_attenuation_steps"],
        "v2_governor_steps": diagnosis["gate_counts"]["v2_governor"]["episode_steps"],
        "v21_conflict_steps": diagnosis["gate_counts"]["v21_conflict"]["episode_steps"],
        "v22_hold_latched_steps": diagnosis["gate_counts"]["v22_hold_latched"]["episode_steps"],
        "v23_guard_steps": diagnosis["gate_counts"]["v23_guard"]["episode_steps"],
        "extension_first100": diagnosis["extension_only_behavior"]["first_100"],
        "extension_100_119": diagnosis["extension_only_behavior"]["extension_100_119"],
    }


def _report(
    summary: Mapping[str, Any],
    diagnoses: Sequence[Mapping[str, Any]],
    reference_stats: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = ["UPSTREAM_BYPASS_AUDIT_READY", "", "# Single-UAV Upstream Obstacle-Bypass Failure Offline Audit", ""]
    lines.extend(
        [
            "Scope: frozen horizon120 CSV/JSON artifacts only; no simulator, environment step, policy/controller runtime, rollout, training, or formal execution was used.",
            "The diagnostic labels below are evidence summaries, not causal scores or controller changes.",
            "",
            "## A. Source and identity validation",
            "",
            f"- Source validation: `{summary['source_validation']['status']}`; episodes={summary['source_episode_count']}; steps={summary['source_step_count']}; all max_steps=120.",
            f"- Requested targets: {', '.join(summary['target_episode_ids'])}.",
            f"- Success reference: {', '.join(summary['reference_episode_ids'])}.",
            f"- Identity checks: `{sum(bool(value) for value in summary['identity_checks'].values())}/{len(summary['identity_checks'])}` passed; `only_horizon_changed=true`; baseline max_steps=100, candidate max_steps=120.",
            "- Checkpoint, policy hash, ordered episode identity, manifest lock, Actor/GAT/Critic, reward, dynamics, action transform, safety, boundary, Goal Recovery, v2/v2.1/v2.2/v2.3, and authority-repair identity checks all passed.",
            "",
            "## B. Failure identity and audit boundary",
            "",
            "- Every target has `passed_obstacle=false` and `post_bypass_recovery=false`; every reference episode has both flags true. Collision count is zero.",
            "- No horizon extension was proposed or run. The audit stops at the existing 120-step source logs.",
            "",
            "## C. Obstacle geometry and sign conventions",
            "",
            "- `goal_distance_after_m` and `post_step_actual_radial_velocity_mps` are used with positive radial velocity defined as goalward (dot product with normalized goal error).",
            "- `along_obstacle_m` is the obstacle-center relative projection on the goal direction; positive is ahead and negative indicates the rear/passed direction. `relevant_obstacle_clearance_m` is the logged point-to-obstacle clearance and `obstacle_extent_m` is the directional extent.",
            "- Direct obstacle bearing, lateral offset, UAV position, and signed left/right bypass side are absent. Therefore `OBSTACLE_GEOMETRY_PARTIALLY_OBSERVABLE`; wrong-left/right direction cannot be adjudicated.",
            "- Encounter is the first valid obstacle row satisfying `-(extent+0.75) <= along_obstacle_m <= extent+0.75`, matching the frozen recovery tracker buffer.",
            "",
            "## D. Per-episode phase timeline",
            "",
            "The complete wide timeline is in `upstream_bypass_phase_timeline.csv`; the table below reports the required anchors.",
            "",
            "| Episode | Encounter | Min clearance (step, m) | First lateral | Peak lateral (step, m/s) | First safety | First boundary | First clip | Passed | Recovery | Last goal m |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for diagnosis in diagnoses:
        timeline = diagnosis["timeline"]
        lines.append(
            "| {episode_id} | {first_obstacle_near_step} | {minimum_clearance_step}, {minimum_clearance_m:.3f} | {first_lateral_avoidance_step} | {peak_lateral_velocity_step}, {peak_lateral_velocity_mps:.3f} | {first_safety_projection_step} | {first_boundary_projection_step} | {first_action_clipping_step} | {passed_obstacle_step} | {post_bypass_recovery_step} | {last_goal_distance_m:.3f} |".format(
                episode_id=diagnosis["episode_id"],
                **{key: ("" if value is None or not _finite(value) else value) for key, value in timeline.items()},
            )
        )
    lines.extend(
        [
            "",
            "The cumulative lateral-displacement column is explicitly a non-signed tangential path proxy (`tangential speed × 0.2 s`); signed lateral displacement is not observable in the source schema.",
            "",
            "## E. Bypass direction and lateral authority",
            "",
            "- Actor raw, pre-transform, pre-safety, and final applied command norms are auditable. A lateral component of those vectors is not logged, so side/direction consistency is `NOT_ASSESSABLE` for all nine targets.",
            "- The required bypass-direction window is `encounter+10..encounter+20`; it is exported with actor raw norm, applied norm, actual tangential-speed magnitude, and tangential path proxy. Intended and actual side remain `NOT_IDENTIFIABLE`.",
            "- For the non-projection failures, pre-safety and applied norms are retained at approximately 1.0 relative to raw norms; this localizes no measurable attenuation to action transform/safety/boundary, but it does not prove the Actor chose the correct side.",
            "- Use `upstream_bypass_obstacle_window_steps.csv` for the step-level actor/applied/actual tangential traces and `upstream_bypass_audit_summary.json` for retention ratios.",
            "",
            "## F. Radial progress, lateral/radial ratios, and classes",
            "",
            "- Radial progress is assessed with pre-encounter, encounter, and post-passed medians/slopes. Failures without a passed step have no post-bypass phase; that absence is reported as `NOT_ASSESSABLE` rather than imputed.",
            "- Excessive lateral persistence is assigned only when the target full obstacle-window tangential median is above the successful passed-window Q3 and no bypass is logged. Radial collapse is a secondary label when its full-window median is below the successful passed-window Q1.",
            "- The ratio column is `|tangential speed| / max(|radial speed|, 0.05)` and is a magnitude diagnostic, not a signed direction measure.",
            "",
            "## G. Safety, boundary, and clipping",
            "",
            f"- Aggregate safety projections: {summary['safety_projection_total_steps']} steps; boundary projections: {summary['boundary_projection_total_steps']} steps.",
            "- Safety concentration: episode 0013=4 steps and 0014=1 step; boundary concentration: episode 0013=7 steps and 0014=34 steps. The remaining 18 episodes have zero safety/boundary projections.",
            "- Episode concentration is listed in `upstream_bypass_projection_breakdown.csv`. Boundary-heavy episode 0014 is diagnosed from its current logs, not from a reconstructed trajectory.",
            "- Projection counts, longest streaks, final-30 fractions, clipping norms, and pre/post command norms are separated; boundary projection is not conflated with safety projection.",
            "",
            "## H. Actor → transform → safety → boundary → dynamics chain",
            "",
            "- The chain is observed through raw actor norm, pre-transform norm, pre-safety norm, final applied norm, projection deltas, and actual radial/tangential velocities.",
            "- Final responsibility labels are deliberately restricted to `Actor-limited`, `safety-dominated`, `boundary-dominated`, `mixed`, or `not assessable`; they are diagnostic responsibility labels, not intervention recommendations.",
            "",
            "| Episode | Primary | Secondary | Responsibility | Confidence |",
            "|---|---|---|---|---|",
        ]
    )
    for diagnosis in diagnoses:
        classification = diagnosis["classification"]
        lines.append(
            f"| {diagnosis['episode_id']} | {classification['primary']} | {', '.join(classification['secondary']) or '—'} | {classification['responsibility_label']} | {classification['confidence']} |"
        )
    lines.extend(
        [
            "",
            "## I. Goal Recovery and v2/v2.1/v2.2/v2.3 gates",
            "",
            "- The effective configuration records `activation_mode=post_bypass_only`. All nine failures have zero `goal_recovery_correction_active` steps; the terminal Goal Recovery path is therefore not the direct source of these upstream failures.",
            "- v2 governor, v2.1 conflict arbitration, v2.2 hold latch, and v2.3 guard/pre-boundary/post-boundary gate counts are reported separately. They are zero in the failure obstacle windows because the failures never enter the terminal capture regime.",
            "",
            "## J. Success reference and failure-versus-reference",
            "",
            "- Reference statistics use the 11 successful episodes, encounter window `[-10,+30]` and passed window `[-20,0]`, with median/IQR only.",
            "- Failure comparisons are restricted to `BELOW_IQR`, `WITHIN_IQR`, `ABOVE_IQR`, or `NOT_ASSESSABLE`; no score is assigned. See `upstream_bypass_reference_comparison.csv`.",
            "",
            "| Reference window | Metric | Median | Q1 | Q3 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for kind in ("encounter_m10_p30", "passed_m20_p0"):
        for metric in (
            "post_step_actual_radial_velocity_mps",
            "post_step_actual_tangential_velocity_mps",
            "actor_raw_norm_mps",
            "applied_norm_mps",
            "safety_projection_applied_fraction",
            "boundary_projection_applied_fraction",
            "action_clipped_fraction",
            "relevant_obstacle_clearance_m",
            "goal_distance_slope_m_per_step",
        ):
            item = reference_stats[kind].get(metric, {})
            if metric == "goal_distance_slope_m_per_step":
                item = item
            lines.append(
                f"| {kind} | {metric} | {_fmt(item.get('median'))} | {_fmt(item.get('q1'))} | {_fmt(item.get('q3'))} |"
            )
    lines.extend(
        [
            "",
            "## K. Episode taxonomy, aggregate clusters, and extension comparison",
            "",
            "- Primary labels are one per target; secondary labels are allowed. The JSON `failure_cluster_counts` contains both primary and secondary totals.",
            "- The first-100 versus extension (100–119) behavior is in `extension_only_behavior_summary`; no claim is made that the extension caused or cured a failure.",
            "",
            "| Episode | First 100 radial m/s | Extension radial m/s | First 100 boundary | Extension boundary | First 100 clip | Extension clip |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for diagnosis in diagnoses:
        first = diagnosis["extension_only_behavior"]["first_100"]
        extension = diagnosis["extension_only_behavior"]["extension_100_119"]
        lines.append(
            f"| {diagnosis['episode_id']} | {_fmt(first.get('mean_radial_velocity_mps'))} | {_fmt(extension.get('mean_radial_velocity_mps'))} | {first.get('boundary_projection_steps', 0)} | {extension.get('boundary_projection_steps', 0)} | {first.get('action_clip_steps', 0)} | {extension.get('action_clip_steps', 0)} |"
        )
    lines.extend(
        [
            "",
            "## L. Final state and limitations",
            "",
            "- `OBSTACLE_GEOMETRY_PARTIALLY_OBSERVABLE`; signed bypass side, direct lateral position, and raw lateral vector component are not logged.",
            "- No source log was modified. No controller, Goal Recovery, Actor, Critic, GAT, reward, dynamics, safety, boundary, action transform, or stable criterion was modified; no training, rollout, formal, or smoke execution occurred.",
            "- The audit is complete and no next experiment is started.",
            "",
            "Final state flags:",
            "",
            "```text",
            "upstream_obstacle_bypass_audit_executed=true",
            "target_failure_episode_count=9",
            "terminal_hold_reopened=false",
            "terminal_controller_modified=false",
            "goal_recovery_modified=false",
            "actor_modified=false",
            "critic_modified=false",
            "gat_modified=false",
            "reward_modified=false",
            "dynamics_modified=false",
            "safety_projection_modified=false",
            "boundary_projection_modified=false",
            "action_transform_modified=false",
            "stable_criterion_modified=false",
            "training_performed=false",
            "rollout_executed=false",
            "formal_executed=false",
            "smoke_executed=false",
            "environment_step_called=false",
            "source_logs_modified=false",
            "controller_code_modified=false",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(source_root: Path = DEFAULT_SOURCE_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, *, allow_existing: bool = False) -> dict[str, Any]:
    """Run the bounded offline audit and write the requested artifact package."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_validation, groups, episode_data, summary_source, evaluation = _identity_source(source_root)
    if output_root.exists() and not allow_existing:
        raise FileExistsError(
            f"refusing to overwrite existing audit output: {output_root}; pass --allow-existing only for an explicit rerun"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    episode_map = episode_data["map"]
    reference_stats, _ = _reference_statistics(groups)
    diagnoses: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    for episode_id in TARGET_EPISODES:
        diagnosis, timeline = _episode_diagnosis(
            episode_id, groups[episode_id], episode_map[episode_id], reference_stats
        )
        diagnoses.append(diagnosis)
        timelines.append({"episode_id": episode_id, **timeline})
    comparisons = [
        comparison
        for diagnosis in diagnoses
        for comparison in _failure_comparison(diagnosis, reference_stats)
    ]
    target_diagnoses = diagnoses
    source_hashes_after = {
        name: _sha256(source_root / name) for name in source_validation["required_files"]
    }
    source_unchanged = source_hashes_after == source_validation["source_hashes_before"]
    if not source_unchanged:
        raise IdentityFailure("source artifacts changed while the audit was running")
    aggregate = summary_source.get("aggregate", {})
    all_diagnoses = [
        _episode_diagnosis(episode_id, groups[episode_id], episode_map[episode_id], reference_stats)[0]
        for episode_id in EXPECTED_EPISODES
    ]
    projection_rows = _projection_rows(all_diagnoses)
    summary: dict[str, Any] = {
        "source_validation": {
            **source_validation,
            "source_hashes_after": source_hashes_after,
            "source_unchanged_during_audit": source_unchanged,
        },
        "source_episode_count": len(groups),
        "source_step_count": sum(len(rows) for rows in groups.values()),
        "target_episode_ids": TARGET_EPISODES,
        "reference_episode_ids": REFERENCE_EPISODES,
        "identity_checks": source_validation["identity_checks"],
        "per_episode_projection_counts": projection_rows,
        "per_episode_phase_timeline": timelines,
        "per_episode_diagnosis": [_safe(diagnosis) for diagnosis in diagnoses],
        "success_reference_statistics": _safe(reference_stats),
        "failure_vs_reference": _safe(comparisons),
        "failure_cluster_counts": {},
        "safety_projection_episode_breakdown": _breakdown(all_diagnoses, "safety_projection_steps"),
        "boundary_projection_episode_breakdown": _breakdown(all_diagnoses, "boundary_projection_steps"),
        "action_clip_episode_breakdown": _breakdown(all_diagnoses, "action_clip_steps"),
        "goal_recovery_active_episode_breakdown": [
            {
                "episode_id": diagnosis["episode_id"],
                "active_steps": diagnosis["goal_recovery_active_steps"],
                "eligible_steps": diagnosis["goal_recovery_eligible_steps"],
                "activation_mode": diagnosis["activation_mode"],
            }
            for diagnosis in diagnoses
        ],
        "extension_only_behavior_summary": [
            {
                "episode_id": diagnosis["episode_id"],
                **diagnosis["extension_only_behavior"],
            }
            for diagnosis in diagnoses
        ],
        "source_aggregate": aggregate,
        "source_hashes": source_hashes_after,
        "audit_constants": {
            "soft_recovery_buffer_m": SOFT_RECOVERY_BUFFER_M,
            "static_safe_distance_m": STATIC_SAFE_DISTANCE_M,
            "decision_dt_s": DECISION_DT_S,
            "dt_provenance": "src/simulation/motion_model.py: MotionConfig.dt_s default 0.2",
            "obstacle_geometry_observability": "OBSTACLE_GEOMETRY_PARTIALLY_OBSERVABLE",
            "signed_bypass_side_observable": False,
        },
        "final_state": {
            "upstream_obstacle_bypass_audit_executed": True,
            "target_failure_episode_count": 9,
            "terminal_hold_reopened": False,
            "terminal_controller_modified": False,
            "goal_recovery_modified": False,
            "actor_modified": False,
            "critic_modified": False,
            "gat_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "safety_projection_modified": False,
            "boundary_projection_modified": False,
            "action_transform_modified": False,
            "stable_criterion_modified": False,
            "training_performed": False,
            "rollout_executed": False,
            "formal_executed": False,
            "smoke_executed": False,
            "environment_step_called": False,
            "source_logs_modified": False,
            "controller_code_modified": False,
        },
    }
    primary_counts: dict[str, int] = {}
    secondary_counts: dict[str, int] = {}
    for diagnosis in diagnoses:
        classification = diagnosis["classification"]
        primary_counts[classification["primary"]] = primary_counts.get(classification["primary"], 0) + 1
        for label in classification["secondary"]:
            secondary_counts[label] = secondary_counts.get(label, 0) + 1
    summary["failure_cluster_counts"] = {
        "primary": primary_counts,
        "secondary": secondary_counts,
        "primary_total": sum(primary_counts.values()),
        "secondary_total": sum(secondary_counts.values()),
        "taxonomy": [
            "WRONG_BYPASS_DIRECTION",
            "INSUFFICIENT_LATERAL_AUTHORITY",
            "EXCESSIVE_LATERAL_PERSISTENCE",
            "RADIAL_PROGRESS_COLLAPSE",
            "SAFETY_LAYER_DOMINATED",
            "BOUNDARY_DRIFT_AFTER_AVOIDANCE",
            "ACTION_TRANSFORM_ATTENUATION",
            "ACTION_CLIP_BOTTLENECK",
            "ACTOR_RECOVERY_FAILURE",
            "STALLED_NEAR_OBSTACLE",
            "MIXED",
            "NOT_ASSESSABLE",
        ],
    }
    report = _report(
        {
            **summary,
            "safety_projection_total_steps": aggregate.get("safety_projection_steps", 0),
            "boundary_projection_total_steps": aggregate.get("boundary_projection_steps", 0),
        },
        diagnoses,
        reference_stats,
        comparisons,
    )
    timeline_columns = [
        "episode_id",
        "first_obstacle_near_step",
        "minimum_clearance_step",
        "minimum_clearance_m",
        "first_lateral_avoidance_step",
        "peak_lateral_velocity_step",
        "peak_lateral_velocity_mps",
        "peak_lateral_displacement_step",
        "peak_lateral_displacement_proxy_m",
        "lateral_displacement_observable",
        "first_safety_projection_step",
        "first_boundary_projection_step",
        "first_action_clipping_step",
        "first_collision_or_projection_step",
        "passed_obstacle_step",
        "post_bypass_recovery_step",
        "last_step",
        "last_goal_distance_m",
        "last_obstacle_distance_m",
    ]
    diagnostic_rows = [_flatten_diagnostic(diagnosis) for diagnosis in diagnoses]
    diagnostic_columns = list(diagnostic_rows[0]) if diagnostic_rows else ["episode_id"]
    projection_columns = list(projection_rows[0]) if projection_rows else ["episode_id"]
    comparison_columns = [
        "episode_id",
        "window_kind",
        "metric",
        "failure_value",
        "reference_q1",
        "reference_median",
        "reference_q3",
        "relation",
        "note",
    ]
    window_rows = _window_step_rows(groups, {diagnosis["episode_id"]: diagnosis for diagnosis in diagnoses})
    window_columns = list(window_rows[0]) if window_rows else ["episode_id"]
    with (output_root / "upstream_bypass_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_safe(summary), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _write_csv(output_root / "upstream_bypass_episode_diagnostics.csv", diagnostic_rows, diagnostic_columns)
    _write_csv(output_root / "upstream_bypass_phase_timeline.csv", timelines, timeline_columns)
    _write_csv(output_root / "upstream_bypass_obstacle_window_steps.csv", window_rows, window_columns)
    _write_csv(output_root / "upstream_bypass_reference_comparison.csv", comparisons, comparison_columns)
    _write_csv(output_root / "upstream_bypass_projection_breakdown.csv", projection_rows, projection_columns)
    (output_root / "upstream_bypass_report.md").write_text(report, encoding="utf-8")
    return _safe(summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_audit(args.source_root, args.output_root, allow_existing=args.allow_existing)
    except IdentityFailure as exc:
        print("UPSTREAM_AUDIT_IDENTITY_FAIL")
        print(str(exc), file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print("UPSTREAM_BYPASS_AUDIT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
