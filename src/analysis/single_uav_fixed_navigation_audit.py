"""Independent, offline audit for the frozen fixed-navigation Stage-1 run.

The evaluator writes truth and command telemetry to CSV files.  This module
only reads those files and the locked manifest.  In particular it does not
construct an environment, advance a step, import a learner, or load a model.
The small :func:`audit_row` and :func:`audit_group` functions are deliberately
pure so that semantic mutations can be tested without a simulator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..control.fixed_navigation import FixedNavigationConfig, FixedWaypointNavigator, fixed_navigation_command
from ..simulation.obstacle_model import BoxObstacle, segment_intersects_obstacle, segment_obstacle_clearance

ROOT = Path(__file__).resolve().parents[2]
STEM = "single_uav_fixed_navigation_v1_seed2026_20261011"
DEFAULT_LOG_ROOT = ROOT / "logs" / "preliminary" / STEM
DEFAULT_DATA_ROOT = ROOT / "data" / "synthetic" / STEM
DEFAULT_CONFIG = ROOT / "data" / "configs" / f"{STEM}.json"
FROZEN_STATISTICS_SHA256 = "a45d8ed0c2aba6943ba4d3cf445cad7e5290d0f47dc86055478d87d6135c8b4e"
UID = "uav_00"
DT_S = 0.2
STRICT_DISTANCE_M = 1.0
STRICT_SPEED_MPS = 0.5
STRICT_HOLD_STEPS = 5
STRICT_PASS_BUFFER_M = 0.75
STRICT_CLEARANCE_M = 2.5
SEMANTIC_TOLERANCE = 1e-6
PHASE_FILES = {
    "development": "development_steps.csv",
    "normal_evaluation": "normal_evaluation_steps.csv",
    "paired": "paired_steps.csv",
    "obstacle": "obstacle_steps.csv",
}
PHASE_CASE_COUNTS = {"development": 6, "normal_evaluation": 36, "paired": 4, "obstacle": 4}
ASSIST_FIELDS = ("pass_commit", "lateral_assist", "radial_assist", "goal_recovery", "terminal_capture")
VECTOR_FIELDS = (
    "ekf_position_pre_m", "ekf_velocity_pre_mps", "reference_position_m",
    "desired_velocity_mps", "requested_acceleration_mps2", "raw_delta_v_mps",
    "axis_limited_delta_v_mps", "requested_delta_v_mps", "env_input_command_mps",
    "reported_applied_delta_v_mps", "actual_delta_v_mps", "execution_correction_delta_v_mps",
    "actual_acceleration_mps2", "true_position_pre_m", "true_velocity_pre_mps",
    "true_position_post_m", "true_velocity_post_mps", "nominal_position_after_command_m",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _native(value: Any) -> Any:
    """Decode evaluator CSV JSON cells while accepting native test values."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    if text[0] in "[{\"":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _float(value: Any, default: float = float("nan")) -> float:
    value = _native(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = -1) -> int:
    value = _native(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    value = _native(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return default


def _vector(value: Any) -> np.ndarray:
    value = _native(value)
    try:
        array = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return np.full(3, np.nan)
    return array


def _finite_vector(value: Any) -> bool:
    array = _vector(value)
    return bool(np.isfinite(array).all())


def _close(left: Any, right: Any, atol: float = 1e-6) -> bool:
    try:
        return bool(np.allclose(np.asarray(_native(left), dtype=float), np.asarray(_native(right), dtype=float), rtol=0.0, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _err(errors: list[str], name: str, condition: bool) -> None:
    if not condition:
        errors.append(name)


def _case_goal(case: Mapping[str, Any]) -> np.ndarray:
    return _vector(case.get("scenario", {}).get("goals", {}).get(UID))


def _physical_box(case: Mapping[str, Any]) -> BoxObstacle | None:
    specs = case.get("scenario", {}).get("obstacle_specs", [])
    physical = [spec for spec in specs if bool(spec.get("physical_collision_enabled", True))]
    if not physical:
        return None
    spec = physical[0]
    return BoxObstacle(
        obstacle_id=str(spec.get("obstacle_id", case.get("scenario_id", "obstacle"))),
        center_m=_vector(spec.get("center_m")),
        size_m=_vector(spec.get("size_m")),
        velocity_mps=_vector(spec.get("velocity_mps", (0.0, 0.0, 0.0))),
        obstacle_type=str(spec.get("obstacle_type", "building")),
        active=bool(spec.get("active", True)),
        physical_collision_enabled=bool(spec.get("physical_collision_enabled", True)),
    )


def _obstacle_projection_bounds(start: np.ndarray, direction: np.ndarray, obstacle: BoxObstacle) -> tuple[float, float]:
    corners = np.asarray(
        [[x, y, z] for x in (obstacle.minimum_m[0], obstacle.maximum_m[0])
         for y in (obstacle.minimum_m[1], obstacle.maximum_m[1])
         for z in (obstacle.minimum_m[2], obstacle.maximum_m[2])], dtype=float,
    )
    values = (corners - start) @ direction
    return float(np.min(values)), float(np.max(values))


def _geometry_hash_independent(scenario: Mapping[str, Any]) -> str:
    """Reproduce the locked geometry hash without importing scenario builders."""

    specs = []
    for spec in scenario.get("obstacle_specs", []):
        specs.append({
            "geometry_type": str(spec["geometry_type"]),
            "center_m": [float(v) for v in spec["center_m"]],
            "size_m": [float(v) for v in spec["size_m"]],
            "physical_collision_enabled": bool(spec["physical_collision_enabled"]),
        })
    payload: dict[str, Any] = {
        "initial_positions": {str(k): [float(v) for v in value] for k, value in scenario.get("initial_positions", {}).items()},
        "goals": {str(k): [float(v) for v in value] for k, value in scenario.get("goals", {}).items()},
        "obstacle_specs": specs,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
    }
    if "obstacles" in scenario:
        payload["obstacles"] = []
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_mode(phase: str) -> str:
    if phase == "paired_coast":
        return "coast_zero_delta_v"
    if phase == "paired_fixed":
        return "fixed_navigation"
    return "fixed_navigation"


def audit_row(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    config: FixedNavigationConfig | Mapping[str, Any],
    *,
    expected_index: int | None = None,
    expected_command: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one row without simulator state.

    The return value always contains ``ok`` and a list of machine-readable
    ``errors``.  It is intentionally useful on its own for mutation tests.
    """

    cfg = config if isinstance(config, FixedNavigationConfig) else FixedNavigationConfig(**dict(config))
    errors: list[str] = []
    phase = str(row.get("phase", ""))
    mode = str(row.get("controller_mode", ""))
    expected_mode = _phase_mode(phase)
    _err(errors, "controller_mode", mode == expected_mode)
    _err(errors, "safety2_enabled", _bool(row.get("safety2_enabled")) is True)
    _err(errors, "assist_mask", str(row.get("assist_mask", "")) == "00000")
    for name in ASSIST_FIELDS:
        _err(errors, name, not _bool(row.get(name)))
    for field in VECTOR_FIELDS:
        _err(errors, f"finite:{field}", _finite_vector(row.get(field)))

    pre_position = _vector(row.get("true_position_pre_m"))
    pre_velocity = _vector(row.get("true_velocity_pre_mps"))
    post_position = _vector(row.get("true_position_post_m"))
    post_velocity = _vector(row.get("true_velocity_post_mps"))
    estimate = np.concatenate((_vector(row.get("ekf_position_pre_m")), _vector(row.get("ekf_velocity_pre_mps"))))
    requested = _vector(row.get("requested_delta_v_mps"))
    input_command = _vector(row.get("env_input_command_mps"))
    reported = _vector(row.get("reported_applied_delta_v_mps"))
    actual = _vector(row.get("actual_delta_v_mps"))
    actual_accel = _vector(row.get("actual_acceleration_mps2"))
    true_delta = post_velocity - pre_velocity
    axis_limit = cfg.max_delta_v_norm_mps / math.sqrt(3.0)
    _err(errors, "reported_applied_finite", bool(np.isfinite(reported).all()))
    _err(errors, "true_delta_v_finite", bool(np.isfinite(true_delta).all()))
    _err(errors, "actuator_semantic_error", _close(reported, true_delta, SEMANTIC_TOLERANCE))
    _err(errors, "actual_delta_v_field", _close(actual, true_delta, SEMANTIC_TOLERANCE))
    _err(errors, "actual_acceleration_field", _close(actual_accel, actual / DT_S, 2e-6))
    _err(errors, "requested_matches_env_input", _close(requested, input_command, SEMANTIC_TOLERANCE))
    _err(errors, "requested_axis_bound", bool(np.isfinite(requested).all() and np.max(np.abs(requested)) <= axis_limit + 1e-9))
    _err(errors, "requested_norm_bound", bool(np.isfinite(requested).all() and np.linalg.norm(requested) <= cfg.max_delta_v_norm_mps + 1e-9))
    _err(errors, "executed_axis_bound", bool(np.isfinite(actual).all() and np.max(np.abs(actual)) <= axis_limit + 1e-9))
    _err(errors, "executed_norm_bound", bool(np.isfinite(actual).all() and np.linalg.norm(actual) <= cfg.max_delta_v_norm_mps + 1e-9))
    _err(errors, "actual_acceleration_norm_bound", bool(np.isfinite(actual_accel).all() and np.linalg.norm(actual_accel) <= 3.0 + 1e-8))
    _err(errors, "execution_correction_field", _close(row.get("execution_correction_delta_v_mps"), reported - requested, SEMANTIC_TOLERANCE))

    refs = case.get("references", [])
    reference_index = _int(row.get("reference_index"))
    if expected_index is not None:
        _err(errors, "waypoint_index", reference_index == int(expected_index))
        reference = _vector(refs[int(expected_index)]) if 0 <= int(expected_index) < len(refs) else np.full(3, np.nan)
    elif 0 <= reference_index < len(refs):
        reference = _vector(refs[reference_index])
    else:
        reference = _vector(row.get("reference_position_m"))
    _err(errors, "reference_field", _close(row.get("reference_position_m"), reference, SEMANTIC_TOLERANCE))
    if expected_command is None and np.isfinite(estimate).all() and np.isfinite(reference).all():
        expected_command = fixed_navigation_command(estimate, reference, cfg)
    if expected_command is not None:
        for field in ("desired_velocity_mps", "requested_acceleration_mps2", "raw_delta_v_mps", "axis_limited_delta_v_mps"):
            _err(errors, f"command:{field}", _close(row.get(field), expected_command[field], 2e-6))
        if mode == "fixed_navigation":
            _err(errors, "fixed_requested_equals_command", _close(requested, expected_command["command_delta_v_mps"], SEMANTIC_TOLERANCE))
        elif mode == "coast_zero_delta_v":
            _err(errors, "coast_zero_requested", bool(np.allclose(requested, 0.0, rtol=0.0, atol=SEMANTIC_TOLERANCE)))

    goal = _case_goal(case)
    distance = float(np.linalg.norm(post_position - goal)) if np.isfinite(post_position).all() and np.isfinite(goal).all() else float("nan")
    speed = float(np.linalg.norm(post_velocity)) if np.isfinite(post_velocity).all() else float("nan")
    collision = _bool(row.get("collision"))
    qualifying = bool(np.isfinite(distance) and np.isfinite(speed) and distance <= STRICT_DISTANCE_M and speed <= STRICT_SPEED_MPS and not collision)
    displacement = reference - _vector(row.get("ekf_position_pre_m"))
    norm = float(np.linalg.norm(displacement))
    direction = displacement / max(norm, 1e-12)
    pre_projection = float(np.dot(pre_velocity, direction)) if np.isfinite(pre_velocity).all() and np.isfinite(direction).all() else float("nan")
    request_projection = float(np.dot(requested, direction)) if np.isfinite(requested).all() and np.isfinite(direction).all() else float("nan")
    reduction = float(-np.dot(actual, direction)) if np.isfinite(actual).all() and np.isfinite(direction).all() else float("nan")
    witness = bool(pre_projection > 0.05 and request_projection < -1e-4 and reduction > 0.001 and speed < float(np.linalg.norm(pre_velocity)))
    if "braking_witness" in row:
        _err(errors, "braking_witness", _bool(row.get("braking_witness")) == witness)
    if "qualifying" in row:
        _err(errors, "qualifying", _bool(row.get("qualifying")) == qualifying)
    for field, expected in (("requested_projection_mps", request_projection), ("pre_true_projection_mps", pre_projection), ("actual_projected_speed_reduction_mps", reduction)):
        if field in row:
            _err(errors, field, math.isfinite(_float(row.get(field))) and math.isclose(_float(row.get(field)), expected, rel_tol=0.0, abs_tol=2e-6))
    if "goal_distance_m_actual" in row:
        _err(errors, "goal_distance", math.isfinite(_float(row.get("goal_distance_m_actual"))) and math.isclose(_float(row.get("goal_distance_m_actual")), distance, rel_tol=0.0, abs_tol=2e-6))
    if "speed_mps_actual" in row:
        _err(errors, "speed", math.isfinite(_float(row.get("speed_mps_actual"))) and math.isclose(_float(row.get("speed_mps_actual")), speed, rel_tol=0.0, abs_tol=2e-6))
    return {"ok": not errors, "errors": errors, "qualifying": qualifying, "braking_witness_expected": witness,
            "goal_distance_m": distance, "speed_mps": speed, "pre_projection_mps": pre_projection,
            "requested_projection_mps": request_projection, "actual_projected_reduction_mps": reduction,
            "reference_index": reference_index}


def audit_group(
    rows: Iterable[Mapping[str, Any]],
    case: Mapping[str, Any],
    config: FixedNavigationConfig | Mapping[str, Any],
    *,
    episode: Mapping[str, Any] | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Audit one ``(phase, scenario_id)`` group and independently derive metrics."""

    cfg = config if isinstance(config, FixedNavigationConfig) else FixedNavigationConfig(**dict(config))
    ordered = sorted(list(rows), key=lambda row: _int(row.get("step")))
    errors: list[str] = []
    _err(errors, "nonempty_group", bool(ordered))
    steps = [_int(row.get("step")) for row in ordered]
    _err(errors, "contiguous_steps", steps == list(range(1, len(ordered) + 1)))
    _err(errors, "at_most_120_steps", bool(ordered) and len(ordered) <= 120)
    expected_phase = str(ordered[0].get("phase", "")) if ordered else ""
    _err(errors, "group_scenario_id", all(str(row.get("scenario_id")) == str(case.get("scenario_id")) for row in ordered))
    _err(errors, "config_hash", config_hash is None or all(str(row.get("config_sha256")) == str(config_hash) for row in ordered))
    navigator = FixedWaypointNavigator(case.get("references", []), cfg)
    physical = _physical_box(case)
    physical_expected = physical is not None
    start = _vector(case.get("scenario", {}).get("initial_positions", {}).get(UID))
    goal = _case_goal(case)
    mission_direction = (goal - start) / max(float(np.linalg.norm(goal - start)), 1e-12)
    relevant = bool(physical is not None and segment_intersects_obstacle(start, goal, physical))
    downstream = _obstacle_projection_bounds(start, mission_direction, physical)[1] if relevant and physical is not None else float("inf")
    passed_physical = False
    collision_permanent = False
    hold = 0
    strict_success = False
    first_qualifying: int | None = None
    row_details: list[dict[str, Any]] = []
    lower_bounds: list[float] = []
    exact_intersections: list[bool] = []
    witness_raw: list[bool] = []
    previous_position = None
    previous_velocity = None
    for index, row in enumerate(ordered):
        estimate = np.concatenate((_vector(row.get("ekf_position_pre_m")), _vector(row.get("ekf_velocity_pre_mps"))))
        # The evaluator advances the waypoint inside ``command`` before
        # emitting the command.  Reproduce that ordering exactly.
        command = None
        if np.isfinite(estimate).all():
            command = navigator.command(estimate)
        expected_index = int(command["reference_index"]) if command is not None else navigator.index
        reference = case.get("references", [])[expected_index] if 0 <= expected_index < len(case.get("references", [])) else row.get("reference_position_m")
        expected_command = command if command is not None else (fixed_navigation_command(estimate, reference, cfg) if np.isfinite(estimate).all() and _finite_vector(reference) else None)
        detail = audit_row(row, case, cfg, expected_index=expected_index, expected_command=expected_command)
        errors.extend(f"step{_int(row.get('step'))}:{error}" for error in detail["errors"])
        collision = _bool(row.get("collision"))
        collision_permanent = collision_permanent or collision
        post_position = _vector(row.get("true_position_post_m"))
        post_velocity = _vector(row.get("true_velocity_post_mps"))
        qualifies = bool(detail["qualifying"])
        if qualifies and first_qualifying is None:
            first_qualifying = _int(row.get("step"))
        if physical is not None:
            pre_position = _vector(row.get("true_position_pre_m"))
            if np.isfinite(pre_position).all() and np.isfinite(post_position).all():
                sampled = segment_obstacle_clearance(pre_position, post_position, physical, samples=33)
                lower = max(0.0, sampled - float(np.linalg.norm(post_position - pre_position)) / 64.0)
                intersects = bool(segment_intersects_obstacle(pre_position, post_position, physical))
                lower_bounds.append(lower)
                exact_intersections.append(intersects)
                projection = float(np.dot(post_position - start, mission_direction))
                current_clearance = float(sampled)
                if relevant and projection >= downstream + STRICT_PASS_BUFFER_M and current_clearance >= STRICT_CLEARANCE_M:
                    passed_physical = True
                _err(errors, f"step{_int(row.get('step'))}:segment_intersects", _bool(row.get("segment_intersects")) == intersects)
                logged_lower = row.get("segment_clearance_lower_bound_m")
                _err(errors, f"step{_int(row.get('step'))}:clearance_lower_bound", logged_lower not in (None, "", "null") and math.isclose(_float(logged_lower), lower, rel_tol=0.0, abs_tol=2e-6))
        else:
            _err(errors, f"step{_int(row.get('step'))}:no_physical_clearance", row.get("segment_clearance_lower_bound_m") in (None, "", "null"))
            _err(errors, f"step{_int(row.get('step'))}:no_segment_intersection", not _bool(row.get("segment_intersects")))
        strict_physical = passed_physical if physical_expected else True
        strict_qualifies = bool(qualifies and strict_physical and not collision_permanent)
        hold = hold + 1 if strict_qualifies else 0
        if strict_qualifies and hold >= STRICT_HOLD_STEPS:
            strict_success = True
        _err(errors, f"step{_int(row.get('step'))}:strict_hold_steps", _int(row.get("strict_hold_steps")) == hold)
        _err(errors, f"step{_int(row.get('step'))}:strict_success", _bool(row.get("strict_success")) == strict_success)
        _err(errors, f"step{_int(row.get('step'))}:physical_pass", _bool(row.get("strict_physical_pass_complete")) == bool(passed_physical if physical_expected else False))
        witness_raw.append(bool(detail["braking_witness_expected"]))
        row_details.append({"step": _int(row.get("step")), "qualifying": qualifies, "strict_qualifying": strict_qualifies,
                            "hold": hold, "strict_success": strict_success, "braking_witness": witness_raw[-1],
                            "goal_distance_m": detail["goal_distance_m"], "speed_mps": detail["speed_mps"]})
        previous_position, previous_velocity = post_position, post_velocity

    first = first_qualifying
    expected_before = [bool(witness and (first is None or step < first)) for witness, step in zip(witness_raw, steps)]
    for row, expected in zip(ordered, expected_before):
        _err(errors, f"step{_int(row.get('step'))}:braking_before_first_hold", _bool(row.get("braking_before_first_hold")) == expected)
    terminal_indices = [index for index, row in enumerate(ordered) if _bool(row.get("terminated")) or _bool(row.get("truncated"))]
    _err(errors, "no_after_terminal", not terminal_indices or terminal_indices == [len(ordered) - 1])
    if ordered:
        _err(errors, "terminal_flags_exclusive", not (_bool(ordered[-1].get("terminated")) and _bool(ordered[-1].get("truncated"))))
    peak_speed = max((detail["speed_mps"] for detail in (audit_row(row, case, cfg) for row in ordered)), default=float("nan"))
    final = row_details[-1] if row_details else {}
    final_row = ordered[-1] if ordered else {}
    collision_count = sum(_bool(row.get("collision")) for row in ordered)
    contact_count = sum(_bool(row.get("physical_contact")) for row in ordered)
    episode_derived = {
        "phase": expected_phase, "scenario_id": str(case.get("scenario_id")), "regime": str(case.get("regime")),
        "noise_seed": _int(case.get("noise_seed")), "steps": len(ordered),
        "success": strict_success, "final_distance_m": final.get("goal_distance_m", float("nan")),
        "final_speed_mps": final.get("speed_mps", float("nan")), "peak_speed_mps": peak_speed,
        "braking_steps_before_first_hold": sum(expected_before), "first_qualifying_step": first,
        "collision": bool(collision_count), "physical_contact": bool(contact_count),
        "boundary_projection_steps": sum(_bool(row.get("boundary_projected")) for row in ordered),
        "safety_projection_steps": sum(_bool(row.get("safety_projected")) for row in ordered),
        "illegal_steps": sum(not _bool(row.get("legal_action_and_acceleration")) for row in ordered),
        "environment_clipped_steps": sum(_bool(row.get("environment_action_clipped")) for row in ordered),
        "physical_pass_complete": bool(passed_physical if physical_expected else False),
        "minimum_clearance_lower_bound_m": min(lower_bounds) if lower_bounds else None,
        "segment_intersects": bool(any(exact_intersections)),
        "timeout": bool(_bool(final_row.get("truncated")) and not strict_success),
        "max_actual_acceleration_mps2": max((_float(np.linalg.norm(_vector(row.get("actual_acceleration_mps2")))) for row in ordered), default=float("nan")),
        "max_position_estimation_error_m": max((_float(np.linalg.norm(_vector(row.get("ekf_position_pre_m")) - _vector(row.get("true_position_pre_m")))) for row in ordered), default=float("nan")),
    }
    if episode is not None:
        for key, expected in episode_derived.items():
            if key not in episode:
                errors.append(f"episode_missing:{key}")
                continue
            actual = episode[key]
            if isinstance(expected, bool):
                _err(errors, f"episode:{key}", _bool(actual) == expected)
            elif expected is None:
                _err(errors, f"episode:{key}", actual in (None, "", "null"))
            elif isinstance(expected, (int, str)):
                _err(errors, f"episode:{key}", str(actual) == str(expected))
            else:
                _err(errors, f"episode:{key}", math.isfinite(float(expected)) and math.isclose(_float(actual), float(expected), rel_tol=0.0, abs_tol=3e-5))
    return {"ok": not errors, "errors": errors, "episode": episode_derived, "rows": row_details,
            "first_qualifying_step": first, "braking_witness_count": sum(witness_raw),
            "braking_before_first_hold_count": sum(expected_before), "physical_expected": physical_expected,
            "exact_segment_intersects": bool(any(exact_intersections))}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _case_index(lock: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    cases: dict[tuple[str, str], Mapping[str, Any]] = {}
    for phase, pool in lock.get("cases", {}).items():
        for case in pool:
            cases[(str(phase), str(case.get("scenario_id")))] = case
    return cases


def _check_manifest(root: Path, data_root: Path, log_root: Path, lock: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Check hashes and independent ID/seed/geometry separation."""

    config_path = root / "data" / "configs" / f"{STEM}.json"
    if config_path.is_file():
        _err(errors, "config_sha256", _sha256(config_path) == str(lock.get("config_sha256")))
    else:
        errors.append("missing_config")
    for relative, expected in (lock.get("sources") or {}).items():
        path = root / Path(str(relative))
        _err(errors, f"source_sha256:{relative}", path.is_file() and _sha256(path).lower() == str(expected).lower())
    cases = [case for pool in lock.get("cases", {}).values() for case in pool]
    ids = [str(case.get("scenario_id")) for case in cases]
    seeds = [_int(case.get("noise_seed")) for case in cases]
    geometries = [str(case.get("geometry_sha256")) for case in cases]
    _err(errors, "manifest_ids_unique", len(ids) == len(set(ids)))
    _err(errors, "manifest_noise_seeds_unique", len(seeds) == len(set(seeds)))
    # Geometry is required to be split between independent phases.  The two
    # left/right obstacle routes intentionally reuse each static box geometry,
    # so uniqueness within the optional obstacle phase is not a contract.
    phase_geometries = {
        str(phase): {str(case.get("geometry_sha256")) for case in pool}
        for phase, pool in lock.get("cases", {}).items()
    }
    split_ok = True
    for left, right in (("development", "normal_evaluation"), ("development", "paired"), ("normal_evaluation", "paired")):
        split_ok = split_ok and not (phase_geometries.get(left, set()) & phase_geometries.get(right, set()))
    split_ok = split_ok and not (phase_geometries.get("obstacle", set()) & (phase_geometries.get("development", set()) | phase_geometries.get("normal_evaluation", set()) | phase_geometries.get("paired", set())))
    _err(errors, "manifest_geometry_phase_splits", split_ok)
    for phase, pool in lock.get("cases", {}).items():
        for case in pool:
            _err(errors, f"geometry_hash:{case.get('scenario_id')}", _geometry_hash_independent(case.get("scenario", {})) == str(case.get("geometry_sha256")))
            _err(errors, f"final_goal:{case.get('scenario_id')}", bool(case.get("references")) and _close(case["references"][-1], _case_goal(case), SEMANTIC_TOLERANCE))
    protected_path = log_root / "protected_inputs_before.json"
    if protected_path.is_file():
        protected = _json(protected_path)
        _err(errors, "protected_count_627", len(protected) == 627)
        for path, expected in protected.items():
            resolved = Path(path)
            _err(errors, f"protected_sha256:{path}", resolved.is_file() and _sha256(resolved).lower() == str(expected).lower())
    else:
        errors.append("missing_protected_inputs_before")
    statistics = root / "data" / "synthetic" / "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011" / "full_train_statistics.json"
    _err(errors, "frozen_statistics_sha256", statistics.is_file() and _sha256(statistics).lower() == FROZEN_STATISTICS_SHA256)
    return {"case_count": len(cases), "scenario_ids": len(set(ids)), "noise_seeds": len(set(seeds)), "geometries": len(set(geometries)),
            "protected_file_count": 627 if protected_path.is_file() else None, "frozen_statistics_sha256": _sha256(statistics) if statistics.is_file() else None}


def _compare_summary_episodes(summary: Mapping[str, Any], episodes: list[Mapping[str, Any]], errors: list[str]) -> None:
    listed = summary.get("episodes")
    _err(errors, "summary_episode_list", isinstance(listed, list))
    if not isinstance(listed, list):
        return
    by_key = {(str(item.get("phase")), str(item.get("scenario_id"))): item for item in listed}
    expected = {(str(item.get("phase")), str(item.get("scenario_id"))): item for item in episodes}
    _err(errors, "summary_episode_keys", set(by_key) == set(expected))
    for key, item in expected.items():
        actual = by_key.get(key, {})
        for field, value in item.items():
            if field not in actual:
                errors.append(f"summary_episode_missing:{key}:{field}")
            elif isinstance(value, bool):
                _err(errors, f"summary_episode:{key}:{field}", _bool(actual[field]) == value)
            elif value is None:
                _err(errors, f"summary_episode:{key}:{field}", actual[field] in (None, "", "null"))
            elif isinstance(value, (int, str)):
                _err(errors, f"summary_episode:{key}:{field}", str(actual[field]) == str(value))
            else:
                _err(errors, f"summary_episode:{key}:{field}", math.isclose(_float(actual[field]), float(value), rel_tol=0.0, abs_tol=3e-5))


def audit_artifacts(
    log_root: str | Path = DEFAULT_LOG_ROOT,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output: str | Path | None = None,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Read existing artifacts and write one exclusive independent report."""

    root = Path(project_root)
    log = Path(log_root)
    data = Path(data_root)
    if not log.is_absolute():
        log = root / log
    if not data.is_absolute():
        data = root / data
    errors: list[str] = []
    lock_path = data / "execution_manifest.lock.json"
    summary_path = log / "execution_summary.json"
    episodes_path = log / "episodes.csv"
    if not lock_path.is_file():
        errors.append(f"missing:{lock_path}")
        lock: Mapping[str, Any] = {}
    else:
        lock = _json(lock_path)
    manifest_details = _check_manifest(root, data, log, lock, errors) if lock else {}
    config_payload = lock.get("controller_config", {})
    try:
        config = FixedNavigationConfig(**config_payload)
    except Exception as exc:  # fail closed while keeping a JSON report
        errors.append(f"invalid_controller_config:{exc}")
        config = FixedNavigationConfig()
    case_map = _case_index(lock)
    all_rows: list[dict[str, str]] = []
    group_results: dict[tuple[str, str], dict[str, Any]] = {}
    missing_files: list[str] = []
    for phase, filename in PHASE_FILES.items():
        path = log / filename
        if not path.is_file():
            if phase != "obstacle":
                missing_files.append(filename)
            continue
        rows = _read_csv(path)
        all_rows.extend(rows)
        for key in {(str(row.get("phase")), str(row.get("scenario_id"))) for row in rows}:
            phase_label, scenario_id = key
            manifest_phase = "paired" if phase_label.startswith("paired_") else phase_label
            case = case_map.get((manifest_phase, scenario_id))
            if case is None:
                errors.append(f"unknown_case:{key}")
                continue
            group = [row for row in rows if str(row.get("phase")) == phase_label and str(row.get("scenario_id")) == scenario_id]
            group_results[key] = audit_group(group, case, config, config_hash=str(lock.get("config_sha256")))
            if not group_results[key]["ok"]:
                errors.extend(f"{phase_label}/{scenario_id}:{value}" for value in group_results[key]["errors"])
    for filename in missing_files:
        errors.append(f"missing:{log / filename}")

    episode_rows = _read_csv(episodes_path) if episodes_path.is_file() else []
    if not episodes_path.is_file():
        errors.append(f"missing:{episodes_path}")
    episode_by_key = {(str(row.get("phase")), str(row.get("scenario_id"))): row for row in episode_rows}
    derived_episodes: list[dict[str, Any]] = []
    for key, result in group_results.items():
        phase_label, scenario_id = key
        episode = episode_by_key.get(key)
        if episode is None:
            errors.append(f"missing_episode:{key}")
        case = case_map.get(("paired" if phase_label.startswith("paired_") else phase_label, scenario_id))
        if case is not None:
            # Re-run the pure group with the episode row so final metrics and counts are compared.
            source_rows = [row for row in all_rows if str(row.get("phase")) == phase_label and str(row.get("scenario_id")) == scenario_id]
            checked = audit_group(source_rows, case, config, episode=episode, config_hash=str(lock.get("config_sha256")))
            group_results[key] = checked
            if not checked["ok"]:
                errors.extend(f"{phase_label}/{scenario_id}:{value}" for value in checked["errors"] if value not in result.get("errors", []))
            derived_episodes.append(checked["episode"])
    _err(errors, "episode_keys_match_groups", set(episode_by_key) == set(group_results))

    phase_gates: dict[str, bool] = {}
    def phase_groups(label: str) -> list[dict[str, Any]]:
        return [result["episode"] for (phase, _), result in group_results.items() if phase == label]
    def clean(items: list[Mapping[str, Any]]) -> bool:
        return all(not _bool(item.get("collision")) and not _bool(item.get("physical_contact"))
                   and _int(item.get("boundary_projection_steps")) == 0 and _int(item.get("safety_projection_steps")) == 0
                   and _int(item.get("illegal_steps")) == 0 and _int(item.get("environment_clipped_steps")) == 0 for item in items)
    development = phase_groups("development")
    phase_gates["development"] = len(development) == 6 and all(item["success"] for item in development) and clean(development)
    normal = phase_groups("normal_evaluation")
    regime_counts = {regime: sum(item.get("regime") == regime for item in normal) for regime in ("rest", "toward", "away", "tangent")}
    bias_cases = {str(case.get("scenario_id")) for case in lock.get("cases", {}).get("normal_evaluation", []) if any(abs(float(v)) > 0 for v in case.get("belief_position_offset_m", [])) or any(abs(float(v)) > 0 for v in case.get("belief_velocity_offset_m", []))}
    bias = [item for item in normal if str(item.get("scenario_id")) in bias_cases]
    rests = [item for item in normal if item.get("regime") == "rest"]
    moving = [item for item in normal if item.get("regime") != "rest"]
    phase_gates["normal_evaluation"] = (len(normal) == 36 and all(item["success"] for item in normal) and clean(normal)
                                         and regime_counts == {"rest": 9, "toward": 9, "away": 9, "tangent": 9}
                                         and len(bias) == 4 and all(item["success"] and clean([item]) for item in bias)
                                         and all(float(item.get("peak_speed_mps", 0)) > 0.5 and int(item.get("braking_steps_before_first_hold", 0)) >= 1 for item in rests)
                                         and sum(int(item.get("braking_steps_before_first_hold", 0)) >= 1 for item in moving) >= 14)
    fixed = phase_groups("paired_fixed")
    coast = phase_groups("paired_coast")
    paired_same = len(fixed) == len(coast) == 4
    if paired_same:
        fkeys = {(item.get("scenario_id"), item.get("noise_seed")) for item in fixed}
        ckeys = {(item.get("scenario_id"), item.get("noise_seed")) for item in coast}
        paired_same = fkeys == ckeys
        for key in fkeys:
            fr = next(result for (phase, sid), result in group_results.items() if phase == "paired_fixed" and sid == str(key[0]))
            cr = next(result for (phase, sid), result in group_results.items() if phase == "paired_coast" and sid == str(key[0]))
            # The step-1 truth state is immutable across the pair.
            frows = [row for row in all_rows if row.get("phase") == "paired_fixed" and row.get("scenario_id") == str(key[0])]
            crows = [row for row in all_rows if row.get("phase") == "paired_coast" and row.get("scenario_id") == str(key[0])]
            if not frows or not crows or not _close(frows[0].get("true_position_pre_m"), crows[0].get("true_position_pre_m")) or not _close(frows[0].get("true_velocity_pre_mps"), crows[0].get("true_velocity_pre_mps")):
                paired_same = False
    phase_gates["paired"] = paired_same and all(item["success"] for item in fixed) and not any(item["success"] for item in coast) and clean(fixed + coast)
    obstacle_path_exists = (log / PHASE_FILES["obstacle"]).is_file()
    obstacle = phase_groups("obstacle")
    phase_gates["obstacle"] = (not obstacle_path_exists or (len(obstacle) == 4 and all(item["success"] and item["physical_pass_complete"] and not item["segment_intersects"] and float(item["minimum_clearance_lower_bound_m"]) >= 2.5 for item in obstacle) and clean(obstacle)))
    if obstacle_path_exists and not (phase_gates["development"] and phase_gates["normal_evaluation"] and phase_gates["paired"]):
        errors.append("obstacle_present_before_prior_gates")

    summary = _json(summary_path) if summary_path.is_file() else {}
    if not summary_path.is_file():
        errors.append(f"missing:{summary_path}")
    else:
        _compare_summary_episodes(summary, derived_episodes, errors)
        counts = summary.get("counts", {})
        _err(errors, "summary_environment_steps", _int(counts.get("environment_steps")) == len(all_rows) and _int(counts.get("environment_steps")) <= 6480)
        for key in ("checkpoint_loads", "trainer_constructions", "ppo_updates", "optimizer_steps"):
            _err(errors, f"summary_no_{key}", _int(counts.get(key)) == 0)
        _err(errors, "summary_no_training", bool(summary.get("no_tuning_or_reruns", False)))
        for phase, gate in phase_gates.items():
            if phase in summary.get("gates", {}):
                _err(errors, f"summary_gate:{phase}", _bool(summary["gates"][phase]) == gate)

    verdict = "passed" if not errors else "failed"
    report: dict[str, Any] = {
        "schema_version": "single_uav_fixed_navigation_independent_audit_v1",
        "status": verdict, "passed": verdict == "passed", "project_root": str(root.resolve()),
        "log_root": str(log.resolve()), "data_root": str(data.resolve()),
        "phase_gates": phase_gates, "manifest": manifest_details,
        "group_count": len(group_results), "row_count": len(all_rows),
        "errors": errors[:500], "error_count": len(errors),
        "read_only_execution": {"environment_steps": 0, "checkpoint_loads": 0, "trainer_constructions": 0, "ppo_updates": 0, "optimizer_steps": 0},
        "independent": True, "no_learning_data_used": True,
    }
    output_path = Path(output) if output is not None else log / "independent_audit.json"
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    report["output"] = str(output_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit_artifacts(args.log_root, data_root=args.data_root, output=args.output)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "phase_gates": report["phase_gates"], "errors": report["error_count"]}, ensure_ascii=False, sort_keys=True))


__all__ = ["audit_artifacts", "audit_group", "audit_row", "main", "DEFAULT_LOG_ROOT", "DEFAULT_DATA_ROOT"]


if __name__ == "__main__":
    main()
