"""Independent, offline audit for the controlled three-UAV navigation run.

The runner writes the evidence consumed here.  This module deliberately does
not import the runner, an environment, a learner, torch, or an optimizer.  It
recomputes navigation commands, synchronized pair distances, message gates,
and the fixed intervention schedule from the saved rows.  Missing artifacts
are reported as failures; no missing statistic is filled in from the runner's
summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..control.fixed_navigation import FixedNavigationConfig, fixed_navigation_command

ROOT = Path(__file__).resolve().parents[2]
STEM = "three_uav_fixed_navigation_v2_seed2026_20261011"
DEFAULT_LOG_ROOT = ROOT / "logs" / "preliminary" / STEM
DEFAULT_DATA_ROOT = ROOT / "data" / "synthetic" / STEM
DEFAULT_CONFIG = ROOT / "data" / "configs" / f"{STEM}.json"
UIDS = ("uav_00", "uav_01", "uav_02")
PAIRS = tuple((UIDS[i], UIDS[j]) for i in range(3) for j in range(i + 1, 3))
DT_S = 0.2
MAX_STEPS = 120
EPISODES = 8
JOINT_STEPS = EPISODES * MAX_STEPS
DV_NORM = 0.6
DV_AXIS = DV_NORM / math.sqrt(3.0)
MAX_SPEED = 12.0
BOUNDS_MIN = np.asarray((0.0, 0.0, 2.0), dtype=float)
BOUNDS_MAX = np.asarray((200.0, 200.0, 60.0), dtype=float)
STRICT_DISTANCE = 1.0
STRICT_SPEED = 0.5
STRICT_HOLD = 5
POTENTIAL_RANGE = 75.0
UWB_RANGE = 100.0
STALE_AGE = 0.6
UPDATE_PERIOD = 0.8
SEMANTIC_TOL = 1e-6
INPUT_TOL = 1e-12
COV_TOL = 1e-8
ASSIST_FIELDS = ("pass_commit", "lateral_exit", "radial_intent", "goal_recovery", "terminal_capture")
TABLES = ("navigation", "joint", "pairs", "flows", "messages", "gates", "fusion_events")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _native(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
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


def _vector(value: Any, size: int = 3) -> np.ndarray:
    value = _native(value)
    try:
        return np.asarray(value, dtype=float).reshape(size)
    except (TypeError, ValueError):
        return np.full(size, np.nan)


def _state(value: Any) -> np.ndarray:
    value = _native(value)
    try:
        state = np.asarray(value, dtype=float).reshape(-1)
        return state[:6] if state.size in (6, 15) else np.full(6, np.nan)
    except (TypeError, ValueError):
        return np.full(6, np.nan)


def _matrix(value: Any, shape: tuple[int, int] = (15, 15)) -> np.ndarray:
    value = _native(value)
    try:
        return np.asarray(value, dtype=float).reshape(shape)
    except (TypeError, ValueError):
        return np.full(shape, np.nan)


def _close(left: Any, right: Any, atol: float = SEMANTIC_TOL) -> bool:
    try:
        return bool(np.allclose(np.asarray(_native(left), dtype=float), np.asarray(_native(right), dtype=float), rtol=0.0, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(_native(value), dtype=float)).all())
    except (TypeError, ValueError):
        return False


def _missing(value: Any) -> bool:
    """Whether a CSV cell or optional vector is absent."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "null"}
    return False


def _err(errors: list[str], name: str, condition: bool) -> None:
    if not condition:
        errors.append(name)


def _row_goal(row: Mapping[str, Any], goal: Any | None = None) -> np.ndarray:
    return _vector(row.get("goal_m") if goal is None else goal)


def _row_estimate(row: Mapping[str, Any]) -> np.ndarray:
    packed = row.get("ekf_state_pre")
    if packed is not None and not (isinstance(packed, str) and packed.strip() in {"", "null"}):
        return _state(packed)
    return np.concatenate((_vector(row.get("ekf_position_pre_m")), _vector(row.get("ekf_velocity_pre_mps"))))


def _config(value: FixedNavigationConfig | Mapping[str, Any] | None = None) -> FixedNavigationConfig:
    if value is None:
        return FixedNavigationConfig()
    if isinstance(value, FixedNavigationConfig):
        return value
    return FixedNavigationConfig(**dict(value))


def packet_id(source: str, target: str, send_time: Any) -> str:
    return f"{source}->{target}@{float(send_time):.17g}"


def synchronized_pair_distance(pre_source: Any, post_source: Any, pre_target: Any, post_target: Any) -> tuple[float, float]:
    """Return closest relative distance using one shared time fraction."""

    pre = _vector(pre_source) - _vector(pre_target)
    delta = (_vector(post_source) - _vector(post_target)) - pre
    norm2 = float(np.dot(delta, delta))
    fraction = 0.0 if norm2 <= 1e-24 else max(0.0, min(1.0, -float(np.dot(pre, delta)) / norm2))
    return float(np.linalg.norm(pre + fraction * delta)), float(fraction)


def audit_navigation_row(
    row: Mapping[str, Any],
    *,
    config: FixedNavigationConfig | Mapping[str, Any] | None = None,
    goal: Any | None = None,
    expected_command: Mapping[str, Any] | None = None,
) -> list[str]:
    """Audit one navigation row and return machine-readable error names.

    The function is pure and intentionally useful in mutation tests.  It
    accepts both the runner's packed six-state fields and split test fixtures.
    """

    cfg = _config(config)
    errors: list[str] = []
    estimate = _row_estimate(row)
    goal_array = _row_goal(row, goal)
    pre_position = _vector(row.get("true_position_pre_m"))
    pre_velocity = _vector(row.get("true_velocity_pre_mps"))
    post_position = _vector(row.get("true_position_post_m"))
    post_velocity = _vector(row.get("true_velocity_post_mps"))
    requested = _vector(row.get("requested_delta_v_mps"))
    env_input = _vector(row.get("env_input_delta_v_mps", row.get("env_input_command_mps")))
    reported = _vector(row.get("reported_applied_delta_v_mps"))
    actual = _vector(row.get("actual_delta_v_mps"))
    actual_acceleration = _vector(row.get("actual_acceleration_mps2"))
    for name, value in (("estimate", estimate), ("goal", goal_array), ("pre_position", pre_position),
                        ("pre_velocity", pre_velocity), ("post_position", post_position),
                        ("post_velocity", post_velocity), ("requested", requested), ("env_input", env_input),
                        ("reported", reported), ("actual", actual), ("actual_acceleration", actual_acceleration)):
        _err(errors, f"finite:{name}", bool(np.isfinite(value).all()))
    _err(errors, "action_input_identity", _close(requested, env_input, INPUT_TOL))
    true_delta = post_velocity - pre_velocity
    _err(errors, "actual_delta_v", _close(actual, true_delta, SEMANTIC_TOL))
    _err(errors, "reported_applied_delta_v", _close(reported, true_delta, SEMANTIC_TOL))
    _err(errors, "actual_acceleration", _close(actual_acceleration, actual / DT_S, 2e-6))
    _err(errors, "axis_bound_requested", bool(np.max(np.abs(requested)) <= DV_AXIS + 1e-9))
    _err(errors, "axis_bound_actual", bool(np.max(np.abs(actual)) <= DV_AXIS + 1e-9))
    _err(errors, "norm_bound_requested", bool(np.linalg.norm(requested) <= DV_NORM + 1e-9))
    _err(errors, "norm_bound_actual", bool(np.linalg.norm(actual) <= DV_NORM + 1e-9))
    _err(errors, "acceleration_bound", bool(np.linalg.norm(actual_acceleration) <= 3.0 + 1e-8))
    _err(errors, "speed_bound", bool(np.linalg.norm(post_velocity) <= MAX_SPEED + 1e-9))
    _err(errors, "position_bounds", bool(np.all(post_position >= BOUNDS_MIN) and np.all(post_position <= BOUNDS_MAX)))
    for field in ("boundary_projected", "safety_projected", "environment_action_clipped"):
        _err(errors, field, not _bool(row.get(field)))
    covariance = _matrix(row.get("ekf_covariance_post"))
    covariance_finite = bool(np.isfinite(covariance).all())
    _err(errors, "covariance_finite", covariance_finite)
    if covariance_finite:
        _err(errors, "covariance_symmetric", float(np.max(np.abs(covariance - covariance.T))) <= COV_TOL)
        _err(errors, "covariance_psd", float(np.linalg.eigvalsh((covariance + covariance.T) / 2.0).min()) >= -COV_TOL)
    else:
        errors.extend(("covariance_symmetric", "covariance_psd"))

    if expected_command is None and np.isfinite(estimate).all() and np.isfinite(goal_array).all():
        try:
            expected_command = fixed_navigation_command(estimate, goal_array, cfg)
        except (TypeError, ValueError):
            expected_command = None
    if expected_command is not None:
        for field in ("desired_velocity_mps", "requested_acceleration_mps2", "raw_delta_v_mps", "axis_limited_delta_v_mps"):
            _err(errors, f"command:{field}", _close(row.get(field), expected_command[field], 2e-6))
        _err(errors, "command:requested_delta_v_mps", _close(requested, expected_command["command_delta_v_mps"], INPUT_TOL))

    distance = float(np.linalg.norm(post_position - goal_array)) if np.isfinite(post_position).all() and np.isfinite(goal_array).all() else float("nan")
    speed = float(np.linalg.norm(post_velocity)) if np.isfinite(post_velocity).all() else float("nan")
    qualifying = bool(math.isfinite(distance) and math.isfinite(speed) and distance <= STRICT_DISTANCE and speed <= STRICT_SPEED and not _bool(row.get("collision")))
    _err(errors, "qualifying", _bool(row.get("qualifying")) == qualifying)
    direction_vector = goal_array - estimate[:3]
    direction = direction_vector / max(float(np.linalg.norm(direction_vector)), 1e-12) if np.isfinite(direction_vector).all() else np.full(3, np.nan)
    pre_projection = float(np.dot(pre_velocity, direction)) if np.isfinite(direction).all() and np.isfinite(pre_velocity).all() else float("nan")
    requested_projection = float(np.dot(requested, direction)) if np.isfinite(direction).all() and np.isfinite(requested).all() else float("nan")
    reduction = float(-np.dot(actual, direction)) if np.isfinite(direction).all() and np.isfinite(actual).all() else float("nan")
    witness = bool(pre_projection > .05 and requested_projection < -1e-4 and reduction > .001 and speed < float(np.linalg.norm(pre_velocity)))
    if "braking_witness" in row:
        _err(errors, "braking_witness", _bool(row.get("braking_witness")) == witness)
    if "pre_true_velocity_projection_mps" in row:
        _err(errors, "pre_projection", math.isfinite(_float(row.get("pre_true_velocity_projection_mps"))) and math.isclose(_float(row.get("pre_true_velocity_projection_mps")), pre_projection, rel_tol=0.0, abs_tol=2e-6))
    if "requested_projection_mps" in row:
        _err(errors, "requested_projection", math.isfinite(_float(row.get("requested_projection_mps"))) and math.isclose(_float(row.get("requested_projection_mps")), requested_projection, rel_tol=0.0, abs_tol=2e-6))
    if "actual_projected_speed_reduction_mps" in row:
        _err(errors, "projected_reduction", math.isfinite(_float(row.get("actual_projected_speed_reduction_mps"))) and math.isclose(_float(row.get("actual_projected_speed_reduction_mps")), reduction, rel_tol=0.0, abs_tol=2e-6))
    if "goal_distance_m_actual" in row:
        _err(errors, "goal_distance", math.isfinite(_float(row.get("goal_distance_m_actual"))) and math.isclose(_float(row.get("goal_distance_m_actual")), distance, rel_tol=0.0, abs_tol=2e-6))
    if "speed_mps_actual" in row:
        _err(errors, "speed", math.isfinite(_float(row.get("speed_mps_actual"))) and math.isclose(_float(row.get("speed_mps_actual")), speed, rel_tol=0.0, abs_tol=2e-6))
    _err(errors, "safety2_enabled", _bool(row.get("safety2_enabled")) is True)
    _err(errors, "assist_mask", str(row.get("assist_mask", "")) == "00000")
    for field in ASSIST_FIELDS:
        if field in row:
            _err(errors, field, not _bool(row.get(field)))
    return errors


def _strict_replay(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    holds = dict.fromkeys(UIDS, 0)
    ever = dict.fromkeys(UIDS, False)
    first_qual = dict.fromkeys(UIDS)
    first_success = dict.fromkeys(UIDS)
    collision_latched = False
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        collision_latched = collision_latched or _bool(row.get("collision"))
        goal = _row_goal(row)
        current: dict[str, bool] = {}
        uid = str(row.get("uav_id") or UIDS[0])
        if uid not in holds:
            uid = UIDS[0]
        post = _vector(row.get("true_position_post_m"))
        velocity = _vector(row.get("true_velocity_post_mps"))
        qualifies = bool(np.isfinite(post).all() and np.isfinite(velocity).all() and np.isfinite(goal).all()
                        and np.linalg.norm(post - goal) <= STRICT_DISTANCE
                        and np.linalg.norm(velocity) <= STRICT_SPEED
                        and not collision_latched)
        current[uid] = qualifies
        if qualifies:
            holds[uid] += 1
            if first_qual[uid] is None:
                first_qual[uid] = index
            if holds[uid] >= STRICT_HOLD:
                ever[uid] = True
                if first_success[uid] is None:
                    first_success[uid] = index
        else:
            holds[uid] = 0
        details.append({"uid": uid, "qualifying": qualifies, "hold_steps": holds[uid], "ever_success": ever[uid],
                        "first_qualifying_step": first_qual[uid], "first_success_step": first_success[uid]})
    return details, collision_latched


def derive_strict_status(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure strict parking replay exposed for audit and mutation tests."""

    return _strict_replay(rows)[0]


def audit_navigation_group(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: FixedNavigationConfig | Mapping[str, Any] | None = None,
    expected_goal_by_uid: Mapping[str, Any] | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """Audit one episode's 360 navigation rows independently."""

    cfg = _config(config)
    ordered = sorted(list(rows), key=lambda row: (_int(row.get("step")), str(row.get("uav_id"))))
    errors: list[str] = []
    _err(errors, "navigation_row_count_360", len(ordered) == 360)
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_uid[str(row.get("uav_id"))].append(row)
        goal = expected_goal_by_uid.get(str(row.get("uav_id"))) if expected_goal_by_uid else None
        errors.extend(f"step{_int(row.get('step'))}/{row.get('uav_id')}:{name}" for name in audit_navigation_row(row, config=cfg, goal=goal))
    _err(errors, "navigation_uids", set(by_uid) == set(UIDS))
    derived: dict[str, Any] = {"steps": 0, "braking_before_first_qualifying": {}, "success": {}}
    for uid in UIDS:
        rows_uid = sorted(by_uid.get(uid, []), key=lambda row: _int(row.get("step")))
        steps = [_int(row.get("step")) for row in rows_uid]
        _err(errors, f"{uid}:row_count_120", len(rows_uid) == MAX_STEPS)
        _err(errors, f"{uid}:contiguous_steps", steps == list(range(1, MAX_STEPS + 1)))
        for previous, current in zip(rows_uid, rows_uid[1:]):
            _err(errors, f"{uid}:truth_position_continuity_step{_int(current.get('step'))}", _close(current.get("true_position_pre_m"), previous.get("true_position_post_m"), 1e-9))
            _err(errors, f"{uid}:truth_velocity_continuity_step{_int(current.get('step'))}", _close(current.get("true_velocity_pre_mps"), previous.get("true_velocity_post_mps"), 1e-9))
            _err(errors, f"{uid}:ekf_continuity_step{_int(current.get('step'))}", _close(current.get("ekf_state_pre"), previous.get("ekf_state_post"), 1e-8))
        details, collision = _strict_replay(rows_uid)
        braking = 0
        first_qualifying: int | None = None
        for row, detail in zip(rows_uid, details):
            if detail["qualifying"] and first_qualifying is None:
                first_qualifying = _int(row.get("step"))
            witness = _bool(row.get("braking_witness"))
            before = witness and (first_qualifying is None or _int(row.get("step")) < first_qualifying)
            braking += int(before)
            _err(errors, f"{uid}:hold_step{_int(row.get('step'))}", _int(row.get("hold_steps")) == int(detail["hold_steps"]))
            _err(errors, f"{uid}:ever_success_step{_int(row.get('step'))}", _bool(row.get("ever_success")) == bool(detail["ever_success"]))
            _err(errors, f"{uid}:first_qualifying_step{_int(row.get('step'))}", _missing(row.get("first_qualifying_step")) if detail["first_qualifying_step"] is None else _int(row.get("first_qualifying_step")) == detail["first_qualifying_step"])
            _err(errors, f"{uid}:first_success_step{_int(row.get('step'))}", _missing(row.get("first_success_step")) if detail["first_success_step"] is None else _int(row.get("first_success_step")) == detail["first_success_step"])
            _err(errors, f"{uid}:braking_before_first{_int(row.get('step'))}", _bool(row.get("braking_before_first_qualifying")) == before)
        if not (case_id == "c2" and uid == "uav_02"):
            _err(errors, f"{uid}:braking_witness_required", braking >= 1)
        derived["braking_before_first_qualifying"][uid] = braking
        derived["success"][uid] = bool(details[-1]["ever_success"]) if details else False
        derived["steps"] = max(derived["steps"], len(rows_uid))
        _err(errors, f"{uid}:collision_free", not collision)
    return {"ok": not errors, "errors": errors, "derived": derived, "rows": ordered}


def audit_pair_rows(
    pair_rows: Iterable[Mapping[str, Any]],
    navigation_rows: Iterable[Mapping[str, Any]],
    *,
    minimum_distance: float = 2.0,
) -> dict[str, Any]:
    """Recompute synchronized pair minima from navigation truth rows."""

    errors: list[str] = []
    nav = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("uav_id"))): row for row in navigation_rows}
    rows = list(pair_rows)
    expected_keys = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("source")), str(row.get("target"))) for row in rows}
    episodes = {str(row.get("episode_id")) for row in rows}
    _err(errors, "pair_row_count", len(rows) == len(episodes) * MAX_STEPS * 3)
    seen_packets: set[tuple[str, int, str, str]] = set()
    by_episode_step: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        episode, step, source, target = str(row.get("episode_id")), _int(row.get("step")), str(row.get("source")), str(row.get("target"))
        key = (episode, step, source, target)
        _err(errors, f"pair_known:{key}", source != target and tuple(sorted((source, target), key=UIDS.index)) in PAIRS)
        _err(errors, f"pair_duplicate:{key}", key not in seen_packets)
        seen_packets.add(key)
        by_episode_step[(episode, step)].append(row)
        left = nav.get((episode, step, source)); right = nav.get((episode, step, target))
        if left is None or right is None:
            errors.append(f"pair_missing_navigation:{key}")
            continue
        distance, fraction = synchronized_pair_distance(left.get("true_position_pre_m"), left.get("true_position_post_m"), right.get("true_position_pre_m"), right.get("true_position_post_m"))
        _err(errors, f"pair_distance:{key}", math.isfinite(_float(row.get("distance_m"))) and math.isclose(_float(row.get("distance_m")), distance, rel_tol=0.0, abs_tol=2e-6))
        _err(errors, f"pair_fraction:{key}", math.isfinite(_float(row.get("time_fraction"))) and math.isclose(_float(row.get("time_fraction")), fraction, rel_tol=0.0, abs_tol=2e-6))
        _err(errors, f"pair_minimum:{key}", distance >= minimum_distance - 1e-8)
    for key, group in by_episode_step.items():
        _err(errors, f"pair_complete:{key}", {(str(row.get("source")), str(row.get("target"))) for row in group} == set(PAIRS))
    return {"ok": not errors, "errors": errors, "minimum_by_episode": {
        episode: min((_float(row.get("distance_m")) for row in rows if str(row.get("episode_id")) == episode), default=float("nan"))
        for episode in episodes
    }}


def _is_blocked(variant: str, source: str, target: str, step: int) -> bool:
    return variant == "cut" and 20 <= step <= 50 and ((source, target) in (("uav_00", "uav_02"), ("uav_02", "uav_00")))


def audit_flow_rows(
    flow_rows: Iterable[Mapping[str, Any]],
    navigation_rows: Iterable[Mapping[str, Any]],
    message_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit each directed communication edge against nav truth and packets."""

    errors: list[str] = []
    nav = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("uav_id"))): row for row in navigation_rows}
    messages = list(message_rows)
    arrivals = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("source")), str(row.get("target")))
                for row in messages if str(row.get("status")) == "arrived"}
    rows = list(flow_rows)
    episodes = {str(row.get("episode_id")) for row in rows}
    _err(errors, "flow_row_count", len(rows) == len(episodes) * MAX_STEPS * 6)
    by_key: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        episode, step = str(row.get("episode_id")), _int(row.get("step"))
        source, target = str(row.get("source")), str(row.get("target"))
        key = (episode, step, source, target)
        by_key[key] = row
        left, right = nav.get((episode, step, source)), nav.get((episode, step, target))
        _err(errors, f"flow_direction:{key}", source in UIDS and target in UIDS and source != target)
        if left is None or right is None:
            errors.append(f"flow_missing_navigation:{key}")
            continue
        pre_distance = float(np.linalg.norm(_vector(left.get("true_position_pre_m")) - _vector(right.get("true_position_pre_m"))))
        post_distance = float(np.linalg.norm(_vector(left.get("true_position_post_m")) - _vector(right.get("true_position_post_m"))))
        pre_potential = pre_distance <= POTENTIAL_RANGE + 1e-9
        potential = post_distance <= POTENTIAL_RANGE + 1e-9
        blocked = _is_blocked(str(row.get("variant")), source, target, step)
        expected_requested = 0.5 if pre_potential and not blocked else 0.0
        expected_applied = 0.5 if pre_potential and potential and not blocked else 0.0
        _err(errors, f"flow_distance:{key}", math.isfinite(_float(row.get("physical_distance_m"))) and math.isclose(_float(row.get("physical_distance_m")), post_distance, rel_tol=0.0, abs_tol=2e-6))
        _err(errors, f"flow_potential:{key}", _bool(row.get("potential")) == potential)
        _err(errors, f"flow_requested_bandwidth:{key}", math.isclose(_float(row.get("requested_bandwidth")), expected_requested, rel_tol=0.0, abs_tol=1e-9))
        _err(errors, f"flow_applied_bandwidth:{key}", math.isclose(_float(row.get("applied_bandwidth")), expected_applied, rel_tol=0.0, abs_tol=1e-9))
        _err(errors, f"flow_controlled_selection:{key}", _bool(row.get("controlled_link_selected")) == bool(expected_requested > 0))
        # Radio loss is stochastic: service permits sending, but does not guarantee it.
        _err(errors, f"flow_transmission:{key}", not _bool(row.get("transmitted")) or expected_applied > 0)
        message_time = row.get("message_send_time_s")
        age = _float(row.get("message_age_s"))
        fresh = False
        if not _missing(message_time) and math.isfinite(_float(message_time)):
            expected_age = step * DT_S - _float(message_time)
            _err(errors, f"flow_message_age:{key}", math.isfinite(age) and math.isclose(age, expected_age, rel_tol=0.0, abs_tol=2e-6))
            fresh = age <= STALE_AGE and age >= -1e-9
        else:
            _err(errors, f"flow_empty_age:{key}", _missing(row.get("message_age_s")))
        _err(errors, f"flow_fresh:{key}", _bool(row.get("fresh")) == fresh)
        effective = potential and fresh
        _err(errors, f"flow_effective:{key}", _bool(row.get("effective_visible")) == effective)
        if effective:
            _err(errors, f"flow_actor_neighbor_state:{key}", _close(row.get("actor_neighbor_state"), row.get("received_message_state"), 1e-8))
            _err(errors, f"flow_actor_neighbor_covariance:{key}", _close(row.get("actor_neighbor_covariance_diagonal"), row.get("received_message_covariance_diagonal"), 1e-8))
            node_state = _native(row.get("actor_node_state"))
            node_array = np.asarray(node_state, dtype=float).reshape(-1) if node_state is not None else np.empty(0)
            message_state = _vector(row.get("received_message_state"), 15)
            _err(errors, f"flow_actor_node_state:{key}", bool(node_array.size == 15 and np.allclose(node_array, message_state, rtol=0.0, atol=1e-8)))
            node_cov = row.get("actor_node_covariance_diagonal")
            node_cov_array = np.asarray(_native(node_cov), dtype=float).reshape(-1) if not _missing(node_cov) else np.empty(0)
            message_cov = _vector(row.get("received_message_covariance_diagonal"), 15)
            _err(errors, f"flow_actor_node_covariance:{key}", bool(node_cov_array.size == 15 and np.allclose(node_cov_array, message_cov, rtol=0.0, atol=1e-8)))
        else:
            _err(errors, f"flow_no_neighbor_leak:{key}", _missing(row.get("actor_neighbor_state")) and _missing(row.get("actor_node_state")))
        previous = row.get("previous_consumed_send_time_s")
        new_since = bool(not _missing(message_time) and (_missing(previous) or not math.isclose(_float(previous), _float(message_time), rel_tol=0.0, abs_tol=1e-12)))
        _err(errors, f"flow_new_packet:{key}", _bool(row.get("new_since_last_consumed")) == new_since)
        _err(errors, f"flow_reused:{key}", _bool(row.get("reused")) == bool(not _missing(message_time) and not new_since))
        newly_arrived = (episode, step, source, target) in arrivals
        _err(errors, f"flow_arrival_correlation:{key}", _bool(row.get("newly_arrived_this_step")) == newly_arrived)
        if not _missing(message_time):
            packet = str(row.get("packet_id", ""))
            expected_packet = packet_id(source, target, message_time)
            _err(errors, f"flow_packet_id:{key}", packet == expected_packet)
            # Visibility may repeat before consumption; uniqueness applies to fusion events.
    return {"ok": not errors, "errors": errors, "rows": rows, "indexed": by_key}


def audit_fusion_rows(
    fusion_rows: Iterable[Mapping[str, Any]],
    flow_rows: Iterable[Mapping[str, Any]],
    navigation_rows: Iterable[Mapping[str, Any]],
    *,
    cooperative: bool,
) -> dict[str, Any]:
    """Check accepted-update accounting and packet/fusion coupling."""

    errors: list[str] = []
    fusion = list(fusion_rows)
    flows = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("source")), str(row.get("target"))): row for row in flow_rows}
    nav = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("uav_id"))): row for row in navigation_rows}
    seen: set[tuple[str, str, str, str]] = set()
    accepted_by_target: Counter[tuple[str, str]] = Counter()
    for event in fusion:
        episode, step = str(event.get("episode_id")), _int(event.get("step"))
        source, target = str(event.get("source_uav_id")), str(event.get("target_uav_id"))
        flow = flows.get((episode, step, source, target))
        key = (episode, source, target, str(event.get("packet_id", "")))
        _err(errors, f"fusion_duplicate:{key}", key not in seen)
        seen.add(key)
        _err(errors, f"fusion_packet_matches_flow:{key}", flow is not None and str(event.get("packet_id")) == str(flow.get("packet_id")))
        accepted = _bool(event.get("accepted"))
        accepted_by_target[(episode, target)] += int(accepted)
        if flow is not None:
            _err(errors, f"fusion_accept_matches_flow:{key}", accepted == _bool(flow.get("fusion_accepted")))
            _err(errors, f"fusion_attempted_new_packet:{key}", _bool(flow.get("gate_new_packet")) is True)
        weight = _float(event.get("ci_weight"))
        _err(errors, f"fusion_ci_weight:{key}", math.isfinite(weight) and 0.0 <= weight <= 1.0)
        before, after, correction = _vector(event.get("state_before"), 15), _vector(event.get("state_after"), 15), _vector(event.get("state_correction"), 15)
        _err(errors, f"fusion_state_correction:{key}", _close(correction, after - before, 1e-7))
        if accepted:
            _err(errors, f"fusion_nonzero_or_zero_innovation:{key}", _finite(correction))
    if not cooperative:
        _err(errors, "fusion_off_no_events", len(fusion) == 0)
    for (episode, step, uid), row in nav.items():
        pre = _int(row.get("valid_neighbor_count_pre"), 0)
        post = _int(row.get("valid_neighbor_count_post"), 0)
        _err(errors, f"fusion_counter_delta_nonnegative:{episode}/{step}/{uid}", post >= pre)
        expected = sum(1 for event in fusion if str(event.get("episode_id")) == episode and _int(event.get("step")) == step and str(event.get("target_uav_id")) == uid and _bool(event.get("accepted")))
        _err(errors, f"fusion_counter_delta:{episode}/{step}/{uid}", post - pre == expected)
    return {"ok": not errors, "errors": errors,
            "accepted_by_target": {f"{episode}/{target}": int(count) for (episode, target), count in accepted_by_target.items()},
            "rows": fusion}


def audit_gate_rows(
    gate_rows: Iterable[Mapping[str, Any]],
    flow_rows: Iterable[Mapping[str, Any]],
    *,
    cooperative: bool,
) -> dict[str, Any]:
    """Audit raw measurement-gate decisions against directed flow rows."""

    errors: list[str] = []
    flows = {(str(row.get("episode_id")), _int(row.get("step")), str(row.get("source")), str(row.get("target"))): row for row in flow_rows}
    rows = list(gate_rows)
    seen: set[tuple[str, int, str, str]] = set()
    for gate in rows:
        episode, step = str(gate.get("episode_id")), _int(gate.get("step"))
        source, target = str(gate.get("source_uav_id")), str(gate.get("target_uav_id"))
        key = (episode, step, source, target)
        _err(errors, f"gate_direction:{key}", source in UIDS and target in UIDS and source != target)
        _err(errors, f"gate_duplicate:{key}", key not in seen)
        seen.add(key)
        flow = flows.get(key)
        if flow is None:
            errors.append(f"gate_missing_flow:{key}")
            continue
        attempted = _bool(gate.get("ekf_update_attempted"))
        accepted = _bool(gate.get("ekf_update_accepted"))
        new_packet = _bool(gate.get("new_packet"))
        arrived = _bool(gate.get("message_arrived"))
        effective = _bool(gate.get("effective_edge"))
        period = _bool(gate.get("update_period_passed"))
        selected = _bool(gate.get("selected_non_none"))
        _err(errors, f"gate_flow_new_packet:{key}", new_packet == _bool(flow.get("gate_new_packet")))
        _err(errors, f"gate_flow_attempted:{key}", attempted == _bool(flow.get("fusion_attempted")))
        _err(errors, f"gate_flow_accepted:{key}", accepted == _bool(flow.get("fusion_accepted")))
        _err(errors, f"gate_selected_receiver:{key}", selected == _bool(flow.get("measurement_selected")))
        _err(errors, f"gate_new_packet_prerequisites:{key}", not new_packet or (arrived and effective and period and selected))
        _err(errors, f"gate_accept_prerequisites:{key}", not accepted or attempted)
        if not cooperative:
            _err(errors, f"gate_fusion_off:{key}", not accepted and not attempted)
    return {"ok": not errors, "errors": errors, "rows": rows}


def _canonical_geometry_hash(scenario: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {
        "initial_positions": {uid: [float(v) for v in scenario.get("initial_positions", {}).get(uid, ())] for uid in UIDS},
        "goals": {uid: [float(v) for v in scenario.get("goals", {}).get(uid, ())] for uid in UIDS},
        "obstacle_specs": [], "bounds_min_m": [0.0, 0.0, 2.0], "bounds_max_m": [200.0, 200.0, 60.0],
    }
    if "obstacles" in scenario:
        payload["obstacles"] = []
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _expected_schedule() -> list[dict[str, Any]]:
    starts = {
        "dev": ((40.0, 40.0, 20.0), (40.0, 48.0, 20.0), (40.0, 44.0, 27.0)),
        "c0": ((80.0, 80.0, 22.0), (80.0, 88.0, 22.0), (80.0, 84.0, 29.0)),
        "c1": ((55.0, 50.0, 20.0), (55.0, 60.0, 20.0), (55.0, 55.0, 28.0)),
        "c2": ((40.0, 100.0, 25.0), (110.0, 100.0, 25.0), (77.0, 108.0, 31.0)),
    }
    goals = {
        "dev": tuple((x + 12.0, y, z) for x, y, z in starts["dev"]),
        "c0": tuple((x + 12.0, y, z) for x, y, z in starts["c0"]),
        "c1": ((75.0, 44.0, 20.0), (75.0, 66.0, 20.0), (75.0, 55.0, 28.0)),
        "c2": ((30.0, 100.0, 25.0), (125.0, 100.0, 25.0), (77.0, 108.0, 31.0)),
    }
    rows = (("three_nav_dev_normal", "dev", "normal", 2026115100, True, "development"),
            ("three_nav_c0_normal", "c0", "normal", 2026115200, True, "evaluation"),
            ("three_nav_c0_cut", "c0", "cut", 2026115200, True, "evaluation"),
            ("three_nav_c1_normal", "c1", "normal", 2026115201, True, "evaluation"),
            ("three_nav_c1_cut", "c1", "cut", 2026115201, True, "evaluation"),
            ("three_nav_c2_normal", "c2", "normal", 2026115202, True, "evaluation"),
            ("three_nav_c2_cut", "c2", "cut", 2026115202, True, "evaluation"),
            ("three_nav_c0_fusion_off", "c0", "fusion_off", 2026115200, False, "fusion_diagnostic"))
    result = []
    for episode_id, case_id, variant, seed, cooperative, stage in rows:
        scene = {"initial_positions": {uid: list(starts[case_id][i]) for i, uid in enumerate(UIDS)},
                 "initial_velocities": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
                 "initial_attitudes": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
                 "goals": {uid: list(goals[case_id][i]) for i, uid in enumerate(UIDS)},
                 "obstacle_specs": [], "obstacles": []}
        result.append({"episode_id": episode_id, "case_id": case_id, "variant": variant, "seed": seed,
                       "cooperative_ekf_enabled": cooperative, "stage": stage, "scenario": scene,
                       "geometry_sha256": _canonical_geometry_hash(scene)})
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _resolve_artifact(root: Path, data: Path, log: Path, name: str) -> Path | None:
    for candidate in (data / name, log / name, root / name):
        if candidate.is_file():
            return candidate
    return None


def _check_manifest(root: Path, data: Path, log: Path, lock: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    config_path = root / "data" / "configs" / f"{STEM}.json"
    _err(errors, "config_exists", config_path.is_file())
    if config_path.is_file():
        _err(errors, "config_sha256", _sha256(config_path).lower() == str(lock.get("config_sha256", "")).lower())
    sources = lock.get("sources") if isinstance(lock.get("sources"), Mapping) else {}
    required_sources = {
        "src/control/fixed_navigation.py", "src/control/fixed_navigation_execution.py", "src/learning/mappo.py",
        "src/experiments/three_uav_fixed_navigation.py", "src/experiments/three_uav_fixed_navigation_contract.py",
        "src/analysis/three_uav_fixed_navigation_audit.py", "src/simulation/environment.py", "src/simulation/ekf.py",
        "src/simulation/graph_interface.py", "src/simulation/topology.py", "src/simulation/motion_model.py",
        "src/simulation/sensor_model.py", "src/synthetic/channel_model.py",
    }
    _err(errors, "manifest_source_set", required_sources.issubset(set(str(key) for key in sources)))
    for relative, expected in sources.items():
        path = root / Path(str(relative))
        _err(errors, f"source_sha256:{relative}", path.is_file() and _sha256(path).lower() == str(expected).lower())
    protected_path = log / "protected_inputs_before.json"
    protected = _json(protected_path) if protected_path.is_file() else {}
    _err(errors, "protected_inputs_count_655", isinstance(protected, Mapping) and len(protected) == 655)
    if isinstance(protected, Mapping):
        for path_text, expected in protected.items():
            path = Path(str(path_text))
            # The archive is intentionally created after the protected snapshot.
            if path.name == "legacy_mappo_before.py.txt":
                continue
            _err(errors, f"protected_sha256:{path_text}", path.is_file() and _sha256(path).lower() == str(expected).lower())
    archive = data / "legacy_mappo_before.py.txt"
    _err(errors, "original_mappo_archive_sha256", archive.is_file() and _sha256(archive).lower() == "a815889f0d05b038d4f20550d763ff59a2782d6439ca4b47e25e69be6d103fcd")
    expected_schedule = _expected_schedule()
    actual_schedule = lock.get("schedule")
    _err(errors, "manifest_schedule_count", isinstance(actual_schedule, list) and len(actual_schedule) == EPISODES)
    if isinstance(actual_schedule, list):
        for expected, actual in zip(expected_schedule, actual_schedule):
            key = str(expected["episode_id"])
            for field in ("episode_id", "case_id", "variant", "seed", "cooperative_ekf_enabled", "stage"):
                _err(errors, f"schedule:{key}:{field}", actual.get(field) == expected[field])
            _err(errors, f"schedule:{key}:geometry", str(actual.get("geometry_sha256")) == expected["geometry_sha256"])
            _err(errors, f"schedule:{key}:geometry_recomputed", _canonical_geometry_hash(actual.get("scenario", {})) == str(actual.get("geometry_sha256")))
    for field, value in (("controlled_authority", True), ("no_training", True), ("maximum_joint_steps", JOINT_STEPS),
                         ("learned_motion_dim", 0), ("neural_samples_actual_execution", 0), ("truth_initialization_inherited", True),
                         ("no_replay_or_tuning", True)):
        _err(errors, f"manifest:{field}", lock.get(field) == value)
    simulation = lock.get("simulation_config", {}) if isinstance(lock.get("simulation_config"), Mapping) else {}
    motion = simulation.get("motion", {}) if isinstance(simulation.get("motion"), Mapping) else {}
    for field, value in (("dt_s", 0.2), ("max_speed_mps", 12.0), ("max_acceleration_mps2", 3.0),
                         ("bounds_min_m", [0.0, 0.0, 2.0]), ("bounds_max_m", [200.0, 200.0, 60.0]),
                         ("safety_distance_m", 2.0)):
        actual = motion.get(field)
        _err(errors, f"simulation_motion:{field}", _close(actual, value, 1e-12) if isinstance(value, list) else math.isclose(_float(actual), float(value), rel_tol=0.0, abs_tol=1e-12))
    _err(errors, "simulation_uav_count", simulation.get("uav_count") == 3)
    _err(errors, "simulation_max_steps", simulation.get("max_steps") == MAX_STEPS)
    _err(errors, "simulation_indexed_stream", simulation.get("random_stream_mode") == "indexed-v1")
    details.update({"protected_file_count": len(protected) if isinstance(protected, Mapping) else None,
                    "source_count": len(sources), "archive_sha256": _sha256(archive) if archive.is_file() else None,
                    "schedule_count": len(actual_schedule) if isinstance(actual_schedule, list) else 0})
    return details


def _check_config(root: Path, config_payload: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    expected = {"position_gain_s_inv": 0.6, "velocity_gain_s_inv": 1.6, "cruise_speed_mps": 2.5,
                "waypoint_radius_m": 0.8, "max_delta_v_norm_mps": 0.6, "dt_s": 0.2}
    nav_path = root / str(config_payload.get("navigation_config", ""))
    dynamics_path = root / str(config_payload.get("baseline_dynamics_config", ""))
    _err(errors, "navigation_config_exists", nav_path.is_file())
    _err(errors, "dynamics_config_exists", dynamics_path.is_file())
    nav_payload = _json(nav_path) if nav_path.is_file() else {}
    controller = nav_payload.get("controller", {})
    for field, value in expected.items():
        _err(errors, f"controller:{field}", math.isclose(_float(controller.get(field)), value, rel_tol=0.0, abs_tol=1e-12))
    _err(errors, "config_uav_ids", config_payload.get("uav_ids") == list(UIDS))
    for field, value in (("joint_step_budget", JOINT_STEPS), ("episodes", EPISODES), ("max_steps", MAX_STEPS),
                         ("gnss_mode", "denied"), ("random_stream_mode", "indexed-v1"), ("safety2_enabled", True),
                         ("assist_mask", "00000"), ("safety_distance_m", 2.0), ("strict_distance_m", 1.0),
                         ("strict_speed_mps", 0.5), ("strict_hold_steps", 5), ("continuous_pair_distance_minimum_m", 2.0)):
        _err(errors, f"config:{field}", config_payload.get(field) == value)
    intervention = config_payload.get("intervention", {})
    for field, value in (("start_step", 20), ("end_step", 50), ("drained_start_step", 25), ("recovery_start_step", 51), ("unblocked_allocation", 0.5), ("redistribute_unused_bandwidth", False)):
        _err(errors, f"intervention:{field}", intervention.get(field) == value)
    _err(errors, "intervention:blocked_edges", intervention.get("blocked_edges") == [["uav_00", "uav_02"], ["uav_02", "uav_00"]])
    return {"navigation_config": str(nav_path), "dynamics_config": str(dynamics_path)}


def _compare_summary(summary: Mapping[str, Any], expected_episode_ids: Sequence[str], errors: list[str]) -> dict[str, Any]:
    counts = summary.get("counts", {})
    expected_counts = {"joint_environment_steps": JOINT_STEPS, "ppo_updates": 0, "optimizer_steps": 0,
                       "checkpoint_loads": 0, "rollout_additions": 0, "policy_samples_executed": 0,
                       "trainer_constructions_actual_run": 0}
    for field, value in expected_counts.items():
        _err(errors, f"summary_count:{field}", _int(counts.get(field)) == value)
    episodes = summary.get("episodes")
    _err(errors, "summary_episode_count", isinstance(episodes, list) and len(episodes) == EPISODES)
    if isinstance(episodes, list):
        _err(errors, "summary_episode_ids", [str(item.get("episode_id")) for item in episodes] == list(expected_episode_ids))
        for item in episodes:
            _err(errors, f"summary_steps:{item.get('episode_id')}", _int(item.get("steps")) == MAX_STEPS)
            _err(errors, f"summary_gate:{item.get('episode_id')}", _bool(item.get("gate_passed")) is True)
    _err(errors, "summary_all_scheduled", _bool(summary.get("all_scheduled_gates_passed")) is True)
    _err(errors, "summary_authority", summary.get("authority") == "controlled_only_neural_probes_never_executed")
    return {"counts": dict(counts), "episode_count": len(episodes) if isinstance(episodes, list) else 0}


def _check_fixture_and_probes(root: Path, data: Path, log: Path, errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    comparison_path = _resolve_artifact(root, data, log, "legacy_action_fixture_comparison.json")
    if comparison_path is None:
        errors.append("missing:legacy_action_fixture_comparison.json")
    else:
        comparison = _json(comparison_path)
        expected = {"passed": True, "configurations": 4, "actions": 8, "exact_equality": True, "environment_steps": 0, "optimizer_steps": 0}
        for key, value in expected.items():
            _err(errors, f"legacy_fixture:{key}", comparison.get(key) == value)
        result["legacy_fixture"] = comparison
    probes_path = _resolve_artifact(root, data, log, "interface_probes.json")
    if probes_path is None:
        errors.append("missing:interface_probes.json")
    else:
        probes = _json(probes_path)
        expected = {"passed": True, "environment_steps": 0, "optimizer_steps": 0, "rollout_length": 0,
                    "checkpoint_loads": 0, "parameters_unchanged": True, "policy_samples_executed": 0,
                    "interface_only_policy_executed": False}
        for key, value in expected.items():
            _err(errors, f"interface_probe:{key}", probes.get(key) == value)
        result["interface_probes"] = probes
    return result


def audit_artifacts(
    log_root: str | Path = DEFAULT_LOG_ROOT,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output: str | Path | None = None,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Audit saved artifacts and write an exclusive independent report."""

    root = Path(project_root).resolve()
    log = Path(log_root)
    data = Path(data_root)
    if not log.is_absolute():
        log = root / log
    if not data.is_absolute():
        data = root / data
    errors: list[str] = []
    lock_path = data / "execution_manifest.lock.json"
    if not lock_path.is_file():
        errors.append(f"missing:{lock_path}")
        lock: Mapping[str, Any] = {}
    else:
        lock = _json(lock_path)
    manifest = _check_manifest(root, data, log, lock, errors) if lock else {}
    config_payload = _json(root / "data" / "configs" / f"{STEM}.json") if (root / "data" / "configs" / f"{STEM}.json").is_file() else {}
    config_details = _check_config(root, config_payload, errors) if config_payload else {}
    schedule = _expected_schedule()
    by_id = {row["episode_id"]: row for row in schedule}
    tables: dict[str, list[dict[str, str]]] = {}
    for table in TABLES:
        path = log / f"{table}.csv"
        if not path.is_file():
            errors.append(f"missing:{path}")
            tables[table] = []
        else:
            tables[table] = _read_csv(path)
    nav_rows = tables["navigation"]
    episodes_seen = {str(row.get("episode_id")) for row in nav_rows}
    _err(errors, "navigation_episode_count", episodes_seen == set(by_id))
    nav_results: dict[str, dict[str, Any]] = {}
    for episode_id, case in by_id.items():
        group = [row for row in nav_rows if str(row.get("episode_id")) == episode_id]
        goals = case["scenario"]["goals"]
        nav_results[episode_id] = audit_navigation_group(group, config=FixedNavigationConfig(), expected_goal_by_uid=goals, case_id=case["case_id"])
        if not nav_results[episode_id]["ok"]:
            errors.extend(f"{episode_id}:{item}" for item in nav_results[episode_id]["errors"])
    pair_result = audit_pair_rows(tables["pairs"], nav_rows)
    flow_result = audit_flow_rows(tables["flows"], nav_rows, tables["messages"])
    fusion_results: dict[str, dict[str, Any]] = {}
    gate_results: dict[str, dict[str, Any]] = {}
    for episode_id, case in by_id.items():
        fusion_group = [row for row in tables["fusion_events"] if str(row.get("episode_id")) == episode_id]
        episode_flows = [row for row in tables["flows"] if str(row.get("episode_id")) == episode_id]
        episode_nav = [row for row in nav_rows if str(row.get("episode_id")) == episode_id]
        episode_gates = [row for row in tables["gates"] if str(row.get("episode_id")) == episode_id]
        fusion_results[episode_id] = audit_fusion_rows(fusion_group, episode_flows, episode_nav, cooperative=bool(case["cooperative_ekf_enabled"]))
        gate_results[episode_id] = audit_gate_rows(episode_gates, episode_flows, cooperative=bool(case["cooperative_ekf_enabled"]))
        if not fusion_results[episode_id]["ok"]:
            errors.extend(f"{episode_id}:{item}" for item in fusion_results[episode_id]["errors"])
        if not gate_results[episode_id]["ok"]:
            errors.extend(f"{episode_id}:{item}" for item in gate_results[episode_id]["errors"])
    if not pair_result["ok"]:
        errors.extend(f"pairs:{item}" for item in pair_result["errors"])
    if not flow_result["ok"]:
        errors.extend(f"flows:{item}" for item in flow_result["errors"])
    joint_rows = tables["joint"]
    _err(errors, "joint_row_count_960", len(joint_rows) == JOINT_STEPS)
    for row in joint_rows:
        episode, step = str(row.get("episode_id")), _int(row.get("step"))
        pair_group = [item for item in tables["pairs"] if str(item.get("episode_id")) == episode and _int(item.get("step")) == step]
        if pair_group:
            expected_min = min(_float(item.get("distance_m")) for item in pair_group)
            _err(errors, f"joint_minimum:{episode}/{step}", math.isclose(_float(row.get("minimum_synchronized_pair_distance_m")), expected_min, rel_tol=0.0, abs_tol=2e-6))
        _err(errors, f"joint_collision:{episode}/{step}", not _bool(row.get("collision")))
    joint_index = {(str(row.get("episode_id")), _int(row.get("step"))): row for row in joint_rows}
    joint_success_by_episode: dict[str, bool] = {}
    for episode_id in by_id:
        nav_by_step = defaultdict(dict)
        for row in nav_rows:
            if str(row.get("episode_id")) == episode_id:
                nav_by_step[_int(row.get("step"))][str(row.get("uav_id"))] = row
        joint_hold = 0
        joint_ever = False
        joint_first: int | None = None
        terminal_seen = False
        for step in range(1, MAX_STEPS + 1):
            nav_step = nav_by_step.get(step, {})
            all_qualifying = len(nav_step) == len(UIDS) and all(_bool(nav_step[uid].get("qualifying")) for uid in UIDS)
            joint_hold = joint_hold + 1 if all_qualifying else 0
            if joint_hold >= STRICT_HOLD:
                joint_ever = True
                if joint_first is None:
                    joint_first = step
            joint = joint_index.get((episode_id, step))
            if joint is not None:
                _err(errors, f"joint_hold:{episode_id}/{step}", _int(joint.get("joint_hold_steps")) == joint_hold)
                _err(errors, f"joint_ever:{episode_id}/{step}", _bool(joint.get("joint_ever_success")) == joint_ever)
                _err(errors, f"joint_first:{episode_id}/{step}", _missing(joint.get("joint_first_success_step")) if joint_first is None else _int(joint.get("joint_first_success_step")) == joint_first)
                terminal = _bool(joint.get("parent_terminated")) or _bool(joint.get("parent_truncated"))
                _err(errors, f"no_early_terminal:{episode_id}/{step}", not terminal or step == MAX_STEPS)
                if terminal:
                    terminal_seen = True
                _err(errors, f"no_rows_after_terminal:{episode_id}/{step}", not terminal_seen or step == MAX_STEPS)
        joint_success_by_episode[episode_id] = joint_ever
    # Common-random-number pairs must be identical through the step before
    # the intervention.  Compare raw saved fields, rather than summary hashes.
    for case_id in ("c0", "c1", "c2"):
        normal_id, cut_id = f"three_nav_{case_id}_normal", f"three_nav_{case_id}_cut"
        normal_rows = {( _int(row.get("step")), str(row.get("uav_id"))): row for row in nav_rows if str(row.get("episode_id")) == normal_id}
        cut_rows = {( _int(row.get("step")), str(row.get("uav_id"))): row for row in nav_rows if str(row.get("episode_id")) == cut_id}
        for key in set(normal_rows) & set(cut_rows):
            if key[0] > 19:
                continue
            left, right = normal_rows[key], cut_rows[key]
            for field in ("true_position_pre_m", "true_velocity_pre_mps", "true_position_post_m", "true_velocity_post_mps", "ekf_state_pre", "ekf_state_post"):
                _err(errors, f"crn_prefix:{case_id}:{key}:{field}", _close(left.get(field), right.get(field), 0.0))
    # Geometry-specific communication witnesses are independently derived from
    # the navigation truth, and make the c2 threshold crossing reviewable.
    geometry_details: dict[str, Any] = {}
    for case_id in ("c0", "c1", "c2"):
        ids = [f"three_nav_{case_id}_normal", f"three_nav_{case_id}_cut"]
        case_flows = [row for row in tables["flows"] if str(row.get("episode_id")) in ids]
        if case_id in ("c0", "c1"):
            stable = all(_bool(row.get("potential")) for row in case_flows)
            _err(errors, f"{case_id}:potential_graph_stable", stable)
            geometry_details[case_id] = {"potential_rows": len(case_flows), "potential_graph_stable": stable}
        else:
            pair_distances = [_float(row.get("physical_distance_m")) for row in case_flows if row.get("source") == "uav_00" and row.get("target") == "uav_01" and str(row.get("episode_id")) == ids[0]]
            bridge = [row for row in case_flows if str(row.get("episode_id")) == ids[0] and row.get("source") in {"uav_00", "uav_01"} and row.get("target") == "uav_02"]
            bridge += [row for row in case_flows if str(row.get("episode_id")) == ids[0] and row.get("source") == "uav_02" and row.get("target") in {"uav_00", "uav_01"}]
            graph_changes = 0
            for source, target in (("uav_00", "uav_01"), ("uav_01", "uav_00")):
                values = [_bool(row.get("potential")) for row in sorted(case_flows, key=lambda item: _int(item.get("step"))) if row.get("source") == source and row.get("target") == target and str(row.get("episode_id")) == ids[0]]
                graph_changes += sum(left != right for left, right in zip(values, values[1:]))
            _err(errors, "c2:distance_min_lt_75", bool(pair_distances) and min(pair_distances) < 75.0)
            _err(errors, "c2:distance_max_gt_75", bool(pair_distances) and max(pair_distances) > 75.0)
            _err(errors, "c2:distance_max_le_100", bool(pair_distances) and max(pair_distances) <= 100.0 + 1e-9)
            _err(errors, "c2:bridge_four_edges", len(bridge) == MAX_STEPS * 4 and all(_bool(row.get("potential")) for row in bridge))
            geometry_details[case_id] = {"distance_min_m": min(pair_distances) if pair_distances else None,
                                         "distance_max_m": max(pair_distances) if pair_distances else None,
                                         "visible_graph_changes": graph_changes, "bridge_rows": len(bridge)}
    probe_details = _check_fixture_and_probes(root, data, log, errors)
    summary_path = log / "execution_summary.json"
    summary_payload: Mapping[str, Any] = {}
    if not summary_path.is_file():
        errors.append(f"missing:{summary_path}")
        summary_details = {}
    else:
        summary_payload = _json(summary_path)
        summary_details = _compare_summary(summary_payload, [row["episode_id"] for row in schedule], errors)
    # Communication and belief-trajectory witnesses are separate from the
    # navigation gate.  They prevent an all-zero or copied diagnostic from
    # being presented as a cooperative mechanism result.
    for case_id in ("c0", "c1", "c2"):
        normal_id = f"three_nav_{case_id}_normal"
        accepted = sum(_bool(row.get("accepted")) for row in tables["fusion_events"] if str(row.get("episode_id")) == normal_id)
        _err(errors, f"{case_id}:normal_cooperative_updates_positive", accepted > 0)
    off_id = "three_nav_c0_fusion_off"
    off_flows = [row for row in tables["flows"] if str(row.get("episode_id")) == off_id]
    _err(errors, "fusion_off:communications_still_transmit", any(_bool(row.get("transmitted")) for row in off_flows))
    _err(errors, "fusion_off:communications_still_arrive", any(_bool(row.get("newly_arrived_this_step")) for row in off_flows))
    summary_episodes = summary_payload.get("episodes", []) if isinstance(summary_payload, Mapping) else []
    summary_by_id = {str(item.get("episode_id")): item for item in summary_episodes} if isinstance(summary_episodes, list) else {}
    normal_hash = summary_by_id.get("three_nav_c0_normal", {}).get("belief_trajectory_sha256")
    off_hash = summary_by_id.get(off_id, {}).get("belief_trajectory_sha256")
    _err(errors, "fusion_off:belief_trajectory_differs", bool(normal_hash and off_hash and str(normal_hash) != str(off_hash)))
    # Apply the contract-level gates only after independent row checks.
    episode_gates: dict[str, bool] = {}
    for episode_id, case in by_id.items():
        nav = nav_results[episode_id]
        accepted = sum(_bool(row.get("accepted")) for row in tables["fusion_events"] if str(row.get("episode_id")) == episode_id)
        parking_passed = all(nav["derived"]["success"].get(uid, False) for uid in UIDS) and joint_success_by_episode.get(episode_id, False)
        navigation_passed = nav["ok"] and (parking_passed or case["variant"] == "fusion_off")
        mechanism = accepted > 0 if case["cooperative_ekf_enabled"] else accepted == 0
        cutoff = True
        if case["variant"] == "cut":
            for source, target in (("uav_00", "uav_02"), ("uav_02", "uav_00")):
                blocked = [row for row in tables["flows"] if str(row.get("episode_id")) == episode_id and row.get("source") == source and row.get("target") == target and 20 <= _int(row.get("step")) <= 50]
                drained = [row for row in blocked if _int(row.get("step")) >= 25]
                recovered = [row for row in tables["flows"] if str(row.get("episode_id")) == episode_id and row.get("source") == source and row.get("target") == target and _int(row.get("step")) >= 51]
                cutoff = cutoff and bool(blocked) and all(_float(row.get("applied_bandwidth")) == 0 and not _bool(row.get("transmitted")) for row in blocked)
                cutoff = cutoff and all(not any(_bool(row.get(name)) for name in ("newly_arrived_this_step", "effective_visible", "fusion_accepted")) for row in drained)
                cutoff = cutoff and all(any(_bool(row.get(name)) for row in recovered) for name in ("transmitted", "newly_arrived_this_step", "fusion_accepted"))
        episode_gates[episode_id] = bool(navigation_passed and mechanism and cutoff)
        _err(errors, f"independent_episode_gate:{episode_id}", episode_gates[episode_id])
    report: dict[str, Any] = {
        "schema_version": "three_uav_fixed_navigation_independent_audit_v1",
        "status": "passed" if not errors else "failed", "passed": not errors,
        "project_root": str(root), "log_root": str(log.resolve()), "data_root": str(data.resolve()),
        "manifest": manifest, "config": config_details, "episode_gates": episode_gates,
        "geometry": geometry_details,
        "row_counts": {table: len(rows) for table, rows in tables.items()},
        "navigation": {episode: {"ok": result["ok"], "errors": len(result["errors"]), "derived": result["derived"]} for episode, result in nav_results.items()},
        "pair_check": {"ok": pair_result["ok"], "errors": len(pair_result["errors"])},
        "flow_check": {"ok": flow_result["ok"], "errors": len(flow_result["errors"])},
        "fusion": {episode: {"ok": result["ok"], "errors": len(result["errors"]), "accepted_by_target": result["accepted_by_target"]} for episode, result in fusion_results.items()},
        "gates": {episode: {"ok": result["ok"], "errors": len(result["errors"])} for episode, result in gate_results.items()},
        "summary": summary_details, "probes": probe_details,
        "errors": errors[:1000], "error_count": len(errors),
        "read_only_execution": {"environment_steps": 0, "optimizer_steps": 0, "checkpoint_loads": 0, "rollout_length": 0, "ppo_updates": 0, "trainer_constructions": 0, "policy_samples_executed": 0},
        "independent": True, "controlled_authority": "controlled_only_neural_probes_never_executed",
        "no_learning_data_used": True,
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
    result = audit_artifacts(args.log_root, data_root=args.data_root, output=args.output)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "errors": result["error_count"]}, ensure_ascii=False, sort_keys=True))


__all__ = [
    "audit_artifacts", "audit_navigation_row", "audit_navigation_group", "audit_pair_rows", "audit_flow_rows",
    "audit_fusion_rows", "derive_strict_status", "synchronized_pair_distance", "packet_id", "main", "DEFAULT_LOG_ROOT", "DEFAULT_DATA_ROOT",
]


if __name__ == "__main__":
    main()
