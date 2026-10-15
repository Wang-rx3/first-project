"""Pure measurement-selection and calibration contracts for three UAVs.

This module deliberately contains no environment, simulator, learning, or
training imports.  The functions operate on the local observations and event
records supplied by the caller, so that legality and the training credit
proxy can be tested independently of a runtime implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


UIDS = ("uav_00", "uav_01", "uav_02")
_NONE = "none"
_BOUNDS_MIN = (0.0, 0.0, 2.0)
_BOUNDS_MAX = (200.0, 200.0, 60.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _hash_geometry(scenario: Mapping[str, Any]) -> str:
    payload = {
        "initial_positions": {
            uid: [float(x) for x in scenario["initial_positions"][uid]] for uid in UIDS
        },
        "goals": {uid: [float(x) for x in scenario["goals"][uid]] for uid in UIDS},
        "obstacles": [],
        "obstacle_specs": [],
        "bounds_min_m": list(_BOUNDS_MIN),
        "bounds_max_m": list(_BOUNDS_MAX),
    }
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _scene(
    scene_id: str,
    split: str,
    seed: int,
    starts: Sequence[Sequence[float]],
    goals: Sequence[Sequence[float]],
    requires_graph_change: bool,
) -> dict[str, Any]:
    if len(starts) != 3 or len(goals) != 3:
        raise ValueError("a scene requires exactly three starts and goals")
    scenario = {
        "initial_positions": {
            uid: [float(x) for x in starts[index]] for index, uid in enumerate(UIDS)
        },
        "initial_velocities": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
        "initial_attitudes": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
        "goals": {uid: [float(x) for x in goals[index]] for index, uid in enumerate(UIDS)},
        "obstacles": [],
        "obstacle_specs": [],
    }
    return {
        "scene_id": str(scene_id),
        "split": str(split),
        "seed": int(seed),
        "requires_graph_change": bool(requires_graph_change),
        "scenario": scenario,
        "geometry_sha256": _hash_geometry(scenario),
    }


def _translate(points: Sequence[Sequence[float]], offset: Sequence[float]) -> list[list[float]]:
    return [[float(point[axis]) + float(offset[axis]) for axis in range(3)] for point in points]


def make_scene_splits() -> dict[str, list[dict[str, Any]]]:
    """Return the fixed four-scene training and three-scene validation splits."""

    compact_x = ((40.0, 40.0, 20.0), (40.0, 52.0, 20.0), (40.0, 46.0, 30.0))
    compact_y = ((55.0, 65.0, 22.0), (67.0, 65.0, 22.0), (61.0, 65.0, 32.0))
    bridge_x = ((35.0, 125.0, 24.0), (103.0, 125.0, 24.0), (69.0, 139.0, 34.0))
    bridge_x_goals = ((80.0, 113.0, 24.0), (153.0, 137.0, 24.0), (119.0, 139.0, 34.0))
    bridge_y = ((130.0, 35.0, 22.0), (130.0, 103.0, 22.0), (144.0, 69.0, 32.0))
    bridge_y_goals = ((118.0, 80.0, 22.0), (142.0, 153.0, 22.0), (144.0, 119.0, 32.0))

    train = [
        _scene("train_compact_x", "train", 2026116100, compact_x, _translate(compact_x, (50, 0, 0)), False),
        _scene("train_compact_y", "train", 2026116101, compact_y, _translate(compact_y, (0, 50, 0)), False),
        _scene("train_bridge_x", "train", 2026116102, bridge_x, bridge_x_goals, True),
        _scene("train_bridge_y", "train", 2026116103, bridge_y, bridge_y_goals, True),
    ]

    # Validation origins and goals are translated copies of three base
    # geometries.  The translations keep every point inside the fixed box and
    # avoid every training origin while retaining static geometry.
    val_compact_starts = _translate(compact_x, (8.0, 8.0, 0.0))
    val_compact_goals = _translate(_translate(compact_x, (50.0, 0.0, 0.0)), (8.0, 8.0, 0.0))
    val_bridge_x_starts = _translate(bridge_x, (8.0, -8.0, 0.0))
    val_bridge_x_goals = _translate(bridge_x_goals, (8.0, -8.0, 0.0))
    val_bridge_y_starts = _translate(bridge_y, (-8.0, 8.0, 0.0))
    val_bridge_y_goals = _translate(bridge_y_goals, (-8.0, 8.0, 0.0))
    validation = [
        _scene("validation_compact_x", "validation", 2026116400, val_compact_starts, val_compact_goals, False),
        _scene("validation_bridge_x", "validation", 2026116401, val_bridge_x_starts, val_bridge_x_goals, True),
        _scene("validation_bridge_y", "validation", 2026116402, val_bridge_y_starts, val_bridge_y_goals, True),
    ]
    return {"train": copy.deepcopy(train), "validation": copy.deepcopy(validation)}


def _own_local(local: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = local.get("local") if isinstance(local, Mapping) else None
    return nested if isinstance(nested, Mapping) else local


def _uid_from_local(local: Mapping[str, Any]) -> str:
    value = local.get("uav_id")
    if value is not None:
        return str(value)
    node_ids = local.get("node_ids")
    if not isinstance(node_ids, (str, bytes)):
        try:
            values = list(node_ids)
        except (TypeError, ValueError):
            values = []
        if values:
            return str(values[0])
    return ""


def _ids(value: Any) -> set[str]:
    if isinstance(value, (str, bytes)):
        return {str(value)}
    try:
        return {str(item) for item in value}
    except (TypeError, ValueError):
        return set()


def _base_mask(local: Mapping[str, Any]) -> np.ndarray:
    value = local.get("base_action_mask", local.get("action_mask"))
    if isinstance(value, Mapping):
        out = np.asarray(
            [bool(value.get(uid, False)) for uid in UIDS]
            + [bool(value.get(_NONE, value.get("none", True)))],
            dtype=bool,
        )
        return out
    try:
        array = np.asarray(value, dtype=bool).reshape(-1) if value is not None else np.ones(4, dtype=bool)
    except (TypeError, ValueError):
        return np.zeros(4, dtype=bool)
    out = np.zeros(4, dtype=bool)
    out[: min(4, array.size)] = array[:4]
    return out


def _finite_covariance(value: Any, *, state_dim: int | None = None) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] != array.shape[1] or (state_dim and array.shape != (state_dim, state_dim)):
        return None
    if not np.isfinite(array).all() or not np.allclose(array, array.T, rtol=1e-9, atol=1e-9):
        return None
    try:
        eigenvalues = np.linalg.eigvalsh(array)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) < -1e-9:
        return None
    return array


def _valid_message(local: Mapping[str, Any], source: str) -> bool:
    messages = local.get("neighbor_messages")
    if not isinstance(messages, Mapping) or not isinstance(messages.get(source), Mapping):
        return False
    message = messages[source]
    try:
        estimate = np.asarray(message.get("estimate", message.get("state_estimate")), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if estimate.shape != (15,) or not np.isfinite(estimate).all():
        return False
    covariance = _finite_covariance(message.get("covariance", message.get("state_covariance")), state_dim=15)
    return covariance is not None


def _eligibility_value(local: Mapping[str, Any], source: str) -> bool:
    value = local.get("measurement_update_eligibility")
    if value is None:
        return False
    if isinstance(value, Mapping):
        return value.get(source) is True or bool(value.get(source, False))
    if isinstance(value, (str, bytes)):
        return False
    try:
        array = np.asarray(value, dtype=bool).reshape(-1)
    except (TypeError, ValueError):
        return bool(value) if isinstance(value, (bool, np.bool_)) else False
    if array.size == 1:
        return bool(array[0])
    index = UIDS.index(source)
    return index < array.size and bool(array[index])


def _legal_sources(local: Mapping[str, Any], *, apply_base_mask: bool = True) -> list[str]:
    uid = _uid_from_local(local)
    candidate = _ids(local.get("candidate_measurement_sources", ()))
    effective = _ids(local.get("effective_sources", ()))
    available = local.get("uwb_available", {})
    if not isinstance(available, Mapping):
        available = {}
    try:
        budget = float(local.get("current_budget_remaining", local.get("measurement_budget_remaining", float("nan"))))
    except (TypeError, ValueError):
        budget = float("nan")
    base = _base_mask(local)
    legal: list[str] = []
    for index, source in enumerate(UIDS):
        if source == uid or source not in candidate or source not in effective:
            continue
        if available.get(source) is not True and not isinstance(available.get(source), np.bool_):
            continue
        if not bool(available.get(source, False)) or not _eligibility_value(local, source):
            continue
        if not math.isfinite(budget) or budget <= 0.0 or not _valid_message(local, source):
            continue
        if apply_base_mask and not bool(base[index]):
            continue
        legal.append(source)
    return legal


def admissible_mask(local: Mapping[str, Any], uid: str | None = None) -> np.ndarray:
    """Return a conservative four-action mask: three sources followed by ``none``."""

    if not isinstance(local, Mapping):
        return np.zeros(4, dtype=bool)
    local_view = dict(_own_local(local))
    if uid is not None:
        local_view["uav_id"] = str(uid)
    base = _base_mask(local_view)
    output = np.zeros(4, dtype=bool)
    # ``none`` is the fail-closed fallback and therefore remains legal even
    # when a malformed/empty base mask or a self-only observation blocks every
    # peer source.
    output[3] = True
    for source in _legal_sources(local_view):
        output[UIDS.index(source)] = True
    return output


def admissible_observations(observations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(observations, Mapping):
        raise ValueError("observations must be a mapping keyed by all UAV IDs")
    if set(observations) != set(UIDS):
        raise ValueError("observations must contain exactly the three UAV IDs")
    result: dict[str, dict[str, Any]] = {}
    for uid in UIDS:
        if not isinstance(observations[uid], Mapping):
            raise ValueError(f"observation for {uid} must be a mapping")
        local = copy.deepcopy(dict(_own_local(observations[uid])))
        local["uav_id"] = uid
        local["action_mask"] = admissible_mask(local, uid=uid)
        result[uid] = local
    return result


def fixed_bandwidth(local: Mapping[str, Any], uid: str | None = None, fraction: float = 0.5) -> dict[str, float]:
    try:
        share = float(fraction)
    except (TypeError, ValueError):
        raise ValueError("fraction must be finite and in [0, 1]") from None
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("fraction must be finite and in [0, 1]")
    local_view = _own_local(local) if isinstance(local, Mapping) else {}
    owner = str(uid if uid is not None else local_view.get("uav_id", ""))
    potential = _ids(local_view.get("potential_targets", ()))
    return {target: share for target in UIDS if target in potential and target != owner}


def communication_actions(
    observations: Mapping[str, Mapping[str, Any]], selections: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(observations, Mapping) or set(observations) != set(UIDS):
        raise ValueError("observations must contain exactly the three UAV IDs")
    if not isinstance(selections, Mapping) or set(selections) != set(UIDS):
        raise ValueError("selections must contain exactly the three UAV IDs")
    actions: dict[str, dict[str, Any]] = {}
    for uid in UIDS:
        raw = observations[uid]
        if not isinstance(raw, Mapping):
            raise ValueError(f"observation for {uid} must be a mapping")
        local = dict(_own_local(raw))
        local["uav_id"] = uid
        selected_value = selections[uid]
        if isinstance(selected_value, Mapping):
            selected_value = selected_value.get("measurement_uav_id")
        selected = str(selected_value) if selected_value is not None else _NONE
        mask = admissible_mask(local, uid=uid)
        if selected not in UIDS or not bool(mask[UIDS.index(selected)]):
            selected = _NONE
        actions[uid] = {
            "measurement_uav_id": selected,
            "bandwidth": fixed_bandwidth(local, uid=uid),
        }
    return actions


def random_selection(local: Mapping[str, Any], rng: Any) -> str:
    if rng is None:
        raise ValueError("rng is required")
    mask = admissible_mask(local)
    choices = [UIDS[index] for index in range(3) if bool(mask[index])]
    if bool(mask[3]):
        choices.append(_NONE)
    if not choices:
        return _NONE
    if hasattr(rng, "choice"):
        return str(rng.choice(choices))
    raise ValueError("rng must provide choice")


def round_robin_selection(local: Mapping[str, Any], step: int) -> str:
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 1:
        raise ValueError("step must be a positive integer")
    legal = _legal_sources(_own_local(local) if isinstance(local, Mapping) else {})
    if not legal:
        return _NONE
    return legal[(int(step) - 1) % len(legal)]


def _range_variance(value: Any) -> float:
    try:
        variance = float(value)
    except (TypeError, ValueError):
        raise ValueError("range_variance_m2 must be finite and positive") from None
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("range_variance_m2 must be finite and positive")
    return variance


def _estimate(local: Mapping[str, Any], source: str) -> np.ndarray | None:
    if source == str(local.get("uav_id", "")):
        value = local.get("estimate", local.get("state_estimate"))
    else:
        messages = local.get("neighbor_messages", {})
        value = messages.get(source, {}).get("estimate", messages.get(source, {}).get("state_estimate")) if isinstance(messages, Mapping) and isinstance(messages.get(source), Mapping) else None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.shape == (15,) and np.isfinite(array).all() else None


def _raw_covariance(local: Mapping[str, Any], source: str) -> np.ndarray | None:
    if source == str(local.get("uav_id", "")):
        value = local.get("covariance", local.get("state_covariance"))
    else:
        messages = local.get("neighbor_messages", {})
        value = messages.get(source, {}).get("covariance", messages.get(source, {}).get("state_covariance")) if isinstance(messages, Mapping) and isinstance(messages.get(source), Mapping) else None
    return _finite_covariance(value, state_dim=15)


def expected_relative_credit(local: Mapping[str, Any], source: str, range_variance_m2: float) -> float:
    """Compute a current-observation linear information credit in ``[0, 1]``."""

    variance = _range_variance(range_variance_m2)
    source = str(source)
    if source not in UIDS or source == str(local.get("uav_id", "")):
        return 0.0
    own_uid = _uid_from_local(local)
    own_estimate = _estimate(local, own_uid)
    neighbor_estimate = _estimate(local, source)
    own_covariance = _raw_covariance(local, own_uid)
    neighbor_covariance = _raw_covariance(local, source)
    if own_estimate is None or neighbor_estimate is None or own_covariance is None or neighbor_covariance is None:
        return 0.0
    relative = neighbor_estimate[:3] - own_estimate[:3]
    norm = float(np.linalg.norm(relative))
    if not math.isfinite(norm) or norm <= 1e-12:
        return 0.0
    direction = relative / norm
    own_position_covariance = own_covariance[:3, :3]
    neighbor_position_covariance = neighbor_covariance[:3, :3]
    local_variance = float(direction @ own_position_covariance @ direction)
    neighbor_variance = float(direction @ neighbor_position_covariance @ direction)
    trace = float(np.trace(own_position_covariance))
    if not all(math.isfinite(value) for value in (local_variance, neighbor_variance, trace)):
        return 0.0
    if local_variance < 0.0 or neighbor_variance < 0.0 or trace <= 0.0:
        return 0.0
    cross = 2.0 * math.sqrt(local_variance * neighbor_variance)
    denominator = local_variance + variance + neighbor_variance + cross
    numerator = float(direction @ own_position_covariance @ own_position_covariance @ direction)
    if not math.isfinite(denominator) or denominator <= 0.0 or not math.isfinite(numerator):
        return 0.0
    return float(np.clip((numerator / denominator) / trace, 0.0, 1.0))


def greedy_scores(local: Mapping[str, Any], credit_scale: float, range_variance_m2: float) -> dict[str, float]:
    try:
        scale = float(credit_scale)
    except (TypeError, ValueError):
        raise ValueError("credit_scale must be finite and positive") from None
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("credit_scale must be finite and positive")
    scores = {uid: 0.0 for uid in UIDS}
    scores[_NONE] = 0.0
    legal = set(_legal_sources(_own_local(local) if isinstance(local, Mapping) else {}))
    for source in UIDS:
        if source in legal:
            raw = expected_relative_credit(local, source, range_variance_m2)
            scores[source] = float(np.clip(raw / scale, 0.0, 1.0) - 0.05)
    return scores


def greedy_selection(local: Mapping[str, Any], credit_scale: float, range_variance_m2: float) -> str:
    scores = greedy_scores(local, credit_scale, range_variance_m2)
    for source in UIDS:
        if scores[source] > 0.0:
            best = source
            break
    else:
        return _NONE
    for source in UIDS[UIDS.index(best) + 1 :]:
        if scores[source] > scores[best]:
            best = source
    return best if scores[best] > 0.0 else _NONE


def relative_linear_credit(event: Mapping[str, Any]) -> float:
    if not isinstance(event, Mapping) or not bool(event.get("accepted", False)):
        return 0.0
    try:
        before = float(event["position_covariance_trace_before_m2"])
        after = float(event["position_covariance_trace_after_linear_m2"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(before) or before <= 0.0 or not math.isfinite(after):
        return 0.0
    return float(np.clip((before - after) / max(before, 1e-12), 0.0, 1.0))


def _event_uid(event: Mapping[str, Any], *, target: bool = True) -> str | None:
    names = ("target_uav_id", "uav_id", "receiver_uav_id") if target else ("uav_id", "target_uav_id")
    for name in names:
        if name in event and str(event[name]) in UIDS:
            return str(event[name])
    return None


def reward_components(
    info: Mapping[str, Any],
    credit_scale: float,
    communication_cost: float = 0.05,
    waste_cost: float = 0.05,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Build shared team rewards from observed update, overhead, and budget rows."""

    try:
        scale = float(credit_scale)
        comm_coef = float(communication_cost)
        waste_coef = float(waste_cost)
    except (TypeError, ValueError):
        raise ValueError("credit_scale and costs must be finite; credit_scale positive") from None
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(comm_coef) or not math.isfinite(waste_coef):
        raise ValueError("credit_scale must be positive and costs finite")
    if not isinstance(info, Mapping):
        raise ValueError("info must be a mapping")

    by_uid: dict[str, dict[str, float]] = {}
    events = info.get("relative_range_update_events", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        events = ()
    # ``measurement_budget_events`` is the native per-transition list in the
    # environment info contract.  Do not substitute a budget estimate or an
    # environment-private cumulative list here.
    budget_rows = info.get("measurement_budget_events", ())
    if not isinstance(budget_rows, Sequence) or isinstance(budget_rows, (str, bytes)):
        budget_rows = ()
    overhead_rows = info.get("measurement_overhead_events", ())
    if not isinstance(overhead_rows, Sequence) or isinstance(overhead_rows, (str, bytes)):
        overhead_rows = ()

    for uid in UIDS:
        uid_events = [event for event in events if isinstance(event, Mapping) and _event_uid(event) == uid]
        raw_credit = min(1.0, sum(relative_linear_credit(event) for event in uid_events))
        budget_consumed = 0.0
        for row in budget_rows:
            if isinstance(row, Mapping) and _event_uid(row, target=False) == uid:
                try:
                    value = float(row.get("budget_consumed", 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                if math.isfinite(value) and value > 0.0:
                    budget_consumed += value
        actual_packets = 0.0
        for row in overhead_rows:
            if not isinstance(row, Mapping) or _event_uid(row) != uid:
                continue
            try:
                value = float(row.get("load_packets_applied", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                actual_packets += value
        scaled_credit = float(np.clip(raw_credit / scale, 0.0, 1.0))
        wasted_units = max(budget_consumed - actual_packets, 0.0)
        communication_penalty = comm_coef * actual_packets
        wasted_penalty = waste_coef * wasted_units
        local_utility = scaled_credit - communication_penalty - wasted_penalty
        by_uid[uid] = {
            "raw_relative_credit_proxy": float(raw_credit),
            "scaled_credit_proxy": scaled_credit,
            "budget_consumed": float(budget_consumed),
            "actual_overhead_packets": float(actual_packets),
            "actual_overhead_bytes_equivalent": float(512.0 * actual_packets),
            "communication_cost": float(communication_penalty),
            "wasted_budget_units": float(wasted_units),
            "wasted_budget_penalty": float(wasted_penalty),
            "local_utility": float(local_utility),
            "team_reward": 0.0,
        }
    team_reward = float(np.mean([by_uid[uid]["local_utility"] for uid in UIDS]))
    rewards = {uid: team_reward for uid in UIDS}
    for uid in UIDS:
        by_uid[uid]["team_reward"] = team_reward
    return rewards, by_uid


def calibration_readiness(
    scene_metrics: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    positive_credits: Sequence[Any],
    credit_scale: float,
    config_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the fixed train-only calibration gates without tuning inputs."""

    failures: list[str] = []
    try:
        scale = float(credit_scale)
    except (TypeError, ValueError):
        scale = float("nan")
    if not math.isfinite(scale) or scale <= 0.0:
        failures.append("credit_scale must be finite and positive")
    required = (
        "min_moving_fraction",
        "min_positive_credit_events",
        "min_separated_decisions",
        "expected_credit_spread_tolerance",
        "min_greedy_positive_decisions",
    )
    gates: dict[str, float] = {}
    for name in required:
        try:
            gates[name] = float(config_gates[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"missing config gate: {name}")
            continue
        if not math.isfinite(gates[name]):
            failures.append(f"nonfinite config gate: {name}")

    rows = list(scene_metrics) if isinstance(scene_metrics, Sequence) else []
    train_rows = [row for row in rows if isinstance(row, Mapping) and row.get("split") == "train"]
    if any(not isinstance(row, Mapping) or row.get("split") != "train" for row in rows):
        failures.append("validation or malformed scene metrics supplied to train calibration")
    if len(train_rows) != 4:
        failures.append(f"expected exactly four train scenes, got {len(train_rows)}")
    scene_ids = [str(row.get("scene_id")) for row in train_rows]
    if len(set(scene_ids)) != len(scene_ids):
        failures.append("duplicate train scene IDs")

    per_scene: dict[str, dict[str, Any]] = {}
    scene_change_counts: dict[str, int] = {}
    for row in train_rows:
        scene_id = str(row.get("scene_id"))
        fractions = row.get("moving_fractions", {})
        values: dict[str, float] = {}
        if not isinstance(fractions, Mapping):
            fractions = {}
        for uid in UIDS:
            try:
                values[uid] = float(fractions[uid])
            except (KeyError, TypeError, ValueError):
                values[uid] = float("nan")
        minimum = min(values.values()) if values and all(math.isfinite(value) for value in values.values()) else float("nan")
        try:
            changes = int(row.get("physical_graph_changes", 0))
        except (TypeError, ValueError):
            changes = 0
        if changes < 0:
            failures.append(f"scene {scene_id} has negative physical graph changes")
        declared = bool(row.get("requires_graph_change", False))
        scene_change_counts[scene_id] = changes
        per_scene[scene_id] = {
            "moving_fractions": values,
            "min_moving_fraction": float(minimum),
            "physical_graph_changes": changes,
            "requires_graph_change": declared,
            "dynamic_declared_and_observed": bool(declared and changes > 0),
        }
        if not math.isfinite(minimum) or (gates.get("min_moving_fraction", float("inf")) > minimum):
            failures.append(f"scene {scene_id} fails minimum moving fraction")
    if not any(item["dynamic_declared_and_observed"] for item in per_scene.values()):
        failures.append("no declared dynamic train scene has a physical graph change")

    finite_positive = [float(value) for value in positive_credits if _finite_positive(value)]
    if len(finite_positive) < int(gates.get("min_positive_credit_events", 10**18)):
        failures.append("insufficient positive finite credit events")

    decisions = list(decision_rows) if isinstance(decision_rows, Sequence) else []
    greedy_positive = 0
    multi_peer = 0
    separated = 0
    maximum_spread = 0.0
    tolerance = gates.get("expected_credit_spread_tolerance", float("inf"))
    train_scene_set = set(scene_ids)
    for row in decisions:
        if not isinstance(row, Mapping):
            failures.append("malformed decision row")
            continue
        decision_scene = row.get("scene_id")
        if decision_scene is None or str(decision_scene) not in train_scene_set:
            failures.append("decision row references a scene outside the supplied TRAIN set")
        if "split" in row and row.get("split") != "train":
            failures.append("decision row is not TRAIN-scoped")
        if "uav_id" in row and str(row.get("uav_id")) not in UIDS:
            failures.append("decision row has an unknown UAV ID")
        if bool(row.get("greedy_positive", False)):
            greedy_positive += 1
        try:
            legal_count = int(row.get("legal_peer_count", 0))
            spread = float(row.get("expected_credit_spread", float("nan")))
        except (TypeError, ValueError):
            legal_count, spread = 0, float("nan")
        if legal_count >= 2:
            multi_peer += 1
            if math.isfinite(spread):
                maximum_spread = max(maximum_spread, spread)
                if spread > tolerance:
                    separated += 1
    if greedy_positive < int(gates.get("min_greedy_positive_decisions", 10**18)):
        failures.append("insufficient greedy-positive decisions")
    if separated < int(gates.get("min_separated_decisions", 10**18)):
        failures.append("insufficient separated multi-peer decisions")
    denominator = len(decisions)
    aggregate = {
        "train_scene_count": len(train_rows),
        "positive_credit_count": len(finite_positive),
        "positive_credit_finite_count": len(finite_positive),
        "greedy_positive_decisions": greedy_positive,
        "decision_count": denominator,
        "multi_peer_decision_count": multi_peer,
        "separated_decision_count": separated,
        "separated_fraction_multi_peer": float(separated / multi_peer) if multi_peer else 0.0,
        "separated_fraction_all_decisions": float(separated / denominator) if denominator else 0.0,
        "maximum_expected_credit_spread": float(maximum_spread),
        "max_expected_credit_spread": float(maximum_spread),
        "physical_graph_changes_total": int(sum(scene_change_counts.values())),
        "declared_dynamic_train_scene_count": int(
            sum(item["dynamic_declared_and_observed"] for item in per_scene.values())
        ),
    }
    return {
        "ready": not failures,
        "failures": failures,
        "aggregate": aggregate,
        "per_scene": per_scene,
        "per_scene_rows": [{"scene_id": key, **value} for key, value in per_scene.items()],
        "scene_change_counts": scene_change_counts,
        "gates": gates,
    }


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


__all__ = [
    "UIDS",
    "admissible_mask",
    "admissible_observations",
    "calibration_readiness",
    "communication_actions",
    "expected_relative_credit",
    "fixed_bandwidth",
    "greedy_scores",
    "greedy_selection",
    "make_scene_splits",
    "random_selection",
    "relative_linear_credit",
    "reward_components",
    "round_robin_selection",
]
