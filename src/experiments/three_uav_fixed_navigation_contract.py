"""Pure contracts for the fixed three-UAV navigation experiment.

The functions in this module build deterministic reset manifests and small
analytic helpers.  They intentionally do not import or construct an
environment, consume randomness, inspect truth state, or run a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


UIDS = ("uav_00", "uav_01", "uav_02")
_BOUNDS_MIN = (0.0, 0.0, 2.0)
_BOUNDS_MAX = (200.0, 200.0, 60.0)


def _jsonable(value: Any) -> Any:
    """Convert scalar-like values to JSON-compatible values for hashing."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    # Keep this helper independent of NumPy while accepting NumPy scalars and
    # arrays if a caller uses them to inspect a geometry hash.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _geometry_hash(scenario: Mapping[str, Any]) -> str:
    """Return the fixed-navigation geometry hash.

    Only the ordered UAV position/goal maps, obstacle geometry, explicit empty
    obstacle reset override, and fixed bounds contribute to this digest.
    Episode identifiers, variants, seeds, and configuration flags are absent.
    """

    payload: dict[str, Any] = {
        "initial_positions": {
            str(uid): [float(value) for value in scenario.get("initial_positions", {}).get(uid, ())]
            for uid in UIDS
        },
        "goals": {
            str(uid): [float(value) for value in scenario.get("goals", {}).get(uid, ())]
            for uid in UIDS
        },
        "obstacle_specs": [],
        "bounds_min_m": list(_BOUNDS_MIN),
        "bounds_max_m": list(_BOUNDS_MAX),
    }
    for spec in scenario.get("obstacle_specs", []):
        if not isinstance(spec, Mapping):
            continue
        # Keep the same geometry-only fields as the historical single-UAV
        # contract.  This schedule has no obstacles, but retaining the full
        # contract makes the helper safe for callers inspecting variants.
        payload["obstacle_specs"].append(
            {
                "geometry_type": str(spec["geometry_type"]),
                "center_m": [float(value) for value in spec["center_m"]],
                "size_m": [float(value) for value in spec["size_m"]],
                "physical_collision_enabled": bool(spec["physical_collision_enabled"]),
            }
        )
    if "obstacles" in scenario:
        payload["obstacles"] = []
    return _canonical(payload)


def _scene(starts: Sequence[Sequence[float]], goals: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(starts) != len(UIDS) or len(goals) != len(UIDS):
        raise ValueError("three-UAV scenes require exactly three starts and goals")
    return {
        "initial_positions": {
            uid: [float(value) for value in starts[index]] for index, uid in enumerate(UIDS)
        },
        "initial_velocities": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
        "initial_attitudes": {uid: [0.0, 0.0, 0.0] for uid in UIDS},
        "goals": {
            uid: [float(value) for value in goals[index]] for index, uid in enumerate(UIDS)
        },
        "obstacle_specs": [],
        "obstacles": [],
    }


def _plus_x(starts: Sequence[Sequence[float]], amount: float = 12.0) -> list[list[float]]:
    return [
        [float(point[0]) + float(amount), float(point[1]), float(point[2])]
        for point in starts
    ]


def make_scenario_schedule() -> list[dict[str, Any]]:
    """Build the exact eight-episode fixed-navigation schedule.

    A fresh scene is returned for every entry and on every call so downstream
    reset handling cannot mutate sibling episodes or alter a later invocation.
    """

    dev_starts = ((40.0, 40.0, 20.0), (40.0, 48.0, 20.0), (40.0, 44.0, 27.0))
    c0_starts = ((80.0, 80.0, 22.0), (80.0, 88.0, 22.0), (80.0, 84.0, 29.0))
    c1_starts = ((55.0, 50.0, 20.0), (55.0, 60.0, 20.0), (55.0, 55.0, 28.0))
    c1_goals = ((75.0, 44.0, 20.0), (75.0, 66.0, 20.0), (75.0, 55.0, 28.0))
    c2_starts = ((40.0, 100.0, 25.0), (110.0, 100.0, 25.0), (77.0, 108.0, 31.0))
    c2_goals = ((30.0, 100.0, 25.0), (125.0, 100.0, 25.0), (77.0, 108.0, 31.0))

    scenes = {
        "dev": _scene(dev_starts, _plus_x(dev_starts)),
        "c0": _scene(c0_starts, _plus_x(c0_starts)),
        "c1": _scene(c1_starts, c1_goals),
        "c2": _scene(c2_starts, c2_goals),
    }
    rows = (
        ("three_nav_dev_normal", "dev", "normal", 2026115100, True, "development"),
        ("three_nav_c0_normal", "c0", "normal", 2026115200, True, "evaluation"),
        ("three_nav_c0_cut", "c0", "cut", 2026115200, True, "evaluation"),
        ("three_nav_c1_normal", "c1", "normal", 2026115201, True, "evaluation"),
        ("three_nav_c1_cut", "c1", "cut", 2026115201, True, "evaluation"),
        ("three_nav_c2_normal", "c2", "normal", 2026115202, True, "evaluation"),
        ("three_nav_c2_cut", "c2", "cut", 2026115202, True, "evaluation"),
        ("three_nav_c0_fusion_off", "c0", "fusion_off", 2026115200, False, "fusion_diagnostic"),
    )
    schedule: list[dict[str, Any]] = []
    for episode_id, case_id, variant, seed, ekf_enabled, stage in rows:
        scene = _scene(
            list(scenes[case_id]["initial_positions"].values()),
            list(scenes[case_id]["goals"].values()),
        )
        schedule.append(
            {
                "episode_id": episode_id,
                "case_id": case_id,
                "variant": variant,
                "seed": int(seed),
                "scenario": scene,
                "cooperative_ekf_enabled": bool(ekf_enabled),
                "stage": stage,
                "geometry_sha256": _geometry_hash(scene),
            }
        )
    return schedule


def _own_local(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the actor-local mapping without inspecting other state fields."""

    nested = observation.get("local")
    if isinstance(nested, Mapping):
        return nested
    return observation


def _as_ids(value: Any) -> set[str]:
    if isinstance(value, (str, bytes)):
        return {str(value)}
    try:
        return {str(item) for item in value}
    except (TypeError, ValueError):
        return set()


def _action_allowed(mask: Any, source: str) -> bool:
    if isinstance(mask, Mapping):
        return bool(mask.get(source, False))
    try:
        index = UIDS.index(source)
        return index < len(mask) and bool(mask[index])
    except (TypeError, ValueError, IndexError):
        return False


def controlled_communication_actions(
    observations: Mapping[str, Mapping[str, Any]],
    step: int,
    cut_enabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build deterministic measurement and outgoing-bandwidth actions.

    Selection uses each UAV's own local candidate/effective sets and optional
    local action mask.  No truth, graph-global, or current-position field is
    read.  Bandwidth remains a fixed half-share per potential outgoing edge;
    the single-edge case deliberately retains the half-share rather than
    redistributing it to one.
    """

    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    if not isinstance(observations, Mapping):
        raise ValueError("observations must be a mapping keyed by UAV ID")
    actions: dict[str, dict[str, Any]] = {}
    for uid in UIDS:
        raw = observations.get(uid, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"observation for {uid} must be a mapping")
        local = _own_local(raw)
        candidates = _as_ids(local.get("candidate_measurement_sources", ()))
        effective = _as_ids(local.get("effective_sources", ()))
        mask_present = "action_mask" in local
        mask = local.get("action_mask")
        legal = sorted(
            (candidates & effective & set(UIDS)) - {uid},
            key=lambda source: UIDS.index(source) if source in UIDS else len(UIDS),
        )
        if mask_present:
            legal = [source for source in legal if _action_allowed(mask, source)]
        selected = legal[(step - 1) % len(legal)] if legal else "none"

        potential = sorted(
            (_as_ids(local.get("potential_targets", ())) & set(UIDS)) - {uid},
            key=UIDS.index,
        )
        bandwidth = {target: 0.5 for target in potential}
        if cut_enabled and 20 <= step <= 50:
            if uid == "uav_00" and "uav_02" in bandwidth:
                bandwidth["uav_02"] = 0.0
            elif uid == "uav_02" and "uav_00" in bandwidth:
                bandwidth["uav_00"] = 0.0
        actions[uid] = {"measurement_uav_id": selected, "bandwidth": bandwidth}
    return actions


def _finite_vec(value: Any, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite three-vector")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite three-vector") from None
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must be a finite three-vector")
    return values  # type: ignore[return-value]


def _position_map(values: Mapping[str, Any], *, name: str) -> dict[str, tuple[float, float, float]]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping keyed by UIDs")
    missing = [uid for uid in UIDS if uid not in values]
    if missing:
        raise ValueError(f"{name} is missing {missing}")
    return {uid: _finite_vec(values[uid], name=f"{name}[{uid}]") for uid in UIDS}


def joint_closest_approach(
    pre_positions: Mapping[str, Any],
    post_positions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Find the synchronized closest approach for each UAV pair.

    Each pair follows one shared piecewise-linear time fraction in ``[0, 1]``.
    This is intentionally different from independently matching segment times
    for each vehicle.
    """

    pre = _position_map(pre_positions, name="pre_positions")
    post = _position_map(post_positions, name="post_positions")
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(UIDS):
        for target in UIDS[source_index + 1 :]:
            relative_pre = tuple(pre[source][axis] - pre[target][axis] for axis in range(3))
            relative_motion = tuple(
                (post[source][axis] - post[target][axis]) - relative_pre[axis] for axis in range(3)
            )
            motion_norm_sq = sum(value * value for value in relative_motion)
            if motion_norm_sq <= 1e-24:
                fraction = 0.0
            else:
                dot = sum(relative_pre[axis] * relative_motion[axis] for axis in range(3))
                fraction = max(0.0, min(1.0, -dot / motion_norm_sq))
            relative_at_fraction = tuple(
                relative_pre[axis] + fraction * relative_motion[axis] for axis in range(3)
            )
            distance = math.sqrt(sum(value * value for value in relative_at_fraction))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "distance_m": float(distance),
                    "time_fraction": float(fraction),
                }
            )
    return rows


class StrictParkingTracker:
    """Track strict individual and simultaneous three-UAV parking holds."""

    def __init__(
        self,
        uav_ids: Sequence[str] = UIDS,
        distance_limit_m: float = 1.0,
        speed_limit_mps: float = 0.5,
        hold_steps: int = 5,
    ) -> None:
        ids = tuple(str(uid) for uid in uav_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("uav_ids must be a nonempty sequence of unique IDs")
        for name, value in (("distance_limit_m", distance_limit_m), ("speed_limit_mps", speed_limit_mps)):
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps < 1:
            raise ValueError("hold_steps must be a positive integer")
        self.uav_ids = ids
        self.distance_limit_m = float(distance_limit_m)
        self.speed_limit_mps = float(speed_limit_mps)
        self.hold_steps = int(hold_steps)
        self._step = 0
        self._holds = {uid: 0 for uid in ids}
        self._ever_success = {uid: False for uid in ids}
        self._first_qualifying_step = {uid: None for uid in ids}
        self._first_success_step = {uid: None for uid in ids}
        self._joint_hold_steps = 0
        self._joint_ever_success = False
        self._joint_first_success_step: int | None = None
        self._collision_latched = False

    def _metric_map(self, values: Mapping[str, Any], *, name: str) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping keyed by UAV ID")
        missing = [uid for uid in self.uav_ids if uid not in values]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        result: dict[str, float] = {}
        for uid in self.uav_ids:
            try:
                value = float(values[uid])
            except (TypeError, ValueError):
                raise ValueError(f"{name}[{uid}] must be finite and nonnegative") from None
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}[{uid}] must be finite and nonnegative")
            result[uid] = value
        return result

    def update(
        self,
        distances: Mapping[str, Any],
        speeds: Mapping[str, Any],
        collision: bool = False,
    ) -> dict[str, Any]:
        """Advance one observation and return current plus sticky status."""

        distance_values = self._metric_map(distances, name="distances")
        speed_values = self._metric_map(speeds, name="speeds")
        if not isinstance(collision, (bool,)):
            raise ValueError("collision must be a boolean")
        self._step += 1
        if collision:
            self._collision_latched = True

        qualifying: dict[str, bool] = {}
        for uid in self.uav_ids:
            current = (
                not self._collision_latched
                and distance_values[uid] <= self.distance_limit_m
                and speed_values[uid] <= self.speed_limit_mps
            )
            qualifying[uid] = current
            if current:
                self._holds[uid] += 1
                if self._first_qualifying_step[uid] is None:
                    self._first_qualifying_step[uid] = self._step
                if self._holds[uid] >= self.hold_steps:
                    self._ever_success[uid] = True
                    if self._first_success_step[uid] is None:
                        self._first_success_step[uid] = self._step
            else:
                self._holds[uid] = 0

        if all(qualifying.values()):
            self._joint_hold_steps += 1
            if self._joint_hold_steps >= self.hold_steps:
                self._joint_ever_success = True
                if self._joint_first_success_step is None:
                    self._joint_first_success_step = self._step
        else:
            self._joint_hold_steps = 0

        return {
            "per_uav": {
                uid: {
                    "qualifying": bool(qualifying[uid]),
                    "hold_steps": int(self._holds[uid]),
                    "ever_success": bool(self._ever_success[uid]),
                    "first_qualifying_step": self._first_qualifying_step[uid],
                    "first_success_step": self._first_success_step[uid],
                }
                for uid in self.uav_ids
            },
            "joint_hold_steps": int(self._joint_hold_steps),
            "joint_ever_success": bool(self._joint_ever_success),
            "joint_first_success_step": self._joint_first_success_step,
            "step": int(self._step),
        }


__all__ = [
    "UIDS",
    "make_scenario_schedule",
    "controlled_communication_actions",
    "joint_closest_approach",
    "StrictParkingTracker",
]
