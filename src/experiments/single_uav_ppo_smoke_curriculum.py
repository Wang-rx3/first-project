"""Deterministic scenario pool and update schedule for the single-UAV PPO smoke run.

The smoke run deliberately keeps scenario construction separate from the runtime
environment.  This module only creates JSON-compatible reset fixtures and an
auditable curriculum schedule.  A runner may pass ``entry["scenario"]`` to the
baseline environment later, but building the curriculum never resets or steps an
environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from math import dist
from typing import Any, Iterable, Mapping, Sequence


BOUNDS_MIN_M = (0.0, 0.0, 2.0)
BOUNDS_MAX_M = (200.0, 200.0, 60.0)
ROLLOUT_STEPS = 120
CALIBRATION_STEPS_PER_UPDATE = 16
PPO_UPDATES = 18


def _json_hash(payload: Mapping[str, Any]) -> str:
    """Hash a canonical JSON payload without depending on dictionary order."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_payload(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return geometry used for leakage checks.

    Obstacle identifiers are intentionally omitted.  A renamed obstacle must not
    make an otherwise identical scene appear to be a new geometry.  Start and
    goal positions are part of the scene geometry because the smoke pool uses
    obstacle-free normal scenes with different short navigation segments.
    """

    obstacle_specs = []
    for spec in scenario.get("obstacle_specs", []):
        obstacle_specs.append(
            {
                "geometry_type": str(spec["geometry_type"]),
                "center_m": [float(value) for value in spec["center_m"]],
                "size_m": [float(value) for value in spec["size_m"]],
                "physical_collision_enabled": bool(
                    spec["physical_collision_enabled"]
                ),
            }
        )
    payload = {
        "initial_positions": {
            str(key): [float(value) for value in values]
            for key, values in scenario.get("initial_positions", {}).items()
        },
        "goals": {
            str(key): [float(value) for value in values]
            for key, values in scenario.get("goals", {}).items()
        },
        "obstacle_specs": obstacle_specs,
        "bounds_min_m": list(BOUNDS_MIN_M),
        "bounds_max_m": list(BOUNDS_MAX_M),
    }
    if "obstacles" in scenario:
        # Preserve the explicit empty-obstacle override in the hash.  This
        # distinguishes a real empty normal scene from the frozen environment
        # fallback that is selected by obstacle_specs=[] alone.
        payload["obstacles"] = []
    return payload


def _geometry_hash(scenario: Mapping[str, Any]) -> str:
    return _json_hash(_geometry_payload(scenario))


def _mirror_y(value: float) -> float:
    return BOUNDS_MIN_M[1] + BOUNDS_MAX_M[1] - float(value)


def _case_id(split: str, kind: str, index: int) -> str:
    return f"smoke_{split}_{kind}_{index:02d}"


def _seed_for(seed: int, ordinal: int, *, noise: bool = False) -> int:
    """Derive disjoint, stable scenario and noise streams from the root seed."""

    # Keeping the two ranges separate makes accidental stream reuse easy to
    # detect in manifests while retaining a simple integer seed for the runtime.
    return int(seed) * 1_000_000 + (50_000 if noise else 1_000) + int(ordinal)


def _box(
    obstacle_id: str,
    center: Sequence[float],
    size: Sequence[float],
) -> dict[str, Any]:
    return {
        "obstacle_id": str(obstacle_id),
        "geometry_type": "box",
        "center_m": [float(value) for value in center],
        "size_m": [float(value) for value in size],
        "physical_collision_enabled": True,
    }


def _scenario(
    *,
    start: Sequence[float],
    goal: Sequence[float],
    velocity: Sequence[float],
    attitude: Sequence[float],
    obstacle_specs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    # These names and mappings are the reset keys consumed by the frozen
    # single-UAV PPO baseline.  Return fresh lists so callers cannot mutate a
    # sibling mirror case through shared state.
    return {
        "initial_positions": {"uav_00": [float(value) for value in start]},
        "initial_velocities": {"uav_00": [float(value) for value in velocity]},
        "initial_attitudes": {"uav_00": [float(value) for value in attitude]},
        "goals": {"uav_00": [float(value) for value in goal]},
        "obstacle_specs": [copy.deepcopy(dict(spec)) for spec in obstacle_specs],
    }


def _entry(
    *,
    split: str,
    kind: str,
    index: int,
    side: str,
    mirror_pair: str | None,
    scenario_seed: int,
    noise_seed: int,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    scene = copy.deepcopy(dict(scenario))
    return {
        "episode_id": _case_id(split, kind, index),
        "scenario_id": _case_id(split, kind, index),
        "kind": kind,
        "side": side,
        "mirror_pair": mirror_pair,
        "split": split,
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "geometry_sha256": _geometry_hash(scene),
        "scenario": scene,
    }


def _normal_entry(split: str, index: int, seed: int, ordinal: int) -> dict[str, Any]:
    # Every normal training segment is a real 4--8 m navigation segment.  The
    # validation positions are outside the training geometry hash set.
    if split == "train":
        starts = ((18.0, 92.0, 8.0), (22.0, 96.0, 10.0), (26.0, 100.0, 12.0),
                  (30.0, 104.0, 14.0), (34.0, 108.0, 16.0), (38.0, 112.0, 18.0))
        lengths = (5.0, 6.0, 7.0, 8.0, 5.5, 7.5)
        lateral_velocity = (0.12, -0.16, 0.10, -0.14, 0.18, -0.12)
    else:
        starts = ((44.0, 88.0, 9.0), (48.0, 116.0, 15.0))
        lengths = (6.5, 7.25)
        lateral_velocity = (0.15, -0.15)
    start = starts[index]
    goal = (start[0] + lengths[index], start[1], start[2])
    scene = _scenario(
        start=start,
        goal=goal,
        velocity=(0.35, lateral_velocity[index], 0.0),
        attitude=(0.0, 0.0, 0.0),
        obstacle_specs=(),
    )
    # The frozen environment interprets an empty obstacle_specs list as a
    # request for its historical default obstacles.  Keep the reset fixture
    # schema compatible while explicitly selecting a truly empty scene.
    scene["obstacles"] = []
    return _entry(
        split=split,
        kind="normal",
        index=index,
        side="none",
        mirror_pair=None,
        scenario_seed=_seed_for(seed, ordinal),
        noise_seed=_seed_for(seed, ordinal, noise=True),
        scenario=scene,
    )


def _obstacle_entry(
    *,
    split: str,
    kind: str,
    index: int,
    side: str,
    mirror_pair: str,
    seed: int,
    ordinal: int,
    mission_y: float,
    start_x: float,
    goal_x: float,
    obstacle_x: float,
    obstacle_size_x: float,
    obstacle_size_y: float,
    obstacle_size_z: float,
    z: float,
    lateral_speed: float,
) -> dict[str, Any]:
    # ``left`` is +Y and ``right`` is -Y.  A box shifted to negative Y is
    # easier to bypass on the left; a positive-Y shift is easier to bypass on
    # the right.  The segment still crosses every box by construction.
    if side == "left":
        obstacle_offset_y = -2.0 if kind == "single" else -2.5
        velocity_y = abs(lateral_speed)
        attitude_y = 0.02
    elif side == "right":
        obstacle_offset_y = 2.0 if kind == "single" else 2.5
        velocity_y = -abs(lateral_speed)
        attitude_y = -0.02
    else:
        raise ValueError(f"obstacle case has invalid side {side!r}")
    start = (start_x, mission_y, z)
    goal = (goal_x, mission_y, z)
    obstacle = _box(
        f"{kind}_obstacle_{split}_{index:02d}_{side}",
        (obstacle_x, mission_y + obstacle_offset_y, z),
        (obstacle_size_x, obstacle_size_y, obstacle_size_z),
    )
    scene = _scenario(
        start=start,
        goal=goal,
        velocity=(0.30, velocity_y, 0.0),
        attitude=(0.0, attitude_y, 0.0),
        obstacle_specs=(obstacle,),
    )
    return _entry(
        split=split,
        kind=kind,
        index=index,
        side=side,
        mirror_pair=mirror_pair,
        scenario_seed=_seed_for(seed, ordinal),
        noise_seed=_seed_for(seed, ordinal, noise=True),
        scenario=scene,
    )


def _build_split(split: str, seed: int, ordinal_start: int) -> list[dict[str, Any]]:
    if split == "train":
        normal_count = 6
        pair_ys_single = (94.0, 98.0, 102.0)
        single_x = (40.0, 42.0, 44.0)
        single_goal_x = (62.0, 64.0, 66.0)
        single_obstacle_x = (50.0, 52.0, 54.0)
        single_size_x = (4.0, 5.0, 6.0)
        single_z = (8.0, 12.0, 16.0)
        pair_ys_full = (88.0, 96.0, 104.0)
        full_x = (30.0, 32.0, 34.0)
        full_goal_x = (70.0, 72.0, 74.0)
        full_obstacle_x = (48.0, 49.0, 50.0)
        full_size_x = (6.0, 7.0, 8.0)
        full_z = (10.0, 14.0, 18.0)
    elif split == "validation":
        normal_count = 2
        pair_ys_single = (90.0, 98.0)
        single_x = (46.0, 48.0)
        single_goal_x = (68.0, 70.0)
        single_obstacle_x = (56.0, 58.0)
        single_size_x = (4.5, 5.5)
        single_z = (20.0, 24.0)
        pair_ys_full = (84.0, 100.0)
        full_x = (36.0, 38.0)
        full_goal_x = (76.0, 78.0)
        full_obstacle_x = (55.0, 56.0)
        full_size_x = (6.5, 7.5)
        full_z = (22.0, 26.0)
    else:
        raise ValueError(f"unsupported split {split!r}")

    pool: list[dict[str, Any]] = []
    ordinal = ordinal_start
    for index in range(normal_count):
        pool.append(_normal_entry(split, index, seed, ordinal))
        ordinal += 1

    for pair_index, mission_y in enumerate(pair_ys_single):
        pair_name = f"smoke_{split}_single_pair_{pair_index:02d}"
        # The first member is the +Y-bypass case.  Its mirrored member is the
        # -Y-bypass case, while the shared mission line remains +X.
        left_index = 2 * pair_index
        right_index = left_index + 1
        pool.append(
            _obstacle_entry(
                split=split,
                kind="single",
                index=left_index,
                side="left",
                mirror_pair=pair_name,
                seed=seed,
                ordinal=ordinal,
                mission_y=mission_y,
                start_x=single_x[pair_index],
                goal_x=single_goal_x[pair_index],
                obstacle_x=single_obstacle_x[pair_index],
                obstacle_size_x=single_size_x[pair_index],
                obstacle_size_y=6.0,
                obstacle_size_z=6.0,
                z=single_z[pair_index],
                lateral_speed=0.18 + 0.02 * pair_index,
            )
        )
        ordinal += 1
        pool.append(
            _obstacle_entry(
                split=split,
                kind="single",
                index=right_index,
                side="right",
                mirror_pair=pair_name,
                seed=seed,
                ordinal=ordinal,
                mission_y=_mirror_y(mission_y),
                start_x=single_x[pair_index],
                goal_x=single_goal_x[pair_index],
                obstacle_x=single_obstacle_x[pair_index],
                obstacle_size_x=single_size_x[pair_index],
                obstacle_size_y=6.0,
                obstacle_size_z=6.0,
                z=single_z[pair_index],
                lateral_speed=0.18 + 0.02 * pair_index,
            )
        )
        ordinal += 1

    for pair_index, mission_y in enumerate(pair_ys_full):
        pair_name = f"smoke_{split}_full_pair_{pair_index:02d}"
        left_index = 2 * pair_index
        right_index = left_index + 1
        pool.append(
            _obstacle_entry(
                split=split,
                kind="full",
                index=left_index,
                side="left",
                mirror_pair=pair_name,
                seed=seed,
                ordinal=ordinal,
                mission_y=mission_y,
                start_x=full_x[pair_index],
                goal_x=full_goal_x[pair_index],
                obstacle_x=full_obstacle_x[pair_index],
                obstacle_size_x=full_size_x[pair_index],
                obstacle_size_y=8.0,
                obstacle_size_z=8.0 + pair_index,
                z=full_z[pair_index],
                lateral_speed=0.22 + 0.02 * pair_index,
            )
        )
        ordinal += 1
        pool.append(
            _obstacle_entry(
                split=split,
                kind="full",
                index=right_index,
                side="right",
                mirror_pair=pair_name,
                seed=seed,
                ordinal=ordinal,
                mission_y=_mirror_y(mission_y),
                start_x=full_x[pair_index],
                goal_x=full_goal_x[pair_index],
                obstacle_x=full_obstacle_x[pair_index],
                obstacle_size_x=full_size_x[pair_index],
                obstacle_size_y=8.0,
                obstacle_size_z=8.0 + pair_index,
                z=full_z[pair_index],
                lateral_speed=0.22 + 0.02 * pair_index,
            )
        )
        ordinal += 1
    return pool


def _segment_intersects_box(
    start: Sequence[float], goal: Sequence[float], center: Sequence[float], size: Sequence[float]
) -> bool:
    """Use a closed slab test to prove that a mission line crosses a box."""

    t_min, t_max = 0.0, 1.0
    for start_value, goal_value, center_value, size_value in zip(
        start, goal, center, size
    ):
        low = float(center_value) - float(size_value) / 2.0
        high = float(center_value) + float(size_value) / 2.0
        delta = float(goal_value) - float(start_value)
        if abs(delta) <= 1e-12:
            if not (low <= float(start_value) <= high):
                return False
            continue
        first = (low - float(start_value)) / delta
        last = (high - float(start_value)) / delta
        if first > last:
            first, last = last, first
        t_min = max(t_min, first)
        t_max = min(t_max, last)
        if t_min > t_max + 1e-12:
            return False
    return t_max >= 0.0 and t_min <= 1.0 and t_min <= t_max + 1e-12


def _assert_vector(value: Any, *, name: str, bounded: bool) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssertionError(f"{name} must be a length-three list")
    if not all(isinstance(item, (int, float)) for item in value):
        raise AssertionError(f"{name} must contain numeric values")
    if bounded and not all(
        low <= float(item) <= high
        for item, low, high in zip(value, BOUNDS_MIN_M, BOUNDS_MAX_M)
    ):
        raise AssertionError(f"{name} falls outside the scene bounds")


def _validate_entry(entry: Mapping[str, Any]) -> None:
    required = {
        "episode_id",
        "scenario_id",
        "kind",
        "side",
        "mirror_pair",
        "split",
        "scenario_seed",
        "noise_seed",
        "geometry_sha256",
        "scenario",
    }
    if not required.issubset(entry):
        raise AssertionError(f"scenario entry is missing {required - set(entry)}")
    if entry["episode_id"] != entry["scenario_id"]:
        raise AssertionError("episode_id and scenario_id must identify the same case")
    scene = entry["scenario"]
    if not isinstance(scene, Mapping):
        raise AssertionError("scenario must be a mapping")
    expected_keys = {
        "initial_positions",
        "initial_velocities",
        "initial_attitudes",
        "goals",
        "obstacle_specs",
    }
    if set(scene) not in (expected_keys, expected_keys | {"obstacles"}):
        raise AssertionError(f"reset scenario keys drifted: {set(scene)}")
    if "obstacles" in scene:
        if entry["kind"] != "normal" or scene["obstacles"] != []:
            raise AssertionError("only normal scenes may explicitly disable obstacles")
    for field in ("initial_positions", "initial_velocities", "initial_attitudes", "goals"):
        mapping = scene[field]
        if set(mapping) != {"uav_00"}:
            raise AssertionError(f"{field} must contain exactly uav_00")
        _assert_vector(mapping["uav_00"], name=f"{field}.uav_00", bounded=field != "initial_velocities" and field != "initial_attitudes")
    initial_velocity = scene["initial_velocities"]["uav_00"]
    if dist(initial_velocity, (0.0, 0.0, 0.0)) > 1.0:
        raise AssertionError("smoke initial velocity must remain modest")
    mission_distance = dist(
        scene["initial_positions"]["uav_00"], scene["goals"]["uav_00"]
    )
    if entry["kind"] == "normal" and not 4.0 <= mission_distance <= 8.0:
        raise AssertionError("normal smoke missions must span 4--8 m")
    if entry["geometry_sha256"] != _geometry_hash(scene):
        raise AssertionError(f"geometry hash mismatch for {entry['scenario_id']}")
    obstacles = scene["obstacle_specs"]
    if entry["kind"] == "normal" and obstacles:
        raise AssertionError("normal smoke scenes must be obstacle-free")
    if entry["kind"] in {"single", "full"} and len(obstacles) != 1:
        raise AssertionError("obstacle smoke scenes must contain exactly one box")
    for obstacle in obstacles:
        if set(obstacle) != {
            "obstacle_id",
            "geometry_type",
            "center_m",
            "size_m",
            "physical_collision_enabled",
        }:
            raise AssertionError("obstacle reset schema drifted")
        if obstacle["geometry_type"] != "box" or not obstacle["physical_collision_enabled"]:
            raise AssertionError("smoke obstacles must be physical boxes")
        _assert_vector(obstacle["center_m"], name="obstacle.center_m", bounded=True)
        _assert_vector(obstacle["size_m"], name="obstacle.size_m", bounded=False)
        if not all(float(size) > 0.0 for size in obstacle["size_m"]):
            raise AssertionError("obstacle sizes must be positive")
        low = [float(c) - float(s) / 2.0 for c, s in zip(obstacle["center_m"], obstacle["size_m"])]
        high = [float(c) + float(s) / 2.0 for c, s in zip(obstacle["center_m"], obstacle["size_m"])]
        if any(lo < bound_low or hi > bound_high for lo, hi, bound_low, bound_high in zip(low, high, BOUNDS_MIN_M, BOUNDS_MAX_M)):
            raise AssertionError("obstacle extents fall outside the scene bounds")
        if not _segment_intersects_box(
            scene["initial_positions"]["uav_00"],
            scene["goals"]["uav_00"],
            obstacle["center_m"],
            obstacle["size_m"],
        ):
            raise AssertionError(f"mission line does not intersect {obstacle['obstacle_id']}")
    if entry["kind"] in {"single", "full"}:
        mission_y = float(scene["initial_positions"]["uav_00"][1])
        dy = float(obstacles[0]["center_m"][1]) - mission_y
        if entry["side"] == "left" and not dy < 0.0:
            raise AssertionError("left case must offset the obstacle to negative Y")
        if entry["side"] == "right" and not dy > 0.0:
            raise AssertionError("right case must offset the obstacle to positive Y")
    elif entry["side"] != "none":
        raise AssertionError("normal case must have side none")


def _assert_mirrors(pool: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in pool:
        if entry["mirror_pair"] is not None:
            groups[str(entry["mirror_pair"])].append(entry)
    for pair_id, pair in groups.items():
        if len(pair) != 2 or {item["side"] for item in pair} != {"left", "right"}:
            raise AssertionError(f"mirror pair {pair_id} is incomplete")
        left = next(item for item in pair if item["side"] == "left")
        right = next(item for item in pair if item["side"] == "right")
        left_scene, right_scene = left["scenario"], right["scenario"]
        for field in ("initial_positions", "goals"):
            left_vector = left_scene[field]["uav_00"]
            right_vector = right_scene[field]["uav_00"]
            if right_vector[0] != left_vector[0] or right_vector[2] != left_vector[2] or right_vector[1] != _mirror_y(left_vector[1]):
                raise AssertionError(f"{pair_id} {field} are not mirrored")
        left_velocity = left_scene["initial_velocities"]["uav_00"]
        right_velocity = right_scene["initial_velocities"]["uav_00"]
        if right_velocity[0] != left_velocity[0] or right_velocity[2] != left_velocity[2] or right_velocity[1] != -left_velocity[1]:
            raise AssertionError(f"{pair_id} velocities are not mirrored")
        left_attitude = left_scene["initial_attitudes"]["uav_00"]
        right_attitude = right_scene["initial_attitudes"]["uav_00"]
        if right_attitude[0] != left_attitude[0] or right_attitude[2] != left_attitude[2] or right_attitude[1] != -left_attitude[1]:
            raise AssertionError(f"{pair_id} attitudes are not mirrored")
        left_box = left_scene["obstacle_specs"][0]
        right_box = right_scene["obstacle_specs"][0]
        if right_box["center_m"][0] != left_box["center_m"][0] or right_box["center_m"][2] != left_box["center_m"][2] or right_box["center_m"][1] != _mirror_y(left_box["center_m"][1]) or right_box["size_m"] != left_box["size_m"]:
            raise AssertionError(f"{pair_id} boxes are not mirrored")
        if left_box["obstacle_id"] == right_box["obstacle_id"]:
            raise AssertionError(f"{pair_id} obstacle IDs must remain distinct")


def _validate_split_separation(train_pool: Sequence[Mapping[str, Any]], validation_pool: Sequence[Mapping[str, Any]]) -> None:
    for field in ("episode_id", "scenario_id", "scenario_seed", "noise_seed", "geometry_sha256"):
        train_values = {entry[field] for entry in train_pool}
        validation_values = {entry[field] for entry in validation_pool}
        if train_values & validation_values:
            raise AssertionError(f"train/validation leakage in {field}")


def _entry_lookup(pool: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in pool:
        kind = str(entry["kind"])
        index = int(str(entry["scenario_id"]).rsplit("_", 1)[-1])
        lookup[(kind, index)] = entry
    return lookup


def _build_schedule(train_pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _entry_lookup(train_pool)
    stage_specs = (
        ("stage1_normal", [("normal", index) for index in range(6)]),
        ("stage2_single", [("normal", 0), ("normal", 1)] + [("single", index) for index in range(4)]),
        ("stage3_full", [("normal", 4), ("normal", 5), ("single", 0), ("single", 1), ("full", 0), ("full", 1)]),
    )
    schedule: list[dict[str, Any]] = []
    global_update = 1
    for stage_index, (stage_name, cases) in enumerate(stage_specs, start=1):
        if len(cases) != 6:
            raise AssertionError("every curriculum stage must contain six updates")
        for update_in_stage, (kind, index) in enumerate(cases, start=1):
            entry = lookup[(kind, index)]
            scene_id = str(entry["scenario_id"])
            schedule.append(
                {
                    "global_update": global_update,
                    "stage": stage_index,
                    "update_in_stage": update_in_stage,
                    "stage_name": stage_name,
                    "scenario_id": scene_id,
                    "kind": str(entry["kind"]),
                    "side": str(entry["side"]),
                    "rollout_steps": ROLLOUT_STEPS,
                    "arms": ["A", "B"],
                    "shared_both_arms": True,
                }
            )
            global_update += 1
    return schedule


def _validate_schedule(schedule: Sequence[Mapping[str, Any]], train_pool: Sequence[Mapping[str, Any]]) -> None:
    if len(schedule) != PPO_UPDATES:
        raise AssertionError(f"expected {PPO_UPDATES} update blocks")
    ids = {str(entry["scenario_id"]) for entry in train_pool}
    expected_stages = ("stage1_normal", "stage2_single", "stage3_full")
    for index, update in enumerate(schedule, start=1):
        if update["global_update"] != index:
            raise AssertionError("global update numbering must be contiguous")
        stage_name = str(update["stage_name"])
        if stage_name not in expected_stages:
            raise AssertionError(f"unknown stage {stage_name}")
        if not 1 <= int(update["update_in_stage"]) <= 6:
            raise AssertionError("stage update must be in 1..6")
        if update["scenario_id"] not in ids:
            raise AssertionError("schedule references an unavailable scene")
        if update["kind"] not in {"normal", "single", "full"}:
            raise AssertionError("schedule references an unknown scenario kind")
        if update["rollout_steps"] != ROLLOUT_STEPS:
            raise AssertionError("schedule rollout budget drifted")
        if update["arms"] != ["A", "B"] or not update["shared_both_arms"]:
            raise AssertionError("schedule must be shared by both arms")
    if [entry["update_in_stage"] for entry in schedule[:6]] != list(range(1, 7)):
        raise AssertionError("stage1 update numbering drifted")
    if [entry["update_in_stage"] for entry in schedule[6:12]] != list(range(1, 7)):
        raise AssertionError("stage2 update numbering drifted")
    if [entry["update_in_stage"] for entry in schedule[12:18]] != list(range(1, 7)):
        raise AssertionError("stage3 update numbering drifted")
    if sum(int(entry["rollout_steps"]) for entry in schedule) != 2_160:
        raise AssertionError("PPO steps per arm must total 2160")


def _validate_pools(train_pool: Sequence[Mapping[str, Any]], validation_pool: Sequence[Mapping[str, Any]]) -> None:
    if len(train_pool) != 18 or len(validation_pool) != 10:
        raise AssertionError("smoke pool sizes must be 18 train and 10 validation")
    for pool, split in ((train_pool, "train"), (validation_pool, "validation")):
        for entry in pool:
            if entry["split"] != split:
                raise AssertionError(f"entry has wrong split {entry['split']!r}")
            _validate_entry(entry)
        if len({entry["scenario_id"] for entry in pool}) != len(pool):
            raise AssertionError(f"duplicate {split} scenario IDs")
        counts = Counter(str(entry["kind"]) for entry in pool)
        expected = {"normal": 6, "single": 6, "full": 6} if split == "train" else {"normal": 2, "single": 4, "full": 4}
        if counts != expected:
            raise AssertionError(f"unexpected {split} kind balance {counts}")
        seeds = [entry["scenario_seed"] for entry in pool]
        noise_seeds = [entry["noise_seed"] for entry in pool]
        if len(set(seeds)) != len(seeds) or len(set(noise_seeds)) != len(noise_seeds):
            raise AssertionError(f"duplicate {split} random seeds")
        _assert_mirrors(pool)
    _validate_split_separation(train_pool, validation_pool)


def build_smoke_curriculum(seed: int = 2026) -> dict[str, Any]:
    """Build and validate the complete deterministic smoke curriculum.

    The returned mapping is safe to serialize as JSON.  It contains the full
    available train/validation pools, the 18 update blocks, and provenance that
    distinguishes available cases from cases actually exposed to PPO.  The six
    available train cases omitted from PPO rollout scheduling are reserved for
    train-only calibration and are never scheduled as PPO rollout scenes.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_pool = _build_split("train", seed, 0)
    validation_pool = _build_split("validation", seed, 100)
    _validate_pools(train_pool, validation_pool)
    schedule = _build_schedule(train_pool)
    _validate_schedule(schedule, train_pool)
    train_by_id = {str(entry["scenario_id"]): entry for entry in train_pool}
    available_ids = list(train_by_id)
    exposed_ids: list[str] = []
    for update in schedule:
        scene_id = str(update["scenario_id"])
        if scene_id not in exposed_ids:
            exposed_ids.append(scene_id)
    calibration_only_ids = [
        str(train_by_id[_case_id("train", "single", index)]["scenario_id"])
        for index in range(4, 6)
    ] + [
        str(train_by_id[_case_id("train", "full", index)]["scenario_id"])
        for index in range(2, 6)
    ]
    provenance = {
        "generator": "single_uav_ppo_smoke_curriculum",
        "version": 1,
        "seed": int(seed),
        "bounds_min_m": list(BOUNDS_MIN_M),
        "bounds_max_m": list(BOUNDS_MAX_M),
        "available_train_case_ids": available_ids,
        "available_validation_case_ids": [str(entry["scenario_id"]) for entry in validation_pool],
        "actually_exposed_case_ids": exposed_ids,
        "calibration_only_case_ids": calibration_only_ids,
        "unscheduled_available_case_ids": [case_id for case_id in available_ids if case_id not in exposed_ids],
        "available_pool_vs_actually_exposed_distinct": len(available_ids) != len(exposed_ids),
        "geometry_hash_definition": "canonical full scene geometry; obstacle_id excluded",
        "train_validation_split_assertions": {
            "ids_disjoint": True,
            "scenario_seeds_disjoint": True,
            "noise_seeds_disjoint": True,
            "geometry_hashes_disjoint": True,
        },
        "schedule_assertions": {
            "update_blocks": len(schedule),
            "rollout_steps_per_arm": sum(int(update["rollout_steps"]) for update in schedule),
            "train_only_calibration_steps": len(schedule) * CALIBRATION_STEPS_PER_UPDATE,
            "calibration_steps_per_update": CALIBRATION_STEPS_PER_UPDATE,
            "calibration_only_case_ids": list(calibration_only_ids),
            "both_arms_shared": all(bool(update["shared_both_arms"]) for update in schedule),
        },
    }
    result = {
        "seed": int(seed),
        "train_pool": train_pool,
        "validation_pool": validation_pool,
        "schedule": schedule,
        "provenance": provenance,
    }
    # A final serialization round trip catches accidental NumPy/scalar values
    # before a runner writes the manifest.
    json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "BOUNDS_MAX_M",
    "BOUNDS_MIN_M",
    "CALIBRATION_STEPS_PER_UPDATE",
    "PPO_UPDATES",
    "ROLLOUT_STEPS",
    "build_smoke_curriculum",
]
