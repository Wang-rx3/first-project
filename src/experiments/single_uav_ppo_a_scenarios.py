"""Frozen scenario and open-loop witness helpers for the PPO-A audit.

The helpers in this module are deliberately deterministic and side-effect free
at import time.  They build a fresh manifest from the historical v1 curriculum
and produce an analytic, fixed-action witness without constructing an
environment.  Environment execution belongs to the root runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CURRICULUM = _PROJECT_ROOT / "data" / "synthetic" / "single_uav_static_ppo_smoke_v1_seed2026" / "curriculum.json"
_OLD_TRAIN_IDS = tuple(f"smoke_train_{kind}_{index:02d}" for kind in ("normal", "single", "full") for index in range(6))


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json(value.item())
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _geometry_hash(scenario: Mapping[str, Any]) -> str:
    """Match the v1 canonical geometry contract while omitting obstacle IDs."""

    obstacle_specs = []
    for spec in scenario.get("obstacle_specs", []):
        if not isinstance(spec, Mapping):
            continue
        obstacle_specs.append(
            {
                "geometry_type": str(spec["geometry_type"]),
                "center_m": [float(value) for value in spec["center_m"]],
                "size_m": [float(value) for value in spec["size_m"]],
                "physical_collision_enabled": bool(spec["physical_collision_enabled"]),
            }
        )
    payload: dict[str, Any] = {
        "initial_positions": {str(key): [float(value) for value in values] for key, values in scenario.get("initial_positions", {}).items()},
        "goals": {str(key): [float(value) for value in values] for key, values in scenario.get("goals", {}).items()},
        "obstacle_specs": obstacle_specs,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
    }
    if "obstacles" in scenario:
        payload["obstacles"] = []
    return _canonical(payload)


def _load_curriculum(path: Path = _CURRICULUM) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen v1 curriculum: {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("train_pool"), list):
        raise ValueError("frozen v1 curriculum must contain a train_pool list")
    return payload


def _scene_entry(
    *,
    episode_id: str,
    kind: str,
    split: str,
    scenario_seed: int,
    noise_seed: int,
    scenario: Mapping[str, Any],
    side: str = "none",
    mirror_pair: str | None = None,
) -> dict[str, Any]:
    scene = copy.deepcopy(dict(scenario))
    scene.setdefault("obstacle_specs", [])
    scene["foundation_task_kind"] = str(kind)
    scene["obstacle_specs"] = copy.deepcopy(scene.get("obstacle_specs", []))
    # The simulator treats an explicit ``obstacles=[]`` reset field as an
    # override that suppresses materialization from ``obstacle_specs``.  That
    # override belongs only to normal scenes; physical cases must retain their
    # static box specifications and omit the empty field.
    if kind == "normal":
        scene["obstacles"] = []
    elif "obstacles" in scene:
        scene["obstacles"] = copy.deepcopy(scene["obstacles"])
    return {
        "episode_id": str(episode_id),
        "scenario_id": str(episode_id),
        "kind": str(kind),
        "side": str(side),
        "mirror_pair": mirror_pair,
        "split": str(split),
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "geometry_sha256": _geometry_hash(scene),
        "scenario": scene,
    }


def _normal_validation(index: int) -> dict[str, Any]:
    i = int(index)
    sign = 1.0 if i % 2 == 0 else -1.0
    # The alternating offsets are intentionally asymmetric to avoid making the
    # validation set a translated copy of the train trajectories.
    dy = 0.35 if i % 2 == 0 else -0.45
    scene = {
        "initial_positions": {"uav_00": [21.25 + 3.2 * i, 93.0 + 1.5 * i, 8.4 + 0.6 * i]},
        "initial_velocities": {"uav_00": [0.25 + 0.1 * (i % 3), 0.08 * sign, 0.0]},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [21.25 + 3.2 * i + 4.2 + 0.48 * i, 93.0 + 1.5 * i + dy, 8.4 + 0.6 * i]},
        "obstacle_specs": [],
        "obstacles": [],
    }
    return _scene_entry(
        episode_id=f"foundation_validation_normal_{i:02d}",
        kind="normal",
        split="validation",
        scenario_seed=2026190000 + i,
        noise_seed=2026180000 + i,
        scenario=scene,
    )


def _translate_physical(old: Mapping[str, Any], *, kind: str, index: int) -> dict[str, Any]:
    scenario = copy.deepcopy(dict(old["scenario"]))
    pair_key = str(old.get("mirror_pair") or "")
    # Pair indices are stable within each physical task family; mirrored cases
    # receive exactly the same translation, preserving the old pair relation.
    pair_index = int(str(pair_key).rsplit("_", 1)[-1]) if pair_key.rsplit("_", 1)[-1].isdigit() else index // 2
    offset = np.asarray([5.35 + 0.13 * pair_index, 4.85 + 0.21 * pair_index, 0.55], dtype=float)

    for field in ("initial_positions", "goals"):
        values = scenario.get(field, {})
        if isinstance(values, MutableMapping):
            for uav_id, point in list(values.items()):
                values[uav_id] = (np.asarray(point, dtype=float) + offset).tolist()
    specs = scenario.get("obstacle_specs", [])
    if isinstance(specs, list):
        for spec in specs:
            if isinstance(spec, MutableMapping) and "center_m" in spec:
                spec["center_m"] = (np.asarray(spec["center_m"], dtype=float) + offset).tolist()
                # IDs are metadata and intentionally not part of the geometry
                # hash, but fresh IDs make the manifest unambiguous.
                spec["obstacle_id"] = f"foundation_{kind}_obstacle_{index:02d}"
    # A few historical entries carry a materialized obstacle list.  Keep it in
    # sync when present, while preserving all non-geometric metadata.
    obstacles = scenario.get("obstacles", [])
    if isinstance(obstacles, list):
        for obstacle in obstacles:
            if isinstance(obstacle, MutableMapping) and "center_m" in obstacle:
                obstacle["center_m"] = (np.asarray(obstacle["center_m"], dtype=float) + offset).tolist()
                obstacle["obstacle_id"] = f"foundation_{kind}_obstacle_{index:02d}"
    pair_name = f"foundation_validation_{kind}_pair_{pair_index:02d}"
    return _scene_entry(
        episode_id=f"foundation_validation_{kind}_{index:02d}",
        kind=kind,
        split="validation",
        scenario_seed=2026191000 + index if kind == "single" else 2026192000 + index,
        noise_seed=2026181000 + index if kind == "single" else 2026182000 + index,
        scenario=scenario,
        side=str(old.get("side", "none")),
        mirror_pair=pair_name,
    )


def _assert_manifest(manifest: Mapping[str, Any], old: Mapping[str, Any]) -> None:
    train = list(manifest["train_pool"])
    normal = list(manifest["validation_normal"])
    physical = list(manifest["validation_single"]) + list(manifest["validation_full"])
    if len(train) != 18 or len(manifest["normal_train"]) != 6 or len(normal) != 8 or len(physical) != 8:
        raise AssertionError("foundation manifest cardinality mismatch")
    old_entries = list(old["train_pool"])
    old_val = list(old.get("validation_pool", []))
    old_ids = {str(item.get("episode_id", item.get("scenario_id", ""))) for item in old_entries + old_val}
    fresh = normal + physical
    ids = [str(item["episode_id"]) for item in fresh]
    if len(set(ids)) != len(ids) or old_ids.intersection(ids):
        raise AssertionError("fresh validation IDs overlap frozen v1 IDs")
    seeds = [(int(item["scenario_seed"]), int(item["noise_seed"])) for item in train + old_val]
    fresh_seeds = [(int(item["scenario_seed"]), int(item["noise_seed"])) for item in fresh]
    if set(seeds).intersection(fresh_seeds):
        raise AssertionError("fresh validation seeds overlap frozen v1 seeds")
    if len({item["geometry_sha256"] for item in train + old_val}) != len(train + old_val):
        # This is a historical assertion, not a requirement to repair old data.
        pass
    if set(item["geometry_sha256"] for item in fresh).intersection(item["geometry_sha256"] for item in train + old_val):
        raise AssertionError("fresh validation geometry overlaps frozen v1 geometry")
    for item in manifest["normal_train"] + normal:
        scenario = item["scenario"]
        if item["kind"] == "normal" and (scenario.get("obstacles") or scenario.get("obstacle_specs")):
            raise AssertionError("normal foundation cases must have no obstacles")
        if scenario.get("foundation_task_kind") != item["kind"]:
            raise AssertionError("foundation_task_kind marker missing")
    for item in physical:
        specs = item["scenario"].get("obstacle_specs", [])
        if item["kind"] not in {"single", "full"} or len(specs) != 1:
            raise AssertionError("physical foundation validation case malformed")
        if "obstacles" in item["scenario"] and item["scenario"]["obstacles"] == []:
            raise AssertionError("physical foundation cases must not carry an empty obstacle override")
        spec = specs[0]
        if not isinstance(spec, Mapping) or spec.get("geometry_type") != "box" or not bool(spec.get("physical_collision_enabled", False)):
            raise AssertionError("physical foundation case must contain one static physical box")


def build_a_foundation_scenarios(seed: int = 2026) -> dict[str, Any]:
    """Return the deterministic 18-train/16-validation foundation manifest."""

    if int(seed) != 2026:
        raise ValueError("foundation scenarios are frozen at seed=2026")
    old = _load_curriculum()
    original_pool = copy.deepcopy(list(old["train_pool"]))
    if len(original_pool) != 18:
        raise ValueError("frozen v1 curriculum must contain exactly 18 training cases")
    # Keep the historical train records byte-for-byte semantically unchanged
    # and expose normal markers only in copied foundation cases.
    # ``train_pool`` is an immutable copy of the source records.  Foundation
    # markers live on the execution-facing normal_train copies so the source
    # curriculum remains directly auditable by ID/geometry/seed.
    train_pool: list[dict[str, Any]] = [copy.deepcopy(dict(item)) for item in original_pool]
    normal_train = [copy.deepcopy(item) for item in train_pool if item.get("kind") == "normal"]
    for entry in normal_train:
        scenario = entry.setdefault("scenario", {})
        scenario["foundation_task_kind"] = "normal"
        scenario.setdefault("obstacle_specs", [])
        scenario.setdefault("obstacles", [])
    old_validation = list(old.get("validation_pool", []))
    if not old_validation:
        # v1 manifests used ``validation`` in a few early snapshots.  The
        # current frozen curriculum uses validation_pool; accept either key.
        old_validation = list(old.get("validation", []))
    old_single = [item for item in old_validation if str(item.get("kind")) == "single"][:4]
    old_full = [item for item in old_validation if str(item.get("kind")) == "full"][:4]
    if len(old_single) != 4 or len(old_full) != 4:
        raise ValueError("frozen v1 curriculum must contain four single and four full validation cases")
    validation_normal = [_normal_validation(i) for i in range(8)]
    validation_single = [_translate_physical(item, kind="single", index=i) for i, item in enumerate(old_single)]
    validation_full = [_translate_physical(item, kind="full", index=i) for i, item in enumerate(old_full)]
    train_single = [item for item in train_pool if item.get("kind") == "single"]
    train_full = [item for item in train_pool if item.get("kind") == "full"]
    # Schedules name the frozen training IDs.  Fresh validation cases are
    # deliberately absent from the schedule and are used only by feasibility
    # and downstream diagnostics.
    schedules = {
        "normal": [str(item["scenario_id"]) for item in normal_train],
        "single": [str(item["scenario_id"]) for item in normal_train[:2]] + [str(item["scenario_id"]) for item in train_single[:4]],
        "full": [str(item["scenario_id"]) for item in normal_train[:2]] + [str(item["scenario_id"]) for item in train_single[:2]] + [str(item["scenario_id"]) for item in train_full[:2]],
    }
    provenance = {
        "manifest": "single_uav_ppo_a_foundation_v1",
        "seed": 2026,
        "source_curriculum": str(_CURRICULUM),
        "source_train_case_ids": [str(item.get("scenario_id", item.get("episode_id", ""))) for item in original_pool],
        "source_validation_case_ids": [str(item.get("scenario_id", item.get("episode_id", ""))) for item in old_validation],
        "train_pool_count": len(train_pool),
        "validation_counts": {"normal": len(validation_normal), "single": len(validation_single), "full": len(validation_full)},
        "geometry_hash_definition": "canonical positions/goals/obstacle geometry; obstacle IDs and metadata excluded",
        "fresh_validation_geometry": True,
        "fresh_validation_seeds": True,
        "old_curriculum_immutable": True,
    }
    manifest = {
        "seed": 2026,
        "train_pool": train_pool,
        "normal_train": normal_train,
        "validation_normal": validation_normal,
        "validation_single": validation_single,
        "validation_full": validation_full,
        "schedules": schedules,
        "provenance": provenance,
    }
    _assert_manifest(manifest, old)
    return manifest


def _base_value(base: Any, name: str, default: Any) -> Any:
    if isinstance(base, Mapping):
        return base.get(name, default)
    return getattr(base, name, default)


def _scene_from(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if "scenario" in value and isinstance(value["scenario"], Mapping):
        return value["scenario"]
    return value


def _polynomial_velocities(v0: np.ndarray, displacement: np.ndarray, *, dt: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n + 1, dtype=float)
    linear = 1.0 - k / float(n)
    shape = k * (float(n) - k)
    base_integral = dt * np.sum(0.5 * (linear[:-1] + linear[1:]))
    shape_integral = dt * np.sum(0.5 * (shape[:-1] + shape[1:]))
    if shape_integral <= 0.0:
        raise ValueError("invalid witness horizon")
    coefficient = (np.asarray(displacement, dtype=float) - np.asarray(v0, dtype=float) * base_integral) / shape_integral
    velocities = linear[:, None] * np.asarray(v0, dtype=float)[None, :] + shape[:, None] * coefficient[None, :]
    return velocities, coefficient


def _integrate(position: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
    points = [np.asarray(position, dtype=float).copy()]
    for left, right in zip(velocities[:-1], velocities[1:]):
        points.append(points[-1] + 0.5 * (left + right) * dt)
    return np.asarray(points, dtype=float)


def _bounded_fallback(v0: np.ndarray, displacement: np.ndarray, *, dt: float, n: int, max_speed: float, max_delta: float) -> np.ndarray:
    """Construct a finite bounded witness if the mandated polynomial is too abrupt.

    The normal foundation cases use the mandated polynomial.  The fallback is
    only needed by long physical cases whose 35-step polynomial violates the
    per-step acceleration limit; it preserves the endpoint and limits.
    """
    target = np.asarray(displacement, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    # Start from a smooth triangular velocity envelope and solve its constant
    # acceleration scale by bisection.  A tiny endpoint correction is applied
    # to the middle velocity samples and then projected to the component-wise
    # Euclidean limit.
    t = np.linspace(0.0, 1.0, n + 1)
    envelope = 4.0 * t * (1.0 - t)
    velocities = np.zeros((n + 1, 3), dtype=float)
    velocities[0] = v0
    for axis in range(3):
        direction = 1.0 if target[axis] >= 0 else -1.0
        lo, hi = 0.0, max_speed
        for _ in range(80):
            amp = (lo + hi) / 2.0
            candidate = v0[axis] * (1.0 - t) + direction * amp * envelope
            distance = dt * np.sum(0.5 * (candidate[:-1] + candidate[1:]))
            if direction * distance < direction * target[axis]:
                lo = amp
            else:
                hi = amp
        velocities[:, axis] = v0[axis] * (1.0 - t) + direction * hi * envelope
    velocities[-1] = 0.0
    for _ in range(8):
        actual = dt * np.sum(0.5 * (velocities[:-1] + velocities[1:]), axis=0)
        error = target - actual
        velocities[1:-1] += error[None, :] / max((n - 1) * dt, 1e-9)
        velocities = np.clip(velocities, -max_speed, max_speed)
        velocities[0] = v0
        velocities[-1] = 0.0
    delta = np.diff(velocities, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    for i, norm in enumerate(norms):
        if norm > max_delta:
            velocities[i + 1] = velocities[i] + delta[i] * (max_delta / norm)
    velocities[-1] = 0.0
    return velocities


def build_open_loop_witness(scene: Mapping[str, Any], base: Any) -> dict[str, Any]:
    """Build a fixed world-frame velocity witness and analytic predictions."""

    scenario = _scene_from(scene)
    try:
        position = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
        velocity = np.asarray(scenario["initial_velocities"]["uav_00"], dtype=float).reshape(3)
        goal = np.asarray(scenario["goals"]["uav_00"], dtype=float).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scenario must define uav_00 initial position, velocity, and goal") from exc
    if not np.isfinite(np.concatenate((position, velocity, goal))).all():
        raise ValueError("witness scenario state must be finite")
    dt = float(_base_value(base, "dt_s", 0.2))
    max_speed = float(_base_value(base, "max_speed_mps", 12.0))
    max_accel = float(_base_value(base, "max_acceleration_mps2", 3.0))
    lower = np.asarray(_base_value(base, "bounds_min_m", (0.0, 0.0, 2.0)), dtype=float)
    upper = np.asarray(_base_value(base, "bounds_max_m", (200.0, 200.0, 60.0)), dtype=float)
    max_delta = max_accel * dt
    displacement = goal - position
    chosen_n: int | None = None
    chosen_velocities: np.ndarray | None = None
    chosen_coefficients: np.ndarray | None = None
    for n in range(30, 36):
        trial, coefficients = _polynomial_velocities(velocity, displacement, dt=dt, n=n)
        trial_positions = _integrate(position, trial, dt)
        if (
            np.linalg.norm(trial, axis=1).max() <= max_speed + 1e-9
            and np.linalg.norm(np.diff(trial, axis=0), axis=1).max() <= max_delta + 1e-9
            and np.all(trial_positions >= lower - 1e-9)
            and np.all(trial_positions <= upper + 1e-9)
        ):
            chosen_n, chosen_velocities, chosen_coefficients = n, trial, coefficients
            break
    fallback = False
    if chosen_velocities is None:
        chosen_n = 35
        chosen_velocities = _bounded_fallback(velocity, displacement, dt=dt, n=chosen_n, max_speed=max_speed, max_delta=max_delta)
        chosen_coefficients = np.zeros(3, dtype=float)
        fallback = True
    predicted_positions = _integrate(position, chosen_velocities, dt)
    # Correct accumulated floating-point drift in the final position while
    # retaining the exact trapezoid construction used by the witness.
    correction = goal - predicted_positions[-1]
    if np.linalg.norm(correction) > 1e-9 and chosen_n > 1:
        chosen_velocities[1:-1] += correction[None, :] / max((chosen_n - 1) * dt, 1e-9)
        chosen_velocities[0] = velocity
        chosen_velocities[-1] = 0.0
        predicted_positions = _integrate(position, chosen_velocities, dt)
    dynamic_actions = np.diff(chosen_velocities, axis=0)
    actions = np.vstack((dynamic_actions, np.zeros((5, 3), dtype=float)))
    predicted_velocities = np.vstack((chosen_velocities, np.zeros((5, 3), dtype=float)))
    predicted_positions_with_hold = [*predicted_positions.tolist()]
    for _ in range(5):
        predicted_positions_with_hold.append(predicted_positions_with_hold[-1])
    action_norms = np.linalg.norm(actions, axis=1)
    speed_norms = np.linalg.norm(predicted_velocities, axis=1)
    if len(actions) > 40:
        raise AssertionError("open-loop witness exceeds action budget")
    return {
        "scenario_id": str(scene.get("scenario_id", scene.get("episode_id", ""))) if isinstance(scene, Mapping) else "",
        "actions": actions.tolist(),
        "dynamic_actions": int(chosen_n),
        "dynamic_steps": int(chosen_n),
        "hold_steps": 5,
        "dt_s": dt,
        "initial_position": position.tolist(),
        "initial_velocity": velocity.tolist(),
        "goal": goal.tolist(),
        "predicted_positions": predicted_positions_with_hold,
        "predicted_velocities": predicted_velocities.tolist(),
        "polynomial_coefficients": np.asarray(chosen_coefficients).tolist(),
        "polynomial_fallback": bool(fallback),
        "nominal_limits": {
            "max_speed_mps": max_speed,
            "max_acceleration_mps2": max_accel,
            "max_delta_v": max_delta,
            "max_predicted_speed_mps": float(speed_norms.max()),
            "max_commanded_delta_v": float(action_norms.max()),
            "speed_ok": bool(speed_norms.max() <= max_speed + 1e-8),
            "acceleration_ok": bool(action_norms.max() <= max_delta + 1e-8),
            "bounds_ok": bool(np.all(np.asarray(predicted_positions_with_hold) >= lower - 1e-8) and np.all(np.asarray(predicted_positions_with_hold) <= upper + 1e-8)),
            "endpoint_error_m": float(np.linalg.norm(np.asarray(predicted_positions[-1]) - goal)),
        },
        "source": "analytic_fixed_world_velocity_polynomial",
        "teacher_data": False,
        "policy_initialization": False,
    }


__all__ = ["build_a_foundation_scenarios", "build_open_loop_witness"]
