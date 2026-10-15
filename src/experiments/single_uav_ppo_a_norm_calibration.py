"""Calibration and leakage-safe full statistics for the A normalization arm.

This module is deliberately separate from the historical smoke/foundation
runner.  Scenario construction, reset-only schema checks, and statistics
serialization are safe to import and test without running a rollout.  The
calibration collector is the only function here that advances an environment;
callers should invoke it once, after their preflight checks.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..learning.mappo import central_state_to_vector
from .single_uav_ppo_baseline import (
    EXPECTED_NODE_FEATURE_DIM,
    FEATURE_INDEX_MAP,
    create_single_uav_ppo_environment,
    create_single_uav_ppo_trainer,
    verify_actor_observation_contract,
)
from .single_uav_ppo_a_support import initialize_a_foundation
from .single_uav_ppo_a_scenarios import (
    _geometry_hash,
    build_a_foundation_scenarios,
)


EXPECTED_CENTRAL_FEATURE_DIM = 174
CALIBRATION_EPISODES = 24
CALIBRATION_STEPS_PER_EPISODE = 120
CALIBRATION_OBSERVATIONS = 2904
CALIBRATION_ENV_STEPS = 2880
_EPS = 1.0e-6
_CLIP = 10.0


def _json_value(value: Any) -> Any:
    """Convert numpy values to strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite JSON value")
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _copy_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(scene))


def _scene_entry(
    *,
    scenario_id: str,
    split: str,
    side: str,
    noise_seed: int,
    scenario: Mapping[str, Any],
    **metadata: Any,
) -> dict[str, Any]:
    scene = _copy_scene(scenario)
    scene.setdefault("obstacles", [])
    scene.setdefault("obstacle_specs", [])
    scene["foundation_task_kind"] = "normal"
    # Explicit empty lists are part of the no-obstacle reset contract.  They
    # prevent the environment from materializing stale physical geometry.
    scene["obstacles"] = []
    scene["obstacle_specs"] = []
    record: dict[str, Any] = {
        "scenario_id": str(scenario_id),
        "episode_id": str(scenario_id),
        "split": str(split),
        "kind": "normal",
        "side": str(side),
        "noise_seed": int(noise_seed),
        "geometry_sha256": _geometry_hash(scene),
        "scenario": scene,
    }
    record.update({str(key): _json_value(value) for key, value in metadata.items()})
    return record


def _old_scene_ids_and_geometry() -> tuple[set[str], set[str], set[int]]:
    """Read the frozen v1/foundation pools used only for overlap assertions."""

    curriculum_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "synthetic"
        / "single_uav_static_ppo_smoke_v1_seed2026"
        / "curriculum.json"
    )
    old_ids: set[str] = set()
    old_geometry: set[str] = set()
    old_seeds: set[int] = set()
    try:
        payload = json.loads(curriculum_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    records = list(payload.get("train_pool", [])) + list(payload.get("validation_pool", []))
    # The foundation manifest has fresh normal validation records derived from
    # this curriculum.  Include it when available so independent calibration
    # cases cannot silently reuse those geometries either.
    try:
        foundation = build_a_foundation_scenarios()
        records.extend(foundation.get("validation_normal", []))
    except Exception:
        # A manifest builder error is reported by its own API; construction of
        # this independent manifest remains deterministic when a frozen input
        # is temporarily unavailable.
        pass
    for record in records:
        if not isinstance(record, Mapping):
            continue
        scenario_id = record.get("scenario_id", record.get("episode_id"))
        if scenario_id is not None:
            old_ids.add(str(scenario_id))
        scenario = record.get("scenario")
        if isinstance(scenario, Mapping):
            try:
                old_geometry.add(str(record.get("geometry_sha256", _geometry_hash(scenario))))
            except Exception:
                pass
        for name in ("scenario_seed", "noise_seed"):
            if name in record:
                try:
                    old_seeds.add(int(record[name]))
                except (TypeError, ValueError):
                    pass
    return old_ids, old_geometry, old_seeds


def _assert_split_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {"calibration": 12, "train": 6, "validation": 8, "challenge": 4}
    if any(key not in manifest for key in expected):
        raise AssertionError(f"split manifest missing keys: {sorted(set(expected) - set(manifest))}")
    for key, count in expected.items():
        values = manifest[key]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != count:
            raise AssertionError(f"{key} must contain exactly {count} scenes")
    scenes = [scene for key in expected for scene in manifest[key]]
    ids = [str(scene.get("scenario_id")) for scene in scenes]
    seeds = [int(scene.get("noise_seed")) for scene in scenes]
    hashes = [str(scene.get("geometry_sha256")) for scene in scenes]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise AssertionError("split scene IDs and noise seeds must be globally unique")
    if any(not item or len(item) != 64 for item in hashes):
        raise AssertionError("every split scene requires a SHA256 geometry hash")
    for scene in scenes:
        if not isinstance(scene.get("scenario"), Mapping):
            raise AssertionError("split scene missing scenario payload")
        if scene.get("kind") != "normal" or scene["scenario"].get("foundation_task_kind") != "normal":
            raise AssertionError("all calibration/validation/challenge scenes must be normal tasks")
        if scene["scenario"].get("obstacles") != [] or scene["scenario"].get("obstacle_specs") != []:
            raise AssertionError("normal split scene must declare empty obstacles and obstacle_specs")
    train_ids = {str(scene.get("scenario_id")) for scene in manifest["train"]}
    for key in ("calibration", "validation", "challenge"):
        if train_ids.intersection(str(scene.get("scenario_id")) for scene in manifest[key]):
            raise AssertionError(f"{key} scene ID overlaps train")


def build_split_manifest() -> dict[str, Any]:
    """Build the independent calibration, train, validation, and challenge splits.

    Values are scene records (the cardinalities are 12/6/8/4).  ``counts``
    and ``provenance`` make the cardinality and anti-overlap contract explicit
    for callers that serialize the manifest.
    """

    foundation = build_a_foundation_scenarios()
    train = [_copy_scene(scene) for scene in foundation["normal_train"]]
    if len(train) != 6:
        raise AssertionError("the frozen foundation normal training pool must contain six scenes")

    old_ids, old_geometry, old_seeds = _old_scene_ids_and_geometry()
    train_ids = {str(scene["scenario_id"]) for scene in train}
    train_geometry = {str(scene["geometry_sha256"]) for scene in train}
    train_seeds = {
        int(scene.get("noise_seed", -1))
        for scene in train
    }

    calibration: list[dict[str, Any]] = []
    axes = (("x", np.array([1.0, 0.0, 0.0])), ("y", np.array([0.0, 1.0, 0.0])), ("z", np.array([0.0, 0.0, 1.0])))
    directions = [(axis, sign, vector * float(sign)) for axis, vector in axes for sign in (1, -1)]
    for magnitude_index, magnitude in enumerate((4.0, 8.0)):
        for direction_index, (axis, sign, direction) in enumerate(directions):
            index = magnitude_index * len(directions) + direction_index
            position = np.array([22.0 + 3.0 * index, 90.0 + 1.5 * index, 14.0 + 0.3 * index])
            goal = position + direction * magnitude
            scenario = {
                "initial_positions": {"uav_00": position.tolist()},
                "initial_velocities": {"uav_00": [0.35, 0.0, 0.0]},
                "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
                "goals": {"uav_00": goal.tolist()},
                "obstacles": [],
                "obstacle_specs": [],
                "foundation_task_kind": "normal",
            }
            calibration.append(
                _scene_entry(
                    scenario_id=f"norm_calibration_{axis}_{'plus' if sign > 0 else 'minus'}_{int(magnitude):02d}m",
                    split="calibration",
                    side=f"{axis}{'+' if sign > 0 else '-'}",
                    noise_seed=202611210000 + index,
                    scenario=scenario,
                    axis=axis,
                    direction_sign=int(sign),
                    magnitude_m=float(magnitude),
                    pair_magnitude_index=int(magnitude_index),
                )
            )

    validation: list[dict[str, Any]] = []
    for index in range(8):
        position = np.array([34.0 + 2.7 * index, 71.0 + 2.1 * index, 10.5 + 0.8 * index])
        y_sign = 1.0 if index % 2 == 0 else -1.0
        z_sign = 1.0 if index % 3 != 0 else -1.0
        x_speed = 0.30 + 0.02 * (index % 8)
        velocity = np.array([x_speed, y_sign * 0.04, 0.0])
        offset = 4.2 + 0.45 * index
        goal = position + np.array([offset, y_sign * 0.35, z_sign * 0.15])
        scenario = {
            "initial_positions": {"uav_00": position.tolist()},
            "initial_velocities": {"uav_00": velocity.tolist()},
            "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
            "goals": {"uav_00": goal.tolist()},
            "obstacles": [],
            "obstacle_specs": [],
            "foundation_task_kind": "normal",
        }
        validation.append(
            _scene_entry(
                scenario_id=f"norm_fresh_validation_{index:02d}",
                split="validation",
                side=f"fresh_{'plus' if y_sign > 0 else 'minus'}",
                noise_seed=202611220000 + index,
                scenario=scenario,
                goal_offset_m=float(offset),
                lateral_offset_m=float(y_sign * 0.35),
                vertical_offset_m=float(z_sign * 0.15),
            )
        )

    challenge_specs = (
        ("stationary", [0.0, 0.0, 0.0], np.array([5.0, 0.0, 0.0])),
        ("stationary", [0.0, 0.0, 0.0], np.array([6.0, 0.0, 0.0])),
        ("fast", [1.2, 0.0, 0.0], np.array([7.0, 0.0, 0.0])),
        ("fast", [1.2, 0.0, 0.0], np.array([8.0, 0.0, 0.0])),
    )
    challenge: list[dict[str, Any]] = []
    for index, (challenge_type, velocity, displacement) in enumerate(challenge_specs):
        position = np.array([67.0 + 4.0 * index, 118.0 + 3.0 * index, 22.0 + 1.0 * index])
        goal = position + displacement
        scenario = {
            "initial_positions": {"uav_00": position.tolist()},
            "initial_velocities": {"uav_00": velocity},
            "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
            "goals": {"uav_00": goal.tolist()},
            "obstacles": [],
            "obstacle_specs": [],
            "foundation_task_kind": "normal",
        }
        challenge.append(
            _scene_entry(
                scenario_id=f"norm_challenge_{challenge_type}_{index:02d}",
                split="challenge",
                side=challenge_type,
                noise_seed=202611230000 + index,
                scenario=scenario,
                challenge_type=challenge_type,
                goal_distance_m=float(np.linalg.norm(displacement)),
            )
        )

    manifest: dict[str, Any] = {
        "calibration": calibration,
        "train": train,
        "validation": validation,
        "challenge": challenge,
        "counts": {"calibration": 12, "train": 6, "validation": 8, "challenge": 4},
        "provenance": {
            "manifest": "single_uav_ppo_a_norm_calibration_v2",
            "seed": 2026,
            "calibration_directions": ["+x", "-x", "+y", "-y", "+z", "-z"],
            "calibration_magnitudes_m": [4.0, 8.0],
            "train_source": "build_a_foundation_scenarios()['normal_train']",
            "old_scene_ids_checked": len(old_ids),
            "old_geometry_hashes_checked": len(old_geometry),
            "old_noise_seeds_checked": len(old_seeds),
            "train_geometry_hashes": sorted(train_geometry),
            "train_noise_seeds": sorted(train_seeds),
        },
    }
    _assert_split_manifest(manifest)
    # The independent split must not recycle any frozen ID/geometry/seed.  The
    # six training records intentionally retain their frozen IDs and geometry.
    independent = calibration + validation + challenge
    if {str(s["scenario_id"]) for s in independent}.intersection(old_ids):
        raise AssertionError("independent split IDs overlap the frozen curriculum/foundation")
    if {str(s["geometry_sha256"]) for s in independent}.intersection(old_geometry | train_geometry):
        raise AssertionError("independent split geometry overlaps the frozen curriculum/foundation")
    if {int(s["noise_seed"]) for s in independent}.intersection(old_seeds | train_seeds):
        raise AssertionError("independent split seeds overlap the frozen curriculum/foundation")
    return manifest


def _scene_scenario(scene: Mapping[str, Any]) -> dict[str, Any]:
    scenario = scene.get("scenario", scene)
    if not isinstance(scenario, Mapping):
        raise ValueError("scene must contain a scenario mapping")
    return copy.deepcopy(dict(scenario))


def _central(info: Mapping[str, Any], base: Any) -> np.ndarray:
    central_state = info.get("central_state")
    if not isinstance(central_state, Mapping):
        raise ValueError("environment reset info is missing central_state")
    vector = central_state_to_vector(
        central_state,
        uav_count=1,
        node_feature_dim=EXPECTED_NODE_FEATURE_DIM,
        edge_feature_dim=11,
        max_obstacles=int(getattr(base, "max_obstacles", 8)),
        include_remaining_time=False,
    )
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != EXPECTED_CENTRAL_FEATURE_DIM or not np.isfinite(vector).all():
        raise ValueError("central state vector must be finite and 174-dimensional")
    return vector


def schema_gate(base: Any, scene: Mapping[str, Any]) -> dict[str, Any]:
    """Run the actor/critic schema gate on a reset-only strict environment."""

    env = create_single_uav_ppo_environment(base)
    scenario = _scene_scenario(scene)
    seed = int(scene.get("noise_seed", 202611210000))
    observations, info = env.reset(seed=seed, scenario_config=scenario)
    uid = env.uav_ids[0]
    raw = np.asarray(observations[uid]["node_features"], dtype=np.float64)
    if raw.shape != (1, EXPECTED_NODE_FEATURE_DIM):
        raise AssertionError(f"actor node shape is {raw.shape}, expected (1,42)")
    central = _central(info, base)
    if not np.allclose(raw[0], central[:EXPECTED_NODE_FEATURE_DIM], atol=1e-5, rtol=0.0):
        raise AssertionError("actor node row does not match central vector first 42 entries")
    estimate = np.asarray(env.ekfs[uid].state, dtype=np.float64).reshape(-1)
    covariance = np.asarray(env.ekfs[uid].covariance, dtype=np.float64)
    expected_diag = np.diag(covariance).reshape(-1)
    goal = np.asarray(env.goals[uid], dtype=np.float64)
    direction = np.asarray(env._local_obstacle_direction_features(estimate), dtype=np.float64).reshape(-1)
    if not np.allclose(raw[0, 3:6], estimate[3:6], atol=1e-5, rtol=0.0):
        raise AssertionError("velocity feature ordering changed")
    if not np.allclose(raw[0, 15:30], expected_diag, atol=1e-5, rtol=0.0):
        raise AssertionError("diagonal EKF covariance feature ordering changed")
    if not np.allclose(raw[0, 30:33], goal - estimate[:3], atol=1e-5, rtol=0.0):
        raise AssertionError("goal delta feature ordering changed")
    if not np.allclose(raw[0, 39:42], direction, atol=1e-5, rtol=0.0):
        raise AssertionError("obstacle direction feature ordering changed")
    contract = verify_actor_observation_contract(env, observations)
    if not np.isfinite(raw).all() or not np.isfinite(central).all():
        raise AssertionError("schema gate received non-finite features")
    return {
        "passed": True,
        "environment": "StrictSingleUAVPPOEnv",
        "env_steps": 0,
        "actual_actor_node_shape": list(raw.shape),
        "actual_central_shape": [int(central.size)],
        "actor_encoder_input_shape": [1, EXPECTED_NODE_FEATURE_DIM],
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "graph_schema": {
            "node_features": [1, EXPECTED_NODE_FEATURE_DIM],
            "central_state_vector": [EXPECTED_CENTRAL_FEATURE_DIM],
            "edge_feature_dim": 11,
            "max_obstacles": int(getattr(base, "max_obstacles", 8)),
        },
        "finite": True,
        "contract": contract,
    }


def _rng_hash(environment: Any) -> str:
    state = getattr(getattr(environment, "rng", None), "bit_generator", None)
    payload = getattr(state, "state", repr(state))
    return _canonical_hash(payload)


def _policy_hashes(policy: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, parameter in policy.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        hashes[str(name)] = digest.hexdigest()
    return hashes


def _execution_record(info: Mapping[str, Any], uid: str, sampled: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    execution = info.get("action_execution", {})
    if isinstance(execution, Mapping):
        item = execution.get(uid, {})
    else:
        item = {}
    if not isinstance(item, Mapping):
        item = {}
    applied = np.asarray(item.get("delta_v_applied", sampled), dtype=np.float64).reshape(3)
    return applied, dict(item)


def _row_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"))


def _write_rows_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError("cannot write an empty calibration log")
    fields = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ValueError("calibration CSV schema drift")
            writer.writerow({key: (_row_json(value) if isinstance(value, (Mapping, list, tuple, np.ndarray)) else value) for key, value in row.items()})


def collect_calibration(
    base: Any,
    manifest: Mapping[str, Any],
    source_path: str | Path,
    old_statistics: Sequence[Any],
    data_root: str | Path,
    log_root: str | Path,
) -> dict[str, Any]:
    """Collect the deterministic 24-episode, 2880-step raw observation bank.

    The collector constructs one fresh control trainer per arm, loads only the
    approved A source, then freezes every policy parameter before deterministic
    collection.  It never adds a transition or performs an optimizer step.
    """

    _assert_split_manifest(manifest)
    calibration = list(manifest["calibration"])
    if len(calibration) != 12:
        raise ValueError("calibration manifest must contain 12 cases")
    if len(old_statistics) < 4:
        raise ValueError("old_statistics must be (normalizer, central, potential, provenance)")
    if not isinstance(source_path, (str, Path)) or not str(source_path):
        raise ValueError("source_path must identify the frozen A source")
    data_dir = Path(data_root)
    log_dir = Path(log_root)
    bank_path = data_dir / "calibration_bank.npz"
    completion_path = log_dir / "calibration_completion.json"
    step_log_path = log_dir / "calibration_steps.csv"
    if bank_path.exists() or completion_path.exists() or step_log_path.exists():
        raise FileExistsError("calibration outputs already exist; refusing to overwrite")

    gate = schema_gate(base, calibration[0])
    old_normalizer, old_central = old_statistics[0], old_statistics[1]
    node_rows: list[np.ndarray] = []
    central_rows: list[np.ndarray] = []
    metadata: dict[str, list[Any]] = {key: [] for key in ("episode_id", "scenario_id", "arm", "source", "split", "kind", "noise_seed", "step")}
    step_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    uid = "uav_00"
    for arm in ("P", "G"):
        # The action logger is intentionally pointed to a unique unused path;
        # calibration does not call reconcile_action_execution and therefore
        # emits no sampled-execution CSV.
        env = create_single_uav_ppo_environment(base)
        action_log_path = log_dir / f".calibration_{arm}_unused_action.csv"
        trainer = create_single_uav_ppo_trainer(
            env,
            base,
            normalizer=old_normalizer,
            central_normalizer=old_central,
            normalizer_provenance=old_statistics[3],
            action_log_path=action_log_path,
        )
        initialization = initialize_a_foundation(trainer, source_path, arm)
        if len(trainer.rollout) != 0 or len(getattr(trainer.optimizer, "state", {})) != 0:
            raise AssertionError("fresh A control trainer must have empty rollout and optimizer state")
        for parameter in trainer.policy.parameters():
            parameter.requires_grad_(False)
        trainer.policy.eval()
        frozen_parameter_hashes = _policy_hashes(trainer.policy)
        if not all(not parameter.requires_grad for parameter in trainer.policy.parameters()):
            raise AssertionError("calibration control policy was not frozen")
        for scene in calibration:
            scenario = _scene_scenario(scene)
            episode_id = f"calibration_train_{arm}_{scene['scenario_id']}"
            scenario["episode_id"] = episode_id
            observations, info = env.reset(seed=int(scene["noise_seed"]), scenario_config=scenario)
            pre_hash = _rng_hash(env)
            tracker_hold = 0
            reached = False
            episode_terminated = False
            episode_truncated = False
            episode_steps = 0
            for step in range(CALIBRATION_STEPS_PER_EPISODE + 1):
                raw_obs = np.asarray(observations[uid]["node_features"], dtype=np.float64)
                if raw_obs.shape != (1, EXPECTED_NODE_FEATURE_DIM) or not np.isfinite(raw_obs).all():
                    raise AssertionError("calibration raw node observation schema drift")
                central = _central(info, base)
                # Preserve every reset/post-step observation in the bank.
                node_rows.append(raw_obs[0].copy())
                central_rows.append(central.copy())
                for key, value in (
                    ("episode_id", episode_id),
                    ("scenario_id", scene["scenario_id"]),
                    ("arm", arm),
                    ("source", arm),
                    ("split", "calibration"),
                    ("kind", "normal"),
                    ("noise_seed", int(scene["noise_seed"])),
                    ("step", step),
                ):
                    metadata[key].append(value)
                if step == CALIBRATION_STEPS_PER_EPISODE:
                    break
                pre_position = np.asarray(env.states[uid].position_m, dtype=np.float64).copy()
                pre_velocity = np.asarray(env.states[uid].velocity_mps, dtype=np.float64).copy()
                pre_distance = float(np.linalg.norm(np.asarray(env.goals[uid]) - pre_position))
                pre_speed = float(np.linalg.norm(pre_velocity))
                decision = trainer.act(observations, info.get("central_state"), deterministic=True)
                if uid not in decision.actions or uid not in decision.samples:
                    raise AssertionError("deterministic A policy decision lacks uav_00 action/sample")
                sampled = np.asarray(decision.samples[uid].delta_v, dtype=np.float64).reshape(3)
                action = decision.actions
                normalized_decision_node = np.asarray(decision.observations[uid]["node_features"], dtype=np.float64).reshape(-1)
                before_rng = _rng_hash(env)
                next_observations, rewards, terminated, truncated, next_info = env.step(action)
                after_rng = _rng_hash(env)
                applied, execution = _execution_record(next_info, uid, sampled)
                post_position = np.asarray(env.states[uid].position_m, dtype=np.float64).copy()
                post_velocity = np.asarray(env.states[uid].velocity_mps, dtype=np.float64).copy()
                post_distance = float(next_info.get("strict_goal_distance_m_actual", np.linalg.norm(np.asarray(env.goals[uid]) - post_position)))
                post_speed = float(next_info.get("strict_speed_mps_actual", np.linalg.norm(post_velocity)))
                qualifies = bool(post_distance <= 1.0 and post_speed <= 0.5 and not next_info.get("collision", False))
                tracker_hold = tracker_hold + 1 if qualifies else 0
                reached = bool(reached or tracker_hold >= 5)
                step_rows.append(
                    {
                        "arm": arm,
                        "phase": "calibration_train",
                        "split": "calibration",
                        "kind": "normal",
                        "scenario_id": scene["scenario_id"],
                        "episode_id": episode_id,
                        "step": step + 1,
                        "pre_truth_position": pre_position.tolist(),
                        "pre_truth_velocity": pre_velocity.tolist(),
                        "post_truth_position": post_position.tolist(),
                        "post_truth_velocity": post_velocity.tolist(),
                        "actual_goal_distance": post_distance,
                        "actual_speed": post_speed,
                        "parking_reporter_hold": tracker_hold,
                        "parking_reporter_reached": reached,
                        "raw_node": raw_obs[0].tolist(),
                        "normalized_node": normalized_decision_node.tolist(),
                        "raw_central": central.tolist(),
                        "normalized_central": np.asarray(old_central.transform(central)).reshape(-1).tolist(),
                        "sample_delta_v": sampled.tolist(),
                        "action_applied_delta_v": applied.tolist(),
                        "correction_delta_v": (applied - sampled).tolist(),
                        "safety_projected": bool(execution.get("safety_projected", False)),
                        "boundary_any_axis": bool(execution.get("boundary_projected", False)),
                        "pre_rng_hash": before_rng,
                        "post_rng_hash": after_rng,
                        "noise_seed": int(scene["noise_seed"]),
                        "reward": float(np.asarray(list(rewards.values()), dtype=float).mean()) if isinstance(rewards, Mapping) and rewards else 0.0,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                episode_steps += 1
                episode_terminated = bool(episode_terminated or terminated)
                episode_truncated = bool(truncated)
                collision = bool(next_info.get("collision", False) or next_info.get("physical_collision", False))
                if collision or terminated or (truncated and step + 1 < CALIBRATION_STEPS_PER_EPISODE):
                    _write_rows_exclusive(step_log_path, step_rows)
                    reason = "collision" if collision else "terminated" if terminated else "truncated_before_120"
                    raise RuntimeError(f"calibration episode failed ({reason}): {episode_id} step={step + 1}")
                observations, info = next_observations, next_info
                pre_hash = after_rng
            if episode_steps != CALIBRATION_STEPS_PER_EPISODE:
                _write_rows_exclusive(step_log_path, step_rows)
                raise AssertionError(f"calibration episode step count is {episode_steps}, expected 120")
            if _policy_hashes(trainer.policy) != frozen_parameter_hashes:
                raise AssertionError("frozen A control policy parameters changed during calibration")
            if len(trainer.rollout) != 0 or len(getattr(trainer.optimizer, "state", {})) != 0:
                raise AssertionError("calibration must not populate PPO rollout or optimizer state")
            episodes.append(
                {
                    "episode_id": episode_id,
                    "scenario_id": scene["scenario_id"],
                    "arm": arm,
                    "split": "calibration",
                    "steps": CALIBRATION_STEPS_PER_EPISODE,
                    "observations": CALIBRATION_STEPS_PER_EPISODE + 1,
                    "terminated": episode_terminated,
                    "truncated_at_120": episode_truncated,
                    "rng_final_hash": pre_hash,
                    "source_initialization": initialization,
                }
            )

    if len(node_rows) != CALIBRATION_OBSERVATIONS or len(central_rows) != CALIBRATION_OBSERVATIONS:
        _write_rows_exclusive(step_log_path, step_rows)
        raise AssertionError("calibration bank count mismatch")
    _write_rows_exclusive(step_log_path, step_rows)
    nodes = np.asarray(node_rows, dtype=np.float64)
    central = np.asarray(central_rows, dtype=np.float64)
    if not np.isfinite(nodes).all() or not np.isfinite(central).all():
        raise FloatingPointError("calibration bank contains non-finite observations")
    if bank_path.exists():
        raise FileExistsError(bank_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        bank_path,
        node_features=nodes,
        central_features=central,
        raw_node_features=nodes,
        raw_central_features=central,
        episode_id=np.asarray(metadata["episode_id"], dtype="U128"),
        scenario_id=np.asarray(metadata["scenario_id"], dtype="U128"),
        arm=np.asarray(metadata["arm"], dtype="U2"),
        source=np.asarray(metadata["source"], dtype="U2"),
        split=np.asarray(metadata["split"], dtype="U32"),
        kind=np.asarray(metadata["kind"], dtype="U16"),
        noise_seed=np.asarray(metadata["noise_seed"], dtype=np.int64),
        step=np.asarray(metadata["step"], dtype=np.int64),
    )
    completion = {
        "passed": True,
        "schema_gate": gate,
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": _sha256(source_path) if Path(source_path).is_file() else None,
        "source_arms": ["P", "G"],
        "teacher_data_count": 0,
        "optimizer_steps": 0,
        "rollout_length": 0,
        "env_steps": CALIBRATION_ENV_STEPS,
        "observations": CALIBRATION_OBSERVATIONS,
        "episodes": len(episodes),
        "all_complete": True,
        "fit_split": "train",
        "bank_path": str(bank_path.resolve()),
        "episodes_detail": episodes,
    }
    _write_json_exclusive(completion_path, completion)
    return {
        "passed": True,
        "schema_gate": gate,
        "bank_path": str(bank_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "step_log_path": str(step_log_path.resolve()),
        "episodes": len(episodes),
        "env_steps": CALIBRATION_ENV_STEPS,
        "observations": CALIBRATION_OBSERVATIONS,
        "teacher_data_count": 0,
        "optimizer_steps": 0,
        "rollout_length": 0,
    }


def _load_bank(bank_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = Path(bank_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        node_key = "node_features" if "node_features" in archive else "raw_node_features"
        central_key = "central_features" if "central_features" in archive else "raw_central_features"
        if node_key not in archive or central_key not in archive:
            raise ValueError("calibration bank must contain node_features and central_features")
        nodes = np.asarray(archive[node_key], dtype=np.float64)
        central = np.asarray(archive[central_key], dtype=np.float64)
        metadata = {key: np.asarray(archive[key]) for key in archive.files if key not in {node_key, central_key, "raw_node_features", "raw_central_features"}}
    if nodes.shape != (CALIBRATION_OBSERVATIONS, EXPECTED_NODE_FEATURE_DIM):
        raise ValueError(f"calibration node bank shape must be (2904,42), got {nodes.shape}")
    if central.shape != (CALIBRATION_OBSERVATIONS, EXPECTED_CENTRAL_FEATURE_DIM):
        raise ValueError(f"calibration central bank shape must be (2904,174), got {central.shape}")
    if not np.isfinite(nodes).all() or not np.isfinite(central).all():
        raise ValueError("calibration bank contains non-finite values")
    required = {"episode_id", "scenario_id", "arm", "source", "split", "kind", "noise_seed", "step"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"calibration bank metadata missing: {missing}")
    for key in required:
        if metadata[key].reshape(-1).size != CALIBRATION_OBSERVATIONS:
            raise ValueError(f"calibration metadata {key} must have 2904 rows")
    split = metadata["split"].astype(str).reshape(-1)
    kind = metadata["kind"].astype(str).reshape(-1)
    arm = metadata["arm"].astype(str).reshape(-1)
    source = metadata["source"].astype(str).reshape(-1)
    episode_id = metadata["episode_id"].astype(str).reshape(-1)
    scenario_id = metadata["scenario_id"].astype(str).reshape(-1)
    steps = metadata["step"].astype(np.int64).reshape(-1)
    if not np.all(split == "calibration") or not np.all(kind == "normal"):
        raise ValueError("calibration bank contains non-calibration or non-normal metadata rows")
    if set(arm) != {"P", "G"}:
        raise ValueError("calibration bank must contain exactly P and G source arms")
    if not np.array_equal(arm, source):
        raise ValueError("calibration bank arm/source metadata disagree")
    unique_episodes = sorted(set(episode_id))
    if len(unique_episodes) != CALIBRATION_EPISODES:
        raise ValueError("calibration bank must contain exactly 24 episodes")
    if len(set(scenario_id)) != 12:
        raise ValueError("calibration bank must contain exactly 12 scenario IDs")
    for episode in unique_episodes:
        mask = episode_id == episode
        if int(mask.sum()) != CALIBRATION_STEPS_PER_EPISODE + 1:
            raise ValueError(f"episode {episode} must contain reset plus 120 observations")
        if not np.array_equal(np.sort(steps[mask]), np.arange(CALIBRATION_STEPS_PER_EPISODE + 1)):
            raise ValueError(f"episode {episode} must contain steps 0..120 exactly once")
        if len(set(arm[mask])) != 1 or len(set(scenario_id[mask])) != 1:
            raise ValueError(f"episode {episode} has inconsistent arm/scenario metadata")
    for scenario in sorted(set(scenario_id)):
        mask = scenario_id == scenario
        if int(mask.sum()) != 2 * (CALIBRATION_STEPS_PER_EPISODE + 1):
            raise ValueError(f"scenario {scenario} must have one complete P and one complete G episode")
        if set(arm[mask]) != {"P", "G"}:
            raise ValueError(f"scenario {scenario} is missing P/G source arm")
    return nodes, central, metadata


def _stats_for_matrix(matrix: np.ndarray, *, signed_slices: Mapping[str, tuple[slice, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    std = matrix.std(axis=0, dtype=np.float64)
    centered_maxabs = np.max(np.abs(matrix - mean), axis=0)
    ptp = np.ptp(matrix, axis=0)
    scales = np.maximum(std, centered_maxabs / 8.0)
    scales = np.where(ptp < _EPS, np.maximum(scales, 1.0), scales)
    scales = np.maximum(scales, _EPS)
    signed_summary: dict[str, Any] = {}
    for name, (feature_slice, floor) in signed_slices.items():
        values = matrix[:, feature_slice]
        # Signed motion/goal vectors are centered at the physical origin.  A
        # per-axis empirical center would rotate or translate their meaning
        # when a direction is absent from a small calibration bank.
        mean[feature_slice] = 0.0
        scalar = max(float(np.sqrt(np.mean(values * values))), float(floor), float(np.max(np.abs(values))) / 8.0, _EPS)
        scales[feature_slice] = scalar
        signed_summary[name] = {
            "slice": [int(feature_slice.start), int(feature_slice.stop)],
            "mean": mean[feature_slice].tolist(),
            "signed_rms": float(np.sqrt(np.mean(values * values))),
            "max_abs": float(np.max(np.abs(values))),
            "scale": float(scalar),
        }
    return mean, std, scales, signed_summary


def _feature_summary(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray, scale: np.ndarray, signed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": int(matrix.shape[0]),
        "mean": mean.tolist(),
        "std": np.asarray(std, dtype=np.float64).tolist(),
        "min": np.min(matrix, axis=0).tolist(),
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
        "max": np.max(matrix, axis=0).tolist(),
        "scale": scale.tolist(),
        "signed_groups": copy.deepcopy(dict(signed)),
    }


def _normalizer_states(nodes: np.ndarray, central: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    node_mean, node_std, node_scale, node_signed = _stats_for_matrix(
        nodes,
        signed_slices={"velocity": (slice(3, 6), 6.0), "goal_delta": (slice(30, 33), 2.4)},
    )
    central_mean, central_std, central_scale, central_signed = _stats_for_matrix(
        central,
        signed_slices={"velocity": (slice(3, 6), 6.0), "goal_delta": (slice(30, 33), 2.4)},
    )
    # The actual normalizer classes provide the canonical state schema.  Their
    # fitted arrays are replaced by the explicit A contract above, preserving
    # float64 full-dataset means/scales and integer sample counts.
    node_normalizer = ObservationNormalizer(eps=_EPS, clip=_CLIP).fit({"uav_00": {"node_features": nodes}})
    node_state = node_normalizer.normalizers["node_features"]
    node_state.mean = node_mean
    node_state.scale = node_scale
    node_state.count = np.full(EXPECTED_NODE_FEATURE_DIM, nodes.shape[0], dtype=np.int64)
    central_normalizer = FeatureNormalizer(eps=_EPS, clip=_CLIP).fit(central)
    central_normalizer.mean = central_mean
    central_normalizer.scale = central_scale
    central_normalizer.count = np.full(EXPECTED_CENTRAL_FEATURE_DIM, central.shape[0], dtype=np.int64)
    return node_normalizer.state_dict(), central_normalizer.state_dict(), {
        "node": _feature_summary(nodes, node_mean, node_std, node_scale, node_signed),
        "central": _feature_summary(central, central_mean, central_std, central_scale, central_signed),
        "signed_scales": {
            "node": node_signed,
            "central": central_signed,
        },
    }


def _write_distribution_csv(
    path: Path,
    nodes: np.ndarray,
    central: np.ndarray,
    summaries: Mapping[str, Any],
    metadata: Mapping[str, np.ndarray],
    old_states: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    rows: list[dict[str, Any]] = []
    steps = np.asarray(metadata["step"], dtype=np.int64).reshape(-1)
    bins: tuple[tuple[str, np.ndarray], ...] = (
        ("reset_0", steps == 0),
        ("steps_1_16", (steps >= 1) & (steps <= 16)),
        ("steps_17_40", (steps >= 17) & (steps <= 40)),
        ("steps_41_80", (steps >= 41) & (steps <= 80)),
        ("steps_81_120", (steps >= 81) & (steps <= 120)),
    )

    def append_condition(condition: str, group: str, matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> None:
        for time_bin, mask in bins:
            values = matrix[mask]
            if values.size == 0:
                continue
            normalized = np.clip((values - mean) / scale, -_CLIP, _CLIP)
            for index in range(matrix.shape[1]):
                rows.append(
                    {
                        "condition": condition,
                        "feature_group": group,
                        "time_bin": time_bin,
                        "feature_index": index,
                        "count": int(values.shape[0]),
                        "mean": float(np.mean(values[:, index])),
                        "std": float(np.std(values[:, index])),
                        "min": float(np.min(values[:, index])),
                        "q01": float(np.quantile(values[:, index], 0.01)),
                        "q50": float(np.quantile(values[:, index], 0.50)),
                        "q99": float(np.quantile(values[:, index], 0.99)),
                        "max": float(np.max(values[:, index])),
                        "signed_rms": float(np.sqrt(np.mean(values[:, index] ** 2))),
                        "max_abs": float(np.max(np.abs(values[:, index]))),
                        "scale": float(scale[index]),
                        "normalized_max_abs": float(np.max(np.abs(normalized[:, index]))),
                        "clip_plus_flag": bool(np.any(normalized[:, index] >= _CLIP)),
                        "clip_minus_flag": bool(np.any(normalized[:, index] <= -_CLIP)),
                        "zero_scale_flag": bool(scale[index] <= _EPS),
                    }
                )

    append_condition(
        "new_full",
        "node",
        nodes,
        np.asarray(summaries["node"]["mean"], dtype=float),
        np.asarray(summaries["node"]["scale"], dtype=float),
    )
    append_condition(
        "new_full",
        "central",
        central,
        np.asarray(summaries["central"]["mean"], dtype=float),
        np.asarray(summaries["central"]["scale"], dtype=float),
    )
    if old_states:
        append_condition(
            "old_short",
            "node",
            nodes,
            np.asarray(old_states["node"]["mean"], dtype=float),
            np.asarray(old_states["node"]["scale"], dtype=float),
        )
        append_condition(
            "old_short",
            "central",
            central,
            np.asarray(old_states["central"]["mean"], dtype=float),
            np.asarray(old_states["central"]["scale"], dtype=float),
        )
    _write_rows_exclusive(path, rows)


def fit_full_statistics(
    bank_path: str | Path,
    potential_statistics: Mapping[str, Any],
    artifact_path: str | Path,
    log_root: str | Path,
) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, Any], dict[str, Any]]:
    """Fit frozen full statistics from the calibration bank only."""

    if not isinstance(potential_statistics, Mapping):
        raise ValueError("potential_statistics must be a mapping")
    potential = copy.deepcopy(dict(potential_statistics))
    nodes, central, metadata = _load_bank(bank_path)
    observation_state, central_state, summaries = _normalizer_states(nodes, central)
    target = Path(artifact_path)
    if target.exists():
        raise FileExistsError(target)
    source_hash = _sha256(bank_path)
    old_states: dict[str, Any] | None = None
    old_statistics_state: dict[str, Any] | None = None
    old_statistics_source: str | None = None
    try:
        from .single_uav_ppo_a_support import SHARED_STATISTICS_DEFAULT, load_shared_a_statistics

        old_artifact = Path(SHARED_STATISTICS_DEFAULT).expanduser().resolve()
        old_norm, old_central, _old_potential, old_provenance = load_shared_a_statistics(old_artifact)
        old_node = old_norm.state_dict()["normalizers"]["node_features"]
        old_central_state = old_central.state_dict()
        old_states = {"node": old_node, "central": old_central_state}
        old_statistics_state = {
            "observation_normalizer": old_norm.state_dict(),
            "central_normalizer": old_central_state,
            "provenance": old_provenance,
        }
        old_statistics_source = str(old_artifact)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
        # Standalone unit tests may supply a synthetic bank without the frozen
        # project artifact.  New statistics remain fully valid; the optional
        # old-short diagnostic is emitted only when the locked state exists.
        old_states = None
    artifact = {
        "observation_normalizer": observation_state,
        "central_normalizer": central_state,
        "potential_statistics": copy.deepcopy(potential),
        "fit_split": "train",
        "frozen_during_ppo_and_validation": True,
        "new_calibration_steps": CALIBRATION_ENV_STEPS,
        "count": CALIBRATION_OBSERVATIONS,
        "node_count": CALIBRATION_OBSERVATIONS,
        "central_count": CALIBRATION_OBSERVATIONS,
        "source_rawbank_sha256": source_hash,
        "source_bank_sha256": source_hash,
        "clip": _CLIP,
        "eps": _EPS,
        "statistics_summary": summaries,
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
        "central_feature_dim": EXPECTED_CENTRAL_FEATURE_DIM,
        "calibration_observation_count": CALIBRATION_OBSERVATIONS,
    }
    if old_statistics_state is not None:
        artifact["old_statistics_state"] = old_statistics_state
        artifact["old_statistics_source"] = old_statistics_source
    _write_json_exclusive(target, artifact)
    distribution_path = Path(log_root) / "feature_distribution.csv"
    _write_distribution_csv(distribution_path, nodes, central, summaries, metadata, old_states)
    artifact_sha = _sha256(target)
    provenance = {
        "artifact_path": str(target.resolve()),
        "source_sha256": artifact_sha,
        "fit_split": "train",
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
        "central_feature_dim": EXPECTED_CENTRAL_FEATURE_DIM,
        "count": CALIBRATION_OBSERVATIONS,
        "node_count": CALIBRATION_OBSERVATIONS,
        "central_count": CALIBRATION_OBSERVATIONS,
        "new_calibration_steps": CALIBRATION_ENV_STEPS,
        "source_rawbank_sha256": source_hash,
        "statistics_frozen": True,
        "refit_performed": False,
        "normalizer_state_sha256": _canonical_hash(observation_state),
        "central_normalizer_state_sha256": _canonical_hash(central_state),
    }
    normalizer = ObservationNormalizer.from_state_dict(observation_state)
    central_normalizer = FeatureNormalizer.from_state_dict(central_state)
    return normalizer, central_normalizer, potential, provenance


class _FrozenObservationNormalizer(ObservationNormalizer):
    def fit(self, *_args: Any, **_kwargs: Any) -> "_FrozenObservationNormalizer":
        raise RuntimeError("full calibration statistics are frozen; refit is forbidden")

    update = fit


class _FrozenFeatureNormalizer(FeatureNormalizer):
    def fit(self, *_args: Any, **_kwargs: Any) -> "_FrozenFeatureNormalizer":
        raise RuntimeError("full calibration central statistics are frozen; refit is forbidden")

    update = fit


def load_full_statistics(path: str | Path) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, Any], dict[str, Any]]:
    """Load and verify the full artifact as a read-only statistics tuple."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    with artifact_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("full statistics artifact must be a JSON object")
    if payload.get("fit_split") != "train" or payload.get("frozen_during_ppo_and_validation") is not True:
        raise ValueError("full statistics artifact must be frozen train statistics")
    if int(payload.get("count", 0)) != CALIBRATION_OBSERVATIONS:
        raise ValueError("full statistics artifact count must be 2904")
    if int(payload.get("new_calibration_steps", -1)) != CALIBRATION_ENV_STEPS:
        raise ValueError("full statistics artifact must record 2880 calibration steps")
    observation_state = payload.get("observation_normalizer")
    central_state = payload.get("central_normalizer")
    if not isinstance(observation_state, Mapping) or not isinstance(central_state, Mapping):
        raise ValueError("full statistics artifact is missing normalizer states")
    observation = ObservationNormalizer.from_state_dict(observation_state)
    central = FeatureNormalizer.from_state_dict(central_state)
    node = observation.normalizers.get("node_features")
    if node is None or node.feature_dim != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError("full observation normalizer must contain 42 node features")
    if central.feature_dim != EXPECTED_CENTRAL_FEATURE_DIM:
        raise ValueError("full central normalizer must contain 174 features")
    for name, values in (("node mean", node.mean), ("node scale", node.scale), ("node count", node.count), ("central mean", central.mean), ("central scale", central.scale), ("central count", central.count)):
        if values is None or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"{name} is non-finite")
    if np.any(np.asarray(node.scale) <= 0.0) or np.any(np.asarray(central.scale) <= 0.0):
        raise ValueError("full normalizer scales must be positive")
    if not np.all(np.asarray(node.count) == CALIBRATION_OBSERVATIONS) or not np.all(np.asarray(central.count) == CALIBRATION_OBSERVATIONS):
        raise ValueError("full normalizer counts must all equal 2904")
    if dict(payload.get("feature_index_map", {})) != FEATURE_INDEX_MAP:
        raise ValueError("full artifact feature_index_map does not match the 42-feature contract")
    potential = payload.get("potential_statistics")
    if not isinstance(potential, Mapping):
        raise ValueError("full artifact is missing potential_statistics")
    artifact_sha = _sha256(artifact_path)
    provenance = {
        "artifact_path": str(artifact_path),
        "source_sha256": artifact_sha,
        "fit_split": "train",
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
        "central_feature_dim": EXPECTED_CENTRAL_FEATURE_DIM,
        "count": CALIBRATION_OBSERVATIONS,
        "node_count": CALIBRATION_OBSERVATIONS,
        "central_count": CALIBRATION_OBSERVATIONS,
        "new_calibration_steps": CALIBRATION_ENV_STEPS,
        "source_rawbank_sha256": payload.get("source_rawbank_sha256"),
        "statistics_frozen": True,
        "refit_performed": False,
        "normalizer_state_sha256": _canonical_hash(observation_state),
        "central_normalizer_state_sha256": _canonical_hash(central_state),
    }
    frozen_observation = _FrozenObservationNormalizer(eps=observation.eps, clip=observation.clip)
    frozen_observation.normalizers = copy.deepcopy(observation.normalizers)
    frozen_central = _FrozenFeatureNormalizer(eps=central.eps, clip=central.clip)
    frozen_central.mean = central.mean.copy() if central.mean is not None else None
    frozen_central.scale = central.scale.copy() if central.scale is not None else None
    frozen_central.count = central.count.copy() if central.count is not None else None
    return frozen_observation, frozen_central, copy.deepcopy(dict(potential)), provenance


__all__ = [
    "build_split_manifest",
    "schema_gate",
    "collect_calibration",
    "fit_full_statistics",
    "load_full_statistics",
    "EXPECTED_NODE_FEATURE_DIM",
    "EXPECTED_CENTRAL_FEATURE_DIM",
    "CALIBRATION_OBSERVATIONS",
    "CALIBRATION_ENV_STEPS",
]
