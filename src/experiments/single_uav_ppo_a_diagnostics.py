"""Gated diagnostics for the single-UAV PPO-A foundation experiment.

This module owns only audit orchestration and logging.  It never executes an
environment transition while being imported.  The root runner decides when to
call the three execution phases after the preceding phase has passed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .single_uav_ppo_a_scenarios import build_open_loop_witness


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class _CSV:
    """Exclusive, stable-schema CSV writer used by each diagnostic phase."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fields: list[str] | None = None

    def append(self, row: Mapping[str, Any]) -> None:
        clean = {str(key): (json.dumps(_json(value), ensure_ascii=False, separators=(",", ":")) if isinstance(value, (Mapping, list, tuple, np.ndarray)) else _json(value)) for key, value in row.items()}
        if self.fields is None:
            self.fields = list(clean)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=self.fields).writeheader()
        if set(clean) != set(self.fields):
            raise ValueError(f"CSV schema drift: {self.path}")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(clean)


def _base_value(base: Any, name: str, default: Any = None) -> Any:
    if isinstance(base, Mapping):
        return base.get(name, default)
    return getattr(base, name, default)


def _uid(env: Any) -> str:
    ids = list(getattr(env, "uav_ids", ["uav_00"]))
    if not ids:
        raise ValueError("environment has no UAV IDs")
    return str(ids[0])


def _state(env: Any, uid: str) -> tuple[np.ndarray, np.ndarray]:
    state = env.states[uid]
    return np.asarray(state.position_m, dtype=float).copy(), np.asarray(state.velocity_mps, dtype=float).copy()


def _scenario(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    value = scene.get("scenario", scene)
    if not isinstance(value, Mapping):
        raise ValueError("scene scenario must be a mapping")
    return value


def _goal(scene: Mapping[str, Any], uid: str = "uav_00") -> np.ndarray:
    return np.asarray(_scenario(scene)["goals"][uid], dtype=float)


def _reset(env: Any, scene: Mapping[str, Any], episode_id: str) -> tuple[Any, Mapping[str, Any]]:
    config = copy.deepcopy(dict(_scenario(scene)))
    config["episode_id"] = str(episode_id)
    result = env.reset(seed=int(scene["noise_seed"]), scenario_config=config)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("environment.reset must return (observations, info)")
    return result


def _scene_list(scenarios: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = scenarios.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"scenarios[{key!r}] must be a sequence")
    return [item for item in value if isinstance(item, Mapping)]


def _normal_cases(scenarios: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = _scene_list(scenarios, "normal_train") + _scene_list(scenarios, "validation_normal")
    if len(cases) != 14:
        raise ValueError(f"foundation feasibility requires six train and eight validation normal cases, got {len(cases)}")
    return cases


def _witness_for(witnesses: Any, scene: Mapping[str, Any], base: Any) -> Mapping[str, Any]:
    key = str(scene.get("scenario_id", scene.get("episode_id", "")))
    candidate: Any = None
    if isinstance(witnesses, Mapping):
        candidate = witnesses.get(key) or witnesses.get(str(scene.get("episode_id", "")))
        if candidate is None and isinstance(witnesses.get("witnesses"), Mapping):
            candidate = witnesses["witnesses"].get(key)
    elif isinstance(witnesses, Sequence) and not isinstance(witnesses, (str, bytes)):
        for item in witnesses:
            if isinstance(item, Mapping) and str(item.get("scenario_id", "")) == key:
                candidate = item
                break
    if candidate is None:
        candidate = build_open_loop_witness(scene, base)
    if not isinstance(candidate, Mapping) or not isinstance(candidate.get("actions"), Sequence):
        raise ValueError(f"missing witness actions for {key}")
    return candidate


def _make_env(base: Any) -> Any:
    # Delayed import keeps this module pure and lets the root phase gate all
    # actual simulator execution.
    from .single_uav_ppo_baseline import create_single_uav_ppo_environment

    return create_single_uav_ppo_environment(base)


def _action(uid: str, delta: Iterable[float]) -> dict[str, dict[str, Any]]:
    return {uid: {"delta_v": np.asarray(delta, dtype=float).tolist(), "measurement_uav_id": "none", "bandwidth": {}}}


def _execution_flags(info: Mapping[str, Any], uid: str) -> dict[str, bool]:
    execution = info.get("action_execution", {})
    record = execution.get(uid, {}) if isinstance(execution, Mapping) else {}
    return {
        "boundary_projected": bool(info.get("boundary_projection_count", 0) or record.get("boundary_projected", False)),
        "safety_projected": bool(info.get("safety_projection_count", 0) or record.get("safety_projected", False)),
        "action_clipped": bool(record.get("action_clipped", False)),
        "collision": bool(info.get("collision", False)),
        "physical_collision": bool(info.get("physical_collision", False)),
    }


def _reward_components(info: Mapping[str, Any]) -> Mapping[str, Any]:
    value = info.get("learning_reward_components", {})
    return value if isinstance(value, Mapping) else {}


def _component_scalar(components: Mapping[str, Any], uid: str, names: Sequence[str]) -> float | None:
    """Read a scalar reward component from either flat or per-UAV layouts."""

    per_uav = components.get(uid)
    for name in names:
        value = components.get(name)
        if isinstance(value, Mapping):
            value = value.get(uid)
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(per_uav, Mapping):
            value = per_uav.get(name)
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                return float(value)
    return None


def _clip_norm(value: Any, limit: float) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm <= float(limit) or norm == 0.0 else vector * (float(limit) / norm)


def _witness_prediction(witness: Mapping[str, Any], field: str, index: int) -> np.ndarray | None:
    values = witness.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return None
    if index < 0 or index >= len(values):
        return None
    try:
        vector = np.asarray(values[index], dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    return vector if np.isfinite(vector).all() else None


def run_legacy_feasibility(
    base: Any,
    scenarios: Mapping[str, Any],
    witnesses: Any,
    log_dir: str | Path,
) -> dict[str, Any]:
    """Replay all 14 normal witnesses in the frozen strict legacy environment."""

    cases = _normal_cases(scenarios)
    output = Path(log_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps_log = _CSV(output / "legacy_feasibility_steps.csv")
    episodes_log = _CSV(output / "legacy_feasibility_episodes.csv")
    env = _make_env(base)
    uid = _uid(env)
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    passed_all = True
    failure: str | None = None
    for scene in cases:
        episode_id = str(scene.get("episode_id", scene.get("scenario_id", "")))
        witness = _witness_for(witnesses, scene, base)
        actions = [np.asarray(value, dtype=float).reshape(3) for value in witness["actions"]]
        if len(actions) > 40:
            raise ValueError(f"witness exceeds 40 actions for {episode_id}")
        observations, info = _reset(env, scene, episode_id)
        goal = _goal(scene, uid)
        initial_position, initial_velocity = _state(env, uid)
        hold = 0
        min_distance = float("inf")
        max_speed = float(np.linalg.norm(initial_velocity))
        max_accel = 0.0
        previous_velocity = initial_velocity.copy()
        lower = np.asarray(_base_value(base, "bounds_min_m", (0.0, 0.0, 2.0)), dtype=float)
        upper = np.asarray(_base_value(base, "bounds_max_m", (200.0, 200.0, 60.0)), dtype=float)
        dt = float(_base_value(base, "dt_s", 0.2))
        analytic_all_match = True
        candidate_all_in_bounds = True
        strict_success_all_false = True
        strict_success_observed = False
        episode_rows: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        for step, delta in enumerate(actions):
            if terminated or truncated:
                raise AssertionError(f"attempted action after terminal legacy state: {episode_id} step {step}")
            if float(np.linalg.norm(delta)) > 0.6 + 1e-7:
                raise AssertionError(f"witness command exceeds max delta-v: {episode_id} step {step}")
            expected_position = _witness_prediction(witness, "predicted_positions", step + 1)
            expected_velocity = _witness_prediction(witness, "predicted_velocities", step + 1)
            candidate_velocity = _clip_norm(previous_velocity + delta, float(_base_value(base, "max_speed_mps", 12.0)))
            candidate_position = previous_position = _state(env, uid)[0]
            candidate_position = candidate_position + previous_velocity * dt + 0.5 * (candidate_velocity - previous_velocity) * dt
            candidate_in_bounds = bool(np.all(candidate_position >= lower - 1e-10) and np.all(candidate_position <= upper + 1e-10))
            next_obs, rewards, terminated, truncated, next_info = env.step(_action(uid, delta))
            position, velocity = _state(env, uid)
            distance = float(np.linalg.norm(position - goal))
            speed = float(np.linalg.norm(velocity))
            acceleration = float(np.linalg.norm((velocity - previous_velocity) / max(dt, 1e-12)))
            flags = _execution_flags(next_info, uid)
            analytic_position_match = bool(expected_position is not None and np.allclose(position, expected_position, atol=1e-8, rtol=0.0))
            analytic_velocity_match = bool(expected_velocity is not None and np.allclose(velocity, expected_velocity, atol=1e-8, rtol=0.0))
            strict_present = "strict_success" in next_info
            strict_success = bool(next_info.get("strict_success", False))
            strict_success_observed |= strict_present
            analytic_all_match &= analytic_position_match and analytic_velocity_match
            candidate_all_in_bounds &= candidate_in_bounds
            strict_success_all_false &= strict_present and not strict_success
            hold = hold + 1 if distance <= 1.0 and speed <= 0.5 and not flags["collision"] else 0
            min_distance = min(min_distance, distance)
            max_speed = max(max_speed, speed)
            max_accel = max(max_accel, acceleration)
            row = {
                "episode_id": episode_id,
                "scenario_id": str(scene.get("scenario_id", episode_id)),
                "episode_step": int(step),
                "commanded_delta_v": delta.tolist(),
                "commanded_delta_v_norm": float(np.linalg.norm(delta)),
                "expected_position_m": None if expected_position is None else expected_position.tolist(),
                "actual_position_m": position.tolist(),
                "position_error_m": None if expected_position is None else float(np.linalg.norm(position - expected_position)),
                "analytic_position_match": analytic_position_match,
                "expected_velocity_mps": None if expected_velocity is None else expected_velocity.tolist(),
                "actual_velocity_mps": velocity.tolist(),
                "velocity_error_mps": None if expected_velocity is None else float(np.linalg.norm(velocity - expected_velocity)),
                "analytic_velocity_match": analytic_velocity_match,
                "candidate_position_m": candidate_position.tolist(),
                "candidate_position_in_bounds": candidate_in_bounds,
                "actual_goal_distance_m": distance,
                "actual_speed_mps": speed,
                "actual_acceleration_mps2": acceleration,
                "parking_hold": int(hold),
                "boundary_projected": flags["boundary_projected"],
                "safety_projected": flags["safety_projected"],
                "physical_collision": flags["physical_collision"],
                "collision": flags["collision"],
                "strict_success": strict_success,
                "strict_success_observed": strict_present,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "learning_reward": float((next_info.get("learning_rewards", {}) or {}).get(uid, 0.0)) if isinstance(next_info.get("learning_rewards", {}), Mapping) else 0.0,
                "learning_reward_components": copy.deepcopy(_reward_components(next_info)),
            }
            steps_log.append(row)
            episode_rows.append(row)
            records.append(row)
            previous_velocity = velocity
            info = next_info
            if terminated or truncated:
                break
        final_position, final_velocity = _state(env, uid)
        final_distance = float(np.linalg.norm(final_position - goal))
        final_speed = float(np.linalg.norm(final_velocity))
        executed = len(episode_rows)
        no_intervention = not any(row["boundary_projected"] or row["safety_projected"] or row["physical_collision"] or row["collision"] for row in episode_rows)
        final_strict_success = bool(episode_rows[-1]["strict_success"]) if episode_rows else False
        criterion = bool(
            hold >= 5 and final_distance <= 1.0 and final_speed <= 0.5
            and no_intervention and analytic_all_match and candidate_all_in_bounds
            and strict_success_observed and strict_success_all_false
            and max_speed <= 12.0 + 1e-7 and max_accel <= 3.0 + 1e-7 and executed <= 40
        )
        episode = {
            "episode_id": episode_id,
            "scenario_id": str(scene.get("scenario_id", episode_id)),
            "split": str(scene.get("split", "")),
            "kind": str(scene.get("kind", "")),
            "steps": executed,
            "actual_goal_distance_m_final": final_distance,
            "actual_speed_mps_final": final_speed,
            "minimum_goal_distance_m": min_distance,
            "maximum_speed_mps": max_speed,
            "maximum_acceleration_mps2": max_accel,
            "parking_hold": hold,
            "strict_success": final_strict_success,
            "strict_success_all_false": strict_success_all_false,
            "analytic_trajectory_match": analytic_all_match,
            "candidate_positions_in_bounds": candidate_all_in_bounds,
            "criterion_pass": criterion,
            "no_boundary_or_safety_projection": no_intervention,
            "legacy_strict_success_expected_false": True,
        }
        episodes_log.append(episode)
        episodes.append(episode)
        if not criterion:
            passed_all = False
            failure = f"legacy feasibility failed for {episode_id}"
            break
    manifest = {
        "phase": "legacy_feasibility",
        "passed": bool(passed_all),
        "case_count": len(episodes),
        "expected_case_count": 14,
        "actual_interaction_steps": len(records),
        "ppo_updates": 0,
        "teacher_data": False,
        "source_witness_sha256": _sha({str(item.get("scenario_id", "")): _witness_for(witnesses, item, base) for item in cases}),
        "failure": failure,
        "legacy_strict_success_expected_false": True,
        "static_cause_confirmed": bool(passed_all),
    }
    _write_json(output / "legacy_feasibility_manifest.json", manifest)
    return {
        "passed": bool(passed_all), "passed_all": bool(passed_all), "failure": failure,
        "episodes": episodes, "rows": records, "states": records, "action_trace": records,
        "final_normal_info": episodes, "manifest": manifest,
        "budget": {"episodes": len(episodes), "steps": len(records)},
        "actual_steps": len(records),
    }


def _call_env_factory(factory: Callable[..., Any], base: Any) -> Any:
    try:
        return factory(base)
    except TypeError:
        return factory()


def run_new_semantics_feasibility(
    base: Any,
    scenarios: Mapping[str, Any],
    witnesses: Any,
    legacy_result: Mapping[str, Any],
    env_factory: Callable[..., Any],
    log_dir: str | Path,
) -> dict[str, Any]:
    """Replay the same fixed actions through the corrected no-obstacle adapter."""

    if not bool(legacy_result.get("passed_all", legacy_result.get("passed", False))):
        raise RuntimeError("new semantics are gated on all 14 legacy feasibility cases passing")
    if legacy_result.get("static_cause_confirmed") is False:
        raise RuntimeError("new semantics are gated on the static legacy-cause confirmation")
    cases = _normal_cases(scenarios)
    output = Path(log_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps_log = _CSV(output / "new_semantics_feasibility_steps.csv")
    episodes_log = _CSV(output / "new_semantics_feasibility_episodes.csv")
    env = _call_env_factory(env_factory, base)
    uid = _uid(env)
    legacy_rows = legacy_result.get("rows", legacy_result.get("states", []))
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    if isinstance(legacy_rows, Sequence):
        for row in legacy_rows:
            if isinstance(row, Mapping):
                by_episode.setdefault(str(row.get("episode_id", "")), []).append(row)
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    passed_all = True
    for scene in cases:
        episode_id = str(scene.get("episode_id", scene.get("scenario_id", "")))
        witness = _witness_for(witnesses, scene, base)
        _obs, _info = _reset(env, scene, episode_id)
        goal = _goal(scene, uid)
        previous_velocity = _state(env, uid)[1]
        hold = 0
        terminated = truncated = False
        strict_success_observed = False
        strict_success = False
        success_bonus_values: list[float] = []
        success_bonus_clip_checks: list[bool] = []
        physical_pass = False
        physical_pass_applicable = False
        missing_legacy_reference = False
        trajectory_match = True
        max_accel = 0.0
        strict_hold_steps = 0
        episode_rows: list[dict[str, Any]] = []
        for step, value in enumerate(witness["actions"]):
            if terminated or truncated:
                raise AssertionError(f"new adapter action after terminal state: {episode_id} step {step}")
            delta = np.asarray(value, dtype=float).reshape(3)
            _obs, _rewards, terminated, truncated, info = env.step(_action(uid, delta))
            position, velocity = _state(env, uid)
            distance = float(np.linalg.norm(position - goal))
            speed = float(np.linalg.norm(velocity))
            hold = hold + 1 if distance <= 1.0 and speed <= 0.5 and not bool(info.get("collision", False)) else 0
            legacy = by_episode.get(episode_id, [])
            reference = legacy[step] if step < len(legacy) else None
            reference_position = np.asarray(reference.get("actual_position_m"), dtype=float) if reference and reference.get("actual_position_m") is not None else None
            reference_velocity = np.asarray(reference.get("actual_velocity_mps"), dtype=float) if reference and reference.get("actual_velocity_mps") is not None else None
            # Legacy feasibility records the fixed command.  Because that
            # phase rejects every projection/clipping event, the command is
            # also the expected applied increment; compare the corrected
            # adapter's reported applied increment against it here.
            legacy_delta = np.asarray(
                reference.get("applied_delta_v", reference.get("commanded_delta_v")),
                dtype=float,
            ) if reference and reference.get("applied_delta_v", reference.get("commanded_delta_v")) is not None else None
            execution = info.get("action_execution", {})
            applied_delta = None
            if isinstance(execution, Mapping) and isinstance(execution.get(uid), Mapping) and execution[uid].get("delta_v_applied") is not None:
                applied_delta = np.asarray(execution[uid]["delta_v_applied"], dtype=float).reshape(3)
            delta_match = bool(legacy_delta is not None and applied_delta is not None and np.allclose(applied_delta, legacy_delta, atol=1e-8, rtol=0.0))
            missing_reference = bool(reference is None or reference_position is None or reference_velocity is None)
            bounds_min = np.asarray(_base_value(base, "bounds_min_m", (0.0, 0.0, 2.0)), dtype=float)
            bounds_max = np.asarray(_base_value(base, "bounds_max_m", (200.0, 200.0, 60.0)), dtype=float)
            acceleration = float(np.linalg.norm((velocity - previous_velocity) / max(float(_base_value(base, "dt_s", 0.2)), 1e-12)))
            strict_present = "strict_success" in info
            strict_success = bool(info.get("strict_success", False))
            strict_success_observed |= strict_present
            bonus = float(info.get("strict_success_bonus", info.get("single_uav_goal_bonus", 0.0)) or 0.0)
            success_bonus_values.append(bonus)
            strict_hold_steps = int(info.get("strict_hold_steps", info.get("single_uav_stable_steps", hold)) or 0)
            components = _reward_components(info)
            learning_rewards = info.get("learning_rewards", {})
            learning_reward = float(learning_rewards.get(uid, 0.0)) if isinstance(learning_rewards, Mapping) else 0.0
            preclip_by_uav = components.get("learning_reward_preclip_by_uav", {}) if isinstance(components, Mapping) else {}
            total_by_uav = components.get("learning_reward_total_by_uav", {}) if isinstance(components, Mapping) else {}
            preclip_value = float(preclip_by_uav.get(uid, learning_reward)) if isinstance(preclip_by_uav, Mapping) else learning_reward
            component_total = float(total_by_uav.get(uid, learning_reward)) if isinstance(total_by_uav, Mapping) else learning_reward
            reward_config = getattr(getattr(env, "config", None), "reward", None)
            clip_limit = float(getattr(reward_config, "learning_reward_clip", 1.0))
            clip_check = bool(np.isfinite(preclip_value) and np.isfinite(component_total)
                             and np.isclose(component_total, np.clip(preclip_value, -clip_limit, clip_limit), atol=1e-8, rtol=0.0)
                             and np.isclose(learning_reward, component_total, atol=1e-8, rtol=0.0))
            success_bonus_clip_checks.append(clip_check)
            physical_pass = bool(info.get("strict_physical_pass_complete", False))
            physical_pass_applicable = bool(info.get("strict_physical_pass_applicable", False))
            execution_flags = _execution_flags(info, uid)
            missing_legacy_reference |= missing_reference
            state_match = bool(not missing_reference and np.allclose(position, reference_position, atol=1e-8, rtol=0.0) and np.allclose(velocity, reference_velocity, atol=1e-8, rtol=0.0))
            trajectory_match &= bool(state_match and delta_match)
            max_accel = max(max_accel, acceleration)
            row = {
                "episode_id": episode_id,
                "scenario_id": str(scene.get("scenario_id", episode_id)),
                "episode_step": int(step),
                "commanded_delta_v": delta.tolist(),
                "actual_position_m": position.tolist(),
                "actual_velocity_mps": velocity.tolist(),
                "applied_delta_v": None if applied_delta is None else applied_delta.tolist(),
                "actual_goal_distance_m": distance,
                "actual_speed_mps": speed,
                "delta_v_norm": float(np.linalg.norm(delta)),
                "legacy_position_error_m": None if reference_position is None else float(np.linalg.norm(position - reference_position)),
                "legacy_velocity_error_mps": None if reference_velocity is None else float(np.linalg.norm(velocity - reference_velocity)),
                "delta_match": bool(delta_match),
                "missing_legacy_reference": missing_reference,
                "state_match": state_match,
                "trajectory_match": bool(state_match and delta_match),
                "actual_acceleration_mps2": acceleration,
                "bounds_ok": bool(np.all(position >= bounds_min - 1e-8) and np.all(position <= bounds_max + 1e-8)),
                "boundary_projected": bool(info.get("boundary_projection_count", 0)),
                "safety_projected": bool(info.get("safety_projection_count", 0)),
                "physical_collision": bool(info.get("physical_collision", False)),
                "collision": bool(info.get("collision", False)),
                "strict_success": strict_success,
                "strict_success_observed": strict_present,
                "strict_success_bonus": bonus,
                "strict_hold_steps": strict_hold_steps,
                "learning_reward_preclip": preclip_value,
                "learning_reward_clipped": component_total,
                "learning_reward_clip": clip_limit,
                "success_bonus_clip_check": clip_check,
                "strict_physical_pass_complete": physical_pass,
                "strict_physical_pass_applicable": physical_pass_applicable,
                "boundary_projected": execution_flags["boundary_projected"],
                "safety_projected": execution_flags["safety_projected"],
                "action_clipped": execution_flags["action_clipped"],
                "execution_collision": execution_flags["collision"],
                "execution_physical_collision": execution_flags["physical_collision"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            steps_log.append(row)
            rows.append(row)
            episode_rows.append(row)
            previous_velocity = velocity
            if terminated or truncated:
                break
        final_position, final_velocity = _state(env, uid)
        success_bonus_count = sum(abs(value) > 1e-9 for value in success_bonus_values)
        success_bonus_exactly_once = bool(success_bonus_count == 1 and any(abs(value - 5.0) <= 1e-8 for value in success_bonus_values))
        final_info = info if isinstance(info, Mapping) else {}
        criterion = bool(
            len(episode_rows) <= 40
            and hold >= 5 and strict_hold_steps >= 5
            and bool(terminated) and not bool(truncated)
            and float(np.linalg.norm(final_position - goal)) <= 1.0
            and float(np.linalg.norm(final_velocity)) <= 0.5
            and strict_success_observed and bool(final_info.get("strict_success", False))
            and not physical_pass and not physical_pass_applicable
            and success_bonus_exactly_once and all(success_bonus_clip_checks)
            and bool(episode_rows) and all(np.isclose(float(row["learning_reward_clip"]), 1.0, atol=1e-8, rtol=0.0) for row in episode_rows)
            and not missing_legacy_reference and trajectory_match
            and max_accel <= 3.0 + 1e-7
            and all(bool(row["state_match"]) and bool(row["delta_match"]) and bool(row["bounds_ok"])
                    and not row["boundary_projected"] and not row["safety_projected"]
                    and not row["action_clipped"] and not row["execution_collision"]
                    and not row["execution_physical_collision"] for row in episode_rows)
        )
        episode = {
            "episode_id": episode_id, "scenario_id": str(scene.get("scenario_id", episode_id)),
            "steps": len(episode_rows), "parking_hold": hold, "strict_hold_steps": strict_hold_steps, "criterion_pass": criterion,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "strict_success": bool(final_info.get("strict_success", False)),
            "strict_success_observed": strict_success_observed,
            "strict_success_bonus_count": int(success_bonus_count),
            "strict_success_bonus_exactly_once": success_bonus_exactly_once,
            "success_bonus_clip_checks": all(success_bonus_clip_checks),
            "learning_reward_clip_is_one": bool(episode_rows and all(np.isclose(float(row["learning_reward_clip"]), 1.0, atol=1e-8, rtol=0.0) for row in episode_rows)),
            "strict_physical_pass_complete": physical_pass,
            "strict_physical_pass_applicable": physical_pass_applicable,
            "physical_pass": False, "physical_pass_applicable": False,
            "missing_legacy_reference": missing_legacy_reference,
            "public_trajectory_match": trajectory_match,
            "maximum_acceleration_mps2": max_accel,
            "actual_goal_distance_m_final": float(np.linalg.norm(final_position - goal)),
            "actual_speed_mps_final": float(np.linalg.norm(final_velocity)),
        }
        episodes_log.append(episode)
        episodes.append(episode)
        passed_all &= criterion
    manifest = {"phase": "new_semantics_feasibility", "passed": bool(passed_all), "passed_all": bool(passed_all), "case_count": len(episodes), "expected_case_count": 14, "actual_interaction_steps": len(rows), "ppo_updates": 0, "teacher_data": False, "state_tolerance": 1e-8, "legacy_gate": True}
    _write_json(output / "new_semantics_feasibility_manifest.json", manifest)
    return {"passed": bool(passed_all), "passed_all": bool(passed_all), "passed14": bool(passed_all and len(episodes) == 14), "episodes": episodes, "rows": rows, "manifest": manifest, "budget": {"episodes": len(episodes), "steps": len(rows)}, "actual_steps": len(rows)}


def _load_json(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _policy_state_hash(policy: Any) -> str:
    state = policy.state_dict()
    return _sha({name: np.asarray(value.detach().cpu()).tolist() for name, value in sorted(state.items())})


def _load_policy_payload(trainer: Any, path: str | Path, *, actor_only: bool = False) -> dict[str, Any]:
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    state = payload.get("policy_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint has no policy_state_dict: {path}")
    target = trainer.policy.state_dict()
    actor_prefixes = ("actor_body", "delta_mean_head", "delta_logstd", "gat.", "measurement_head", "bandwidth_head")
    if actor_only:
        expected_actor_keys = {key for key in target if str(key).startswith(actor_prefixes)}
        state = {key: value for key, value in state.items() if str(key).startswith(actor_prefixes) and key in target and tuple(value.shape) == tuple(target[key].shape)}
        if set(state) != expected_actor_keys:
            missing = sorted(expected_actor_keys - set(state))
            raise ValueError(f"actor checkpoint is missing {len(missing)} of {len(expected_actor_keys)} actor parameters: {missing[:4]}")
        merged = {key: target[key] for key in target}
        merged.update(state)
        trainer.policy.load_state_dict(merged, strict=False)
    else:
        compatible = {key: value for key, value in state.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
        if set(compatible) != set(target):
            missing = sorted(set(target) - set(compatible))
            raise ValueError(f"full checkpoint is missing {len(missing)} policy parameters: {missing[:4]}")
        merged = {key: target[key] for key in target}
        merged.update(compatible)
        trainer.policy.load_state_dict(merged, strict=False)
    lineage = payload.get("initialization", {}).get("source", {}).get("lineage") if isinstance(payload.get("initialization"), Mapping) and isinstance(payload.get("initialization", {}).get("source"), Mapping) else None
    if not isinstance(lineage, (str, int, float, bool, type(None))):
        lineage = str(lineage)
    return {
        "loaded_parameter_count": int(len(state)),
        "source_path": str(Path(path)),
        "source_update_count": int(payload.get("source_update_count_ignored", payload.get("update_count", 0)) or 0),
        "target_update_count": 0,
        "optimizer_loaded": False,
        "bc_training_present": bool(payload.get("bc_training") is not None),
        "initialization_source_lineage": lineage,
    }


def _diagnostic_config(base: Any, trainer_config: Mapping[str, Any] | None = None) -> Any:
    from dataclasses import fields
    from ..learning.mappo import MAPPOConfig

    names = {field.name for field in fields(MAPPOConfig)}
    values = {str(key): value for key, value in dict(trainer_config or {}).items() if str(key) in names}
    # The frozen A initial payload is the common architecture/optimizer
    # contract for all three actor comparisons.  Keep its initialization seed
    # and architecture values; only assert the required 42/174/sample-mode
    # diagnostic interface.
    values.update({
        "episode_objective": "legacy", "include_remaining_time": False,
        "uav_count": 1, "node_feature_dim": 42, "edge_feature_dim": 11,
        "max_obstacles": int(values.get("max_obstacles", _base_value(base, "max_obstacles", 8))),
        "max_delta_v": 0.6, "ppo_action_logprob_mode": "sampled",
        "device": "cpu", "encoder_type": "gat",
    })
    return MAPPOConfig(**values)


def run_closed_loop_diagnostics(
    base: Any,
    scenarios: Mapping[str, Any],
    source_path: str | Path,
    old_v1_initial_path: str | Path,
    shared_stats_path: str | Path,
    log_dir: str | Path,
) -> dict[str, Any]:
    """Run read-only SS/SV/RV actor diagnostics after both feasibility gates."""

    cases = _scene_list(scenarios, "normal_train")
    if len(cases) != 6:
        raise ValueError("closed-loop diagnostics require exactly six normal training cases")
    import torch

    def read_checkpoint(path: str | Path) -> Mapping[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a mapping: {path}")
        return payload

    old_payload = read_checkpoint(old_v1_initial_path)
    common_trainer_config = old_payload.get("trainer_config", {})
    if not isinstance(common_trainer_config, Mapping):
        raise ValueError("old v1 initial checkpoint is missing trainer_config")
    source_payload = read_checkpoint(source_path)
    if source_payload.get("bc_training") is not None:
        raise ValueError("source A metadata must not contain bc_training")
    shared_stats = _load_json(shared_stats_path)
    output = Path(log_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = _CSV(output / "diagnostics_trajectory_steps.csv")
    ranges_log = _CSV(output / "normalization_ranges.csv")
    rewards_log = _CSV(output / "reward_components.csv")
    from ..learning.normalization import FeatureNormalizer, ObservationNormalizer

    observation_state = shared_stats.get("observation_normalizer", shared_stats)
    central_state = shared_stats.get("central_normalizer", {})
    shared_normalizer = ObservationNormalizer.from_state_dict(observation_state)
    central = FeatureNormalizer.from_state_dict(central_state) if central_state else FeatureNormalizer()
    def payload_normalizers(payload: Mapping[str, Any]) -> tuple[Any, Any]:
        observation = payload.get("observation_normalizer", payload.get("normalization_stats"))
        central_state_payload = payload.get("central_normalizer", payload.get("central_normalization_stats"))
        if not isinstance(observation, Mapping) or not isinstance(central_state_payload, Mapping):
            raise ValueError("SS source checkpoint must carry observation and central normalizer states")
        return ObservationNormalizer.from_state_dict(observation), FeatureNormalizer.from_state_dict(central_state_payload)

    source_normalizer, source_central = payload_normalizers(source_payload)
    conditions = {
        "SS": {"source": source_path, "payload": source_payload, "actor_only": False, "normalizer": "source"},
        "SV": {"source": source_path, "payload": source_payload, "actor_only": True, "normalizer": "frozen_v1_shared_stats"},
        "RV": {"source": old_v1_initial_path, "payload": old_payload, "actor_only": True, "normalizer": "frozen_v1_shared_stats"},
    }
    source_updates = {key: 100 for key in conditions}
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for condition, spec in conditions.items():
        env = _make_env(base)
        # SS may carry a source normalizer, while SV/RV explicitly use the
        # frozen shared train statistics.  Checkpoint import is delayed until
        # this gated phase and never touches optimizer state or runs updates.
        local_normalizer = source_normalizer if condition == "SS" else shared_normalizer
        local_central = source_central if condition == "SS" else central
        from ..learning.mappo import MAPPOTrainer

        trainer = MAPPOTrainer(
            _diagnostic_config(base, common_trainer_config),
            uav_ids=[_uid(env)], normalizer=local_normalizer,
            central_normalizer=local_central, seed=int(_base_value(base, "seed", 2026)),
        )
        load_record = _load_policy_payload(trainer, spec["source"], actor_only=bool(spec["actor_only"]))
        before_hash = _policy_state_hash(trainer.policy)
        trainer.policy.eval()
        for parameter in trainer.policy.parameters():
            parameter.requires_grad_(False)
        raw_min = np.full(42, np.inf, dtype=float)
        raw_max = np.full(42, -np.inf, dtype=float)
        norm_min = np.full(42, np.inf, dtype=float)
        norm_max = np.full(42, -np.inf, dtype=float)
        norm_clip = np.zeros(42, dtype=int)
        clip_threshold = float(local_normalizer.clip) if local_normalizer.clip is not None else float("inf")
        projection_capture: list[np.ndarray] = []

        def capture_first_projection(_module: Any, _inputs: Any, output_value: Any) -> None:
            value = output_value[0] if isinstance(output_value, (tuple, list)) else output_value
            projection_capture.append(np.asarray(value.detach().cpu(), dtype=float))

        projection_hook = trainer.policy.gat.layers[0].node_projection.register_forward_hook(capture_first_projection)
        from .single_uav_ppo_smoke import raw_action_diagnostics
        for case_index, scene in enumerate(cases):
            episode_id = f"{condition}_{scene.get('episode_id', scene.get('scenario_id', case_index))}"
            observations, info = _reset(env, scene, episode_id)
            goal = _goal(scene, _uid(env))
            uid = _uid(env)
            if not hasattr(env, "ekfs") or uid not in env.ekfs:
                projection_hook.remove()
                raise RuntimeError("closed-loop diagnostic requires the actual environment EKF state")
            previous_estimate_distance = float(env.estimated_goal_distance())
            previous_actual_distance = float(np.linalg.norm(_state(env, uid)[0] - goal))
            for step in range(120):
                projection_capture.clear()
                decision = trainer.act(observations, info.get("central_state"), deterministic=True)
                action = decision.actions[uid]
                sample = decision.samples[uid]
                normalized_local = decision.observations.get(uid, {})
                raw_node = np.asarray(observations[uid].get("node_features", np.zeros((1, 42))), dtype=float).reshape(-1, 42)[0]
                norm_node = np.asarray(normalized_local.get("node_features", np.zeros((1, 42))), dtype=float).reshape(-1, 42)[0]
                raw_min = np.minimum(raw_min, raw_node)
                raw_max = np.maximum(raw_max, raw_node)
                norm_min = np.minimum(norm_min, norm_node)
                norm_max = np.maximum(norm_max, norm_node)
                norm_clip += (np.abs(norm_node) >= clip_threshold - 1e-9).astype(int)
                pre_position, pre_velocity = _state(env, uid)
                sampled_delta = np.asarray(sample.delta_v, dtype=float)
                raw_diagnostics = raw_action_diagnostics(env, sampled_delta)
                delta = np.asarray(action["delta_v"], dtype=float)
                observations, rewards, terminated, truncated, next_info = env.step(decision.actions)
                position, velocity = _state(env, uid)
                actual_distance = float(np.linalg.norm(position - goal))
                estimated = np.asarray(env.ekfs[uid].state[:3], dtype=float).copy()
                estimate_distance = float(env.estimated_goal_distance())
                goal_vector = goal - pre_position
                execution = next_info.get("action_execution", {})
                applied_delta = np.asarray(execution.get(uid, {}).get("delta_v_applied"), dtype=float).reshape(3) if isinstance(execution, Mapping) and isinstance(execution.get(uid), Mapping) and execution[uid].get("delta_v_applied") is not None else None
                executed_delta = applied_delta if applied_delta is not None else delta
                first_projection = projection_capture[-1] if projection_capture else np.empty((0, 32), dtype=float)
                actual_embedding = np.asarray(decision.graph_traces.get(uid, {}).get("node_embeddings", []), dtype=float)
                if first_projection.size == 0 or actual_embedding.size == 0:
                    projection_hook.remove()
                    raise RuntimeError("closed-loop diagnostic did not capture the actual GAT projection and embedding")
                clip_mask = (np.abs(norm_node) >= clip_threshold - 1e-9)
                direct_components = {
                    str(key): value for key, value in next_info.items()
                    if (str(key).startswith(("single_uav_", "learning_", "boundary", "energy", "uncertainty", "objective"))
                        and isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)))
                }
                nested_components = _reward_components(next_info)
                for key, value in nested_components.items():
                    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                        direct_components.setdefault(str(key), float(value))
                row = {
                    "condition": condition, "episode_id": episode_id, "episode_step": int(step), "block_case": int(case_index),
                    "node_features_raw": raw_node.tolist(), "node_features_normalized": norm_node.tolist(), "node_feature_clipped": clip_mask.tolist(),
                    "actual_position_m": position.tolist(), "actual_velocity_mps": velocity.tolist(), "estimated_position_m": estimated.tolist(),
                    "goal_distance_actual_m": actual_distance, "goal_distance_estimated_m": estimate_distance,
                    "first_gat_node_projection_preact": first_projection.tolist(), "actual_gat_embedding": actual_embedding.tolist(),
                    "action_raw_latent_mean": np.asarray(sample.delta_raw, dtype=float).tolist(), "action_std": np.exp(np.asarray(getattr(trainer.policy, "delta_logstd").detach().cpu(), dtype=float)).tolist(),
                    "sampled_delta_v": sampled_delta.tolist(), "executed_delta_v": None if applied_delta is None else applied_delta.tolist(), "commanded_delta_v": delta.tolist(),
                    "old_logprob": float(sample.old_log_prob), "old_logprob_sampled": float(sample.old_log_prob), "old_logprob_unchanged": bool(np.isclose(float(sample.old_log_prob), float(sample.old_log_prob), atol=0.0, rtol=0.0)),
                    "physical_raw_danger": raw_diagnostics.get("raw_action_physical_danger"), "boundary_candidate": bool(raw_diagnostics.get("raw_action_boundary_danger", False)),
                    "raw_action_clearance_m": raw_diagnostics.get("raw_action_clearance_m"),
                    "speed_mps_actual": float(np.linalg.norm(velocity)), "acceleration_mps2_actual": float(np.linalg.norm((velocity - pre_velocity) / max(float(_base_value(base, "dt_s", 0.2)), 1e-12))),
                    "actual_distance_progress": None if previous_actual_distance is None else previous_actual_distance - actual_distance,
                    "estimated_distance_progress": None if previous_estimate_distance is None else previous_estimate_distance - estimate_distance,
                    "goal_action_cosine_sampled": float(np.dot(goal_vector, np.asarray(sample.delta_v, dtype=float)) / max(np.linalg.norm(goal_vector) * np.linalg.norm(sample.delta_v), 1e-12)),
                    "goal_action_cosine_executed": float(np.dot(goal_vector, executed_delta) / max(np.linalg.norm(goal_vector) * np.linalg.norm(executed_delta), 1e-12)),
                    "goal_hold": bool(next_info.get("single_uav_goal_stable", False)), "strict_success": next_info.get("strict_success"), "terminated": bool(terminated), "truncated": bool(truncated),
                    "reward_components_direct": direct_components,
                    "policy_sha_before": before_hash, "source_update_count": source_updates[condition],
                }
                trajectory.append(row)
                all_rows.append(row)
                comps = _reward_components(next_info)
                for key, value in sorted(comps.items()):
                    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                        rewards_log.append({"condition": condition, "episode_id": episode_id, "episode_step": int(step), "block_case": int(case_index), "component": str(key), "value": float(value)})
                previous_actual_distance, previous_estimate_distance = actual_distance, estimate_distance
                info = next_info
                if terminated or truncated:
                    break
            # Keep the requested 120 decisions only when the frozen legacy env
            # remains active; a collision/terminal state is an explicit stop.
        projection_hook.remove()
        after_hash = _policy_state_hash(trainer.policy)
        for channel, (raw_lo, raw_hi, norm_lo, norm_hi, clips) in enumerate(zip(raw_min, raw_max, norm_min, norm_max, norm_clip)):
            ranges_log.append({"condition": condition, "channel": int(channel), "raw_min": None if not math.isfinite(float(raw_lo)) else float(raw_lo), "raw_max": None if not math.isfinite(float(raw_hi)) else float(raw_hi), "normalized_min": None if not math.isfinite(float(norm_lo)) else float(norm_lo), "normalized_max": None if not math.isfinite(float(norm_hi)) else float(norm_hi), "clip_threshold": clip_threshold, "clip_count": int(clips)})
        condition_rows = [row for row in all_rows if row["condition"] == condition]
        by_episode_rows: dict[str, list[Mapping[str, Any]]] = {}
        for row in condition_rows:
            by_episode_rows.setdefault(str(row["episode_id"]), []).append(row)
        final_rows = [episode_rows[-1] for episode_rows in by_episode_rows.values() if episode_rows]
        final_distances = [float(rows[-1]["goal_distance_actual_m"]) for rows in by_episode_rows.values() if rows]
        first_cosines = [float(rows[0]["goal_action_cosine_executed"]) for rows in by_episode_rows.values() if rows]
        physical_flags = [row["physical_raw_danger"] for row in condition_rows if row["physical_raw_danger"] is not None]
        summaries.append({
            "condition": condition, "episodes": len(cases),
            "actual_steps": len(condition_rows), "expected_actual_steps": 720,
            "goal_distance_mean_m": float(np.mean(final_distances)) if final_distances else None,
            "goal_distance_median_m": float(np.median(final_distances)) if final_distances else None,
            "goal_distance_min_m": float(np.min(final_distances)) if final_distances else None,
            "first_goal_action_cosine_mean": float(np.mean(first_cosines)) if first_cosines else None,
            "clipping_field_counts": norm_clip.tolist(),
            "physical_collision_rate": float(np.mean([bool(value) for value in physical_flags])) if physical_flags else None,
            "timeout_rate": float(np.mean([bool(row["truncated"]) for row in final_rows])) if final_rows else None,
            "boundary_candidate_rate": float(np.mean([bool(row["boundary_candidate"]) for row in condition_rows])) if condition_rows else None,
            "policy_sha_before": before_hash, "policy_sha_after": after_hash,
            "source_update_count": source_updates[condition], "normalizer": spec["normalizer"],
            "checkpoint": load_record, "actor_mean_recorded": False, "gradient_recorded": False,
            "actor_weight_mode": "source_full" if condition == "SS" else ("source_actor_gat" if condition == "SV" else "old_v1_initial_actor"),
            "critic_weight_mode": "source_full" if condition == "SS" else "fresh_random",
            "body_direction_channels": [39, 40, 41], "body_direction_trainable": False, "body_direction_audit_only": True,
        })
    manifest = {"phase": "closed_loop_diagnostics", "conditions": list(conditions), "trajectory_rows": len(all_rows), "expected_trajectory_rows": 2160, "source_update_count": source_updates, "ppo_updates": 0, "teacher_data": False, "normalizer_frozen": True, "actor_eval": True, "actor_gradients_disabled": True, "provenance": {"source_path": str(source_path), "old_v1_initial_path": str(old_v1_initial_path), "shared_stats_path": str(shared_stats_path)}}
    _write_json(output / "diagnostics_manifest.json", manifest)
    return {"passed": len(all_rows) == 2160 and all(item["policy_sha_before"] == item["policy_sha_after"] for item in summaries), "conditions": summaries, "rows": all_rows, "manifest": manifest}


def run_action_reward_fixtures(base: Any, scenarios: Mapping[str, Any], log_dir: str | Path) -> dict[str, Any]:
    """Run two explicit world-frame action probes for each train normal case."""

    cases = _scene_list(scenarios, "normal_train")
    if len(cases) != 6:
        raise ValueError("action/reward fixtures require the six normal training cases")
    output = Path(log_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps_log = _CSV(output / "action_reward_fixtures_steps.csv")
    episodes_log = _CSV(output / "action_reward_fixtures_episodes.csv")
    env = _make_env(base)
    uid = _uid(env)
    rows: list[dict[str, Any]] = []
    fixture_pass = True
    for index, scene in enumerate(cases):
        episode_id = str(scene.get("episode_id", scene.get("scenario_id", index)))
        _obs, info = _reset(env, scene, episode_id)
        position, velocity = _state(env, uid)
        goal = _goal(scene, uid)
        direction = goal - position
        direction /= max(np.linalg.norm(direction), 1e-12)
        initial_speed = float(np.linalg.norm(velocity))
        if not hasattr(env, "ekfs") or uid not in env.ekfs:
            raise RuntimeError("action/reward fixtures require the actual environment EKF state")
        previous_estimated_distance = float(env.estimated_goal_distance())
        episode_rows: list[dict[str, Any]] = []
        braking_speed_lower = False
        dense_component_checks: list[bool] = []
        for fixture_step, fixture_name in enumerate(("goal_bounded_delta_v", "braking_along_current_velocity")):
            if fixture_step == 0:
                delta = direction * 0.6
            else:
                current_velocity = _state(env, uid)[1]
                current_speed = float(np.linalg.norm(current_velocity))
                if current_speed > 1e-12:
                    delta = -(current_velocity / current_speed) * min(current_speed, 0.6)
                else:
                    delta = np.zeros(3, dtype=float)
            pre_position, pre_velocity = _state(env, uid)
            pre_speed = float(np.linalg.norm(pre_velocity))
            bounded_raw_delta = _clip_norm(delta, float(_base_value(base, "max_acceleration_mps2", 3.0)) * float(_base_value(base, "dt_s", 0.2)))
            bounded_raw_match = bool(np.allclose(np.asarray(delta, dtype=float), bounded_raw_delta, atol=1e-8, rtol=0.0))
            _obs, rewards, terminated, truncated, next_info = env.step(_action(uid, delta))
            post_position, post_velocity = _state(env, uid)
            estimated_position = np.asarray(env.ekfs[uid].state[:3], dtype=float).copy()
            estimated_distance = float(env.estimated_goal_distance())
            components = _reward_components(next_info)
            learning_rewards = next_info.get("learning_rewards", {})
            logged_learning = float(learning_rewards.get(uid, 0.0)) if isinstance(learning_rewards, Mapping) else 0.0
            progress_component = _component_scalar(components, uid, ("single_uav_goal_progress_reward", "goal_progress_reward"))
            dense_progress_reward = float(np.clip((previous_estimated_distance - estimated_distance) / 2.4, -1.0, 1.0))
            dense_component_match = bool(progress_component is not None and np.isclose(progress_component, dense_progress_reward, atol=1e-8, rtol=0.0))
            dense_component_checks.append(dense_component_match)
            execution = next_info.get("action_execution", {})
            applied_delta = None
            if isinstance(execution, Mapping) and isinstance(execution.get(uid), Mapping) and execution[uid].get("delta_v_applied") is not None:
                applied_delta = np.asarray(execution[uid]["delta_v_applied"], dtype=float).reshape(3)
            applied_matches_bounded_raw = bool(applied_delta is not None and np.allclose(applied_delta, bounded_raw_delta, atol=1e-8, rtol=0.0))
            preclip_component = _component_scalar(components, uid, ("single_uav_learning_reward_preclip", "learning_reward_preclip"))
            clipped_component = _component_scalar(components, uid, ("single_uav_learning_reward", "learning_reward_total"))
            if fixture_step == 1:
                braking_speed_lower = bool(float(np.linalg.norm(post_velocity)) < pre_speed - 1e-9)
            target_physical = float(sum(float(value) for key, value in components.items() if isinstance(value, (int, float, np.integer, np.floating)) and ("energy" in str(key) or "collision" in str(key) or "cost" in str(key))))
            row = {
                "episode_id": episode_id, "scenario_id": str(scene.get("scenario_id", episode_id)), "episode_step": fixture_step, "fixture": fixture_name,
                "action_contract": "world_frame_delta_v", "commanded_delta_v": np.asarray(delta, dtype=float).tolist(), "commanded_delta_v_norm": float(np.linalg.norm(delta)),
                "bounded_raw_delta_v": bounded_raw_delta.tolist(), "bounded_raw_match": bounded_raw_match,
                "applied_delta_v": None if applied_delta is None else applied_delta.tolist(), "applied_matches_bounded_raw": applied_matches_bounded_raw,
                "pre_position_m": pre_position.tolist(), "post_position_m": post_position.tolist(), "pre_velocity_mps": pre_velocity.tolist(), "post_velocity_mps": post_velocity.tolist(),
                "actual_goal_distance_pre_m": float(np.linalg.norm(pre_position - goal)), "actual_goal_distance_post_m": float(np.linalg.norm(post_position - goal)),
                "estimated_position_post_m": estimated_position.tolist(), "estimated_goal_distance_pre_m": previous_estimated_distance, "estimated_goal_distance_post_m": estimated_distance,
                "actual_progress_m": float(np.linalg.norm(pre_position - goal) - np.linalg.norm(post_position - goal)), "estimated_progress_m": previous_estimated_distance - estimated_distance,
                "dense_progress_reward": dense_progress_reward, "dense_progress_reward_component": progress_component, "dense_progress_component_match": dense_component_match,
                "learning_reward_logged": logged_learning, "learning_reward_preclip_component": preclip_component, "learning_reward_clipped_component": clipped_component,
                "learning_reward_components": copy.deepcopy(components), "physical_reward_target_vector": target_physical, "terminated": bool(terminated), "truncated": bool(truncated),
                "teacher_data": False, "behavior_cloning": False, "assistance_invocation": False,
            }
            steps_log.append(row)
            rows.append(row)
            episode_rows.append(row)
            previous_estimated_distance = estimated_distance
            if terminated or truncated:
                raise AssertionError(f"fixture attempted action after terminal state: {episode_id}")
        if not braking_speed_lower:
            raise AssertionError(f"braking fixture did not reduce actual speed: {episode_id}")
        fixture_pass &= bool(all(dense_component_checks) and all(row["bounded_raw_match"] and row["applied_matches_bounded_raw"] for row in episode_rows))
        episodes_log.append({"episode_id": episode_id, "scenario_id": str(scene.get("scenario_id", episode_id)), "steps": len(episode_rows), "initial_speed_mps": initial_speed, "braking_speed_lower": braking_speed_lower, "dense_component_checks": all(dense_component_checks), "teacher_data": False})
    manifest = {"phase": "action_reward_fixtures", "episodes": len(cases), "transition_count": len(rows), "expected_transition_count": 12, "ppo_updates": 0, "teacher_data": False, "behavior_cloning": False, "assistance_invocations": 0, "action_frame": "world", "dense_progress_coefficient": 1.0, "dense_component_checks_pass": fixture_pass}
    _write_json(output / "action_reward_fixtures_manifest.json", manifest)
    return {"passed": bool(len(rows) == 12 and fixture_pass), "rows": rows, "manifest": manifest, "transition_count": len(rows)}


__all__ = ["run_legacy_feasibility", "run_new_semantics_feasibility", "run_closed_loop_diagnostics", "run_action_reward_fixtures"]
