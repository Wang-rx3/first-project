"""Paired, read-only diagnosis of the single-UAV terminal approach.

This module compares two already-trained checkpoints on exactly the same
normal far-distance test scenarios.  It deliberately does not call
``MAPPOTrainer.update`` and does not add transitions to a rollout.  The output
is intended to answer a narrower question than a new training run:

    where does the fine-tuned policy stop progressing from 8 m to 1 m?

Distance segments use an explicit half-open convention:

    8--5 m -> [5, 8)
    5--3 m -> [3, 5)
    3--1 m -> [1, 3)
    goal radius -> [0, 1)

The convention is recorded in the diagnostic manifest so a point at exactly
3 m is not silently assigned to two different mechanisms.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _episode_metrics,
    _run_episode,
    _scale_seed,
    _scenario_for,
    _write_attention_summary,
    _write_episode_metrics,
    _write_scale_manifest,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)


APPROACH_SEGMENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "8_5m",
        "label": "8-5 m",
        "lower_m": 5.0,
        "upper_m": 8.0,
        "next": "5_3m",
    },
    {
        "name": "5_3m",
        "label": "5-3 m",
        "lower_m": 3.0,
        "upper_m": 5.0,
        "next": "3_1m",
    },
    {
        "name": "3_1m",
        "label": "3-1 m",
        "lower_m": 1.0,
        "upper_m": 3.0,
        "next": "goal_radius",
    },
)

DISTANCE_BINS: tuple[dict[str, Any], ...] = (
    {"name": "1_2m", "label": "1-2 m", "lower_m": 1.0, "upper_m": 2.0},
    {"name": "2_3m", "label": "2-3 m", "lower_m": 2.0, "upper_m": 3.0},
    {"name": "3_3p5m", "label": "3-3.5 m", "lower_m": 3.0, "upper_m": 3.5},
    {"name": "3p5_5m", "label": "3.5-5 m", "lower_m": 3.5, "upper_m": 5.0},
    {"name": "5_8m", "label": "5-8 m", "lower_m": 5.0, "upper_m": 8.0},
)

POLICY_NAMES: tuple[str, ...] = ("baseline", "fine_tuned")
POLICY_LABELS = {
    "baseline": "coef=0.25 baseline",
    "fine_tuned": "navigation-preserving fine-tuned",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        columns = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                output[column] = value
            writer.writerow(output)


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite(values: Iterable[Any]) -> np.ndarray:
    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers.append(number)
    return np.asarray(numbers, dtype=float)


def _mean(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _max(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite(values)
    return float(np.max(numbers)) if numbers.size else float(default)


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def approach_segment_for_distance(distance_m: float) -> str:
    """Map a goal distance to the auditable approach segment contract."""

    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return "invalid"
    if not np.isfinite(distance):
        return "invalid"
    if 0.0 <= distance < 1.0:
        return "goal_radius"
    for segment in APPROACH_SEGMENTS:
        if float(segment["lower_m"]) <= distance < float(segment["upper_m"]):
            return str(segment["name"])
    if distance >= 8.0:
        return "outside_near_goal"
    return "below_zero"


def distance_bin_for_distance(distance_m: float) -> str:
    """Return the fine-grained bin used to inspect the 3 m transition."""

    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return "invalid"
    if not np.isfinite(distance):
        return "invalid"
    for distance_bin in DISTANCE_BINS:
        if float(distance_bin["lower_m"]) <= distance < float(distance_bin["upper_m"]):
            return str(distance_bin["name"])
    return "outside_diagnostic_range"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_payload(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = scenario.get("scenario")
    return nested if isinstance(nested, Mapping) else scenario


def _dt_from_scenario(scenario: Mapping[str, Any], default: float = 0.2) -> float:
    payload = _scenario_payload(scenario)
    contract = payload.get("motion_action_contract")
    if isinstance(contract, Mapping):
        value = _number(contract, "decision_interval_s", default)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return float(default)


def _timestamp_key(row: Mapping[str, Any]) -> float:
    return round(_number(row, "timestamp"), 6)


def _vector_from_row(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    return np.asarray([_number(row, name) for name in names], dtype=float)


def _goal_vector(scenario: Mapping[str, Any]) -> np.ndarray | None:
    payload = _scenario_payload(scenario)
    goals = payload.get("goals", {})
    if not isinstance(goals, Mapping) or "uav_00" not in goals:
        return None
    try:
        goal = np.asarray(goals["uav_00"], dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    return goal if np.isfinite(goal).all() else None


def _step_rows_from_episode(
    episode_path: Path,
    *,
    policy: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
    scenario: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join reward, action, and ground-truth rows into step-level diagnostics."""

    rewards = _read_csv(episode_path / "rewards.csv")
    actions = _read_csv(episode_path / "actions.csv")
    truth = _read_csv(episode_path / "ground_truth.csv")
    if not truth:
        truth = _read_csv(episode_path / "trajectory.csv")
    actions_by_time = {_timestamp_key(row): row for row in actions}
    truth_by_time = {_timestamp_key(row): row for row in truth}
    goal = _goal_vector(scenario)
    dt = _dt_from_scenario(scenario)
    rows: list[dict[str, Any]] = []
    previous_timestamp: float | None = None
    for index, reward in enumerate(rewards):
        timestamp = _number(reward, "timestamp", (index + 1) * dt)
        action = actions_by_time.get(round(timestamp, 6), {})
        truth_row = truth_by_time.get(round(timestamp, 6), {})
        current_distance = _number(reward, "current_goal_distance")
        previous_distance = _number(reward, "previous_goal_distance")
        delta_distance = _number(reward, "goal_progress_delta")
        if not np.isfinite(delta_distance) and np.isfinite(previous_distance) and np.isfinite(current_distance):
            delta_distance = previous_distance - current_distance
        step_dt = dt
        if previous_timestamp is not None and timestamp > previous_timestamp:
            step_dt = timestamp - previous_timestamp
        radial_progress = (
            delta_distance / step_dt
            if np.isfinite(delta_distance) and step_dt > 0.0
            else float("nan")
        )

        velocity = _vector_from_row(truth_row, ("vx", "vy", "vz"))
        radial_velocity = float("nan")
        if goal is not None and np.isfinite(velocity).all():
            position = _vector_from_row(truth_row, ("x", "y", "z"))
            direction = goal - position
            norm = float(np.linalg.norm(direction))
            if np.isfinite(norm) and norm > 1e-9:
                radial_velocity = float(np.dot(velocity, direction / norm))
        speed = _number(reward, "single_uav_speed_mps")
        if not np.isfinite(speed) and np.isfinite(velocity).all():
            speed = float(np.linalg.norm(velocity))
        target_speed = _number(reward, "single_uav_target_speed_mps")
        braking_signal = _number(reward, "single_uav_braking_action_signal")
        effective_signal = _number(reward, "single_uav_effective_braking_signal")
        action_braking = _number(action, "braking_action")
        action_effective = _number(action, "brake_effective")
        if not np.isfinite(braking_signal):
            braking_signal = action_braking
        if not np.isfinite(effective_signal):
            effective_signal = action_effective
        # The reward log stores a continuous braking *strength* in
        # ``single_uav_braking_action_signal``.  Fractions of braking actions
        # must use the action-log binary event when it is available; a
        # strength of 0.3 is still a braking action, not a non-braking step.
        braking_event = (
            int(action_braking > 0.5)
            if np.isfinite(action_braking)
            else int(np.isfinite(braking_signal) and braking_signal > 1e-9)
        )
        effective_event = (
            int(action_effective > 0.5)
            if np.isfinite(action_effective)
            else int(np.isfinite(effective_signal) and effective_signal > 1e-9)
        )

        row = {
            "policy": policy,
            "policy_label": POLICY_LABELS[policy],
            "episode_index": int(episode_index),
            "episode_id": str(episode_path.name),
            "seed": int(seed),
            "scenario_hash": scenario_hash,
            "step_index": int(index),
            "timestamp_s": float(timestamp),
            "goal_distance_m": current_distance,
            "previous_goal_distance_m": previous_distance,
            "delta_goal_distance_m": delta_distance,
            "radial_progress_mps": radial_progress,
            "radial_velocity_toward_mps": radial_velocity,
            "total_speed_mps": speed,
            "target_speed_mps": target_speed,
            "speed_excess_mps": _number(reward, "single_uav_speed_excess_mps"),
            "braking_action_signal": braking_signal,
            "effective_braking_signal": effective_signal,
            "braking_action": braking_event,
            "effective_braking": effective_event,
            "commanded_braking_delta_v_mps": _number(action, "commanded_braking_delta_v_mps"),
            "applied_braking_delta_v_mps": _number(action, "braking_delta_v_mps"),
            "pre_speed_mps": _number(action, "pre_speed_mps"),
            "post_speed_mps": _number(action, "post_speed_mps"),
            "actual_speed_reduction_when_braking_mps": (
                max(
                    0.0,
                    _number(action, "pre_speed_mps")
                    - _number(action, "post_speed_mps"),
                )
                if braking_event
                and np.isfinite(_number(action, "pre_speed_mps"))
                and np.isfinite(_number(action, "post_speed_mps"))
                else float("nan")
            ),
            "braking_progress_signal": _number(reward, "single_uav_braking_progress_signal"),
            "braking_progress_reward": _number(reward, "single_uav_braking_progress_reward"),
            "terminal_radial_progress_active": _number(
                reward, "single_uav_terminal_radial_progress_active", 0.0
            ),
            "terminal_radial_progress_m": _number(
                reward, "single_uav_terminal_radial_progress_m"
            ),
            "terminal_radial_progress_speed_excess_mps": _number(
                reward, "single_uav_terminal_radial_progress_speed_excess_mps"
            ),
            "terminal_radial_progress_speed_gate": _number(
                reward, "single_uav_terminal_radial_progress_speed_gate"
            ),
            "terminal_radial_progress_signal": _number(
                reward, "single_uav_terminal_radial_progress_signal"
            ),
            "terminal_radial_progress_reward": _number(
                reward, "single_uav_terminal_radial_progress_reward"
            ),
            "target_velocity_action_error": _number(reward, "single_uav_target_velocity_action_error"),
            "target_velocity_action_reward": _number(reward, "single_uav_target_velocity_action_reward"),
            "goal_progress_signal": _number(reward, "single_uav_goal_progress_signal"),
            "goal_progress_reward": _number(reward, "single_uav_goal_progress_reward"),
            "goal_direction_signal": _number(reward, "single_uav_goal_direction_signal"),
            "goal_direction_reward": _number(reward, "single_uav_goal_direction_reward"),
            "stable_steps": _number(reward, "single_uav_stable_steps"),
            "stable_hold_steps_required": _number(reward, "single_uav_stable_hold_steps"),
            "goal_stable": int(_number(reward, "single_uav_goal_stable", 0.0) > 0.5),
            "approach_segment": approach_segment_for_distance(current_distance),
            "distance_bin": distance_bin_for_distance(current_distance),
        }
        rows.append(row)
        previous_timestamp = timestamp
    return rows


def _episode_segment_rows(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create one row per policy/episode/approach segment."""

    output: list[dict[str, Any]] = []
    labels = [str(item["name"]) for item in APPROACH_SEGMENTS]
    for segment in APPROACH_SEGMENTS:
        name = str(segment["name"])
        selected = [row for row in step_rows if row.get("approach_segment") == name]
        next_name = str(segment["next"])
        continued = any(
            row.get("approach_segment") == next_name
            or (next_name == "goal_radius" and _number(row, "goal_distance_m") < 1.0)
            for row in step_rows
        )
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "episode_index": int(episode_index),
                "seed": int(seed),
                "segment": name,
                "segment_label": str(segment["label"]),
                "lower_m": float(segment["lower_m"]),
                "upper_m": float(segment["upper_m"]),
                "entered_segment": int(bool(selected)),
                "continued_to_next_segment": int(bool(selected) and continued),
                "step_count": len(selected),
                "first_distance_m": _number(selected[0], "goal_distance_m") if selected else float("nan"),
                "last_distance_m": _number(selected[-1], "goal_distance_m") if selected else float("nan"),
                "mean_radial_progress_mps": _mean(row.get("radial_progress_mps") for row in selected),
                "mean_delta_goal_distance_m": _mean(row.get("delta_goal_distance_m") for row in selected),
                "mean_radial_velocity_toward_mps": _mean(row.get("radial_velocity_toward_mps") for row in selected),
                "mean_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
                "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in selected),
                "braking_action_fraction": _mean(row.get("braking_action") for row in selected),
                "effective_braking_fraction": _mean(row.get("effective_braking") for row in selected),
                "mean_commanded_braking_delta_v_mps": _mean(
                    row.get("commanded_braking_delta_v_mps")
                    for row in selected
                    if _number(row, "braking_action") > 0.5
                ),
                "mean_applied_braking_delta_v_mps": _mean(
                    row.get("applied_braking_delta_v_mps")
                    for row in selected
                    if _number(row, "braking_action") > 0.5
                ),
                "mean_actual_speed_reduction_when_braking_mps": _mean(
                    row.get("actual_speed_reduction_when_braking_mps")
                    for row in selected
                    if _number(row, "braking_action") > 0.5
                ),
                "mean_braking_progress_signal": _mean(row.get("braking_progress_signal") for row in selected),
                "mean_braking_progress_reward": _mean(row.get("braking_progress_reward") for row in selected),
                "terminal_radial_progress_active_fraction": _mean(
                    row.get("terminal_radial_progress_active") for row in selected
                ),
                "mean_terminal_radial_progress_m": _mean(
                    row.get("terminal_radial_progress_m") for row in selected
                ),
                "mean_terminal_radial_progress_speed_excess_mps": _mean(
                    row.get("terminal_radial_progress_speed_excess_mps") for row in selected
                ),
                "mean_terminal_radial_progress_speed_gate": _mean(
                    row.get("terminal_radial_progress_speed_gate") for row in selected
                ),
                "mean_terminal_radial_progress_signal": _mean(
                    row.get("terminal_radial_progress_signal") for row in selected
                ),
                "mean_terminal_radial_progress_reward": _mean(
                    row.get("terminal_radial_progress_reward") for row in selected
                ),
                "mean_target_velocity_action_reward": _mean(row.get("target_velocity_action_reward") for row in selected),
                "mean_goal_direction_signal": _mean(row.get("goal_direction_signal") for row in selected),
                "mean_goal_direction_reward": _mean(row.get("goal_direction_reward") for row in selected),
                "mean_goal_progress_reward": _mean(row.get("goal_progress_reward") for row in selected),
            }
        )
    return output


def _episode_summary_row(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
) -> dict[str, Any]:
    distances = _finite(row.get("goal_distance_m") for row in step_rows)
    holds = _finite(row.get("stable_steps") for row in step_rows)
    minimum = float(np.min(distances)) if distances.size else float("nan")
    stable = int(any(int(row.get("goal_stable", 0)) > 0 for row in step_rows))
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_index": int(episode_index),
        "seed": int(seed),
        "scenario_hash": scenario_hash,
        "entered_8m": int(np.isfinite(minimum) and minimum <= 8.0),
        "entered_1m": int(np.isfinite(minimum) and minimum <= 1.0),
        "stable_goal": stable,
        "minimum_goal_distance_m": minimum,
        "final_goal_distance_m": _number(step_rows[-1], "goal_distance_m") if step_rows else float("nan"),
        "mean_speed_mps": _mean(row.get("total_speed_mps") for row in step_rows),
        "mean_radial_progress_mps": _mean(row.get("radial_progress_mps") for row in step_rows),
        "maximum_consecutive_goal_hold_steps": int(round(_max(holds, default=0.0))),
        "hold_steps_ge_1": int(_max(holds, default=0.0) >= 1.0),
        "hold_steps_ge_3": int(_max(holds, default=0.0) >= 3.0),
        "hold_steps_ge_5": int(_max(holds, default=0.0) >= 5.0),
        "step_count": len(step_rows),
    }


def _pooled_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    segment: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    name = str(segment["name"])
    episode_rows = [row for row in rows if row.get("segment") == name]
    entered = [row for row in episode_rows if int(row.get("entered_segment", 0))]
    continuation = [row for row in entered if int(row.get("continued_to_next_segment", 0))]
    count = len({int(row.get("episode_index", -1)) for row in rows})
    steps = [row for row in step_rows if row.get("approach_segment") == name]
    braking_steps = [row for row in steps if _number(row, "braking_action") > 0.5]
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "segment": name,
        "segment_label": str(segment["label"]),
        "lower_m": float(segment["lower_m"]),
        "upper_m": float(segment["upper_m"]),
        "episodes_total": count,
        "episodes_entered": len(entered),
        "episodes_entered_rate": _rate(len(entered), count),
        "episodes_continued_to_next": len(continuation),
        "episodes_continued_to_next_rate": _rate(len(continuation), len(entered)),
        "pooled_step_count": len(steps),
        "mean_radial_progress_mps": _mean(row.get("radial_progress_mps") for row in steps),
        "mean_delta_goal_distance_m": _mean(row.get("delta_goal_distance_m") for row in steps),
        "mean_radial_velocity_toward_mps": _mean(row.get("radial_velocity_toward_mps") for row in steps),
        "mean_speed_mps": _mean(row.get("total_speed_mps") for row in steps),
        "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in steps),
        "braking_action_fraction": _mean(row.get("braking_action") for row in steps),
        "effective_braking_fraction": _mean(row.get("effective_braking") for row in steps),
        "mean_commanded_braking_delta_v_mps": _mean(
            row.get("commanded_braking_delta_v_mps") for row in braking_steps
        ),
        "mean_applied_braking_delta_v_mps": _mean(
            row.get("applied_braking_delta_v_mps") for row in braking_steps
        ),
        "mean_actual_speed_reduction_when_braking_mps": _mean(
            row.get("actual_speed_reduction_when_braking_mps") for row in braking_steps
        ),
        "mean_braking_progress_signal": _mean(row.get("braking_progress_signal") for row in steps),
        "mean_braking_progress_reward": _mean(row.get("braking_progress_reward") for row in steps),
        "terminal_radial_progress_active_fraction": _mean(
            row.get("terminal_radial_progress_active") for row in steps
        ),
        "mean_terminal_radial_progress_m": _mean(
            row.get("terminal_radial_progress_m") for row in steps
        ),
        "mean_terminal_radial_progress_speed_excess_mps": _mean(
            row.get("terminal_radial_progress_speed_excess_mps") for row in steps
        ),
        "mean_terminal_radial_progress_speed_gate": _mean(
            row.get("terminal_radial_progress_speed_gate") for row in steps
        ),
        "mean_terminal_radial_progress_signal": _mean(
            row.get("terminal_radial_progress_signal") for row in steps
        ),
        "mean_terminal_radial_progress_reward": _mean(
            row.get("terminal_radial_progress_reward") for row in steps
        ),
        "mean_target_velocity_action_reward": _mean(row.get("target_velocity_action_reward") for row in steps),
        "mean_goal_direction_signal": _mean(row.get("goal_direction_signal") for row in steps),
        "mean_goal_direction_reward": _mean(row.get("goal_direction_reward") for row in steps),
        "mean_goal_progress_reward": _mean(row.get("goal_progress_reward") for row in steps),
        "mean_dwell_steps_per_entering_episode": _mean(row.get("step_count") for row in entered),
        "maximum_dwell_steps": int(round(_max((row.get("step_count") for row in entered), default=0.0))),
        "total_dwell_steps": int(round(float(np.sum(_finite(row.get("step_count") for row in entered))))),
    }


def _distance_bin_summary(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    distance_bin: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(distance_bin["name"])
    selected = [row for row in step_rows if row.get("distance_bin") == name]
    episode_count = len({int(row.get("episode_index", -1)) for row in selected})
    braking_steps = [row for row in selected if _number(row, "braking_action") > 0.5]
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "distance_bin": name,
        "distance_bin_label": str(distance_bin["label"]),
        "lower_m": float(distance_bin["lower_m"]),
        "upper_m": float(distance_bin["upper_m"]),
        "episodes_with_samples": episode_count,
        "step_count": len(selected),
        "mean_distance_m": _mean(row.get("goal_distance_m") for row in selected),
        "mean_radial_progress_mps": _mean(row.get("radial_progress_mps") for row in selected),
        "mean_delta_goal_distance_m": _mean(row.get("delta_goal_distance_m") for row in selected),
        "mean_radial_velocity_toward_mps": _mean(row.get("radial_velocity_toward_mps") for row in selected),
        "mean_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
        "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in selected),
        "braking_action_fraction": _mean(row.get("braking_action") for row in selected),
        "effective_braking_fraction": _mean(row.get("effective_braking") for row in selected),
        "mean_commanded_braking_delta_v_mps": _mean(
            row.get("commanded_braking_delta_v_mps") for row in braking_steps
        ),
        "mean_applied_braking_delta_v_mps": _mean(
            row.get("applied_braking_delta_v_mps") for row in braking_steps
        ),
        "mean_actual_speed_reduction_when_braking_mps": _mean(
            row.get("actual_speed_reduction_when_braking_mps") for row in braking_steps
        ),
        "mean_braking_progress_signal": _mean(row.get("braking_progress_signal") for row in selected),
        "mean_braking_progress_reward": _mean(row.get("braking_progress_reward") for row in selected),
        "terminal_radial_progress_active_fraction": _mean(
            row.get("terminal_radial_progress_active") for row in selected
        ),
        "mean_terminal_radial_progress_m": _mean(
            row.get("terminal_radial_progress_m") for row in selected
        ),
        "mean_terminal_radial_progress_speed_excess_mps": _mean(
            row.get("terminal_radial_progress_speed_excess_mps") for row in selected
        ),
        "mean_terminal_radial_progress_speed_gate": _mean(
            row.get("terminal_radial_progress_speed_gate") for row in selected
        ),
        "mean_terminal_radial_progress_signal": _mean(
            row.get("terminal_radial_progress_signal") for row in selected
        ),
        "mean_terminal_radial_progress_reward": _mean(
            row.get("terminal_radial_progress_reward") for row in selected
        ),
        "mean_target_velocity_action_reward": _mean(row.get("target_velocity_action_reward") for row in selected),
        "mean_goal_direction_signal": _mean(row.get("goal_direction_signal") for row in selected),
        "mean_goal_direction_reward": _mean(row.get("goal_direction_reward") for row in selected),
        "mean_goal_progress_reward": _mean(row.get("goal_progress_reward") for row in selected),
    }


def _plot_diagnostics(
    *,
    step_rows: Sequence[Mapping[str, Any]],
    segment_summary: Sequence[Mapping[str, Any]],
    distance_summary: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#2563eb", "fine_tuned": "#dc2626"}
    names = [str(item["name"]) for item in DISTANCE_BINS]
    centers = {
        str(item["name"]): (float(item["lower_m"]) + float(item["upper_m"])) / 2.0
        for item in DISTANCE_BINS
    }

    def policy_rows(policy: str) -> list[Mapping[str, Any]]:
        return [row for row in step_rows if row.get("policy") == policy]

    def time_means(policy: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        selected = policy_rows(policy)
        by_step: dict[int, list[float]] = {}
        times: dict[int, float] = {}
        for row in selected:
            value = _number(row, key)
            step = int(row.get("step_index", -1))
            if np.isfinite(value) and step >= 0:
                by_step.setdefault(step, []).append(value)
                times[step] = _number(row, "timestamp_s", step * 0.2)
        keys = sorted(by_step)
        return (
            np.asarray([times[key] for key in keys], dtype=float),
            np.asarray([np.mean(by_step[key]) for key in keys], dtype=float),
        )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for policy in POLICY_NAMES:
        x, y = time_means(policy, "goal_distance_m")
        if x.size:
            axes[0, 0].plot(x, y, linewidth=2.0, color=colors[policy], label=POLICY_LABELS[policy])
    axes[0, 0].axhline(8.0, color="#f59e0b", linestyle=":", label="8 m")
    axes[0, 0].axhline(5.0, color="#a855f7", linestyle="--", label="5 m")
    axes[0, 0].axhline(3.0, color="#0f766e", linestyle="--", label="3 m")
    axes[0, 0].axhline(1.0, color="#dc2626", linestyle=":", label="1 m")
    axes[0, 0].set_title("Mean goal distance versus time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Goal distance (m)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    for policy in POLICY_NAMES:
        selected = [row for row in distance_summary if row.get("policy") == policy]
        selected_by_name = {str(row["distance_bin"]): row for row in selected}
        x = [centers[name] for name in names if name in selected_by_name]
        y = [_number(selected_by_name[name], "mean_radial_progress_mps") for name in names if name in selected_by_name]
        if x:
            axes[0, 1].plot(x, y, marker="o", linewidth=2.0, color=colors[policy], label=POLICY_LABELS[policy])
    axes[0, 1].axvline(3.0, color="#111827", linestyle="--", label="3 m boundary")
    axes[0, 1].axhline(0.0, color="#6b7280", linewidth=1.0)
    axes[0, 1].set_title("Radial progress versus distance")
    axes[0, 1].set_xlabel("Goal distance (m)")
    axes[0, 1].set_ylabel("Distance decrease rate (m/s)")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    for policy in POLICY_NAMES:
        selected = [row for row in distance_summary if row.get("policy") == policy]
        selected_by_name = {str(row["distance_bin"]): row for row in selected}
        x = [centers[name] for name in names if name in selected_by_name]
        speed = [_number(selected_by_name[name], "mean_speed_mps") for name in names if name in selected_by_name]
        target = [_number(selected_by_name[name], "mean_target_speed_mps") for name in names if name in selected_by_name]
        if x:
            axes[1, 0].plot(x, speed, marker="o", linewidth=2.0, color=colors[policy], label=f"{POLICY_LABELS[policy]} speed")
            axes[1, 0].plot(x, target, marker="^", linestyle="--", linewidth=1.5, color=colors[policy], alpha=0.7, label=f"{POLICY_LABELS[policy]} target")
    axes[1, 0].axvline(3.0, color="#111827", linestyle="--", label="3 m boundary")
    axes[1, 0].set_title("Actual speed and target speed versus distance")
    axes[1, 0].set_xlabel("Goal distance (m)")
    axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7)

    for policy in POLICY_NAMES:
        selected = [row for row in distance_summary if row.get("policy") == policy]
        selected_by_name = {str(row["distance_bin"]): row for row in selected}
        x = [centers[name] for name in names if name in selected_by_name]
        braking = [_number(selected_by_name[name], "braking_action_fraction") for name in names if name in selected_by_name]
        effective = [_number(selected_by_name[name], "effective_braking_fraction") for name in names if name in selected_by_name]
        if x:
            axes[1, 1].plot(x, braking, marker="o", linewidth=2.0, color=colors[policy], label=f"{POLICY_LABELS[policy]} action")
            axes[1, 1].plot(x, effective, marker="s", linestyle="--", linewidth=1.5, color=colors[policy], alpha=0.7, label=f"{POLICY_LABELS[policy]} effective")
    axes[1, 1].axvline(3.0, color="#111827", linestyle="--", label="3 m boundary")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Braking action versus distance")
    axes[1, 1].set_xlabel("Goal distance (m)")
    axes[1, 1].set_ylabel("Fraction of step samples")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)
    figure.suptitle("Paired frozen terminal-approach diagnostic")
    combined_path = output_dir / "terminal_approach_diagnostic.png"
    figure.savefig(combined_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for policy in POLICY_NAMES:
        x, y = time_means(policy, "radial_progress_mps")
        if x.size:
            axis.plot(x, y, linewidth=2.0, color=colors[policy], label=POLICY_LABELS[policy])
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title("Mean radial progress versus time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Distance decrease rate (m/s)")
    axis.grid(alpha=0.25)
    axis.legend()
    radial_time_path = output_dir / "radial_progress_vs_time.png"
    figure.savefig(radial_time_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for policy in POLICY_NAMES:
        selected = [row for row in distance_summary if row.get("policy") == policy]
        selected_by_name = {str(row["distance_bin"]): row for row in selected}
        x = [centers[name] for name in names if name in selected_by_name]
        speed = [_number(selected_by_name[name], "mean_speed_mps") for name in names if name in selected_by_name]
        target = [_number(selected_by_name[name], "mean_target_speed_mps") for name in names if name in selected_by_name]
        if x:
            axis.plot(x, speed, marker="o", color=colors[policy], linewidth=2.0, label=f"{POLICY_LABELS[policy]} speed")
            axis.plot(x, target, marker="^", color=colors[policy], linestyle="--", alpha=0.7, label=f"{POLICY_LABELS[policy]} target")
    axis.axvline(3.0, color="#111827", linestyle="--", label="3 m boundary")
    axis.set_title("Speed and target speed versus goal distance")
    axis.set_xlabel("Goal distance (m)")
    axis.set_ylabel("Speed (m/s)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    speed_distance_path = output_dir / "speed_target_vs_distance.png"
    figure.savefig(speed_distance_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    segment_positions = np.arange(len(APPROACH_SEGMENTS))
    width = 0.18
    for policy_index, policy in enumerate(POLICY_NAMES):
        selected = [row for row in segment_summary if row.get("policy") == policy]
        values = {str(row["segment"]): _number(row, "braking_action_fraction") for row in selected}
        effective = {str(row["segment"]): _number(row, "effective_braking_fraction") for row in selected}
        offsets = segment_positions + (policy_index - 0.5) * width
        axis.bar(offsets, [values.get(str(item["name"]), 0.0) for item in APPROACH_SEGMENTS], width=width, alpha=0.85, color=colors[policy], label=f"{POLICY_LABELS[policy]} action")
        axis.plot(offsets, [effective.get(str(item["name"]), 0.0) for item in APPROACH_SEGMENTS], marker="o", color="#111827", linestyle="--", label=f"{POLICY_LABELS[policy]} effective")
    axis.set_xticks(segment_positions, [str(item["label"]) for item in APPROACH_SEGMENTS])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of step samples")
    axis.set_title("Braking fractions by terminal approach segment")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)
    braking_path = output_dir / "braking_vs_distance.png"
    figure.savefig(braking_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "combined": str(combined_path.resolve()),
        "radial_progress_vs_time": str(radial_time_path.resolve()),
        "speed_target_vs_distance": str(speed_distance_path.resolve()),
        "braking_vs_distance": str(braking_path.resolve()),
    }


def _prepare_output_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"diagnostic output is not empty: {path}; use --force for this exact directory"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_policy_manifest(
    *,
    policy_root: Path,
    scale_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hashes: Mapping[str, Any],
    policy_hash_after: str,
    checkpoint_sha256_after: str,
) -> dict[str, Any]:
    _write_scale_manifest(scale_root=scale_root, config=config, episodes=episode_summaries)
    manifest_path = scale_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "single_uav_paired_terminal_approach_diagnostic",
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "policy_role": policy_root.name,
            "policy_source": "frozen_checkpoint",
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256_before": checkpoint_sha256,
                "sha256_after": checkpoint_sha256_after,
                "unchanged": checkpoint_sha256 == checkpoint_sha256_after,
            },
            "policy_identity": {
                **dict(policy_hashes),
                "after_evaluation": policy_hash_after,
                "unchanged": policy_hashes.get("loaded_policy_hash") == policy_hash_after,
            },
            "paired_evaluation_contract": {
                "scenario_distribution": "normal far-distance test only",
                "shared_seed_index": "test seed indices 0..episodes-1",
                "scenario_reused_between_policies": True,
                "curriculum_used": False,
                "update_policy": False,
            },
            "approach_segment_contract": {
                "boundary_rule": "lower inclusive, upper exclusive",
                "segments": _jsonable(list(APPROACH_SEGMENTS)),
                "goal_radius": "[0,1) m",
            },
            "read_only_contract": [
                "MAPPOTrainer.update was not called",
                "no rollout transition was added",
                "optimizer_state_dict was not loaded or written",
                "reward, curriculum, dynamics, action-space, and PPO parameters were not changed",
            ],
        }
    )
    _write_json(manifest_path, _jsonable(manifest))
    _write_json(
        policy_root / "checkpoint_identity.json",
        _jsonable(
            {
                "policy_role": policy_root.name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256_before": checkpoint_sha256,
                "checkpoint_sha256_after": checkpoint_sha256_after,
                "checkpoint_unchanged": checkpoint_sha256 == checkpoint_sha256_after,
                "policy_hashes": {
                    **dict(policy_hashes),
                    "after_evaluation": policy_hash_after,
                    "policy_unchanged": policy_hashes.get("loaded_policy_hash") == policy_hash_after,
                },
                "policy_updates_performed": False,
            }
        ),
    )
    return manifest


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return dict(payload), _sha256(path)


def evaluate_terminal_approach_diagnostic(
    baseline_checkpoint: str | Path,
    fine_tuned_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the paired, frozen normal far-distance terminal approach diagnostic."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    output_root = Path(output_dir)
    _prepare_output_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(baseline_checkpoint)
    fine_path = Path(fine_tuned_checkpoint)
    baseline_payload, baseline_sha256 = _load_checkpoint(baseline_path)
    fine_payload, fine_sha256 = _load_checkpoint(fine_path)
    baseline_config = _config_from_checkpoint(
        baseline_payload,
        output_dir=output_root / "baseline",
        progress=progress,
    )
    fine_config = _config_from_checkpoint(
        fine_payload,
        output_dir=output_root / "fine_tuned",
        progress=progress,
    )
    if baseline_config.uav_count != 1 or fine_config.uav_count != 1:
        raise ValueError("terminal approach diagnostic requires two 1-UAV checkpoints")
    if baseline_config.seed != fine_config.seed:
        raise ValueError("baseline and fine-tuned checkpoints must use the same base seed")
    if baseline_config.scenario_seed != fine_config.scenario_seed:
        raise ValueError("baseline and fine-tuned checkpoints must use the same scenario seed")
    if baseline_config.max_steps != fine_config.max_steps:
        raise ValueError("baseline and fine-tuned checkpoints must use the same max_steps")

    # Generate each normal far-distance test scenario exactly once from a
    # curriculum-free config, then pass a deep copy to both frozen policies.
    scenario_config = replace(
        baseline_config,
        output_dir=output_root / "shared_scenarios",
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        force=True,
    )
    paired_scenarios: list[dict[str, Any]] = []
    for episode_index in range(int(episodes)):
        seed = _scale_seed(scenario_config.seed, scenario_config.uav_count, "test", episode_index)
        scenario = _scenario_for(
            scenario_config,
            seed,
            split="test",
            episode_index=episode_index,
            phase="test",
        )
        paired_scenarios.append(
            {
                "episode_index": episode_index + 1,
                "seed": seed,
                "scenario": copy.deepcopy(scenario),
                "scenario_hash": _canonical_hash(scenario),
                "curriculum_used": bool("single_uav_braking_curriculum" in scenario),
            }
        )
    if any(item["curriculum_used"] for item in paired_scenarios):
        raise RuntimeError("paired diagnostic scenario generation unexpectedly used curriculum")

    payloads = {"baseline": baseline_payload, "fine_tuned": fine_payload}
    checkpoint_paths = {"baseline": baseline_path, "fine_tuned": fine_path}
    checkpoint_hashes = {"baseline": baseline_sha256, "fine_tuned": fine_sha256}
    configs = {"baseline": baseline_config, "fine_tuned": fine_config}
    all_step_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    all_episode_segment_rows: list[dict[str, Any]] = []
    policy_reports: dict[str, Any] = {}

    import torch

    for policy in POLICY_NAMES:
        policy_root = output_root / policy
        scale_root = policy_root / "scale_1"
        scale_root.mkdir(parents=True, exist_ok=True)
        config = configs[policy]
        _copy_calibration_files(scale_root, config)
        normalization = payloads[policy].get("normalization_stats")
        central = payloads[policy].get("central_normalization_stats")
        if isinstance(normalization, Mapping):
            _write_json(
                scale_root / "normalization_stats.json",
                _jsonable(
                    {
                        "normalization_stats": dict(normalization),
                        "central_normalization_stats": dict(central) if isinstance(central, Mapping) else None,
                        "source": "frozen_checkpoint",
                    }
                ),
            )
        trainer, policy_hashes, _ = _load_frozen_trainer(
            payloads[policy],
            config,
            metric_scales=config.metric_scales,
        )
        trainer.policy.eval()
        episode_summaries: list[dict[str, Any]] = []
        episode_metrics: list[dict[str, Any]] = []
        attention_rows: list[dict[str, Any]] = []
        episode_rows_for_policy: list[dict[str, Any]] = []
        segment_rows_for_policy: list[dict[str, Any]] = []
        serial_base = 610001 if policy == "baseline" else 620001
        for item in paired_scenarios:
            episode_index = int(item["episode_index"])
            seed = int(item["seed"])
            serial = serial_base + episode_index - 1
            with torch.no_grad():
                summary, _update, returned_rows = _run_episode(
                    config=config,
                    trainer=trainer,
                    scale_root=scale_root,
                    phase="terminal_approach_diagnostic",
                    split="test",
                    serial=serial,
                    seed=seed,
                    update_policy=False,
                    update_index=None,
                    episode_index=episode_index,
                    episode_total=int(episodes),
                    metric_scales=config.metric_scales,
                    flush_update=False,
                    scenario_config=copy.deepcopy(item["scenario"]),
                )
            episode_path = scale_root / "test" / f"episode_{serial:06d}"
            metric_row = returned_rows[-1] if returned_rows and isinstance(returned_rows[-1], Mapping) else _episode_metrics(episode_path, summary)
            episode_metrics.append(dict(metric_row))
            attention_rows.extend(
                row for row in returned_rows[:-1] if isinstance(row, Mapping) and "layer" in row
            )
            step_rows = _step_rows_from_episode(
                episode_path,
                policy=policy,
                episode_index=episode_index,
                seed=seed,
                scenario_hash=str(item["scenario_hash"]),
                scenario=item["scenario"],
            )
            episode_row = _episode_summary_row(
                step_rows,
                policy=policy,
                episode_index=episode_index,
                seed=seed,
                scenario_hash=str(item["scenario_hash"]),
            )
            segment_rows = _episode_segment_rows(
                step_rows,
                policy=policy,
                episode_index=episode_index,
                seed=seed,
            )
            all_step_rows.extend(step_rows)
            all_episode_rows.append(episode_row)
            all_episode_segment_rows.extend(segment_rows)
            episode_rows_for_policy.append(episode_row)
            segment_rows_for_policy.extend(segment_rows)
            episode_entry = dict(summary)
            episode_entry.update(
                {
                    "evaluation_only": True,
                    "policy_role": policy,
                    "paired_episode_index": episode_index,
                    "paired_seed": seed,
                    "paired_scenario_hash": item["scenario_hash"],
                    "curriculum_used": False,
                    "policy_updates_performed": False,
                }
            )
            episode_summaries.append(episode_entry)

        _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", episode_metrics)
        _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)
        checkpoint_sha256_after = _sha256(checkpoint_paths[policy])
        policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
        manifest = _write_policy_manifest(
            policy_root=policy_root,
            scale_root=scale_root,
            config=config,
            episode_summaries=episode_summaries,
            checkpoint_path=checkpoint_paths[policy],
            checkpoint_sha256=checkpoint_hashes[policy],
            policy_hashes=policy_hashes,
            policy_hash_after=policy_hash_after,
            checkpoint_sha256_after=checkpoint_sha256_after,
        )
        validation = validate_episode_benchmark(scale_root)
        policy_reports[policy] = {
            "policy_label": POLICY_LABELS[policy],
            "checkpoint": {
                "path": str(checkpoint_paths[policy].resolve()),
                "sha256_before": checkpoint_hashes[policy],
                "sha256_after": checkpoint_sha256_after,
                "unchanged": checkpoint_hashes[policy] == checkpoint_sha256_after,
                "policy_hashes": {
                    **policy_hashes,
                    "after_evaluation": policy_hash_after,
                    "policy_unchanged": policy_hashes.get("loaded_policy_hash") == policy_hash_after,
                },
            },
            "episodes": len(episode_rows_for_policy),
            "segment_episode_rows": len(segment_rows_for_policy),
            "schema_validation": validation,
            "manifest": str((scale_root / "manifest.json").resolve()),
        }

    segment_summary: list[dict[str, Any]] = []
    distance_summary: list[dict[str, Any]] = []
    for policy in POLICY_NAMES:
        policy_segment_rows = [row for row in all_episode_segment_rows if row.get("policy") == policy]
        policy_step_rows = [row for row in all_step_rows if row.get("policy") == policy]
        for segment in APPROACH_SEGMENTS:
            segment_summary.append(
                _pooled_summary(
                    policy_segment_rows,
                    policy=policy,
                    segment=segment,
                    step_rows=policy_step_rows,
                )
            )
        for distance_bin in DISTANCE_BINS:
            distance_summary.append(
                _distance_bin_summary(policy_step_rows, policy=policy, distance_bin=distance_bin)
            )

    paired_rows: list[dict[str, Any]] = []
    by_policy_episode = {
        policy: {int(row["episode_index"]): row for row in all_episode_rows if row.get("policy") == policy}
        for policy in POLICY_NAMES
    }
    for item in paired_scenarios:
        index = int(item["episode_index"])
        baseline_row = by_policy_episode["baseline"][index]
        fine_row = by_policy_episode["fine_tuned"][index]
        paired_rows.append(
            {
                "episode_index": index,
                "seed": int(item["seed"]),
                "scenario_hash": item["scenario_hash"],
                "scenario_hash_shared": True,
                "baseline_minimum_goal_distance_m": baseline_row["minimum_goal_distance_m"],
                "fine_tuned_minimum_goal_distance_m": fine_row["minimum_goal_distance_m"],
                "fine_tuned_minus_baseline_minimum_distance_m": (
                    float(fine_row["minimum_goal_distance_m"]) - float(baseline_row["minimum_goal_distance_m"])
                ),
                "baseline_entered_8m": baseline_row["entered_8m"],
                "fine_tuned_entered_8m": fine_row["entered_8m"],
                "baseline_entered_1m": baseline_row["entered_1m"],
                "fine_tuned_entered_1m": fine_row["entered_1m"],
                "baseline_stable_goal": baseline_row["stable_goal"],
                "fine_tuned_stable_goal": fine_row["stable_goal"],
                "baseline_max_hold_steps": baseline_row["maximum_consecutive_goal_hold_steps"],
                "fine_tuned_max_hold_steps": fine_row["maximum_consecutive_goal_hold_steps"],
            }
        )

    plot_paths = _plot_diagnostics(
        step_rows=all_step_rows,
        segment_summary=segment_summary,
        distance_summary=distance_summary,
        output_dir=plots_root,
    )
    _write_rows_csv(logs_root / "step_metrics.csv", all_step_rows)
    _write_rows_csv(logs_root / "episode_metrics.csv", all_episode_rows)
    _write_rows_csv(logs_root / "episode_segment_metrics.csv", all_episode_segment_rows)
    _write_rows_csv(logs_root / "segment_summary.csv", segment_summary)
    _write_rows_csv(logs_root / "distance_bin_summary.csv", distance_summary)
    _write_rows_csv(logs_root / "paired_episode_comparison.csv", paired_rows)
    scenario_hashes = [str(item["scenario_hash"]) for item in paired_scenarios]
    # A fixed ``scenario_seed`` intentionally makes the scene geometry hash
    # repeat across episode indices.  The important paired condition is that
    # each episode's exact scenario object is passed to both policies; unique
    # hashes across episodes are neither required nor desirable here.
    scenario_hashes_recorded = len(scenario_hashes) == int(episodes)
    shared_scenario_hash_count = len(set(scenario_hashes))
    all_policy_schema_ok = all(
        bool(report["schema_validation"].get("ok", False))
        for report in policy_reports.values()
    )
    all_read_only_ok = all(
        bool(report["checkpoint"].get("unchanged", False))
        and bool(report["checkpoint"]["policy_hashes"].get("policy_unchanged", False))
        for report in policy_reports.values()
    )
    root_manifest = {
        "dataset_id": "single_uav_paired_terminal_approach_diagnostic",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "uav_count": 1,
        "episodes_per_policy": int(episodes),
        "policies": {
            policy: {
                "label": POLICY_LABELS[policy],
                "path": f"{policy}/scale_1",
                "checkpoint": str(checkpoint_paths[policy].resolve()),
                "checkpoint_sha256": checkpoint_hashes[policy],
            }
            for policy in POLICY_NAMES
        },
        "paired_scenario_contract": {
            "split": "test",
            "distribution": "normal far-distance",
            "seed_indices": list(range(int(episodes))),
            "scenario_hashes_recorded": scenario_hashes_recorded,
            "shared_scenario_hash_count": shared_scenario_hash_count,
            "same_scenario_passed_to_both_policies": True,
            "curriculum_used": False,
        },
        "approach_segment_contract": {
            "boundary_rule": "lower inclusive, upper exclusive",
            "segments": _jsonable(list(APPROACH_SEGMENTS)),
            "goal_radius": "[0,1) m",
        },
        "read_only_contract": {
            "no_training_update_call": True,
            "no_transition_added": True,
            "optimizer_state_not_loaded": True,
            "reward_curriculum_dynamics_action_space_unchanged": True,
        },
        "policies_reports": policy_reports,
    }
    _write_json(output_root / "manifest.json", _jsonable(root_manifest))
    _write_json(logs_root / "paired_scenario_manifest.json", _jsonable(root_manifest["paired_scenario_contract"]))

    summary = {
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "uav_count": 1,
        "episodes_per_policy": int(episodes),
        "total_policy_episodes": int(episodes) * len(POLICY_NAMES),
        "policies": policy_reports,
        "segment_summary": segment_summary,
        "distance_bin_summary": distance_summary,
        "paired_episode_comparison": paired_rows,
        "files": {
            "step_metrics_csv": str((logs_root / "step_metrics.csv").resolve()),
            "episode_metrics_csv": str((logs_root / "episode_metrics.csv").resolve()),
            "episode_segment_metrics_csv": str((logs_root / "episode_segment_metrics.csv").resolve()),
            "segment_summary_csv": str((logs_root / "segment_summary.csv").resolve()),
            "distance_bin_summary_csv": str((logs_root / "distance_bin_summary.csv").resolve()),
            "paired_episode_comparison_csv": str((logs_root / "paired_episode_comparison.csv").resolve()),
            "manifest": str((output_root / "manifest.json").resolve()),
            "plots": plot_paths,
        },
        "paired_scenario_checks": {
            "same_scenario_passed_to_both_policies": True,
            "curriculum_used_in_any_scenario": any(item["curriculum_used"] for item in paired_scenarios),
            "scenario_hashes_recorded": scenario_hashes_recorded,
            "shared_scenario_hash_count": shared_scenario_hash_count,
        },
        "read_only_checks": {
            "all_checkpoints_unchanged": all_read_only_ok,
            "all_policies_unchanged": all_read_only_ok,
            "no_training_update_call": True,
            "no_transition_added": True,
            "optimizer_state_not_loaded": True,
        },
        "schema_validation": {
            "ok": bool(all_policy_schema_ok),
            "policies": {
                policy: report["schema_validation"] for policy, report in policy_reports.items()
            },
        },
        "scope": {
            "normal_far_distance_only": True,
            "validation_or_training_evaluated": False,
            "goal_hold_reward_enabled": False,
            "reward_or_curriculum_modified": False,
            "diagnostic_boundary_rule": "[5,8), [3,5), [1,3), [0,1) goal radius",
        },
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "APPROACH_SEGMENTS",
    "DISTANCE_BINS",
    "approach_segment_for_distance",
    "distance_bin_for_distance",
    "evaluate_terminal_approach_diagnostic",
]
