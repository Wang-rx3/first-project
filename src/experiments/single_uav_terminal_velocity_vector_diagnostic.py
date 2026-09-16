"""Frozen C-vs-D terminal velocity-vector diagnosis.

This module compares the early-smooth mainline checkpoint with the rejected
terminal-radial-progress ablation on the same normal far-distance scenarios.
It is deliberately evaluation-only: no rollout transition is added and no
policy, optimizer, reward, or curriculum state is updated.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import braking_distance_target_speed
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import (
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


POLICY_NAMES: tuple[str, ...] = ("c_early_smooth", "d_radial_ablation")
POLICY_LABELS: dict[str, str] = {
    "c_early_smooth": "C early_smooth-only",
    "d_radial_ablation": "D radial-progress ablation",
}

TERMINAL_BAND = {
    "name": "1_3m",
    "label": "1 < d <= 3 m",
    "lower_m": 1.0,
    "upper_m": 3.0,
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
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def _mean(values: Iterable[Any]) -> float:
    values_array = _finite(values)
    return float(np.mean(values_array)) if values_array.size else float("nan")


def _median(values: Iterable[Any]) -> float:
    values_array = _finite(values)
    return float(np.median(values_array)) if values_array.size else float("nan")


def _percentile(values: Iterable[Any], percentile: float) -> float:
    values_array = _finite(values)
    return float(np.percentile(values_array, percentile)) if values_array.size else float("nan")


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _timestamp_key(row: Mapping[str, Any]) -> float:
    return round(_number(row, "timestamp", _number(row, "timestamp_s", float("nan"))), 6)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _goal_from_scenario(scenario: Mapping[str, Any]) -> np.ndarray:
    goals = scenario.get("goals", {})
    if not isinstance(goals, Mapping) or "uav_00" not in goals:
        raise ValueError("diagnostic scenario is missing goals.uav_00")
    goal = np.asarray(goals["uav_00"], dtype=float).reshape(3)
    if not np.isfinite(goal).all():
        raise ValueError("diagnostic goal is not finite")
    return goal


def _vector(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    values = np.asarray(
        [_number(row, f"{prefix}{axis}") for axis in ("x", "y", "z")],
        dtype=float,
    )
    return values


def _target_speed(config: Any, distance_m: float) -> float:
    return float(
        braking_distance_target_speed(
            float(distance_m),
            goal_threshold_m=float(config.single_uav_goal_threshold_m),
            near_goal_radius_m=float(config.single_uav_near_goal_radius_m),
            stable_speed_mps=float(config.single_uav_goal_speed_threshold_mps),
            braking_acceleration_mps2=float(config.single_uav_braking_acceleration_mps2),
            braking_margin_m=float(config.single_uav_braking_margin_m),
            far_target_speed_mps=float(config.single_uav_target_speed_far_mps),
            max_speed_mps=12.0,
            speed_profile=str(config.single_uav_braking_distance_speed_profile),
        )
    )


def velocity_vector_step_rows(
    *,
    episode_path: Path,
    policy: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
    scenario: Mapping[str, Any],
    config: Any,
) -> list[dict[str, Any]]:
    """Join EKF estimates and reward rows into policy-observable vector data."""

    rewards = _read_csv(episode_path / "rewards.csv")
    estimates = _read_csv(episode_path / "estimates.csv")
    actions = _read_csv(episode_path / "actions.csv")
    estimates_by_time = {_timestamp_key(row): row for row in estimates}
    actions_by_time = {_timestamp_key(row): row for row in actions}
    goal = _goal_from_scenario(scenario)
    rows: list[dict[str, Any]] = []
    for step_index, reward in enumerate(rewards):
        timestamp = _number(reward, "timestamp", (step_index + 1) * 0.2)
        estimate = estimates_by_time.get(round(timestamp, 6), {})
        action = actions_by_time.get(round(timestamp, 6), {})
        position = _vector(estimate, "")
        velocity = _vector(estimate, "v")
        valid_position = np.isfinite(position).all()
        valid_velocity = np.isfinite(velocity).all()
        goal_vector = goal - position if valid_position else np.full(3, np.nan)
        distance = float(np.linalg.norm(goal_vector)) if np.isfinite(goal_vector).all() else float("nan")
        if not np.isfinite(distance):
            distance = _number(reward, "current_goal_distance")
            goal_vector = np.full(3, np.nan)
        if np.isfinite(goal_vector).all() and distance > 1e-9:
            goal_direction = goal_vector / distance
        else:
            goal_direction = np.full(3, np.nan)
        speed = float(np.linalg.norm(velocity)) if valid_velocity else float("nan")
        target_speed = _number(reward, "single_uav_target_speed_mps")
        if not np.isfinite(target_speed) and np.isfinite(distance):
            target_speed = _target_speed(config, distance)
        target_velocity = (
            goal_direction * target_speed
            if np.isfinite(goal_direction).all() and np.isfinite(target_speed)
            else np.full(3, np.nan)
        )
        radial_velocity = (
            float(np.dot(velocity, goal_direction))
            if valid_velocity and np.isfinite(goal_direction).all()
            else float("nan")
        )
        tangential_speed = (
            float(np.sqrt(max(0.0, speed * speed - radial_velocity * radial_velocity)))
            if np.isfinite(speed) and np.isfinite(radial_velocity)
            else float("nan")
        )
        if valid_velocity and speed > 1e-9 and np.isfinite(goal_direction).all():
            direction_angle_error_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(np.dot(velocity / speed, goal_direction), -1.0, 1.0)
                    )
                )
            )
        else:
            direction_angle_error_deg = float("nan")
        vector_error = (
            float(np.linalg.norm(velocity - target_velocity))
            if valid_velocity and np.isfinite(target_velocity).all()
            else float("nan")
        )
        active = int(
            np.isfinite(distance)
            and float(TERMINAL_BAND["lower_m"]) < distance <= float(TERMINAL_BAND["upper_m"])
        )
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "episode_index": int(episode_index),
                "seed": int(seed),
                "episode_id": str(episode_path.name),
                "scenario_hash": scenario_hash,
                "step_index": int(step_index),
                "timestamp_s": float(timestamp),
                "estimated_goal_distance_m": distance,
                "terminal_vector_band_active": active,
                "total_speed_mps": speed,
                "radial_velocity_toward_goal_mps": radial_velocity,
                "tangential_speed_mps": tangential_speed,
                "direction_angle_error_deg": direction_angle_error_deg,
                "target_speed_mps": target_speed,
                "current_velocity_x_mps": float(velocity[0]) if valid_velocity else float("nan"),
                "current_velocity_y_mps": float(velocity[1]) if valid_velocity else float("nan"),
                "current_velocity_z_mps": float(velocity[2]) if valid_velocity else float("nan"),
                "target_velocity_x_mps": float(target_velocity[0]) if np.isfinite(target_velocity[0]) else float("nan"),
                "target_velocity_y_mps": float(target_velocity[1]) if np.isfinite(target_velocity[1]) else float("nan"),
                "target_velocity_z_mps": float(target_velocity[2]) if np.isfinite(target_velocity[2]) else float("nan"),
                "velocity_vector_error_mps": vector_error,
                "braking_action": _number(action, "braking_action", _number(reward, "single_uav_braking_action_signal", 0.0)),
                "effective_braking": _number(action, "brake_effective", _number(reward, "single_uav_effective_braking_signal", 0.0)),
                "braking_progress_signal": _number(reward, "single_uav_braking_progress_signal", 0.0),
                "target_velocity_action_reward": _number(reward, "single_uav_target_velocity_action_reward", 0.0),
                "goal_direction_reward": _number(reward, "single_uav_goal_direction_reward", 0.0),
                "goal_progress_reward": _number(reward, "single_uav_goal_progress_reward", 0.0),
            }
        )
    return rows


def velocity_vector_episode_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_index: int,
    seed: int,
    scenario_hash: str,
) -> dict[str, Any]:
    distances = _finite(row.get("estimated_goal_distance_m") for row in rows)
    terminal = [row for row in rows if int(_number(row, "terminal_vector_band_active", 0.0)) == 1]
    minimum_distance = float(np.min(distances)) if distances.size else float("nan")
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_index": int(episode_index),
        "seed": int(seed),
        "scenario_hash": scenario_hash,
        "entered_3m": int(distances.size > 0 and np.any(distances <= 3.0)),
        "minimum_goal_distance_m": minimum_distance,
        "terminal_1_3m_dwell_steps": len(terminal),
        "mean_radial_velocity_toward_goal_mps": _mean(
            row.get("radial_velocity_toward_goal_mps") for row in terminal
        ),
        "mean_tangential_speed_mps": _mean(row.get("tangential_speed_mps") for row in terminal),
        "mean_total_speed_mps": _mean(row.get("total_speed_mps") for row in terminal),
        "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in terminal),
        "mean_velocity_vector_error_mps": _mean(
            row.get("velocity_vector_error_mps") for row in terminal
        ),
        "mean_direction_angle_error_deg": _mean(
            row.get("direction_angle_error_deg") for row in terminal
        ),
    }


def velocity_vector_policy_summary(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    terminal = [
        row
        for row in step_rows
        if int(_number(row, "terminal_vector_band_active", 0.0)) == 1
    ]
    episodes = [row for row in episode_rows if row.get("policy") == policy]
    entered = sum(int(row.get("entered_3m", 0)) for row in episodes)
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": len(episodes),
        "entered_3m_episodes": entered,
        "entered_3m_rate": _rate(entered, len(episodes)),
        "entered_3m_minimum_distance_m": _mean(
            row.get("minimum_goal_distance_m") for row in episodes if int(row.get("entered_3m", 0))
        ),
        "global_minimum_goal_distance_m": (
            float(np.min(_finite(row.get("minimum_goal_distance_m") for row in episodes)))
            if _finite(row.get("minimum_goal_distance_m") for row in episodes).size
            else float("nan")
        ),
        "terminal_1_3m_step_count": len(terminal),
        "mean_terminal_1_3m_radial_velocity_toward_goal_mps": _mean(
            row.get("radial_velocity_toward_goal_mps") for row in terminal
        ),
        "mean_terminal_1_3m_tangential_speed_mps": _mean(
            row.get("tangential_speed_mps") for row in terminal
        ),
        "mean_terminal_1_3m_total_speed_mps": _mean(
            row.get("total_speed_mps") for row in terminal
        ),
        "mean_terminal_1_3m_target_speed_mps": _mean(
            row.get("target_speed_mps") for row in terminal
        ),
        "mean_terminal_1_3m_velocity_vector_error_mps": _mean(
            row.get("velocity_vector_error_mps") for row in terminal
        ),
        "median_terminal_1_3m_velocity_vector_error_mps": _median(
            row.get("velocity_vector_error_mps") for row in terminal
        ),
        "p90_terminal_1_3m_velocity_vector_error_mps": _percentile(
            (row.get("velocity_vector_error_mps") for row in terminal),
            90.0,
        ),
        "mean_terminal_1_3m_direction_angle_error_deg": _mean(
            row.get("direction_angle_error_deg") for row in terminal
        ),
        "mean_terminal_1_3m_dwell_steps_per_entered_episode": _mean(
            row.get("terminal_1_3m_dwell_steps")
            for row in episodes
            if int(row.get("entered_3m", 0))
        ),
    }


def assess_vector_hypothesis(
    policy_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report whether D shows the proposed vector-misalignment signature."""

    c = policy_summary["c_early_smooth"]
    d = policy_summary["d_radial_ablation"]

    def delta(key: str) -> float:
        return float(d.get(key, float("nan"))) - float(c.get(key, float("nan")))

    criteria = {
        "d_higher_tangential_speed": delta("mean_terminal_1_3m_tangential_speed_mps") > 1e-6,
        "d_higher_vector_error": delta("mean_terminal_1_3m_velocity_vector_error_mps") > 1e-6,
        "d_higher_direction_angle_error": delta("mean_terminal_1_3m_direction_angle_error_deg") > 1e-6,
        "d_high_total_but_lower_radial": (
            delta("mean_terminal_1_3m_total_speed_mps") > 1e-6
            and delta("mean_terminal_1_3m_radial_velocity_toward_goal_mps") < -1e-6
        ),
    }
    finite_evidence = all(
        np.isfinite(float(policy_summary[policy].get(key, float("nan"))))
        for policy in POLICY_NAMES
        for key in (
            "mean_terminal_1_3m_tangential_speed_mps",
            "mean_terminal_1_3m_velocity_vector_error_mps",
            "mean_terminal_1_3m_direction_angle_error_deg",
            "mean_terminal_1_3m_total_speed_mps",
            "mean_terminal_1_3m_radial_velocity_toward_goal_mps",
        )
    )
    return {
        "criteria": criteria,
        "supporting_criteria_count": int(sum(criteria.values())),
        "finite_evidence": bool(finite_evidence),
        "supported": bool(finite_evidence and any(criteria.values())),
        "d_minus_c": {
            key: delta(key)
            for key in (
                "mean_terminal_1_3m_tangential_speed_mps",
                "mean_terminal_1_3m_velocity_vector_error_mps",
                "mean_terminal_1_3m_direction_angle_error_deg",
                "mean_terminal_1_3m_total_speed_mps",
                "mean_terminal_1_3m_radial_velocity_toward_goal_mps",
            )
        },
        "decision": (
            "proceed_to_E_vector_correction"
            if finite_evidence and any(criteria)
            else "stop_and_report_vector_hypothesis_not_supported"
        ),
    }


def _plot_velocity_vector_diagnostic(
    *,
    step_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"c_early_smooth": "#2563eb", "d_radial_ablation": "#dc2626"}
    labels = {policy: POLICY_LABELS[policy] for policy in POLICY_NAMES}
    bins = np.linspace(1.0, 3.0, 21)
    centers = (bins[:-1] + bins[1:]) / 2.0

    def binned(policy: str, key: str) -> np.ndarray:
        selected = [
            row
            for row in step_rows
            if row.get("policy") == policy
            and int(_number(row, "terminal_vector_band_active", 0.0)) == 1
        ]
        values: list[float] = []
        distances = np.asarray([_number(row, "estimated_goal_distance_m") for row in selected])
        for lower, upper in zip(bins[:-1], bins[1:]):
            values.append(
                _mean(
                    row.get(key)
                    for row, distance in zip(selected, distances)
                    if lower <= distance < upper
                )
            )
        return np.asarray(values, dtype=float)

    def time_means(policy: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        selected = [row for row in step_rows if row.get("policy") == policy]
        by_step: dict[int, list[float]] = {}
        for row in selected:
            step = int(_number(row, "step_index", -1))
            value = _number(row, key)
            if step >= 0 and np.isfinite(value):
                by_step.setdefault(step, []).append(value)
        keys = sorted(by_step)
        return (
            np.asarray([key * 0.2 for key in keys], dtype=float),
            np.asarray([np.mean(by_step[key]) for key in keys], dtype=float),
        )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for policy in POLICY_NAMES:
        axes[0, 0].plot(
            centers,
            binned(policy, "total_speed_mps"),
            color=colors[policy],
            linewidth=2.0,
            label=f"{labels[policy]} total",
        )
        axes[0, 0].plot(
            centers,
            binned(policy, "target_speed_mps"),
            color=colors[policy],
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[policy]} target",
        )
    axes[0, 0].set_title("Speed and target speed in 1--3 m")
    axes[0, 0].set_xlabel("Estimated goal distance (m)")
    axes[0, 0].set_ylabel("Speed (m/s)")

    for policy in POLICY_NAMES:
        axes[0, 1].plot(
            centers,
            binned(policy, "radial_velocity_toward_goal_mps"),
            color=colors[policy],
            linewidth=2.0,
            label=labels[policy],
        )
        axes[0, 1].plot(
            centers,
            binned(policy, "tangential_speed_mps"),
            color=colors[policy],
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[policy]} tangential",
        )
    axes[0, 1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0, 1].set_title("Radial and tangential velocity")
    axes[0, 1].set_xlabel("Estimated goal distance (m)")
    axes[0, 1].set_ylabel("Velocity (m/s)")

    for policy in POLICY_NAMES:
        axes[1, 0].plot(
            centers,
            binned(policy, "velocity_vector_error_mps"),
            color=colors[policy],
            linewidth=2.0,
            label=labels[policy],
        )
    axes[1, 0].set_title("Velocity-vector error")
    axes[1, 0].set_xlabel("Estimated goal distance (m)")
    axes[1, 0].set_ylabel("||v - v_target|| (m/s)")

    for policy in POLICY_NAMES:
        axes[1, 1].plot(
            centers,
            binned(policy, "direction_angle_error_deg"),
            color=colors[policy],
            linewidth=2.0,
            label=labels[policy],
        )
    axes[1, 1].set_title("Velocity direction-angle error")
    axes[1, 1].set_xlabel("Estimated goal distance (m)")
    axes[1, 1].set_ylabel("Angle error (deg)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Frozen C vs D terminal velocity-vector diagnostic")
    distance_path = output_dir / "terminal_velocity_vector_vs_distance.png"
    figure.savefig(distance_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for policy in POLICY_NAMES:
        for axis, key, title, ylabel in (
            (axes[0, 0], "estimated_goal_distance_m", "Goal distance", "Distance (m)"),
            (axes[0, 1], "radial_velocity_toward_goal_mps", "Radial velocity", "Velocity (m/s)"),
            (axes[1, 0], "velocity_vector_error_mps", "Vector error", "Error (m/s)"),
            (axes[1, 1], "tangential_speed_mps", "Tangential speed", "Speed (m/s)"),
        ):
            x, y = time_means(policy, key)
            if x.size:
                axis.plot(x, y, color=colors[policy], linewidth=2.0, label=labels[policy])
            axis.set_title(title)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    figure.suptitle("Frozen C vs D terminal vector signals versus time")
    time_path = output_dir / "terminal_velocity_vector_vs_time.png"
    figure.savefig(time_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "distance": str(distance_path.resolve()),
        "time": str(time_path.resolve()),
    }


def _prepare_output_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"diagnostic output is not empty: {path}; use --force for this exact directory"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def evaluate_c_vs_d_terminal_velocity_vector_diagnostic(
    c_checkpoint: str | Path,
    d_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the frozen paired diagnostic and return its evidence report."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    output_root = Path(output_dir)
    _prepare_output_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {
        "c_early_smooth": Path(c_checkpoint),
        "d_radial_ablation": Path(d_checkpoint),
    }
    payloads: dict[str, Mapping[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    configs: dict[str, Any] = {}
    for policy, path in checkpoint_paths.items():
        import torch

        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload must be a mapping: {path}")
        payloads[policy] = dict(payload)
        checkpoint_hashes[policy] = _sha256(path)
        configs[policy] = _config_from_checkpoint(
            payload,
            output_dir=output_root / policy,
            progress=progress,
        )
    if any(configs[policy].uav_count != 1 for policy in POLICY_NAMES):
        raise ValueError("C vs D velocity-vector diagnostic requires 1-UAV checkpoints")
    if configs["c_early_smooth"].seed != configs["d_radial_ablation"].seed:
        raise ValueError("C and D must use the same base seed")
    if configs["c_early_smooth"].scenario_seed != configs["d_radial_ablation"].scenario_seed:
        raise ValueError("C and D must use the same scenario seed")

    scenario_config = replace(
        configs["c_early_smooth"],
        output_dir=output_root / "shared_scenarios",
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        force=True,
    )
    paired_scenarios: list[dict[str, Any]] = []
    for episode_index in range(int(episodes)):
        seed = _scale_seed(
            scenario_config.seed,
            scenario_config.uav_count,
            "test",
            episode_index,
        )
        scenario = _scenario_for(
            scenario_config,
            seed,
            split="test",
            episode_index=episode_index,
            phase="test",
        )
        if "single_uav_braking_curriculum" in scenario or "single_uav_terminal_braking_curriculum" in scenario:
            raise RuntimeError("paired C/D diagnostic unexpectedly used curriculum")
        paired_scenarios.append(
            {
                "episode_index": episode_index + 1,
                "seed": seed,
                "scenario": copy.deepcopy(scenario),
                "scenario_hash": _canonical_hash(scenario),
            }
        )

    all_step_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    policy_reports: dict[str, Any] = {}
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
        policy_episode_rows: list[dict[str, Any]] = []
        serial_base = 610001 if policy == "c_early_smooth" else 620001
        for item in paired_scenarios:
            episode_index = int(item["episode_index"])
            seed = int(item["seed"])
            with torch.no_grad():
                summary, _update, returned_rows = _run_episode(
                    config=config,
                    trainer=trainer,
                    scale_root=scale_root,
                    phase="terminal_velocity_vector_diagnostic",
                    split="test",
                    serial=serial_base + episode_index - 1,
                    seed=seed,
                    update_policy=False,
                    update_index=None,
                    episode_index=episode_index,
                    episode_total=int(episodes),
                    metric_scales=config.metric_scales,
                    flush_update=False,
                    scenario_config=copy.deepcopy(item["scenario"]),
                )
            episode_path = scale_root / "test" / f"episode_{serial_base + episode_index - 1:06d}"
            episode_rows = velocity_vector_step_rows(
                episode_path=episode_path,
                policy=policy,
                episode_index=episode_index,
                seed=seed,
                scenario_hash=str(item["scenario_hash"]),
                scenario=item["scenario"],
                config=config,
            )
            episode_row = velocity_vector_episode_summary(
                episode_rows,
                policy=policy,
                episode_index=episode_index,
                seed=seed,
                scenario_hash=str(item["scenario_hash"]),
            )
            all_step_rows.extend(episode_rows)
            all_episode_rows.append(episode_row)
            policy_episode_rows.append(episode_row)
            episode_metrics.append(_episode_metrics(episode_path, summary))
            attention_rows.extend(
                row for row in returned_rows[:-1] if isinstance(row, Mapping) and "layer" in row
            )
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
        _write_json(
            scale_root / "diagnostics" / "frozen_policy_identity.json",
            _jsonable(
                {
                    "policy": policy,
                    "checkpoint_path": str(checkpoint_paths[policy].resolve()),
                    "checkpoint_sha256_before": checkpoint_hashes[policy],
                    "checkpoint_sha256_after": _sha256(checkpoint_paths[policy]),
                    "loaded_policy_hash": policy_hashes.get("loaded_policy_hash"),
                    "after_evaluation_policy_hash": _state_dict_hash(trainer.policy.state_dict()),
                    "policy_updates_performed": False,
                }
            ),
        )
        manifest = _write_scale_manifest(scale_root=scale_root, config=config, episodes=episode_summaries)
        manifest.update(
            {
                "dataset_id": "single_uav_c_vs_d_terminal_velocity_vector_diagnostic",
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "policy_role": policy,
                "checkpoint": {
                    "path": str(checkpoint_paths[policy].resolve()),
                    "sha256_before": checkpoint_hashes[policy],
                    "sha256_after": _sha256(checkpoint_paths[policy]),
                    "unchanged": checkpoint_hashes[policy] == _sha256(checkpoint_paths[policy]),
                },
                "paired_evaluation_contract": {
                    "same_normal_far_distance_scenarios": True,
                    "curriculum_used": False,
                    "policy_updates_performed": False,
                    "terminal_band": "1 < post-action estimated distance <= 3 m",
                },
            }
        )
        _write_json(scale_root / "manifest.json", _jsonable(manifest))
        validation = validate_episode_benchmark(scale_root)
        policy_summary = velocity_vector_policy_summary(
            all_episode_rows,
            all_step_rows,
            policy=policy,
        )
        policy_reports[policy] = {
            "label": POLICY_LABELS[policy],
            "checkpoint": {
                "path": str(checkpoint_paths[policy].resolve()),
                "sha256": checkpoint_hashes[policy],
                "unchanged": checkpoint_hashes[policy] == _sha256(checkpoint_paths[policy]),
                "policy_hash_unchanged": policy_hashes.get("loaded_policy_hash")
                == _state_dict_hash(trainer.policy.state_dict()),
            },
            "episodes": len(policy_episode_rows),
            "schema_validation": validation,
            "summary": policy_summary,
        }

    _write_rows_csv(logs_root / "step_metrics.csv", all_step_rows)
    _write_rows_csv(logs_root / "episode_metrics.csv", all_episode_rows)
    policy_summaries = {
        policy: policy_reports[policy]["summary"] for policy in POLICY_NAMES
    }
    hypothesis = assess_vector_hypothesis(policy_summaries)
    plot_paths = _plot_velocity_vector_diagnostic(
        step_rows=all_step_rows,
        output_dir=plots_root,
    )
    scenario_manifest = {
        "episodes": int(episodes),
        "scenario_hashes": [item["scenario_hash"] for item in paired_scenarios],
        "same_scenario_passed_to_both_policies": True,
        "curriculum_used": False,
        "split": "test",
    }
    _write_json(logs_root / "paired_scenario_manifest.json", _jsonable(scenario_manifest))
    summary = {
        "dataset_id": "single_uav_c_vs_d_terminal_velocity_vector_diagnostic",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "uav_count": 1,
        "episodes_per_policy": int(episodes),
        "policies": {
            policy: {
                "label": POLICY_LABELS[policy],
                "checkpoint_path": str(checkpoint_paths[policy].resolve()),
                "checkpoint_sha256": checkpoint_hashes[policy],
                "summary": policy_reports[policy]["summary"],
                "schema_validation": policy_reports[policy]["schema_validation"],
            }
            for policy in POLICY_NAMES
        },
        "terminal_band_contract": {
            "rule": "1 < post-action estimated goal distance <= 3 m",
            "target_vector": "pre-action estimated goal direction multiplied by target speed at pre-action estimated distance",
            "velocity_vector_error": "norm(current estimated velocity - frozen target velocity vector)",
        },
        "paired_scenario_contract": scenario_manifest,
        "hypothesis_assessment": hypothesis,
        "plots": plot_paths,
        "schema_validation_ok": all(
            bool(policy_reports[policy]["schema_validation"].get("ok", False))
            for policy in POLICY_NAMES
        ),
        "read_only_ok": all(
            bool(policy_reports[policy]["checkpoint"]["unchanged"])
            and bool(policy_reports[policy]["checkpoint"]["policy_hash_unchanged"])
            for policy in POLICY_NAMES
        ),
    }
    _write_json(output_root / "manifest.json", _jsonable(summary))
    _write_json(logs_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "POLICY_NAMES",
    "POLICY_LABELS",
    "velocity_vector_step_rows",
    "velocity_vector_episode_summary",
    "velocity_vector_policy_summary",
    "assess_vector_hypothesis",
    "evaluate_c_vs_d_terminal_velocity_vector_diagnostic",
]
