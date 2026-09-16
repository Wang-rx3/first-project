"""Cross-episode diagnostics for single-UAV navigation failure analysis.

The analysis is intentionally downstream of the canonical episode logs.  It
does not alter the simulator or the policy; it quantifies how often a policy
reaches the near-goal and stable-goal regions and whether its speed decreases
as the estimated goal distance decreases.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .plotting import _number, _read_csv, _timestamp
from ..simulation.motion_model import (
    BRAKING_DIAGNOSTIC_TOLERANCE_MPS,
    braking_distance_target_speed,
)


SPLITS = ("train", "validation", "test")
SPLIT_COLORS = {
    "train": "#2563eb",
    "validation": "#f59e0b",
    "test": "#dc2626",
}


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.max(finite)) if finite.size else float("nan")


def _distance_target_speed(
    distance_m: float,
    *,
    goal_threshold_m: float,
    near_goal_radius_m: float,
    stable_speed_mps: float,
    far_target_speed_mps: float,
    braking_acceleration_mps2: float = 2.0,
    braking_margin_m: float = 2.0,
    max_speed_mps: float = 12.0,
) -> float:
    """Mirror the environment's distance-dependent target-speed schedule."""

    distance = float(distance_m)
    if distance <= goal_threshold_m:
        return float(stable_speed_mps)
    if distance >= near_goal_radius_m or near_goal_radius_m <= goal_threshold_m:
        return float(far_target_speed_mps)
    fraction = (distance - goal_threshold_m) / (
        near_goal_radius_m - goal_threshold_m
    )
    return float(
        stable_speed_mps
        + fraction * (far_target_speed_mps - stable_speed_mps)
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _trajectory_speed_by_timestamp(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> dict[float, float]:
    values: dict[float, list[float]] = {}
    for row in trajectory_rows:
        velocity = np.asarray(
            [_number(row.get(axis)) for axis in ("vx", "vy", "vz")],
            dtype=float,
        )
        speed = float(np.linalg.norm(velocity))
        if np.isfinite(speed):
            values.setdefault(_timestamp(row), []).append(speed)
    return {timestamp: _finite_mean(speeds) for timestamp, speeds in values.items()}


def _action_response_metrics(
    action_rows: Sequence[Mapping[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
    *,
    near_goal_radius_m: float,
) -> dict[str, Any]:
    """Summarize the recorded action-to-motion braking response.

    Braking is defined from the executed action, not inferred from a plotted
    trajectory: ``braking_action`` is the negative projection of applied
    ``delta_v`` onto the pre-action velocity.  Rates use action samples as
    their denominator and near-goal rates use only action samples whose exact
    reward timestamp is inside the near-goal radius.
    """

    reward_by_timestamp = {
        _timestamp(row): row
        for row in reward_rows
        if np.isfinite(_timestamp(row))
    }
    records: list[dict[str, float]] = []
    for row in action_rows:
        timestamp = _timestamp(row)
        reward = reward_by_timestamp.get(timestamp, {})
        record = {
            "pre_speed_mps": _number(row.get("pre_speed_mps")),
            "post_speed_mps": _number(row.get("post_speed_mps")),
            "speed_reduction_mps": _number(row.get("speed_reduction_mps")),
            "applied_longitudinal_delta_v_mps": _number(
                row.get("applied_longitudinal_delta_v_mps")
            ),
            "commanded_braking_delta_v_mps": _number(
                row.get("commanded_braking_delta_v_mps")
            ),
            "braking_delta_v_mps": _number(row.get("braking_delta_v_mps")),
            "braking_action": _number(row.get("braking_action")),
            "brake_effective": _number(row.get("brake_effective")),
            "brake_blocked": _number(row.get("brake_blocked")),
            "speed_limit_limited": _number(row.get("speed_limit_limited"), 0.0),
            "distance_m": _number(reward.get("current_goal_distance")),
        }
        required = (
            "pre_speed_mps",
            "post_speed_mps",
            "speed_reduction_mps",
            "applied_longitudinal_delta_v_mps",
            "commanded_braking_delta_v_mps",
            "braking_delta_v_mps",
            "braking_action",
            "brake_effective",
            "brake_blocked",
        )
        if all(np.isfinite(record[key]) for key in required):
            record["near_goal"] = float(
                np.isfinite(record["distance_m"])
                and record["distance_m"] <= near_goal_radius_m
            )
            record["commanded_braking_action"] = float(
                record["commanded_braking_delta_v_mps"]
                > BRAKING_DIAGNOSTIC_TOLERANCE_MPS
            )
            records.append(record)

    def _subset(prefix: str, selected: Sequence[Mapping[str, float]]) -> dict[str, Any]:
        action_count = len(selected)
        commanded_count = sum(
            int(row["commanded_braking_action"] > 0.5) for row in selected
        )
        braking_count = sum(int(row["braking_action"] > 0.5) for row in selected)
        effective_count = sum(int(row["brake_effective"] > 0.5) for row in selected)
        blocked_count = sum(int(row["brake_blocked"] > 0.5) for row in selected)
        speed_limit_count = sum(
            int(row["speed_limit_limited"] > 0.5) for row in selected
        )
        braking_speed_reductions = [
            row["speed_reduction_mps"]
            for row in selected
            if row["braking_action"] > 0.5
        ]
        applied_braking = [
            row["braking_delta_v_mps"]
            for row in selected
            if row["braking_action"] > 0.5
        ]
        return {
            f"{prefix}action_sample_count": action_count,
            f"{prefix}commanded_braking_action_count": commanded_count,
            f"{prefix}braking_action_count": braking_count,
            f"{prefix}effective_braking_action_count": effective_count,
            f"{prefix}blocked_braking_action_count": blocked_count,
            f"{prefix}speed_limit_limited_count": speed_limit_count,
            f"{prefix}braking_action_fraction": (
                braking_count / action_count if action_count else float("nan")
            ),
            f"{prefix}effective_braking_given_action_fraction": (
                effective_count / braking_count if braking_count else float("nan")
            ),
            f"{prefix}blocked_braking_given_command_fraction": (
                blocked_count / commanded_count if commanded_count else float("nan")
            ),
            f"{prefix}speed_limit_limited_fraction": (
                speed_limit_count / action_count if action_count else float("nan")
            ),
            f"{prefix}mean_applied_braking_delta_v_mps": _finite_mean(applied_braking),
            f"{prefix}mean_speed_reduction_when_braking_mps": _finite_mean(
                braking_speed_reductions
            ),
        }

    near_records = [row for row in records if row["near_goal"] > 0.5]
    return {
        **_subset("", records),
        **_subset("near_goal_", near_records),
    }


def _episode_rows(
    scale_path: Path,
    *,
    goal_threshold_m: float,
    near_goal_radius_m: float,
    speed_threshold_mps: float,
    speed_control_mode: str,
    target_speed_far_mps: float,
    braking_acceleration_mps2: float = 2.0,
    braking_margin_m: float = 2.0,
    braking_distance_speed_profile: str = "legacy",
    max_speed_mps: float = 12.0,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    episode_rows: list[dict[str, Any]] = []
    speed_distance_rows: list[dict[str, float]] = []
    for split in SPLITS:
        split_path = scale_path / split
        if not split_path.is_dir():
            continue
        for episode_path in sorted(split_path.glob("episode_*")):
            if not episode_path.is_dir():
                continue
            rewards = _read_csv(episode_path / "rewards.csv")
            if not rewards:
                continue
            scenario_path = episode_path / "scenario.json"
            try:
                scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                scenario = {}
            trajectory_speed = _trajectory_speed_by_timestamp(
                _read_csv(episode_path / "trajectory.csv")
            )
            action_metrics = _action_response_metrics(
                _read_csv(episode_path / "actions.csv")
                if (episode_path / "actions.csv").is_file()
                else [],
                rewards,
                near_goal_radius_m=near_goal_radius_m,
            )
            distances: list[float] = []
            speeds: list[float] = []
            target_speeds: list[float] = []
            speed_excesses: list[float] = []
            stable_flags: list[bool] = []
            stable_step_values: list[float] = []
            stable_hold_step_values: list[float] = []
            for row in rewards:
                distance = _number(row.get("current_goal_distance"))
                speed = _number(row.get("single_uav_speed_mps"))
                if not np.isfinite(speed):
                    speed = trajectory_speed.get(_timestamp(row), float("nan"))
                target_speed = _number(row.get("single_uav_target_speed_mps"))
                if not np.isfinite(target_speed):
                    target_speed = (
                        (
                            _distance_target_speed(
                                distance,
                                goal_threshold_m=goal_threshold_m,
                                near_goal_radius_m=near_goal_radius_m,
                                stable_speed_mps=speed_threshold_mps,
                                far_target_speed_mps=target_speed_far_mps,
                            )
                            if speed_control_mode == "distance_target_speed"
                            else braking_distance_target_speed(
                                distance,
                                goal_threshold_m=goal_threshold_m,
                                near_goal_radius_m=near_goal_radius_m,
                                stable_speed_mps=speed_threshold_mps,
                                braking_acceleration_mps2=braking_acceleration_mps2,
                                braking_margin_m=braking_margin_m,
                                far_target_speed_mps=target_speed_far_mps,
                                max_speed_mps=max_speed_mps,
                                speed_profile=braking_distance_speed_profile,
                            )
                        )
                        if speed_control_mode
                        in {"distance_target_speed", "braking_distance"}
                        and np.isfinite(distance)
                        else 0.0
                    )
                speed_excess = _number(row.get("single_uav_speed_excess_mps"))
                if not np.isfinite(speed_excess):
                    speed_excess = (
                        max(0.0, speed - target_speed)
                        if (
                            speed_control_mode
                            in {"distance_target_speed", "braking_distance"}
                            and np.isfinite(distance)
                            and (
                                speed_control_mode == "braking_distance"
                                or distance <= near_goal_radius_m
                            )
                            and np.isfinite(speed)
                        )
                        else (
                            speed
                            if (
                                speed_control_mode == "fixed_near_goal"
                                and np.isfinite(distance)
                                and distance <= near_goal_radius_m
                                and np.isfinite(speed)
                            )
                            else 0.0
                        )
                    )
                if np.isfinite(distance):
                    distances.append(distance)
                    speeds.append(speed)
                    target_speeds.append(target_speed)
                    speed_excesses.append(speed_excess)
                    if np.isfinite(speed):
                        speed_distance_rows.append(
                            {
                                "split_index": float(SPLITS.index(split)),
                                "distance_m": float(distance),
                                "speed_mps": float(speed),
                            }
                        )
                stable_value = _number(row.get("single_uav_goal_stable"))
                stable_steps = _number(row.get("single_uav_stable_steps"))
                stable_hold_steps = _number(row.get("single_uav_stable_hold_steps"))
                if np.isfinite(stable_steps):
                    stable_step_values.append(stable_steps)
                if np.isfinite(stable_hold_steps):
                    stable_hold_step_values.append(stable_hold_steps)
                if np.isfinite(stable_value):
                    stable_flags.append(stable_value > 0.5)
                elif np.isfinite(distance) and np.isfinite(speed):
                    stable_flags.append(
                        distance <= goal_threshold_m and speed <= speed_threshold_mps
                    )

            if not distances:
                continue
            near_indices = [
                index for index, distance in enumerate(distances) if distance <= near_goal_radius_m
            ]
            goal_indices = [
                index for index, distance in enumerate(distances) if distance <= goal_threshold_m
            ]
            near_speeds = [
                speeds[index]
                for index in near_indices
                if index < len(speeds) and np.isfinite(speeds[index])
            ]
            near_target_speeds = [
                target_speeds[index]
                for index in near_indices
                if index < len(target_speeds) and np.isfinite(target_speeds[index])
            ]
            near_speed_excesses = [
                speed_excesses[index]
                for index in near_indices
                if index < len(speed_excesses) and np.isfinite(speed_excesses[index])
            ]
            # ``speeds`` is aligned with finite-distance rows in the canonical
            # reward log; a trajectory fallback remains timestamp-aligned in
            # the same loop above.
            maximum_stable_steps = _finite_max(stable_step_values)
            episode_rows.append(
                {
                    "split": split,
                    "episode_id": str(episode_path.name),
                    "seed": (
                        scenario.get("seed", "")
                        if isinstance(scenario, Mapping)
                        else ""
                    ),
                    "sample_count": len(distances),
                    "minimum_goal_distance_m": float(min(distances)),
                    "final_goal_distance_m": float(distances[-1]),
                    "entered_near_goal": int(bool(near_indices)),
                    "entered_goal_radius": int(bool(goal_indices)),
                    "stable_goal": int(any(stable_flags)),
                    "maximum_stable_steps": maximum_stable_steps,
                    "maximum_consecutive_goal_hold_steps": maximum_stable_steps,
                    "required_stable_hold_steps": (
                        float(max(stable_hold_step_values))
                        if stable_hold_step_values
                        else float("nan")
                    ),
                    "speed_at_near_goal_entry_mps": (
                        float(speeds[near_indices[0]])
                        if near_indices and near_indices[0] < len(speeds)
                        else float("nan")
                    ),
                    "mean_speed_within_near_goal_mps": _finite_mean(near_speeds),
                    "minimum_speed_within_near_goal_mps": (
                        float(min(near_speeds)) if near_speeds else float("nan")
                    ),
                    "mean_target_speed_within_near_goal_mps": _finite_mean(
                        near_target_speeds
                    ),
                    "mean_speed_excess_within_near_goal_mps": _finite_mean(
                        near_speed_excesses
                    ),
                    "overspeed_fraction_within_near_goal": (
                        float(
                            np.mean(
                                np.asarray(near_speed_excesses, dtype=float) > 1e-9
                            )
                        )
                        if near_speed_excesses
                        else float("nan")
                    ),
                    "termination_reason": (
                        scenario.get("termination_reason", "")
                        if isinstance(scenario, Mapping)
                        else ""
                    ),
                    **action_metrics,
                }
            )
    return episode_rows, speed_distance_rows


def _split_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    speed_threshold_mps: float,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        subset = [row for row in rows if str(row.get("split", "")) == split]
        count = len(subset)
        entering_near = sum(int(row.get("entered_near_goal", 0)) for row in subset)
        entering_goal = sum(int(row.get("entered_goal_radius", 0)) for row in subset)
        stable = sum(int(row.get("stable_goal", 0)) for row in subset)
        maximum_consecutive_goal_hold_steps = _finite_max(
            [
                _number(
                    row.get(
                        "maximum_consecutive_goal_hold_steps",
                        row.get("maximum_stable_steps"),
                    )
                )
                for row in subset
            ]
        )
        maximum_consecutive_goal_hold_values = [
            _number(
                row.get(
                    "maximum_consecutive_goal_hold_steps",
                    row.get("maximum_stable_steps"),
                )
            )
            for row in subset
        ]
        hold_threshold_counts = {
            threshold: sum(
                int(np.isfinite(value) and value >= threshold)
                for value in maximum_consecutive_goal_hold_values
            )
            for threshold in (1, 3, 5)
        }
        near_entry_speeds = [
            _number(row.get("speed_at_near_goal_entry_mps"))
            for row in subset
        ]
        near_mean_speeds = [
            _number(row.get("mean_speed_within_near_goal_mps"))
            for row in subset
        ]
        near_target_speeds = [
            _number(row.get("mean_target_speed_within_near_goal_mps"))
            for row in subset
        ]
        near_speed_excesses = [
            _number(row.get("mean_speed_excess_within_near_goal_mps"))
            for row in subset
        ]
        overspeed_fractions = [
            _number(row.get("overspeed_fraction_within_near_goal"))
            for row in subset
        ]
        action_sample_count = sum(
            int(_number(row.get("action_sample_count"), 0.0)) for row in subset
        )
        commanded_braking_count = sum(
            int(_number(row.get("commanded_braking_action_count"), 0.0))
            for row in subset
        )
        braking_action_count = sum(
            int(_number(row.get("braking_action_count"), 0.0)) for row in subset
        )
        effective_braking_count = sum(
            int(_number(row.get("effective_braking_action_count"), 0.0))
            for row in subset
        )
        blocked_braking_count = sum(
            int(_number(row.get("blocked_braking_action_count"), 0.0))
            for row in subset
        )
        speed_limit_limited_count = sum(
            int(_number(row.get("speed_limit_limited_count"), 0.0))
            for row in subset
        )
        near_action_sample_count = sum(
            int(_number(row.get("near_goal_action_sample_count"), 0.0))
            for row in subset
        )
        near_commanded_braking_count = sum(
            int(_number(row.get("near_goal_commanded_braking_action_count"), 0.0))
            for row in subset
        )
        near_braking_action_count = sum(
            int(_number(row.get("near_goal_braking_action_count"), 0.0))
            for row in subset
        )
        near_effective_braking_count = sum(
            int(_number(row.get("near_goal_effective_braking_action_count"), 0.0))
            for row in subset
        )
        near_blocked_braking_count = sum(
            int(_number(row.get("near_goal_blocked_braking_action_count"), 0.0))
            for row in subset
        )
        near_speed_limit_limited_count = sum(
            int(_number(row.get("near_goal_speed_limit_limited_count"), 0.0))
            for row in subset
        )
        near_applied_braking = [
            _number(row.get("near_goal_mean_applied_braking_delta_v_mps"))
            for row in subset
        ]
        near_speed_reductions = [
            _number(row.get("near_goal_mean_speed_reduction_when_braking_mps"))
            for row in subset
        ]
        summary[split] = {
            "episode_count": count,
            "minimum_goal_distance_m": min(
                (_number(row.get("minimum_goal_distance_m")) for row in subset),
                default=float("nan"),
            ),
            "entered_near_goal_episodes": entering_near,
            "entered_near_goal_rate": entering_near / count if count else 0.0,
            "entered_goal_radius_episodes": entering_goal,
            "entered_goal_radius_rate": entering_goal / count if count else 0.0,
            "stable_goal_episodes": stable,
            "stable_goal_rate": stable / count if count else 0.0,
            "maximum_stable_steps": maximum_consecutive_goal_hold_steps,
            "maximum_consecutive_goal_hold_steps": maximum_consecutive_goal_hold_steps,
            "mean_maximum_consecutive_goal_hold_steps": _finite_mean(
                maximum_consecutive_goal_hold_values
            ),
            "maximum_consecutive_goal_hold_steps_max": maximum_consecutive_goal_hold_steps,
            "episodes_with_hold_steps_ge_1": hold_threshold_counts[1],
            "episodes_with_hold_steps_ge_1_rate": (
                hold_threshold_counts[1] / count if count else 0.0
            ),
            "episodes_with_hold_steps_ge_3": hold_threshold_counts[3],
            "episodes_with_hold_steps_ge_3_rate": (
                hold_threshold_counts[3] / count if count else 0.0
            ),
            "episodes_with_hold_steps_ge_5": hold_threshold_counts[5],
            "episodes_with_hold_steps_ge_5_rate": (
                hold_threshold_counts[5] / count if count else 0.0
            ),
            "mean_speed_at_near_goal_entry_mps": _finite_mean(near_entry_speeds),
            "mean_speed_within_near_goal_mps": _finite_mean(near_mean_speeds),
            "mean_target_speed_within_near_goal_mps": _finite_mean(
                near_target_speeds
            ),
            "mean_speed_excess_within_near_goal_mps": _finite_mean(
                near_speed_excesses
            ),
            "mean_overspeed_fraction_within_near_goal": _finite_mean(
                overspeed_fractions
            ),
            "action_sample_count": action_sample_count,
            "commanded_braking_action_count": commanded_braking_count,
            "braking_action_count": braking_action_count,
            "effective_braking_action_count": effective_braking_count,
            "blocked_braking_action_count": blocked_braking_count,
            "speed_limit_limited_count": speed_limit_limited_count,
            "braking_action_fraction": (
                braking_action_count / action_sample_count
                if action_sample_count
                else float("nan")
            ),
            "effective_braking_given_action_fraction": (
                effective_braking_count / braking_action_count
                if braking_action_count
                else float("nan")
            ),
            "blocked_braking_given_command_fraction": (
                blocked_braking_count / commanded_braking_count
                if commanded_braking_count
                else float("nan")
            ),
            "speed_limit_limited_fraction": (
                speed_limit_limited_count / action_sample_count
                if action_sample_count
                else float("nan")
            ),
            "near_goal_action_sample_count": near_action_sample_count,
            "near_goal_commanded_braking_action_count": near_commanded_braking_count,
            "near_goal_braking_action_count": near_braking_action_count,
            "near_goal_effective_braking_action_count": near_effective_braking_count,
            "near_goal_blocked_braking_action_count": near_blocked_braking_count,
            "near_goal_speed_limit_limited_count": near_speed_limit_limited_count,
            "near_goal_braking_action_fraction": (
                near_braking_action_count / near_action_sample_count
                if near_action_sample_count
                else float("nan")
            ),
            "near_goal_effective_braking_given_action_fraction": (
                near_effective_braking_count / near_braking_action_count
                if near_braking_action_count
                else float("nan")
            ),
            "near_goal_blocked_braking_given_command_fraction": (
                near_blocked_braking_count / near_commanded_braking_count
                if near_commanded_braking_count
                else float("nan")
            ),
            "near_goal_speed_limit_limited_fraction": (
                near_speed_limit_limited_count / near_action_sample_count
                if near_action_sample_count
                else float("nan")
            ),
            "near_goal_mean_applied_braking_delta_v_mps": _finite_mean(
                near_applied_braking
            ),
            "near_goal_mean_speed_reduction_when_braking_mps": _finite_mean(
                near_speed_reductions
            ),
            "speed_threshold_mps": float(speed_threshold_mps),
        }
    return summary


def _plot_summary(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, Any]],
    *,
    goal_threshold_m: float,
    near_goal_radius_m: float,
    speed_threshold_mps: float,
    speed_control_mode: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    distance_axis = axes[0, 0]
    for split in SPLITS:
        subset = [row for row in rows if str(row.get("split", "")) == split]
        if not subset:
            continue
        distance_axis.plot(
            np.arange(1, len(subset) + 1),
            [_number(row.get("minimum_goal_distance_m")) for row in subset],
            marker=".",
            linewidth=1.0,
            markersize=4,
            color=SPLIT_COLORS[split],
            label=split,
        )
    distance_axis.axhline(
        goal_threshold_m,
        color="#dc2626",
        linestyle="--",
        label=f"{goal_threshold_m:g} m goal radius",
    )
    distance_axis.axhline(
        near_goal_radius_m,
        color="#f59e0b",
        linestyle=":",
        label=f"{near_goal_radius_m:g} m near-goal radius",
    )
    distance_axis.set_title("Minimum estimated goal distance")
    distance_axis.set_xlabel("Episode index within split")
    distance_axis.set_ylabel("Minimum distance (m)")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend(fontsize=8)

    rate_axis = axes[0, 1]
    x = np.arange(len(SPLITS), dtype=float)
    width = 0.24
    for offset, key, label, color in (
        (-width, "entered_near_goal_rate", f"Entered {near_goal_radius_m:g} m", "#f59e0b"),
        (0.0, "entered_goal_radius_rate", f"Entered {goal_threshold_m:g} m", "#dc2626"),
        (width, "stable_goal_rate", "Stable arrival", "#16a34a"),
    ):
        rate_axis.bar(
            x + offset,
            [float(summary.get(split, {}).get(key, 0.0)) for split in SPLITS],
            width,
            label=label,
            color=color,
        )
    rate_axis.set_xticks(x, SPLITS)
    rate_axis.set_ylim(0.0, 1.0)
    rate_axis.set_ylabel("Episode rate")
    rate_axis.set_title("Goal-region entry rates")
    rate_axis.grid(axis="y", alpha=0.25)
    rate_axis.legend(fontsize=8)

    speed_axis = axes[1, 0]
    speed_axis.plot(
        x,
        [
            _number(summary.get(split, {}).get("mean_speed_at_near_goal_entry_mps"))
            for split in SPLITS
        ],
        marker="o",
        color="#7c3aed",
        label=f"Speed at {near_goal_radius_m:g} m entry",
    )
    speed_axis.plot(
        x,
        [
            _number(summary.get(split, {}).get("mean_speed_within_near_goal_mps"))
            for split in SPLITS
        ],
        marker="s",
        color="#0891b2",
        label=f"Mean speed within {near_goal_radius_m:g} m",
    )
    if speed_control_mode in {"distance_target_speed", "braking_distance"}:
        speed_axis.plot(
            x,
            [
                _number(
                    summary.get(split, {}).get(
                        "mean_target_speed_within_near_goal_mps"
                    )
                )
                for split in SPLITS
            ],
            marker="^",
            color="#111827",
            label="Mean distance target speed",
        )
    speed_axis.axhline(
        speed_threshold_mps,
        color="#166534",
        linestyle="--",
        label="Stable speed limit",
    )
    speed_axis.set_xticks(x, SPLITS)
    speed_axis.set_ylabel("Speed (m/s)")
    speed_axis.set_title(f"Speed after entering the {near_goal_radius_m:g} m region")
    speed_axis.grid(alpha=0.25)
    speed_axis.legend(fontsize=8)

    relation_axis = axes[1, 1]
    for split in SPLITS:
        subset = [row for row in rows if str(row.get("split", "")) == split]
        distances = np.asarray(
            [_number(row.get("minimum_goal_distance_m")) for row in subset],
            dtype=float,
        )
        speeds = np.asarray(
            [_number(row.get("mean_speed_within_near_goal_mps")) for row in subset],
            dtype=float,
        )
        valid = np.isfinite(distances) & np.isfinite(speeds)
        if valid.any():
            relation_axis.scatter(
                distances[valid],
                speeds[valid],
                alpha=0.75,
                s=24,
                color=SPLIT_COLORS[split],
                label=f"{split} near-goal episodes",
            )
    relation_axis.axvline(
        near_goal_radius_m,
        color="#f59e0b",
        linestyle=":",
        label=f"{near_goal_radius_m:g} m boundary",
    )
    relation_axis.axhline(
        speed_threshold_mps,
        color="#166534",
        linestyle="--",
        label=f"{speed_threshold_mps:g} m/s limit",
    )
    relation_axis.set_xlabel("Minimum estimated goal distance (m)")
    relation_axis.set_ylabel(f"Mean speed within {near_goal_radius_m:g} m (m/s)")
    relation_axis.set_title("Near-goal distance versus speed")
    relation_axis.grid(alpha=0.25)
    relation_axis.legend(fontsize=8)
    figure.suptitle("Single-UAV navigation failure analysis")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_speed_distance(
    speed_distance_rows: Sequence[Mapping[str, float]],
    *,
    goal_threshold_m: float,
    near_goal_radius_m: float,
    speed_threshold_mps: float,
    speed_control_mode: str,
    target_speed_far_mps: float,
    braking_acceleration_mps2: float,
    braking_margin_m: float,
    braking_distance_speed_profile: str,
    max_speed_mps: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, zoom in zip(axes, (False, True)):
        for split_index, split in enumerate(SPLITS):
            subset = [
                row
                for row in speed_distance_rows
                if int(row.get("split_index", -1)) == split_index
                and (not zoom or row.get("distance_m", float("nan")) <= near_goal_radius_m * 2.0)
            ]
            if subset:
                axis.scatter(
                    [row["distance_m"] for row in subset],
                    [row["speed_mps"] for row in subset],
                    s=9,
                    alpha=0.22 if not zoom else 0.42,
                    color=SPLIT_COLORS[split],
                    label=split,
                )
        if speed_control_mode in {"distance_target_speed", "braking_distance"}:
            maximum_distance = near_goal_radius_m * 2.0
            if not zoom:
                maximum_distance = max(
                    near_goal_radius_m * 1.25,
                    max(
                        (
                            float(row.get("distance_m", float("nan")))
                            for row in speed_distance_rows
                            if np.isfinite(float(row.get("distance_m", float("nan"))))
                        ),
                        default=near_goal_radius_m,
                    ),
                )
            target_distances = np.linspace(0.0, maximum_distance, 300)
            target_speeds = [
                (
                    _distance_target_speed(
                        distance,
                        goal_threshold_m=goal_threshold_m,
                        near_goal_radius_m=near_goal_radius_m,
                        stable_speed_mps=speed_threshold_mps,
                        far_target_speed_mps=target_speed_far_mps,
                    )
                    if speed_control_mode == "distance_target_speed"
                    else braking_distance_target_speed(
                        distance,
                        goal_threshold_m=goal_threshold_m,
                        near_goal_radius_m=near_goal_radius_m,
                        stable_speed_mps=speed_threshold_mps,
                        braking_acceleration_mps2=braking_acceleration_mps2,
                        braking_margin_m=braking_margin_m,
                        far_target_speed_mps=target_speed_far_mps,
                        max_speed_mps=max_speed_mps,
                        speed_profile=braking_distance_speed_profile,
                    )
                )
                for distance in target_distances
            ]
            axis.plot(
                target_distances,
                target_speeds,
                color="#111827",
                linewidth=1.6,
                label=(
                    "Distance target speed"
                    if speed_control_mode == "distance_target_speed"
                    else "Stopping-distance target speed"
                ),
            )
        axis.axvline(
            goal_threshold_m,
            color="#dc2626",
            linestyle="--",
            label=f"{goal_threshold_m:g} m goal radius",
        )
        axis.axvline(
            near_goal_radius_m,
            color="#f59e0b",
            linestyle=":",
            label=f"{near_goal_radius_m:g} m near-goal radius",
        )
        axis.axhline(
            speed_threshold_mps,
            color="#166534",
            linestyle="--",
            label=f"{speed_threshold_mps:g} m/s stable limit",
        )
        axis.set_xlabel("Estimated goal distance (m)")
        axis.set_ylabel("Actual speed used by reward (m/s)")
        axis.set_title("Near-goal zoom" if zoom else "All recorded samples")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Speed-distance relation v(d) | diagnostic only")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_braking_summary(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    near_goal_radius_m: float,
    output_path: Path,
) -> None:
    """Plot whether executed actions brake and whether braking reduces speed."""

    import matplotlib.pyplot as plt

    x = np.arange(len(SPLITS), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.24
    rate_specs = (
        (
            -width,
            "near_goal_braking_action_fraction",
            "Braking action / action sample",
            "#dc2626",
        ),
        (
            0.0,
            "near_goal_effective_braking_given_action_fraction",
            "Effective braking / braking action",
            "#16a34a",
        ),
        (
            width,
            "near_goal_blocked_braking_given_command_fraction",
            "Blocked braking / command",
            "#f59e0b",
        ),
    )
    for offset, key, label, color in rate_specs:
        values = np.asarray(
            [_number(summary.get(split, {}).get(key)) for split in SPLITS], dtype=float
        )
        axes[0].bar(
            x + offset,
            np.nan_to_num(values, nan=0.0),
            width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, SPLITS)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Rate")
    axes[0].set_title(f"Braking diagnostics inside {near_goal_radius_m:g} m")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    response_specs = (
        (
            -width / 2,
            "near_goal_mean_applied_braking_delta_v_mps",
            "Mean applied braking Δv",
            "#7c3aed",
        ),
        (
            width / 2,
            "near_goal_mean_speed_reduction_when_braking_mps",
            "Mean speed reduction",
            "#2563eb",
        ),
    )
    for offset, key, label, color in response_specs:
        values = np.asarray(
            [_number(summary.get(split, {}).get(key)) for split in SPLITS], dtype=float
        )
        axes[1].bar(
            x + offset,
            np.nan_to_num(values, nan=0.0),
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, SPLITS)
    axes[1].set_ylabel("m/s")
    axes[1].set_title("Executed braking response")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    sample_text = "; ".join(
        f"{split}: n={int(_number(summary.get(split, {}).get('near_goal_action_sample_count'), 0.0))}"
        for split in SPLITS
    )
    figure.suptitle(
        "Single-UAV action-space braking audit | diagnostic only\n" + sample_text
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_single_uav_navigation(
    scale_path: str | Path,
    *,
    output_dir: str | Path,
    metrics_output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Create cross-episode navigation metrics and speed-distance figures."""

    scale_path = Path(scale_path)
    output_dir = Path(output_dir)
    metrics_output_path = Path(metrics_output_path)
    if not scale_path.is_dir():
        raise FileNotFoundError(f"single-UAV scale directory does not exist: {scale_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"analysis output directory is not empty: {output_dir}; use force=True"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_paths = sorted(scale_path.glob("*/episode_*/scenario.json"))
    reward_config: Mapping[str, Any] = {}
    if scenario_paths:
        try:
            scenario = json.loads(scenario_paths[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            scenario = {}
        if isinstance(scenario, Mapping):
            configured = scenario.get("reward_config", {})
            reward_config = configured if isinstance(configured, Mapping) else {}
    goal_threshold_m = _number(
        reward_config.get("single_uav_goal_threshold_m"), 1.0
    )
    near_goal_radius_m = _number(
        reward_config.get("single_uav_near_goal_radius_m"), 5.0
    )
    speed_threshold_mps = _number(
        reward_config.get("single_uav_goal_speed_threshold_mps"), 0.5
    )
    speed_control_mode = str(
        reward_config.get("single_uav_speed_control_mode", "fixed_near_goal")
    )
    target_speed_far_mps = _number(
        reward_config.get("single_uav_target_speed_far_mps"), 6.0
    )
    braking_acceleration_mps2 = _number(
        reward_config.get("single_uav_braking_acceleration_mps2"), 2.0
    )
    braking_margin_m = _number(
        reward_config.get("single_uav_braking_margin_m"), 2.0
    )
    braking_distance_speed_profile = str(
        reward_config.get("single_uav_braking_distance_speed_profile", "legacy")
    )
    motion_contract = (
        scenario.get("motion_action_contract", {})
        if isinstance(scenario, Mapping)
        else {}
    )
    max_speed_mps = _number(
        motion_contract.get("max_speed_mps")
        if isinstance(motion_contract, Mapping)
        else None,
        12.0,
    )
    stable_hold_steps = _number(
        reward_config.get("single_uav_stable_hold_steps"), 1.0
    )
    terminal_radial_progress_reward_coef = _number(
        reward_config.get("single_uav_terminal_radial_progress_reward_coef"), 0.0
    )
    rows, speed_distance_rows = _episode_rows(
        scale_path,
        goal_threshold_m=goal_threshold_m,
        near_goal_radius_m=near_goal_radius_m,
        speed_threshold_mps=speed_threshold_mps,
        speed_control_mode=speed_control_mode,
        target_speed_far_mps=target_speed_far_mps,
        braking_acceleration_mps2=braking_acceleration_mps2,
        braking_margin_m=braking_margin_m,
        braking_distance_speed_profile=braking_distance_speed_profile,
        max_speed_mps=max_speed_mps,
    )
    if not rows:
        raise ValueError(f"scale directory contains no usable reward episodes: {scale_path}")
    rows = sorted(rows, key=lambda row: (SPLITS.index(str(row["split"])), str(row["episode_id"])))
    _write_csv(metrics_output_path, rows)
    summary = _split_summary(rows, speed_threshold_mps=speed_threshold_mps)
    summary_plot = output_dir / "navigation_failure_analysis.png"
    relation_plot = output_dir / "speed_distance_relation.png"
    braking_plot = output_dir / "braking_action_summary.png"
    _plot_summary(
        rows,
        summary,
        goal_threshold_m=goal_threshold_m,
        near_goal_radius_m=near_goal_radius_m,
        speed_threshold_mps=speed_threshold_mps,
        speed_control_mode=speed_control_mode,
        output_path=summary_plot,
    )
    _plot_speed_distance(
        speed_distance_rows,
        goal_threshold_m=goal_threshold_m,
        near_goal_radius_m=near_goal_radius_m,
        speed_threshold_mps=speed_threshold_mps,
        speed_control_mode=speed_control_mode,
        target_speed_far_mps=target_speed_far_mps,
        braking_acceleration_mps2=braking_acceleration_mps2,
        braking_margin_m=braking_margin_m,
        braking_distance_speed_profile=braking_distance_speed_profile,
        max_speed_mps=max_speed_mps,
        output_path=relation_plot,
    )
    _plot_braking_summary(
        summary,
        near_goal_radius_m=near_goal_radius_m,
        output_path=braking_plot,
    )
    manifest = {
        "figure_type": "single_uav_navigation_failure_analysis",
        "scale_path": str(scale_path.resolve()),
        "uav_count": 1,
        "source": {
            "episode_reward_logs": "scale/{train,validation,test}/episode_*/rewards.csv",
            "episode_scenario_logs": "scale/{train,validation,test}/episode_*/scenario.json",
            "episode_action_logs": "scale/{train,validation,test}/episode_*/actions.csv",
            "actual_speed_fallback": "trajectory.csv vx/vy/vz when rewards.csv lacks single_uav_speed_mps",
        },
        "seed_set": sorted({row.get("seed") for row in rows}, key=str),
        "thresholds": {
            "goal_threshold_m": float(goal_threshold_m),
            "near_goal_radius_m": float(near_goal_radius_m),
            "speed_threshold_mps": float(speed_threshold_mps),
            "speed_control_mode": speed_control_mode,
            "target_speed_far_mps": float(target_speed_far_mps),
            "braking_acceleration_mps2": float(braking_acceleration_mps2),
            "braking_margin_m": float(braking_margin_m),
            "braking_distance_speed_profile": braking_distance_speed_profile,
            "stable_hold_steps": float(stable_hold_steps),
            "terminal_radial_progress_reward_coef": float(
                terminal_radial_progress_reward_coef
            ),
            "terminal_radial_progress_activation_band_m": [
                float(goal_threshold_m),
                float(reward_config.get("single_uav_terminal_braking_radius_m", 3.0)),
            ],
            "max_speed_mps": float(max_speed_mps),
            "braking_tolerance_mps": float(BRAKING_DIAGNOSTIC_TOLERANCE_MPS),
        },
        "denominator": "episode count within each split for goal rates; step samples for v(d); action samples for braking rates; near-goal braking rates condition on action samples with an exact reward timestamp inside the near-goal radius",
        "transformations": {
            "minimum_goal_distance": "minimum finite current_goal_distance per episode",
            "entry_rates": "indicator that any finite step enters the configured radius",
            "maximum_consecutive_goal_hold_steps": "maximum finite consecutive stable-goal counter per episode, aggregated by split; sourced from single_uav_stable_steps",
            "hold_duration_summary": "mean and maximum of per-episode maximum_consecutive_goal_hold_steps, plus episode counts/rates at thresholds >=1, >=3, and >=5; denominator is all episodes in the split",
            "near_goal_speed": "speed at first near-goal sample and mean/min speed over near-goal samples",
            "speed_distance_relation": "raw finite step samples; no smoothing, resampling, or filtering beyond finite-value checks",
            "distance_target_speed": "piecewise-linear target speed from the recorded reward configuration; diagnostic overlay only",
            "braking_distance": "stopping-distance-limited target speed using the recorded conservative deceleration and braking margin; diagnostic overlay only",
            "braking_action_response": "raw actions.csv execution diagnostics; applied longitudinal delta_v < 0 means braking, brake_effective additionally requires post-action speed < pre-action speed, and no smoothing is applied",
            "speed_limit_limited": "reported by the motion model when the requested next speed exceeds max_speed_mps and the executed speed is at the cap",
        },
        "summary": summary,
        "claim_boundary": "single-seed, fixed-scenario navigation diagnostic; it does not establish convergence, generalization, or multi-UAV communication performance",
        "outputs": {
            "metrics_csv": str(metrics_output_path.resolve()),
            "navigation_failure_analysis": str(summary_plot.resolve()),
            "speed_distance_relation": str(relation_plot.resolve()),
            "braking_action_summary": str(braking_plot.resolve()),
        },
    }
    manifest_path = output_dir / "navigation_analysis_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "scale_path": str(scale_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "metrics_csv": str(metrics_output_path.resolve()),
        "plots": manifest["outputs"],
        "manifest": str(manifest_path.resolve()),
        "summary": summary,
    }


__all__ = ["analyze_single_uav_navigation"]
