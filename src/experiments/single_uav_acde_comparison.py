"""Read-only A/C/D/E comparison for the single-UAV terminal experiments.

This module deliberately operates on already-written normal far-distance test
artifacts.  It does not construct a trainer, sample a policy, add a rollout,
or update any parameters.  The four policies are:

* A: original ``coef=0.25`` baseline;
* C: early-smooth speed-control fine-tuning;
* D: early-smooth plus terminal radial-progress ablation;
* E: early-smooth plus terminal target-velocity-vector correction.

All rows use the same half-open distance contract as the terminal-approach
diagnostic: ``8--5 m = [5, 8)``, ``5--3 m = [3, 5)``, and
``3--1 m = [1, 3)``.  The comparison is intentionally downstream of the
canonical episode CSVs so it cannot alter the experiment that produced them.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .single_uav_terminal_approach_diagnostic import (
    APPROACH_SEGMENTS,
    DISTANCE_BINS,
    _canonical_hash,
    _number as _approach_number,
    _read_csv as _approach_read_csv,
    _step_rows_from_episode,
)


POLICY_ORDER: tuple[str, ...] = (
    "baseline",
    "early_smooth",
    "early_smooth_radial_progress",
    "early_smooth_vector_correction",
)

POLICY_LABELS: dict[str, str] = {
    "baseline": "A: coef=0.25 baseline",
    "early_smooth": "C: early-smooth speed-control",
    "early_smooth_radial_progress": "D: early-smooth + radial-progress",
    "early_smooth_vector_correction": "E: early-smooth + vector correction",
}

SEGMENT_ORDER: tuple[str, ...] = ("8_5m", "5_3m", "3_1m")
DISTANCE_BIN_ORDER: tuple[str, ...] = (
    "1_2m",
    "2_3m",
    "3_3p5m",
    "3p5_5m",
    "5_8m",
)

TEST_COLUMNS: tuple[str, ...] = (
    "policy",
    "policy_label",
    "episodes",
    "entered_8m_count",
    "entered_8m_rate",
    "entered_5m_count",
    "entered_5m_rate",
    "entered_3m_count",
    "entered_3m_rate",
    "entered_1m_count",
    "entered_1m_rate",
    "stable_goal_count",
    "stable_goal_rate",
    "minimum_goal_distance_m",
    "mean_speed_at_near_goal_entry_mps",
    "mean_speed_within_near_goal_mps",
    "mean_target_speed_within_near_goal_mps",
    "mean_speed_excess_within_near_goal_mps",
    "mean_maximum_consecutive_goal_hold_steps",
    "maximum_consecutive_goal_hold_steps_max",
    "episodes_with_hold_steps_ge_1",
    "episodes_with_hold_steps_ge_3",
    "episodes_with_hold_steps_ge_5",
    "hold_steps_ge_1_rate",
    "hold_steps_ge_3_rate",
    "hold_steps_ge_5_rate",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
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
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    return _approach_number(row, key, default)


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


def _minimum(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return float(np.min(numbers)) if numbers.size else float("nan")


def _maximum(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite(values)
    return float(np.max(numbers)) if numbers.size else float(default)


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _timestamp(row: Mapping[str, Any]) -> float:
    return round(_number(row, "timestamp", float("nan")), 6)


def _scenario_signature(scenario: Mapping[str, Any]) -> str:
    """Hash only the generated scenario, excluding run-specific metadata."""

    generated = scenario.get("scenario_generation")
    payload: Mapping[str, Any] = (
        generated if isinstance(generated, Mapping) else scenario
    )
    return _canonical_hash(payload)


def _terminal_vector_fields(
    step_rows: list[dict[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Attach E's vector-correction fields to shared diagnostic step rows."""

    rewards_by_time = {_timestamp(row): row for row in reward_rows}
    for row in step_rows:
        reward = rewards_by_time.get(round(_number(row, "timestamp_s"), 6), {})
        row.update(
            {
                "terminal_vector_active": _number(
                    reward, "single_uav_terminal_target_velocity_vector_active", 0.0
                ),
                "terminal_vector_error_pre_mps": _number(
                    reward, "single_uav_terminal_target_velocity_vector_error_pre_mps"
                ),
                "terminal_vector_error_post_mps": _number(
                    reward, "single_uav_terminal_target_velocity_vector_error_post_mps"
                ),
                "terminal_vector_error_reduction_mps": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_error_reduction_mps",
                ),
                "terminal_vector_signal": _number(
                    reward, "single_uav_terminal_target_velocity_vector_signal"
                ),
                "terminal_vector_reward": _number(
                    reward, "single_uav_terminal_target_velocity_vector_reward", 0.0
                ),
                "terminal_vector_coef": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_reward_coef",
                    0.0,
                ),
            }
        )


def _episode_summary(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_index: int,
    scenario_signature: str,
) -> dict[str, Any]:
    distances = _finite(row.get("goal_distance_m") for row in step_rows)
    minimum = float(np.min(distances)) if distances.size else float("nan")
    hold_values = _finite(row.get("stable_steps") for row in step_rows)
    maximum_hold = int(round(_maximum(hold_values, default=0.0)))
    stable = int(any(int(_number(row, "goal_stable", 0.0)) > 0 for row in step_rows))
    near = [row for row in step_rows if _number(row, "goal_distance_m") <= 8.0]
    near_entry = near[0] if near else {}
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episode_index": int(episode_index),
        "episode_id": str(step_rows[0].get("episode_id", "")) if step_rows else "",
        "scenario_signature": scenario_signature,
        "step_count": len(step_rows),
        "entered_8m": int(np.isfinite(minimum) and minimum <= 8.0),
        "entered_5m": int(np.isfinite(minimum) and minimum <= 5.0),
        "entered_3m": int(np.isfinite(minimum) and minimum <= 3.0),
        "entered_1m": int(np.isfinite(minimum) and minimum <= 1.0),
        "stable_goal": stable,
        "minimum_goal_distance_m": minimum,
        "final_goal_distance_m": (
            _number(step_rows[-1], "goal_distance_m") if step_rows else float("nan")
        ),
        "speed_at_near_goal_entry_mps": _number(
            near_entry, "total_speed_mps"
        ),
        "mean_speed_within_near_goal_mps": _mean(
            row.get("total_speed_mps") for row in near
        ),
        "mean_target_speed_within_near_goal_mps": _mean(
            row.get("target_speed_mps") for row in near
        ),
        "mean_speed_excess_within_near_goal_mps": _mean(
            row.get("speed_excess_mps") for row in near
        ),
        "maximum_consecutive_goal_hold_steps": maximum_hold,
        "hold_steps_ge_1": int(maximum_hold >= 1),
        "hold_steps_ge_3": int(maximum_hold >= 3),
        "hold_steps_ge_5": int(maximum_hold >= 5),
    }


def _load_policy_test(
    root: Path,
    *,
    policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    scale_root = root / "scale_1"
    test_root = scale_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {test_root}")
    all_steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    signatures: dict[int, str] = {}
    episode_paths = sorted(path for path in test_root.glob("episode_*") if path.is_dir())
    for episode_index, episode_path in enumerate(episode_paths, start=1):
        scenario_path = episode_path / "scenario.json"
        try:
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid scenario file: {scenario_path}") from exc
        signature = _scenario_signature(scenario)
        signatures[episode_index] = signature
        reward_rows = _read_csv(episode_path / "rewards.csv")
        step_rows = _step_rows_from_episode(
            episode_path,
            # The shared helper uses its own label lookup.  We override the
            # policy fields immediately below, keeping its metric contract.
            policy="baseline",
            episode_index=episode_index,
            seed=int(_number(scenario, "seed", episode_index)),
            scenario_hash=signature,
            scenario=scenario,
        )
        for row in step_rows:
            row["policy"] = policy
            row["policy_label"] = POLICY_LABELS[policy]
        _terminal_vector_fields(step_rows, reward_rows)
        all_steps.extend(step_rows)
        episodes.append(
            _episode_summary(
                step_rows,
                policy=policy,
                episode_index=episode_index,
                scenario_signature=signature,
            )
        )
    return episodes, all_steps, signatures


def _test_summary_row(
    episodes: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    count = len(episodes)
    row: dict[str, Any] = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": count,
    }
    for threshold, suffix in ((8, "8m"), (5, "5m"), (3, "3m"), (1, "1m")):
        key = f"entered_{suffix}"
        entered = sum(int(item.get(key, 0)) for item in episodes)
        row[f"entered_{suffix}_count"] = entered
        row[f"entered_{suffix}_rate"] = _rate(entered, count)
    stable = sum(int(item.get("stable_goal", 0)) for item in episodes)
    row["stable_goal_count"] = stable
    row["stable_goal_rate"] = _rate(stable, count)
    row["minimum_goal_distance_m"] = _minimum(
        item.get("minimum_goal_distance_m") for item in episodes
    )
    row["mean_speed_at_near_goal_entry_mps"] = _mean(
        item.get("speed_at_near_goal_entry_mps") for item in episodes
    )
    row["mean_speed_within_near_goal_mps"] = _mean(
        item.get("mean_speed_within_near_goal_mps") for item in episodes
    )
    row["mean_target_speed_within_near_goal_mps"] = _mean(
        item.get("mean_target_speed_within_near_goal_mps") for item in episodes
    )
    row["mean_speed_excess_within_near_goal_mps"] = _mean(
        item.get("mean_speed_excess_within_near_goal_mps") for item in episodes
    )
    hold_values = [item.get("maximum_consecutive_goal_hold_steps") for item in episodes]
    row["mean_maximum_consecutive_goal_hold_steps"] = _mean(hold_values)
    row["maximum_consecutive_goal_hold_steps_max"] = _maximum(hold_values)
    for threshold, suffix in ((1, "1"), (3, "3"), (5, "5")):
        hold_count = sum(
            int(_number(item, "maximum_consecutive_goal_hold_steps", 0.0) >= threshold)
            for item in episodes
        )
        row[f"episodes_with_hold_steps_ge_{suffix}"] = hold_count
        row[f"hold_steps_ge_{suffix}_rate"] = _rate(hold_count, count)
    return row


def _segment_summary_rows(
    episodes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    episode_ids = {int(row.get("episode_index", -1)) for row in episodes}
    output: list[dict[str, Any]] = []
    for segment in APPROACH_SEGMENTS:
        name = str(segment["name"])
        selected = [row for row in steps if row.get("approach_segment") == name]
        entered_ids = {int(row.get("episode_index", -1)) for row in selected}
        next_name = str(segment["next"])
        next_ids = {
            int(row.get("episode_index", -1))
            for row in steps
            if row.get("approach_segment") == next_name
        }
        braking = [row for row in selected if _number(row, "braking_action") > 0.5]
        vector_active = [
            row for row in selected if _number(row, "terminal_vector_active") > 0.5
        ]
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "segment": name,
                "segment_label": str(segment["label"]),
                "lower_m": float(segment["lower_m"]),
                "upper_m": float(segment["upper_m"]),
                "episodes_total": len(episode_ids),
                "episodes_entered": len(entered_ids),
                "episodes_entered_rate": _rate(len(entered_ids), len(episode_ids)),
                "episodes_continued_to_next": len(entered_ids & next_ids),
                "episodes_continued_to_next_rate": _rate(
                    len(entered_ids & next_ids), len(entered_ids)
                ),
                "pooled_step_count": len(selected),
                "mean_radial_progress_mps": _mean(
                    row.get("radial_progress_mps") for row in selected
                ),
                "mean_delta_goal_distance_m": _mean(
                    row.get("delta_goal_distance_m") for row in selected
                ),
                "mean_radial_velocity_toward_mps": _mean(
                    row.get("radial_velocity_toward_mps") for row in selected
                ),
                "mean_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
                "mean_target_speed_mps": _mean(
                    row.get("target_speed_mps") for row in selected
                ),
                "braking_action_fraction": _mean(
                    row.get("braking_action") for row in selected
                ),
                "effective_braking_fraction": _mean(
                    row.get("effective_braking") for row in selected
                ),
                "mean_commanded_braking_delta_v_mps": _mean(
                    row.get("commanded_braking_delta_v_mps") for row in braking
                ),
                "mean_applied_braking_delta_v_mps": _mean(
                    row.get("applied_braking_delta_v_mps") for row in braking
                ),
                "mean_actual_speed_reduction_when_braking_mps": _mean(
                    row.get("actual_speed_reduction_when_braking_mps") for row in braking
                ),
                "mean_braking_progress_signal": _mean(
                    row.get("braking_progress_signal") for row in selected
                ),
                "mean_braking_progress_reward": _mean(
                    row.get("braking_progress_reward") for row in selected
                ),
                "mean_target_velocity_action_reward": _mean(
                    row.get("target_velocity_action_reward") for row in selected
                ),
                "mean_goal_direction_signal": _mean(
                    row.get("goal_direction_signal") for row in selected
                ),
                "mean_goal_direction_reward": _mean(
                    row.get("goal_direction_reward") for row in selected
                ),
                "mean_goal_progress_reward": _mean(
                    row.get("goal_progress_reward") for row in selected
                ),
                "mean_terminal_vector_error_pre_mps": _mean(
                    row.get("terminal_vector_error_pre_mps") for row in vector_active
                ),
                "mean_terminal_vector_error_post_mps": _mean(
                    row.get("terminal_vector_error_post_mps") for row in vector_active
                ),
                "mean_terminal_vector_error_reduction_mps": _mean(
                    row.get("terminal_vector_error_reduction_mps") for row in vector_active
                ),
                "terminal_vector_active_fraction": _rate(
                    len(vector_active), len(selected)
                ),
                "mean_terminal_vector_signal": _mean(
                    row.get("terminal_vector_signal") for row in vector_active
                ),
                "mean_terminal_vector_reward": _mean(
                    row.get("terminal_vector_reward") for row in selected
                ),
                "terminal_vector_reward_max": _maximum(
                    row.get("terminal_vector_reward") for row in selected
                ),
                "mean_dwell_steps_per_entering_episode": _mean(
                    sum(
                        1
                        for row in selected
                        if int(row.get("episode_index", -1)) == episode_index
                    )
                    for episode_index in entered_ids
                ),
                "maximum_dwell_steps": int(
                    max(
                        (
                            sum(
                                1
                                for row in selected
                                if int(row.get("episode_index", -1)) == episode_index
                            )
                            for episode_index in entered_ids
                        ),
                        default=0,
                    )
                ),
            }
        )
    return output


def _distance_summary_rows(
    steps: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in DISTANCE_BIN_ORDER:
        distance_bin = bin_by_name[name]
        selected = [row for row in steps if row.get("distance_bin") == name]
        braking = [row for row in selected if _number(row, "braking_action") > 0.5]
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "distance_bin": name,
                "distance_bin_label": str(distance_bin["label"]),
                "lower_m": float(distance_bin["lower_m"]),
                "upper_m": float(distance_bin["upper_m"]),
                "episodes_with_samples": len(
                    {int(row.get("episode_index", -1)) for row in selected}
                ),
                "step_count": len(selected),
                "mean_distance_m": _mean(row.get("goal_distance_m") for row in selected),
                "mean_radial_progress_mps": _mean(
                    row.get("radial_progress_mps") for row in selected
                ),
                "mean_delta_goal_distance_m": _mean(
                    row.get("delta_goal_distance_m") for row in selected
                ),
                "mean_radial_velocity_toward_mps": _mean(
                    row.get("radial_velocity_toward_mps") for row in selected
                ),
                "mean_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
                "mean_target_speed_mps": _mean(
                    row.get("target_speed_mps") for row in selected
                ),
                "braking_action_fraction": _mean(
                    row.get("braking_action") for row in selected
                ),
                "effective_braking_fraction": _mean(
                    row.get("effective_braking") for row in selected
                ),
                "mean_commanded_braking_delta_v_mps": _mean(
                    row.get("commanded_braking_delta_v_mps") for row in braking
                ),
                "mean_applied_braking_delta_v_mps": _mean(
                    row.get("applied_braking_delta_v_mps") for row in braking
                ),
                "mean_actual_speed_reduction_when_braking_mps": _mean(
                    row.get("actual_speed_reduction_when_braking_mps") for row in braking
                ),
                "mean_terminal_vector_reward": _mean(
                    row.get("terminal_vector_reward") for row in selected
                ),
            }
        )
    return output


def _vector_summary_row(
    steps: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    episode_count: int,
) -> dict[str, Any]:
    active = [row for row in steps if _number(row, "terminal_vector_active") > 0.5]
    positive = [row for row in active if _number(row, "terminal_vector_reward") > 0.0]
    rewards = _finite(row.get("terminal_vector_reward") for row in steps)
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": episode_count,
        "activation_band_m": "(1,3] estimated post-action distance",
        "step_count_total": len(steps),
        "active_step_count": len(active),
        "positive_reward_step_count": len(positive),
        "active_step_fraction": _rate(len(active), len(steps)),
        "positive_reward_given_active_rate": _rate(len(positive), len(active)),
        "coefficient_observed": _maximum(
            (row.get("terminal_vector_coef") for row in steps), default=0.0
        ),
        "mean_error_pre_mps": _mean(
            row.get("terminal_vector_error_pre_mps") for row in active
        ),
        "mean_error_post_mps": _mean(
            row.get("terminal_vector_error_post_mps") for row in active
        ),
        "mean_error_reduction_mps": _mean(
            row.get("terminal_vector_error_reduction_mps") for row in active
        ),
        "mean_signal": _mean(row.get("terminal_vector_signal") for row in active),
        "mean_reward_per_active_step": _mean(
            row.get("terminal_vector_reward") for row in active
        ),
        "mean_reward_per_all_steps": _mean(
            row.get("terminal_vector_reward") for row in steps
        ),
        "total_reward": float(np.sum(rewards)) if rewards.size else 0.0,
        "maximum_reward": _maximum(rewards, default=0.0),
        "reward_nonnegative": bool(np.all(rewards >= -1e-12)) if rewards.size else True,
        "reward_bound_ok": bool(
            np.all((rewards >= -1e-12) & (rewards <= 1.0 + 1e-12))
        )
        if rewards.size
        else True,
    }


def _plot_comparison(
    *,
    test_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    distance_rows: Sequence[Mapping[str, Any]],
    vector_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#2563eb",
        "early_smooth": "#059669",
        "early_smooth_radial_progress": "#dc2626",
        "early_smooth_vector_correction": "#7c3aed",
    }

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    metric_names = (
        "entered_8m_count",
        "entered_5m_count",
        "entered_3m_count",
        "entered_1m_count",
        "stable_goal_count",
    )
    metric_labels = ("8 m", "5 m", "3 m", "1 m", "stable")
    positions = np.arange(len(metric_names))
    width = 0.19
    for index, policy in enumerate(POLICY_ORDER):
        row = next(item for item in test_rows if item.get("policy") == policy)
        values = [_number(row, metric) for metric in metric_names]
        axis.bar(
            positions + (index - 1.5) * width,
            values,
            width=width,
            color=colors[policy],
            label=POLICY_LABELS[policy],
        )
    axis.set_xticks(positions, metric_labels)
    axis.set_ylim(0, max(20, max(_number(row, "episodes") for row in test_rows)))
    axis.set_ylabel("Episodes out of 20")
    axis.set_title("A/C/D/E normal far-distance test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    test_path = output_dir / "acde_test_summary.png"
    figure.savefig(test_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    centers = {"1_2m": 1.5, "2_3m": 2.5, "3_3p5m": 3.25, "3p5_5m": 4.25, "5_8m": 6.5}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for policy in POLICY_ORDER:
        selected = [row for row in distance_rows if row.get("policy") == policy]
        by_name = {str(row.get("distance_bin")): row for row in selected}
        names = [name for name in DISTANCE_BIN_ORDER if name in by_name]
        x = [centers[name] for name in names]
        axes[0, 0].plot(
            x,
            [_number(by_name[name], "mean_radial_progress_mps") for name in names],
            marker="o",
            linewidth=2,
            color=colors[policy],
            label=POLICY_LABELS[policy],
        )
        axes[0, 1].plot(
            x,
            [_number(by_name[name], "mean_speed_mps") for name in names],
            marker="o",
            linewidth=2,
            color=colors[policy],
            label=f"{POLICY_LABELS[policy]} actual",
        )
        axes[0, 1].plot(
            x,
            [_number(by_name[name], "mean_target_speed_mps") for name in names],
            marker="^",
            linestyle="--",
            linewidth=1.4,
            color=colors[policy],
            alpha=0.7,
            label=f"{POLICY_LABELS[policy]} target",
        )
        axes[1, 0].plot(
            x,
            [_number(by_name[name], "braking_action_fraction") for name in names],
            marker="o",
            linewidth=2,
            color=colors[policy],
            label=POLICY_LABELS[policy],
        )

    for axis_item in axes.flat:
        axis_item.grid(alpha=0.25)
    axes[0, 0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0, 0].set_title("Radial progress versus goal distance")
    axes[0, 0].set_xlabel("Goal distance (m)")
    axes[0, 0].set_ylabel("Distance decrease rate (m/s)")
    axes[0, 1].set_title("Actual and target speed")
    axes[0, 1].set_xlabel("Goal distance (m)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[1, 0].set_title("Braking action fraction")
    axes[1, 0].set_xlabel("Goal distance (m)")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_ylim(0.0, 1.0)
    segment_centers = np.arange(len(SEGMENT_ORDER))
    for policy in POLICY_ORDER:
        selected = [row for row in segment_rows if row.get("policy") == policy]
        by_name = {str(row.get("segment")): row for row in selected}
        axes[1, 1].plot(
            segment_centers,
            [
                _number(by_name[name], "mean_terminal_vector_reward")
                if name in by_name
                else float("nan")
                for name in SEGMENT_ORDER
            ],
            marker="o",
            linewidth=2,
            color=colors[policy],
            label=POLICY_LABELS[policy],
        )
    axes[1, 1].set_title("Terminal vector-correction reward")
    axes[1, 1].set_xticks(segment_centers, ("8-5 m", "5-3 m", "3-1 m"))
    axes[1, 1].set_xlabel("Approach segment")
    axes[1, 1].set_ylabel("Mean reward per step")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=6, ncol=2)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    segment_path = output_dir / "acde_terminal_segments.png"
    figure.savefig(segment_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for policy in POLICY_ORDER:
        row = next(item for item in vector_rows if item.get("policy") == policy)
        axes[0].bar(
            POLICY_ORDER.index(policy),
            _number(row, "mean_error_reduction_mps", 0.0),
            color=colors[policy],
            label=POLICY_LABELS[policy],
        )
        axes[1].bar(
            POLICY_ORDER.index(policy),
            _number(row, "mean_reward_per_active_step", 0.0),
            color=colors[policy],
        )
    axes[0].set_title("Vector error reduction in (1, 3] m")
    axes[0].set_ylabel("m/s")
    axes[1].set_title("Vector-correction reward in active steps")
    axes[1].set_ylabel("Reward")
    axes[0].set_xticks(np.arange(len(POLICY_ORDER)), ("A", "C", "D", "E"))
    axes[1].set_xticks(np.arange(len(POLICY_ORDER)), ("A", "C", "D", "E"))
    for axis_item in axes:
        axis_item.grid(axis="y", alpha=0.25)
    vector_path = output_dir / "acde_vector_correction_summary.png"
    figure.savefig(vector_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "test_summary": str(test_path.resolve()),
        "terminal_segments": str(segment_path.resolve()),
        "vector_summary": str(vector_path.resolve()),
    }


def build_acde_comparison(
    *,
    baseline_root: str | Path,
    c_root: str | Path,
    d_root: str | Path,
    e_root: str | Path,
    output_logs: str | Path,
    output_plots: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Build the read-only A/C/D/E comparison from existing test artifacts."""

    output_logs = Path(output_logs)
    output_plots = Path(output_plots)
    for path in (output_logs, output_plots):
        if path.exists() and any(path.iterdir()):
            if not force:
                raise FileExistsError(f"non-empty comparison output: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    roots = {
        "baseline": Path(baseline_root),
        "early_smooth": Path(c_root),
        "early_smooth_radial_progress": Path(d_root),
        "early_smooth_vector_correction": Path(e_root),
    }
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    signatures_by_policy: dict[str, dict[int, str]] = {}

    for policy in POLICY_ORDER:
        episodes, steps, signatures = _load_policy_test(roots[policy], policy=policy)
        signatures_by_policy[policy] = signatures
        episode_rows.extend(episodes)
        step_rows.extend(steps)
        test_rows.append(_test_summary_row(episodes, policy=policy))
        segment_rows.extend(_segment_summary_rows(episodes, steps, policy=policy))
        distance_rows.extend(_distance_summary_rows(steps, policy=policy))
        vector_rows.append(
            _vector_summary_row(steps, policy=policy, episode_count=len(episodes))
        )

    reference_signatures = signatures_by_policy[POLICY_ORDER[0]]
    scenario_signatures_equal = all(
        signatures_by_policy[policy] == reference_signatures for policy in POLICY_ORDER[1:]
    )
    test_episode_counts_equal = len({int(row["episodes"]) for row in test_rows}) == 1

    _write_csv(output_logs / "acde_test_summary.csv", test_rows, TEST_COLUMNS)
    _write_csv(output_logs / "acde_episode_summary.csv", episode_rows)
    _write_csv(output_logs / "acde_segment_summary.csv", segment_rows)
    _write_csv(output_logs / "acde_distance_bin_summary.csv", distance_rows)
    _write_csv(output_logs / "acde_terminal_vector_summary.csv", vector_rows)
    _write_csv(output_logs / "acde_step_metrics.csv", step_rows)
    plot_paths = _plot_comparison(
        test_rows=test_rows,
        segment_rows=segment_rows,
        distance_rows=distance_rows,
        vector_rows=vector_rows,
        output_dir=output_plots,
    )

    required_files = [
        output_logs / "acde_test_summary.csv",
        output_logs / "acde_episode_summary.csv",
        output_logs / "acde_segment_summary.csv",
        output_logs / "acde_distance_bin_summary.csv",
        output_logs / "acde_terminal_vector_summary.csv",
        output_logs / "acde_step_metrics.csv",
        *(Path(path) for path in plot_paths.values()),
    ]
    summary = {
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "policy_order": list(POLICY_ORDER),
        "policies": test_rows,
        "contract": {
            "normal_far_distance_test_only": True,
            "same_test_episode_count": test_episode_counts_equal,
            "same_generated_scenarios": scenario_signatures_equal,
            "distance_boundary_rule": "lower inclusive, upper exclusive",
            "segments": {
                "8_5m": "[5,8)",
                "5_3m": "[3,5)",
                "3_1m": "[1,3)",
            },
            "vector_correction_band": "(1,3] estimated post-action distance",
            "no_training_or_policy_update": True,
        },
        "sources": {policy: str(path.resolve()) for policy, path in roots.items()},
        "scenario_signature_by_policy": signatures_by_policy,
        "schema_validation": {
            "required_files_exist": all(path.is_file() for path in required_files),
            "test_rows": len(test_rows) == len(POLICY_ORDER),
            "segment_rows": len(segment_rows) == len(POLICY_ORDER) * len(SEGMENT_ORDER),
            "vector_rows": len(vector_rows) == len(POLICY_ORDER),
            "ok": (
                all(path.is_file() for path in required_files)
                and len(test_rows) == len(POLICY_ORDER)
                and len(segment_rows) == len(POLICY_ORDER) * len(SEGMENT_ORDER)
                and len(vector_rows) == len(POLICY_ORDER)
            ),
        },
        "files": {
            "test_summary_csv": str((output_logs / "acde_test_summary.csv").resolve()),
            "episode_summary_csv": str((output_logs / "acde_episode_summary.csv").resolve()),
            "segment_summary_csv": str((output_logs / "acde_segment_summary.csv").resolve()),
            "distance_bin_summary_csv": str(
                (output_logs / "acde_distance_bin_summary.csv").resolve()
            ),
            "terminal_vector_summary_csv": str(
                (output_logs / "acde_terminal_vector_summary.csv").resolve()
            ),
            "step_metrics_csv": str((output_logs / "acde_step_metrics.csv").resolve()),
            "plots": plot_paths,
        },
    }
    _write_json(output_logs / "acde_summary.json", summary)
    _write_json(
        output_logs / "acde_manifest.json",
        {
            "dataset_id": "single_uav_acde_read_only_comparison",
            "evaluation_only": True,
            "training_performed": False,
            "policy_order": list(POLICY_ORDER),
            "contract": summary["contract"],
            "sources": summary["sources"],
        },
    )
    return summary


__all__ = [
    "POLICY_ORDER",
    "POLICY_LABELS",
    "build_acde_comparison",
]
