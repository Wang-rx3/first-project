"""Read-only A/C/E/F comparison for the single-UAV terminal experiments.

The comparison consumes existing normal far-distance ``test`` artifacts only.
It never constructs a trainer, samples a new action, or updates a checkpoint.
The four policies are the original baseline (A), early-smooth speed control
(C), ungated terminal velocity-vector correction (E), and the radial-gated
variant (F).
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .single_uav_terminal_approach_diagnostic import (
    _canonical_hash,
    _number as _approach_number,
    _step_rows_from_episode,
)


POLICY_ORDER: tuple[str, ...] = ("baseline", "early_smooth", "vector_correction", "radial_gated")
POLICY_LABELS: dict[str, str] = {
    "baseline": "A: coef=0.25 baseline",
    "early_smooth": "C: early-smooth",
    "vector_correction": "E: ungated vector correction",
    "radial_gated": "F: radial-gated vector correction",
}
SEGMENTS: tuple[tuple[str, str, float, float], ...] = (
    ("8_5m", "8-5 m", 5.0, 8.0),
    ("5_3m", "5-3 m", 3.0, 5.0),
    ("3_1m", "3-1 m", 1.0, 3.0),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    return _approach_number(row, key, default)


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


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = _finite(values)
    return float(np.mean(numbers)) if numbers.size else float(default)


def _minimum(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return float(np.min(numbers)) if numbers.size else float("nan")


def _maximum(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite(values)
    return float(np.max(numbers)) if numbers.size else float(default)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _timestamp(row: Mapping[str, Any]) -> float:
    return round(_number(row, "timestamp", float("nan")), 6)


def _scenario_signature(scenario: Mapping[str, Any]) -> str:
    generated = scenario.get("scenario_generation")
    payload = generated if isinstance(generated, Mapping) else scenario
    return _canonical_hash(payload)


def _attach_vector_fields(
    step_rows: list[dict[str, Any]], reward_rows: Sequence[Mapping[str, Any]]
) -> None:
    rewards_by_time = {_timestamp(row): row for row in reward_rows}
    for row in step_rows:
        reward = rewards_by_time.get(round(_number(row, "timestamp_s"), 6), {})
        row.update(
            {
                "terminal_vector_active": _number(
                    reward, "single_uav_terminal_target_velocity_vector_active", 0.0
                ),
                "terminal_vector_error_pre_mps": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_error_pre_mps",
                    0.0,
                ),
                "terminal_vector_error_post_mps": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_error_post_mps",
                    0.0,
                ),
                "terminal_vector_error_reduction_mps": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_error_reduction_mps",
                    0.0,
                ),
                "terminal_vector_signal": _number(
                    reward, "single_uav_terminal_target_velocity_vector_signal", 0.0
                ),
                "terminal_vector_reward": _number(
                    reward, "single_uav_terminal_target_velocity_vector_reward", 0.0
                ),
                "terminal_vector_coef": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_reward_coef",
                    0.0,
                ),
                "terminal_vector_activation_distance_m": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_activation_distance_m",
                    _number(row, "goal_distance_m", 0.0),
                ),
                "terminal_vector_post_radial_velocity_mps": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_post_radial_velocity_mps",
                    0.0,
                ),
                "terminal_vector_radial_gate": _number(
                    reward,
                    "single_uav_terminal_target_velocity_vector_radial_gate",
                    0.0,
                ),
                "terminal_vector_gate_enabled": bool(
                    _number(
                        reward,
                        "single_uav_terminal_target_velocity_vector_radial_gate_enabled",
                        0.0,
                    )
                    > 0.5
                ),
            }
        )


def _load_policy(root: Path, policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    test_root = root / "scale_1" / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {test_root}")
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    signatures: dict[int, str] = {}
    episode_paths = sorted(path for path in test_root.glob("episode_*") if path.is_dir())
    for index, episode_path in enumerate(episode_paths, start=1):
        scenario = json.loads((episode_path / "scenario.json").read_text(encoding="utf-8"))
        signature = _scenario_signature(scenario)
        signatures[index] = signature
        rows = _step_rows_from_episode(
            episode_path,
            policy="baseline",
            episode_index=index,
            seed=int(_number(scenario, "seed", index)),
            scenario_hash=signature,
            scenario=scenario,
        )
        for row in rows:
            row["policy"] = policy
            row["policy_label"] = POLICY_LABELS[policy]
        _attach_vector_fields(rows, _read_csv(episode_path / "rewards.csv"))
        minimum = _minimum(row.get("goal_distance_m") for row in rows)
        near = [row for row in rows if _number(row, "goal_distance_m") <= 8.0]
        maximum_hold = int(round(_maximum((row.get("stable_steps") for row in rows))))
        episodes.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "episode_index": index,
                "episode_id": rows[0].get("episode_id", "") if rows else "",
                "scenario_signature": signature,
                "entered_8m": int(np.isfinite(minimum) and minimum <= 8.0),
                "entered_5m": int(np.isfinite(minimum) and minimum <= 5.0),
                "entered_3m": int(np.isfinite(minimum) and minimum <= 3.0),
                "entered_1m": int(np.isfinite(minimum) and minimum <= 1.0),
                "stable_goal": int(any(_number(row, "goal_stable", 0.0) > 0.5 for row in rows)),
                "minimum_goal_distance_m": minimum,
                "mean_speed_at_near_goal_entry_mps": _number(near[0], "total_speed_mps") if near else float("nan"),
                "mean_speed_within_near_goal_mps": _mean(row.get("total_speed_mps") for row in near),
                "mean_target_speed_within_near_goal_mps": _mean(row.get("target_speed_mps") for row in near),
                "mean_speed_excess_within_near_goal_mps": _mean(row.get("speed_excess_mps") for row in near),
                "maximum_consecutive_goal_hold_steps": maximum_hold,
            }
        )
        steps.extend(rows)
    return episodes, steps, signatures


def _test_row(episodes: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    count = len(episodes)
    row: dict[str, Any] = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": count,
    }
    for threshold, name in ((8, "8m"), (5, "5m"), (3, "3m"), (1, "1m")):
        entered = sum(int(item.get(f"entered_{name}", 0)) for item in episodes)
        row[f"entered_{name}_count"] = entered
        row[f"entered_{name}_rate"] = _rate(entered, count)
    stable = sum(int(item.get("stable_goal", 0)) for item in episodes)
    row["stable_goal_count"] = stable
    row["stable_goal_rate"] = _rate(stable, count)
    row["minimum_goal_distance_m"] = _minimum(
        item.get("minimum_goal_distance_m") for item in episodes
    )
    for key in (
        "mean_speed_at_near_goal_entry_mps",
        "mean_speed_within_near_goal_mps",
        "mean_target_speed_within_near_goal_mps",
        "mean_speed_excess_within_near_goal_mps",
    ):
        row[key] = _mean(item.get(key) for item in episodes)
    holds = [item.get("maximum_consecutive_goal_hold_steps", 0) for item in episodes]
    row["mean_maximum_consecutive_goal_hold_steps"] = _mean(holds, default=0.0)
    row["maximum_consecutive_goal_hold_steps_max"] = _maximum(holds)
    for threshold, name in ((1, "1"), (3, "3"), (5, "5")):
        hold_count = sum(
            int(_number(item, "maximum_consecutive_goal_hold_steps", 0.0) >= threshold)
            for item in episodes
        )
        row[f"episodes_with_hold_steps_ge_{name}"] = hold_count
        row[f"hold_steps_ge_{name}_rate"] = _rate(hold_count, count)
    return row


def _segment_rows(steps: Sequence[Mapping[str, Any]], policy: str, episode_count: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, label, lower, upper in SEGMENTS:
        selected = [row for row in steps if row.get("approach_segment") == name]
        entered_ids = {int(row.get("episode_index", -1)) for row in selected}
        next_name = {"8_5m": "5_3m", "5_3m": "3_1m", "3_1m": ""}[name]
        next_ids = {
            int(row.get("episode_index", -1))
            for row in steps
            if next_name and row.get("approach_segment") == next_name
        }
        brakes = [row for row in selected if _number(row, "braking_action", 0.0) > 0.5]
        vector_active = [row for row in selected if _number(row, "terminal_vector_active", 0.0) > 0.5]
        output.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "segment": name,
                "segment_label": label,
                "lower_m": lower,
                "upper_m": upper,
                "episodes_total": episode_count,
                "episodes_entered": len(entered_ids),
                "episodes_entered_rate": _rate(len(entered_ids), episode_count),
                "episodes_continued_to_next": len(entered_ids & next_ids),
                "episodes_continued_to_next_rate": _rate(len(entered_ids & next_ids), len(entered_ids)),
                "step_count": len(selected),
                "mean_radial_progress_mps": _mean(row.get("radial_progress_mps") for row in selected),
                "mean_delta_goal_distance_m": _mean(row.get("delta_goal_distance_m") for row in selected),
                "mean_radial_velocity_toward_mps": _mean(row.get("radial_velocity_toward_mps") for row in selected),
                "mean_speed_mps": _mean(row.get("total_speed_mps") for row in selected),
                "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in selected),
                "braking_action_fraction": _mean(row.get("braking_action") for row in selected),
                "effective_braking_fraction": _mean(row.get("effective_braking") for row in selected),
                "mean_applied_braking_delta_v_mps": _mean(row.get("applied_braking_delta_v_mps") for row in brakes),
                "mean_actual_speed_reduction_when_braking_mps": _mean(row.get("actual_speed_reduction_when_braking_mps") for row in brakes),
                "mean_goal_progress_reward": _mean(row.get("goal_progress_reward") for row in selected),
                "mean_goal_direction_reward": _mean(row.get("goal_direction_reward") for row in selected),
                "mean_braking_progress_reward": _mean(row.get("braking_progress_reward") for row in selected),
                "mean_target_velocity_action_reward": _mean(row.get("target_velocity_action_reward") for row in selected),
                "vector_active_fraction": _rate(len(vector_active), len(selected)),
                "mean_vector_error_pre_mps": _mean(row.get("terminal_vector_error_pre_mps") for row in vector_active),
                "mean_vector_error_post_mps": _mean(row.get("terminal_vector_error_post_mps") for row in vector_active),
                "mean_vector_error_reduction_mps": _mean(row.get("terminal_vector_error_reduction_mps") for row in vector_active),
                "mean_vector_radial_gate": _mean(row.get("terminal_vector_radial_gate") for row in vector_active),
                "mean_vector_reward": _mean(row.get("terminal_vector_reward") for row in selected),
            }
        )
    return output


def _vector_row(steps: Sequence[Mapping[str, Any]], policy: str, episode_count: int) -> dict[str, Any]:
    active = [row for row in steps if _number(row, "terminal_vector_active", 0.0) > 0.5]
    positive = [row for row in active if _number(row, "terminal_vector_reward", 0.0) > 1e-12]
    gate_enabled = any(bool(row.get("terminal_vector_gate_enabled", False)) for row in steps)
    misreward = [
        row
        for row in positive
        if gate_enabled and _number(row, "terminal_vector_post_radial_velocity_mps", 0.0) <= 1e-9
    ]
    rewards = _finite(row.get("terminal_vector_reward") for row in steps)
    coefficient = _maximum((row.get("terminal_vector_coef") for row in steps), 0.0)
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": episode_count,
        "gate_enabled": gate_enabled,
        "activation_distance_source": "previous_estimated_goal_distance_m" if gate_enabled else "current_goal_distance_m",
        "active_band_m": "(1,3]",
        "active_step_count": len(active),
        "positive_reward_step_count": len(positive),
        "positive_reward_with_nonpositive_radial_step_count": len(misreward),
        "positive_reward_with_nonpositive_radial_fraction": _rate(len(misreward), len(positive)),
        "active_step_fraction": _rate(len(active), len(steps)),
        "positive_reward_given_active_rate": _rate(len(positive), len(active)),
        "coefficient_observed": coefficient,
        "mean_error_pre_mps": _mean(row.get("terminal_vector_error_pre_mps") for row in active),
        "mean_error_post_mps": _mean(row.get("terminal_vector_error_post_mps") for row in active),
        "mean_error_reduction_mps": _mean(row.get("terminal_vector_error_reduction_mps") for row in active),
        "mean_radial_gate": _mean(row.get("terminal_vector_radial_gate") for row in active),
        "mean_post_radial_velocity_mps": _mean(row.get("terminal_vector_post_radial_velocity_mps") for row in active),
        "mean_reward_per_active_step": _mean(row.get("terminal_vector_reward") for row in active),
        "total_reward": float(np.sum(rewards)) if rewards.size else 0.0,
        "maximum_reward": _maximum(rewards),
        "reward_nonnegative": bool(np.all(rewards >= -1e-12)) if rewards.size else True,
        "reward_bound_ok": bool(np.all(rewards <= coefficient + 1e-9)) if rewards.size else True,
    }


def _plot(test_rows: Sequence[Mapping[str, Any]], segment_rows: Sequence[Mapping[str, Any]], vector_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#2563eb", "early_smooth": "#059669", "vector_correction": "#f59e0b", "radial_gated": "#7c3aed"}
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    metric_names = ("entered_8m_count", "entered_5m_count", "entered_3m_count", "entered_1m_count", "stable_goal_count")
    labels = ("8 m", "5 m", "3 m", "1 m", "stable")
    x = np.arange(len(metric_names))
    width = 0.20
    for index, policy in enumerate(POLICY_ORDER):
        row = next(row for row in test_rows if row["policy"] == policy)
        axis.bar(x + (index - 1.5) * width, [_number(row, key, 0.0) for key in metric_names], width=width, color=colors[policy], label=POLICY_LABELS[policy])
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(20, max(_number(row, "episodes", 0.0) for row in test_rows)))
    axis.set_ylabel("Episodes out of 20")
    axis.set_title("A/C/E/F normal far-distance test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    test_path = output_dir / "acef_test_summary.png"
    figure.savefig(test_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    centers = np.arange(len(SEGMENTS))
    for policy in POLICY_ORDER:
        rows = [row for row in segment_rows if row["policy"] == policy]
        axes[0, 0].plot(centers, [_number(row, "mean_radial_velocity_toward_mps") for row in rows], marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
        axes[0, 1].plot(centers, [_number(row, "mean_speed_mps") for row in rows], marker="o", linewidth=2, color=colors[policy], label=f"{POLICY_LABELS[policy]} actual")
        axes[0, 1].plot(centers, [_number(row, "mean_target_speed_mps") for row in rows], marker="^", linestyle="--", linewidth=1.3, color=colors[policy], alpha=0.7, label=f"{POLICY_LABELS[policy]} target")
        axes[1, 0].plot(centers, [_number(row, "braking_action_fraction") for row in rows], marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
        axes[1, 1].plot(centers, [_number(row, "mean_vector_radial_gate") for row in rows], marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
    for axis_item in axes.flat:
        axis_item.grid(alpha=0.25)
    axes[0, 0].axhline(0.0, color="#111827", linewidth=1)
    axes[0, 0].set_title("Radial velocity toward goal")
    axes[0, 0].set_ylabel("m/s")
    axes[0, 1].set_title("Actual and target speed")
    axes[0, 1].set_ylabel("m/s")
    axes[1, 0].set_title("Braking action fraction")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_title("Terminal vector radial gate")
    axes[1, 1].set_ylim(0, 1)
    for axis_item in axes[1, :]:
        axis_item.set_xticks(centers, [item[1] for item in SEGMENTS])
        axis_item.set_xlabel("Approach segment")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=6, ncol=2)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    segment_path = output_dir / "acef_terminal_segments.png"
    figure.savefig(segment_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for index, policy in enumerate(POLICY_ORDER):
        row = next(row for row in vector_rows if row["policy"] == policy)
        axes[0].bar(index, _number(row, "mean_error_reduction_mps", 0.0), color=colors[policy])
        axes[1].bar(index, _number(row, "mean_radial_gate", 0.0), color=colors[policy])
        axes[2].bar(index, _number(row, "positive_reward_with_nonpositive_radial_step_count", 0.0), color=colors[policy])
    axes[0].set_title("Vector error reduction")
    axes[0].set_ylabel("m/s")
    axes[1].set_title("Mean radial gate")
    axes[1].set_ylabel("Gate")
    axes[2].set_title("Positive reward with non-positive radial")
    axes[2].set_ylabel("Steps")
    axes[1].set_ylim(0, 1)
    short = ("A", "C", "E", "F")
    for axis_item in axes:
        axis_item.set_xticks(np.arange(len(POLICY_ORDER)), short)
        axis_item.grid(axis="y", alpha=0.25)
    vector_path = output_dir / "acef_vector_gate_summary.png"
    figure.savefig(vector_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"test_summary": str(test_path.resolve()), "terminal_segments": str(segment_path.resolve()), "vector_gate_summary": str(vector_path.resolve())}


def build_acef_comparison(*, baseline_root: str | Path, c_root: str | Path, e_root: str | Path, f_root: str | Path, output_logs: str | Path, output_plots: str | Path, force: bool = False) -> dict[str, Any]:
    """Build a read-only A/C/E/F comparison from existing test artifacts."""

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
        "vector_correction": Path(e_root),
        "radial_gated": Path(f_root),
    }
    all_episode_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    signatures: dict[str, dict[int, str]] = {}
    for policy in POLICY_ORDER:
        episodes, steps, policy_signatures = _load_policy(roots[policy], policy)
        signatures[policy] = policy_signatures
        all_episode_rows.extend(episodes)
        all_step_rows.extend(steps)
        test_rows.append(_test_row(episodes, policy))
        segment_rows.extend(_segment_rows(steps, policy, len(episodes)))
        vector_rows.append(_vector_row(steps, policy, len(episodes)))
    reference = signatures[POLICY_ORDER[0]]
    same_scenarios = all(signatures[policy] == reference for policy in POLICY_ORDER[1:])
    plot_paths = _plot(test_rows, segment_rows, vector_rows, output_plots)
    _write_csv(output_logs / "acef_test_summary.csv", test_rows)
    _write_csv(output_logs / "acef_episode_summary.csv", all_episode_rows)
    _write_csv(output_logs / "acef_segment_summary.csv", segment_rows)
    _write_csv(output_logs / "acef_terminal_vector_summary.csv", vector_rows)
    _write_csv(output_logs / "acef_step_metrics.csv", all_step_rows)
    required = [
        output_logs / "acef_test_summary.csv",
        output_logs / "acef_episode_summary.csv",
        output_logs / "acef_segment_summary.csv",
        output_logs / "acef_terminal_vector_summary.csv",
        output_logs / "acef_step_metrics.csv",
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
            "same_test_episode_count": len({int(row["episodes"]) for row in test_rows}) == 1,
            "same_generated_scenarios": same_scenarios,
            "no_training_or_policy_update": True,
            "segments": {name: f"[{lower},{upper})" for name, _, lower, upper in SEGMENTS},
            "F_activation_distance": "previous_estimated_goal_distance_m",
            "F_radial_gate": "clip(post_radial_velocity / max(target_speed_pre, 1e-6), 0, 1)",
        },
        "sources": {policy: str(path.resolve()) for policy, path in roots.items()},
        "scenario_signature_by_policy": signatures,
        "schema_validation": {
            "required_files_exist": all(path.is_file() for path in required),
            "test_rows": len(test_rows) == len(POLICY_ORDER),
            "segment_rows": len(segment_rows) == len(POLICY_ORDER) * len(SEGMENTS),
            "vector_rows": len(vector_rows) == len(POLICY_ORDER),
            "same_scenarios": same_scenarios,
            "ok": (
                all(path.is_file() for path in required)
                and len(test_rows) == len(POLICY_ORDER)
                and len(segment_rows) == len(POLICY_ORDER) * len(SEGMENTS)
                and len(vector_rows) == len(POLICY_ORDER)
                and same_scenarios
            ),
        },
        "files": {
            "test_summary_csv": str((output_logs / "acef_test_summary.csv").resolve()),
            "episode_summary_csv": str((output_logs / "acef_episode_summary.csv").resolve()),
            "segment_summary_csv": str((output_logs / "acef_segment_summary.csv").resolve()),
            "terminal_vector_summary_csv": str((output_logs / "acef_terminal_vector_summary.csv").resolve()),
            "step_metrics_csv": str((output_logs / "acef_step_metrics.csv").resolve()),
            "plots": plot_paths,
        },
    }
    _write_json(output_logs / "acef_summary.json", summary)
    _write_json(output_logs / "acef_manifest.json", {"dataset_id": "single_uav_acef_read_only_comparison", **summary["contract"], "sources": summary["sources"]})
    return summary


__all__ = ["POLICY_ORDER", "POLICY_LABELS", "build_acef_comparison"]
