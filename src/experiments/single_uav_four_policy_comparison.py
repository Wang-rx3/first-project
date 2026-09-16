"""Read-only four-policy comparison for the single-UAV C-route experiments.

The module combines already-generated formal-test and frozen-diagnostic CSVs.
It never loads a trainer and never updates a policy.  The comparison is kept
separate from training so that the four rows remain directly traceable to the
following frozen policies:

* A: original ``coef=0.25`` baseline;
* B: old navigation-preserving fine-tuning;
* C: early-smooth speed-control fine-tuning;
* D: early-smooth plus terminal radial-progress reward fine-tuning.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POLICY_ORDER: tuple[str, ...] = (
    "baseline",
    "old_fine_tuned",
    "early_smooth",
    "early_smooth_radial_progress",
)

POLICY_LABELS: dict[str, str] = {
    "baseline": "A: coef=0.25 baseline",
    "old_fine_tuned": "B: old navigation-preserving fine-tuned",
    "early_smooth": "C: early-smooth speed-control fine-tuned",
    "early_smooth_radial_progress": "D: early-smooth + terminal radial-progress",
}

SOURCE_POLICY_MAP = {
    "baseline": "baseline",
    "old_fine_tuned": "old_fine_tuned",
    "new_speed_control_adjusted": "early_smooth",
}

SEGMENT_ORDER = ("8_5m", "5_3m", "3_1m")
DISTANCE_BIN_ORDER = ("1_2m", "2_3m", "3_3p5m", "3p5_5m", "5_8m")

TEST_COLUMNS = (
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
    "mean_near_goal_applied_braking_delta_v_mps",
    "mean_near_goal_speed_reduction_when_braking_mps",
    "hold_steps_ge_1_rate",
    "hold_steps_ge_3_rate",
    "hold_steps_ge_5_rate",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        # Use a stable first-seen ordering for rows with heterogeneous
        # diagnostic columns.
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        columns = tuple(ordered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite(values: Iterable[Any]) -> np.ndarray:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=float)


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


def _normalise_existing_test_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row in rows:
        source_policy = str(source_row.get("policy", ""))
        policy = SOURCE_POLICY_MAP.get(source_policy)
        if policy is None:
            continue
        row = dict(source_row)
        row["policy"] = policy
        row["policy_label"] = POLICY_LABELS[policy]
        output.append(row)
    return output


def _d_test_row(
    formal_navigation_rows: Sequence[Mapping[str, Any]],
    diagnostic_step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build D's test row from the formal navigation and paired step logs."""

    navigation_rows = [row for row in formal_navigation_rows if str(row.get("split", "")) == "test"]
    step_groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in diagnostic_step_rows:
        if str(row.get("policy", "")) != "fine_tuned":
            continue
        try:
            episode_index = int(row.get("episode_index", -1))
        except (TypeError, ValueError):
            continue
        step_groups.setdefault(episode_index, []).append(row)

    minima = [_minimum(_number(row, "goal_distance_m") for row in rows) for rows in step_groups.values()]
    minima = [value for value in minima if np.isfinite(value)]
    episode_count = len(navigation_rows) or len(step_groups)
    entered = lambda threshold: sum(1 for value in minima if value <= threshold)
    hold_values = [
        _number(row, "maximum_consecutive_goal_hold_steps")
        for row in navigation_rows
    ]
    hold_values = [value for value in hold_values if np.isfinite(value)]

    row: dict[str, Any] = {
        "policy": "early_smooth_radial_progress",
        "policy_label": POLICY_LABELS["early_smooth_radial_progress"],
        "episodes": episode_count,
        "entered_8m_count": entered(8.0),
        "entered_5m_count": entered(5.0),
        "entered_3m_count": entered(3.0),
        "entered_1m_count": entered(1.0),
        "stable_goal_count": sum(
            int(_number(item, "stable_goal", 0.0) > 0.5) for item in navigation_rows
        ),
        "minimum_goal_distance_m": _minimum(
            _number(row, "minimum_goal_distance_m") for row in navigation_rows
        ) if navigation_rows else _minimum(minima),
        "mean_speed_at_near_goal_entry_mps": _mean(
            _number(row, "speed_at_near_goal_entry_mps") for row in navigation_rows
        ),
        "mean_speed_within_near_goal_mps": _mean(
            _number(row, "mean_speed_within_near_goal_mps") for row in navigation_rows
        ),
        "mean_target_speed_within_near_goal_mps": _mean(
            _number(row, "mean_target_speed_within_near_goal_mps") for row in navigation_rows
        ),
        "mean_speed_excess_within_near_goal_mps": _mean(
            _number(row, "mean_speed_excess_within_near_goal_mps") for row in navigation_rows
        ),
        "mean_maximum_consecutive_goal_hold_steps": _mean(hold_values),
        "maximum_consecutive_goal_hold_steps_max": _maximum(hold_values),
        "episodes_with_hold_steps_ge_1": sum(value >= 1.0 for value in hold_values),
        "episodes_with_hold_steps_ge_3": sum(value >= 3.0 for value in hold_values),
        "episodes_with_hold_steps_ge_5": sum(value >= 5.0 for value in hold_values),
        "mean_near_goal_applied_braking_delta_v_mps": _mean(
            _number(row, "near_goal_mean_applied_braking_delta_v_mps")
            for row in navigation_rows
        ),
        "mean_near_goal_speed_reduction_when_braking_mps": _mean(
            _number(row, "near_goal_mean_speed_reduction_when_braking_mps")
            for row in navigation_rows
        ),
    }
    for threshold, suffix in ((8.0, "8m"), (5.0, "5m"), (3.0, "3m"), (1.0, "1m")):
        row[f"entered_{suffix}_rate"] = _rate(int(row[f"entered_{suffix}_count"]), episode_count)
    row["stable_goal_rate"] = _rate(int(row["stable_goal_count"]), episode_count)
    for threshold, suffix in ((1.0, "1"), (3.0, "3"), (5.0, "5")):
        count = int(row[f"episodes_with_hold_steps_ge_{suffix}"])
        row[f"hold_steps_ge_{suffix}_rate"] = _rate(count, episode_count)
    return row


def _normalise_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_policy: str,
    target_policy: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row in rows:
        if str(source_row.get("policy", "")) != source_policy:
            continue
        row = dict(source_row)
        row["policy"] = target_policy
        row["policy_label"] = POLICY_LABELS[target_policy]
        output.append(row)
    return output


def _plot_comparison(
    *,
    test_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    distance_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#2563eb",
        "old_fine_tuned": "#dc2626",
        "early_smooth": "#059669",
        "early_smooth_radial_progress": "#7c3aed",
    }
    labels = [POLICY_LABELS[policy] for policy in POLICY_ORDER]

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    metric_names = ("entered_8m_count", "entered_5m_count", "entered_3m_count", "entered_1m_count", "stable_goal_count")
    metric_labels = ("8 m", "5 m", "3 m", "1 m", "stable")
    positions = np.arange(len(metric_names))
    width = 0.19
    for index, policy in enumerate(POLICY_ORDER):
        row = next(item for item in test_rows if item.get("policy") == policy)
        values = [_number(row, metric) for metric in metric_names]
        axis.bar(positions + (index - 1.5) * width, values, width=width, color=colors[policy], label=labels[index])
    axis.set_xticks(positions, metric_labels)
    axis.set_ylim(0, max(20, max(_number(row, "episodes") for row in test_rows)))
    axis.set_ylabel("Episodes out of 20")
    axis.set_title("Four-policy normal far-distance test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    test_path = output_dir / "four_policy_test_summary.png"
    figure.savefig(test_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    centers = {
        "1_2m": 1.5,
        "2_3m": 2.5,
        "3_3p5m": 3.25,
        "3p5_5m": 4.25,
        "5_8m": 6.5,
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for policy in POLICY_ORDER:
        selected = [row for row in distance_rows if row.get("policy") == policy]
        by_name = {str(row.get("distance_bin")): row for row in selected}
        x = [centers[name] for name in DISTANCE_BIN_ORDER if name in by_name]
        progress = [_number(by_name[name], "mean_radial_progress_mps") for name in DISTANCE_BIN_ORDER if name in by_name]
        speed = [_number(by_name[name], "mean_speed_mps") for name in DISTANCE_BIN_ORDER if name in by_name]
        target = [_number(by_name[name], "mean_target_speed_mps") for name in DISTANCE_BIN_ORDER if name in by_name]
        braking = [_number(by_name[name], "braking_action_fraction") for name in DISTANCE_BIN_ORDER if name in by_name]
        if x:
            axes[0, 0].plot(x, progress, marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
            axes[0, 1].plot(x, speed, marker="o", linewidth=2, color=colors[policy], label=f"{POLICY_LABELS[policy]} speed")
            axes[0, 1].plot(x, target, marker="^", linestyle="--", linewidth=1.4, color=colors[policy], alpha=0.7, label=f"{POLICY_LABELS[policy]} target")
            axes[1, 0].plot(x, braking, marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
        selected_segments = [row for row in segment_rows if row.get("policy") == policy]
        segment_by_name = {str(row.get("segment")): row for row in selected_segments}
        x_segment = np.arange(len(SEGMENT_ORDER))
        y_segment = [_number(segment_by_name[name], "mean_terminal_radial_progress_reward") if name in segment_by_name else 0.0 for name in SEGMENT_ORDER]
        axes[1, 1].plot(x_segment, y_segment, marker="o", linewidth=2, color=colors[policy], label=POLICY_LABELS[policy])
    for axis in list(axes.flat)[:3]:
        axis.axvline(3.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.grid(alpha=0.25)
    axes[0, 0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0, 0].set_title("Radial progress by goal distance")
    axes[0, 0].set_xlabel("Goal distance (m)")
    axes[0, 0].set_ylabel("Distance decrease rate (m/s)")
    axes[0, 1].set_title("Actual and target speed")
    axes[0, 1].set_xlabel("Goal distance (m)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[1, 0].set_title("Braking action fraction")
    axes[1, 0].set_xlabel("Goal distance (m)")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_title("Terminal radial-progress reward by segment")
    axes[1, 1].set_xticks(np.arange(len(SEGMENT_ORDER)), ("8-5 m", "5-3 m", "3-1 m"))
    axes[1, 1].set_xlabel("Approach segment")
    axes[1, 1].set_ylabel("Mean reward")
    axes[1, 1].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=6, ncol=2)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    diagnostic_path = output_dir / "four_policy_terminal_approach_comparison.png"
    figure.savefig(diagnostic_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "test_summary": str(test_path.resolve()),
        "terminal_approach": str(diagnostic_path.resolve()),
    }


def build_four_policy_comparison(
    *,
    three_policy_logs: str | Path,
    d_formal_logs: str | Path,
    d_diagnostic_logs: str | Path,
    reference_diagnostic_logs: str | Path | None = None,
    output_logs: str | Path,
    output_plots: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Combine existing A/B/C/D outputs without retraining or reevaluation."""

    output_logs = Path(output_logs)
    output_plots = Path(output_plots)
    for path in (output_logs, output_plots):
        if path.exists() and any(path.iterdir()):
            if not force:
                raise FileExistsError(f"non-empty comparison output: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    three_policy_logs = Path(three_policy_logs)
    d_formal_logs = Path(d_formal_logs)
    d_diagnostic_logs = Path(d_diagnostic_logs)
    reference_diagnostic_logs = (
        Path(reference_diagnostic_logs) if reference_diagnostic_logs is not None else None
    )
    test_rows = _normalise_existing_test_rows(_read_csv(three_policy_logs / "three_policy_test_summary.csv"))
    test_rows.append(
        _d_test_row(
            _read_csv(d_formal_logs / "single_uav_navigation_analysis.csv"),
            _read_csv(d_diagnostic_logs / "step_metrics.csv"),
        )
    )
    test_rows = [row for policy in POLICY_ORDER for row in test_rows if row.get("policy") == policy]

    segment_rows = _normalise_existing_test_rows([])
    source_segments = _read_csv(three_policy_logs / "three_policy_segment_summary.csv")
    segment_rows.extend(
        row
        for row in (
            dict(source_row) | {"policy": SOURCE_POLICY_MAP.get(str(source_row.get("policy", "")), "")}
            for source_row in source_segments
        )
        if row.get("policy") in POLICY_ORDER[:3]
    )
    for row in segment_rows:
        row["policy_label"] = POLICY_LABELS[str(row["policy"])]
    segment_rows.extend(
        _normalise_diagnostic_rows(
            _read_csv(d_diagnostic_logs / "segment_summary.csv"),
            source_policy="fine_tuned",
            target_policy="early_smooth_radial_progress",
        )
    )

    distance_rows: list[dict[str, Any]] = []
    source_distances = _read_csv(three_policy_logs / "three_policy_distance_bin_summary.csv")
    for source_row in source_distances:
        policy = SOURCE_POLICY_MAP.get(str(source_row.get("policy", "")))
        if policy in POLICY_ORDER[:3]:
            row = dict(source_row)
            row["policy"] = policy
            row["policy_label"] = POLICY_LABELS[policy]
            distance_rows.append(row)
    distance_rows.extend(
        _normalise_diagnostic_rows(
            _read_csv(d_diagnostic_logs / "distance_bin_summary.csv"),
            source_policy="fine_tuned",
            target_policy="early_smooth_radial_progress",
        )
    )

    cross_policy_scenario_hashes_equal: bool | None = None
    if reference_diagnostic_logs is not None:
        reference_rows = _read_csv(reference_diagnostic_logs / "step_metrics.csv")
        reference_hashes = {
            int(row["episode_index"]): str(row.get("scenario_hash", ""))
            for row in reference_rows
            if str(row.get("policy", "")) == "baseline"
        }
        d_hashes = {
            int(row["episode_index"]): str(row.get("scenario_hash", ""))
            for row in _read_csv(d_diagnostic_logs / "step_metrics.csv")
            if str(row.get("policy", "")) == "baseline"
        }
        cross_policy_scenario_hashes_equal = reference_hashes == d_hashes and bool(reference_hashes)

    _write_csv(output_logs / "four_policy_test_summary.csv", test_rows, TEST_COLUMNS)
    _write_csv(output_logs / "four_policy_segment_summary.csv", segment_rows)
    _write_csv(output_logs / "four_policy_distance_bin_summary.csv", distance_rows)

    d_reward_rows = [
        row
        for row in distance_rows
        if row.get("policy") == "early_smooth_radial_progress"
    ]
    _write_csv(output_logs / "D_terminal_radial_progress_distance_summary.csv", d_reward_rows)
    plot_paths = _plot_comparison(
        test_rows=test_rows,
        segment_rows=segment_rows,
        distance_rows=distance_rows,
        output_dir=output_plots,
    )
    summary = {
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "policy_order": list(POLICY_ORDER),
        "policies": test_rows,
        "sources": {
            "three_policy_logs": str(three_policy_logs.resolve()),
            "D_formal_logs": str(d_formal_logs.resolve()),
            "D_frozen_diagnostic_logs": str(d_diagnostic_logs.resolve()),
            "reference_diagnostic_logs": (
                str(reference_diagnostic_logs.resolve())
                if reference_diagnostic_logs is not None
                else None
            ),
        },
        "contract": {
            "same_normal_far_distance_test_contract": True,
            "cross_policy_scenario_hashes_equal": cross_policy_scenario_hashes_equal,
            "D_reward_coef": 0.5,
            "D_reward_activation_band_m": "(1,3] estimated post-action distance",
            "D_goal_hold_reward": 0.0,
            "D_stable_hold_steps": 5,
            "no_new_training_or_policy_update": True,
        },
        "files": {
            "test_summary_csv": str((output_logs / "four_policy_test_summary.csv").resolve()),
            "segment_summary_csv": str((output_logs / "four_policy_segment_summary.csv").resolve()),
            "distance_summary_csv": str((output_logs / "four_policy_distance_bin_summary.csv").resolve()),
            "D_reward_summary_csv": str((output_logs / "D_terminal_radial_progress_distance_summary.csv").resolve()),
            "plots": plot_paths,
        },
    }
    _write_json(output_logs / "four_policy_summary.json", summary)
    _write_json(
        output_logs / "four_policy_manifest.json",
        {
            "dataset_id": "single_uav_four_policy_read_only_comparison",
            "evaluation_only": True,
            "training_performed": False,
            "policy_order": list(POLICY_ORDER),
            "source_contract": summary["contract"],
            "sources": summary["sources"],
        },
    )
    return summary


__all__ = ["POLICY_ORDER", "POLICY_LABELS", "build_four_policy_comparison"]
