"""Read-only diagnostics for the action -> EKF -> learning-feedback chain.

The diagnostic is intentionally separate from training.  It consumes only the
already written ``train`` and ``validation`` artifacts and never opens a
``test`` directory.  Its four outputs answer the narrow questions that are
otherwise easy to conflate:

* do different candidate neighbours produce different EKF outcomes;
* does an accepted update generate delayed training credit;
* is ``none`` selected because of the timing mask or because the policy chose
  it while a candidate was eligible; and
* is the number of accepted updates related to episode-level RMSE.

The covariance reduction used here is the EKF event's pre-restoration linear
diagnostic.  It is not presented as the final physical covariance reduction;
the conservative restored covariance remains the reporting state.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import position_error_metrics


UAV_COUNT = 3
PACKET_BYTES = 512
SPLITS = ("train", "validation")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: _jsonable(row.get(key, ""))
                for key in keys
            }
            for row in rows
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return value


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = _float(row, key, float(default))
    return int(round(value)) if np.isfinite(value) else default


def _bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _episode_dirs(scale: Path, split: str) -> list[Path]:
    root = scale / split
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("episode_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _event_rows(scale: Path, split: str) -> list[dict[str, Any]]:
    """Read exactly one split event stream; never fall back to test data."""

    path = scale / "diagnostics" / f"{split}_cooperative_updates.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _episode_metrics(episode: Path) -> dict[str, Any]:
    ground_truth = _read_csv(episode / "ground_truth.csv")
    estimates = _read_csv(episode / "estimates.csv")
    metrics = position_error_metrics(ground_truth, estimates)
    actions = _read_csv(episode / "actions.csv")
    rewards = _read_csv(episode / "rewards.csv")
    selected = sum(
        str(row.get("measurement_uav_id", "none")) not in {"", "none", str(row.get("uav_id", ""))}
        for row in actions
    )
    windows = max(1, len(actions) // UAV_COUNT)
    events = []
    # The split-specific event stream is read by the caller.  This local list
    # only keeps the metrics helper independent of the global split stream.
    return {
        "episode_id": episode.name,
        "scenario_id": str(actions[0].get("scenario_id", "")) if actions else "",
        "selected_measurement_count": int(selected),
        "actual_measurement_bytes": int(selected * PACKET_BYTES),
        "windows": int(windows),
        "accepted_updates_per_window": float("nan"),
        "position_rmse_norm_m": _float(metrics, "rmse_norm_m"),
        "position_p95_norm_m": _float(metrics, "p95_norm_m"),
        "position_error_sample_count": _int(metrics, "sample_count"),
        "reward_rows": len(rewards),
    }


def _action_rows(scale: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for episode in _episode_dirs(scale, split):
        path = episode / "actions.csv"
        if path.is_file():
            rows.extend(_read_csv(path))
    return rows


def _reward_rows(scale: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for episode in _episode_dirs(scale, split):
        path = episode / "rewards.csv"
        if path.is_file():
            rows.extend(_read_csv(path))
    return rows


def _event_by_episode(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id", ""))].append(dict(row))
    return grouped


def _neighbor_outcomes(
    scale: Path,
    method: str,
    split: str,
    event_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], int] = defaultdict(int)
    for row in action_rows:
        source = str(row.get("measurement_uav_id", "none"))
        target = str(row.get("uav_id", ""))
        if source not in {"", "none", target}:
            selected[(target, source)] += 1

    grouped = defaultdict(list)
    for row in event_rows:
        pair = (str(row.get("target_uav_id", "")), str(row.get("source_uav_id", "")))
        grouped[pair].append(row)

    pairs = sorted(set(selected) | set(grouped))
    rows: list[dict[str, Any]] = []
    for target, source in pairs:
        events = grouped[(target, source)]
        accepted = [row for row in events if _bool(row, "accepted")]
        all_reductions: list[float] = []
        accepted_reductions: list[float] = []
        final_deltas: list[float] = []
        nis: list[float] = []
        corrections: list[float] = []
        directional_information: list[float] = []
        for event in events:
            before = _float(event, "position_covariance_trace_before_m2")
            linear_after = _float(event, "position_covariance_trace_after_linear_m2")
            final_after = _float(event, "position_covariance_trace_after_final_m2")
            if np.isfinite(before) and before > 1e-12 and np.isfinite(linear_after):
                reduction = float(np.clip((before - linear_after) / before, -1.0, 1.0))
                all_reductions.append(reduction)
                if _bool(event, "accepted"):
                    accepted_reductions.append(reduction)
            if np.isfinite(before) and np.isfinite(final_after):
                final_deltas.append(float(final_after - before))
            for key, values in (("nis", nis), ("position_correction_norm_m", corrections)):
                value = _float(event, key)
                if np.isfinite(value):
                    values.append(value)
            value = _float(event, "local_direction_variance_m2")
            effective = _float(event, "effective_measurement_variance_m2")
            if np.isfinite(value) and value >= 0.0 and np.isfinite(effective) and effective > 0.0:
                directional_information.append(
                    float(np.clip(value / (value + effective), 0.0, 1.0))
                )
        rows.append(
            {
                "method": method,
                "split": split,
                "source_uav_id": source,
                "target_uav_id": target,
                "selected_count": int(selected[(target, source)]),
                "event_count": len(events),
                "accepted_count": len(accepted),
                "accepted_rate_given_event": float(len(accepted) / len(events)) if events else "NA",
                "linear_covariance_reduction_mean": _mean(all_reductions),
                "accepted_linear_covariance_reduction_mean": _mean(accepted_reductions),
                "final_covariance_trace_delta_mean": _mean(final_deltas),
                "nis_mean": _mean(nis),
                "position_correction_norm_mean_m": _mean(corrections),
                "directional_information_fraction_mean": _mean(directional_information),
                "covariance_feedback_source": "pre_restoration_linear_update_for_training_only",
                "source_scale": str(scale),
            }
        )
    return rows


def _eligibility_summary(
    scale: Path,
    method: str,
    split: str,
    action_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(action_rows)
    candidate = sum(_int(row, "candidate_count_before_timing") for row in action_rows)
    eligible = sum(_int(row, "eligible_candidate_count") for row in action_rows)
    eligible_action_rows = sum(
        _int(row, "eligible_candidate_count") > 0 for row in action_rows
    )
    filtered = sum(_int(row, "timing_filtered_candidate_count") for row in action_rows)
    selected = sum(
        str(row.get("measurement_uav_id", "none")) not in {"", "none", str(row.get("uav_id", ""))}
        for row in action_rows
    )
    forced = sum(_int(row, "none_forced_by_timing_mask") for row in action_rows)
    none_eligible = sum(_int(row, "none_when_candidate_eligible") for row in action_rows)
    budget_filtered = sum(_int(row, "budget_filtered") for row in action_rows)
    return {
        "method": method,
        "split": split,
        "action_rows": total,
        "candidate_count_before_timing": candidate,
        "eligible_candidate_count": eligible,
        "timing_filtered_candidate_count": filtered,
        "selected_non_none_count": int(selected),
        "eligible_action_row_count": int(eligible_action_rows),
        "none_forced_by_timing_mask_count": forced,
        "none_when_candidate_eligible_count": none_eligible,
        "budget_filtered_count": budget_filtered,
        "selected_rate_given_eligible": float(selected / eligible) if eligible else "NA",
        "selected_rate_given_eligible_action_row": (
            float(selected / eligible_action_rows) if eligible_action_rows else "NA"
        ),
        "none_forced_rate_given_candidate": float(forced / candidate) if candidate else "NA",
        "none_policy_rate_given_eligible": float(none_eligible / eligible) if eligible else "NA",
        "none_policy_rate_given_eligible_action_row": (
            float(none_eligible / eligible_action_rows) if eligible_action_rows else "NA"
        ),
        "timing_mask_enabled_rows": sum(
            _int(row, "candidate_count_before_timing") != _int(row, "eligible_candidate_count")
            for row in action_rows
        ),
        "source_scale": str(scale),
    }


def _accepted_rmse_rows(
    scale: Path,
    method: str,
    split: str,
    event_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in _episode_dirs(scale, split):
        metrics = _episode_metrics(episode)
        events = list(event_by_episode.get(episode.name, []))
        accepted = [event for event in events if _bool(event, "accepted")]
        beneficial = 0
        harmful = 0
        for event in accepted:
            before = _float(event, "true_position_error_before_m")
            after = _float(event, "true_position_error_after_m")
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            if after < before - 1e-3:
                beneficial += 1
            elif after > before + 1e-3:
                harmful += 1
        count = len(accepted)
        rows.append(
            {
                "method": method,
                "split": split,
                "episode_id": episode.name,
                "selected_measurement_count": metrics["selected_measurement_count"],
                "actual_measurement_bytes": metrics["actual_measurement_bytes"],
                "accepted_update_count": count,
                "accepted_updates_per_window": float(count / metrics["windows"]),
                "beneficial_fraction": float(beneficial / count) if count else "NA",
                "harmful_fraction": float(harmful / count) if count else "NA",
                "position_rmse_norm_m": metrics["position_rmse_norm_m"],
                "position_p95_norm_m": metrics["position_p95_norm_m"],
                "windows": metrics["windows"],
                "source_scale": str(scale),
            }
        )
    return rows


def _delayed_feedback_rows(
    scale: Path,
    method: str,
    split: str,
    event_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in _episode_dirs(scale, split):
        reward_path = episode / "rewards.csv"
        reward_rows = _read_csv(reward_path) if reward_path.is_file() else []
        # The reward is team-shared and written once per UAV.  Use one row per
        # timestamp to avoid multiplying delayed credit by the UAV count.
        by_timestamp: dict[str, dict[str, str]] = {}
        for row in reward_rows:
            by_timestamp.setdefault(str(row.get("timestamp", "")), row)
        feedback_values = [
            _float(row, "accepted_update_feedback_reward", 0.0)
            for row in by_timestamp.values()
        ]
        feedback_values = [value for value in feedback_values if np.isfinite(value)]
        records = sum(
            len(_json_records(row.get("released_update_feedback_records_json", "[]")))
            for row in by_timestamp.values()
        )
        selected_records = sum(
            len(_json_records(row.get("accepted_update_feedback_records_json", "[]")))
            for row in by_timestamp.values()
        )
        events = list(event_by_episode.get(episode.name, []))
        rows.append(
            {
                "method": method,
                "split": split,
                "episode_id": episode.name,
                "accepted_event_count": sum(_bool(event, "accepted") for event in events),
                "feedback_selected_record_count": selected_records,
                "feedback_released_record_count": records,
                "feedback_bearing_window_count": sum(abs(value) > 1e-12 for value in feedback_values),
                "feedback_reward_sum": float(sum(feedback_values)),
                "feedback_reward_mean_per_window": _mean(feedback_values),
                "feedback_delay_steps": _int(next(iter(by_timestamp.values()), {}), "accepted_update_feedback_delay_steps"),
                "learning_feedback_mode": str(
                    next(iter(by_timestamp.values()), {}).get(
                        "learning_feedback_mode", "not_recorded"
                    )
                ),
                "feedback_used_for_ppo": int(split == "train"),
                "feedback_covariance_source": str(
                    next(iter(by_timestamp.values()), {}).get(
                        "accepted_update_feedback_covariance_source", "not_applicable"
                    )
                ),
                "source_scale": str(scale),
            }
        )
    return rows


def _json_records(value: Any) -> list[Mapping[str, Any]]:
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(result) if isinstance(result, list) else []


def _diagnostic_figure(
    neighbor_rows: Sequence[Mapping[str, Any]],
    rmse_rows: Sequence[Mapping[str, Any]],
    eligibility_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    png: Path,
    pdf: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in neighbor_rows:
        if row.get("split") == "validation":
            grouped[str(row.get("method"))].append(row)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {"baseline_mappo": "#d62728", "local_only_fixed": "#7f7f7f", "random_cooperative": "#ff7f0e", "budgeted_greedy_ig": "#9467bd"}
    for method in sorted(grouped):
        for row in sorted(grouped[method], key=lambda item: (str(item.get("target_uav_id")), str(item.get("source_uav_id")))):
            value = _float(row, "accepted_linear_covariance_reduction_mean")
            if np.isfinite(value):
                labels.append(f"{method}\n{row.get('target_uav_id')}<-{row.get('source_uav_id')}")
                values.append(value)
                colors.append(palette.get(method, "#1f77b4"))
    if values:
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)), labels, rotation=70, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("accepted linear covariance reduction")
    ax.set_title("Per-neighbour EKF outcome (diagnostic proxy)")

    ax = axes[0, 1]
    for method in sorted({str(row.get("method")) for row in rmse_rows}):
        group = [row for row in rmse_rows if row.get("method") == method and row.get("split") == "validation"]
        x = np.asarray([_float(row, "accepted_updates_per_window") for row in group], dtype=float)
        y = np.asarray([_float(row, "position_rmse_norm_m") for row in group], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(x[mask], y[mask], label=method, alpha=0.75, s=24, color=palette.get(method))
    ax.set_xlabel("accepted updates / window")
    ax.set_ylabel("episode position RMSE (m)")
    ax.set_title("Accepted updates versus RMSE (episode unit)")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 0]
    validation_eligibility = [row for row in eligibility_rows if row.get("split") == "validation"]
    methods = [str(row.get("method")) for row in validation_eligibility]
    x = np.arange(len(methods))
    forced = np.asarray([_float(row, "none_forced_by_timing_mask_count", 0.0) for row in validation_eligibility])
    policy = np.asarray([_float(row, "none_when_candidate_eligible_count", 0.0) for row in validation_eligibility])
    width = 0.38
    if len(methods):
        ax.bar(x - width / 2, forced, width, label="none forced by timing", color="#17becf")
        ax.bar(x + width / 2, policy, width, label="none while candidate eligible", color="#bcbd22")
        ax.set_xticks(x, methods, rotation=35, ha="right")
    ax.set_ylabel("action rows")
    ax.set_title("Why the policy selected none")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    plotted = False
    for method in sorted({str(row.get("method")) for row in feedback_rows}):
        group = [row for row in feedback_rows if row.get("method") == method and row.get("split") == "train"]
        x = np.asarray([_float(row, "accepted_event_count") for row in group], dtype=float)
        y = np.asarray([_float(row, "feedback_reward_sum") for row in group], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any() and np.any(np.abs(y[mask]) > 1e-12):
            ax.scatter(x[mask], y[mask], label=method, alpha=0.75, s=24, color=palette.get(method))
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No non-zero delayed feedback in train", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("accepted updates / episode")
    ax.set_ylabel("delayed feedback reward / episode")
    ax.set_title("Action-linked delayed feedback")
    if plotted:
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Action → EKF update → delayed feedback diagnostic; train/validation only")
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def run_action_feedback_diagnostic(
    paths: Mapping[str, str | Path],
    logs_dir: str | Path,
    plots_dir: str | Path,
    *,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate the four-chain diagnostic without touching test artifacts."""

    logs = Path(logs_dir)
    plots = Path(plots_dir)
    if logs.exists() and any(logs.iterdir()):
        raise FileExistsError(f"diagnostic log directory is non-empty: {logs}")
    if plots.exists() and any(plots.iterdir()):
        raise FileExistsError(f"diagnostic plot directory is non-empty: {plots}")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    selected_methods = tuple(methods or paths.keys())
    neighbor_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    rmse_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = defaultdict(int)
    for method in selected_methods:
        if method not in paths:
            raise KeyError(f"diagnostic method is absent from method paths: {method}")
        scale = Path(paths[method])
        for split in SPLITS:
            episodes = _episode_dirs(scale, split)
            if not episodes:
                continue
            split_counts[split] += len(episodes)
            action_rows = _action_rows(scale, split)
            events = _event_rows(scale, split)
            event_grouped = _event_by_episode(events)
            neighbor_rows.extend(_neighbor_outcomes(scale, method, split, events, action_rows))
            eligibility_rows.append(_eligibility_summary(scale, method, split, action_rows))
            rmse_rows.extend(_accepted_rmse_rows(scale, method, split, event_grouped))
            feedback_rows.extend(_delayed_feedback_rows(scale, method, split, event_grouped))

    _write_csv(logs / "neighbor_update_outcomes.csv", neighbor_rows)
    _write_csv(logs / "action_eligibility_summary.csv", eligibility_rows)
    _write_csv(logs / "accepted_updates_rmse_relation.csv", rmse_rows)
    _write_csv(logs / "delayed_feedback_signal.csv", feedback_rows)
    _diagnostic_figure(
        neighbor_rows,
        rmse_rows,
        eligibility_rows,
        feedback_rows,
        plots / "action_update_feedback_diagnostic.png",
        plots / "action_update_feedback_diagnostic.pdf",
    )
    report = {
        "status": "complete",
        "source_splits": list(SPLITS),
        "test_access_attempts": 0,
        "methods": list(selected_methods),
        "episode_counts": dict(split_counts),
        "rows": {
            "neighbor_update_outcomes": len(neighbor_rows),
            "action_eligibility_summary": len(eligibility_rows),
            "accepted_updates_rmse_relation": len(rmse_rows),
            "delayed_feedback_signal": len(feedback_rows),
        },
        "interpretation_limits": [
            "linear_covariance_reduction is pre-restoration EKF diagnostic credit, not final physical covariance improvement",
            "RMSE relation is descriptive at episode level and is not a causal estimate",
            "missing/no-update quality proportions are represented by NA rather than zero",
            "train and validation are kept separate; test is not opened",
        ],
        "files": {
            "neighbor_update_outcomes": str(logs / "neighbor_update_outcomes.csv"),
            "action_eligibility_summary": str(logs / "action_eligibility_summary.csv"),
            "accepted_updates_rmse_relation": str(logs / "accepted_updates_rmse_relation.csv"),
            "delayed_feedback_signal": str(logs / "delayed_feedback_signal.csv"),
            "figure_png": str(plots / "action_update_feedback_diagnostic.png"),
            "figure_pdf": str(plots / "action_update_feedback_diagnostic.pdf"),
        },
    }
    _write_json(logs / "diagnostic_report.json", report)
    return report


__all__ = ["run_action_feedback_diagnostic"]
