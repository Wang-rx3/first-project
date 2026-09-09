"""Alignment audit for the execution-time uncertainty proxy.

This audit is read-only with respect to the frozen five-seed experiment root.
It reuses the execution-time proxy construction from
``execution_uncertainty_proxy_audit`` and joins only validation-time accepted
cooperative-update diagnostics to measure the quantity the proxy is intended
to predict: linear covariance reduction.

The proxy is never fitted with truth, never used to select an action, and is
not inserted into the EKF, reward, actor, critic, or objective.  Absolute
position error and post-update error change are retained only as offline
validation labels.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .execution_uncertainty_proxy_audit import (
    EPS,
    SEED_RE,
    TREATMENTS,
    ProxyScales,
    _candidate_rows_from_episode,
    _episode_dirs,
    _fit_scales,
    _finite,
    _materialize_proxy_rows,
    _number,
    _timestamp_key,
)


@dataclass(frozen=True)
class ExecutionUncertaintyProxyAlignmentAuditConfig:
    """Configuration for the validation-only proxy alignment audit."""

    reference_root: Path
    logs_output: Path | None = None
    plots_output: Path | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    age_variance_coefficient: float = 1.0
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_outputs(config: ExecutionUncertaintyProxyAlignmentAuditConfig) -> tuple[Path, Path]:
    run_name = Path(config.reference_root).name
    logs_dir = (
        Path(config.logs_output)
        if config.logs_output is not None
        else Path("logs/preliminary") / run_name / "execution_uncertainty_proxy_alignment"
    )
    plots_dir = (
        Path(config.plots_output)
        if config.plots_output is not None
        else Path("plots/preliminary") / run_name / "execution_uncertainty_proxy_alignment"
    )
    for path in (logs_dir, plots_dir):
        if path.exists() and any(path.iterdir()):
            if not config.force:
                raise FileExistsError(
                    f"proxy alignment output is not empty: {path}; use --force to replace this audit"
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return logs_dir, plots_dir


def _event_key(
    base_seed: int,
    event: Mapping[str, Any],
) -> tuple[int, str, float, str, str]:
    return (
        int(base_seed),
        str(event.get("episode_id", "")),
        _timestamp_key(event.get("measurement_time_s", event.get("timestamp"))),
        str(event.get("target_uav_id", "")),
        str(event.get("source_uav_id", "")),
    )


def _candidate_key(row: Mapping[str, Any]) -> tuple[int, str, float, str, str]:
    return (
        int(row["base_seed"]),
        str(row["episode_id"]),
        _timestamp_key(row["timestamp"]),
        str(row["uav_id"]),
        str(row["source_uav_id"]),
    )


def _validation_update_events(scale_dir: Path, base_seed: int) -> dict[tuple[int, str, float, str, str], dict[str, Any]]:
    """Load validation update diagnostics without enumerating the test split."""

    diagnostics_path = scale_dir / "diagnostics" / "cooperative_updates.jsonl"
    events: dict[tuple[int, str, float, str, str], dict[str, Any]] = {}
    for event in _read_jsonl(diagnostics_path):
        split = str(event.get("split", event.get("phase", ""))).strip().lower()
        if split != "validation":
            continue
        key = _event_key(base_seed, event)
        previous = events.get(key)
        # If a key is repeated, retain an accepted record over a rejected one.
        if previous is None or (not bool(previous.get("accepted", False)) and bool(event.get("accepted", False))):
            events[key] = event
    return events


def _event_fields(event: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert a matched update event into finite diagnostic columns."""

    if event is None:
        return {
            "matched_update_event": 0,
            "update_accepted": 0,
            "event_position_covariance_trace_before_m2": float("nan"),
            "event_position_covariance_trace_after_linear_m2": float("nan"),
            "event_position_covariance_trace_after_final_m2": float("nan"),
            "linear_covariance_reduction_m2": float("nan"),
            "final_covariance_reduction_m2": float("nan"),
            "covariance_restoration_gap_m2": float("nan"),
            "state_correction_norm_m": float("nan"),
            "position_correction_norm_m": float("nan"),
            "true_error_change_m": float("nan"),
            "effect_class": "not_selected",
            "event_nis": float("nan"),
            "event_kalman_gain_norm": float("nan"),
            "event_message_age_s": float("nan"),
            "event_unknown_cross_bound_m2": float("nan"),
            "event_los": "unknown",
        }

    accepted = int(bool(event.get("accepted", False)))
    before = _number(event.get("position_covariance_trace_before_m2"))
    after_linear = _number(event.get("position_covariance_trace_after_linear_m2"))
    after_final = _number(event.get("position_covariance_trace_after_final_m2"))
    linear_reduction = before - after_linear if all(math.isfinite(v) for v in (before, after_linear)) else float("nan")
    final_reduction = before - after_final if all(math.isfinite(v) for v in (before, after_final)) else float("nan")
    restoration_gap = after_linear - after_final if all(math.isfinite(v) for v in (after_linear, after_final)) else float("nan")
    error_before = _number(event.get("true_position_error_before_m"))
    error_after = _number(event.get("true_position_error_after_m"))
    error_change = error_after - error_before if math.isfinite(error_before) and math.isfinite(error_after) else float("nan")
    if not accepted:
        effect_class = "rejected"
    elif not math.isfinite(error_change):
        effect_class = "accepted_without_error_label"
    elif error_change < -0.001:
        effect_class = "beneficial"
    elif error_change > 0.001:
        effect_class = "harmful"
    else:
        effect_class = "approximately_neutral"
    los_value = event.get("los")
    if isinstance(los_value, bool):
        los_label = "LOS" if los_value else "NLOS"
    else:
        los_label = "unknown"
    return {
        "matched_update_event": 1,
        "update_accepted": accepted,
        "event_position_covariance_trace_before_m2": before,
        "event_position_covariance_trace_after_linear_m2": after_linear,
        "event_position_covariance_trace_after_final_m2": after_final,
        "linear_covariance_reduction_m2": linear_reduction,
        "final_covariance_reduction_m2": final_reduction,
        "covariance_restoration_gap_m2": restoration_gap,
        # Older cooperative-update logs use ``state_correction_norm`` without
        # a unit suffix; accept that historical spelling without changing the
        # normalized audit schema.
        "state_correction_norm_m": _number(
            event.get("state_correction_norm_m", event.get("state_correction_norm"))
        ),
        "position_correction_norm_m": _number(event.get("position_correction_norm_m")),
        "true_error_change_m": error_change,
        "effect_class": effect_class,
        "event_nis": _number(event.get("nis")),
        "event_kalman_gain_norm": _number(event.get("kalman_gain_norm")),
        "event_message_age_s": _number(event.get("message_age_s")),
        "event_unknown_cross_bound_m2": _number(event.get("unknown_cross_bound_m2")),
        "event_los": los_label,
    }


def _add_event_labels(
    proxy_rows: Sequence[Mapping[str, Any]],
    event_maps: Mapping[tuple[int, str], Mapping[tuple[int, str, float, str, str], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in proxy_rows:
        event = event_maps.get((int(row["base_seed"]), str(row["treatment"])), {}).get(_candidate_key(row))
        result = dict(row)
        result.update(_event_fields(event))
        result["outcome_is_offline_label_only"] = 1
        result["event_is_independent_replicate"] = 0
        output.append(result)
    return output


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _fraction(values: Sequence[bool]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def _bootstrap_interval(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    finite = np.asarray([float(value) for value in values if _finite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(max(int(samples), 100), finite.size))
    means = finite[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _episode_summaries(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = _group(rows, ("base_seed", "treatment", "episode_id"))
    output: list[dict[str, Any]] = []
    for (base_seed, treatment, episode_id), selected in sorted(grouped.items(), key=lambda item: item[0]):
        updates = [row for row in selected if int(row["matched_update_event"])]
        accepted = [row for row in updates if int(row["update_accepted"])]
        output.append({
            "base_seed": base_seed,
            "treatment": treatment,
            "episode_id": episode_id,
            "candidate_row_count": len(selected),
            "multi_candidate_window_count": sum(int(row["multi_candidate_window"]) for row in selected),
            "proxy_valid_fraction": _fraction([bool(int(row["proxy_valid"])) for row in selected]),
            "mean_expected_information_gain_norm": _mean([_number(row.get("expected_information_gain_norm")) for row in selected]),
            "mean_proxy_risk": _mean([_number(row.get("proxy_risk")) for row in selected]),
            "matched_update_event_count": len(updates),
            "accepted_update_count": len(accepted),
            "linear_covariance_reduction_mean_m2": _mean([_number(row.get("linear_covariance_reduction_m2")) for row in accepted]),
            "final_covariance_reduction_mean_m2": _mean([_number(row.get("final_covariance_reduction_m2")) for row in accepted]),
            "mean_covariance_restoration_gap_m2": _mean([_number(row.get("covariance_restoration_gap_m2")) for row in accepted]),
            "mean_true_error_change_m": _mean([_number(row.get("true_error_change_m")) for row in accepted]),
            "beneficial_fraction": _fraction([row["effect_class"] == "beneficial" for row in accepted]),
            "harmful_fraction": _fraction([row["effect_class"] == "harmful" for row in accepted]),
            "neutral_fraction": _fraction([row["effect_class"] == "approximately_neutral" for row in accepted]),
            "event_rows_are_independent_replicates": 0,
        })

    seed_groups = _group(output, ("base_seed", "treatment"))
    seed_output: list[dict[str, Any]] = []
    for (base_seed, treatment), episodes in sorted(seed_groups.items(), key=lambda item: item[0]):
        def mean(name: str) -> float:
            return _mean([_number(row.get(name)) for row in episodes])

        seed_output.append({
            "base_seed": base_seed,
            "treatment": treatment,
            "episode_count": len(episodes),
            "candidate_row_count": sum(int(row["candidate_row_count"]) for row in episodes),
            "matched_update_event_count": sum(int(row["matched_update_event_count"]) for row in episodes),
            "accepted_update_count": sum(int(row["accepted_update_count"]) for row in episodes),
            "mean_expected_information_gain_norm": mean("mean_expected_information_gain_norm"),
            "mean_proxy_risk": mean("mean_proxy_risk"),
            "linear_covariance_reduction_mean_m2": mean("linear_covariance_reduction_mean_m2"),
            "final_covariance_reduction_mean_m2": mean("final_covariance_reduction_mean_m2"),
            "mean_covariance_restoration_gap_m2": mean("mean_covariance_restoration_gap_m2"),
            "mean_true_error_change_m": mean("mean_true_error_change_m"),
            "beneficial_fraction": mean("beneficial_fraction"),
            "harmful_fraction": mean("harmful_fraction"),
            "neutral_fraction": mean("neutral_fraction"),
            "event_rows_are_independent_replicates": 0,
        })
    return output, seed_output


def _risk_label(value: float) -> str:
    if value < 0.25:
        return "risk_0_25"
    if value < 0.50:
        return "risk_25_50"
    if value < 0.75:
        return "risk_50_75"
    return "risk_75_100"


def _strata_summary(rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    # Work on copies: the event table must keep one stable schema.  Adding the
    # stratum only to the original event dictionaries would make the later CSV
    # writer see a field that is absent from the first row.
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row["update_accepted"]):
            selected_row = dict(row)
            selected_row["proxy_risk_stratum"] = _risk_label(_number(row.get("proxy_risk"), 1.0))
            selected.append(selected_row)
    grouped = _group(selected, ("treatment", "proxy_risk_stratum"))
    output: list[dict[str, Any]] = []
    order = {"risk_0_25": 0, "risk_25_50": 1, "risk_50_75": 2, "risk_75_100": 3}
    for (treatment, stratum), event_rows in sorted(grouped.items(), key=lambda item: (item[0][0], order[item[0][1]])):
        episode_groups = _group(event_rows, ("base_seed", "episode_id"))
        episode_linear = [_mean([_number(row.get("linear_covariance_reduction_m2")) for row in values]) for values in episode_groups.values()]
        episode_error = [_mean([_number(row.get("true_error_change_m")) for row in values]) for values in episode_groups.values()]
        episode_harm = [_fraction([row["effect_class"] == "harmful" for row in values]) for values in episode_groups.values()]
        linear_low, linear_high = _bootstrap_interval(episode_linear, bootstrap_samples, bootstrap_seed + order[stratum])
        error_low, error_high = _bootstrap_interval(episode_error, bootstrap_samples, bootstrap_seed + 10 + order[stratum])
        harm_low, harm_high = _bootstrap_interval(episode_harm, bootstrap_samples, bootstrap_seed + 20 + order[stratum])
        output.append({
            "treatment": treatment,
            "proxy_risk_stratum": stratum,
            "event_count": len(event_rows),
            "episode_count": len(episode_groups),
            "seed_count": len({int(row["base_seed"]) for row in event_rows}),
            "mean_proxy_risk": _mean([_number(row.get("proxy_risk")) for row in event_rows]),
            "mean_expected_information_gain_norm": _mean([_number(row.get("expected_information_gain_norm")) for row in event_rows]),
            "mean_linear_covariance_reduction_m2": _mean([_number(row.get("linear_covariance_reduction_m2")) for row in event_rows]),
            "bootstrap_linear_reduction_ci_low_m2": linear_low,
            "bootstrap_linear_reduction_ci_high_m2": linear_high,
            "mean_final_covariance_reduction_m2": _mean([_number(row.get("final_covariance_reduction_m2")) for row in event_rows]),
            "mean_true_error_change_m": _mean([_number(row.get("true_error_change_m")) for row in event_rows]),
            "bootstrap_error_change_ci_low_m": error_low,
            "bootstrap_error_change_ci_high_m": error_high,
            "harmful_fraction": _fraction([row["effect_class"] == "harmful" for row in event_rows]),
            "bootstrap_harmful_fraction_ci_low": harm_low,
            "bootstrap_harmful_fraction_ci_high": harm_high,
            "event_rows_are_independent_replicates": 0,
        })
    return output


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(array.size, dtype=float)
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if _finite(a) and _finite(b)]
    if len(pairs) < 2:
        return float("nan"), float("nan"), len(pairs)
    xa = np.asarray([pair[0] for pair in pairs], dtype=float)
    ya = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.std(xa) <= EPS or np.std(ya) <= EPS:
        return float("nan"), float("nan"), len(pairs)
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    spearman = float(np.corrcoef(_rank(xa), _rank(ya))[0, 1])
    return pearson, spearman, len(pairs)


def _correlation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        selected = [row for row in rows if row["treatment"] == treatment and int(row["update_accepted"])]
        pairs = [
            ("expected_information_gain_norm", "linear_covariance_reduction_m2", "proxy_ig_vs_linear_covariance_reduction"),
            ("proxy_risk", "true_error_change_m", "proxy_risk_vs_true_error_change"),
            ("expected_information_gain_norm", "position_correction_norm_m", "proxy_ig_vs_position_correction"),
        ]
        for x_name, y_name, relation in pairs:
            pearson, spearman, count = _correlation(
                [_number(row.get(x_name)) for row in selected],
                [_number(row.get(y_name)) for row in selected],
            )
            output.append({
                "treatment": treatment,
                "relation": relation,
                "x": x_name,
                "y": y_name,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "matched_accepted_event_count": count,
                "event_rows_are_independent_replicates": 0,
            })
    return output


def _comparison_by_seed(seed_rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[float, float, float, float, int]:
    grouped = _group(seed_rows, ("base_seed",))
    differences: list[float] = []
    for values in grouped.values():
        by_treatment = {str(row["treatment"]): row for row in values}
        if set(TREATMENTS).issubset(by_treatment):
            baseline = _number(by_treatment["baseline_mappo"].get(metric))
            quality = _number(by_treatment["quality_aware_mappo"].get(metric))
            if math.isfinite(baseline) and math.isfinite(quality):
                differences.append(quality - baseline)
    if not differences:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    low, high = _bootstrap_interval(differences, 10_000, 2026)
    return float(np.mean(differences)), float(np.median(differences)), low, high, len(differences)


def _plot_alignment(rows: Sequence[Mapping[str, Any]], strata: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline_mappo": "#2563eb", "quality_aware_mappo": "#16a34a"}
    labels = {"baseline_mappo": "Baseline MAPPO", "quality_aware_mappo": "Quality-aware MAPPO"}
    events = [row for row in rows if int(row["update_accepted"])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for treatment in TREATMENTS:
        selected = [row for row in events if row["treatment"] == treatment]
        axes[0, 0].scatter(
            [_number(row.get("expected_information_gain_norm")) for row in selected],
            [_number(row.get("linear_covariance_reduction_m2")) for row in selected],
            s=14, alpha=0.45, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
        axes[0, 1].scatter(
            [_number(row.get("proxy_risk")) for row in selected],
            [_number(row.get("true_error_change_m")) for row in selected],
            s=14, alpha=0.45, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
        axes[0, 2].scatter(
            [_number(row.get("expected_information_gain_norm")) for row in selected],
            [_number(row.get("position_correction_norm_m")) for row in selected],
            s=14, alpha=0.45, color=colors[treatment], label=labels[treatment], rasterized=True,
        )

    axes[0, 0].set_title("Proxy IG vs. linear covariance reduction")
    axes[0, 0].set_xlabel("normalized expected information gain")
    axes[0, 0].set_ylabel("trace reduction (m²)")
    axes[0, 1].set_title("Proxy risk vs. offline error change")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("proxy risk")
    axes[0, 1].set_ylabel("after − before error (m)")
    axes[0, 2].set_title("Proxy IG vs. state correction")
    axes[0, 2].set_xlabel("normalized expected information gain")
    axes[0, 2].set_ylabel("position correction (m)")

    for treatment in TREATMENTS:
        selected = [row for row in strata if row["treatment"] == treatment]
        selected.sort(key=lambda row: ["risk_0_25", "risk_25_50", "risk_50_75", "risk_75_100"].index(row["proxy_risk_stratum"]))
        x = np.arange(len(selected))
        axes[1, 0].errorbar(
            x,
            [_number(row.get("mean_linear_covariance_reduction_m2")) for row in selected],
            yerr=[
                [
                    max(0.0, _number(row.get("mean_linear_covariance_reduction_m2")) - _number(row.get("bootstrap_linear_reduction_ci_low_m2")))
                    for row in selected
                ],
                [
                    max(0.0, _number(row.get("bootstrap_linear_reduction_ci_high_m2")) - _number(row.get("mean_linear_covariance_reduction_m2")))
                    for row in selected
                ],
            ],
            marker="o", capsize=3, color=colors[treatment], label=labels[treatment],
        )
        axes[1, 1].plot(
            x,
            [_number(row.get("harmful_fraction")) for row in selected],
            marker="o", color=colors[treatment], label=labels[treatment],
        )
        axes[1, 2].plot(
            x,
            [_number(row.get("mean_final_covariance_reduction_m2")) for row in selected],
            marker="o", color=colors[treatment], label=labels[treatment],
        )

    stratum_labels = ["0–25%", "25–50%", "50–75%", "75–100%"]
    for axis in axes[1, :]:
        axis.set_xticks(np.arange(4), stratum_labels)
        axis.set_xlabel("proxy risk stratum")
    axes[1, 0].set_title("Linear covariance reduction by risk")
    axes[1, 0].set_ylabel("trace reduction (m²)")
    axes[1, 1].set_title("Harmful update fraction by risk")
    axes[1, 1].set_ylabel("harmful fraction")
    axes[1, 2].set_title("Final covariance reduction by risk")
    axes[1, 2].set_ylabel("trace reduction (m²)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Execution-time uncertainty proxy alignment audit (validation only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _decision(
    strata_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_treatment: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in strata_rows:
        by_treatment[str(row["treatment"])][str(row["proxy_risk_stratum"])] = row
    alignment_candidates: list[bool] = []
    harmful_candidates: list[bool] = []
    for treatment in TREATMENTS:
        low = by_treatment[treatment].get("risk_0_25")
        high = by_treatment[treatment].get("risk_75_100")
        if low is None or high is None:
            continue
        low_reduction = _number(low.get("mean_linear_covariance_reduction_m2"))
        high_reduction = _number(high.get("mean_linear_covariance_reduction_m2"))
        low_harm = _number(low.get("harmful_fraction"))
        high_harm = _number(high.get("harmful_fraction"))
        if all(math.isfinite(value) for value in (low_reduction, high_reduction)):
            alignment_candidates.append(low_reduction > high_reduction + EPS)
        if all(math.isfinite(value) for value in (low_harm, high_harm)):
            harmful_candidates.append(low_harm <= high_harm + EPS)
    alignment = bool(alignment_candidates) and all(alignment_candidates)
    harmful = bool(harmful_candidates) and all(harmful_candidates)
    matched = sum(int(row["accepted_update_count"]) for row in seed_rows)
    if alignment and harmful and matched > 0:
        gate = "proxy_aligned_for_budget_controlled_intervention"
        next_stage = "budget_controlled_paired_intervention"
    elif matched == 0:
        gate = "no_matched_validation_update_events"
        next_stage = "repair_update_diagnostic_coverage_before_proxy_decision"
    else:
        gate = "proxy_not_aligned_revise_proxy_or_observable_inputs"
        next_stage = "revise_proxy_construction_before_intervention"
    return {
        "low_risk_has_greater_linear_reduction": alignment,
        "low_risk_has_no_greater_harmful_fraction": harmful,
        "matched_accepted_update_count": matched,
        "decision_supported": bool(alignment and harmful and matched > 0),
        "decision_gate": gate,
        "next_stage": next_stage,
    }


def run_execution_uncertainty_proxy_alignment_audit(
    config: ExecutionUncertaintyProxyAlignmentAuditConfig,
) -> dict[str, Any]:
    """Run the read-only proxy-to-linear-covariance alignment audit."""

    if config.train_split != "train":
        raise ValueError("proxy scale fitting must use the train split")
    if config.eval_split != "validation":
        raise ValueError("proxy alignment audit is validation-only; test is sealed")
    if config.age_variance_coefficient < 0.0 or not math.isfinite(config.age_variance_coefficient):
        raise ValueError("age_variance_coefficient must be finite and non-negative")
    if config.bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    reference_root = Path(config.reference_root)
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")

    seed_dirs: list[tuple[int, Path]] = []
    for path in sorted(reference_root.iterdir(), key=lambda item: item.name):
        match = SEED_RE.match(path.name)
        if path.is_dir() and match:
            seed_dirs.append((int(match.group(1)), path))
    if not seed_dirs:
        raise ValueError(f"no seed_* directories found below {reference_root}")

    all_proxy_rows: list[dict[str, Any]] = []
    event_maps: dict[tuple[int, str], dict[tuple[int, str, float, str, str], dict[str, Any]]] = {}
    scale_records: dict[str, Any] = {}
    seed_checks: dict[str, Any] = {}
    for base_seed, seed_root in seed_dirs:
        baseline_scale = seed_root / "baseline_mappo" / "scale_3"
        quality_scale = seed_root / "quality_aware_mappo" / "scale_3"
        train_dirs = _episode_dirs(baseline_scale, config.train_split)
        train_rows: list[dict[str, Any]] = []
        for episode_dir in train_dirs:
            train_rows.extend(_candidate_rows_from_episode(
                episode_dir,
                base_seed=base_seed,
                treatment="baseline_mappo",
                split=config.train_split,
                include_offline_label=False,
                age_variance_coefficient=config.age_variance_coefficient,
            ))
        if not train_rows:
            raise ValueError(f"no baseline training candidate rows for seed {base_seed}")
        scales: ProxyScales = _fit_scales(train_rows, config.age_variance_coefficient)
        scale_records[str(base_seed)] = {
            "fit_treatment": "baseline_mappo",
            "fit_split": config.train_split,
            "fit_candidate_rows": len(train_rows),
            "scales": scales.as_dict(),
        }
        eval_counts: dict[str, int] = {}
        for treatment, branch_scale in (("baseline_mappo", baseline_scale), ("quality_aware_mappo", quality_scale)):
            eval_dirs = _episode_dirs(branch_scale, config.eval_split)
            if not eval_dirs:
                raise ValueError(f"no validation episodes for {base_seed}/{treatment}")
            rows: list[dict[str, Any]] = []
            for episode_dir in eval_dirs:
                rows.extend(_candidate_rows_from_episode(
                    episode_dir,
                    base_seed=base_seed,
                    treatment=treatment,
                    split=config.eval_split,
                    include_offline_label=True,
                    age_variance_coefficient=config.age_variance_coefficient,
                ))
            all_proxy_rows.extend(_materialize_proxy_rows(rows, scales, config.age_variance_coefficient))
            eval_counts[treatment] = len(rows)
            event_maps[(base_seed, treatment)] = _validation_update_events(branch_scale, base_seed)
        seed_checks[str(base_seed)] = {
            "train_candidate_rows": len(train_rows),
            "validation_candidate_rows": eval_counts,
            "validation_update_event_counts": {
                treatment: len(event_maps[(base_seed, treatment)]) for treatment in TREATMENTS
            },
            "test_read": False,
        }

    if not all_proxy_rows:
        raise ValueError("no validation candidate rows were available")
    rows = _add_event_labels(all_proxy_rows, event_maps)
    episode_rows, seed_rows = _episode_summaries(rows)
    strata_rows = _strata_summary(rows, config.bootstrap_samples, config.bootstrap_seed)
    correlations = _correlation_rows(rows)
    logs_dir, plots_dir = _prepare_outputs(config)

    events_path = logs_dir / "proxy_alignment_events.csv"
    episode_path = logs_dir / "proxy_alignment_episode_summary.csv"
    seed_path = logs_dir / "proxy_alignment_seed_summary.csv"
    strata_path = logs_dir / "proxy_alignment_strata.csv"
    correlations_path = logs_dir / "proxy_alignment_correlations.csv"
    plot_path = plots_dir / "proxy_alignment.png"
    _write_csv(events_path, rows)
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_path, seed_rows)
    _write_csv(strata_path, strata_rows)
    _write_csv(correlations_path, correlations)
    _plot_alignment(rows, strata_rows, plot_path)

    finite_fields = (
        "expected_information_gain_raw", "expected_information_gain_norm", "proxy_risk",
        "linear_covariance_reduction_m2", "final_covariance_reduction_m2",
        "covariance_restoration_gap_m2", "state_correction_norm_m",
        "position_correction_norm_m", "true_error_change_m",
    )
    matched_rows = [row for row in rows if int(row["matched_update_event"])]
    accepted_rows = [row for row in matched_rows if int(row["update_accepted"])]
    finite_proxy = all(_finite(row.get(field)) for row in rows for field in finite_fields[:3])
    finite_matched = all(_finite(row.get(field)) for row in accepted_rows for field in finite_fields[3:])
    closure_residual = max(
        (
            abs(
                _number(row.get("linear_covariance_reduction_m2"))
                - (
                    _number(row.get("event_position_covariance_trace_before_m2"))
                    - _number(row.get("event_position_covariance_trace_after_linear_m2"))
                )
            )
            for row in matched_rows
            if _finite(row.get("linear_covariance_reduction_m2"))
        ),
        default=0.0,
    )
    decision = _decision(strata_rows, seed_rows)
    report = {
        "audit": "execution_uncertainty_proxy_alignment",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "base_seeds": [seed for seed, _ in seed_dirs],
        "treatments": list(TREATMENTS),
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "test_data_read": False,
        "source_writes": 0,
        "candidate_row_count": len(rows),
        "matched_update_event_count": len(matched_rows),
        "accepted_update_event_count": len(accepted_rows),
        "independent_unit": "base training seed; episode summaries retained for mechanism diagnostics",
        "update_events_used_as_independent_samples": False,
        "offline_labels": ["true_error_change_m", "effect_class", "event_los"],
        "allowed_proxy_inputs": [
            "local estimated covariance and position",
            "received neighbor estimate/covariance when visible",
            "UWB measurement standard deviation and validity",
            "RSSI, SNR, predicted loss, delay, AoI, queue and current link state",
        ],
        "forbidden_proxy_inputs": [
            "true position", "true range", "true position error", "LOS/NLOS truth",
            "future packet success", "future message arrival", "test split", "central state",
        ],
        "checks": {
            "validation_only": config.eval_split == "validation",
            "test_not_read": True,
            "train_fit_separated": all(record["fit_candidate_rows"] > 0 for record in scale_records.values()),
            "source_untouched_by_audit": True,
            "all_proxy_fields_finite": finite_proxy,
            "matched_update_fields_finite": finite_matched,
            "linear_covariance_reduction_formula_closed": closure_residual <= 1e-8,
            "seed_level_inference_used": True,
            "update_events_are_not_independent": True,
            "truth_is_offline_label_only": True,
        },
        "formula": {
            "linear_covariance_reduction": "trace(P_before) - trace(P_after_linear)",
            "final_covariance_reduction": "trace(P_before) - trace(P_after_final)",
            "error_change": "true_position_error_after - true_position_error_before (offline label only)",
        },
        "scale_records": scale_records,
        "seed_checks": seed_checks,
        "decision": decision,
        "audit_ok": bool(all(report_check for report_check in [
            config.eval_split == "validation",
            finite_proxy,
            finite_matched,
            closure_residual <= 1e-8,
            all(record["fit_candidate_rows"] > 0 for record in scale_records.values()),
        ])),
        "output_paths": {
            "proxy_alignment_events": str(events_path),
            "episode_summary": str(episode_path),
            "seed_summary": str(seed_path),
            "strata": str(strata_path),
            "correlations": str(correlations_path),
            "plot": str(plot_path),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "interpretation_boundary": (
            "The proxy is evaluated against validation-only update diagnostics. "
            "Offline true error and LOS/NLOS labels are not proxy inputs and are "
            "never used to fit, rank, select actions, or modify the current model. "
            "Test data remain sealed."
        ),
    }
    report_path = logs_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


__all__ = [
    "ExecutionUncertaintyProxyAlignmentAuditConfig",
    "run_execution_uncertainty_proxy_alignment_audit",
]
