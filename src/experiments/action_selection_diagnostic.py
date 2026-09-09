"""Read-only train/validation diagnostics for the measurement action head.

This module intentionally never enumerates a test directory.  Its outputs are
diagnostic evidence for the candidate-to-action link, not a replacement for
validation localization or communication performance figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .baseline import _read_json
from .mainline_baseline_comparison import _write_table
from ..synthetic.episode_generator import _write_json, _write_csv


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _array(value: Any, *, dtype=float) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=dtype)


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def _diagnostic_rows(scale: Path, split: str, method: str) -> list[dict[str, Any]]:
    source = scale / "diagnostics" / f"{split}_measurement_quality.jsonl"
    result: list[dict[str, Any]] = []
    for row in _read_jsonl(source):
        probabilities = _array(row.get("measurement_probabilities", []))
        logits = _array(row.get("measurement_logits_raw", []))
        mask = _array(row.get("candidate_action_mask", []))
        source_ids = [str(value) for value in row.get("candidate_source_ids", [])]
        quality = _array(row.get("candidate_quality_scores", []))
        candidate_indices = [
            index
            for index in range(min(len(mask), len(source_ids)))
            if bool(mask[index] > 0.5)
        ]
        selected = str(row.get("measurement_uav_id", "none"))
        selected_index = source_ids.index(selected) if selected in source_ids else len(source_ids)
        eligible = len(candidate_indices)
        best_logit = float(np.max(logits[candidate_indices])) if candidate_indices and logits.size > max(candidate_indices) else float("nan")
        none_logit = float(
            logits[len(source_ids)]
            if logits.size > len(source_ids)
            else row.get("none_logit_raw", float("nan"))
        )
        candidate_probabilities = [
            float(probabilities[index])
            for index in candidate_indices
            if probabilities.size > index and np.isfinite(probabilities[index])
        ]
        entropy = float(
            -np.sum(
                probabilities[probabilities > 0.0]
                * np.log(np.maximum(probabilities[probabilities > 0.0], 1e-12))
            )
            if probabilities.size
            else float("nan")
        )
        result.append(
            {
                "method": method,
                "split": split,
                "episode_id": row.get("episode_id", ""),
                "step": int(row.get("step", 0)),
                "uav_id": row.get("uav_id", ""),
                "measurement_head_mode": row.get("measurement_head_mode", ""),
                "selected_measurement": selected,
                "selected_index": int(selected_index),
                "candidate_count": int(eligible),
                "candidate_available": int(eligible > 0),
                "non_none_selected": int(selected != "none"),
                "none_probability": float(row.get("none_probability", float("nan"))),
                "selected_probability": float(row.get("measurement_probability", float("nan"))),
                "best_candidate_logit": best_logit,
                "none_logit": none_logit,
                "best_candidate_minus_none_logit": (
                    best_logit - none_logit
                    if np.isfinite(best_logit) and np.isfinite(none_logit)
                    else float("nan")
                ),
                "entropy": entropy,
                "selected_candidate_quality": (
                    float(quality[selected_index])
                    if selected_index < quality.size and selected != "none"
                    else float("nan")
                ),
                "candidate_quality_mean": (
                    float(np.mean(quality[candidate_indices]))
                    if candidate_indices and quality.size > max(candidate_indices)
                    else float("nan")
                ),
                "candidate_quality_max": (
                    float(np.max(quality[candidate_indices]))
                    if candidate_indices and quality.size > max(candidate_indices)
                    else float("nan")
                ),
                "deterministic_action": int(row.get("deterministic_action", split != "train")),
                "selection_reason": (
                    "policy_none_with_candidate"
                    if selected == "none" and eligible > 0
                    else "policy_neighbor"
                    if selected != "none"
                    else "no_candidate"
                    if eligible == 0
                    else "none"
                ),
                "candidate_probabilities_json": json.dumps(candidate_probabilities),
                "candidate_quality_scores_json": json.dumps(quality.tolist()),
            }
        )
    return result


def run_action_selection_diagnostic(
    paths: Mapping[str, str | Path],
    logs: str | Path,
    plots: str | Path,
    *,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create the v9 action-selection diagnostic from train/validation only."""

    logs = Path(logs)
    plots = Path(plots)
    for destination in (logs, plots):
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"action-selection diagnostic output is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
    selected_methods = tuple(methods or paths.keys())
    rows: list[dict[str, Any]] = []
    source_files: list[str] = []
    for method in selected_methods:
        scale = Path(paths[method])
        for split in ("train", "validation"):
            source = scale / "diagnostics" / f"{split}_measurement_quality.jsonl"
            source_files.append(str(source))
            rows.extend(_diagnostic_rows(scale, split, method))
    if not rows:
        raise ValueError("no train/validation measurement-quality diagnostics were found")
    _write_table(logs / "action_selection_rows.csv", rows)
    summaries: list[dict[str, Any]] = []
    for (method, split), group in sorted(
        ((key, [row for row in rows if (row["method"], row["split"]) == key]) for key in {(r["method"], r["split"]) for r in rows}),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        eligible = [row for row in group if row["candidate_available"]]
        summaries.append(
            {
                "method": method,
                "split": split,
                "rows": len(group),
                "eligible_rows": len(eligible),
                "non_none_rate_all_rows": _finite_mean(row["non_none_selected"] for row in group),
                "non_none_rate_when_candidate_eligible": _finite_mean(row["non_none_selected"] for row in eligible),
                "none_probability_mean": _finite_mean(row["none_probability"] for row in group),
                "none_probability_p95": float(np.nanpercentile([row["none_probability"] for row in group], 95)) if group else float("nan"),
                "best_minus_none_logit_mean": _finite_mean(row["best_candidate_minus_none_logit"] for row in eligible),
                "entropy_mean": _finite_mean(row["entropy"] for row in group),
                "candidate_quality_mean": _finite_mean(row["candidate_quality_mean"] for row in eligible),
                "selected_candidate_quality_mean": _finite_mean(row["selected_candidate_quality"] for row in eligible),
                "policy_none_with_candidate_rows": sum(row["selection_reason"] == "policy_none_with_candidate" for row in group),
            }
        )
    _write_table(logs / "action_selection_summary.csv", summaries)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = [f"{row['method']}\n{row['split']}" for row in summaries]
    x = np.arange(len(summaries))
    axes[0, 0].bar(x, [row["non_none_rate_when_candidate_eligible"] for row in summaries], color=["#2563eb" if row["split"] == "train" else "#0f766e" for row in summaries])
    axes[0, 0].set_ylabel("Non-none rate when candidate eligible")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 1].bar(x, [row["none_probability_mean"] for row in summaries], color=["#2563eb" if row["split"] == "train" else "#0f766e" for row in summaries])
    axes[0, 1].set_ylabel("Mean none probability")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[1, 0].bar(x, [row["best_minus_none_logit_mean"] for row in summaries], color=["#2563eb" if row["split"] == "train" else "#0f766e" for row in summaries])
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("Best candidate logit − none logit")
    axes[1, 1].bar(x, [row["candidate_quality_mean"] for row in summaries], color=["#2563eb" if row["split"] == "train" else "#0f766e" for row in summaries])
    axes[1, 1].set_ylabel("Mean actor-visible candidate quality")
    for axis in axes.reshape(-1):
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Seed=42 action-selection diagnostic | train vs validation", fontsize=11)
    figure_paths = []
    for extension in ("png", "pdf"):
        path = plots / f"action_selection_diagnostic.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(str(path.resolve()))
    plt.close(fig)
    report = {
        "audit_ok": True,
        "source_files": source_files,
        "test_access_attempts": 0,
        "methods": list(selected_methods),
        "rows": len(rows),
        "summary_rows": summaries,
        "claim_boundary": "diagnostic_only; validation RMSE and communication metrics remain the performance evidence",
        "figures": figure_paths,
    }
    _write_json(logs / "audit_report.json", report)
    return report


__all__ = ["run_action_selection_diagnostic"]
