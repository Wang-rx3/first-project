"""Validation-only audit for localization metric-definition consistency.

The formal experiment historically stores a component-wise coordinate RMSE in
``position_rmse_m``.  EKF diagnostics, however, reason about one Euclidean
position error per UAV and timestamp.  This module computes the latter without
changing any source episode, then reports both definitions side by side.  All
inference is reduced to one row per training seed; episodes are paired
diagnostics and are not treated as independent replicates.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import position_error_metrics


DEFAULT_SEEDS = (12136, 13145, 14154, 15163, 16172)
DEFAULT_BRANCHES = (
    "baseline_mappo",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo",
    "communication_constrained_proxy_guided_mappo",
)


@dataclass(frozen=True)
class MetricConsistencyAuditConfig:
    """Configuration for the validation-only metric audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "communication_constrained_proxy_mappo_seedset_12136_16172/"
        "metric_consistency_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    expected_episodes: int = 20
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("metric consistency audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty unique sequence")
        if self.expected_episodes < 1:
            raise ValueError("expected_episodes must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: MetricConsistencyAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[metric-consistency] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_mean(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_number(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float(default)


def _safe_std(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_number(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size > 1 else float(default)


def _bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _seed_branch_dirs(
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[tuple[int, str, Path]]:
    output: list[tuple[int, str, Path]] = []
    for seed in seeds:
        for branch in branches:
            scale = reference_root / f"seed_{int(seed)}" / branch / "scale_3"
            if not scale.is_dir():
                raise FileNotFoundError(f"missing scale_3 directory: {scale}")
            validation = scale / "validation"
            if not validation.is_dir():
                raise FileNotFoundError(f"missing validation directory: {validation}")
            output.append((int(seed), str(branch), scale))
    return output


def _episode_dirs(scale_root: Path, split: str) -> list[Path]:
    directory = scale_root / split
    return sorted(
        (path for path in directory.iterdir() if path.is_dir() and path.name.startswith("episode_")),
        key=lambda path: path.name,
    )


def _metric_rows(scale_root: Path, split: str) -> dict[str, dict[str, str]]:
    path = scale_root / "diagnostics" / f"{split}_episode_metrics.csv"
    if not path.is_file():
        path = scale_root / "diagnostics" / "episode_metrics.csv"
    return {
        str(row.get("episode_id", "")): row
        for row in _read_csv(path)
        if str(row.get("split", "")) == split and str(row.get("episode_id", ""))
    }


def _accepted_updates(scale_root: Path, split: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    path = scale_root / "diagnostics" / f"{split}_cooperative_updates.jsonl"
    if not path.is_file():
        path = scale_root / "diagnostics" / "cooperative_updates.jsonl"
    for row in _read_jsonl(path):
        if str(row.get("split", "")) != split:
            continue
        if not bool(row.get("accepted", False)):
            continue
        episode_id = str(row.get("episode_id", ""))
        if episode_id:
            counts[episode_id] += 1
    return dict(counts)


def _episode_rows(
    *,
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    split: str,
    config: MetricConsistencyAuditConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed, branch, scale_root in _seed_branch_dirs(reference_root, seeds, branches):
        metric_rows = _metric_rows(scale_root, split)
        update_counts = _accepted_updates(scale_root, split)
        episode_paths = _episode_dirs(scale_root, split)
        for index, episode_path in enumerate(episode_paths, start=1):
            episode_id = episode_path.name
            source = metric_rows.get(episode_id)
            if source is None:
                raise ValueError(f"missing {split} metrics for {seed}/{branch}/{episode_id}")
            position = position_error_metrics(
                _read_csv(episode_path / "ground_truth.csv"),
                _read_csv(episode_path / "estimates.csv"),
            )
            component_rmse = _number(source.get("position_rmse_m"))
            canonical_rmse = _number(position.get("rmse_norm_m"))
            canonical_p95 = _number(position.get("p95_norm_m"))
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "split": split,
                    "episode_id": episode_id,
                    "scenario_id": str(source.get("scenario_id", "")),
                    "position_rmse_norm_m": canonical_rmse,
                    "position_error_p95_norm_m": canonical_p95,
                    "position_rmse_component_m": component_rmse,
                    "j_loc": _number(source.get("j_loc")),
                    "j_com": _number(source.get("j_com")),
                    "mean_reward": _number(source.get("mean_reward")),
                    "accepted_update_count": int(update_counts.get(episode_id, 0)),
                    "eventful_episode": int(update_counts.get(episode_id, 0) > 0),
                    "canonical_sample_count": int(position.get("sample_count", 0.0)),
                    "metric_definition": "vector_norm_rmse_p95",
                    "truth_used_offline_only": 1,
                }
            )
            if config.progress and (index == 1 or index == len(episode_paths) or index % 5 == 0):
                _progress(config, f"{branch} seed={seed}: processed validation episode {index}/{len(episode_paths)}")
    return output


def _seed_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        groups[(int(row["base_seed"]), str(row["branch"]))].append(row)
    metrics = (
        "position_rmse_norm_m",
        "position_error_p95_norm_m",
        "position_rmse_component_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "accepted_update_count",
        "eventful_episode",
        "canonical_sample_count",
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in branches:
            group = groups.get((int(seed), str(branch)), [])
            if not group:
                raise ValueError(f"no episode rows for seed={seed}, branch={branch}")
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": str(branch),
                "split": "validation",
                "sample_unit": "training_seed_after_episode_mean",
                "episode_count": len(group),
                "truth_used_offline_only": 1,
            }
            for metric in metrics:
                row[metric] = _safe_mean(item.get(metric) for item in group)
            row["component_to_norm_ratio"] = (
                float(row["position_rmse_component_m"]) / float(row["position_rmse_norm_m"])
                if float(row["position_rmse_norm_m"]) > 0.0
                else float("nan")
            )
            output.append(row)
    return output


def _paired_contrasts(
    seed_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
    config: MetricConsistencyAuditConfig,
) -> list[dict[str, Any]]:
    by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in seed_rows}
    baseline = str(branches[0])
    metrics = (
        "position_rmse_norm_m",
        "position_error_p95_norm_m",
        "position_rmse_component_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "accepted_update_count",
        "eventful_episode",
    )
    output: list[dict[str, Any]] = []
    for branch in branches[1:]:
        for metric in metrics:
            differences = [
                _number(by_key[(int(seed), str(branch))].get(metric))
                - _number(by_key[(int(seed), baseline)].get(metric))
                for seed in seeds
            ]
            differences = [value for value in differences if math.isfinite(value)]
            low, high = _bootstrap_ci(
                differences,
                config.bootstrap_samples,
                config.bootstrap_seed + len(output),
            )
            output.append(
                {
                    "contrast": f"{branch}_minus_{baseline}",
                    "metric": metric,
                    "paired_seed_count": len(differences),
                    "total_seed_count": len(seeds),
                    "mean_paired_difference": _safe_mean(differences),
                    "median_paired_difference": float(np.median(differences)) if differences else float("nan"),
                    "paired_difference_std": _safe_std(differences),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": int(math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0)),
                    "seed_level_inference": 1,
                }
            )
    return output


def _prepare_outputs(config: MetricConsistencyAuditConfig) -> tuple[Path, Path]:
    logs_dir = Path(config.output_dir)
    plots_dir = (
        Path(config.plots_dir)
        if config.plots_dir is not None
        else Path("plots") / "preliminary" / Path(config.reference_root).name / "metric_consistency_audit"
    )
    managed_logs = (
        "episode_metric_consistency.csv",
        "seed_metric_consistency.csv",
        "paired_contrasts.csv",
        "audit_report.json",
    )
    managed_plots = ("metric_consistency.png",)
    for directory in (logs_dir, plots_dir):
        directory = directory.resolve()
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"audit output is not empty: {directory}; use --force")
        directory.mkdir(parents=True, exist_ok=True)
        if config.force:
            managed = managed_logs if directory == logs_dir.resolve() else managed_plots
            for name in managed:
                path = directory / name
                if path.is_file():
                    path.unlink()
    return logs_dir, plots_dir


def _plot_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    branches: Sequence[str],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "#16a34a",
        "communication_constrained_proxy_guided_mappo": "#dc2626",
    }
    labels = {
        "baseline_mappo": "Baseline",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "Proxy-guided",
        "communication_constrained_proxy_guided_mappo": "Communication-constrained",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axis = axes[0, 0]
    for branch in branches:
        selected = [row for row in seed_rows if str(row["branch"]) == str(branch)]
        selected.sort(key=lambda row: int(row["base_seed"]))
        axis.plot(
            [str(row["base_seed"]) for row in selected],
            [_number(row["position_rmse_norm_m"]) for row in selected],
            marker="o",
            label=labels.get(str(branch), str(branch)),
            color=colors.get(str(branch)),
        )
    axis.set_title("Canonical vector-norm RMSE by seed")
    axis.set_ylabel("RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    x = np.arange(len(branches), dtype=float)
    width = 0.8 / max(1, len(branches))
    for offset, branch in enumerate(branches):
        values = [
            _number(row["component_to_norm_ratio"])
            for row in seed_rows
            if str(row["branch"]) == str(branch)
        ]
        axis.boxplot(values, positions=[x[offset]], widths=width, patch_artist=True, boxprops={"facecolor": colors.get(branch, "#6b7280")})
    axis.axhline(np.sqrt(3.0), color="#6b7280", linestyle="--", linewidth=1, label=r"$\sqrt{3}$ reference")
    axis.set_xticks(x, [labels.get(branch, branch) for branch in branches], rotation=20, ha="right")
    axis.set_title("Legacy component RMSE / canonical norm RMSE")
    axis.set_ylabel("ratio")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for branch in branches:
        values = [
            _number(row["position_rmse_component_m"])
            for row in seed_rows
            if str(row["branch"]) == str(branch)
        ]
        canonical = [
            _number(row["position_rmse_norm_m"])
            for row in seed_rows
            if str(row["branch"]) == str(branch)
        ]
        axis.scatter(canonical, values, label=labels.get(branch, branch), color=colors.get(branch), alpha=0.85)
    finite_values = [
        (_number(row["position_rmse_norm_m"]), _number(row["position_rmse_component_m"]))
        for row in seed_rows
    ]
    max_value = max((max(pair) for pair in finite_values if all(math.isfinite(value) for value in pair)), default=1.0)
    axis.plot([0.0, max_value], [0.0, max_value], color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("Metric-definition comparison")
    axis.set_xlabel("canonical norm RMSE (m)")
    axis.set_ylabel("legacy component RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    rmse_contrasts = [
        row for row in contrasts
        if str(row["metric"]) == "position_rmse_norm_m"
    ]
    positions = np.arange(len(rmse_contrasts), dtype=float)
    values = np.asarray([_number(row["mean_paired_difference"]) for row in rmse_contrasts], dtype=float)
    lows = np.asarray([_number(row["bootstrap_95_ci_low"]) for row in rmse_contrasts], dtype=float)
    highs = np.asarray([_number(row["bootstrap_95_ci_high"]) for row in rmse_contrasts], dtype=float)
    if rmse_contrasts:
        axis.errorbar(positions, values, yerr=np.vstack([values - lows, highs - values]), fmt="o", capsize=4, color="#111827")
    axis.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1)
    axis.set_xticks(positions, [str(row["contrast"]).replace("_minus_", "\n−") for row in rmse_contrasts], rotation=20)
    axis.set_title("Canonical RMSE paired contrasts")
    axis.set_ylabel("seed-level difference (m)")
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Localization metric consistency audit (validation only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_metric_consistency_audit(config: MetricConsistencyAuditConfig) -> dict[str, Any]:
    """Run the read-only validation metric-definition audit."""

    if config.split != "validation":
        raise ValueError("metric consistency audit is validation-only")
    reference_root = Path(config.reference_root).resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")

    episode_rows = _episode_rows(
        reference_root=reference_root,
        seeds=config.seeds,
        branches=config.branches,
        split=config.split,
        config=config,
    )
    seed_rows = _seed_rows(episode_rows, config.seeds, config.branches)
    contrasts = _paired_contrasts(seed_rows, config.seeds, config.branches, config)
    cell_counts = {
        f"{seed}::{branch}": sum(
            int(row["base_seed"]) == int(seed) and str(row["branch"]) == str(branch)
            for row in episode_rows
        )
        for seed in config.seeds
        for branch in config.branches
    }
    required_numeric = (
        "position_rmse_norm_m",
        "position_error_p95_norm_m",
        "position_rmse_component_m",
        "j_loc",
        "j_com",
        "mean_reward",
    )
    all_finite = all(
        math.isfinite(_number(row.get(field)))
        for row in episode_rows
        for field in required_numeric
    )
    component_norm_differences = [
        abs(_number(row["position_rmse_component_m"]) - _number(row["position_rmse_norm_m"]))
        for row in episode_rows
    ]
    component_norm_differences = [value for value in component_norm_differences if math.isfinite(value)]
    ratios = [_number(row["component_to_norm_ratio"]) for row in seed_rows]
    ratios = [value for value in ratios if math.isfinite(value)]
    logs_dir, plots_dir = _prepare_outputs(config)
    episode_path = logs_dir / "episode_metric_consistency.csv"
    seed_path = logs_dir / "seed_metric_consistency.csv"
    contrast_path = logs_dir / "paired_contrasts.csv"
    plot_path = plots_dir / "metric_consistency.png"
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_path, seed_rows)
    _write_csv(contrast_path, contrasts)
    _plot_audit(episode_rows, seed_rows, contrasts, config.branches, plot_path)

    report: dict[str, Any] = {
        "audit": "metric_consistency",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "split": config.split,
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": list(config.branches),
        "independent_unit": "training seed after episode-level reduction",
        "episode_rows_are_not_independent": True,
        "test_data_read": False,
        "test_data_used": False,
        "source_writes": 0,
        "metric_definitions": {
            "position_rmse_norm_m": "sqrt(mean(||p_hat - p_true||_2^2)) over UAV-timestamp rows",
            "position_error_p95_norm_m": "95th percentile of ||p_hat - p_true||_2 over UAV-timestamp rows",
            "position_rmse_component_m": "legacy sqrt(mean of x/y/z coordinate residual squares)",
            "component_metric_is_retained_as_legacy": True,
        },
        "episode_counts": cell_counts,
        "checks": {
            "validation_only": config.split == "validation",
            "test_not_read": True,
            "source_untouched_by_audit": True,
            "all_required_metric_fields_finite": all_finite,
            "all_cells_have_expected_episode_count": all(
                count == int(config.expected_episodes) for count in cell_counts.values()
            ),
            "seed_level_inference_used": True,
            "algorithm_state_changed": False,
        },
        "discrepancy": {
            "mean_absolute_component_minus_norm_m": _safe_mean(component_norm_differences, 0.0),
            "max_absolute_component_minus_norm_m": max(component_norm_differences, default=0.0),
            "mean_component_to_norm_ratio": _safe_mean(ratios, 0.0),
            "expected_component_to_norm_ratio_one_over_sqrt3": float(1.0 / np.sqrt(3.0)),
            "expected_norm_to_component_ratio_sqrt3": float(np.sqrt(3.0)),
        },
        "decision": {
            "primary_reason": "metric_definition_mismatch" if component_norm_differences else "metric_definitions_consistent",
            "recommendation": (
                "Use the vector-norm RMSE/P95 for all new comparisons and keep the legacy component metric only for traceability. "
                "Do not compare event-level norm changes with legacy episode RMSE."
                if component_norm_differences
                else "The canonical and legacy metric definitions are numerically aligned."
            ),
        },
        "audit_ok": bool(
            bool(episode_rows)
            and bool(seed_rows)
            and all_finite
            and all(count == int(config.expected_episodes) for count in cell_counts.values())
        ),
        "outputs": {
            "episode_metric_consistency": str(episode_path),
            "seed_metric_consistency": str(seed_path),
            "paired_contrasts": str(contrast_path),
            "audit_report": str(logs_dir / "audit_report.json"),
            "plot": str(plot_path),
        },
        "interpretation_boundary": (
            "True positions are read only as offline validation labels for metric auditing. They are not passed to actor, critic, EKF, reward, "
            "proxy fitting, or policy selection. The audit does not retrain or modify source data."
        ),
    }
    (logs_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_BRANCHES",
    "DEFAULT_SEEDS",
    "MetricConsistencyAuditConfig",
    "run_metric_consistency_audit",
]
