"""Read-only audit of cross-seed measurement-policy stability.

The completed policy-mechanism audit established that quality features reach the
actor and that a selected measurement creates the intended queue event.  This
module does not replay or retrain an environment.  It only joins the existing
validation-only mechanism summaries with the frozen training metric logs so
that large seed-to-seed differences in measurement behaviour can be diagnosed
without changing any experiment artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _jsonable
from .policy_mechanism_audit import (
    SUPPORTED_BRANCHES,
    _finite_trace_value,
    _float,
    _read_csv,
    _read_json,
    _read_jsonl,
    _safe_mean,
    _write_csv,
)


DEFAULT_SEEDS = (7071, 8080, 9099, 10118, 11127)
DEFAULT_BRANCHES = SUPPORTED_BRANCHES
MANAGED_FILES = (
    "training_stability.csv",
    "validation_policy_stability.csv",
    "seed_comparison.csv",
    "audit_report.json",
)
PLOT_FILE = "measurement_policy_stability.png"


@dataclass(frozen=True)
class MeasurementPolicyStabilityAuditConfig:
    """Configuration for a validation-only, non-mutating stability audit."""

    reference_root: Path
    mechanism_audit_root: Path | None = None
    output_dir: Path = Path(
        "logs/preliminary/"
        "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/"
        "measurement_policy_stability_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("measurement policy stability audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty unique sequence")
        unknown = sorted(set(self.branches) - set(SUPPORTED_BRANCHES))
        if unknown:
            raise ValueError("unsupported branch(es): " + ", ".join(unknown))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    return value


def _prepare_output(path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"audit output is not empty: {path}; use --force")
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in MANAGED_FILES:
            target = path / name
            if target.exists():
                target.unlink()


def _stats(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([_float(row.get(key), float("nan")) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{key}_mean": 0.0,
            f"{key}_std": 0.0,
            f"{key}_min": 0.0,
            f"{key}_max": 0.0,
        }
    return {
        f"{key}_mean": float(values.mean()),
        f"{key}_std": float(values.std(ddof=0)),
        f"{key}_min": float(values.min()),
        f"{key}_max": float(values.max()),
    }


def _finite_numeric_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
    for row in rows:
        for key in keys:
            value = _float(row.get(key), float("nan"))
            if not math.isfinite(value):
                return False
    return True


def summarize_training_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, branch: str
) -> dict[str, Any]:
    """Summarize one seed/branch's train-update metrics.

    This is deliberately descriptive.  The saturation and entropy indicators
    are named heuristics, not claims that the optimizer has collapsed.
    """

    train_rows = [row for row in rows if str(row.get("phase", "train")) == "train"]
    metric_keys = (
        "actor_loss",
        "critic_loss",
        "loss",
        "entropy",
        "gradient_norm",
        "clip_fraction",
        "mean_reward",
        "mean_action_saturation",
        "max_action_saturation",
    )
    result: dict[str, Any] = {
        "base_seed": int(seed),
        "branch": branch,
        "phase": "train",
        "update_count": len(train_rows),
        "update_min": int(_float(train_rows[0].get("update"), 0.0)) if train_rows else 0,
        "update_max": int(_float(train_rows[-1].get("update"), 0.0)) if train_rows else 0,
        "finite_all": int(
            bool(train_rows) and all(_finite_trace_value(row) for row in train_rows)
        ),
        "collision_rows": int(
            sum(bool(row.get("collision_occurred", False)) for row in train_rows)
        ),
    }
    for key in metric_keys:
        result.update(_stats(train_rows, key))

    entropy_values = [
        _float(row.get("entropy"), float("nan")) for row in train_rows
    ]
    entropy_values = [value for value in entropy_values if math.isfinite(value)]
    first_entropy = entropy_values[0] if entropy_values else 0.0
    last_entropy = entropy_values[-1] if entropy_values else 0.0
    result["entropy_first"] = float(first_entropy)
    result["entropy_last"] = float(last_entropy)
    result["entropy_delta"] = float(last_entropy - first_entropy)
    result["entropy_ratio"] = float(last_entropy / first_entropy) if abs(first_entropy) > 1e-12 else 0.0

    clip_values = [
        _float(row.get("clip_fraction"), float("nan")) for row in train_rows
    ]
    clip_values = [value for value in clip_values if math.isfinite(value)]
    result["clip_fraction_ge_0_2_fraction"] = float(
        _safe_mean([value >= 0.2 for value in clip_values], default=0.0)
    )
    max_saturation = result["max_action_saturation_max"]
    result["action_saturation_observed"] = int(max_saturation >= 0.999999)
    result["entropy_ratio_below_half"] = int(
        bool(train_rows) and 0.0 < result["entropy_ratio"] < 0.5
    )
    result["heuristic_training_dynamics_flag"] = int(
        result["action_saturation_observed"]
        or result["entropy_ratio_below_half"]
        or result["clip_fraction_max"] > 0.5
    )
    return result


def _source_paths(mechanism_root: Path) -> dict[str, Path]:
    return {
        "trace": mechanism_root / "measurement_policy_trace.jsonl",
        "feature_summary": mechanism_root / "policy_feature_summary.csv",
        "action_summary": mechanism_root / "action_probability_summary.csv",
        "overhead_summary": mechanism_root / "measurement_overhead_summary.csv",
        "seed_summary": mechanism_root / "seed_level_mechanism_summary.csv",
        "audit_report": mechanism_root / "audit_report.json",
    }


def _resolve_mechanism_root(config: MeasurementPolicyStabilityAuditConfig) -> Path:
    if config.mechanism_audit_root is not None:
        return Path(config.mechanism_audit_root).resolve()
    return (
        Path("logs/preliminary")
        / Path(config.reference_root).resolve().name
        / "policy_mechanism_audit_v3"
    ).resolve()


def _load_training_summaries(
    reference_root: Path, seeds: Sequence[int], branches: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {}
    for seed in seeds:
        for branch in branches:
            path = (
                reference_root
                / f"seed_{int(seed)}"
                / branch
                / "scale_3"
                / "diagnostics"
                / "training_metrics.jsonl"
            )
            rows = _read_jsonl(path)
            summary = summarize_training_rows(rows, seed=int(seed), branch=branch)
            summary["source_path"] = str(path)
            summaries.append(summary)
            completeness[f"{int(seed)}::{branch}"] = {
                "exists": int(path.exists()),
                "row_count": len(rows),
                "train_row_count": summary["update_count"],
            }
    return summaries, completeness


def _merge_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (int(float(row.get("base_seed"))), str(row.get("branch", "")))
        except (TypeError, ValueError):
            continue
        result[key] = dict(row)
    return result


def _load_validation_summaries(
    mechanism_root: Path, seeds: Sequence[int], branches: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(mechanism_root)
    action_rows = _read_csv(paths["action_summary"])
    overhead_rows = _read_csv(paths["overhead_summary"])
    seed_rows = _read_csv(paths["seed_summary"])
    wanted = {(int(seed), branch) for seed in seeds for branch in branches}
    actions = _merge_by_key(
        [row for row in action_rows if (int(float(row.get("base_seed", -1))), str(row.get("branch", ""))) in wanted]
    )
    overhead = _merge_by_key(
        [row for row in overhead_rows if (int(float(row.get("base_seed", -1))), str(row.get("branch", ""))) in wanted]
    )
    seed_metrics = _merge_by_key(
        [row for row in seed_rows if (int(float(row.get("base_seed", -1))), str(row.get("branch", ""))) in wanted]
    )

    output: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {}
    numeric_keys = (
        "measurement_selection_rate",
        "none_selection_rate",
        "mean_none_probability",
        "mean_selected_probability",
        "measurement_count",
        "effective_measurement_count",
        "accepted_update_count",
        "harmful_fraction",
        "beneficial_fraction",
        "neutral_fraction",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "risk_probability_pearson_r",
        "mean_queue_before_packets",
        "mean_queue_with_measurement_packets",
        "mean_queue_next_packets",
        "measurement_overhead_packets",
        "measurement_budget_usage_rate",
    )
    for seed in seeds:
        for branch in branches:
            key = (int(seed), branch)
            merged: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": branch,
                "split": "validation",
                "source": "policy_mechanism_audit_v3",
            }
            for source in (seed_metrics.get(key, {}), actions.get(key, {}), overhead.get(key, {})):
                merged.update(source)
            # Keep the stability table's names explicit while accepting the
            # historical v3 column names used by the overhead summary.
            merged["measurement_overhead_packets"] = _float(
                merged.get("measurement_overhead_packets", merged.get("load_packets_applied")),
                0.0,
            )
            merged["measurement_budget_usage_rate"] = _float(
                merged.get("measurement_budget_usage_rate", merged.get("budget_usage_rate")),
                0.0,
            )
            merged["finite_all"] = int(_finite_numeric_rows([merged], numeric_keys))
            output.append(merged)
            completeness[f"{int(seed)}::{branch}"] = {
                "seed_summary": int(key in seed_metrics),
                "action_summary": int(key in actions),
                "overhead_summary": int(key in overhead),
            }
    return output, completeness


def _comparison_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    by_key = _merge_by_key(validation_rows)
    metrics = (
        "measurement_selection_rate",
        "effective_measurement_count",
        "none_selection_rate",
        "accepted_update_count",
        "harmful_fraction",
        "beneficial_fraction",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "measurement_count",
        "measurement_overhead_packets",
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = by_key.get((int(seed), "baseline_mappo"), {})
        if not baseline:
            continue
        for branch in branches:
            if branch == "baseline_mappo":
                continue
            treatment = by_key.get((int(seed), branch), {})
            if not treatment:
                continue
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "comparison": f"{branch}_minus_baseline_mappo",
                "baseline_branch": "baseline_mappo",
                "treatment_branch": branch,
                "split": "validation",
            }
            for metric in metrics:
                row[f"baseline_{metric}"] = _float(baseline.get(metric), 0.0)
                row[f"treatment_{metric}"] = _float(treatment.get(metric), 0.0)
                row[f"delta_{metric}"] = row[f"treatment_{metric}"] - row[f"baseline_{metric}"]
            output.append(row)
    return output


def _rate_variation(rows: Sequence[Mapping[str, Any]], branches: Sequence[str]) -> dict[str, Any]:
    by_branch: dict[str, list[float]] = {branch: [] for branch in branches}
    for row in rows:
        branch = str(row.get("branch", ""))
        if branch in by_branch:
            by_branch[branch].append(_float(row.get("measurement_selection_rate"), 0.0))
    details: dict[str, Any] = {}
    extreme = False
    for branch, values in by_branch.items():
        finite = [value for value in values if math.isfinite(value)]
        minimum = min(finite) if finite else 0.0
        maximum = max(finite) if finite else 0.0
        rate_range = maximum - minimum
        branch_extreme = bool(rate_range > 0.5 or (minimum < 0.05 and maximum > 0.3))
        extreme = extreme or branch_extreme
        details[branch] = {
            "seed_count": len(finite),
            "min": minimum,
            "max": maximum,
            "range": rate_range,
            "extreme_variation": int(branch_extreme),
        }
    return {"by_branch": details, "extreme_measurement_variation": int(extreme)}


def _write_plot(
    plot_path: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "baseline_mappo": "baseline",
        "quality_aware_mappo": "quality-aware",
        "budgeted_quality_aware_mappo": "budgeted-quality",
    }
    colors = {
        "baseline_mappo": "#2f6de1",
        "quality_aware_mappo": "#d92d2d",
        "budgeted_quality_aware_mappo": "#1b9e5a",
    }
    validation = _merge_by_key(validation_rows)
    training = _merge_by_key(training_rows)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)

    def draw_validation(axis: Any, metric: str, title: str, ylabel: str, *, symlog: bool = False) -> None:
        for branch in branches:
            values = [_float(validation.get((int(seed), branch), {}).get(metric), 0.0) for seed in seeds]
            axis.plot(seeds, values, marker="o", label=labels.get(branch, branch), color=colors.get(branch))
        axis.set_title(title)
        axis.set_xlabel("base seed")
        axis.set_ylabel(ylabel)
        if symlog:
            axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)

    def draw_training(axis: Any, metric: str, title: str, ylabel: str) -> None:
        for branch in branches:
            values = [_float(training.get((int(seed), branch), {}).get(metric), 0.0) for seed in seeds]
            axis.plot(seeds, values, marker="o", label=labels.get(branch, branch), color=colors.get(branch))
        axis.set_title(title)
        axis.set_xlabel("base seed")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    draw_validation(axes[0, 0], "measurement_selection_rate", "Measurement selection rate", "fraction")
    draw_validation(axes[0, 1], "effective_measurement_count", "Effective measurements", "count", symlog=True)
    draw_validation(axes[0, 2], "none_selection_rate", "None selection rate", "fraction")
    draw_validation(axes[1, 0], "position_rmse_m", "Validation position RMSE", "m")
    draw_validation(axes[1, 1], "j_com", "Validation J_com", "cost")
    draw_validation(axes[1, 2], "mean_reward", "Validation mean reward", "reward")
    draw_training(axes[2, 0], "actor_loss_std", "Actor-loss variability", "std")
    draw_training(axes[2, 1], "entropy_ratio", "Train entropy last / first", "ratio")
    draw_training(axes[2, 2], "gradient_norm_max", "Maximum gradient norm", "norm")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=max(1, len(branches)),
        )
    fig.suptitle(
        "Measurement-policy stability audit (validation-only diagnostic; seed-level series)",
        fontsize=14,
        y=1.025,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def run_measurement_policy_stability_audit(
    config: MeasurementPolicyStabilityAuditConfig,
) -> dict[str, Any]:
    """Run the non-mutating cross-seed stability audit."""

    reference_root = Path(config.reference_root).resolve()
    mechanism_root = _resolve_mechanism_root(config)
    output_dir = Path(config.output_dir).resolve()
    plots_dir = (
        Path(config.plots_dir).resolve()
        if config.plots_dir is not None
        else (Path("plots/preliminary") / reference_root.name / "measurement_policy_stability_audit").resolve()
    )
    if output_dir == reference_root or output_dir.is_relative_to(reference_root):
        raise ValueError("audit output must not be inside the reference experiment root")
    _prepare_output(output_dir, force=config.force)
    plots_dir.mkdir(parents=True, exist_ok=True)

    source_paths = _source_paths(mechanism_root)
    source_exists = {name: int(path.exists()) for name, path in source_paths.items()}
    if not all(source_exists.values()):
        missing = [name for name, exists in source_exists.items() if not exists]
        raise FileNotFoundError(
            "missing policy mechanism audit source file(s): " + ", ".join(missing)
        )

    source_report = _read_json_object(source_paths["audit_report"])
    training_rows, training_completeness = _load_training_summaries(
        reference_root, config.seeds, config.branches
    )
    validation_rows, validation_completeness = _load_validation_summaries(
        mechanism_root, config.seeds, config.branches
    )
    comparison_rows = _comparison_rows(validation_rows, config.seeds, config.branches)

    training_columns = sorted({key for row in training_rows for key in row})
    validation_columns = sorted({key for row in validation_rows for key in row})
    comparison_columns = sorted({key for row in comparison_rows for key in row})
    _write_csv(output_dir / "training_stability.csv", training_columns, training_rows)
    _write_csv(
        output_dir / "validation_policy_stability.csv", validation_columns, validation_rows
    )
    _write_csv(output_dir / "seed_comparison.csv", comparison_columns, comparison_rows)

    expected_keys = {(int(seed), branch) for seed in config.seeds for branch in config.branches}
    training_keys = {(int(row["base_seed"]), str(row["branch"])) for row in training_rows}
    validation_keys = {(int(row["base_seed"]), str(row["branch"])) for row in validation_rows}
    training_complete = training_keys == expected_keys and all(
        item["exists"] == 1 and item["train_row_count"] > 0
        for item in training_completeness.values()
    )
    validation_complete = validation_keys == expected_keys and all(
        item["seed_summary"] == 1
        and item["action_summary"] == 1
        and item["overhead_summary"] == 1
        for item in validation_completeness.values()
    )
    training_finite = all(int(row.get("finite_all", 0)) == 1 for row in training_rows)
    validation_finite = all(int(row.get("finite_all", 0)) == 1 for row in validation_rows)
    source_checks = dict(source_report.get("checks", {}))
    source_audit_ok = bool(source_report.get("audit_ok", False))
    source_validation_only = bool(
        source_report.get("test_data_used", True) is False
        and source_checks.get("all_validation_only", False)
    )
    warmup_only = bool(source_checks.get("quality_models_warmup_only", False))
    variation = _rate_variation(validation_rows, config.branches)
    training_saturation = [
        {
            "base_seed": int(row["base_seed"]),
            "branch": str(row["branch"]),
            "action_saturation_observed": int(row.get("action_saturation_observed", 0)),
            "entropy_ratio": _float(row.get("entropy_ratio"), 0.0),
            "clip_fraction_max": _float(row.get("clip_fraction_max"), 0.0),
        }
        for row in training_rows
        if int(row.get("heuristic_training_dynamics_flag", 0)) == 1
    ]

    audit_report: dict[str, Any] = {
        "audit": "measurement_policy_training_stability_audit",
        "schema_version": "measurement-policy-stability-v1",
        "reference_root": str(reference_root),
        "mechanism_audit_root": str(mechanism_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "split": config.split,
        "branches": list(config.branches),
        "seeds": [int(seed) for seed in config.seeds],
        "seed_set": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_and_window_rows_not_independent": True,
        "test_data_used": False,
        "training_source": "scale_3/diagnostics/training_metrics.jsonl, phase=train",
        "validation_source": "policy_mechanism_audit_v3 summaries, validation-only",
        "checks": {
            "source_files_exist": bool(all(source_exists.values())),
            "source_mechanism_audit_ok": source_audit_ok,
            "source_validation_only": source_validation_only,
            "quality_models_warmup_only": warmup_only,
            "training_branch_complete": bool(training_complete),
            "validation_branch_complete": bool(validation_complete),
            "training_rows_finite": bool(training_finite),
            "validation_rows_finite": bool(validation_finite),
            "original_data_not_written": True,
        },
        "source_checks": source_checks,
        "source_files": source_exists,
        "measurement_rate_variation": variation,
        "training_dynamics": {
            "heuristic_thresholds": {
                "max_action_saturation_ge": 0.999999,
                "entropy_ratio_below": 0.5,
                "clip_fraction_max_above": 0.5,
            },
            "possible_saturation_or_instability_rows": training_saturation,
        },
        "diagnostic_focus": {
            "high_measurement_seed_8080": [
                row for row in validation_rows if int(row["base_seed"]) == 8080
            ],
            "low_measurement_seed_9099": [
                row for row in validation_rows if int(row["base_seed"]) == 9099
            ],
        },
        "findings": {
            "feature_and_queue_link_verified": bool(
                source_audit_ok
                and source_checks.get("feature_injection_ok", False)
                and source_checks.get("payload_invariant", False)
            ),
            "actor_risk_response_verified": bool(
                source_checks.get("pre_action_mask_validity", False)
                and source_report.get("policy_response", {}).get("actor_response_by_branch")
            ),
            "measurement_rate_seed_sensitive": bool(
                variation["extreme_measurement_variation"]
            ),
            "training_dynamics_possible_saturation": bool(training_saturation),
            "primary_finding": "policy_training_or_measurement_rate_instability"
            if variation["extreme_measurement_variation"]
            else "no_extreme_measurement_rate_variation_detected",
        },
        "decision": "stabilize_measurement_policy_before_core_model_changes",
        "decision_supported": False,
        "claim_boundary": (
            "This is a descriptive mechanism diagnostic. It does not establish "
            "causal superiority, generalization, or a paper-level performance claim."
        ),
    }
    audit_ok = bool(
        all(source_exists.values())
        and source_audit_ok
        and source_validation_only
        and warmup_only
        and training_complete
        and validation_complete
        and training_finite
        and validation_finite
    )
    audit_report["audit_ok"] = audit_ok
    (output_dir / "audit_report.json").write_text(
        json.dumps(_jsonable(audit_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_path = plots_dir / PLOT_FILE
    _write_plot(plot_path, validation_rows, training_rows, config.seeds, config.branches)

    summary = {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "audit_report": str(output_dir / "audit_report.json"),
        "plot": str(plot_path),
        "seeds": [int(seed) for seed in config.seeds],
        "branches": list(config.branches),
        "decision": audit_report["decision"],
        "primary_finding": audit_report["findings"]["primary_finding"],
        "audit_ok": audit_ok,
        "decision_supported": False,
    }
    if config.progress:
        print(
            "[measurement-policy-stability] "
            f"seeds={len(config.seeds)} branches={len(config.branches)} "
            f"decision={audit_report['decision']}",
            flush=True,
        )
    return summary


__all__ = [
    "MeasurementPolicyStabilityAuditConfig",
    "run_measurement_policy_stability_audit",
    "summarize_training_rows",
]
