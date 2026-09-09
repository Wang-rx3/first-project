"""Read-only attribution audit for the 3-UAV mainline diagnostic curves.

The audit consumes frozen train diagnostics and validation-only seed summaries.
It never opens the test split, updates a trainer, or mutates checkpoints.  Event
and update rows are used only for mechanism diagnosis; the statistical unit
remains the training seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ekf_sensor_interaction_audit import (
    EKFSensorInteractionAuditConfig,
    run_ekf_sensor_interaction_audit,
)
from .final_mainline_audit import (
    BASELINE_BRANCH,
    DEFAULT_BRANCHES,
    DEFAULT_SEEDS,
    PROPOSED_BRANCH,
)


LOW_REWARD_THRESHOLD = -0.7
ROLLING_WINDOW = 5
ATTRIBUTION_STATES = {"confirmed", "ruled_out", "inconclusive"}
MANAGED_FILES = (
    "training_reward_loss_attribution.csv",
    "collision_stratified_training.csv",
    "train_validation_cost_shift.csv",
    "expected_actual_cost_gap.csv",
    "update_quality_per_window.csv",
    "jloc_sensitivity.csv",
    "seed_outlier_attribution.csv",
    "audit_report.json",
)
PLOT_FILES = (
    "training_reward_collision_overlay.png",
    "loss_critic_attribution.png",
    "measurement_update_rmse_path.png",
    "expected_actual_communication_gap.png",
    "jloc_rmse_sensitivity.png",
    "seed_paired_changes.png",
)
BRANCH_COLORS = {
    BASELINE_BRANCH: "#356AE6",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo": "#18A566",
    PROPOSED_BRANCH: "#D92D2D",
}
BRANCH_LABELS = {
    BASELINE_BRANCH: "Local-EKF-GAT-MAPPO (Proposed Method)",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo": "Proxy-guided ablation",
    PROPOSED_BRANCH: "Deprecated communication-constrained branch",
}


@dataclass(frozen=True)
class MainlineCurveCauseAuditConfig:
    """Configuration for a frozen-result curve attribution audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
        "curve_cause_audit"
    )
    plots_dir: Path | None = None
    source_logs_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    expected_training_updates: int = 100
    expected_validation_episodes: int = 20
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    include_sensor_interaction: bool = True
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("mainline curve cause audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty unique sequence")
        if BASELINE_BRANCH not in self.branches or PROPOSED_BRANCH not in self.branches:
            raise ValueError("branches must include baseline and proposed branches")
        if self.expected_training_updates < 1 or self.expected_validation_episodes < 1:
            raise ValueError("expected update and episode counts must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: MainlineCurveCauseAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[mainline-curve-cause] {message}", flush=True)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = 0) -> int:
    result = _number(value, float(default))
    return int(round(result)) if math.isfinite(result) else int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_number(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _std(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_number(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if array.size > 1 else float(default)


def _correlation(x_values: Iterable[Any], y_values: Iterable[Any]) -> float:
    pairs = [
        (_number(x), _number(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(_number(x)) and math.isfinite(_number(y))
    ]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing audit source table: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing audit source report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _prepare_outputs(config: MainlineCurveCauseAuditConfig) -> tuple[Path, Path]:
    reference = Path(config.reference_root).resolve()
    logs = Path(config.output_dir).resolve()
    plots = (
        Path(config.plots_dir).resolve()
        if config.plots_dir is not None
        else (Path("plots/preliminary") / reference.name / "curve_cause_audit").resolve()
    )
    for path in (logs, plots):
        if path == reference or path.is_relative_to(reference):
            raise ValueError("audit outputs must not be inside the frozen experiment root")
    if config.force:
        for path in (logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("curve-cause audit output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(
    reference: Path,
    source_logs: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
) -> dict[str, str]:
    files = [
        source_logs / "final_mainline_audit" / "training_metrics.csv",
        source_logs / "final_mainline_audit" / "seed_level_metrics.csv",
        source_logs / "final_mainline_audit" / "paired_contrasts.csv",
        source_logs / "final_mainline_audit" / "validation_communication_cost.csv",
    ]
    files.extend(
        reference / f"seed_{int(seed)}" / branch / "scale_3" / "checkpoints" / "final.pt"
        for seed in seeds
        for branch in branches
    )
    return {
        str(path.resolve()): _sha256(path)
        for path in files
        if path.is_file()
    }


def _trailing_mean(values: Sequence[float], window: int = ROLLING_WINDOW) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - int(window) + 1)
        output.append(_mean(values[start : index + 1]))
    return output


def _training_rows(
    source_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = {(int(seed), str(branch)) for seed in seeds for branch in branches}
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        key = (_integer(row.get("base_seed")), str(row.get("branch", "")))
        if key in wanted and str(row.get("phase", "")) == "train":
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: _integer(row.get("update")))
        rewards = [_number(row.get("mean_reward")) for row in ordered]
        rolling = _trailing_mean(rewards)
        for row, rolling_reward in zip(ordered, rolling):
            reward = _number(row.get("mean_reward"))
            collision = int(_bool(row.get("collision_occurred")))
            output.append(
                {
                    "base_seed": seed,
                    "branch": branch,
                    "update": _integer(row.get("update")),
                    "phase": "train",
                    "mean_reward_raw": reward,
                    "mean_reward_trailing_mean_5": rolling_reward,
                    "rolling_window": ROLLING_WINDOW,
                    "collision_occurred": collision,
                    "low_reward_below_minus_0_7": int(reward < LOW_REWARD_THRESHOLD),
                    "loss": _number(row.get("loss")),
                    "actor_loss": _number(row.get("actor_loss")),
                    "critic_loss": _number(row.get("critic_loss")),
                    "weighted_critic_loss": 0.5 * _number(row.get("critic_loss")),
                    "entropy_combined_heads": _number(row.get("entropy")),
                    "clip_fraction": _number(row.get("clip_fraction")),
                    "raw_curve_preserved": 1,
                    "smoothing_used_for_inference": 0,
                }
            )
    return output


def _collision_summaries(
    training_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in training_rows:
        groups[(_integer(row.get("base_seed")), str(row.get("branch")))].append(row)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in branches:
            rows = groups.get((int(seed), str(branch)), [])
            collision = [row for row in rows if _bool(row.get("collision_occurred"))]
            noncollision = [row for row in rows if not _bool(row.get("collision_occurred"))]
            low = [row for row in rows if _number(row.get("mean_reward_raw")) < LOW_REWARD_THRESHOLD]
            loss_corr = _correlation(
                [row.get("loss") for row in rows],
                [row.get("critic_loss") for row in rows],
            )
            critic_collision = _mean(row.get("critic_loss") for row in collision)
            critic_noncollision = _mean(row.get("critic_loss") for row in noncollision)
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": str(branch),
                    "training_update_count": len(rows),
                    "rollout_episodes_per_update": 1,
                    "collision_update_count": len(collision),
                    "collision_update_fraction": len(collision) / max(1, len(rows)),
                    "low_reward_update_count": len(low),
                    "low_reward_collision_count": sum(
                        int(_bool(row.get("collision_occurred"))) for row in low
                    ),
                    "collision_fraction_among_low_reward": (
                        sum(int(_bool(row.get("collision_occurred"))) for row in low) / len(low)
                        if low else 0.0
                    ),
                    "mean_reward_collision": _mean(row.get("mean_reward_raw") for row in collision),
                    "mean_reward_noncollision": _mean(row.get("mean_reward_raw") for row in noncollision),
                    "noncollision_reward_std": _std(row.get("mean_reward_raw") for row in noncollision),
                    "mean_loss_collision": _mean(row.get("loss") for row in collision),
                    "mean_loss_noncollision": _mean(row.get("loss") for row in noncollision),
                    "mean_critic_loss_collision": critic_collision,
                    "mean_critic_loss_noncollision": critic_noncollision,
                    "critic_collision_to_noncollision_ratio": (
                        critic_collision / critic_noncollision
                        if math.isfinite(critic_collision)
                        and math.isfinite(critic_noncollision)
                        and abs(critic_noncollision) > 1.0e-12
                        else float("nan")
                    ),
                    "loss_critic_correlation": loss_corr,
                    "rows_are_training_updates_not_independent_seeds": 1,
                }
            )
    return output


def _seed_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (_integer(row.get("base_seed")), str(row.get("branch", ""))): row
        for row in rows
    }


def _is_validation_seed_row(row: Mapping[str, Any]) -> bool:
    """Accept the legacy ``nan`` split cell only in the validation-only table.

    ``final_mainline_audit/seed_level_metrics.csv`` is itself produced from a
    validation-only reducer, but older versions serialized the derived split
    cell as the literal string ``nan``.  The parent audit report is separately
    required to declare validation-only/test-not-read before these rows are
    consumed.
    """

    value = str(row.get("split", "")).strip().lower()
    return value in {"validation", "", "nan"}


def _cost_target_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, float]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split", "")) != "validation":
            continue
        groups[(_integer(row.get("base_seed")), str(row.get("treatment", "")))].append(row)
    return {
        key: {
            "target": _mean(row.get("target") for row in values),
            "expected": _mean(row.get("expected_communication_cost") for row in values),
            "actual": _mean(row.get("actual_communication_cost") for row in values),
        }
        for key, values in groups.items()
    }


def _train_validation_shift_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    metrics = _seed_index(seed_rows)
    costs = _cost_target_index(cost_rows)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = metrics[(int(seed), BASELINE_BRANCH)]
        proposed = metrics[(int(seed), PROPOSED_BRANCH)]
        target = costs[(int(seed), PROPOSED_BRANCH)]["target"]
        baseline_actual = _number(baseline.get("actual_communication_cost"))
        proposed_actual = _number(proposed.get("actual_communication_cost"))
        output.append(
            {
                "base_seed": int(seed),
                "source_target_split": "baseline_mappo/train",
                "evaluation_split": "validation",
                "communication_cost_target": target,
                "baseline_validation_actual_cost": baseline_actual,
                "proposed_validation_expected_cost": _number(proposed.get("expected_communication_cost")),
                "proposed_validation_actual_cost": proposed_actual,
                "target_minus_baseline_validation_actual": target - baseline_actual,
                "proposed_minus_baseline_validation_actual": proposed_actual - baseline_actual,
                "absolute_train_target_validation_shift": abs(target - baseline_actual),
                "shift_over_0_10": int(abs(target - baseline_actual) > 0.10),
                "baseline_validation_measurement_count": _number(baseline.get("measurement_count")),
                "proposed_validation_measurement_count": _number(proposed.get("measurement_count")),
                "measurement_count_difference": (
                    _number(proposed.get("measurement_count"))
                    - _number(baseline.get("measurement_count"))
                ),
                "validation_not_used_for_target_fitting": 1,
            }
        )
    return output


def _expected_actual_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = _seed_index(seed_rows)
    costs = _cost_target_index(cost_rows)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in branches:
            metric = metrics[(int(seed), str(branch))]
            cost = costs[(int(seed), str(branch))]
            expected = cost["expected"]
            actual = cost["actual"]
            target = cost["target"]
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": str(branch),
                    "split": "validation",
                    "validation_execution_mode": "deterministic_argmax",
                    "expected_communication_cost": expected,
                    "actual_selected_communication_cost": actual,
                    "actual_minus_expected": actual - expected,
                    "baseline_train_target": target,
                    "expected_within_target_plus_0_05": int(expected <= target + 0.05 + 1.0e-12),
                    "actual_within_target_plus_0_05": int(actual <= target + 0.05 + 1.0e-12),
                    "measurement_count": _number(metric.get("measurement_count")),
                    "j_com": _number(metric.get("j_com")),
                    "seed_is_independent_unit": 1,
                }
            )
    return output


def _update_quality_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = _seed_index(seed_rows)
    output: list[dict[str, Any]] = []
    fields = (
        "accepted_update_count",
        "accepted_updates_per_window",
        "update_coverage_any",
        "accepted_update_rate",
        "beneficial_fraction",
        "neutral_fraction",
        "harmful_fraction",
        "beneficial_per_window",
        "harmful_per_window",
        "h1_benefit_retained_fraction",
        "h2_benefit_retained_fraction",
        "h3_benefit_retained_fraction",
        "h5_benefit_retained_fraction",
        "h1_benefit_loss_observed_fraction",
        "h3_benefit_loss_observed_fraction",
        "h5_benefit_loss_observed_fraction",
    )
    for seed in seeds:
        for branch in branches:
            source = metrics[(int(seed), str(branch))]
            row: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": str(branch),
                "split": "validation",
                "episode_count": _integer(source.get("episode_count")),
                "no_update_episode_count": _integer(source.get("no_update_episode_count")),
                "event_rows_are_not_independent": 1,
                "seed_is_independent_unit": 1,
            }
            row.update({field: _number(source.get(field)) for field in fields})
            output.append(row)
    return output


def _jloc_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = _seed_index(seed_rows)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in branches:
            source = metrics[(int(seed), str(branch))]
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": str(branch),
                    "split": "validation",
                    "position_rmse_norm_m": _number(source.get("position_rmse_norm_m")),
                    "position_p95_norm_m": _number(source.get("position_p95_norm_m")),
                    "j_loc": _number(source.get("j_loc")),
                    "covariance_position_trace": _number(source.get("covariance_position_trace")),
                    "j_loc_saturation_fraction": _number(source.get("j_loc_saturation_fraction")),
                    "linear_cov_reduction_event_fraction": _number(
                        source.get("linear_cov_reduction_event_fraction")
                    ),
                    "j_loc_immediate_change_abs_max": _number(
                        source.get("j_loc_immediate_change_abs_max")
                    ),
                    "j_loc_semantics": "clipped covariance uncertainty, not state-mean RMSE",
                    "seed_is_independent_unit": 1,
                }
            )
    return output


def _outlier_rows(
    seed_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    metrics = _seed_index(seed_rows)
    output: list[dict[str, Any]] = []
    fields = (
        "position_rmse_norm_m",
        "position_p95_norm_m",
        "measurement_count",
        "accepted_updates_per_window",
        "beneficial_per_window",
        "harmful_per_window",
        "actual_communication_cost",
        "expected_communication_cost",
        "j_com",
        "mean_reward",
        "j_loc",
    )
    for seed in seeds:
        baseline = metrics[(int(seed), BASELINE_BRANCH)]
        proposed = metrics[(int(seed), PROPOSED_BRANCH)]
        row: dict[str, Any] = {
            "base_seed": int(seed),
            "comparison": f"{PROPOSED_BRANCH}_minus_{BASELINE_BRANCH}",
        }
        for field in fields:
            row[f"delta_{field}"] = _number(proposed.get(field)) - _number(baseline.get(field))
        row["rmse_worse"] = int(row["delta_position_rmse_norm_m"] > 0.0)
        row["p95_worse"] = int(row["delta_position_p95_norm_m"] > 0.0)
        row["actual_cost_worse"] = int(row["delta_actual_communication_cost"] > 0.0)
        row["harmful_per_window_worse"] = int(row["delta_harmful_per_window"] > 0.0)
        row["measurement_count_increased"] = int(row["delta_measurement_count"] > 0.0)
        row["outlier_score"] = sum(
            int(row[key])
            for key in (
                "rmse_worse",
                "p95_worse",
                "actual_cost_worse",
                "harmful_per_window_worse",
                "measurement_count_increased",
            )
        )
        output.append(row)
    return output


def _status(condition: bool, evidence_available: bool = True) -> str:
    if not evidence_available:
        return "inconclusive"
    return "confirmed" if condition else "ruled_out"


def _attributions(
    collision_rows: Sequence[Mapping[str, Any]],
    shift_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    jloc_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
    source_report: Mapping[str, Any],
    sensor_report: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    low_count = sum(_integer(row.get("low_reward_update_count")) for row in collision_rows)
    low_collision = sum(_integer(row.get("low_reward_collision_count")) for row in collision_rows)
    collision_fraction = low_collision / max(1, low_count)
    reward_gaps = [
        _number(row.get("mean_reward_collision")) - _number(row.get("mean_reward_noncollision"))
        for row in collision_rows
    ]
    correlations = [_number(row.get("loss_critic_correlation")) for row in collision_rows]
    critic_ratios = [
        _number(row.get("critic_collision_to_noncollision_ratio")) for row in collision_rows
    ]
    noncollision_sd = _mean(row.get("noncollision_reward_std") for row in collision_rows)
    shifted = sum(_integer(row.get("shift_over_0_10")) for row in shift_rows)
    proposed_gap = [row for row in gap_rows if str(row.get("branch")) == PROPOSED_BRANCH]
    argmax_gap_count = sum(
        int(_number(row.get("actual_minus_expected")) > 0.05) for row in proposed_gap
    )
    cost_worse_seeds = sum(
        int(_number(row.get("delta_actual_communication_cost")) > 0.0)
        for row in outlier_rows
    )
    proposed_quality = [
        row for row in quality_rows if str(row.get("branch")) == PROPOSED_BRANCH
    ]
    h5_retained = _mean(row.get("h5_benefit_retained_fraction") for row in proposed_quality)
    proposed_jloc = [row for row in jloc_rows if str(row.get("branch")) == PROPOSED_BRANCH]
    baseline_jloc = {
        _integer(row.get("base_seed")): row
        for row in jloc_rows
        if str(row.get("branch")) == BASELINE_BRANCH
    }
    jloc_delta = _mean(
        _number(row.get("j_loc"))
        - _number(baseline_jloc[_integer(row.get("base_seed"))].get("j_loc"))
        for row in proposed_jloc
    )
    rmse_delta = _mean(row.get("delta_position_rmse_norm_m") for row in outlier_rows)
    saturation = _mean(row.get("j_loc_saturation_fraction") for row in jloc_rows)
    sensor_decision = dict((sensor_report or {}).get("decision", {}))
    sensor_available = bool(sensor_report)
    sensor_cause = str(sensor_decision.get("persistence_primary_cause", ""))
    sensor_jloc_break = bool((sensor_report or {}).get("jloc_objective_break", False))
    sensor_h5_retained = _number(
        sensor_decision.get("benefit_retained_at_decision_horizon")
    )
    sensor_vo_fraction = _number(
        dict(sensor_decision.get("next_transition_type_fraction", {})).get(
            "visual_odometry_update"
        )
    )
    init_ok = bool(source_report.get("checks", {}).get("shared_initialization_hashes_equal", False))
    if not init_ok:
        init_ok = bool(source_report.get("initialization_audit", {}).get("shared_initialization_hashes_equal", False))

    results = {
        "collision_mixture_dominates_reward_curve": {
            "status": _status(
                low_count > 0
                and collision_fraction >= 0.95
                and _mean(reward_gaps) <= -0.20
            ),
            "low_reward_update_count": low_count,
            "collision_fraction_among_low_reward": collision_fraction,
            "mean_collision_minus_noncollision_reward": _mean(reward_gaps),
        },
        "critic_target_variance_dominates_loss": {
            "status": _status(
                _mean(correlations) >= 0.99 and _mean(critic_ratios) >= 2.0
            ),
            "mean_loss_critic_correlation": _mean(correlations),
            "mean_collision_to_noncollision_critic_ratio": _mean(critic_ratios),
        },
        "single_episode_rollout_variance": {
            "status": _status(noncollision_sd >= 0.05),
            "rollout_episodes_per_update": 1,
            "mean_within_seed_noncollision_reward_std": noncollision_sd,
            "diagnostic_threshold": 0.05,
        },
        "train_validation_cost_shift": {
            "status": _status(shifted >= 3),
            "seed_count_with_absolute_shift_over_0_10": shifted,
            "total_seed_count": len(shift_rows),
        },
        "expected_argmax_cost_mismatch": {
            "status": _status(argmax_gap_count >= 3 and cost_worse_seeds >= 2),
            "proposed_seed_count_actual_minus_expected_over_0_05": argmax_gap_count,
            "paired_seed_count_actual_cost_worse_than_baseline": cost_worse_seeds,
            "validation_execution_mode": "deterministic_argmax",
        },
        "ekf_benefit_not_persistent": {
            "status": _status(
                h5_retained < 0.50
                and sensor_cause in {
                    "visual_odometry_interaction",
                    "prediction_propagation",
                    "covariance_restoration_objective_break",
                },
                sensor_available,
            ),
            "proposed_h5_benefit_retained_fraction": h5_retained,
            "sensor_interaction_h5_benefit_retained_fraction": sensor_h5_retained,
            "next_visual_odometry_transition_fraction": sensor_vo_fraction,
            "sensor_interaction_primary_cause": sensor_cause or None,
        },
        "jloc_objective_insensitive": {
            "status": _status(
                abs(jloc_delta) <= 0.001
                and abs(rmse_delta) >= 0.005
                and saturation >= 0.10
                and sensor_jloc_break,
                sensor_available,
            ),
            "mean_proposed_minus_baseline_j_loc": jloc_delta,
            "mean_proposed_minus_baseline_rmse_norm_m": rmse_delta,
            "mean_j_loc_saturation_fraction": saturation,
            "sensor_audit_jloc_objective_break": sensor_jloc_break,
        },
        "update_coverage_any_is_saturated": {
            "status": _status(
                all(abs(_number(row.get("update_coverage_any")) - 1.0) <= 1.0e-12 for row in quality_rows)
            ),
            "recommended_primary_update_metrics": [
                "accepted_updates_per_window",
                "beneficial_per_window",
                "harmful_per_window",
            ],
        },
        "shared_initialization_confound": {
            "status": "ruled_out" if init_ok else "inconclusive",
            "shared_initialization_hashes_equal": init_ok,
        },
        "numeric_or_schema_failure": {
            "status": "ruled_out" if bool(source_report.get("audit_ok", False)) else "inconclusive",
            "source_audit_ok": bool(source_report.get("audit_ok", False)),
        },
    }
    if any(item["status"] not in ATTRIBUTION_STATES for item in results.values()):
        raise RuntimeError("invalid attribution state")
    return results


def _plot_training_reward(
    rows: Sequence[Mapping[str, Any]], branches: Sequence[str], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(branches), 1, figsize=(10, 3.1 * len(branches)), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis, branch in zip(axes_array, branches):
        selected = [row for row in rows if str(row.get("branch")) == branch]
        by_update: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            by_update[_integer(row.get("update"))].append(row)
        updates = sorted(by_update)
        raw = [_mean(row.get("mean_reward_raw") for row in by_update[update]) for update in updates]
        smooth = _trailing_mean(raw)
        axis.plot(updates, raw, color=BRANCH_COLORS.get(branch), alpha=0.55, linewidth=1.2, label="raw cross-seed mean")
        axis.plot(updates, smooth, color=BRANCH_COLORS.get(branch), linewidth=2.2, label="5-update trailing mean (display only)")
        for row in selected:
            if _bool(row.get("collision_occurred")):
                axis.scatter(_integer(row.get("update")), _number(row.get("mean_reward_raw")), color="#111827", marker="x", s=18, alpha=0.45)
        axis.axhline(LOW_REWARD_THRESHOLD, color="#6B7280", linestyle="--", linewidth=1)
        axis.set_ylabel("Mean team reward")
        axis.set_title(BRANCH_LABELS.get(branch, branch))
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, loc="lower right")
    axes_array[-1].set_xlabel("Update")
    fig.suptitle("Training reward with collision updates (diagnostic only)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_loss(
    rows: Sequence[Mapping[str, Any]], branches: Sequence[str], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(branches), figsize=(5.1 * len(branches), 4.2), sharey=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis, branch in zip(axes_array, branches):
        selected = [row for row in rows if str(row.get("branch")) == branch]
        by_update: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            by_update[_integer(row.get("update"))].append(row)
        updates = sorted(by_update)
        total = [_mean(row.get("loss") for row in by_update[update]) for update in updates]
        critic = [_mean(row.get("weighted_critic_loss") for row in by_update[update]) for update in updates]
        axis.plot(updates, total, color="#111827", linewidth=1.4, label="total loss")
        axis.plot(updates, critic, color=BRANCH_COLORS.get(branch), linewidth=1.5, alpha=0.85, label="0.5 × critic loss")
        collision_updates = sorted({
            _integer(row.get("update")) for row in selected if _bool(row.get("collision_occurred"))
        })
        for update in collision_updates:
            axis.axvline(update, color="#DC2626", alpha=0.06, linewidth=1)
        axis.set_title(BRANCH_LABELS.get(branch, branch))
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    axes_array[0].set_ylabel("Loss")
    axes_array[0].legend(fontsize=8)
    fig.suptitle("MAPPO total-loss attribution (red bands: collision updates)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_measurement_path(rows: Sequence[Mapping[str, Any]], branches: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for branch in branches:
        selected = [row for row in rows if str(row.get("branch")) == branch]
        color = BRANCH_COLORS.get(branch)
        label = BRANCH_LABELS.get(branch, branch)
        axes[0].scatter(
            [_number(row.get("accepted_updates_per_window")) for row in selected],
            [_number(row.get("position_rmse_norm_m")) for row in selected],
            color=color, label=label, s=48,
        )
        axes[1].scatter(
            [_number(row.get("harmful_per_window")) for row in selected],
            [_number(row.get("position_rmse_norm_m")) for row in selected],
            color=color, label=label, s=48,
        )
    axes[0].set(xlabel="Accepted updates per window", ylabel="Position RMSE norm (m)", title="Update coverage vs localization")
    axes[1].set(xlabel="Harmful updates per window", ylabel="Position RMSE norm (m)", title="Harmful-update load vs localization")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_cost_gap(rows: Sequence[Mapping[str, Any]], branches: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    all_values: list[float] = []
    for branch in branches:
        selected = [row for row in rows if str(row.get("branch")) == branch]
        x = [_number(row.get("expected_communication_cost")) for row in selected]
        y = [_number(row.get("actual_selected_communication_cost")) for row in selected]
        all_values.extend(x + y)
        axis.scatter(x, y, color=BRANCH_COLORS.get(branch), label=BRANCH_LABELS.get(branch, branch), s=55)
        # Label only the proposed branch. Labeling all 15 points obscures the
        # dense ablation/proposed cluster and adds no inferential information.
        if branch == PROPOSED_BRANCH:
            for row, xv, yv in zip(selected, x, y):
                axis.annotate(
                    str(_integer(row.get("base_seed"))),
                    (xv, yv),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    lower = min(all_values, default=0.0)
    upper = max(all_values, default=1.0)
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="actual = expected")
    axis.set(xlabel="Expected communication cost", ylabel="Deterministic argmax actual cost", title="Expected–actual communication-cost gap")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_jloc(rows: Sequence[Mapping[str, Any]], branches: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for branch in branches:
        selected = [row for row in rows if str(row.get("branch")) == branch]
        color = BRANCH_COLORS.get(branch)
        label = BRANCH_LABELS.get(branch, branch)
        axes[0].scatter(
            [_number(row.get("position_rmse_norm_m")) for row in selected],
            [_number(row.get("j_loc")) for row in selected],
            color=color, label=label, s=50,
        )
        axes[1].scatter(
            [_number(row.get("j_loc_saturation_fraction")) for row in selected],
            [_number(row.get("position_rmse_norm_m")) for row in selected],
            color=color, label=label, s=50,
        )
    axes[0].set(xlabel="Position RMSE norm (m)", ylabel="J_loc", title="RMSE and covariance objective")
    axes[1].set(xlabel="J_loc saturation fraction", ylabel="Position RMSE norm (m)", title="Clipping saturation and RMSE")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_paired(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("delta_position_rmse_norm_m", "RMSE Δ (m)"),
        ("delta_actual_communication_cost", "Actual cost Δ"),
        ("delta_harmful_per_window", "Harmful/window Δ"),
        ("delta_accepted_updates_per_window", "Accepted/window Δ"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    seeds = [_integer(row.get("base_seed")) for row in rows]
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [_number(row.get(metric)) for row in rows]
        colors = ["#18A566" if value < 0.0 else "#D92D2D" for value in values]
        axis.bar([str(seed) for seed in seeds], values, color=colors, alpha=0.85)
        axis.axhline(0.0, color="#111827", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Proposed minus baseline seed-paired changes")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_mainline_curve_cause_audit(
    config: MainlineCurveCauseAuditConfig,
) -> dict[str, Any]:
    """Run the read-only 3-UAV curve attribution audit."""

    reference = Path(config.reference_root).resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"reference experiment root not found: {reference}")
    source_logs = (
        Path(config.source_logs_dir).resolve()
        if config.source_logs_dir is not None
        else (Path("logs/preliminary") / reference.name).resolve()
    )
    final_logs = source_logs / "final_mainline_audit"
    source_report = _read_json(source_logs / "audit_report.json")
    final_report = _read_json(final_logs / "audit_report.json")
    training_source = _read_csv(final_logs / "training_metrics.csv")
    seed_source = _read_csv(final_logs / "seed_level_metrics.csv")
    cost_source = _read_csv(final_logs / "validation_communication_cost.csv")
    _read_csv(final_logs / "paired_contrasts.csv")

    before_hashes = _source_hashes(reference, source_logs, config.seeds, config.branches)
    logs, plots = _prepare_outputs(config)
    _progress(config, "building frozen train/validation attribution tables")
    training_rows = _training_rows(training_source, config.seeds, config.branches)
    collision_rows = _collision_summaries(training_rows, config.seeds, config.branches)
    selected_seed_rows = [
        row for row in seed_source
        if _integer(row.get("base_seed")) in set(config.seeds)
        and str(row.get("branch")) in set(config.branches)
        and _is_validation_seed_row(row)
    ]
    shift_rows = _train_validation_shift_rows(selected_seed_rows, cost_source, config.seeds)
    gap_rows = _expected_actual_rows(
        selected_seed_rows, cost_source, config.seeds, config.branches
    )
    quality_rows = _update_quality_rows(selected_seed_rows, config.seeds, config.branches)
    jloc_rows = _jloc_rows(selected_seed_rows, config.seeds, config.branches)
    outlier_rows = _outlier_rows(selected_seed_rows, config.seeds)

    sensor_report: dict[str, Any] | None = None
    if config.include_sensor_interaction:
        _progress(config, "rechecking EKF-to-next-sensor interaction on the stabilized seeds")
        sensor_summary = run_ekf_sensor_interaction_audit(
            EKFSensorInteractionAuditConfig(
                reference_root=reference,
                output_dir=logs / "ekf_sensor_interaction",
                plots_dir=plots / "ekf_sensor_interaction",
                seeds=config.seeds,
                branches=config.branches,
                split="validation",
                horizons=(1, 2, 3, 5),
                expected_episodes=config.expected_validation_episodes,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
                force=False,
                progress=config.progress,
            )
        )
        # The sensor-interaction runner returns the report object directly;
        # its persisted report path is available under ``outputs``.
        sensor_report = dict(sensor_summary)

    attributions = _attributions(
        collision_rows,
        shift_rows,
        gap_rows,
        quality_rows,
        jloc_rows,
        outlier_rows,
        source_report,
        sensor_report,
    )

    _write_csv(logs / MANAGED_FILES[0], training_rows)
    _write_csv(logs / MANAGED_FILES[1], collision_rows)
    _write_csv(logs / MANAGED_FILES[2], shift_rows)
    _write_csv(logs / MANAGED_FILES[3], gap_rows)
    _write_csv(logs / MANAGED_FILES[4], quality_rows)
    _write_csv(logs / MANAGED_FILES[5], jloc_rows)
    _write_csv(logs / MANAGED_FILES[6], outlier_rows)

    _progress(config, "rendering diagnostic-only figures from the derived CSV tables")
    _plot_training_reward(training_rows, config.branches, plots / PLOT_FILES[0])
    _plot_loss(training_rows, config.branches, plots / PLOT_FILES[1])
    measurement_path_rows = [
        {
            **row,
            "position_rmse_norm_m": _number(
                _seed_index(selected_seed_rows)[
                    (_integer(row.get("base_seed")), str(row.get("branch")))
                ].get("position_rmse_norm_m")
            ),
        }
        for row in quality_rows
    ]
    _plot_measurement_path(measurement_path_rows, config.branches, plots / PLOT_FILES[2])
    _plot_cost_gap(gap_rows, config.branches, plots / PLOT_FILES[3])
    _plot_jloc(jloc_rows, config.branches, plots / PLOT_FILES[4])
    _plot_paired(outlier_rows, plots / PLOT_FILES[5])

    after_hashes = _source_hashes(reference, source_logs, config.seeds, config.branches)
    expected_training_cells = len(config.seeds) * len(config.branches)
    training_complete = (
        len(collision_rows) == expected_training_cells
        and all(
            _integer(row.get("training_update_count")) == config.expected_training_updates
            for row in collision_rows
        )
    )
    validation_complete = (
        len(selected_seed_rows) == expected_training_cells
        and all(
            _integer(row.get("episode_count")) == config.expected_validation_episodes
            for row in selected_seed_rows
        )
    )
    source_unchanged = before_hashes == after_hashes
    all_outputs_exist = all((logs / name).is_file() for name in MANAGED_FILES[:-1]) and all(
        (plots / name).is_file() for name in PLOT_FILES
    )
    sensor_ok = (not config.include_sensor_interaction) or bool(
        sensor_report and sensor_report.get("audit_ok", False)
    )
    audit_ok = bool(
        training_complete
        and validation_complete
        and source_unchanged
        and all_outputs_exist
        and sensor_ok
        and bool(final_report.get("audit_ok", False))
    )
    confirmed = [name for name, item in attributions.items() if item["status"] == "confirmed"]
    ruled_out = [name for name, item in attributions.items() if item["status"] == "ruled_out"]
    inconclusive = [name for name, item in attributions.items() if item["status"] == "inconclusive"]
    report: dict[str, Any] = {
        "audit": "mainline_curve_cause_audit",
        "status": "read_only_train_and_validation_diagnostic",
        "reference_root": str(reference),
        "source_logs_dir": str(source_logs),
        "split_used_for_performance_judgment": "validation",
        "train_usage": "training curves and baseline-train communication target provenance only",
        "test_data_read": False,
        "test_data_used": False,
        "trainer_update_called": False,
        "optimizer_step_called": False,
        "environment_replayed": False,
        "uav_count": 3,
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": [str(branch) for branch in config.branches],
        "independent_statistical_unit": "training seed",
        "training_updates_are_diagnostic_not_independent_samples": True,
        "validation_events_are_diagnostic_not_independent_samples": True,
        "figure_contract": {
            "classification": "diagnostic-only; not automatically manuscript-admissible",
            "raw_training_curve_preserved": True,
            "rolling_mean_window": ROLLING_WINDOW,
            "rolling_mean_purpose": "display only",
            "rolling_mean_used_for_inference": False,
            "uncertainty_policy": "no invented uncertainty; seed-level inference remains in final_mainline_audit",
        },
        "attributions": attributions,
        "confirmed_causes": confirmed,
        "ruled_out_causes": ruled_out,
        "inconclusive_causes": inconclusive,
        "priority_after_audit": [
            "If visual_odometry_interaction is confirmed, test only train-fitted VO quality covariance/gating in seed=2026 smoke.",
            "If expected_argmax_cost_mismatch is confirmed, align the communication constraint with deterministic execution without validation fitting.",
            "If non-collision single-episode variance remains confirmed, test rollout_episodes_per_update=4 with terminal-aware GAE.",
            "Keep J_loc as a covariance-uncertainty metric and use per-window update metrics in figures.",
        ],
        "checks": {
            "five_seed_three_branch_layout": len(config.seeds) == 5 and len(config.branches) == 3,
            "training_updates_complete": training_complete,
            "validation_episode_counts_complete": validation_complete,
            "validation_only_for_performance": config.split == "validation",
            "test_not_read": True,
            "checkpoint_and_source_hashes_unchanged": source_unchanged,
            "source_file_hash_count": len(before_hashes),
            "sensor_interaction_audit_ok": sensor_ok,
            "all_declared_outputs_exist": all_outputs_exist,
            "source_final_mainline_audit_ok": bool(final_report.get("audit_ok", False)),
        },
        "audit_ok": audit_ok,
        "decision_supported": bool(audit_ok and not inconclusive),
        "outputs": {
            **{Path(name).stem: str(logs / name) for name in MANAGED_FILES[:-1]},
            **{Path(name).stem: str(plots / name) for name in PLOT_FILES},
            "sensor_interaction_audit": (
                str(logs / "ekf_sensor_interaction" / "audit_report.json")
                if config.include_sensor_interaction else None
            ),
            "audit_report": str(logs / "audit_report.json"),
        },
        "interpretation_boundary": (
            "The audit attributes observed curve behavior but does not prove causality or algorithm superiority. "
            "Collision/update rows are nested diagnostics. Only seed-level paired summaries support performance claims."
        ),
    }
    _write_json(logs / "audit_report.json", _finite_json(report))
    _progress(config, f"complete audit_ok={audit_ok} confirmed={len(confirmed)}")
    return {
        "output_dir": str(logs),
        "plots_dir": str(plots),
        "audit_ok": audit_ok,
        "decision_supported": report["decision_supported"],
        "confirmed_causes": confirmed,
        "audit_report": str(logs / "audit_report.json"),
        "figures": [str(plots / name) for name in PLOT_FILES],
    }


__all__ = [
    "MainlineCurveCauseAuditConfig",
    "run_mainline_curve_cause_audit",
    "_attributions",
    "_collision_summaries",
    "_trailing_mean",
]
