"""Train and audit a train-calibrated measurement-rate control branch.

The historical MAPPO branches remain untouched.  This experiment adds one
new branch, ``rate_controlled_quality_aware_mappo``, whose training update
contains a differentiable penalty on the expected non-``none`` measurement
rate.  Targets are fitted only from the baseline branch's training split for
the same seed and UAV, while validation is used for the decision report and
test is generated but sealed.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .baseline import _read_csv, _read_json
from .formal import (
    FormalExperimentConfig,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)
from .localization_coupling_intervention import _bootstrap_interval
from .quality_aware_mappo import (
    DEFAULT_EKF_CONFIG,
    DEFAULT_PROFILE,
    _episode_rows_for_branch,
    _event_summary,
    _finite,
    _formal_base_config,
    _measurement_budget_summary,
    _position_error_p95,
    _prepare_batch_output,
    _quality_summary,
    _read_jsonl,
    _write_table,
)


DEFAULT_SEEDS = (7071, 8080, 9099, 10118, 11127)
BRANCHES = (
    "baseline_mappo",
    "budgeted_quality_aware_mappo",
    "rate_controlled_quality_aware_mappo",
)
RATE_BRANCH = "rate_controlled_quality_aware_mappo"
QUALITY_BRANCHES = {"budgeted_quality_aware_mappo", RATE_BRANCH}
METRICS = (
    "mean_reward",
    "position_rmse_m",
    "position_error_p95_m",
    "j_loc",
    "j_com",
    "joint_objective",
    "accepted_update_count",
    "beneficial_fraction",
    "harmful_fraction",
    "neutral_fraction",
    "nlos_update_fraction",
    "stale_update_fraction",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "connectivity",
    "energy_proxy",
    "collision_rate",
    "none_selection_rate",
    "mean_selected_quality_risk",
    "quality_feature_enabled",
    "measurement_count",
    "measurement_overhead_packets",
    "measurement_budget_usage_rate",
)
LOWER_IS_BETTER = {
    "position_rmse_m",
    "position_error_p95_m",
    "j_loc",
    "j_com",
    "joint_objective",
    "harmful_fraction",
    "nlos_update_fraction",
    "stale_update_fraction",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "collision_rate",
    "none_selection_rate",
    "measurement_count",
    "measurement_overhead_packets",
    "measurement_budget_usage_rate",
}


@dataclass(frozen=True)
class MeasurementRateControlledMAPPOConfig:
    """Configuration for the three-branch rate-stability experiment."""

    output_dir: Path
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    warmup_episodes: int = 20
    train_episodes: int = 100
    validation_episodes: int = 20
    test_episodes: int = 20
    scenario_profile_path: Path = DEFAULT_PROFILE
    ekf_config_path: Path = DEFAULT_EKF_CONFIG
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    device: str = "cpu"
    measurement_overhead_enabled: bool = True
    measurement_quality_logit_penalty: float = 1.0
    measurement_rate_constraint_coef: float = 0.5
    measurement_rate_tolerance: float = 0.05
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("rate-controlled MAPPO is currently fixed to 3 UAVs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if min(
            self.max_steps,
            self.warmup_episodes,
            self.train_episodes,
            self.validation_episodes,
            self.test_episodes,
        ) < 1:
            raise ValueError("steps and all episode counts must be positive")
        if self.device != "cpu":
            raise ValueError("rate-controlled MAPPO currently supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("this experiment requires measurement_overhead_enabled=True")
        if not np.isfinite(self.measurement_quality_logit_penalty) or self.measurement_quality_logit_penalty < 0.0:
            raise ValueError("measurement_quality_logit_penalty must be finite and non-negative")
        if not np.isfinite(self.measurement_rate_constraint_coef) or self.measurement_rate_constraint_coef < 0.0:
            raise ValueError("measurement_rate_constraint_coef must be finite and non-negative")
        if not np.isfinite(self.measurement_rate_tolerance) or not 0.0 <= self.measurement_rate_tolerance <= 1.0:
            raise ValueError("measurement_rate_tolerance must be finite and in [0, 1]")


def _progress(config: MeasurementRateControlledMAPPOConfig, message: str) -> None:
    if config.progress:
        print(f"[rate-controlled-mappo] {message}", flush=True)


def _fit_measurement_rate_targets(
    baseline_scale: Path,
    *,
    uav_count: int,
    base_seed: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit per-UAV rates from baseline train actions only.

    ``base_seed`` is optional for compatibility with older unit-test callers,
    but every formal multi-seed run supplies it.  Persisting the seed on each
    row is important because otherwise rows from different seed runs cannot be
    joined back to their source data without relying on file order.
    """

    train_root = baseline_scale / "train"
    episode_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"baseline train episodes are missing under {train_root}")
    uav_ids = [f"uav_{index:02d}" for index in range(uav_count)]
    selected = {uav_id: 0 for uav_id in uav_ids}
    windows = {uav_id: 0 for uav_id in uav_ids}
    for episode_root in episode_dirs:
        for row in _read_csv(episode_root / "actions.csv"):
            uav_id = str(row.get("uav_id", ""))
            if uav_id not in windows:
                continue
            windows[uav_id] += 1
            if str(row.get("measurement_uav_id", "none")) != "none":
                selected[uav_id] += 1
    if any(windows[uav_id] <= 0 for uav_id in uav_ids):
        raise RuntimeError(f"baseline train action windows are missing under {train_root}")
    targets = {
        uav_id: float(selected[uav_id] / windows[uav_id])
        for uav_id in uav_ids
    }
    rows = [
        {
            "base_seed": int(base_seed) if base_seed is not None else "",
            "uav_id": uav_id,
            "baseline_train_selected_measurements": int(selected[uav_id]),
            "baseline_train_decision_windows": int(windows[uav_id]),
            "target_measurement_rate": targets[uav_id],
            "source_split": "train",
            "source_branch": "baseline_mappo",
        }
        for uav_id in uav_ids
    ]
    return targets, rows


def _fit_measurement_budgets(baseline_scale: Path, *, uav_count: int) -> dict[str, int]:
    train_root = baseline_scale / "train"
    episode_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"baseline train episodes are missing under {train_root}")
    uav_ids = [f"uav_{index:02d}" for index in range(uav_count)]
    per_episode: list[dict[str, int]] = []
    for episode_root in episode_dirs:
        counts = {uav_id: 0 for uav_id in uav_ids}
        for row in _read_csv(episode_root / "actions.csv"):
            uav_id = str(row.get("uav_id", ""))
            if uav_id in counts and str(row.get("measurement_uav_id", "none")) != "none":
                counts[uav_id] += 1
        per_episode.append(counts)
    return {
        uav_id: max(0, int(np.rint(np.mean([row[uav_id] for row in per_episode]))))
        for uav_id in uav_ids
    }


def _formal_config(
    config: MeasurementRateControlledMAPPOConfig,
    *,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
    treatment: str,
    budgets: Mapping[str, int] | None = None,
    targets: Mapping[str, float] | None = None,
) -> FormalExperimentConfig:
    quality_enabled = treatment in QUALITY_BRANCHES
    budget_enabled = treatment in QUALITY_BRANCHES
    rate_enabled = treatment == RATE_BRANCH
    base = _formal_base_config(
        # The helper only accesses the shared experiment fields and therefore
        # accepts this config structurally; construct an explicit command line
        # below so manifests identify this experiment correctly.
        config,  # type: ignore[arg-type]
        seed,
        output_dir,
        ekf_parameters,
    )
    command_line = (
        "run-rate-controlled-quality-mappo "
        f"--seed {seed} --steps {config.max_steps} "
        f"--warmup-episodes {config.warmup_episodes} "
        f"--train-episodes {config.train_episodes} "
        f"--validation-episodes {config.validation_episodes} "
        f"--test-episodes {config.test_episodes} --measurement-overhead-enabled "
        f"--quality-logit-penalty {config.measurement_quality_logit_penalty:g} "
        f"--measurement-rate-coef {config.measurement_rate_constraint_coef:g} "
        f"--measurement-rate-tolerance {config.measurement_rate_tolerance:g}"
    )
    return FormalExperimentConfig(
        **{
            **asdict(base),
            "output_dir": output_dir,
            "command_line": command_line,
            "mappo_variant": treatment,
            "measurement_overhead_enabled": True,
            "measurement_quality_logit_penalty": (
                float(config.measurement_quality_logit_penalty) if quality_enabled else 0.0
            ),
            "measurement_budget_enabled": budget_enabled,
            "measurement_budget_by_uav": dict(budgets) if budget_enabled and budgets is not None else None,
            "measurement_rate_constraint_enabled": rate_enabled,
            "measurement_rate_constraint_coef": (
                float(config.measurement_rate_constraint_coef) if rate_enabled else 0.0
            ),
            "measurement_rate_tolerance": float(config.measurement_rate_tolerance),
            "measurement_rate_target_by_uav": (
                {str(key): float(value) for key, value in targets.items()}
                if rate_enabled and targets is not None
                else None
            ),
        }
    )


def _branch_config_summary(seed_root: Path, seed: int) -> tuple[bool, dict[str, Any]]:
    baseline = _read_json(seed_root / "baseline_mappo" / "scale_3" / "manifest.json")
    base_config = dict(baseline.get("config", {}))
    ignored = {
        "mappo_variant",
        "output_dir",
        "measurement_quality_logit_penalty",
        "measurement_budget_enabled",
        "measurement_budget_by_uav",
        "measurement_rate_constraint_enabled",
        "measurement_rate_constraint_coef",
        "measurement_rate_tolerance",
        "measurement_rate_target_by_uav",
    }
    for key in ignored:
        base_config.pop(key, None)
    equal = True
    details: dict[str, Any] = {}
    for branch in BRANCHES:
        manifest = _read_json(seed_root / branch / "scale_3" / "manifest.json")
        branch_config = dict(manifest.get("config", {}))
        raw = dict(branch_config)
        for key in ignored:
            branch_config.pop(key, None)
        equal = equal and branch_config == base_config
        details[branch] = {
            "mappo_variant": raw.get("mappo_variant"),
            "measurement_quality_logit_penalty": raw.get("measurement_quality_logit_penalty", 0.0),
            "measurement_budget_enabled": raw.get("measurement_budget_enabled", False),
            "measurement_rate_constraint_enabled": raw.get("measurement_rate_constraint_enabled", False),
            "measurement_rate_constraint_coef": raw.get("measurement_rate_constraint_coef", 0.0),
            "measurement_rate_tolerance": raw.get("measurement_rate_tolerance", 0.05),
        }
    return equal, details


def _run_seed(
    config: MeasurementRateControlledMAPPOConfig,
    *,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    ekf_parameters = load_selected_ekf_config(config.ekf_config_path)
    seed_root = root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    template = _formal_config(
        config,
        seed=seed,
        output_dir=seed_root / "metric_scale_probe",
        ekf_parameters=ekf_parameters,
        treatment="baseline_mappo",
    )
    metric_scales = fit_shared_training_metric_scales(template, (config.uav_count,))
    summaries: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    quality_models: dict[str, Any] = {}
    shared_quality_model: Mapping[str, Any] | None = None
    budgets: dict[str, int] | None = None
    targets: dict[str, float] | None = None
    target_rows: list[dict[str, Any]] = []

    for treatment in BRANCHES:
        branch_root = seed_root / treatment
        if treatment == "baseline_mappo":
            branch_budgets = None
            branch_targets = None
        else:
            if budgets is None or targets is None:
                raise RuntimeError("baseline train targets must be fitted before quality branches")
            branch_budgets = budgets
            branch_targets = targets if treatment == RATE_BRANCH else None
        _progress(config, f"seed {seed}: training {treatment}")
        formal_config = _formal_config(
            config,
            seed=seed,
            output_dir=branch_root,
            ekf_parameters=ekf_parameters,
            treatment=treatment,
            budgets=branch_budgets,
            targets=branch_targets,
        )
        formal_config = FormalExperimentConfig(
            **{
                **asdict(formal_config),
                "metric_scales": metric_scales,
                "metric_scale_source": "pooled_training_warmup_p95",
            }
        )
        summary = run_preliminary(formal_config, measurement_quality_model=shared_quality_model)
        scale_root = branch_root / "scale_3"
        report = validate_episode_benchmark(scale_root)
        if not bool(report.get("ok", False)):
            raise RuntimeError(f"episode validation failed for seed {seed}, {treatment}: {report}")
        model_path = scale_root / "calibration" / "measurement_quality_model.json"
        model = _read_json(model_path) if model_path.exists() else {}
        summaries[treatment] = summary
        reports[treatment] = report
        quality_models[treatment] = model
        if treatment == "baseline_mappo":
            if not model:
                raise RuntimeError(f"baseline warm-up quality model is missing for seed {seed}")
            shared_quality_model = model
            budgets = _fit_measurement_budgets(scale_root, uav_count=config.uav_count)
            targets, target_rows = _fit_measurement_rate_targets(
                scale_root,
                uav_count=config.uav_count,
                base_seed=seed,
            )

    configs_equal, config_details = _branch_config_summary(seed_root, seed)
    baseline_model = json.dumps(quality_models["baseline_mappo"], sort_keys=True)
    quality_models_equal = all(
        baseline_model == json.dumps(quality_models[branch], sort_keys=True)
        for branch in QUALITY_BRANCHES
    )
    return {
        "base_seed": int(seed),
        "seed_root": str(seed_root),
        "branches": {
            branch: {
                "root": str(seed_root / branch),
                "scale_root": str(seed_root / branch / "scale_3"),
                "manifest": str(seed_root / branch / "scale_3" / "manifest.json"),
                "validation_report_ok": bool(reports[branch].get("ok", False)),
                **config_details[branch],
            }
            for branch in BRANCHES
        },
        "measurement_budgets": dict(budgets or {}),
        "measurement_rate_targets": dict(targets or {}),
        "measurement_rate_target_rows": target_rows,
        "warmup_quality_models_identical_within_seed": quality_models_equal,
        "branch_configs_equal_except_expected_controls": configs_equal,
        "validation_reports": reports,
        "test_data_used": False,
    }


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


def _seed_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in rows}):
        for split in ("validation", "test"):
            for branch in BRANCHES:
                selected = [
                    row for row in rows
                    if int(row["base_seed"]) == seed
                    and str(row["split"]) == split
                    and str(row["treatment"]) == branch
                ]
                if not selected:
                    raise ValueError(f"missing rows for seed={seed}, split={split}, branch={branch}")
                output.append(
                    {
                        "base_seed": seed,
                        "split": split,
                        "treatment": branch,
                        "episode_count": len(selected),
                        **{metric: _mean(row.get(metric) for row in selected) for metric in METRICS},
                    }
                )
    return output


def _paired_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["base_seed"]), str(row["split"]), str(row["episode_id"]))
        pairs.setdefault(key, {})[str(row["treatment"])] = row
    output: list[dict[str, Any]] = []
    for (seed, split, episode_id), pair in sorted(pairs.items()):
        if set(pair) != set(BRANCHES):
            raise ValueError(f"unpaired episode {seed}/{split}/{episode_id}: {set(pair)}")
        for branch in BRANCHES[1:]:
            for metric in METRICS:
                baseline = _finite(pair[BRANCHES[0]].get(metric))
                treatment = _finite(pair[branch].get(metric))
                difference = treatment - baseline
                if not np.isfinite(difference):
                    raise ValueError(f"non-finite paired contrast for {metric}")
                output.append(
                    {
                        "base_seed": seed,
                        "split": split,
                        "episode_id": episode_id,
                        "comparison": f"{branch}_minus_baseline_mappo",
                        "metric": metric,
                        "baseline_value": baseline,
                        "treatment_value": treatment,
                        "paired_difference": difference,
                        "improvement_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    }
                )
    return output


def _cross_seed_summary(
    contrasts: Sequence[Mapping[str, Any]],
    config: MeasurementRateControlledMAPPOConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for split in ("validation", "test"):
        comparisons = sorted({str(row["comparison"]) for row in contrasts if str(row["split"]) == split})
        for comparison in comparisons:
            for metric in METRICS:
                per_seed: dict[int, list[float]] = {}
                for row in contrasts:
                    if (
                        str(row["split"]) == split
                        and str(row["comparison"]) == comparison
                        and str(row["metric"]) == metric
                    ):
                        per_seed.setdefault(int(row["base_seed"]), []).append(_finite(row["paired_difference"]))
                seed_values = np.asarray(
                    [_mean(per_seed[seed], default=float("nan")) for seed in sorted(per_seed)],
                    dtype=float,
                )
                if seed_values.size == 0 or not np.isfinite(seed_values).all():
                    continue
                low, high = _bootstrap_interval(seed_values, config.bootstrap_samples, rng)
                lower = metric in LOWER_IS_BETTER
                output.append(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        "seed_count": int(seed_values.size),
                        "mean_across_seeds": float(np.mean(seed_values)),
                        "std_across_seeds": float(np.std(seed_values, ddof=1)) if seed_values.size > 1 else 0.0,
                        "bootstrap_95_ci_low": float(low),
                        "bootstrap_95_ci_high": float(high),
                        "improvement_direction": "lower" if lower else "higher",
                        "ci_supports_improvement": bool(high < 0.0 if lower else low > 0.0),
                        "ci_supports_non_worsening": bool(high <= 0.0 if lower else low >= 0.0),
                    }
                )
    return output


def _validation_rates(
    root: Path,
    seeds: Sequence[int],
    targets_by_seed: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES:
            scale_root = root / f"seed_{seed}" / branch / "scale_3"
            counts = {f"uav_{index:02d}": [0, 0] for index in range(3)}
            split_root = scale_root / "validation"
            for episode_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                for row in _read_csv(episode_root / "actions.csv"):
                    uav_id = str(row.get("uav_id", ""))
                    if uav_id not in counts:
                        continue
                    counts[uav_id][1] += 1
                    if str(row.get("measurement_uav_id", "none")) != "none":
                        counts[uav_id][0] += 1
            rates = {
                uav_id: (float(value[0] / value[1]) if value[1] else 0.0)
                for uav_id, value in counts.items()
            }
            target = targets_by_seed[seed]
            deviations = [abs(rates[uav_id] - target[uav_id]) for uav_id in sorted(rates)]
            rows.append(
                {
                    "base_seed": seed,
                    "treatment": branch,
                    "split": "validation",
                    "measurement_rate_mean": float(np.mean(list(rates.values()))),
                    "measurement_rate_std_across_uav": float(np.std(list(rates.values()))),
                    "mean_abs_deviation_from_baseline_train_target": float(np.mean(deviations)),
                    "max_abs_deviation_from_baseline_train_target": float(np.max(deviations)),
                    "uav_rates": rates,
                    "target_rates": dict(target),
                }
            )
    return rows


def _training_rate_rows(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = root / f"seed_{seed}" / RATE_BRANCH / "scale_3" / "diagnostics" / "training_metrics.jsonl"
        for row in _read_jsonl(path):
            payload = {"base_seed": seed, "treatment": RATE_BRANCH, **row}
            for key in (
                "loss",
                "actor_loss",
                "critic_loss",
                "entropy",
                "clip_fraction",
                "gradient_norm",
                "measurement_rate_constraint_loss",
                "measurement_rate_constraint_enabled",
            ):
                payload[key] = _finite(payload.get(key), 0.0)
            rows.append(payload)
    return rows


def _budget_violations(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES[1:]:
            branch_root = root / f"seed_{seed}" / branch / "scale_3"
            for split in ("train", "validation", "test"):
                split_root = branch_root / split
                for episode_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                    scenario = _read_json(episode_root / "scenario.json")
                    limits = {
                        str(key): int(_finite(value, 0.0))
                        for key, value in (scenario.get("measurement_budget_limits", {}) or {}).items()
                    }
                    counts: dict[str, int] = {}
                    for row in _read_csv(episode_root / "actions.csv"):
                        uav_id = str(row.get("uav_id", ""))
                        if str(row.get("measurement_uav_id", "none")) != "none":
                            counts[uav_id] = counts.get(uav_id, 0) + 1
                    for uav_id, count in counts.items():
                        limit = limits.get(uav_id, 0)
                        if count > limit:
                            violations.append(
                                {
                                    "base_seed": seed,
                                    "branch": branch,
                                    "split": split,
                                    "episode_id": episode_root.name,
                                    "uav_id": uav_id,
                                    "count": count,
                                    "limit": limit,
                                }
                            )
    return violations


def _audit(
    config: MeasurementRateControlledMAPPOConfig,
    seed_runs: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seed_means: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
    validation_rates: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    budget_violations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    structural = (
        len(seed_runs) == len(config.seeds)
        and all(
            bool(run.get("branches", {}).get(branch, {}).get("validation_report_ok", False))
            for run in seed_runs for branch in BRANCHES
        )
        and all(
            int(row.get("episode_count", 0))
            == (config.validation_episodes if str(row.get("split")) == "validation" else config.test_episodes)
            for row in seed_means
        )
    )
    variants_ok = all(
        run.get("branches", {}).get("baseline_mappo", {}).get("mappo_variant") == "baseline_mappo"
        and run.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get("mappo_variant")
        == "budgeted_quality_aware_mappo"
        and run.get("branches", {}).get(RATE_BRANCH, {}).get("mappo_variant") == RATE_BRANCH
        for run in seed_runs
    )
    quality_ok = all(
        not bool(run.get("branches", {}).get("baseline_mappo", {}).get("measurement_budget_enabled", True))
        and bool(run.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get("measurement_budget_enabled", False))
        and bool(run.get("branches", {}).get(RATE_BRANCH, {}).get("measurement_budget_enabled", False))
        and not bool(run.get("branches", {}).get("baseline_mappo", {}).get("measurement_rate_constraint_enabled", True))
        and not bool(run.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get("measurement_rate_constraint_enabled", True))
        and bool(run.get("branches", {}).get(RATE_BRANCH, {}).get("measurement_rate_constraint_enabled", False))
        for run in seed_runs
    )
    targets_ok = (
        len(target_rows) == len(config.seeds) * config.uav_count
        and all(
            str(row.get("source_split")) == "train"
            and str(row.get("source_branch")) == "baseline_mappo"
            and 0.0 <= _finite(row.get("target_measurement_rate"), -1.0) <= 1.0
            for row in target_rows
        )
    )
    finite = all(
        np.isfinite(_finite(row.get(metric)))
        for row in episode_rows for metric in METRICS
    ) and all(
        np.isfinite(_finite(row.get("measurement_rate_constraint_loss"), 0.0))
        for row in training_rows
    )
    rates_finite = all(
        np.isfinite(_finite(row.get("measurement_rate_mean")))
        and np.isfinite(_finite(row.get("mean_abs_deviation_from_baseline_train_target")))
        for row in validation_rates
    )
    expected_rate_rows = len(config.seeds) * len(BRANCHES)
    rate_rows_complete = len(validation_rates) == expected_rate_rows
    rate_branch_rows = [row for row in validation_rates if str(row.get("treatment")) == RATE_BRANCH]
    within_target_count = sum(
        _finite(row.get("max_abs_deviation_from_baseline_train_target"), float("inf")) <= 0.10
        for row in rate_branch_rows
    )
    branch_ranges: dict[str, float] = {}
    branch_stds: dict[str, float] = {}
    for branch in BRANCHES[1:]:
        values = np.asarray(
            [_finite(row.get("measurement_rate_mean")) for row in validation_rates if str(row.get("treatment")) == branch],
            dtype=float,
        )
        branch_ranges[branch] = float(np.ptp(values)) if values.size else float("nan")
        branch_stds[branch] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    stability_supported = bool(
        rate_rows_complete
        and within_target_count >= int(math.ceil(0.8 * len(config.seeds)))
        and np.isfinite(branch_ranges.get(RATE_BRANCH, float("nan")))
        and np.isfinite(branch_ranges.get("budgeted_quality_aware_mappo", float("nan")))
        and branch_ranges[RATE_BRANCH] < branch_ranges["budgeted_quality_aware_mappo"]
        and branch_stds[RATE_BRANCH] < branch_stds["budgeted_quality_aware_mappo"]
    )
    mean_by_branch: dict[str, dict[str, float]] = {}
    for branch in BRANCHES:
        selected = [
            row for row in seed_means
            if str(row.get("split")) == "validation" and str(row.get("treatment")) == branch
        ]
        mean_by_branch[branch] = {
            metric: _mean(
                (row.get(metric) for row in selected),
                default=float("nan"),
            )
            for metric in ("position_rmse_m", "j_com", "harmful_fraction", "mean_reward")
        }
    rate_mean = mean_by_branch.get(RATE_BRANCH, {})
    baseline_mean = mean_by_branch.get("baseline_mappo", {})
    not_clearly_worsened = bool(
        rate_mean.get("position_rmse_m", float("inf"))
        <= baseline_mean.get("position_rmse_m", float("inf")) + max(0.05, 0.10 * abs(baseline_mean.get("position_rmse_m", 0.0)))
        and rate_mean.get("j_com", float("inf"))
        <= baseline_mean.get("j_com", float("inf")) + max(0.01, 0.10 * abs(baseline_mean.get("j_com", 0.0)))
        and rate_mean.get("harmful_fraction", float("inf"))
        <= baseline_mean.get("harmful_fraction", float("inf")) + 0.05
    )
    audit_ok = bool(
        structural
        and variants_ok
        and quality_ok
        and targets_ok
        and finite
        and rates_finite
        and not budget_violations
        and all(bool(run.get("warmup_quality_models_identical_within_seed", False)) for run in seed_runs)
        and all(bool(run.get("branch_configs_equal_except_expected_controls", False)) for run in seed_runs)
    )
    return {
        "experiment": "measurement_rate_controlled_quality_aware_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evidence_status": "engineering_stability_validation_only",
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_used_for_selection": False,
        "test_data_generated_and_sealed": True,
        "fit_sources": {
            "measurement_rate_targets": "baseline_mappo/train actions only",
            "quality_model": "warm-up actor-visible observations only",
            "measurement_budgets": "baseline_mappo/train actions only",
        },
        "fixed_controls": {
            "measurement_rate_constraint_coef": float(config.measurement_rate_constraint_coef),
            "measurement_rate_tolerance": float(config.measurement_rate_tolerance),
            "measurement_overhead_enabled": True,
            "quality_logit_penalty": float(config.measurement_quality_logit_penalty),
        },
        "checks": {
            "seed_count": len(config.seeds),
            "all_branch_validation_reports_ok": structural,
            "branch_variants_correct": variants_ok,
            "quality_budget_rate_controls_correct": quality_ok,
            "train_only_targets_complete": targets_ok,
            "all_episode_metrics_finite": finite,
            "validation_rate_rows_complete": rate_rows_complete,
            "training_constraint_rows_finite": all(
                np.isfinite(_finite(row.get("measurement_rate_constraint_loss"), 0.0)) for row in training_rows
            ),
            "budget_violations": len(budget_violations),
            "warmup_quality_models_identical_within_seed": all(
                bool(run.get("warmup_quality_models_identical_within_seed", False)) for run in seed_runs
            ),
            "branch_configs_equal_except_expected_controls": all(
                bool(run.get("branch_configs_equal_except_expected_controls", False)) for run in seed_runs
            ),
            "measurement_rate_range": branch_ranges,
            "measurement_rate_std": branch_stds,
            "rate_branch_within_target_count": int(within_target_count),
            "required_within_target_count": int(math.ceil(0.8 * len(config.seeds))),
            "not_clearly_worsened": not_clearly_worsened,
        },
        "decision_gate": (
            "measurement_policy_stability_supported"
            if stability_supported and not_clearly_worsened
            else "measurement_policy_stability_not_supported"
        ),
        "audit_ok": audit_ok,
        "stability_supported": stability_supported,
        "decision_supported": bool(audit_ok and stability_supported and not_clearly_worsened),
        "interpretation_boundary": (
            "This run tests whether a fixed train-calibrated expected measurement-rate constraint "
            "stabilizes the measurement policy. It does not by itself establish algorithmic superiority, "
            "a final reward redesign, collision safety, or scalability to 5/7 UAVs."
        ),
    }


def _plot(
    validation_rates: Sequence[Mapping[str, Any]],
    seed_means: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "budgeted_quality_aware_mappo": "#dc2626",
        RATE_BRANCH: "#16a34a",
    }
    labels = {
        "baseline_mappo": "Baseline MAPPO",
        "budgeted_quality_aware_mappo": "Budgeted quality-aware",
        RATE_BRANCH: "Rate-controlled quality-aware",
    }
    seeds = sorted({int(row["base_seed"]) for row in validation_rates})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for branch in BRANCHES:
        selected = [row for row in validation_rates if str(row.get("treatment")) == branch]
        selected.sort(key=lambda row: int(row["base_seed"]))
        axes[0, 0].plot(
            [int(row["base_seed"]) for row in selected],
            [float(row["measurement_rate_mean"]) for row in selected],
            marker="o",
            label=labels[branch],
            color=colors[branch],
        )
        axes[0, 1].plot(
            [int(row["base_seed"]) for row in selected],
            [float(row["mean_abs_deviation_from_baseline_train_target"]) for row in selected],
            marker="o",
            label=labels[branch],
            color=colors[branch],
        )
    axes[0, 0].set_title("Validation measurement rate")
    axes[0, 0].set_ylabel("Non-none selection rate")
    axes[0, 1].set_title("Deviation from baseline train target")
    axes[0, 1].set_ylabel("Mean absolute deviation")
    for axis in axes[0]:
        axis.set_xlabel("Seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    for axis, metric, title in (
        (axes[1, 0], "position_rmse_m", "Validation position RMSE (m)"),
        (axes[1, 1], "j_com", "Validation communication cost"),
    ):
        for branch in BRANCHES:
            selected = [
                row for row in seed_means
                if str(row.get("split")) == "validation" and str(row.get("treatment")) == branch
            ]
            selected.sort(key=lambda row: int(row["base_seed"]))
            axis.plot(
                [int(row["base_seed"]) for row in selected],
                [_finite(row.get(metric), 0.0) for row in selected],
                marker="o",
                label=labels[branch],
                color=colors[branch],
            )
        axis.set_title(title)
        axis.set_xlabel("Seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / "measurement_policy_stability.png"
    figure.suptitle("Measurement-rate stability diagnostic (validation; test sealed)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def run_measurement_rate_controlled_mappo(
    config: MeasurementRateControlledMAPPOConfig,
) -> dict[str, Any]:
    """Run the Smoke or formal multi-seed rate-control experiment."""

    root, logs_dir, plots_dir = _prepare_batch_output(config)  # type: ignore[arg-type]
    if not Path(config.scenario_profile_path).exists():
        raise FileNotFoundError(f"scenario profile not found: {config.scenario_profile_path}")
    if not Path(config.ekf_config_path).exists():
        raise FileNotFoundError(f"EKF config not found: {config.ekf_config_path}")
    seed_runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    targets_by_seed: dict[int, dict[str, float]] = {}
    for seed in config.seeds:
        run = _run_seed(config, seed=int(seed), root=root)
        seed_runs.append(run)
        target_rows.extend(run["measurement_rate_target_rows"])
        targets_by_seed[int(seed)] = {
            str(key): float(value)
            for key, value in (run.get("measurement_rate_targets") or {}).items()
        }
        for branch in BRANCHES:
            episode_rows.extend(
                _episode_rows_for_branch(
                    root / f"seed_{int(seed)}" / branch / "scale_3",
                    branch,
                    int(seed),
                )
            )
    seed_means = _seed_means(episode_rows)
    contrasts = _paired_contrasts(episode_rows)
    cross_seed = _cross_seed_summary(contrasts, config)
    validation_rates = _validation_rates(root, config.seeds, targets_by_seed)
    training_rows = _training_rate_rows(root, config.seeds)
    budget_violations = _budget_violations(root, config.seeds)
    audit = _audit(
        config,
        seed_runs,
        episode_rows,
        seed_means,
        cross_seed,
        validation_rates,
        target_rows,
        training_rows,
        budget_violations,
    )
    plot = _plot(validation_rates, seed_means, plots_dir)

    _write_table(logs_dir / "measurement_rate_targets.csv", target_rows)
    _write_table(logs_dir / "training_rate_constraint.csv", training_rows)
    _write_table(logs_dir / "validation_policy_metrics.csv", [
        row for row in seed_means if str(row.get("split")) == "validation"
    ])
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "seed_comparison.csv", seed_means)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "validation_measurement_rates.csv", validation_rates)
    _write_table(logs_dir / "budget_violations.csv", budget_violations)
    _write_json(logs_dir / "audit_report.json", audit)
    manifest = {
        "dataset_id": "python_sim_measurement_rate_controlled_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_stability_validation_only",
        "uav_count": config.uav_count,
        "seed_set": [int(seed) for seed in config.seeds],
        "training": {
            "warmup_episodes": config.warmup_episodes,
            "train_episodes": config.train_episodes,
            "validation_episodes": config.validation_episodes,
            "test_episodes_generated_but_sealed": config.test_episodes,
        },
        "branches": {
            "baseline_mappo": "historical actor; no quality, budget, or rate constraint",
            "budgeted_quality_aware_mappo": "historical quality-aware actor plus fixed train-derived budget",
            RATE_BRANCH: "quality-aware actor plus fixed train-derived budget and train-only expected rate constraint",
        },
        "measurement_rate_constraint": {
            "coefficient": float(config.measurement_rate_constraint_coef),
            "tolerance": float(config.measurement_rate_tolerance),
            "loss_applies_during": "training_updates_only",
            "target_source": "baseline_mappo/train actions only",
            "target_is_per_seed_and_uav": True,
        },
        "measurement_overhead": {
            "enabled": True,
            "equivalent_packet_size_bytes": 512,
            "one_equivalent_packet_per_selected_measurement": True,
        },
        "independent_unit": "base training seed",
        "test_data_used_for_selection": False,
        "outputs": {
            "measurement_rate_targets": str(logs_dir / "measurement_rate_targets.csv"),
            "training_rate_constraint": str(logs_dir / "training_rate_constraint.csv"),
            "validation_policy_metrics": str(logs_dir / "validation_policy_metrics.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "seed_comparison": str(logs_dir / "seed_comparison.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
            "comparison_plot": str(plot),
        },
        "audit": audit,
    }
    _write_json(root / "manifest.json", manifest)
    return {
        "output_dir": str(root),
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "seeds": [int(seed) for seed in config.seeds],
        "audit_ok": bool(audit["audit_ok"]),
        "stability_supported": bool(audit["stability_supported"]),
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
        "comparison_plot": str(plot),
        "audit_report": str(logs_dir / "audit_report.json"),
    }


__all__ = [
    "BRANCHES",
    "DEFAULT_SEEDS",
    "MeasurementRateControlledMAPPOConfig",
    "RATE_BRANCH",
    "run_measurement_rate_controlled_mappo",
]
