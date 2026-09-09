"""Adaptive measurement-rate constrained MAPPO experiment.

This module is deliberately separate from the historical fixed-rate runner.
It reuses the same environment and episode writer, but adds one new branch,
``adaptive_budgeted_quality_aware_mappo``.  The branch keeps the frozen
quality features and train-derived measurement budget, while its train-only
rate penalty is weighted by one detached multiplier per UAV.  The multiplier
is updated after each MAPPO update and is persisted in the checkpoint.

The baseline and the historical budgeted branch in this experiment are
controls.  They use the same scene profile, overhead mechanism, optimizer
settings, and train/validation/test separation.  Validation and test never
update the multiplier or fit any target.  Test is generated for sealing and
is not used by the decision gate.
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
from .measurement_rate_controlled_mappo import (
    _fit_measurement_budgets,
    _fit_measurement_rate_targets,
)
from .quality_aware_mappo import (
    DEFAULT_EKF_CONFIG,
    DEFAULT_PROFILE,
    _episode_rows_for_branch,
    _event_summary,
    _finite,
    _measurement_budget_summary,
    _position_error_p95,
    _quality_summary,
    _read_jsonl,
    _write_table,
)


DEFAULT_SEEDS = (7071, 8080, 9099, 10118, 11127)
BASELINE_BRANCH = "baseline_mappo"
BUDGETED_BRANCH = "budgeted_quality_aware_mappo"
ADAPTIVE_BRANCH = "adaptive_budgeted_quality_aware_mappo"
BRANCHES = (BASELINE_BRANCH, BUDGETED_BRANCH, ADAPTIVE_BRANCH)
QUALITY_BRANCHES = {BUDGETED_BRANCH, ADAPTIVE_BRANCH}
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


@dataclass(frozen=True)
class AdaptiveMeasurementRateMAPPOConfig:
    """Configuration for the adaptive three-branch stability experiment."""

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
    measurement_rate_tolerance: float = 0.05
    measurement_rate_dual_initial: float = 0.0
    measurement_rate_dual_step_size: float = 0.25
    measurement_rate_dual_max: float = 5.0
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("adaptive measurement-rate MAPPO is currently fixed to 3 UAVs")
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
            raise ValueError("adaptive measurement-rate MAPPO currently supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("this experiment requires measurement_overhead_enabled=True")
        for name, value in (
            ("measurement_quality_logit_penalty", self.measurement_quality_logit_penalty),
            ("measurement_rate_tolerance", self.measurement_rate_tolerance),
            ("measurement_rate_dual_initial", self.measurement_rate_dual_initial),
            ("measurement_rate_dual_step_size", self.measurement_rate_dual_step_size),
            ("measurement_rate_dual_max", self.measurement_rate_dual_max),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.measurement_rate_tolerance) <= 1.0:
            raise ValueError("measurement_rate_tolerance must be in [0, 1]")
        if float(self.measurement_rate_dual_initial) > float(self.measurement_rate_dual_max):
            raise ValueError("measurement_rate_dual_initial must not exceed measurement_rate_dual_max")
        if float(self.measurement_rate_dual_step_size) <= 0.0:
            raise ValueError("measurement_rate_dual_step_size must be positive")
        if float(self.measurement_rate_dual_max) <= 0.0:
            raise ValueError("measurement_rate_dual_max must be positive")


def _progress(config: AdaptiveMeasurementRateMAPPOConfig, message: str) -> None:
    if config.progress:
        print(f"[adaptive-rate-mappo] {message}", flush=True)


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


def _prepare_batch_output(
    config: AdaptiveMeasurementRateMAPPOConfig,
) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / root.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / root.name
    targets = (root, logs, plots)
    if not config.force:
        nonempty = [str(path) for path in targets if path.exists() and any(path.iterdir())]
        if nonempty:
            raise FileExistsError(
                "adaptive MAPPO output is not empty; use --force to replace only: "
                + ", ".join(nonempty)
            )
    else:
        for path in targets:
            if path.exists():
                shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
    return root, logs, plots


def _formal_config(
    config: AdaptiveMeasurementRateMAPPOConfig,
    *,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
    treatment: str,
    budgets: Mapping[str, int] | None = None,
    targets: Mapping[str, float] | None = None,
) -> FormalExperimentConfig:
    base = FormalExperimentConfig(
        output_dir=output_dir,
        uav_count=config.uav_count,
        seed=int(seed),
        max_steps=config.max_steps,
        warmup_episodes=config.warmup_episodes,
        train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes,
        test_episodes=config.test_episodes,
        device=config.device,
        force=False,
        vo_position_std_m=float(ekf_parameters["vo_position_std_m"]),
        process_position_std_m=float(ekf_parameters["process_position_std_m"]),
        process_velocity_std_mps=float(ekf_parameters["process_velocity_std_mps"]),
        nlos_variance_multiplier=float(ekf_parameters["nlos_variance_multiplier"]),
        scenario_profile_path=Path(config.scenario_profile_path),
        ekf_config_source=str(config.ekf_config_path),
        command_line=(
            "run-adaptive-budgeted-quality-mappo "
            f"--seed {seed} --steps {config.max_steps} "
            f"--warmup-episodes {config.warmup_episodes} "
            f"--train-episodes {config.train_episodes} "
            f"--validation-episodes {config.validation_episodes} "
            f"--test-episodes {config.test_episodes} --measurement-overhead-enabled "
            f"--quality-logit-penalty {config.measurement_quality_logit_penalty:g} "
            f"--dual-initial {config.measurement_rate_dual_initial:g} "
            f"--dual-step-size {config.measurement_rate_dual_step_size:g} "
            f"--dual-max {config.measurement_rate_dual_max:g}"
        ),
        progress=config.progress,
        mappo_variant=treatment,
        measurement_overhead_enabled=True,
        measurement_quality_logit_penalty=(
            float(config.measurement_quality_logit_penalty) if treatment in QUALITY_BRANCHES else 0.0
        ),
        measurement_budget_enabled=treatment in QUALITY_BRANCHES,
        measurement_budget_by_uav=(dict(budgets) if treatment in QUALITY_BRANCHES and budgets is not None else None),
        measurement_rate_constraint_enabled=treatment == ADAPTIVE_BRANCH,
        # The adaptive branch uses the detached multiplier loss instead of the
        # historical fixed coefficient.  Setting this to zero avoids adding
        # two rate penalties while keeping the legacy branch untouched.
        measurement_rate_constraint_coef=0.0,
        measurement_rate_tolerance=float(config.measurement_rate_tolerance),
        measurement_rate_target_by_uav=(dict(targets) if treatment == ADAPTIVE_BRANCH and targets is not None else None),
        measurement_rate_dual_enabled=treatment == ADAPTIVE_BRANCH,
        measurement_rate_dual_initial=float(config.measurement_rate_dual_initial),
        measurement_rate_dual_step_size=float(config.measurement_rate_dual_step_size),
        measurement_rate_dual_max=float(config.measurement_rate_dual_max),
    )
    return base


def _run_seed(
    config: AdaptiveMeasurementRateMAPPOConfig,
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
        treatment=BASELINE_BRANCH,
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
        if treatment == BASELINE_BRANCH:
            branch_budgets = None
            branch_targets = None
        else:
            if budgets is None or targets is None:
                raise RuntimeError("baseline train targets must be fitted before quality branches")
            branch_budgets = budgets
            branch_targets = targets if treatment == ADAPTIVE_BRANCH else None
        branch_root = seed_root / treatment
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
        if treatment == BASELINE_BRANCH:
            if not model:
                raise RuntimeError(f"baseline warm-up quality model is missing for seed {seed}")
            shared_quality_model = model
            budgets = _fit_measurement_budgets(scale_root, uav_count=config.uav_count)
            targets, target_rows = _fit_measurement_rate_targets(
                scale_root,
                uav_count=config.uav_count,
                base_seed=seed,
            )

    baseline_model = json.dumps(quality_models[BASELINE_BRANCH], sort_keys=True)
    quality_models_equal = all(
        baseline_model == json.dumps(quality_models[branch], sort_keys=True)
        for branch in QUALITY_BRANCHES
    )
    baseline_manifest = _read_json(seed_root / BASELINE_BRANCH / "scale_3" / "manifest.json")
    baseline_config = dict(baseline_manifest.get("config", {}))
    ignored = {
        "mappo_variant",
        "output_dir",
        "measurement_quality_logit_penalty",
        "measurement_budget_enabled",
        "measurement_budget_by_uav",
        "measurement_rate_constraint_enabled",
        "measurement_rate_constraint_coef",
        "measurement_rate_target_by_uav",
        "measurement_rate_dual_enabled",
        "measurement_rate_dual_initial",
        "measurement_rate_dual_step_size",
        "measurement_rate_dual_max",
    }
    for key in ignored:
        baseline_config.pop(key, None)
    branch_configs_equal = True
    details: dict[str, Any] = {}
    for treatment in BRANCHES:
        manifest = _read_json(seed_root / treatment / "scale_3" / "manifest.json")
        raw = dict(manifest.get("config", {}))
        comparable = dict(raw)
        for key in ignored:
            comparable.pop(key, None)
        branch_configs_equal = branch_configs_equal and comparable == baseline_config
        details[treatment] = {
            "mappo_variant": raw.get("mappo_variant"),
            "quality_features_enabled": bool(
                summaries[treatment].get("measurement_quality", {}).get("enabled", False)
            ),
            "measurement_budget_enabled": bool(summaries[treatment].get("measurement_budget_enabled", False)),
            "measurement_rate_constraint_enabled": bool(
                summaries[treatment].get("measurement_rate_constraint", {}).get("enabled", False)
            ),
            "measurement_rate_dual_enabled": bool(
                summaries[treatment].get("measurement_rate_constraint", {}).get("dual_enabled", False)
            ),
            "measurement_rate_dual_state": summaries[treatment].get("measurement_rate_dual_state", {}),
            "validation_report_ok": bool(reports[treatment].get("ok", False)),
            "scale_root": str(seed_root / treatment / "scale_3"),
        }
    return {
        "base_seed": int(seed),
        "seed_root": str(seed_root),
        "branches": details,
        "measurement_budgets": dict(budgets or {}),
        "measurement_rate_targets": dict(targets or {}),
        "measurement_rate_target_rows": target_rows,
        "warmup_quality_models_identical": quality_models_equal,
        "branch_configs_equal_except_expected_controls": branch_configs_equal,
        "validation_reports": reports,
        "test_data_used": False,
    }


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
                output.append({
                    "base_seed": seed,
                    "split": split,
                    "treatment": branch,
                    "episode_count": len(selected),
                    **{
                        metric: _mean(item.get(metric) for item in selected)
                        for metric in METRICS
                    },
                })
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
        baseline = pair[BASELINE_BRANCH]
        for treatment in BRANCHES[1:]:
            for metric in METRICS:
                baseline_value = _finite(baseline.get(metric))
                treatment_value = _finite(pair[treatment].get(metric))
                difference = treatment_value - baseline_value
                if not np.isfinite(difference):
                    raise ValueError(f"non-finite paired contrast for {metric}")
                output.append({
                    "base_seed": seed,
                    "split": split,
                    "episode_id": episode_id,
                    "comparison": f"{treatment}_minus_{BASELINE_BRANCH}",
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "treatment_value": treatment_value,
                    "paired_difference": difference,
                    "improvement_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                })
    return output


def _cross_seed_summary(
    contrasts: Sequence[Mapping[str, Any]],
    config: AdaptiveMeasurementRateMAPPOConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for split in ("validation", "test"):
        for comparison in sorted({
            str(row["comparison"]) for row in contrasts if str(row["split"]) == split
        }):
            for metric in METRICS:
                values_by_seed: dict[int, list[float]] = {}
                for row in contrasts:
                    if (
                        str(row["split"]) == split
                        and str(row["comparison"]) == comparison
                        and str(row["metric"]) == metric
                    ):
                        values_by_seed.setdefault(int(row["base_seed"]), []).append(
                            _finite(row["paired_difference"])
                        )
                if not values_by_seed:
                    continue
                seed_values = np.asarray(
                    [_mean(values_by_seed[seed]) for seed in sorted(values_by_seed)],
                    dtype=float,
                )
                if not np.isfinite(seed_values).all():
                    raise ValueError(f"non-finite seed contrast for {split}/{comparison}/{metric}")
                low, high = _bootstrap_interval(seed_values, config.bootstrap_samples, rng)
                lower = metric in LOWER_IS_BETTER
                output.append({
                    "split": split,
                    "comparison": comparison,
                    "metric": metric,
                    "seed_count": int(seed_values.size),
                    "mean_across_seeds": float(np.mean(seed_values)),
                    "std_across_seeds": float(np.std(seed_values, ddof=1)) if seed_values.size > 1 else 0.0,
                    "bootstrap_95_ci_low": float(low),
                    "bootstrap_95_ci_high": float(high),
                    "improvement_direction": "lower" if lower else "higher",
                    "improved_seed_count": int(np.sum(seed_values < 0.0 if lower else seed_values > 0.0)),
                    "not_worsened_seed_count": int(np.sum(seed_values <= 0.0 if lower else seed_values >= 0.0)),
                    "ci_supports_improvement": bool(high < 0.0 if lower else low > 0.0),
                    "ci_supports_non_worsening": bool(high <= 0.0 if lower else low >= 0.0),
                })
    return output


def _validation_rates(
    root: Path,
    seeds: Sequence[int],
    targets_by_seed: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES:
            counts = {f"uav_{index:02d}": [0, 0] for index in range(3)}
            split_root = root / f"seed_{seed}" / branch / "scale_3" / "validation"
            for episode_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                for row in _read_csv(episode_root / "actions.csv"):
                    uav_id = str(row.get("uav_id", ""))
                    if uav_id not in counts:
                        continue
                    counts[uav_id][1] += 1
                    if str(row.get("measurement_uav_id", "none")) != "none":
                        counts[uav_id][0] += 1
            rates = {
                uav_id: float(value[0] / value[1]) if value[1] else 0.0
                for uav_id, value in counts.items()
            }
            target = targets_by_seed[seed]
            deviations = [abs(rates[uav_id] - target[uav_id]) for uav_id in sorted(rates)]
            rows.append({
                "base_seed": int(seed),
                "treatment": branch,
                "split": "validation",
                "measurement_rate_mean": float(np.mean(list(rates.values()))),
                "measurement_rate_std_across_uav": float(np.std(list(rates.values()))),
                "mean_abs_deviation_from_baseline_train_target": float(np.mean(deviations)),
                "max_abs_deviation_from_baseline_train_target": float(np.max(deviations)),
                "uav_rates": json.dumps(rates, sort_keys=True),
                "target_rates": json.dumps(target, sort_keys=True),
            })
    return rows


def _training_rows(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES:
            path = root / f"seed_{seed}" / branch / "scale_3" / "diagnostics" / "training_metrics.jsonl"
            for source in _read_jsonl(path):
                row = {"base_seed": int(seed), "treatment": branch, **source}
                for key, default in (
                    ("loss", 0.0),
                    ("actor_loss", 0.0),
                    ("critic_loss", 0.0),
                    ("entropy", 0.0),
                    ("clip_fraction", 0.0),
                    ("gradient_norm", 0.0),
                    ("measurement_rate_constraint_loss", 0.0),
                    ("measurement_rate_constraint_weighted_loss", 0.0),
                    ("measurement_rate_dual_loss", 0.0),
                    ("measurement_rate_dual_enabled", 0.0),
                ):
                    row[key] = _finite(row.get(key), default)
                for uav_index in range(3):
                    uav_id = f"uav_{uav_index:02d}"
                    for prefix in ("measurement_rate_dual", "measurement_rate_dual_update", "measurement_rate_violation"):
                        key = f"{prefix}_{uav_id}"
                        row[key] = _finite(row.get(key), 0.0)
                rows.append(row)
    return rows


def _budget_violations(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in (BUDGETED_BRANCH, ADAPTIVE_BRANCH):
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
                        if count > limits.get(uav_id, 0):
                            violations.append({
                                "base_seed": int(seed),
                                "branch": branch,
                                "split": split,
                                "episode_id": episode_root.name,
                                "uav_id": uav_id,
                                "count": int(count),
                                "limit": int(limits.get(uav_id, 0)),
                            })
    return violations


def _checkpoint_dual_audit(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = root / f"seed_{seed}" / ADAPTIVE_BRANCH / "scale_3" / "checkpoints" / "final.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("measurement_rate_dual_state", {})
        valid = set(state) == {"uav_00", "uav_01", "uav_02"}
        values: dict[str, float] = {}
        for uav_id in ("uav_00", "uav_01", "uav_02"):
            value = _finite(state.get(uav_id), float("nan"))
            values[uav_id] = value
            valid = valid and np.isfinite(value) and 0.0 <= value <= 5.0
        rows.append({"base_seed": int(seed), "checkpoint": str(path), "dual_state_valid": bool(valid), **values})
    return rows


def _audit(
    config: AdaptiveMeasurementRateMAPPOConfig,
    seed_runs: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seed_means: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
    validation_rates: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    budget_violations: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = int(math.ceil(0.8 * len(config.seeds)))
    structural = (
        len(seed_runs) == len(config.seeds)
        and all(
            bool(run.get("branches", {}).get(branch, {}).get("validation_report_ok", False))
            for run in seed_runs for branch in BRANCHES
        )
        and all(
            int(row.get("episode_count", 0)) == (
                config.validation_episodes if str(row.get("split")) == "validation" else config.test_episodes
            )
            for row in seed_means
        )
    )
    variants_ok = all(
        run.get("branches", {}).get(BASELINE_BRANCH, {}).get("mappo_variant") == BASELINE_BRANCH
        and run.get("branches", {}).get(BUDGETED_BRANCH, {}).get("mappo_variant") == BUDGETED_BRANCH
        and run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("mappo_variant") == ADAPTIVE_BRANCH
        for run in seed_runs
    )
    controls_ok = all(
        not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("quality_features_enabled", True))
        and bool(run.get("branches", {}).get(BUDGETED_BRANCH, {}).get("quality_features_enabled", False))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("quality_features_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_budget_enabled", True))
        and bool(run.get("branches", {}).get(BUDGETED_BRANCH, {}).get("measurement_budget_enabled", False))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("measurement_budget_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_rate_constraint_enabled", True))
        and not bool(run.get("branches", {}).get(BUDGETED_BRANCH, {}).get("measurement_rate_constraint_enabled", True))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("measurement_rate_constraint_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_rate_dual_enabled", True))
        and not bool(run.get("branches", {}).get(BUDGETED_BRANCH, {}).get("measurement_rate_dual_enabled", True))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("measurement_rate_dual_enabled", False))
        for run in seed_runs
    )
    target_ok = (
        len(target_rows) == len(config.seeds) * config.uav_count
        and all(
            str(row.get("source_split")) == "train"
            and str(row.get("source_branch")) == BASELINE_BRANCH
            and 0.0 <= _finite(row.get("target_measurement_rate"), -1.0) <= 1.0
            for row in target_rows
        )
    )
    finite = all(
        np.isfinite(_finite(row.get(metric)))
        for row in episode_rows for metric in METRICS
    )
    training_finite = all(
        np.isfinite(_finite(row.get("loss"), 0.0))
        and np.isfinite(_finite(row.get("measurement_rate_dual_loss"), 0.0))
        and all(np.isfinite(_finite(row.get(f"measurement_rate_dual_uav_{index:02d}"), 0.0)) for index in range(3))
        for row in training_rows
    )
    dual_bounds = all(
        0.0 <= _finite(row.get(f"uav_{index:02d}"), float("nan")) <= 5.0
        for row in checkpoint_rows for index in range(3)
    ) and all(bool(row.get("dual_state_valid", False)) for row in checkpoint_rows)
    rates_complete = len(validation_rates) == len(config.seeds) * len(BRANCHES)
    adaptive_rates = [
        row for row in validation_rates if str(row.get("treatment")) == ADAPTIVE_BRANCH
    ]
    within_target_count = sum(
        _finite(row.get("max_abs_deviation_from_baseline_train_target"), float("inf")) <= 0.10
        for row in adaptive_rates
    )
    comparison = f"{ADAPTIVE_BRANCH}_minus_{BASELINE_BRANCH}"
    validation_contrasts = {
        str(row["metric"]): row
        for row in cross_seed
        if str(row.get("split")) == "validation" and str(row.get("comparison")) == comparison
    }
    rmse = validation_contrasts.get("position_rmse_m", {})
    harmful = validation_contrasts.get("harmful_fraction", {})
    j_com = validation_contrasts.get("j_com", {})
    reward = validation_contrasts.get("mean_reward", {})
    adaptive_means = {
        metric: _mean(
            row.get(metric) for row in seed_means
            if str(row.get("split")) == "validation" and str(row.get("treatment")) == ADAPTIVE_BRANCH
        )
        for metric in ("position_rmse_m", "j_com", "harmful_fraction", "mean_reward")
    }
    baseline_means = {
        metric: _mean(
            row.get(metric) for row in seed_means
            if str(row.get("split")) == "validation" and str(row.get("treatment")) == BASELINE_BRANCH
        )
        for metric in ("position_rmse_m", "j_com", "harmful_fraction", "mean_reward")
    }
    performance_gate = bool(
        int(rmse.get("improved_seed_count", 0)) >= required
        and int(harmful.get("not_worsened_seed_count", 0)) >= required
        and bool(rmse.get("ci_supports_improvement", False))
        and bool(harmful.get("ci_supports_non_worsening", False))
        and bool(j_com.get("ci_supports_non_worsening", False))
        and bool(reward.get("ci_supports_non_worsening", False))
    )
    stability_supported = bool(
        rates_complete
        and within_target_count >= required
        and performance_gate
    )
    audit_ok = bool(
        structural
        and variants_ok
        and controls_ok
        and target_ok
        and finite
        and training_finite
        and dual_bounds
        and not budget_violations
        and all(bool(run.get("warmup_quality_models_identical", False)) for run in seed_runs)
        and all(bool(run.get("branch_configs_equal_except_expected_controls", False)) for run in seed_runs)
    )
    return {
        "experiment": "adaptive_budgeted_quality_aware_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evidence_status": "engineering_stability_validation_only",
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_used_for_selection": False,
        "test_data_generated_and_sealed": True,
        "fit_sources": {
            "measurement_rate_targets": "baseline_mappo/train actions only",
            "measurement_budgets": "baseline_mappo/train actions only",
            "quality_model": "warm-up actor-visible observations only",
        },
        "fixed_controls": {
            "quality_logit_penalty": float(config.measurement_quality_logit_penalty),
            "measurement_overhead_enabled": True,
            "measurement_rate_tolerance": float(config.measurement_rate_tolerance),
            "dual_initial": float(config.measurement_rate_dual_initial),
            "dual_step_size": float(config.measurement_rate_dual_step_size),
            "dual_max": float(config.measurement_rate_dual_max),
        },
        "checks": {
            "seed_count": len(config.seeds),
            "branch_count": len(BRANCHES),
            "all_branch_validation_reports_ok": structural,
            "branch_variants_correct": variants_ok,
            "control_flags_correct": controls_ok,
            "train_only_targets_complete": target_ok,
            "all_episode_metrics_finite": finite,
            "training_rows_finite": training_finite,
            "checkpoint_dual_state_finite_and_bounded": dual_bounds,
            "budget_violations": len(budget_violations),
            "validation_rate_rows_complete": rates_complete,
            "adaptive_within_target_count": int(within_target_count),
            "required_within_target_count": required,
            "performance_gate": performance_gate,
            "adaptive_mean_metrics": adaptive_means,
            "baseline_mean_metrics": baseline_means,
        },
        "decision_gate": (
            "adaptive_measurement_rate_mappo_supported_across_seeds"
            if audit_ok and stability_supported
            else "adaptive_measurement_rate_mappo_not_yet_supported_across_seeds"
        ),
        "audit_ok": audit_ok,
        "stability_supported": stability_supported,
        "decision_supported": bool(audit_ok and stability_supported),
        "interpretation_boundary": (
            "This experiment tests whether a detached per-UAV adaptive multiplier stabilizes the "
            "quality-aware measurement policy under fixed queue overhead and train-fitted budgets. "
            "It does not establish final algorithmic superiority, collision safety, or scalability to 5/7 UAVs."
        ),
    }


def _plot(
    validation_rates: Sequence[Mapping[str, Any]],
    seed_means: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        BASELINE_BRANCH: "#2563eb",
        BUDGETED_BRANCH: "#dc2626",
        ADAPTIVE_BRANCH: "#16a34a",
    }
    labels = {
        BASELINE_BRANCH: "Baseline MAPPO",
        BUDGETED_BRANCH: "Budgeted quality-aware",
        ADAPTIVE_BRANCH: "Adaptive budgeted quality-aware",
    }
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for branch in BRANCHES:
        selected = sorted(
            [row for row in validation_rates if str(row.get("treatment")) == branch],
            key=lambda row: int(row["base_seed"]),
        )
        axes[0, 0].plot(
            [int(row["base_seed"]) for row in selected],
            [_finite(row.get("measurement_rate_mean"), 0.0) for row in selected],
            marker="o", label=labels[branch], color=colors[branch],
        )
        axes[0, 1].plot(
            [int(row["base_seed"]) for row in selected],
            [_finite(row.get("max_abs_deviation_from_baseline_train_target"), 0.0) for row in selected],
            marker="o", label=labels[branch], color=colors[branch],
        )
    axes[0, 0].set_title("Validation measurement rate")
    axes[0, 0].set_ylabel("Non-none rate")
    axes[0, 1].set_title("Maximum target deviation")
    axes[0, 1].set_ylabel("Absolute deviation")
    for axis in axes[0, :2]:
        axis.set_xlabel("Seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    for axis, metric, title in (
        (axes[0, 2], "measurement_rate_dual_uav_00", "Adaptive dual (UAV 00)"),
        (axes[1, 0], "position_rmse_m", "Validation position RMSE (m)"),
        (axes[1, 1], "j_com", "Validation communication cost"),
        (axes[1, 2], "mean_reward", "Validation team reward"),
    ):
        for branch in BRANCHES:
            selected = sorted(
                [row for row in seed_means if str(row.get("treatment")) == branch and str(row.get("split")) == "validation"],
                key=lambda row: int(row["base_seed"]),
            )
            if metric.startswith("measurement_rate_dual"):
                values = []
                for selected_row in selected:
                    if branch != ADAPTIVE_BRANCH:
                        values.append(0.0)
                        continue
                    values.append(_mean(
                        (
                            row.get(metric, 0.0)
                            for row in training_rows
                            if int(row.get("base_seed", -1)) == int(selected_row["base_seed"])
                            and str(row.get("treatment")) == ADAPTIVE_BRANCH
                        ),
                        default=0.0,
                    ))
            else:
                values = [_finite(row.get(metric), 0.0) for row in selected]
            axis.plot(
                [int(row["base_seed"]) for row in selected], values,
                marker="o", label=labels[branch], color=colors[branch],
            )
        axis.set_title(title)
        axis.set_xlabel("Seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / "adaptive_measurement_rate_mappo.png"
    figure.suptitle("Adaptive measurement-rate MAPPO stability (validation; test sealed)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def run_adaptive_budgeted_quality_aware_mappo(
    config: AdaptiveMeasurementRateMAPPOConfig,
) -> dict[str, Any]:
    """Run the Smoke or formal adaptive measurement-rate experiment."""

    root, logs_dir, plots_dir = _prepare_batch_output(config)
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
        targets_by_seed[int(seed)] = dict(run.get("measurement_rate_targets") or {})
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
    training_rows = _training_rows(root, config.seeds)
    budget_violations = _budget_violations(root, config.seeds)
    checkpoint_rows = _checkpoint_dual_audit(root, config.seeds)
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
        checkpoint_rows,
    )
    plot = _plot(validation_rates, seed_means, training_rows, plots_dir)

    _write_table(logs_dir / "measurement_rate_targets.csv", target_rows)
    _write_table(logs_dir / "training_rate_constraint.csv", training_rows)
    _write_table(logs_dir / "validation_policy_metrics.csv", [
        row for row in seed_means if str(row.get("split")) == "validation"
    ])
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "seed_comparison.csv", seed_means)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "cross_seed_summary.csv", cross_seed)
    _write_table(logs_dir / "validation_measurement_rates.csv", validation_rates)
    _write_table(logs_dir / "budget_violations.csv", budget_violations)
    _write_table(logs_dir / "checkpoint_dual_audit.csv", checkpoint_rows)
    _write_json(logs_dir / "audit_report.json", audit)
    manifest = {
        "dataset_id": "python_sim_adaptive_budgeted_quality_aware_mappo",
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
            BASELINE_BRANCH: "historical actor with communication overhead; no quality, budget, or rate control",
            BUDGETED_BRANCH: "quality-aware actor plus fixed train-derived budget",
            ADAPTIVE_BRANCH: "quality-aware actor plus fixed budget and detached per-UAV adaptive rate multipliers",
        },
        "measurement_rate_constraint": {
            "adaptive_branch": ADAPTIVE_BRANCH,
            "tolerance": float(config.measurement_rate_tolerance),
            "dual_initial": float(config.measurement_rate_dual_initial),
            "dual_step_size": float(config.measurement_rate_dual_step_size),
            "dual_max": float(config.measurement_rate_dual_max),
            "loss_applies_during": "training_updates_only",
            "multiplier_update": "after_train_trainer_update_only; detached; per_uav",
            "target_source": "baseline_mappo/train actions only",
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
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "seed_comparison": str(logs_dir / "seed_comparison.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "cross_seed_summary": str(logs_dir / "cross_seed_summary.csv"),
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
    "ADAPTIVE_BRANCH",
    "AdaptiveMeasurementRateMAPPOConfig",
    "BRANCHES",
    "DEFAULT_SEEDS",
    "run_adaptive_budgeted_quality_aware_mappo",
]
