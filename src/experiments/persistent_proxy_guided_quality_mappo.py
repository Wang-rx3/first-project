"""Persistent information-gain guided MAPPO experiment.

This module adds one narrow policy-side mechanism to the existing communication
and cooperative-localization experiments.  The persistent proxy forecasts the
*covariance benefit* of a candidate range update for a fixed number of prediction
steps.  It is fitted from baseline training observations only and is used as a
detached teacher signal for the measurement head.  The EKF, reward, GAT, radio,
queue and motion implementations are intentionally unchanged.

Validation is the pre-registered decision split.  Test episodes are generated
and sealed for later reporting, but never enter fitting, model selection or the
decision gate.  The independent statistical unit is the training seed.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .baseline import _read_csv, _read_json
from .communication_constrained_proxy_mappo import (
    fit_measurement_communication_cost_targets,
    _candidate_cost_map,
)
from .execution_uncertainty_proxy_audit import _candidate_rows_from_episode
from .localization_coupling_intervention import _bootstrap_interval
from .measurement_rate_controlled_mappo import _fit_measurement_budgets
from .formal import (
    FormalExperimentConfig,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)
from .proxy_guided_adaptive_budgeted_quality_aware_mappo import _train_proxy_model
from .quality_aware_mappo import (
    DEFAULT_EKF_CONFIG,
    DEFAULT_PROFILE,
    _episode_rows_for_branch,
    _finite,
    _read_jsonl,
    _write_table,
)
from ..learning.measurement_proxy import fit_persistent_proxy_model


DEFAULT_SEEDS = (17181, 18190, 19209, 20218, 21227)
BASELINE_BRANCH = "baseline_mappo"
PERSISTENT_BRANCH = "persistent_proxy_guided_quality_mappo"
BRANCHES = (BASELINE_BRANCH, PERSISTENT_BRANCH)

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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


def _timestamp(value: Any) -> float:
    try:
        return round(float(value), 9)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class PersistentProxyMAPPOConfig:
    """Configuration for the persistent-proxy two-branch experiment."""

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
    persistent_horizon: int = 3
    # The corrected teacher charges only communication cost above the
    # baseline-train target.  Keep this enabled by default for new persistent
    # proxy runs; the MAPPO/formal defaults remain false for old checkpoints.
    persistent_proxy_target_relative_cost: bool = True
    proxy_age_variance_coefficient: float = 1.0
    proxy_teacher_coef: float = 0.25
    proxy_temperature: float = 0.25
    communication_cost_weight: float = 1.0
    communication_cost_tolerance: float = 0.05
    communication_dual_initial: float = 0.0
    communication_dual_step_size: float = 0.25
    communication_dual_max: float = 5.0
    fixed_packet_cost_fraction: float = 0.5
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("persistent proxy MAPPO is currently fixed to 3 UAVs")
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
            raise ValueError("persistent proxy MAPPO currently supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("this experiment requires measurement_overhead_enabled=True")
        if isinstance(self.persistent_horizon, bool) or int(self.persistent_horizon) < 1:
            raise ValueError("persistent_horizon must be a positive integer")
        if not isinstance(self.persistent_proxy_target_relative_cost, bool):
            raise ValueError("persistent_proxy_target_relative_cost must be boolean")
        for name, value in (
            ("measurement_quality_logit_penalty", self.measurement_quality_logit_penalty),
            ("proxy_age_variance_coefficient", self.proxy_age_variance_coefficient),
            ("proxy_teacher_coef", self.proxy_teacher_coef),
            ("proxy_temperature", self.proxy_temperature),
            ("communication_cost_weight", self.communication_cost_weight),
            ("communication_cost_tolerance", self.communication_cost_tolerance),
            ("communication_dual_initial", self.communication_dual_initial),
            ("communication_dual_step_size", self.communication_dual_step_size),
            ("communication_dual_max", self.communication_dual_max),
            ("fixed_packet_cost_fraction", self.fixed_packet_cost_fraction),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.proxy_temperature <= 0.0:
            raise ValueError("proxy_temperature must be positive")
        if not 0.0 <= self.communication_cost_tolerance <= 1.0:
            raise ValueError("communication_cost_tolerance must be in [0, 1]")
        if not 0.0 <= self.fixed_packet_cost_fraction <= 1.0:
            raise ValueError("fixed_packet_cost_fraction must be in [0, 1]")
        if self.communication_dual_max <= 0.0:
            raise ValueError("communication_dual_max must be positive")
        if self.communication_dual_initial > self.communication_dual_max:
            raise ValueError("communication_dual_initial must not exceed communication_dual_max")
        if self.communication_dual_step_size <= 0.0:
            raise ValueError("communication_dual_step_size must be positive")


def _progress(config: PersistentProxyMAPPOConfig, message: str) -> None:
    if config.progress:
        print(f"[persistent-proxy-mappo] {message}", flush=True)


def _prepare_outputs(config: PersistentProxyMAPPOConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = (
        Path(config.logs_dir)
        if config.logs_dir is not None
        else Path("logs/preliminary") / root.name
    )
    plots = (
        Path(config.plots_dir)
        if config.plots_dir is not None
        else Path("plots/preliminary") / root.name
    )
    if config.force:
        for path in (root, logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (root, logs, plots)):
        raise FileExistsError("persistent proxy output is not empty; use --force")
    for path in (root, logs, plots):
        path.mkdir(parents=True, exist_ok=True)
    return root, logs, plots


def _formal_config(
    config: PersistentProxyMAPPOConfig,
    *,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
    branch: str,
    metric_scales: Mapping[str, float] | None = None,
    budgets: Mapping[str, int] | None = None,
    communication_targets: Mapping[str, float] | None = None,
    persistent_model: Mapping[str, Any] | None = None,
) -> FormalExperimentConfig:
    persistent = branch == PERSISTENT_BRANCH
    if persistent and (
        budgets is None or communication_targets is None or persistent_model is None
    ):
        raise ValueError("persistent branch requires budgets, communication targets and proxy model")
    return FormalExperimentConfig(
        output_dir=output_dir,
        uav_count=3,
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
        metric_scales=dict(metric_scales) if metric_scales is not None else None,
        metric_scale_source=(
            "pooled_training_warmup_p95" if metric_scales is not None else "training_warmup_p95"
        ),
        command_line=(
            "run-persistent-proxy-guided-quality-mappo "
            f"--seed {seed} --persistent-horizon {config.persistent_horizon} "
            f"--measurement-overhead-enabled --quality-logit-penalty "
            f"{config.measurement_quality_logit_penalty:g}"
        ),
        progress=config.progress,
        mappo_variant=branch,
        measurement_overhead_enabled=True,
        measurement_quality_logit_penalty=(
            float(config.measurement_quality_logit_penalty) if persistent else 0.0
        ),
        measurement_budget_enabled=persistent,
        measurement_budget_by_uav=dict(budgets) if persistent else None,
        measurement_rate_constraint_enabled=False,
        measurement_rate_target_by_uav=None,
        measurement_rate_dual_enabled=False,
        measurement_proxy_features_enabled=False,
        measurement_proxy_guidance_enabled=False,
        measurement_proxy_teacher_coef=float(config.proxy_teacher_coef),
        measurement_proxy_temperature=float(config.proxy_temperature),
        measurement_proxy_communication_cost_weight=float(config.communication_cost_weight),
        measurement_proxy_model=None,
        measurement_persistent_proxy_features_enabled=persistent,
        measurement_persistent_proxy_guidance_enabled=persistent,
        measurement_persistent_proxy_horizon=int(config.persistent_horizon),
    measurement_persistent_proxy_model=(
            dict(persistent_model) if persistent else None
        ),
        measurement_persistent_proxy_target_relative_cost=(
            bool(config.persistent_proxy_target_relative_cost) if persistent else False
        ),
        measurement_communication_constraint_enabled=persistent,
        measurement_communication_cost_target_by_uav=(
            dict(communication_targets) if persistent else None
        ),
        measurement_communication_cost_tolerance=float(config.communication_cost_tolerance),
        measurement_communication_dual_initial=float(config.communication_dual_initial),
        measurement_communication_dual_step_size=float(config.communication_dual_step_size),
        measurement_communication_dual_max=float(config.communication_dual_max),
        measurement_communication_fixed_packet_fraction=float(config.fixed_packet_cost_fraction),
    )


def _training_rows_for_persistent_model(
    baseline_scale: Path,
    seed: int,
    age_variance_coefficient: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_root = baseline_scale / "train"
    for episode_root in sorted(path for path in train_root.iterdir() if path.is_dir()):
        rows.extend(
            _candidate_rows_from_episode(
                episode_root,
                base_seed=seed,
                treatment=BASELINE_BRANCH,
                split="train",
                include_offline_label=False,
                age_variance_coefficient=age_variance_coefficient,
            )
        )
    return rows


def _fit_persistent_model(
    baseline_scale: Path,
    *,
    seed: int,
    config: PersistentProxyMAPPOConfig,
    ekf_parameters: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _training_rows_for_persistent_model(
        baseline_scale, seed, config.proxy_age_variance_coefficient
    )
    if not rows:
        raise RuntimeError(f"no baseline training rows for persistent proxy: {baseline_scale}")
    base_model = _train_proxy_model(
        baseline_scale, seed, config.proxy_age_variance_coefficient
    )
    model = fit_persistent_proxy_model(
        rows,
        base_model,
        horizon=int(config.persistent_horizon),
        decision_dt_s=0.2,
        process_position_std_m=float(ekf_parameters["process_position_std_m"]),
        process_velocity_std_mps=float(ekf_parameters["process_velocity_std_mps"]),
        age_variance_coefficient=float(config.proxy_age_variance_coefficient),
    )
    model.update(
        {
            "schema_version": "execution-persistent-proxy-v1",
            "source_split": "train",
            "source_branch": BASELINE_BRANCH,
            "fit_scope": "baseline_mappo/train actor-visible graph observations only",
            "candidate_row_count": len(rows),
            "validation_used_for_fitting": False,
            "test_used_for_fitting": False,
            "persistent_horizon": int(config.persistent_horizon),
            "forbidden_inputs": [
                "true_position",
                "true_error",
                "true_range",
                "physical_los_nlos",
                "future_state",
                "future_packet_outcome",
                "post_update_nis",
            ],
        }
    )
    return model, rows


def _branch_config_audit(seed_root: Path) -> tuple[bool, dict[str, Any]]:
    manifests = {
        branch: _read_json(seed_root / branch / "scale_3" / "manifest.json")
        for branch in BRANCHES
    }
    integration_summaries = {
        branch: _read_json(
            seed_root / branch / "scale_3" / "checks" / "integration_summary.json"
        )
        for branch in BRANCHES
    }
    baseline = dict(manifests[BASELINE_BRANCH].get("config", {}))
    ignored = {
        "mappo_variant",
        "output_dir",
        "measurement_quality_logit_penalty",
        "measurement_budget_enabled",
        "measurement_budget_by_uav",
        "measurement_persistent_proxy_features_enabled",
        "measurement_persistent_proxy_guidance_enabled",
        "measurement_persistent_proxy_horizon",
        "measurement_persistent_proxy_model",
        "measurement_persistent_proxy_target_relative_cost",
        "measurement_communication_constraint_enabled",
        "measurement_communication_cost_target_by_uav",
        "measurement_communication_cost_tolerance",
        "measurement_communication_dual_initial",
        "measurement_communication_dual_step_size",
        "measurement_communication_dual_max",
        "measurement_communication_fixed_packet_fraction",
    }
    for key in ignored:
        baseline.pop(key, None)
    equal = True
    details: dict[str, Any] = {}
    for branch, manifest in manifests.items():
        raw = dict(manifest.get("config", {}))
        summary = integration_summaries[branch]
        comparable = dict(raw)
        for key in ignored:
            comparable.pop(key, None)
        equal = equal and comparable == baseline
        details[branch] = {
            "mappo_variant": raw.get("mappo_variant"),
            "measurement_quality_enabled": bool(
                summary.get("measurement_quality", {}).get("enabled", False)
            ),
            "measurement_budget_enabled": bool(raw.get("measurement_budget_enabled", False)),
            "persistent_proxy_features_enabled": bool(
                summary.get("measurement_persistent_proxy", {}).get("features_enabled", False)
            ),
            "persistent_proxy_guidance_enabled": bool(
                summary.get("measurement_persistent_proxy", {}).get("guidance_enabled", False)
            ),
            "communication_constraint_enabled": bool(
                summary.get("measurement_communication_constraint", {}).get("enabled", False)
            ),
            "measurement_overhead_enabled": bool(raw.get("measurement_overhead_enabled", False)),
        }
    return equal, details


def _run_seed(
    config: PersistentProxyMAPPOConfig,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    ekf = load_selected_ekf_config(config.ekf_config_path)
    seed_root = root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    template = _formal_config(
        config,
        seed=seed,
        output_dir=seed_root / "metric_scale_probe",
        ekf_parameters=ekf,
        branch=BASELINE_BRANCH,
    )
    metric_scales = fit_shared_training_metric_scales(template, (config.uav_count,))
    _progress(config, f"seed {seed}: training {BASELINE_BRANCH}")
    baseline_formal = _formal_config(
        config,
        seed=seed,
        output_dir=seed_root / BASELINE_BRANCH,
        ekf_parameters=ekf,
        branch=BASELINE_BRANCH,
        metric_scales=metric_scales,
    )
    run_preliminary(baseline_formal)
    baseline_scale = seed_root / BASELINE_BRANCH / "scale_3"
    baseline_report = validate_episode_benchmark(baseline_scale)
    if not bool(baseline_report.get("ok", False)):
        raise RuntimeError(f"baseline episode validation failed: {baseline_report}")

    quality_model_path = baseline_scale / "calibration" / "measurement_quality_model.json"
    quality_model = _read_json(quality_model_path)
    budgets = _fit_measurement_budgets(baseline_scale, uav_count=3)
    persistent_model, model_rows = _fit_persistent_model(
        baseline_scale, seed=seed, config=config, ekf_parameters=ekf
    )
    communication_targets, communication_target_rows = fit_measurement_communication_cost_targets(
        baseline_scale,
        seed=seed,
        proxy_model=_train_proxy_model(
            baseline_scale, seed, config.proxy_age_variance_coefficient
        ),
        fixed_packet_cost_fraction=config.fixed_packet_cost_fraction,
    )

    _progress(config, f"seed {seed}: training {PERSISTENT_BRANCH}")
    persistent_formal = _formal_config(
        config,
        seed=seed,
        output_dir=seed_root / PERSISTENT_BRANCH,
        ekf_parameters=ekf,
        branch=PERSISTENT_BRANCH,
        metric_scales=metric_scales,
        budgets=budgets,
        communication_targets=communication_targets,
        persistent_model=persistent_model,
    )
    run_preliminary(
        persistent_formal,
        measurement_quality_model=quality_model,
        measurement_persistent_proxy_model=persistent_model,
    )
    persistent_scale = seed_root / PERSISTENT_BRANCH / "scale_3"
    persistent_report = validate_episode_benchmark(persistent_scale)
    if not bool(persistent_report.get("ok", False)):
        raise RuntimeError(f"persistent episode validation failed: {persistent_report}")
    configs_equal, config_details = _branch_config_audit(seed_root)
    return {
        "base_seed": int(seed),
        "seed_root": str(seed_root),
        "branches": {
            BASELINE_BRANCH: {
                "scale_root": str(baseline_scale),
                "validation_report_ok": bool(baseline_report.get("ok", False)),
                **config_details[BASELINE_BRANCH],
            },
            PERSISTENT_BRANCH: {
                "scale_root": str(persistent_scale),
                "validation_report_ok": bool(persistent_report.get("ok", False)),
                **config_details[PERSISTENT_BRANCH],
            },
        },
        "measurement_budgets": {str(key): int(value) for key, value in budgets.items()},
        "measurement_communication_targets": {
            str(key): float(value) for key, value in communication_targets.items()
        },
        "measurement_communication_target_rows": communication_target_rows,
        "persistent_proxy_model": _jsonable(persistent_model),
        "persistent_proxy_training_row_count": len(model_rows),
        "warmup_quality_model_path": str(quality_model_path),
        "branch_configs_equal_except_expected_controls": bool(configs_equal),
        "validation_reports": {
            BASELINE_BRANCH: baseline_report,
            PERSISTENT_BRANCH: persistent_report,
        },
        "test_data_used_for_selection": False,
    }


def _seed_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seeds = sorted({int(row["base_seed"]) for row in rows})
    for seed in seeds:
        for split in ("validation", "test"):
            for branch in BRANCHES:
                selected = [
                    row
                    for row in rows
                    if int(row["base_seed"]) == seed
                    and str(row.get("split")) == split
                    and str(row.get("treatment")) == branch
                ]
                if not selected:
                    raise ValueError(f"missing rows for {seed}/{split}/{branch}")
                output.append(
                    {
                        "base_seed": seed,
                        "split": split,
                        "treatment": branch,
                        "episode_count": len(selected),
                        **{metric: _mean(item.get(metric) for item in selected) for metric in METRICS},
                    }
                )
    return output


def _paired_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["base_seed"]), str(row.get("split")), str(row["episode_id"]))
        pairs.setdefault(key, {})[str(row["treatment"])] = row
    output: list[dict[str, Any]] = []
    for (seed, split, episode_id), pair in sorted(pairs.items()):
        if set(pair) != set(BRANCHES):
            raise ValueError(f"unpaired episode {seed}/{split}/{episode_id}: {set(pair)}")
        for metric in METRICS:
            baseline = _finite(pair[BASELINE_BRANCH].get(metric))
            persistent = _finite(pair[PERSISTENT_BRANCH].get(metric))
            difference = persistent - baseline
            if not np.isfinite(difference):
                raise ValueError(f"non-finite paired contrast for {metric}")
            output.append(
                {
                    "base_seed": seed,
                    "split": split,
                    "episode_id": episode_id,
                    "comparison": f"{PERSISTENT_BRANCH}_minus_{BASELINE_BRANCH}",
                    "metric": metric,
                    "baseline_value": baseline,
                    "treatment_value": persistent,
                    "paired_difference": difference,
                    "improvement_direction": (
                        "lower" if metric in LOWER_IS_BETTER else "higher"
                    ),
                }
            )
    return output


def _cross_seed(
    contrasts: Sequence[Mapping[str, Any]],
    config: PersistentProxyMAPPOConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for split in ("validation", "test"):
        for metric in METRICS:
            values_by_seed: dict[int, list[float]] = {}
            for row in contrasts:
                if str(row.get("split")) == split and str(row.get("metric")) == metric:
                    values_by_seed.setdefault(int(row["base_seed"]), []).append(
                        _finite(row.get("paired_difference"))
                    )
            if not values_by_seed:
                continue
            seed_values = np.asarray(
                [_mean(values_by_seed[seed]) for seed in sorted(values_by_seed)], dtype=float
            )
            if not np.isfinite(seed_values).all():
                raise ValueError(f"non-finite seed contrast for {split}/{metric}")
            low, high = _bootstrap_interval(seed_values, config.bootstrap_samples, rng)
            lower = metric in LOWER_IS_BETTER
            output.append(
                {
                    "split": split,
                    "comparison": f"{PERSISTENT_BRANCH}_minus_{BASELINE_BRANCH}",
                    "metric": metric,
                    "seed_count": int(seed_values.size),
                    "mean_across_seeds": float(np.mean(seed_values)),
                    "median_across_seeds": float(np.median(seed_values)),
                    "std_across_seeds": float(np.std(seed_values, ddof=1))
                    if seed_values.size > 1
                    else 0.0,
                    "bootstrap_95_ci_low": float(low),
                    "bootstrap_95_ci_high": float(high),
                    "improvement_direction": "lower" if lower else "higher",
                    "improved_seed_count": int(np.sum(seed_values < 0.0 if lower else seed_values > 0.0)),
                    "not_worsened_seed_count": int(np.sum(seed_values <= 0.0 if lower else seed_values >= 0.0)),
                    "ci_supports_improvement": bool(low > 0.0 if not lower else high < 0.0),
                    "ci_supports_non_worsening": bool(low >= 0.0 if not lower else high <= 0.0),
                }
            )
    return output


def _budget_violations(
    root: Path,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for seed in seeds:
        scale = root / f"seed_{seed}" / PERSISTENT_BRANCH / "scale_3"
        for split in ("train", "validation", "test"):
            split_root = scale / split
            for episode in sorted(path for path in split_root.iterdir() if path.is_dir()):
                scenario = _read_json(episode / "scenario.json")
                limits = {
                    str(key): int(float(value))
                    for key, value in (scenario.get("measurement_budget_limits", {}) or {}).items()
                }
                counts: dict[str, int] = {}
                for row in _read_csv(episode / "actions.csv"):
                    if str(row.get("measurement_uav_id", "none")) != "none":
                        uav = str(row.get("uav_id", ""))
                        counts[uav] = counts.get(uav, 0) + 1
                for uav, count in counts.items():
                    limit = limits.get(uav, 0)
                    if count > limit:
                        violations.append(
                            {
                                "base_seed": int(seed),
                                "split": split,
                                "episode_id": episode.name,
                                "uav_id": uav,
                                "count": int(count),
                                "limit": int(limit),
                            }
                        )
    return violations


def _dual_audit(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = (
            root / f"seed_{seed}" / PERSISTENT_BRANCH / "scale_3" / "checkpoints" / "final.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("measurement_communication_dual_state", {})
        valid = set(state) == {"uav_00", "uav_01", "uav_02"}
        row: dict[str, Any] = {"base_seed": int(seed), "checkpoint": str(checkpoint)}
        for index in range(3):
            value = _finite(state.get(f"uav_{index:02d}"), float("nan"))
            row[f"communication_dual_uav_{index:02d}"] = value
            valid = valid and np.isfinite(value) and 0.0 <= value <= 5.0
        row["dual_state_valid"] = bool(valid)
        rows.append(row)
    return rows


def _validation_communication_cost(
    root: Path,
    seeds: Sequence[int],
    targets: Mapping[int, Mapping[str, float]],
    models: Mapping[int, Mapping[str, Any]],
    fixed_fraction: float,
) -> list[dict[str, Any]]:
    """Recompute expected candidate cost from validation actions only."""

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        target = targets[int(seed)]
        for branch in BRANCHES:
            scale = root / f"seed_{seed}" / branch / "scale_3"
            cost_map = _candidate_cost_map(
                scale,
                seed,
                "validation",
                models[int(seed)],
                fixed_fraction,
            )
            per_uav: dict[str, list[float]] = {f"uav_{i:02d}": [] for i in range(3)}
            selected_count: dict[str, int] = {key: 0 for key in per_uav}
            for episode in sorted(path for path in (scale / "validation").iterdir() if path.is_dir()):
                for action in _read_csv(episode / "actions.csv"):
                    uav = str(action.get("uav_id", ""))
                    if uav not in per_uav:
                        continue
                    selected = str(action.get("measurement_uav_id", "none"))
                    cost = 0.0
                    if selected != "none":
                        selected_count[uav] += 1
                        key = (episode.name, _timestamp(action.get("timestamp")), uav, selected)
                        cost = _finite(cost_map.get(key), 1.0)
                    per_uav[uav].append(cost)
            means = {uav: _mean(values, 0.0) for uav, values in per_uav.items()}
            rows.append(
                {
                    "base_seed": int(seed),
                    "split": "validation",
                    "treatment": branch,
                    "selected_measurements_total": int(sum(selected_count.values())),
                    "expected_cost_mean": float(np.mean(list(means.values()))),
                    "target_cost_mean": float(np.mean(list(target.values()))) if target else 0.0,
                    "max_uav_cost_excess": float(
                        max(means[uav] - float(target.get(uav, 0.0)) for uav in means)
                    ),
                    **{f"expected_cost_{uav}": value for uav, value in means.items()},
                    **{f"target_cost_{uav}": float(target.get(uav, 0.0)) for uav in means},
                }
            )
    return rows


def _training_metrics(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES:
            path = (
                root / f"seed_{seed}" / branch / "scale_3" / "diagnostics" / "training_metrics.jsonl"
            )
            for item in _read_jsonl(path):
                row = {"base_seed": int(seed), "treatment": branch, **item}
                for key, value in list(row.items()):
                    if key not in {"base_seed", "treatment"} and isinstance(value, (int, float, str)):
                        parsed = _finite(value, float("nan"))
                        if np.isfinite(parsed):
                            row[key] = parsed
                rows.append(row)
    return rows


def _audit(
    config: PersistentProxyMAPPOConfig,
    runs: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seed_means: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
    communication_cost_rows: Sequence[Mapping[str, Any]],
    budget_violations: Sequence[Mapping[str, Any]],
    dual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = int(math.ceil(0.8 * len(config.seeds)))
    structural = (
        len(runs) == len(config.seeds)
        and all(
            bool(run.get("branches", {}).get(branch, {}).get("validation_report_ok", False))
            for run in runs
            for branch in BRANCHES
        )
        and all(
            int(row.get("episode_count", 0))
            == (config.validation_episodes if str(row.get("split")) == "validation" else config.test_episodes)
            for row in seed_means
        )
    )
    variants_ok = all(
        run.get("branches", {}).get(BASELINE_BRANCH, {}).get("mappo_variant") == BASELINE_BRANCH
        and run.get("branches", {}).get(PERSISTENT_BRANCH, {}).get("mappo_variant")
        == PERSISTENT_BRANCH
        for run in runs
    )
    controls_ok = all(
        not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_quality_enabled", True))
        and bool(run.get("branches", {}).get(PERSISTENT_BRANCH, {}).get("measurement_quality_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("persistent_proxy_features_enabled", True))
        and bool(run.get("branches", {}).get(PERSISTENT_BRANCH, {}).get("persistent_proxy_features_enabled", False))
        and bool(run.get("branches", {}).get(PERSISTENT_BRANCH, {}).get("communication_constraint_enabled", False))
        for run in runs
    )
    config_equal = all(
        bool(run.get("branch_configs_equal_except_expected_controls", False)) for run in runs
    )
    finite = all(
        np.isfinite(_finite(row.get(metric))) for row in episode_rows for metric in METRICS
    )
    dual_ok = len(dual_rows) == len(config.seeds) and all(
        bool(row.get("dual_state_valid", False)) for row in dual_rows
    )
    proxy_ok = all(
        isinstance(run.get("persistent_proxy_model"), Mapping)
        and run.get("persistent_proxy_model", {}).get("source_split") == "train"
        and not bool(run.get("persistent_proxy_model", {}).get("validation_used_for_fitting", True))
        and not bool(run.get("persistent_proxy_model", {}).get("test_used_for_fitting", True))
        for run in runs
    )
    overhead_ok = all(
        bool(run.get("branches", {}).get(branch, {}).get("measurement_overhead_enabled", False))
        for run in runs
        for branch in BRANCHES
    )
    cost_complete = len(communication_cost_rows) == len(config.seeds) * len(BRANCHES)
    cost_finite = cost_complete and all(
        np.isfinite(_finite(row.get("expected_cost_mean"), float("nan")))
        and np.isfinite(_finite(row.get("target_cost_mean"), float("nan")))
        for row in communication_cost_rows
    )
    cost_target_ok = cost_finite and all(
        _finite(row.get("max_uav_cost_excess"), float("inf")) <= 0.05 + 1e-9
        for row in communication_cost_rows
        if str(row.get("treatment")) == PERSISTENT_BRANCH
    )
    primary = {
        str(row["metric"]): row
        for row in cross_seed
        if str(row.get("split")) == "validation"
    }
    rmse = primary.get("position_rmse_m", {})
    harmful = primary.get("harmful_fraction", {})
    j_com = primary.get("j_com", {})
    reward = primary.get("mean_reward", {})
    performance = (
        int(rmse.get("improved_seed_count", 0)) >= required
        and bool(rmse.get("ci_supports_improvement", False))
        and int(harmful.get("not_worsened_seed_count", 0)) >= required
        and _finite(harmful.get("bootstrap_95_ci_high"), float("inf")) <= 0.02
        and _finite(j_com.get("bootstrap_95_ci_high"), float("inf")) <= 0.005
        and _finite(reward.get("bootstrap_95_ci_low"), float("-inf")) >= -0.005
    )
    audit_ok = bool(
        structural
        and variants_ok
        and controls_ok
        and config_equal
        and finite
        and dual_ok
        and proxy_ok
        and overhead_ok
        and cost_complete
        and cost_finite
        and not budget_violations
    )
    decision_supported = bool(audit_ok and cost_target_ok and performance)
    return {
        "experiment": "persistent_proxy_guided_quality_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evidence_status": "engineering_stability_validation_only",
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_used_for_selection": False,
        "test_data_generated_and_sealed": True,
        "fit_sources": {
            "persistent_proxy": "baseline_mappo/train actor-visible graph observations only",
            "measurement_budget": "baseline_mappo/train actions only",
            "communication_cost_target": "baseline_mappo/train actor-visible link quality only",
            "quality_model": "warmup actor-visible observations only",
        },
        "fixed_controls": {
            "measurement_overhead_enabled": True,
            "persistent_horizon": int(config.persistent_horizon),
            "quality_logit_penalty": float(config.measurement_quality_logit_penalty),
            "teacher_coef": float(config.proxy_teacher_coef),
            "temperature": float(config.proxy_temperature),
            "communication_cost_weight": float(config.communication_cost_weight),
            "communication_cost_tolerance": float(config.communication_cost_tolerance),
            "communication_dual_step_size": float(config.communication_dual_step_size),
            "communication_dual_max": float(config.communication_dual_max),
            "fixed_packet_cost_fraction": float(config.fixed_packet_cost_fraction),
        },
        "checks": {
            "seed_count": len(config.seeds),
            "branch_count": len(BRANCHES),
            "structural": structural,
            "variants_correct": variants_ok,
            "persistent_controls_correct": controls_ok,
            "branch_configs_equal_except_expected_controls": config_equal,
            "all_episode_metrics_finite": finite,
            "dual_state_finite": dual_ok,
            "persistent_proxy_fit_train_only": proxy_ok,
            "measurement_overhead_enabled": overhead_ok,
            "communication_cost_rows_complete": cost_complete,
            "communication_cost_rows_finite": cost_finite,
            "communication_cost_target_within_tolerance": cost_target_ok,
            "budget_violations": len(budget_violations),
            "rmse_improved_seed_count": int(rmse.get("improved_seed_count", 0)),
            "harmful_not_worsened_seed_count": int(harmful.get("not_worsened_seed_count", 0)),
            "required_seed_count": required,
            "primary_comparison": f"{PERSISTENT_BRANCH}_minus_{BASELINE_BRANCH}",
            "rmse_validation_ci": [
                _finite(rmse.get("bootstrap_95_ci_low"), float("nan")),
                _finite(rmse.get("bootstrap_95_ci_high"), float("nan")),
            ],
            "j_com_validation_ci": [
                _finite(j_com.get("bootstrap_95_ci_low"), float("nan")),
                _finite(j_com.get("bootstrap_95_ci_high"), float("nan")),
            ],
            "harmful_fraction_validation_ci": [
                _finite(harmful.get("bootstrap_95_ci_low"), float("nan")),
                _finite(harmful.get("bootstrap_95_ci_high"), float("nan")),
            ],
            "mean_reward_validation_ci": [
                _finite(reward.get("bootstrap_95_ci_low"), float("nan")),
                _finite(reward.get("bootstrap_95_ci_high"), float("nan")),
            ],
        },
        "decision_gate": (
            "persistent_proxy_guided_mappo_supported"
            if decision_supported
            else "persistent_proxy_guided_mappo_not_yet_supported"
        ),
        "audit_ok": audit_ok,
        "decision_supported": decision_supported,
        "interpretation_boundary": (
            "This experiment tests a fixed-horizon execution-visible proxy and a train-derived "
            "communication-cost constraint on the measurement head. It does not alter the EKF, "
            "reward, GAT, radio or queue models and does not establish final paper superiority."
        ),
    }


def _plot(
    seed_means: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {BASELINE_BRANCH: "#2563eb", PERSISTENT_BRANCH: "#dc2626"}
    labels = {BASELINE_BRANCH: "Baseline MAPPO", PERSISTENT_BRANCH: "Persistent proxy MAPPO"}
    metrics = (
        ("position_rmse_m", "Position RMSE (m)"),
        ("j_loc", "J_loc"),
        ("j_com", "J_com"),
        ("mean_reward", "Mean team reward"),
        ("measurement_count", "Measurements / episode"),
        ("harmful_fraction", "Harmful update fraction"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    validation = [row for row in seed_means if str(row.get("split")) == "validation"]
    for axis, (metric, title) in zip(axes.flat, metrics):
        values_by_branch: list[np.ndarray] = []
        for branch in BRANCHES:
            values = np.asarray(
                [
                    _finite(row.get(metric), float("nan"))
                    for row in validation
                    if str(row.get("treatment")) == branch
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            values_by_branch.append(values)
        positions = np.arange(len(BRANCHES), dtype=float)
        means = [float(np.mean(values)) if values.size else 0.0 for values in values_by_branch]
        stds = [
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            for values in values_by_branch
        ]
        axis.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=[colors[branch] for branch in BRANCHES],
        )
        for position, values in zip(positions, values_by_branch):
            if values.size:
                axis.scatter(
                    np.full(values.size, position),
                    values,
                    color="black",
                    s=18,
                    zorder=3,
                )
        axis.set_title(title)
        axis.set_xticks(positions, [labels[branch] for branch in BRANCHES], rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Persistent proxy MAPPO: validation seed-level diagnostics")
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / "persistent_proxy_guided_quality_mappo.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def run_persistent_proxy_guided_quality_mappo(
    config: PersistentProxyMAPPOConfig,
) -> dict[str, Any]:
    root, logs_dir, plots_dir = _prepare_outputs(config)
    if not Path(config.scenario_profile_path).is_file():
        raise FileNotFoundError(f"scenario profile not found: {config.scenario_profile_path}")
    if not Path(config.ekf_config_path).is_file():
        raise FileNotFoundError(f"EKF config not found: {config.ekf_config_path}")

    runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    models: dict[int, Mapping[str, Any]] = {}
    targets: dict[int, Mapping[str, float]] = {}
    for seed in config.seeds:
        run = _run_seed(config, int(seed), root)
        runs.append(run)
        models[int(seed)] = dict(run["persistent_proxy_model"])
        targets[int(seed)] = {
            str(key): float(value)
            for key, value in run["measurement_communication_targets"].items()
        }
        target_rows.extend(run["measurement_communication_target_rows"])
        for uav_id, value in run["measurement_budgets"].items():
            budget_rows.append(
                {
                    "base_seed": int(seed),
                    "uav_id": uav_id,
                    "measurement_budget": int(value),
                    "source_branch": BASELINE_BRANCH,
                    "source_split": "train",
                }
            )
        for branch in BRANCHES:
            episode_rows.extend(
                _episode_rows_for_branch(
                    root / f"seed_{int(seed)}" / branch / "scale_3",
                    branch,
                    int(seed),
                )
            )

    means = _seed_means(episode_rows)
    contrasts = _paired_contrasts(episode_rows)
    cross_seed = _cross_seed(contrasts, config)
    communication_cost_rows = _validation_communication_cost(
        root, config.seeds, targets, models, config.fixed_packet_cost_fraction
    )
    violations = _budget_violations(root, config.seeds)
    dual_rows = _dual_audit(root, config.seeds)
    training_rows = _training_metrics(root, config.seeds)
    audit = _audit(
        config,
        runs,
        episode_rows,
        means,
        cross_seed,
        communication_cost_rows,
        violations,
        dual_rows,
    )
    plot = _plot(means, cross_seed, plots_dir)

    _write_table(logs_dir / "measurement_budgets.csv", budget_rows)
    _write_table(logs_dir / "measurement_communication_cost_targets.csv", target_rows)
    _write_table(logs_dir / "validation_communication_cost.csv", communication_cost_rows)
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "seed_metrics.csv", means)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "cross_seed_summary.csv", cross_seed)
    _write_table(logs_dir / "training_metrics.csv", training_rows)
    _write_table(logs_dir / "budget_violations.csv", violations)
    _write_table(logs_dir / "communication_dual_audit.csv", dual_rows)
    _write_json(logs_dir / "persistent_proxy_parameters.json", _jsonable(models))
    _write_json(logs_dir / "audit_report.json", _jsonable(audit))
    manifest = {
        "dataset_id": "python_sim_persistent_proxy_guided_quality_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_stability_validation_only",
        "uav_count": config.uav_count,
        "seed_set": [int(seed) for seed in config.seeds],
        "branches": {
            BASELINE_BRANCH: "historical actor with measurement overhead",
            PERSISTENT_BRANCH: (
                "fixed-horizon execution-visible persistent covariance-benefit proxy, "
                "detached teacher and train-only communication-cost constraint"
            ),
        },
        "persistent_proxy": {
            "horizon": int(config.persistent_horizon),
            "utility": (
                "persistent_information_gain_norm - communication_cost_weight * "
                "max(0, proxy_cost_norm - baseline_train_target_cost)"
            ),
            "teacher_coefficient": float(config.proxy_teacher_coef),
            "temperature": float(config.proxy_temperature),
            "communication_cost_weight": float(config.communication_cost_weight),
            "fit_source": "baseline_mappo/train actor-visible graph observations only",
            "forbidden_inputs": [
                "true_position",
                "true_error",
                "true_range",
                "physical_los_nlos",
                "future_state",
                "future_packet_outcome",
                "post_update_nis",
            ],
        },
        "communication_constraint": {
            "enabled": True,
            "fit_source": "baseline_mappo/train only",
            "fixed_packet_cost_fraction": float(config.fixed_packet_cost_fraction),
            "equivalent_packet_size_bytes": 512,
            "dual_updates": "training updates only",
        },
        "independent_unit": "base training seed",
        "test_data_used_for_selection": False,
        "outputs": {
            "audit_report": str(logs_dir / "audit_report.json"),
            "comparison_plot": str(plot),
        },
        "audit": audit,
    }
    _write_json(root / "manifest.json", _jsonable(manifest))
    return {
        "output_dir": str(root),
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "seeds": [int(seed) for seed in config.seeds],
        "audit_ok": bool(audit["audit_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "audit_report": str(logs_dir / "audit_report.json"),
        "comparison_plot": str(plot),
    }


@dataclass(frozen=True)
class PersistentProxyAuditConfig:
    """Read-only proxy audit configuration."""

    reference_root: Path
    output_dir: Path
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    persistent_horizon: int = 3
    age_variance_coefficient: float = 1.0
    fixed_packet_cost_fraction: float = 0.5
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if isinstance(self.persistent_horizon, bool) or int(self.persistent_horizon) < 1:
            raise ValueError("persistent_horizon must be positive")
        if not math.isfinite(float(self.age_variance_coefficient)) or self.age_variance_coefficient < 0.0:
            raise ValueError("age_variance_coefficient must be finite and non-negative")
        if not 0.0 <= self.fixed_packet_cost_fraction <= 1.0:
            raise ValueError("fixed_packet_cost_fraction must be in [0, 1]")


def run_persistent_proxy_audit(config: PersistentProxyAuditConfig) -> dict[str, Any]:
    """Fit and audit the persistent proxy from train, then inspect validation."""

    logs = Path(config.output_dir)
    plots = Path(config.plots_dir) if config.plots_dir is not None else logs.parent / "plots" / logs.name
    if config.force:
        for path in (logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("persistent proxy audit output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    all_valid = True
    all_deterministic = True
    all_spread = True
    for seed in config.seeds:
        baseline_scale = Path(config.reference_root) / f"seed_{seed}" / BASELINE_BRANCH / "scale_3"
        if not baseline_scale.is_dir():
            raise FileNotFoundError(f"missing baseline scale: {baseline_scale}")
        # Train-only model fit.  No validation/test rows are traversed here.
        base_model = _train_proxy_model(
            baseline_scale, int(seed), config.age_variance_coefficient
        )
        train_rows = _training_rows_for_persistent_model(
            baseline_scale, int(seed), config.age_variance_coefficient
        )
        persistent_model = fit_persistent_proxy_model(
            train_rows,
            base_model,
            horizon=int(config.persistent_horizon),
            age_variance_coefficient=float(config.age_variance_coefficient),
        )
        persistent_model["source_split"] = "train"
        persistent_model["validation_used_for_fitting"] = False
        persistent_model["test_used_for_fitting"] = False
        models[str(seed)] = _jsonable(persistent_model)
        valid_count = 0
        total_count = 0
        multi_windows: dict[tuple[str, str], list[float]] = {}
        validation_root = baseline_scale / "validation"
        for episode in sorted(path for path in validation_root.iterdir() if path.is_dir()):
            raw_rows = _candidate_rows_from_episode(
                episode,
                base_seed=int(seed),
                treatment=BASELINE_BRANCH,
                split="validation",
                include_offline_label=True,
                age_variance_coefficient=config.age_variance_coefficient,
            )
            from ..learning.measurement_proxy import persistent_information_gain_from_row

            for row in raw_rows:
                raw, normalized, valid = persistent_information_gain_from_row(
                    row, persistent_model, horizon=int(config.persistent_horizon)
                )
                total_count += 1
                valid_count += int(bool(valid and np.isfinite(normalized)))
                key = (str(row["episode_id"]), str(row["timestamp"]))
                multi_windows.setdefault(key, []).append(float(normalized) if np.isfinite(normalized) else float("nan"))
                events.append(
                    {
                        "base_seed": int(seed),
                        "split": "validation",
                        "episode_id": row["episode_id"],
                        "timestamp": row["timestamp"],
                        "uav_id": row["uav_id"],
                        "source_uav_id": row["source_uav_id"],
                        "candidate_count": row["candidate_count"],
                        "persistent_horizon": int(config.persistent_horizon),
                        "persistent_gain_raw": raw,
                        "persistent_gain_norm": normalized,
                        "proxy_cost_norm": float(
                            np.clip(
                                1.0
                                - _finite(row.get("packet_loss_pred"), 1.0),
                                0.0,
                                1.0,
                            )
                        ),
                        "proxy_valid": int(bool(valid and np.isfinite(normalized))),
                        "offline_position_error_m": row.get("offline_position_error_m", float("nan")),
                    }
                )
        multi_candidates = [values for values in multi_windows.values() if len(values) >= 2]
        nonzero_spread = sum(
            bool(np.isfinite(values).all() and max(values) - min(values) > 1e-8)
            for values in multi_candidates
        )
        valid_rate = valid_count / total_count if total_count else 0.0
        spread_rate = nonzero_spread / len(multi_candidates) if multi_candidates else 0.0
        # Recompute the deterministic ordering twice from the same values.
        deterministic = all(
            sorted(range(len(values)), key=lambda index: (-values[index], index))
            == sorted(range(len(values)), key=lambda index: (-values[index], index))
            for values in multi_candidates
            if np.isfinite(values).all()
        )
        all_valid = all_valid and valid_rate >= 0.95
        all_spread = all_spread and (spread_rate >= 0.80 if multi_candidates else False)
        all_deterministic = all_deterministic and deterministic
        seed_summaries.append(
            {
                "base_seed": int(seed),
                "source_split": "train",
                "validation_candidate_count": int(total_count),
                "validation_proxy_valid_rate": float(valid_rate),
                "multi_candidate_window_count": len(multi_candidates),
                "nonzero_spread_window_count": int(nonzero_spread),
                "nonzero_spread_rate": float(spread_rate),
                "deterministic_ordering": bool(deterministic),
                "validation_used_for_fitting": False,
                "test_used_for_fitting": False,
            }
        )
    audit = {
        "experiment": "persistent_proxy_audit",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "fit_source": "baseline_mappo/train actor-visible graph observations only",
        "test_data_used": False,
        "checks": {
            "all_proxy_fields_finite": bool(all_valid),
            "multi_candidate_nonzero_spread": bool(all_spread),
            "deterministic_ordering": bool(all_deterministic),
            "validation_only_evaluation": True,
        },
        "audit_ok": bool(all_valid and all_spread and all_deterministic),
        "decision_supported": bool(all_valid and all_spread and all_deterministic),
        "decision_gate": (
            "persistent_proxy_ready_for_smoke"
            if all_valid and all_spread and all_deterministic
            else "persistent_proxy_requires_repair"
        ),
        "interpretation_boundary": (
            "This is a proxy-structure audit. Offline error labels are reported only for "
            "validation diagnostics and are never used to fit, rank or control actions."
        ),
    }
    _write_table(logs / "persistent_proxy_events.csv", events)
    _write_table(logs / "persistent_proxy_seed_summary.csv", seed_summaries)
    _write_json(logs / "persistent_proxy_parameters.json", models)
    _write_json(logs / "audit_report.json", audit)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    if events:
        valid_events = [row for row in events if int(row.get("proxy_valid", 0))]
        axes[0].scatter(
            [_finite(row.get("persistent_gain_norm"), 0.0) for row in valid_events],
            [_finite(row.get("proxy_cost_norm"), 0.0) for row in valid_events],
            s=8,
            alpha=0.35,
        )
        axes[1].hist(
            [_finite(row.get("persistent_gain_norm"), 0.0) for row in valid_events],
            bins=20,
        )
        axes[2].bar(
            [str(row["base_seed"]) for row in seed_summaries],
            [float(row["nonzero_spread_rate"]) for row in seed_summaries],
        )
    axes[0].set_title("Persistent gain vs cost")
    axes[1].set_title("Normalized persistent gain")
    axes[2].set_title("Nonzero candidate spread")
    for axis in axes:
        axis.grid(alpha=0.25)
    plot_path = plots / "persistent_proxy_audit.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return {
        "output_dir": str(logs),
        "plots_dir": str(plots),
        "audit_ok": bool(audit["audit_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "audit_report": str(logs / "audit_report.json"),
        "plot": str(plot_path),
    }


__all__ = [
    "BASELINE_BRANCH",
    "BRANCHES",
    "DEFAULT_SEEDS",
    "PERSISTENT_BRANCH",
    "PersistentProxyAuditConfig",
    "PersistentProxyMAPPOConfig",
    "run_persistent_proxy_audit",
    "run_persistent_proxy_guided_quality_mappo",
]
