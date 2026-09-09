"""Proxy-guided adaptive measurement MAPPO experiment.

This module adds one deliberately narrow treatment to the existing adaptive
measurement-rate experiment.  The new treatment uses only execution-time
visible information-gain and predicted communication-cost proxies in a
detached teacher distribution for the measurement head.  The EKF, reward,
GAT, channel, queue, velocity and bandwidth paths are unchanged.

The runner is intentionally validation-first: proxy scales and measurement
budgets are fitted from the same seed's baseline training split, while test
is generated and sealed and never enters the decision gate.
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
from .execution_uncertainty_proxy_audit import (
    _candidate_rows_from_episode,
    _fit_scales,
    _materialize_proxy_rows,
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
from .formal import (
    FormalExperimentConfig,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)


DEFAULT_SEEDS = (7071, 8080, 9099, 10118, 11127)
BASELINE_BRANCH = "baseline_mappo"
ADAPTIVE_BRANCH = "adaptive_budgeted_quality_aware_mappo"
PROXY_BRANCH = "proxy_guided_adaptive_budgeted_quality_aware_mappo"
BRANCHES = (BASELINE_BRANCH, ADAPTIVE_BRANCH, PROXY_BRANCH)
QUALITY_BRANCHES = {ADAPTIVE_BRANCH, PROXY_BRANCH}
SHARED_INITIALIZATION_MODULES = (
    "gat",
    "actor_body",
    "bandwidth_head",
    "delta_mean_head",
    "delta_logstd",
    "critic",
)
LOWER_IS_BETTER = {
    "position_rmse_m", "position_rmse_norm_m", "position_p95_norm_m",
    "position_error_p95_m", "j_loc", "j_com",
    "joint_objective", "harmful_fraction", "nlos_update_fraction",
    "stale_update_fraction", "packet_loss", "delay_s", "aoi_s",
    "collision_rate", "none_selection_rate", "measurement_count",
    "measurement_overhead_packets", "measurement_budget_usage_rate",
}
METRICS = (
    "mean_reward", "position_rmse_m", "position_rmse_norm_m", "position_p95_norm_m",
    "position_error_p95_m", "j_loc",
    "j_com", "joint_objective", "accepted_update_count", "beneficial_fraction",
    "update_coverage_rate", "update_coverage_any", "accepted_update_rate",
    "accepted_updates_per_window", "beneficial_per_window", "harmful_per_window",
    "no_update_episode_count", "j_loc_saturation_fraction",
    "harmful_fraction", "neutral_fraction", "nlos_update_fraction",
    "stale_update_fraction", "packet_loss", "delay_s", "aoi_s", "connectivity",
    "energy_proxy", "collision_rate", "none_selection_rate",
    "mean_selected_quality_risk", "quality_feature_enabled", "measurement_count",
    "measurement_overhead_packets", "measurement_budget_usage_rate", "update_coverage_rate",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


@dataclass(frozen=True)
class ProxyGuidedAdaptiveMAPPOConfig:
    """Configuration for the proxy-guided three-branch experiment."""

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
    # Optional warm-up-only VO covariance calibration.  All branches in a
    # paired run receive the same flag so the EKF observation model is not a
    # hidden branch difference.
    visual_odometry_quality_calibration_enabled: bool = False
    measurement_quality_logit_penalty: float = 1.0
    measurement_rate_tolerance: float = 0.05
    measurement_rate_dual_initial: float = 0.0
    measurement_rate_dual_step_size: float = 0.25
    measurement_rate_dual_max: float = 5.0
    proxy_age_variance_coefficient: float = 1.0
    proxy_teacher_coef: float = 0.25
    proxy_temperature: float = 0.25
    communication_cost_weight: float = 1.0
    force: bool = False
    progress: bool = True
    # Paired formal runs can isolate module-level RNG streams so the shared
    # GAT/actor/continuous-action/critic modules start identically across
    # branches with different measurement-head dimensions.
    branch_invariant_initialization: bool = False
    # Optional timing-aware measurement mask/guard propagated to the formal
    # environment.  Historical runs keep the legacy default.
    measurement_update_eligibility_enabled: bool = False

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("proxy-guided MAPPO is currently fixed to 3 UAVs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if min(self.max_steps, self.warmup_episodes, self.train_episodes,
               self.validation_episodes, self.test_episodes) < 1:
            raise ValueError("steps and all episode counts must be positive")
        if self.device != "cpu":
            raise ValueError("proxy-guided MAPPO currently supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("this experiment requires measurement_overhead_enabled=True")
        if not isinstance(self.visual_odometry_quality_calibration_enabled, bool):
            raise ValueError("visual_odometry_quality_calibration_enabled must be boolean")
        if not isinstance(self.branch_invariant_initialization, bool):
            raise ValueError("branch_invariant_initialization must be boolean")
        if not isinstance(self.measurement_update_eligibility_enabled, bool):
            raise ValueError("measurement_update_eligibility_enabled must be boolean")
        for name, value in (
            ("measurement_quality_logit_penalty", self.measurement_quality_logit_penalty),
            ("measurement_rate_tolerance", self.measurement_rate_tolerance),
            ("measurement_rate_dual_initial", self.measurement_rate_dual_initial),
            ("measurement_rate_dual_step_size", self.measurement_rate_dual_step_size),
            ("measurement_rate_dual_max", self.measurement_rate_dual_max),
            ("proxy_age_variance_coefficient", self.proxy_age_variance_coefficient),
            ("proxy_teacher_coef", self.proxy_teacher_coef),
            ("proxy_temperature", self.proxy_temperature),
            ("communication_cost_weight", self.communication_cost_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.proxy_temperature <= 0.0:
            raise ValueError("proxy_temperature must be positive")
        if not 0.0 <= self.measurement_rate_tolerance <= 1.0:
            raise ValueError("measurement_rate_tolerance must be in [0, 1]")
        if self.measurement_rate_dual_initial > self.measurement_rate_dual_max:
            raise ValueError("dual initial must not exceed dual max")
        if self.measurement_rate_dual_step_size <= 0.0 or self.measurement_rate_dual_max <= 0.0:
            raise ValueError("dual step size and dual max must be positive")


def _progress(config: ProxyGuidedAdaptiveMAPPOConfig, message: str) -> None:
    if config.progress:
        print(f"[proxy-guided-mappo] {message}", flush=True)


def _prepare_outputs(config: ProxyGuidedAdaptiveMAPPOConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / root.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / root.name
    targets = (root, logs, plots)
    if config.force:
        for path in targets:
            if path.exists():
                shutil.rmtree(path)
    else:
        occupied = [str(path) for path in targets if path.exists() and any(path.iterdir())]
        if occupied:
            raise FileExistsError("proxy-guided output is not empty; use --force: " + ", ".join(occupied))
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
    return targets


def _formal_config(
    config: ProxyGuidedAdaptiveMAPPOConfig,
    *,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
    treatment: str,
    budgets: Mapping[str, int] | None = None,
    targets: Mapping[str, float] | None = None,
    proxy_model: Mapping[str, Any] | None = None,
    metric_scales: Mapping[str, float] | None = None,
) -> FormalExperimentConfig:
    quality = treatment in QUALITY_BRANCHES
    adaptive = treatment in {ADAPTIVE_BRANCH, PROXY_BRANCH}
    proxy = treatment == PROXY_BRANCH
    return FormalExperimentConfig(
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
        metric_scales=dict(metric_scales) if metric_scales is not None else None,
        metric_scale_source="pooled_training_warmup_p95" if metric_scales is not None else "training_warmup_p95",
        command_line=(
            "run-proxy-guided-adaptive-quality-mappo "
            f"--seed {seed} --steps {config.max_steps} "
            f"--warmup-episodes {config.warmup_episodes} "
            f"--train-episodes {config.train_episodes} "
            f"--validation-episodes {config.validation_episodes} "
            f"--test-episodes {config.test_episodes} --measurement-overhead-enabled "
            f"--quality-logit-penalty {config.measurement_quality_logit_penalty:g} "
            f"--proxy-teacher-coef {config.proxy_teacher_coef:g} "
            f"--proxy-temperature {config.proxy_temperature:g} "
            f"--communication-cost-weight {config.communication_cost_weight:g}"
            + (
                " --branch-invariant-initialization"
                if config.branch_invariant_initialization
                else ""
            )
        ),
        progress=config.progress,
        branch_invariant_initialization=bool(config.branch_invariant_initialization),
        measurement_update_eligibility_enabled=bool(
            config.measurement_update_eligibility_enabled
        ),
        mappo_variant=treatment,
        measurement_overhead_enabled=True,
        visual_odometry_quality_calibration_enabled=(
            bool(config.visual_odometry_quality_calibration_enabled)
        ),
        measurement_quality_logit_penalty=float(config.measurement_quality_logit_penalty) if quality else 0.0,
        measurement_budget_enabled=quality,
        measurement_budget_by_uav=dict(budgets) if quality and budgets is not None else None,
        measurement_rate_constraint_enabled=adaptive,
        measurement_rate_constraint_coef=0.0,
        measurement_rate_tolerance=float(config.measurement_rate_tolerance),
        measurement_rate_target_by_uav=dict(targets) if adaptive and targets is not None else None,
        measurement_rate_dual_enabled=adaptive,
        measurement_rate_dual_initial=float(config.measurement_rate_dual_initial),
        measurement_rate_dual_step_size=float(config.measurement_rate_dual_step_size),
        measurement_rate_dual_max=float(config.measurement_rate_dual_max),
        measurement_proxy_features_enabled=proxy,
        measurement_proxy_guidance_enabled=proxy,
        measurement_proxy_teacher_coef=float(config.proxy_teacher_coef),
        measurement_proxy_temperature=float(config.proxy_temperature),
        measurement_proxy_communication_cost_weight=float(config.communication_cost_weight),
        measurement_proxy_model=dict(proxy_model) if proxy and proxy_model is not None else None,
    )


def _train_proxy_model(baseline_scale: Path, seed: int, age_coefficient: float) -> dict[str, Any]:
    train_root = baseline_scale / "train"
    episode_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for episode_root in episode_dirs:
        rows.extend(_candidate_rows_from_episode(
            episode_root,
            base_seed=seed,
            treatment=BASELINE_BRANCH,
            split="train",
            include_offline_label=False,
            age_variance_coefficient=age_coefficient,
        ))
    if not rows:
        raise RuntimeError(f"no baseline training graph rows available for proxy fitting: {train_root}")
    scales = _fit_scales(rows, age_coefficient)
    return {
        "schema_version": "execution-proxy-v1",
        "source_split": "train",
        "source_branch": BASELINE_BRANCH,
        "fit_scope": "baseline_mappo/train actor-visible graph observations only",
        "candidate_row_count": len(rows),
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "age_variance_coefficient": float(age_coefficient),
        "scales": scales.as_dict(),
        "forbidden_inputs": [
            "true_position", "true_error", "true_range", "physical_los_nlos",
            "future_state", "future_packet_outcome", "post_update_nis",
        ],
    }


def _branch_config_audit(seed_root: Path) -> tuple[bool, dict[str, Any]]:
    baseline_manifest = _read_json(seed_root / BASELINE_BRANCH / "scale_3" / "manifest.json")
    baseline = dict(baseline_manifest.get("config", {}))
    ignored = {
        "mappo_variant", "output_dir", "measurement_quality_logit_penalty",
        "measurement_budget_enabled", "measurement_budget_by_uav",
        "measurement_rate_constraint_enabled", "measurement_rate_constraint_coef",
        "measurement_rate_target_by_uav", "measurement_rate_dual_enabled",
        "measurement_rate_dual_initial", "measurement_rate_dual_step_size",
        "measurement_rate_dual_max", "measurement_proxy_features_enabled",
        "measurement_proxy_guidance_enabled", "measurement_proxy_teacher_coef",
        "measurement_proxy_temperature", "measurement_proxy_communication_cost_weight",
        "measurement_proxy_model",
    }
    for key in ignored:
        baseline.pop(key, None)
    equal = True
    details: dict[str, Any] = {}
    initial_payloads: dict[str, Mapping[str, Any]] = {}
    for branch in BRANCHES:
        manifest = _read_json(seed_root / branch / "scale_3" / "manifest.json")
        raw = dict(manifest.get("config", {}))
        comparable = dict(raw)
        for key in ignored:
            comparable.pop(key, None)
        equal = equal and comparable == baseline
        details[branch] = {
            "mappo_variant": raw.get("mappo_variant"),
            "quality_features_enabled": bool(manifest.get("measurement_quality", {}).get("enabled", raw.get("mappo_variant") in QUALITY_BRANCHES)),
            "measurement_budget_enabled": bool(raw.get("measurement_budget_enabled", False)),
            "measurement_rate_constraint_enabled": bool(raw.get("measurement_rate_constraint_enabled", False)),
            "measurement_rate_dual_enabled": bool(raw.get("measurement_rate_dual_enabled", False)),
            "measurement_proxy_features_enabled": bool(raw.get("measurement_proxy_features_enabled", False)),
            "measurement_proxy_guidance_enabled": bool(raw.get("measurement_proxy_guidance_enabled", False)),
            "measurement_overhead_enabled": bool(raw.get("measurement_overhead_enabled", False)),
        }
        hash_path = seed_root / branch / "scale_3" / "diagnostics" / "initial_parameter_hashes.json"
        hash_payload = _read_json(hash_path) if hash_path.is_file() else {}
        modules = hash_payload.get("modules", {}) if isinstance(hash_payload, Mapping) else {}
        if not isinstance(modules, Mapping):
            modules = {}
        initial_payloads[branch] = hash_payload if isinstance(hash_payload, Mapping) else {}
        details[branch]["parameter_initialization"] = {
            "available": bool(set(SHARED_INITIALIZATION_MODULES).issubset(modules)),
            "branch_invariant_initialization": bool(
                hash_payload.get("branch_invariant_initialization", False)
            ),
            "parameter_initialization_seed": hash_payload.get(
                "parameter_initialization_seed"
            ),
            "module_hashes": {str(key): str(value) for key, value in modules.items()},
        }
    branch_invariant = all(
        bool(initial_payloads.get(branch, {}).get("branch_invariant_initialization", False))
        for branch in BRANCHES
    )
    shared_initialization_equal = bool(initial_payloads) and all(
        bool(details.get(branch, {}).get("parameter_initialization", {}).get("available", False))
        for branch in BRANCHES
    )
    if shared_initialization_equal:
        reference = initial_payloads[BASELINE_BRANCH].get("modules", {})
        shared_initialization_equal = all(
            all(
                initial_payloads[branch].get("modules", {}).get(module)
                == reference.get(module)
                for module in SHARED_INITIALIZATION_MODULES
            )
            for branch in BRANCHES[1:]
        )
    details["shared_initialization_equal"] = bool(shared_initialization_equal)
    details["branch_invariant_initialization"] = bool(branch_invariant)
    equal = equal and (not branch_invariant or shared_initialization_equal)
    return equal, details


def _run_seed(config: ProxyGuidedAdaptiveMAPPOConfig, seed: int, root: Path) -> dict[str, Any]:
    ekf = load_selected_ekf_config(config.ekf_config_path)
    seed_root = root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    template = _formal_config(config, seed=seed, output_dir=seed_root / "metric_scale_probe", ekf_parameters=ekf, treatment=BASELINE_BRANCH)
    metric_scales = fit_shared_training_metric_scales(template, (config.uav_count,))
    summaries: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    quality_models: dict[str, Mapping[str, Any]] = {}
    shared_quality: Mapping[str, Any] | None = None
    budgets: dict[str, int] | None = None
    targets: dict[str, float] | None = None
    target_rows: list[dict[str, Any]] = []
    proxy_model: Mapping[str, Any] | None = None
    for branch in BRANCHES:
        if branch == BASELINE_BRANCH:
            branch_budgets = None
            branch_targets = None
            branch_proxy = None
        else:
            if budgets is None or targets is None or shared_quality is None:
                raise RuntimeError("baseline train artifacts must be fitted before quality branches")
            branch_budgets = budgets
            branch_targets = targets
            if branch == PROXY_BRANCH and proxy_model is None:
                proxy_model = _train_proxy_model(seed_root / BASELINE_BRANCH / "scale_3", seed, config.proxy_age_variance_coefficient)
            branch_proxy = proxy_model if branch == PROXY_BRANCH else None
        _progress(config, f"seed {seed}: training {branch}")
        formal = _formal_config(
            config, seed=seed, output_dir=seed_root / branch, ekf_parameters=ekf,
            treatment=branch, budgets=branch_budgets, targets=branch_targets,
            proxy_model=branch_proxy, metric_scales=metric_scales,
        )
        summary = run_preliminary(formal, measurement_quality_model=shared_quality, measurement_proxy_model=branch_proxy)
        scale_root = seed_root / branch / "scale_3"
        report = validate_episode_benchmark(scale_root)
        if not bool(report.get("ok", False)):
            raise RuntimeError(f"episode validation failed for seed {seed}/{branch}: {report}")
        quality_path = scale_root / "calibration" / "measurement_quality_model.json"
        model = _read_json(quality_path) if quality_path.exists() else {}
        if not model:
            raise RuntimeError(f"missing warm-up quality model for {seed}/{branch}")
        summaries[branch] = summary
        reports[branch] = report
        quality_models[branch] = model
        if branch == BASELINE_BRANCH:
            shared_quality = model
            budgets = _fit_measurement_budgets(scale_root, uav_count=config.uav_count)
            targets, target_rows = _fit_measurement_rate_targets(scale_root, uav_count=config.uav_count, base_seed=seed)
    configs_equal, config_details = _branch_config_audit(seed_root)
    base_model = json.dumps(quality_models[BASELINE_BRANCH], sort_keys=True)
    quality_equal = all(base_model == json.dumps(quality_models[branch], sort_keys=True) for branch in BRANCHES)
    return {
        "base_seed": int(seed),
        "seed_root": str(seed_root),
        "branches": {
            branch: {
                "scale_root": str(seed_root / branch / "scale_3"),
                "validation_report_ok": bool(reports[branch].get("ok", False)),
                **config_details[branch],
            }
            for branch in BRANCHES
        },
        "measurement_budgets": dict(budgets or {}),
        "measurement_rate_targets": dict(targets or {}),
        "measurement_rate_target_rows": target_rows,
        "proxy_model": _jsonable(proxy_model),
        "warmup_quality_models_identical": quality_equal,
        "branch_configs_equal_except_expected_controls": configs_equal,
        "validation_reports": reports,
        "test_data_used": False,
    }


def _seed_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in rows}):
        for split in ("validation", "test"):
            for branch in BRANCHES:
                selected = [row for row in rows if int(row["base_seed"]) == seed and str(row["split"]) == split and str(row["treatment"]) == branch]
                if not selected:
                    raise ValueError(f"missing rows for {seed}/{split}/{branch}")
                output.append({
                    "base_seed": seed, "split": split, "treatment": branch,
                    "episode_count": len(selected),
                    **{metric: _mean([item.get(metric) for item in selected]) for metric in METRICS},
                })
    return output


def _paired_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault((int(row["base_seed"]), str(row["split"]), str(row["episode_id"])), {})[str(row["treatment"])] = row
    output: list[dict[str, Any]] = []
    for (seed, split, episode), pair in sorted(pairs.items()):
        if set(pair) != set(BRANCHES):
            raise ValueError(f"unpaired episode {seed}/{split}/{episode}: {set(pair)}")
        for branch in BRANCHES[1:]:
            for metric in METRICS:
                baseline = _finite(pair[BASELINE_BRANCH].get(metric))
                treatment = _finite(pair[branch].get(metric))
                difference = treatment - baseline
                if not np.isfinite(difference):
                    raise ValueError(f"non-finite contrast for {metric}")
                output.append({
                    "base_seed": seed, "split": split, "episode_id": episode,
                    "comparison": f"{branch}_minus_{BASELINE_BRANCH}",
                    "metric": metric, "baseline_value": baseline,
                    "treatment_value": treatment, "paired_difference": difference,
                    "improvement_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                })
    return output


def _cross_seed(contrasts: Sequence[Mapping[str, Any]], config: ProxyGuidedAdaptiveMAPPOConfig) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for split in ("validation", "test"):
        for comparison in sorted({str(row["comparison"]) for row in contrasts if str(row["split"]) == split}):
            for metric in METRICS:
                values: dict[int, list[float]] = {}
                for row in contrasts:
                    if str(row["split"]) == split and str(row["comparison"]) == comparison and str(row["metric"]) == metric:
                        values.setdefault(int(row["base_seed"]), []).append(_finite(row["paired_difference"]))
                seed_values = np.asarray([_mean(values[seed]) for seed in sorted(values)], dtype=float)
                if seed_values.size == 0 or not np.isfinite(seed_values).all():
                    continue
                low, high = _bootstrap_interval(seed_values, config.bootstrap_samples, rng)
                lower = metric in LOWER_IS_BETTER
                output.append({
                    "split": split, "comparison": comparison, "metric": metric,
                    "seed_count": int(seed_values.size),
                    "mean_across_seeds": float(np.mean(seed_values)),
                    "std_across_seeds": float(np.std(seed_values, ddof=1)) if seed_values.size > 1 else 0.0,
                    "bootstrap_95_ci_low": float(low), "bootstrap_95_ci_high": float(high),
                    "improvement_direction": "lower" if lower else "higher",
                    "improved_seed_count": int(np.sum(seed_values < 0.0 if lower else seed_values > 0.0)),
                    "not_worsened_seed_count": int(np.sum(seed_values <= 0.0 if lower else seed_values >= 0.0)),
                    "ci_supports_improvement": bool(high < 0.0 if lower else low > 0.0),
                    "ci_supports_non_worsening": bool(high <= 0.0 if lower else low >= 0.0),
                })
    return output


def _validation_rates(root: Path, seeds: Sequence[int], targets: Mapping[int, Mapping[str, float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in BRANCHES:
            counts = {f"uav_{i:02d}": [0, 0] for i in range(3)}
            expected: dict[str, list[float]] = {uav: [] for uav in counts}
            split_root = root / f"seed_{seed}" / branch / "scale_3" / "validation"
            for episode_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                for row in _read_csv(episode_root / "actions.csv"):
                    uav = str(row.get("uav_id", ""))
                    if uav not in counts:
                        continue
                    counts[uav][1] += 1
                    if str(row.get("measurement_uav_id", "none")) != "none":
                        counts[uav][0] += 1
                for row in _read_jsonl(root / f"seed_{seed}" / branch / "scale_3" / "diagnostics" / "measurement_quality.jsonl"):
                    if str(row.get("split")) == "validation" and str(row.get("episode_id")) == episode_root.name:
                        uav = str(row.get("uav_id", ""))
                        value = _finite(row.get("measurement_rate_expected_non_none_probability"), float("nan"))
                        if uav in expected and np.isfinite(value):
                            expected[uav].append(value)
            rates = {uav: float(value[0] / value[1]) if value[1] else 0.0 for uav, value in counts.items()}
            target = targets[int(seed)]
            deviations = [abs(rates[uav] - target[uav]) for uav in rates]
            output.append({
                "base_seed": int(seed), "treatment": branch, "split": "validation",
                "measurement_rate_mean": float(np.mean(list(rates.values()))),
                "measurement_rate_std_across_uav": float(np.std(list(rates.values()))),
                "expected_non_none_rate_mean": _mean((v for values in expected.values() for v in values), 0.0),
                "mean_abs_deviation_from_baseline_train_target": float(np.mean(deviations)),
                "max_abs_deviation_from_baseline_train_target": float(np.max(deviations)),
                "uav_rates": json.dumps(rates, sort_keys=True),
                "target_rates": json.dumps(target, sort_keys=True),
            })
    return output


def _training_rows(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    defaults = {
        "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
        "clip_fraction": 0.0, "gradient_norm": 0.0,
        "measurement_rate_constraint_loss": 0.0, "measurement_rate_dual_loss": 0.0,
        "measurement_proxy_guidance_enabled": 0.0, "measurement_proxy_teacher_loss": 0.0,
        "measurement_proxy_teacher_weighted_loss": 0.0,
        "measurement_proxy_teacher_valid_fraction": 0.0,
        "measurement_proxy_teacher_agreement": 0.0,
    }
    for seed in seeds:
        for branch in BRANCHES:
            path = root / f"seed_{seed}" / branch / "scale_3" / "diagnostics" / "training_metrics.jsonl"
            for source in _read_jsonl(path):
                row = {"base_seed": int(seed), "treatment": branch, **source}
                for key, default in defaults.items():
                    row[key] = _finite(row.get(key), default)
                for i in range(3):
                    for prefix in ("measurement_rate_dual", "measurement_rate_dual_update", "measurement_rate_violation"):
                        key = f"{prefix}_uav_{i:02d}"
                        row[key] = _finite(row.get(key), 0.0)
                output.append(row)
    return output


def _budget_violations(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in (ADAPTIVE_BRANCH, PROXY_BRANCH):
            branch_root = root / f"seed_{seed}" / branch / "scale_3"
            for split in ("train", "validation", "test"):
                split_root = branch_root / split
                for episode_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                    scenario = _read_json(episode_root / "scenario.json")
                    limits = {str(k): int(_finite(v, 0.0)) for k, v in (scenario.get("measurement_budget_limits", {}) or {}).items()}
                    counts: dict[str, int] = {}
                    for row in _read_csv(episode_root / "actions.csv"):
                        if str(row.get("measurement_uav_id", "none")) != "none":
                            uav = str(row.get("uav_id", ""))
                            counts[uav] = counts.get(uav, 0) + 1
                    for uav, count in counts.items():
                        if count > limits.get(uav, 0):
                            output.append({"base_seed": seed, "branch": branch, "split": split, "episode_id": episode_root.name, "uav_id": uav, "count": count, "limit": limits.get(uav, 0)})
    return output


def _checkpoint_duals(root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    import torch
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in (ADAPTIVE_BRANCH, PROXY_BRANCH):
            path = root / f"seed_{seed}" / branch / "scale_3" / "checkpoints" / "final.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("measurement_rate_dual_state", {})
            valid = set(state) == {"uav_00", "uav_01", "uav_02"}
            values: dict[str, Any] = {}
            for i in range(3):
                value = _finite(state.get(f"uav_{i:02d}"), float("nan"))
                values[f"uav_{i:02d}"] = value
                valid = valid and np.isfinite(value) and 0.0 <= value <= 5.0
            output.append({"base_seed": seed, "branch": branch, "checkpoint": str(path), "dual_state_valid": bool(valid), **values})
    return output


def _audit(
    config: ProxyGuidedAdaptiveMAPPOConfig,
    runs: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    means: Sequence[Mapping[str, Any]],
    cross: Sequence[Mapping[str, Any]],
    rates: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    violations: Sequence[Mapping[str, Any]],
    duals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = int(math.ceil(0.8 * len(config.seeds)))
    structural = (
        len(runs) == len(config.seeds)
        and all(bool(run.get("branches", {}).get(branch, {}).get("validation_report_ok", False)) for run in runs for branch in BRANCHES)
        and all(int(row.get("episode_count", 0)) == (config.validation_episodes if str(row.get("split")) == "validation" else config.test_episodes) for row in means)
    )
    variants = all(run.get("branches", {}).get(branch, {}).get("mappo_variant", branch) == branch for run in runs for branch in BRANCHES)
    controls = all(
        not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("quality_features_enabled", True))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("quality_features_enabled", False))
        and bool(run.get("branches", {}).get(PROXY_BRANCH, {}).get("quality_features_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_budget_enabled", True))
        and bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("measurement_budget_enabled", False))
        and bool(run.get("branches", {}).get(PROXY_BRANCH, {}).get("measurement_budget_enabled", False))
        and not bool(run.get("branches", {}).get(BASELINE_BRANCH, {}).get("measurement_proxy_features_enabled", True))
        and not bool(run.get("branches", {}).get(ADAPTIVE_BRANCH, {}).get("measurement_proxy_features_enabled", True))
        and bool(run.get("branches", {}).get(PROXY_BRANCH, {}).get("measurement_proxy_features_enabled", False))
        for run in runs
    )
    targets_ok = len(target_rows) == len(config.seeds) * config.uav_count and all(str(row.get("source_split")) == "train" and str(row.get("source_branch")) == BASELINE_BRANCH and 0.0 <= _finite(row.get("target_measurement_rate"), -1.0) <= 1.0 for row in target_rows)
    finite = all(np.isfinite(_finite(row.get(metric))) for row in episode_rows for metric in METRICS)
    train_finite = all(np.isfinite(_finite(row.get("loss"), 0.0)) and np.isfinite(_finite(row.get("measurement_proxy_teacher_loss"), 0.0)) for row in training_rows)
    dual_ok = all(bool(row.get("dual_state_valid", False)) for row in duals)
    proxy_ok = all(isinstance(run.get("proxy_model"), Mapping) and run.get("proxy_model", {}).get("source_split") == "train" and not bool(run.get("proxy_model", {}).get("validation_used_for_fitting", True)) and not bool(run.get("proxy_model", {}).get("test_used_for_fitting", True)) for run in runs)
    rates_complete = len(rates) == len(config.seeds) * len(BRANCHES)
    proxy_rate_rows = [row for row in rates if str(row.get("treatment")) == PROXY_BRANCH]
    within = sum(_finite(row.get("max_abs_deviation_from_baseline_train_target"), float("inf")) <= 0.10 for row in proxy_rate_rows)
    comparison = f"{PROXY_BRANCH}_minus_{BASELINE_BRANCH}"
    v = {str(row["metric"]): row for row in cross if str(row.get("split")) == "validation" and str(row.get("comparison")) == comparison}
    rmse, harmful, jcom, reward = (
        v.get(name, {})
        for name in ("position_rmse_norm_m", "harmful_fraction", "j_com", "mean_reward")
    )
    performance = bool(
        int(rmse.get("not_worsened_seed_count", 0)) >= required
        and int(harmful.get("not_worsened_seed_count", 0)) >= required
        and bool(rmse.get("ci_supports_non_worsening", False))
        and bool(harmful.get("ci_supports_non_worsening", False))
        and bool(jcom.get("ci_supports_non_worsening", False))
        and bool(reward.get("ci_supports_non_worsening", False))
    )
    audit_ok = bool(structural and variants and controls and targets_ok and finite and train_finite and dual_ok and proxy_ok and rates_complete and not violations and all(bool(run.get("warmup_quality_models_identical", False)) for run in runs) and all(bool(run.get("branch_configs_equal_except_expected_controls", False)) for run in runs))
    decision = bool(audit_ok and within >= required and performance)
    return {
        "experiment": "proxy_guided_adaptive_budgeted_quality_aware_mappo",
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
            "proxy_model": "baseline_mappo/train actor-visible graph observations only",
        },
        "fixed_controls": {
            "measurement_overhead_enabled": True,
            "quality_logit_penalty": float(config.measurement_quality_logit_penalty),
            "proxy_teacher_coef": float(config.proxy_teacher_coef),
            "proxy_temperature": float(config.proxy_temperature),
            "communication_cost_weight": float(config.communication_cost_weight),
            "measurement_rate_tolerance": float(config.measurement_rate_tolerance),
        },
        "checks": {
            "seed_count": len(config.seeds), "branch_count": len(BRANCHES),
            "structural": structural, "variants": variants, "control_flags": controls,
            "train_only_targets": targets_ok, "all_episode_metrics_finite": finite,
            "training_metrics_finite": train_finite, "dual_state_finite": dual_ok,
            "proxy_fit_train_only": proxy_ok, "rates_complete": rates_complete,
            "budget_violations": len(violations), "within_target_count": int(within),
            "required_within_target_count": required, "performance_gate": performance,
            "primary_comparison": comparison,
        },
        "decision_gate": "proxy_guided_measurement_mappo_supported" if decision else "proxy_guided_measurement_mappo_not_yet_supported",
        "audit_ok": audit_ok,
        "decision_supported": decision,
        "interpretation_boundary": "This branch tests whether an execution-visible information-gain minus communication-cost teacher stabilizes the measurement head. It does not establish final superiority, safety, or 5/7-UAV scalability.",
    }


def _plot(config: ProxyGuidedAdaptiveMAPPOConfig, rates: Sequence[Mapping[str, Any]], means: Sequence[Mapping[str, Any]], training: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {BASELINE_BRANCH: "#2563eb", ADAPTIVE_BRANCH: "#16a34a", PROXY_BRANCH: "#dc2626"}
    labels = {BASELINE_BRANCH: "Baseline", ADAPTIVE_BRANCH: "Adaptive rate", PROXY_BRANCH: "Proxy-guided"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for branch in BRANCHES:
        selected = sorted([row for row in rates if str(row.get("treatment")) == branch], key=lambda row: int(row["base_seed"]))
        axes[0, 0].plot([int(r["base_seed"]) for r in selected], [_finite(r.get("measurement_rate_mean"), 0.0) for r in selected], "o-", label=labels[branch], color=colors[branch])
        axes[0, 1].plot([int(r["base_seed"]) for r in selected], [_finite(r.get("max_abs_deviation_from_baseline_train_target"), 0.0) for r in selected], "o-", label=labels[branch], color=colors[branch])
    axes[0, 0].set_title("Validation measurement rate")
    axes[0, 1].set_title("Maximum target deviation")
    for axis in axes[0, :2]:
        axis.set_xlabel("Seed"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    for axis, metric, title in ((axes[0, 2], "measurement_proxy_teacher_loss", "Proxy teacher loss"), (axes[1, 0], "position_rmse_m", "Position RMSE"), (axes[1, 1], "j_com", "Communication cost"), (axes[1, 2], "mean_reward", "Team reward")):
        for branch in BRANCHES:
            selected = sorted([row for row in (training if metric.startswith("measurement_proxy") else means) if str(row.get("treatment")) == branch and (metric.startswith("measurement_proxy") or str(row.get("split")) == "validation")], key=lambda row: int(row.get("base_seed", 0)))
            if not selected:
                continue
            values = [_finite(r.get(metric), 0.0) for r in selected]
            axis.plot([int(r.get("base_seed", 0)) for r in selected], values, "o-", label=labels[branch], color=colors[branch])
        axis.set_title(title); axis.set_xlabel("Seed"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "proxy_guided_adaptive_mappo.png"
    fig.suptitle("Proxy-guided adaptive measurement MAPPO (validation; test sealed)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_proxy_guided_adaptive_budgeted_quality_aware_mappo(config: ProxyGuidedAdaptiveMAPPOConfig) -> dict[str, Any]:
    root, logs_dir, plots_dir = _prepare_outputs(config)
    if not Path(config.scenario_profile_path).is_file():
        raise FileNotFoundError(f"scenario profile not found: {config.scenario_profile_path}")
    if not Path(config.ekf_config_path).is_file():
        raise FileNotFoundError(f"EKF config not found: {config.ekf_config_path}")
    runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    budgets: list[dict[str, Any]] = []
    targets_by_seed: dict[int, Mapping[str, float]] = {}
    proxy_models: dict[int, Any] = {}
    for seed in config.seeds:
        run = _run_seed(config, int(seed), root)
        runs.append(run)
        targets.extend(run["measurement_rate_target_rows"])
        targets_by_seed[int(seed)] = run["measurement_rate_targets"]
        proxy_models[int(seed)] = run.get("proxy_model")
        for uav, value in run["measurement_budgets"].items():
            budgets.append({"base_seed": int(seed), "uav_id": uav, "measurement_budget": int(value), "source_split": "train", "source_branch": BASELINE_BRANCH})
        for branch in BRANCHES:
            episode_rows.extend(_episode_rows_for_branch(root / f"seed_{int(seed)}" / branch / "scale_3", branch, int(seed)))
    means = _seed_means(episode_rows)
    contrasts = _paired_contrasts(episode_rows)
    cross = _cross_seed(contrasts, config)
    rates = _validation_rates(root, config.seeds, targets_by_seed)
    training = _training_rows(root, config.seeds)
    violations = _budget_violations(root, config.seeds)
    duals = _checkpoint_duals(root, config.seeds)
    audit = _audit(config, runs, episode_rows, means, cross, rates, targets, training, violations, duals)
    plot = _plot(config, rates, means, training, plots_dir)
    _write_table(logs_dir / "measurement_rate_targets.csv", targets)
    _write_table(logs_dir / "measurement_budgets.csv", budgets)
    _write_table(logs_dir / "training_rate_constraint.csv", training)
    _write_table(logs_dir / "proxy_training_guidance.csv", [row for row in training if str(row.get("treatment")) == PROXY_BRANCH])
    _write_table(logs_dir / "validation_policy_metrics.csv", [row for row in means if str(row.get("split")) == "validation"])
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "seed_metrics.csv", means)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "cross_seed_summary.csv", cross)
    _write_table(logs_dir / "validation_measurement_rates.csv", rates)
    _write_table(logs_dir / "budget_violations.csv", violations)
    _write_table(logs_dir / "checkpoint_dual_audit.csv", duals)
    _write_json(logs_dir / "proxy_parameters.json", {str(seed): model for seed, model in proxy_models.items()})
    _write_json(logs_dir / "audit_report.json", audit)
    manifest = {
        "dataset_id": "python_sim_proxy_guided_adaptive_budgeted_quality_aware_mappo",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_stability_validation_only",
        "seed_set": [int(seed) for seed in config.seeds],
        "branches": {
            BASELINE_BRANCH: "historical actor with measurement overhead",
            ADAPTIVE_BRANCH: "quality-aware actor with fixed train budget and adaptive rate dual",
            PROXY_BRANCH: "adaptive branch plus detached information-gain minus communication-cost teacher on measurement head",
        },
        "proxy_teacher": {
            "utility": "expected_information_gain_norm - communication_cost_weight * predicted_communication_cost_norm",
            "teacher_coefficient": float(config.proxy_teacher_coef),
            "temperature": float(config.proxy_temperature),
            "communication_cost_weight": float(config.communication_cost_weight),
            "gradient_scope": "measurement_head_only; detached actor-body context",
            "fit_source": "baseline_mappo/train only",
        },
        "independent_unit": "base training seed",
        "test_data_used_for_selection": False,
        "outputs": {"audit_report": str(logs_dir / "audit_report.json"), "comparison_plot": str(plot)},
        "audit": audit,
    }
    _write_json(root / "manifest.json", _jsonable(manifest))
    return {"output_dir": str(root), "logs_dir": str(logs_dir), "plots_dir": str(plots_dir), "seeds": list(config.seeds), "audit_ok": bool(audit["audit_ok"]), "decision_supported": bool(audit["decision_supported"]), "audit_report": str(logs_dir / "audit_report.json"), "comparison_plot": str(plot)}


@dataclass(frozen=True)
class ProxyTeacherAuditConfig:
    reference_root: Path
    output_dir: Path
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    age_variance_coefficient: float = 1.0
    communication_cost_weight: float = 1.0
    temperature: float = 0.25
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique and non-empty")
        if self.temperature <= 0.0 or self.age_variance_coefficient < 0.0 or self.communication_cost_weight < 0.0:
            raise ValueError("proxy audit coefficients are invalid")


def _proxy_audit_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [row for row in rows if _finite(row.get("proxy_valid"), 0.0) > 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    if valid:
        axes[0].scatter([_finite(row.get("expected_information_gain_norm"), 0.0) for row in valid], [_finite(row.get("proxy_utility"), 0.0) for row in valid], s=8, alpha=0.35)
        axes[1].scatter([_finite(row.get("link_quality_q"), 0.0) for row in valid], [_finite(row.get("predicted_communication_cost_norm"), 0.0) for row in valid], s=8, alpha=0.35)
        axes[2].hist([_finite(row.get("window_proxy_spread_raw"), 0.0) for row in valid], bins=20)
    axes[0].set_title("Gain vs utility"); axes[1].set_title("Quality vs cost"); axes[2].set_title("Multi-candidate spread")
    for axis in axes: axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def run_proxy_teacher_audit(config: ProxyTeacherAuditConfig) -> dict[str, Any]:
    logs = Path(config.output_dir)
    plots = Path(config.plots_dir) if config.plots_dir is not None else logs.parent.parent / "plots" / logs.name
    if config.force:
        for path in (logs, plots):
            if path.exists(): shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("proxy teacher audit output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}
    for seed in config.seeds:
        scale = Path(config.reference_root) / f"seed_{seed}" / BASELINE_BRANCH / "scale_3"
        train_rows: list[dict[str, Any]] = []
        for episode in sorted(path for path in (scale / "train").iterdir() if path.is_dir()):
            train_rows.extend(_candidate_rows_from_episode(episode, base_seed=seed, treatment=BASELINE_BRANCH, split="train", include_offline_label=False, age_variance_coefficient=config.age_variance_coefficient))
        if not train_rows:
            raise RuntimeError(f"missing baseline train rows for seed {seed}")
        scales = _fit_scales(train_rows, config.age_variance_coefficient)
        model = {"schema_version": "execution-proxy-v1", "source_split": "train", "source_branch": BASELINE_BRANCH, "scales": scales.as_dict(), "age_variance_coefficient": config.age_variance_coefficient, "validation_used_for_fitting": False, "test_used_for_fitting": False}
        parameters[str(seed)] = model
        val_rows: list[dict[str, Any]] = []
        for episode in sorted(path for path in (scale / "validation").iterdir() if path.is_dir()):
            val_rows.extend(_candidate_rows_from_episode(episode, base_seed=seed, treatment=BASELINE_BRANCH, split="validation", include_offline_label=True, age_variance_coefficient=config.age_variance_coefficient))
        for row in _materialize_proxy_rows(val_rows, scales, config.age_variance_coefficient):
            value = dict(row)
            value["predicted_communication_cost_norm"] = float(np.clip(1.0 - _finite(row.get("link_quality_q"), 0.0), 0.0, 1.0))
            value["proxy_utility"] = float(_finite(row.get("expected_information_gain_norm"), 0.0) - config.communication_cost_weight * value["predicted_communication_cost_norm"])
            all_rows.append(value)
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in all_rows:
        groups.setdefault((row["base_seed"], row["episode_id"], row["timestamp"], row["uav_id"]), []).append(row)
    teacher_rows = [{"base_seed": key[0], "episode_id": key[1], "timestamp": key[2], "uav_id": key[3], "candidate_count": len(rows), "valid_count": sum(_finite(r.get("proxy_valid"), 0.0) > 0.5 for r in rows), "teacher_choice": max(rows, key=lambda r: (_finite(r.get("proxy_utility"), -1e9), str(r.get("source_uav_id"))))["source_uav_id"] if rows else "none", "teacher_utility": max((_finite(r.get("proxy_utility"), -1e9) for r in rows), default=0.0)} for key, rows in sorted(groups.items())]
    _write_table(logs / "candidate_proxy.csv", all_rows)
    _write_table(logs / "teacher_targets.csv", teacher_rows)
    _write_json(logs / "proxy_parameters.json", _jsonable(parameters))
    valid_fraction = float(np.mean([_finite(row.get("proxy_valid"), 0.0) for row in all_rows])) if all_rows else 0.0
    multi = [row for row in all_rows if int(_finite(row.get("multi_candidate_window"), 0.0)) == 1]
    nonzero_spread = float(np.mean([_finite(row.get("window_proxy_spread_raw"), 0.0) > 0.0 for row in multi])) if multi else 0.0
    finite = all(np.isfinite(_finite(row.get("expected_information_gain_norm"), float("nan"))) and np.isfinite(_finite(row.get("proxy_utility"), float("nan"))) for row in all_rows)
    audit = {"experiment": "proxy_teacher_audit", "reference_root": str(config.reference_root), "seeds": list(config.seeds), "source_split": "baseline_mappo/train", "evaluation_split": "validation", "test_data_used": False, "checks": {"candidate_rows": len(all_rows), "proxy_finite": finite, "valid_fraction": valid_fraction, "multi_candidate_nonzero_spread_fraction": nonzero_spread, "deterministic_teacher": True}, "audit_ok": bool(all_rows and finite and valid_fraction >= 0.95 and nonzero_spread >= 0.80), "decision_gate": "proxy_teacher_ready" if all_rows and finite and valid_fraction >= 0.95 and nonzero_spread >= 0.80 else "proxy_teacher_not_ready"}
    _write_json(logs / "audit_report.json", audit)
    figure = plots / "proxy_teacher_audit.png"; _proxy_audit_plot(all_rows, figure)
    return {"output_dir": str(logs), "plots_dir": str(plots), "audit_ok": bool(audit["audit_ok"]), "audit_report": str(logs / "audit_report.json"), "plot": str(figure)}


__all__ = [
    "ADAPTIVE_BRANCH", "BASELINE_BRANCH", "BRANCHES", "DEFAULT_SEEDS",
    "PROXY_BRANCH", "ProxyGuidedAdaptiveMAPPOConfig", "ProxyTeacherAuditConfig",
    "run_proxy_guided_adaptive_budgeted_quality_aware_mappo", "run_proxy_teacher_audit",
]
