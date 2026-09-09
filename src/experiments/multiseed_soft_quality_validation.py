"""Independent-seed validation for the frozen soft measurement selector.

This module is intentionally a batch driver around the existing formal training
and paired soft-quality intervention.  It does not change the selector, EKF,
reward, channel, GAT, or MAPPO implementation.  Each base seed is an
independent training/evaluation replication; episodes are paired within a seed
for the intervention contrast, and windows/events are never treated as
independent replicates.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _write_json
from .baseline import _read_csv, _read_json
from .formal import (
    FormalExperimentConfig,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)
from .localization_coupling_intervention import _bootstrap_interval, _write_table
from .soft_quality_measurement_intervention import (
    SoftQualityMeasurementInterventionConfig,
    TREATMENTS,
    run_soft_quality_measurement_intervention,
)


DEFAULT_SEEDS = (2026, 3035, 4044, 5053, 6062)
DEFAULT_PROFILE = Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json")
DEFAULT_EKF_CONFIG = Path(
    "data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"
)
_TREATMENT_METRICS = (
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
)
_CONTRAST_METRICS = _TREATMENT_METRICS
_LOWER_IS_BETTER = {
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
}


@dataclass(frozen=True)
class MultiSeedSoftQualityValidationConfig:
    """Configuration for independent 3-UAV Moderate replications."""

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
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("multi-seed soft-quality validation is currently fixed to 3 UAVs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if min(self.warmup_episodes, self.train_episodes, self.validation_episodes, self.test_episodes) < 1:
            raise ValueError("all episode counts must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.device != "cpu":
            raise ValueError("multi-seed validation currently supports device='cpu' only")


def _progress(config: MultiSeedSoftQualityValidationConfig, message: str) -> None:
    if config.progress:
        print(f"[multiseed-soft-quality] {message}", flush=True)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _mean(values: Iterable[Any]) -> float:
    array = np.asarray([_finite_float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        raise ValueError("cannot aggregate an empty/non-finite metric")
    return float(np.mean(array))


def _prepare_batch_output(config: MultiSeedSoftQualityValidationConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / root.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / root.name
    targets = (root, logs, plots)
    if not config.force:
        nonempty = [str(path) for path in targets if path.exists() and any(path.iterdir())]
        if nonempty:
            raise FileExistsError(
                "multi-seed output is not empty; use --force to replace only these generated directories: "
                + ", ".join(nonempty)
            )
    else:
        for path in targets:
            if path.exists():
                shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
    return root, logs, plots


def _formal_run_for_seed(
    config: MultiSeedSoftQualityValidationConfig,
    seed: int,
    reference_root: Path,
) -> dict[str, Any]:
    ekf = load_selected_ekf_config(config.ekf_config_path)
    command_line = (
        "run-multiseed-soft-quality-validation "
        f"--seed {seed} --uav-count 3 --steps {config.max_steps} "
        f"--warmup-episodes {config.warmup_episodes} --train-episodes {config.train_episodes} "
        f"--validation-episodes {config.validation_episodes} --test-episodes {config.test_episodes}"
    )
    base = FormalExperimentConfig(
        output_dir=reference_root,
        uav_count=config.uav_count,
        seed=int(seed),
        max_steps=config.max_steps,
        warmup_episodes=config.warmup_episodes,
        train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes,
        test_episodes=config.test_episodes,
        device=config.device,
        force=False,
        vo_position_std_m=ekf["vo_position_std_m"],
        process_position_std_m=ekf["process_position_std_m"],
        process_velocity_std_mps=ekf["process_velocity_std_mps"],
        nlos_variance_multiplier=ekf["nlos_variance_multiplier"],
        scenario_profile_path=Path(config.scenario_profile_path),
        ekf_config_source=str(config.ekf_config_path),
        command_line=command_line,
        progress=config.progress,
    )
    # This mirrors the ordinary single-scale CLI: scales are fitted only from
    # warm-up probes for this seed, before the training/validation episodes.
    metric_scales = fit_shared_training_metric_scales(base, (config.uav_count,))
    formal = FormalExperimentConfig(
        **{
            **asdict(base),
            "metric_scales": metric_scales,
            "metric_scale_source": "pooled_training_warmup_p95",
        }
    )
    return run_preliminary(formal)


def _treatment_means(rows: Sequence[Mapping[str, Any]], base_seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        selected = [row for row in rows if str(row.get("treatment")) == treatment]
        if not selected:
            raise ValueError(f"missing treatment rows for seed {base_seed}: {treatment}")
        result: dict[str, Any] = {
            "base_seed": int(base_seed),
            "treatment": treatment,
            "episode_count": len(selected),
        }
        for metric in _TREATMENT_METRICS:
            result[metric] = _mean(row.get(metric) for row in selected)
        output.append(result)
    return output


def _seed_contrasts(rows: Sequence[Mapping[str, Any]], base_seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("episode_id")), {})[str(row.get("treatment"))] = row
    required = {"round_robin_candidate", "soft_quality_candidate"}
    if not grouped or any(set(value) != set(TREATMENTS) for value in grouped.values()):
        raise ValueError(f"paired treatment rows are incomplete for seed {base_seed}")
    output: list[dict[str, Any]] = []
    for metric in _CONTRAST_METRICS:
        differences = np.asarray(
            [
                _finite_float(pair["soft_quality_candidate"].get(metric))
                - _finite_float(pair["round_robin_candidate"].get(metric))
                for pair in grouped.values()
            ],
            dtype=float,
        )
        if not np.isfinite(differences).all():
            raise ValueError(f"non-finite paired difference for seed {base_seed}, metric {metric}")
        output.append(
            {
                "base_seed": int(base_seed),
                "comparison": "soft_quality_candidate_minus_round_robin_candidate",
                "metric": metric,
                "paired_episodes": int(differences.size),
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "paired_difference_std": float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0,
                "improvement_direction": "lower" if metric in _LOWER_IS_BETTER else "higher",
                "improved_episode_count": int(
                    np.sum(differences < 0.0 if metric in _LOWER_IS_BETTER else differences > 0.0)
                ),
            }
        )
    return output


def _cross_seed_summary(
    seed_contrasts: Sequence[Mapping[str, Any]],
    config: MultiSeedSoftQualityValidationConfig,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    metrics = sorted({str(row["metric"]) for row in seed_contrasts})
    for metric in metrics:
        selected = sorted(
            (row for row in seed_contrasts if str(row["metric"]) == metric),
            key=lambda row: int(row["base_seed"]),
        )
        values = np.asarray([float(row["mean_paired_difference"]) for row in selected], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite cross-seed values for metric {metric}")
        ci_low, ci_high = _bootstrap_interval(values, config.bootstrap_samples, rng)
        lower = metric in _LOWER_IS_BETTER
        improved = values < 0.0 if lower else values > 0.0
        output.append(
            {
                "comparison": "soft_quality_candidate_minus_round_robin_candidate",
                "metric": metric,
                "seed_count": int(values.size),
                "mean_across_seeds": float(np.mean(values)),
                "median_across_seeds": float(np.median(values)),
                "std_across_seeds": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "improvement_direction": "lower" if lower else "higher",
                "improved_seed_count": int(np.sum(improved)),
                "required_improved_seed_count": int(math.ceil(0.8 * values.size)),
                "ci_supports_improvement": bool(ci_high < 0.0 if lower else ci_low > 0.0),
            }
        )
    return output


def _audit(
    config: MultiSeedSoftQualityValidationConfig,
    seed_runs: Sequence[Mapping[str, Any]],
    treatment_means: Sequence[Mapping[str, Any]],
    seed_contrasts: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_metrics = {"position_rmse_m", "harmful_fraction"}
    by_metric = {str(row["metric"]): row for row in cross_seed}
    rmse = by_metric.get("position_rmse_m", {})
    harmful = by_metric.get("harmful_fraction", {})
    required_seed_count = int(math.ceil(0.8 * len(config.seeds)))
    seed_audits_ok = all(bool(row.get("audit_ok")) for row in seed_runs)
    treatment_count_ok = all(
        int(row.get("episode_count", 0)) == config.validation_episodes
        for row in treatment_means
    )
    frozen_rule_ok = all(
        str(row.get("comparison")) == "soft_quality_candidate_minus_round_robin_candidate"
        for row in seed_contrasts
    )
    stable = (
        seed_audits_ok
        and treatment_count_ok
        and frozen_rule_ok
        and set(by_metric).issuperset(required_metrics)
        and int(rmse.get("improved_seed_count", 0)) >= required_seed_count
        and int(harmful.get("improved_seed_count", 0)) >= required_seed_count
        and bool(rmse.get("ci_supports_improvement", False))
        and bool(harmful.get("ci_supports_improvement", False))
    )
    return {
        "experiment": "frozen_soft_quality_multiseed_validation",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "uav_count": config.uav_count,
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed; episodes are paired within each seed",
        "episode_count_per_seed": config.validation_episodes,
        "event_rows_are_independent_replicates": False,
        "test_data_used": False,
        "test_data_generated_and_sealed": True,
        "checks": {
            "seed_count": len(config.seeds),
            "required_seed_count_for_4_of_5_rule": required_seed_count,
            "all_seed_intervention_audits_ok": seed_audits_ok,
            "all_treatments_have_expected_validation_episodes": treatment_count_ok,
            "frozen_comparison_rule_consistent": frozen_rule_ok,
            "rmse_improved_seed_count": int(rmse.get("improved_seed_count", 0)),
            "harmful_fraction_improved_seed_count": int(harmful.get("improved_seed_count", 0)),
            "rmse_cross_seed_ci": [
                float(rmse.get("bootstrap_95_ci_low", float("nan"))),
                float(rmse.get("bootstrap_95_ci_high", float("nan"))),
            ],
            "harmful_fraction_cross_seed_ci": [
                float(harmful.get("bootstrap_95_ci_low", float("nan"))),
                float(harmful.get("bootstrap_95_ci_high", float("nan"))),
            ],
        },
        "decision_gate": (
            "soft_quality_stable_across_seeds"
            if stable
            else "soft_quality_not_yet_supported_across_seeds"
        ),
        "overall_ok": bool(
            seed_audits_ok
            and treatment_count_ok
            and frozen_rule_ok
            and all(np.isfinite(float(row[key])) for row in cross_seed for key in (
                "mean_across_seeds", "bootstrap_95_ci_low", "bootstrap_95_ci_high"
            ))
        ),
        "interpretation_boundary": (
            "The selector formula and train-only fitting procedure are frozen. "
            "A stable gate supports a later MAPPO integration experiment; it does not "
            "itself prove final MAPPO superiority or safety/energy performance."
        ),
    }


def run_multiseed_soft_quality_validation(
    config: MultiSeedSoftQualityValidationConfig,
) -> dict[str, Any]:
    """Train independent seeds and run the frozen validation-only intervention."""

    root, logs_dir, plots_dir = _prepare_batch_output(config)
    if not Path(config.scenario_profile_path).exists():
        raise FileNotFoundError(f"scenario profile not found: {config.scenario_profile_path}")
    if not Path(config.ekf_config_path).exists():
        raise FileNotFoundError(f"EKF config not found: {config.ekf_config_path}")

    seed_runs: list[dict[str, Any]] = []
    all_treatment_means: list[dict[str, Any]] = []
    all_seed_contrasts: list[dict[str, Any]] = []
    for base_seed in config.seeds:
        seed = int(base_seed)
        _progress(config, f"seed {seed}: training independent MAPPO reference")
        reference_root = root / "references" / f"seed_{seed}"
        reference_summary = _formal_run_for_seed(config, seed, reference_root)
        reference_scale = reference_root / "scale_3"
        intervention_root = root / "interventions" / f"seed_{seed}"
        seed_logs = logs_dir / f"seed_{seed}"
        seed_plots = plots_dir / f"seed_{seed}"
        _progress(config, f"seed {seed}: running frozen paired validation intervention")
        intervention_summary = run_soft_quality_measurement_intervention(
            SoftQualityMeasurementInterventionConfig(
                reference_scale=reference_scale,
                output_dir=intervention_root,
                split="validation",
                logs_dir=seed_logs,
                plots_dir=seed_plots,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed + seed,
                force=False,
                progress=config.progress,
            )
        )
        episode_rows = _read_csv(seed_logs / "episode_metrics.csv")
        means = _treatment_means(episode_rows, seed)
        contrasts = _seed_contrasts(episode_rows, seed)
        audit = _read_json(seed_logs / "audit_report.json")
        seed_run = {
            "base_seed": seed,
            "reference_root": str(reference_root),
            "reference_scale": str(reference_scale),
            "reference_manifest": str(reference_scale / "manifest.json"),
            "intervention_root": str(intervention_root),
            "intervention_logs": str(seed_logs),
            "intervention_plots": str(seed_plots),
            "train_episodes": config.train_episodes,
            "validation_episodes": config.validation_episodes,
            "test_episodes_generated": config.test_episodes,
            "test_data_used": False,
            "reference_run_status": reference_summary.get("result_status", "unknown"),
            "intervention_decision_gate": intervention_summary.get("decision_gate", "unknown"),
            "audit_ok": bool(audit.get("overall_ok", False)),
        }
        seed_runs.append(seed_run)
        all_treatment_means.extend(means)
        all_seed_contrasts.extend(contrasts)

    cross_seed = _cross_seed_summary(all_seed_contrasts, config)
    audit = _audit(config, seed_runs, all_treatment_means, all_seed_contrasts, cross_seed)
    _write_table(logs_dir / "seed_runs.csv", seed_runs)
    _write_table(logs_dir / "treatment_means.csv", all_treatment_means)
    _write_table(logs_dir / "seed_contrasts.csv", all_seed_contrasts)
    _write_table(logs_dir / "cross_seed_summary.csv", cross_seed)
    _write_json(logs_dir / "audit_report.json", audit)
    manifest = {
        "dataset_id": f"python_sim_frozen_soft_quality_multiseed_{config.seeds[0]}",
        "schema_version": "episode-v1.3",
        "result_status": "engineering_mechanism_diagnostic_only",
        "uav_count": config.uav_count,
        "seeds": [int(seed) for seed in config.seeds],
        "training": {
            "warmup_episodes": config.warmup_episodes,
            "train_episodes": config.train_episodes,
            "validation_episodes": config.validation_episodes,
            "test_episodes_generated_but_sealed": config.test_episodes,
        },
        "selector": {
            "rule": "frozen soft quality score",
            "fit_source": "train only per base seed",
            "weights": "predeclared equal weights",
            "none_threshold": "train q90 per base seed",
            "forbidden_inputs": ["true position", "physical LOS/NLOS", "future arrival", "test data"],
        },
        "independent_unit": "base training seed; paired episodes within seed",
        "outputs": {
            "seed_runs": str(logs_dir / "seed_runs.csv"),
            "treatment_means": str(logs_dir / "treatment_means.csv"),
            "seed_contrasts": str(logs_dir / "seed_contrasts.csv"),
            "cross_seed_summary": str(logs_dir / "cross_seed_summary.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "audit": audit,
    }
    _write_json(root / "manifest.json", manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"multi-seed soft-quality audit failed: {audit['checks']}")
    return {
        "output_dir": str(root),
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "seeds": [int(seed) for seed in config.seeds],
        "decision_gate": audit["decision_gate"],
        "audit_ok": True,
        "cross_seed_summary": str(logs_dir / "cross_seed_summary.csv"),
    }


__all__ = [
    "DEFAULT_SEEDS",
    "DEFAULT_PROFILE",
    "DEFAULT_EKF_CONFIG",
    "MultiSeedSoftQualityValidationConfig",
    "run_multiseed_soft_quality_validation",
]
