"""Multi-seed MAPPO comparison with an actor-visible quality-aware branch.

The experiment trains two policies from the same deterministic configuration for
each base seed:

``baseline_mappo``
    The historical actor.  Its measurement head receives the original graph
    context and action mask only.

``quality_aware_mappo``
    The same actor with two additional ID-aligned measurement features: a
    frozen, warm-up-fitted risk vector and a validity vector.

``budgeted_quality_aware_mappo``
    The quality-aware actor with one additional remaining-budget feature and
    an environment-enforced fixed measurement budget fitted from baseline
    training episodes only.

The runner deliberately treats the training seed as the independent unit.
Validation is used for the decision gate and test is generated but sealed until
the final report.  This module does not use true position, physical LOS/NLOS,
future packet outcomes, or offline error labels to construct actor inputs.
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
from ..simulation.ekf import EKFConfig
from .baseline import _read_csv, _read_json
from .formal import (
    FormalExperimentConfig,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)
from .metrics import position_error_metrics
from .localization_coupling_intervention import _bootstrap_interval


DEFAULT_SEEDS = (2026, 3035, 4044, 5053, 6062)
DEFAULT_PROFILE = Path(
    "data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"
)
DEFAULT_EKF_CONFIG = Path(
    "data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"
)
TREATMENTS = (
    "baseline_mappo",
    "quality_aware_mappo",
    "budgeted_quality_aware_mappo",
)

_METRICS = (
    "mean_reward",
    "position_rmse_m",
    "position_rmse_norm_m",
    "position_p95_norm_m",
    "position_error_p95_m",
    "j_loc",
    "j_com",
    "joint_objective",
    "accepted_update_count",
    "update_coverage_rate",
    "update_coverage_any",
    "accepted_update_rate",
    "accepted_updates_per_window",
    "beneficial_per_window",
    "harmful_per_window",
    "no_update_episode_count",
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
    "j_loc_saturation_fraction",
)
_PLOT_METRICS = (
    "position_rmse_norm_m",
    "harmful_fraction",
    "accepted_update_count",
    "none_selection_rate",
)
_LOWER_IS_BETTER = {
    "position_rmse_m",
    "position_rmse_norm_m",
    "position_p95_norm_m",
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
    "j_loc_saturation_fraction",
    "no_update_episode_count",
}


@dataclass(frozen=True)
class QualityAwareMAPPOConfig:
    """Configuration for the formal 3-UAV three-branch comparison."""

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
    measurement_overhead_enabled: bool = False
    measurement_quality_logit_penalty: float = 1.0
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("quality-aware MAPPO integration is currently fixed to 3 UAVs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if min(
            self.warmup_episodes,
            self.train_episodes,
            self.validation_episodes,
            self.test_episodes,
        ) < 1:
            raise ValueError("warmup, train, validation, and test episode counts must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.device != "cpu":
            raise ValueError("quality-aware MAPPO currently supports device='cpu' only")
        if not math.isfinite(self.measurement_quality_logit_penalty) or self.measurement_quality_logit_penalty < 0.0:
            raise ValueError("measurement_quality_logit_penalty must be finite and non-negative")


def _progress(config: QualityAwareMAPPOConfig, message: str) -> None:
    if config.progress:
        print(f"[quality-aware-mappo] {message}", flush=True)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    array = np.asarray([_finite(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _quantile(values: Iterable[Any], quantile: float, default: float = float("nan")) -> float:
    array = np.asarray([_finite(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if array.size else float(default)


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                columns.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _prepare_batch_output(config: QualityAwareMAPPOConfig) -> tuple[Path, Path, Path]:
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
    targets = (root, logs, plots)
    if not config.force:
        nonempty = [str(path) for path in targets if path.exists() and any(path.iterdir())]
        if nonempty:
            raise FileExistsError(
                "quality-aware MAPPO output is not empty; use --force to replace only: "
                + ", ".join(nonempty)
            )
    else:
        for path in targets:
            if path.exists():
                shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
    return root, logs, plots


def _formal_base_config(
    config: QualityAwareMAPPOConfig,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
) -> FormalExperimentConfig:
    command_line = (
        "run-quality-aware-mappo "
        f"--seed {seed} --steps {config.max_steps} "
        f"--warmup-episodes {config.warmup_episodes} "
        f"--train-episodes {config.train_episodes} "
        f"--validation-episodes {config.validation_episodes} "
        f"--test-episodes {config.test_episodes}"
    )
    if config.measurement_overhead_enabled:
        command_line += " --measurement-overhead-enabled"
    command_line += f" --quality-logit-penalty {config.measurement_quality_logit_penalty:g}"
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
        command_line=command_line,
        progress=config.progress,
        mappo_variant="baseline_mappo",
        measurement_overhead_enabled=config.measurement_overhead_enabled,
        measurement_quality_logit_penalty=config.measurement_quality_logit_penalty,
    )


def _fit_baseline_measurement_budgets(
    branch_scale: Path,
    *,
    uav_count: int,
) -> dict[str, int]:
    """Fit fixed per-UAV measurement budgets from baseline train episodes only."""

    train_root = branch_scale / "train"
    episode_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"baseline train episodes are missing under {train_root}")
    uav_ids = [f"uav_{index:02d}" for index in range(uav_count)]
    counts_by_episode: list[dict[str, int]] = []
    for episode_root in episode_dirs:
        rows = _read_csv(episode_root / "actions.csv")
        counts = {uav_id: 0 for uav_id in uav_ids}
        for row in rows:
            uav_id = str(row.get("uav_id", ""))
            if uav_id in counts and str(row.get("measurement_uav_id", "none")) != "none":
                counts[uav_id] += 1
        counts_by_episode.append(counts)
    return {
        uav_id: max(
            0,
            int(np.rint(np.mean([counts[uav_id] for counts in counts_by_episode]))),
        )
        for uav_id in uav_ids
    }


def _run_seed_branches(
    config: QualityAwareMAPPOConfig,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    ekf_parameters = load_selected_ekf_config(config.ekf_config_path)
    seed_root = root / f"seed_{int(seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)

    # Fit common metric scales once from the same warm-up probe.  The baseline
    # branch fits the actor-quality model once from warm-up snapshots; the
    # quality-aware branch reuses that exact model so the frozen rule is fitted
    # once per seed, not once per treatment.
    template = _formal_base_config(config, seed, seed_root / "metric_scale_probe", ekf_parameters)
    metric_scales = fit_shared_training_metric_scales(template, (config.uav_count,))
    branch_summaries: dict[str, Any] = {}
    branch_reports: dict[str, Any] = {}
    quality_models: dict[str, Any] = {}
    shared_quality_model: Mapping[str, Any] | None = None
    measurement_budgets: dict[str, int] | None = None
    for treatment in TREATMENTS:
        branch_root = seed_root / treatment
        _progress(config, f"seed {seed}: training {treatment}")
        quality_enabled = treatment in {"quality_aware_mappo", "budgeted_quality_aware_mappo"}
        budget_enabled = treatment == "budgeted_quality_aware_mappo"
        if budget_enabled and measurement_budgets is None:
            raise RuntimeError("measurement budgets were not fitted from the baseline train split")
        formal_config = FormalExperimentConfig(
            **{
                **asdict(template),
                "output_dir": branch_root,
                "metric_scales": metric_scales,
                "metric_scale_source": "pooled_training_warmup_p95",
                "mappo_variant": treatment,
                "measurement_quality_logit_penalty": (
                    float(config.measurement_quality_logit_penalty)
                    if quality_enabled
                    else 0.0
                ),
                "measurement_budget_enabled": budget_enabled,
                "measurement_budget_by_uav": (
                    dict(measurement_budgets) if budget_enabled else None
                ),
            }
        )
        summary = run_preliminary(
            formal_config,
            measurement_quality_model=shared_quality_model,
        )
        scale_root = branch_root / "scale_3"
        report = validate_episode_benchmark(scale_root)
        if not bool(report.get("ok", False)):
            raise RuntimeError(f"episode validation failed for seed {seed}, {treatment}: {report}")
        model_path = scale_root / "calibration" / "measurement_quality_model.json"
        model = _read_json(model_path) if model_path.exists() else {}
        branch_summaries[treatment] = summary
        branch_reports[treatment] = report
        quality_models[treatment] = model
        if treatment == "baseline_mappo":
            if not model:
                raise RuntimeError(f"baseline warm-up quality model is missing for seed {seed}")
            shared_quality_model = model
            measurement_budgets = _fit_baseline_measurement_budgets(
                scale_root,
                uav_count=config.uav_count,
            )

    baseline_model = json.dumps(quality_models["baseline_mappo"], sort_keys=True)
    quality_model_strings = {
        treatment: json.dumps(quality_models[treatment], sort_keys=True)
        for treatment in TREATMENTS
    }
    baseline_manifest = _read_json(seed_root / "baseline_mappo" / "scale_3" / "manifest.json")
    baseline_config = dict(baseline_manifest.get("config", {}))
    comparable_config_keys = {
        "mappo_variant",
        "output_dir",
        "measurement_quality_logit_penalty",
        "measurement_budget_enabled",
        "measurement_budget_by_uav",
    }
    for key in comparable_config_keys:
        baseline_config.pop(key, None)
    branch_config_equal = True
    for treatment in TREATMENTS[1:]:
        manifest = _read_json(seed_root / treatment / "scale_3" / "manifest.json")
        treatment_config = dict(manifest.get("config", {}))
        for key in comparable_config_keys:
            treatment_config.pop(key, None)
        branch_config_equal = branch_config_equal and baseline_config == treatment_config
    return {
        "base_seed": int(seed),
        "seed_root": str(seed_root),
        "branches": {
            treatment: {
                "root": str(seed_root / treatment),
                "scale_root": str(seed_root / treatment / "scale_3"),
                "manifest": str(seed_root / treatment / "scale_3" / "manifest.json"),
                "validation_report_ok": bool(branch_reports[treatment].get("ok", False)),
                "mappo_variant": branch_summaries[treatment].get("mappo_variant", treatment),
                "quality_features_enabled": bool(
                    branch_summaries[treatment].get("measurement_quality", {}).get("enabled", False)
                ),
                "measurement_overhead_enabled": bool(
                    branch_summaries[treatment].get("measurement_overhead_enabled", False)
                ),
                "measurement_quality_logit_penalty": float(
                    branch_summaries[treatment].get("measurement_quality", {}).get(
                        "logit_penalty", 0.0
                    )
                ),
                "measurement_budget_enabled": bool(
                    branch_summaries[treatment].get("measurement_budget_enabled", False)
                ),
                "measurement_budget_by_uav": (
                    {
                        str(key): int(value)
                        for key, value in (
                            branch_summaries[treatment].get("measurement_budget_by_uav") or {}
                        ).items()
                    }
                    if branch_summaries[treatment].get("measurement_budget_by_uav") is not None
                    else None
                ),
            }
            for treatment in TREATMENTS
        },
        "common_metric_scales": {str(key): float(value) for key, value in metric_scales.items()},
        "measurement_budgets": dict(measurement_budgets or {}),
        "warmup_quality_models_identical": all(
            baseline_model == quality_model_strings[treatment]
            for treatment in TREATMENTS[1:]
        ),
        "branch_configs_equal_except_variant": branch_config_equal,
        "validation_reports": branch_reports,
        "test_data_used": False,
    }


def _position_error_p95(episode_root: Path) -> float:
    truth_rows = _read_csv(episode_root / "ground_truth.csv")
    estimate_rows = _read_csv(episode_root / "estimates.csv")
    truth: dict[tuple[str, float], np.ndarray] = {}
    for row in truth_rows:
        key = (str(row.get("uav_id", "")), round(_finite(row.get("timestamp")), 6))
        values = np.asarray([_finite(row.get(axis)) for axis in ("x", "y", "z")], dtype=float)
        if np.isfinite(values).all():
            truth[key] = values
    errors: list[float] = []
    for row in estimate_rows:
        key = (str(row.get("uav_id", "")), round(_finite(row.get("timestamp")), 6))
        estimate = np.asarray([_finite(row.get(axis)) for axis in ("x", "y", "z")], dtype=float)
        if key in truth and np.isfinite(estimate).all():
            errors.append(float(np.linalg.norm(estimate - truth[key])))
    return _quantile(errors, 0.95, default=0.0)


def _event_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row.get("accepted", False))]
    deltas = np.asarray(
        [
            _finite(row.get("true_position_error_after_m"))
            - _finite(row.get("true_position_error_before_m"))
            for row in accepted
        ],
        dtype=float,
    )
    deltas = deltas[np.isfinite(deltas)]
    beneficial = float(np.sum(deltas < -0.001))
    harmful = float(np.sum(deltas > 0.001))
    neutral = float(max(0, deltas.size - int(beneficial) - int(harmful)))
    count = float(max(1, deltas.size))
    stale_limit = float(EKFConfig().stale_message_limit_s)
    available = bool(deltas.size)
    return {
        "accepted_update_count": float(deltas.size),
        # This is an episode-level availability indicator.  The legacy
        # fractions remain numeric for backward compatibility, while the
        # status field makes an episode with no accepted update explicit NA
        # for any update-quality interpretation.
        "update_coverage_rate": float(available),
        "update_quality_available": float(available),
        "update_quality_event_count": float(deltas.size),
        "update_quality_status": "available" if available else "NA",
        "beneficial_fraction": beneficial / count,
        "harmful_fraction": harmful / count,
        "neutral_fraction": neutral / count,
        "nlos_update_fraction": _mean(
            (1.0 - float(bool(row.get("los", True))) for row in accepted), default=0.0
        ),
        "stale_update_fraction": _mean(
            (_finite(row.get("message_age_s"), 0.0) > stale_limit for row in accepted), default=0.0
        ),
        "mean_update_gain_norm": _mean(
            (row.get("kalman_gain_norm") for row in accepted), default=0.0
        ),
        "mean_position_correction_m": _mean(
            (row.get("position_correction_norm_m") for row in accepted), default=0.0
        ),
    }


def _quality_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "none_selection_rate": 0.0,
            "mean_selected_quality_risk": -1.0,
            "mean_selected_quality_valid": 0.0,
            "quality_feature_enabled": 0.0,
        }
    none = [str(row.get("measurement_uav_id", "none")) == "none" for row in rows]
    selected_risks = [
        _finite(row.get("selected_quality_risk"))
        for row in rows
        if str(row.get("measurement_uav_id", "none")) != "none"
    ]
    selected_risks = [value for value in selected_risks if np.isfinite(value)]
    valid = [_finite(row.get("selected_quality_valid"), 0.0) for row in rows]
    enabled = [_finite(row.get("quality_feature_enabled"), 0.0) for row in rows]
    return {
        "none_selection_rate": float(np.mean(none)),
        "mean_selected_quality_risk": float(np.mean(selected_risks)) if selected_risks else -1.0,
        "mean_selected_quality_valid": _mean(valid, default=0.0),
        "quality_feature_enabled": _mean(enabled, default=0.0),
    }


def _measurement_budget_summary(episode_root: Path) -> dict[str, float]:
    """Summarize measurement load and fixed-budget use from episode logs."""

    action_rows = _read_csv(episode_root / "actions.csv")
    measurement_count = float(
        sum(
            str(row.get("measurement_uav_id", "none")) != "none"
            for row in action_rows
        )
    )
    overhead_rows = _read_csv(episode_root / "measurement_overhead.csv")
    overhead_packets = float(
        sum(
            max(0.0, _finite(row.get("load_packets_applied"), 0.0))
            for row in overhead_rows
        )
    )
    scenario = _read_json(episode_root / "scenario.json")
    limits_payload = scenario.get("measurement_budget_limits", {})
    limits = {
        str(key): max(0, int(_finite(value, 0.0)))
        for key, value in limits_payload.items()
        if np.isfinite(_finite(value, 0.0))
    }
    total_budget = float(sum(limits.values()))
    usage = measurement_count / total_budget if total_budget > 0.0 else 0.0
    return {
        "measurement_count": measurement_count,
        "measurement_overhead_packets": overhead_packets,
        "measurement_budget_usage_rate": float(np.clip(usage, 0.0, 1.0)),
    }


def _episode_rows_for_branch(
    branch_scale: Path,
    treatment: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    metrics_rows = _read_csv(branch_scale / "diagnostics" / "episode_metrics.csv")
    events = _read_jsonl(branch_scale / "diagnostics" / "cooperative_updates.jsonl")
    quality = _read_jsonl(branch_scale / "diagnostics" / "measurement_quality.jsonl")
    output: list[dict[str, Any]] = []
    for row in metrics_rows:
        split = str(row.get("split", ""))
        episode_id = str(row.get("episode_id", ""))
        episode_root = branch_scale / split / episode_id
        event_rows = [
            event
            for event in events
            if str(event.get("split", "")) == split
            and str(event.get("episode_id", "")) == episode_id
        ]
        quality_rows = [
            event
            for event in quality
            if str(event.get("split", "")) == split
            and str(event.get("episode_id", "")) == episode_id
        ]
        values: dict[str, Any] = dict(row)
        position_metrics = position_error_metrics(
            _read_csv(episode_root / "ground_truth.csv"),
            _read_csv(episode_root / "estimates.csv"),
        )
        values.update(
            {
                "base_seed": int(base_seed),
                "treatment": treatment,
                "position_error_p95_m": _position_error_p95(episode_root),
                "position_rmse_norm_m": _finite(
                    row.get("position_rmse_norm_m"),
                    position_metrics.get("rmse_norm_m", 0.0),
                ),
                "position_p95_norm_m": _finite(
                    row.get("position_p95_norm_m"),
                    position_metrics.get("p95_norm_m", 0.0),
                ),
                **_event_summary(event_rows),
                **_quality_summary(quality_rows),
                **_measurement_budget_summary(episode_root),
                "event_row_count": len(event_rows),
                "quality_log_row_count": len(quality_rows),
            }
        )
        # Historical episode writers predate the canonical update-coverage
        # fields.  Derive finite compatibility values from this episode's
        # rewards/events without changing the original artifacts.
        if "j_loc_saturation_fraction" not in values:
            reward_rows = _read_csv(episode_root / "rewards.csv")
            uncertainty = np.asarray(
                [_finite(item.get("normalized_uncertainty"), float("nan")) for item in reward_rows],
                dtype=float,
            )
            uncertainty = uncertainty[np.isfinite(uncertainty)]
            values["j_loc_saturation_fraction"] = (
                float(np.mean(uncertainty >= 1.0 - 1e-12)) if uncertainty.size else 0.0
            )
        accepted = max(0.0, _finite(values.get("accepted_update_count"), 0.0))
        reward_count = len(_read_csv(episode_root / "rewards.csv"))
        uav_count = max(1, int(_finite(values.get("uav_count"), 3.0)))
        decision_windows = max(1.0, round(reward_count / uav_count))
        values.setdefault("update_coverage_any", float(accepted > 0.0))
        values.setdefault("accepted_update_rate", accepted / max(1.0, decision_windows * uav_count))
        values.setdefault("accepted_updates_per_window", accepted / decision_windows)
        values.setdefault("beneficial_per_window", max(0.0, _finite(values.get("beneficial_fraction"), 0.0) * accepted) / decision_windows)
        values.setdefault("harmful_per_window", max(0.0, _finite(values.get("harmful_fraction"), 0.0) * accepted) / decision_windows)
        values.setdefault("no_update_episode_count", float(accepted <= 0.0))
        for key in _METRICS:
            if not np.isfinite(_finite(values.get(key))):
                raise ValueError(
                    f"non-finite metric {key} in {branch_scale} {split} {episode_id}"
                )
        output.append(values)
    return output


def _seed_treatment_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in rows}):
        for split in ("validation", "test"):
            for treatment in TREATMENTS:
                selected = [
                    row
                    for row in rows
                    if int(row["base_seed"]) == seed
                    and str(row["split"]) == split
                    and str(row["treatment"]) == treatment
                ]
                if not selected:
                    raise ValueError(f"missing rows for seed={seed}, split={split}, treatment={treatment}")
                output.append(
                    {
                        "base_seed": seed,
                        "split": split,
                        "treatment": treatment,
                        "episode_count": len(selected),
                        **{
                            metric: _mean((row.get(metric) for row in selected))
                            for metric in _METRICS
                        },
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
        if set(pair) != set(TREATMENTS):
            raise ValueError(f"unpaired episode {seed}/{split}/{episode_id}: {set(pair)}")
        for metric in _METRICS:
            baseline = _finite(pair["baseline_mappo"].get(metric))
            for treatment in TREATMENTS:
                if treatment == "baseline_mappo":
                    continue
                value = _finite(pair[treatment].get(metric))
                difference = value - baseline
                if not np.isfinite(difference):
                    raise ValueError(f"non-finite paired contrast for {metric}")
                output.append(
                    {
                        "base_seed": seed,
                        "split": split,
                        "episode_id": episode_id,
                        "comparison": f"{treatment}_minus_baseline_mappo",
                        "metric": metric,
                        "baseline_value": baseline,
                        "treatment_value": value,
                        "paired_difference": difference,
                        "improvement_direction": "lower" if metric in _LOWER_IS_BETTER else "higher",
                    }
                )
    return output


def _cross_seed_summary(
    contrasts: Sequence[Mapping[str, Any]],
    config: QualityAwareMAPPOConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for split in ("validation", "test"):
        for comparison in sorted(
            {str(row["comparison"]) for row in contrasts if str(row["split"]) == split}
        ):
            for metric in _METRICS:
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
                    [_mean(values_by_seed[seed], default=float("nan")) for seed in sorted(values_by_seed)],
                    dtype=float,
                )
                if not np.isfinite(seed_values).all():
                    raise ValueError(f"non-finite cross-seed contrast for {split}/{comparison}/{metric}")
                ci_low, ci_high = _bootstrap_interval(seed_values, config.bootstrap_samples, rng)
                lower = metric in _LOWER_IS_BETTER
                improved = seed_values < 0.0 if lower else seed_values > 0.0
                not_worsened = seed_values <= 0.0 if lower else seed_values >= 0.0
                output.append(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        "seed_count": int(seed_values.size),
                        "mean_across_seeds": float(np.mean(seed_values)),
                        "median_across_seeds": float(np.median(seed_values)),
                        "std_across_seeds": float(np.std(seed_values, ddof=1)) if seed_values.size > 1 else 0.0,
                        "bootstrap_95_ci_low": float(ci_low),
                        "bootstrap_95_ci_high": float(ci_high),
                        "improvement_direction": "lower" if lower else "higher",
                        "improved_seed_count": int(np.sum(improved)),
                        "not_worsened_seed_count": int(np.sum(not_worsened)),
                        "required_improved_seed_count": int(math.ceil(0.8 * seed_values.size)),
                        "ci_supports_improvement": bool(
                            ci_high < 0.0 if lower else ci_low > 0.0
                        ),
                        "ci_supports_non_worsening": bool(
                            ci_high <= 0.0 if lower else ci_low >= 0.0
                        ),
                    }
                )
    return output


def _audit(
    config: QualityAwareMAPPOConfig,
    seed_runs: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    treatment_means: Sequence[Mapping[str, Any]],
    cross_seed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation_rows = [row for row in cross_seed if str(row.get("split")) == "validation"]
    primary_comparison = "budgeted_quality_aware_mappo_minus_baseline_mappo"
    primary_rows = [
        row for row in validation_rows if str(row.get("comparison")) == primary_comparison
    ]
    # Keep a clear fallback for callers that provide only the historical two
    # branches; the new three-branch experiment always uses the budgeted pair.
    if not primary_rows:
        primary_comparison = "quality_aware_mappo_minus_baseline_mappo"
        primary_rows = [
            row for row in validation_rows if str(row.get("comparison")) == primary_comparison
        ]
    by_metric = {str(row["metric"]): row for row in primary_rows}
    rmse = by_metric.get("position_rmse_norm_m", by_metric.get("position_rmse_m", {}))
    harmful = by_metric.get("harmful_fraction", {})
    j_com = by_metric.get("j_com", {})
    required = int(math.ceil(0.8 * len(config.seeds)))
    structural = all(bool(row.get("validation_reports_ok", True)) for row in seed_runs)
    structural = structural and all(
        bool(row.get("branches", {}).get(treatment, {}).get("validation_report_ok", False))
        for row in seed_runs
        for treatment in TREATMENTS
    )
    variant_ok = all(
        str(row.get("branches", {}).get("baseline_mappo", {}).get("mappo_variant")) == "baseline_mappo"
        and str(row.get("branches", {}).get("quality_aware_mappo", {}).get("mappo_variant")) == "quality_aware_mappo"
        and str(row.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get("mappo_variant"))
        == "budgeted_quality_aware_mappo"
        for row in seed_runs
    )
    feature_ok = all(
        not bool(row.get("branches", {}).get("baseline_mappo", {}).get("quality_features_enabled", True))
        and bool(row.get("branches", {}).get("quality_aware_mappo", {}).get("quality_features_enabled", False))
        and bool(
            row.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get(
                "quality_features_enabled", False
            )
        )
        for row in seed_runs
    )
    budget_ok = all(
        not bool(row.get("branches", {}).get("baseline_mappo", {}).get("measurement_budget_enabled", True))
        and not bool(row.get("branches", {}).get("quality_aware_mappo", {}).get("measurement_budget_enabled", True))
        and bool(
            row.get("branches", {}).get("budgeted_quality_aware_mappo", {}).get(
                "measurement_budget_enabled", False
            )
        )
        and all(
            int(value) >= 0
            for value in (row.get("measurement_budgets", {}) or {}).values()
        )
        for row in seed_runs
    )
    overhead_ok = all(
        bool(row.get("branches", {}).get(treatment, {}).get("measurement_overhead_enabled", False))
        == bool(config.measurement_overhead_enabled)
        for row in seed_runs
        for treatment in TREATMENTS
    )
    paired_counts_ok = all(
        int(row.get("episode_count", 0))
        == (config.validation_episodes if str(row.get("split")) == "validation" else config.test_episodes)
        for row in treatment_means
    )
    finite = all(
        np.isfinite(_finite(row.get(metric)))
        for row in episode_rows
        for metric in _METRICS
    )
    stable = (
        len(config.seeds) >= 5
        and int(rmse.get("improved_seed_count", 0)) >= required
        and int(harmful.get("not_worsened_seed_count", 0)) >= required
        and bool(rmse.get("ci_supports_improvement", False))
        and bool(harmful.get("ci_supports_non_worsening", False))
        and bool(j_com.get("ci_supports_non_worsening", False))
    )
    decision_supported = bool(
        structural
        and variant_ok
        and feature_ok
        and budget_ok
        and overhead_ok
        and paired_counts_ok
        and finite
        and stable
    )
    return {
        "experiment": "budgeted_quality_aware_mappo_vs_baseline_multiseed",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evidence_status": "engineering_integration_validation_only",
        "uav_count": config.uav_count,
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_used_for_selection": False,
        "test_data_generated_and_sealed": True,
        "quality_fit_source": "warmup_actor_visible_observations_only",
        "forbidden_quality_inputs": [
            "true_position",
            "true_error",
            "physical_los_nlos",
            "future_state",
            "future_packet_outcome",
        ],
        "checks": {
            "seed_count": len(config.seeds),
            "required_improved_seed_count": required,
            "all_validation_reports_ok": structural,
            "branch_variants_correct": variant_ok,
            "quality_feature_branch_only": feature_ok,
            "budget_branch_correct": budget_ok,
            "measurement_overhead_matches_config": overhead_ok,
            "paired_episode_counts_ok": paired_counts_ok,
            "all_episode_metrics_finite": finite,
            "warmup_quality_models_identical_within_seed": all(
                bool(row.get("warmup_quality_models_identical", False)) for row in seed_runs
            ),
            "branch_configs_equal_except_variant": all(
                bool(row.get("branch_configs_equal_except_variant", False)) for row in seed_runs
            ),
            "rmse_improved_seed_count": int(rmse.get("improved_seed_count", 0)),
            "harmful_fraction_improved_seed_count": int(harmful.get("improved_seed_count", 0)),
            "harmful_fraction_not_worsened_seed_count": int(
                harmful.get("not_worsened_seed_count", 0)
            ),
            "primary_comparison": primary_comparison,
            "rmse_validation_ci": [
                float(rmse.get("bootstrap_95_ci_low", float("nan"))),
                float(rmse.get("bootstrap_95_ci_high", float("nan"))),
            ],
            "harmful_fraction_validation_ci": [
                float(harmful.get("bootstrap_95_ci_low", float("nan"))),
                float(harmful.get("bootstrap_95_ci_high", float("nan"))),
            ],
            "j_com_validation_ci": [
                float(j_com.get("bootstrap_95_ci_low", float("nan"))),
                float(j_com.get("bootstrap_95_ci_high", float("nan"))),
            ],
        },
        "decision_gate": (
            "budgeted_quality_aware_mappo_supported_across_seeds"
            if decision_supported
            else "budgeted_quality_aware_mappo_not_yet_supported_across_seeds"
        ),
        "audit_ok": bool(
            structural
            and variant_ok
            and feature_ok
            and budget_ok
            and overhead_ok
            and paired_counts_ok
            and finite
        ),
        "decision_supported": decision_supported,
        "interpretation_boundary": (
            "This experiment tests whether actor-visible quality features plus a fixed train-fitted measurement "
            "budget improve the measurement decision under queue overhead. "
            "It does not establish superiority for 5/7 UAVs, collision safety, energy trade-offs, or a final "
            "paper claim until the multi-seed evidence gate is satisfied."
        ),
    }


def _plot_comparison(
    treatment_means: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "quality_aware_mappo": "#16a34a",
        "budgeted_quality_aware_mappo": "#dc2626",
    }
    labels = {
        "baseline_mappo": "Baseline MAPPO",
        "quality_aware_mappo": "Quality-aware MAPPO",
        "budgeted_quality_aware_mappo": "Budgeted quality-aware MAPPO",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, _PLOT_METRICS):
        width = min(0.8 / len(TREATMENTS), 0.24)
        offsets = (np.arange(len(TREATMENTS), dtype=float) - (len(TREATMENTS) - 1) / 2.0) * width
        for offset, treatment in zip(offsets, TREATMENTS):
            selected = [
                row
                for row in treatment_means
                if str(row.get("split")) == "validation" and str(row.get("treatment")) == treatment
            ]
            values = np.asarray([_finite(row.get(metric)) for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if values.size else 0.0
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            axis.bar(
                [0 + offset],
                [mean],
                width=width,
                yerr=[std],
                capsize=4,
                color=colors[treatment],
                label=labels[treatment],
            )
            selected_test = [
                row
                for row in treatment_means
                if str(row.get("split")) == "test" and str(row.get("treatment")) == treatment
            ]
            test_values = np.asarray([_finite(row.get(metric)) for row in selected_test], dtype=float)
            test_values = test_values[np.isfinite(test_values)]
            test_mean = float(np.mean(test_values)) if test_values.size else 0.0
            axis.bar(
                [1 + offset],
                [test_mean],
                width=width,
                color=colors[treatment],
                alpha=0.45,
                hatch="//",
            )
        axis.set_title(metric)
        axis.set_xticks([0, 1], ["validation", "test (sealed)"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Baseline vs quality-aware MAPPO (seed-level means; test is sealed)")
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / "quality_aware_mappo_comparison.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def run_quality_aware_mappo(config: QualityAwareMAPPOConfig) -> dict[str, Any]:
    """Run the three-branch, independent-seed budgeted MAPPO experiment."""

    root, logs_dir, plots_dir = _prepare_batch_output(config)
    if not Path(config.scenario_profile_path).exists():
        raise FileNotFoundError(f"scenario profile not found: {config.scenario_profile_path}")
    if not Path(config.ekf_config_path).exists():
        raise FileNotFoundError(f"EKF config not found: {config.ekf_config_path}")

    seed_runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_run = _run_seed_branches(config, int(seed), root)
        seed_runs.append(seed_run)
        for treatment in TREATMENTS:
            branch_scale = root / f"seed_{int(seed)}" / treatment / "scale_3"
            episode_rows.extend(_episode_rows_for_branch(branch_scale, treatment, int(seed)))

    treatment_means = _seed_treatment_means(episode_rows)
    contrasts = _paired_contrasts(episode_rows)
    cross_seed = _cross_seed_summary(contrasts, config)
    audit = _audit(config, seed_runs, episode_rows, treatment_means, cross_seed)
    plot = _plot_comparison(treatment_means, plots_dir)

    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "treatment_means.csv", treatment_means)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "cross_seed_summary.csv", cross_seed)
    _write_table(logs_dir / "seed_runs.csv", seed_runs)
    _write_json(logs_dir / "audit_report.json", audit)
    manifest = {
        "dataset_id": f"python_sim_quality_aware_mappo_{config.seeds[0]}",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_integration_validation_only",
        "uav_count": config.uav_count,
        "seed_set": [int(seed) for seed in config.seeds],
        "training": {
            "warmup_episodes": config.warmup_episodes,
            "train_episodes": config.train_episodes,
            "validation_episodes": config.validation_episodes,
            "test_episodes_generated_but_sealed": config.test_episodes,
        },
        "measurement_overhead": {
            "enabled": bool(config.measurement_overhead_enabled),
            "equivalent_packet_size_bytes": 512,
            "one_equivalent_packet_per_selected_measurement": True,
            "mac_protocol_modeled": False,
            "automatic_retransmission_modeled": False,
        },
        "branches": {
            "baseline_mappo": "original actor; no measurement quality input",
            "quality_aware_mappo": "measurement head receives warm-up-fitted risk and validity vectors only",
            "budgeted_quality_aware_mappo": (
                "quality-aware measurement head plus remaining train-fitted budget feature; "
                "environment masks exhausted candidates"
            ),
        },
        "quality_rule": {
            "fit_source": "warmup actor-visible observation snapshots per seed",
            "weights": "frozen equal weights",
            "forbidden_inputs": audit["forbidden_quality_inputs"],
            "bandwidth_velocity_ekf_gat_critic_reward_unchanged": True,
        },
        "independent_unit": "base training seed",
        "test_data_used_for_selection": False,
        "outputs": {
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "treatment_means": str(logs_dir / "treatment_means.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "cross_seed_summary": str(logs_dir / "cross_seed_summary.csv"),
            "seed_runs": str(logs_dir / "seed_runs.csv"),
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
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
        "comparison_plot": str(plot),
        "audit_report": str(logs_dir / "audit_report.json"),
    }


__all__ = [
    "DEFAULT_EKF_CONFIG",
    "DEFAULT_PROFILE",
    "DEFAULT_SEEDS",
    "QualityAwareMAPPOConfig",
    "run_quality_aware_mappo",
]
