"""Communication-cost constrained proxy-guided MAPPO experiment.

The baseline training split is the sole source of the per-UAV communication
cost targets, proxy scales, measurement budgets and shared quality model.
Validation drives the pre-registered decision gate; test is generated and
sealed but is never read by the analysis in this module.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .baseline import _read_csv, _read_json
from .execution_uncertainty_proxy_audit import (
    ProxyScales,
    _candidate_rows_from_episode,
    _materialize_proxy_rows,
)
from .formal import fit_shared_training_metric_scales, load_selected_ekf_config, run_preliminary
from .localization_coupling_intervention import _bootstrap_interval
from .measurement_rate_controlled_mappo import _fit_measurement_budgets, _fit_measurement_rate_targets
from .proxy_guided_adaptive_budgeted_quality_aware_mappo import (
    BASELINE_BRANCH,
    PROXY_BRANCH,
    ProxyGuidedAdaptiveMAPPOConfig,
    _formal_config,
    _train_proxy_model,
)
from .quality_aware_mappo import (
    DEFAULT_EKF_CONFIG, DEFAULT_PROFILE, _event_summary, _finite,
    _measurement_budget_summary, _position_error_p95, _quality_summary,
    _read_jsonl, _write_table,
)
from .metrics import position_error_metrics


COMMUNICATION_BRANCH = "communication_constrained_proxy_guided_mappo"
BRANCHES = (BASELINE_BRANCH, PROXY_BRANCH, COMMUNICATION_BRANCH)
DEFAULT_CONFIRMATORY_SEEDS = (12136, 13145, 14154, 15163, 16172)
SHARED_INITIALIZATION_MODULES = (
    "gat",
    "actor_body",
    "bandwidth_head",
    "delta_mean_head",
    "delta_logstd",
    "critic",
)
METRICS = (
    "mean_reward", "position_rmse_m", "position_rmse_norm_m", "position_p95_norm_m",
    "position_error_p95_m", "j_loc", "j_com",
    "joint_objective", "harmful_fraction", "beneficial_fraction", "neutral_fraction",
    "accepted_update_count", "update_coverage_rate", "update_coverage_any",
    "accepted_update_rate", "accepted_updates_per_window", "beneficial_per_window",
    "harmful_per_window", "no_update_episode_count", "j_loc_saturation_fraction",
    "packet_loss", "delay_s", "aoi_s", "connectivity",
    "measurement_count", "measurement_overhead_packets", "measurement_budget_usage_rate",
)


def _mean(values: Sequence[Any]) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    shifted = array - np.max(array)
    result = np.exp(shifted)
    return result / max(float(np.sum(result)), 1e-12)


def _timestamp(value: Any) -> float:
    return round(float(value), 9)


@dataclass(frozen=True)
class CommunicationConstrainedProxyMAPPOConfig:
    output_dir: Path
    seeds: tuple[int, ...] = DEFAULT_CONFIRMATORY_SEEDS
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
    visual_odometry_quality_calibration_enabled: bool = False
    measurement_quality_logit_penalty: float = 1.0
    proxy_teacher_coef: float = 0.25
    proxy_temperature: float = 0.25
    communication_cost_weight: float = 1.0
    communication_cost_tolerance: float = 0.05
    communication_dual_initial: float = 0.0
    communication_dual_step_size: float = 0.25
    communication_dual_max: float = 5.0
    fixed_packet_cost_fraction: float = 0.5
    # When enabled, every branch gets the same warm-up-only visual-odometry
    # covariance calibration through the shared FormalExperimentConfig.
    force: bool = False
    progress: bool = True
    # Isolate module-level initialization streams for fair paired branches.
    branch_invariant_initialization: bool = False
    # Optional timing-aware measurement mask/guard propagated to every formal
    # branch in a new run; disabled by default for historical compatibility.
    measurement_update_eligibility_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if min(self.max_steps, self.warmup_episodes, self.train_episodes,
               self.validation_episodes, self.test_episodes) < 1:
            raise ValueError("steps and episode counts must be positive")
        if self.device != "cpu":
            raise ValueError("communication-constrained experiment currently supports CPU only")
        if not isinstance(self.visual_odometry_quality_calibration_enabled, bool):
            raise ValueError("visual_odometry_quality_calibration_enabled must be boolean")
        if not isinstance(self.branch_invariant_initialization, bool):
            raise ValueError("branch_invariant_initialization must be boolean")
        if not isinstance(self.measurement_update_eligibility_enabled, bool):
            raise ValueError("measurement_update_eligibility_enabled must be boolean")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        for name, value in (
            ("measurement_quality_logit_penalty", self.measurement_quality_logit_penalty),
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
        if self.proxy_temperature <= 0.0 or self.communication_dual_step_size <= 0.0:
            raise ValueError("temperature and dual step must be positive")
        if self.communication_dual_max <= 0.0 or self.communication_dual_initial > self.communication_dual_max:
            raise ValueError("invalid communication dual bounds")
        if not 0.0 <= self.communication_cost_tolerance <= 1.0:
            raise ValueError("communication cost tolerance must be in [0, 1]")
        if not 0.0 <= self.fixed_packet_cost_fraction <= 1.0:
            raise ValueError("fixed packet cost fraction must be in [0, 1]")


@dataclass(frozen=True)
class MeasurementCommunicationCostAuditConfig:
    reference_root: Path
    output_dir: Path
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = (7071, 8080, 9099, 10118, 11127)
    fixed_packet_cost_fraction: float = 0.5
    force: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not 0.0 <= float(self.fixed_packet_cost_fraction) <= 1.0:
            raise ValueError("fixed packet cost fraction must be in [0, 1]")


def _prepare(root: Path, logs: Path, plots: Path, force: bool) -> None:
    if force:
        for path in (root, logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (root, logs, plots)):
        raise FileExistsError("output is not empty; use --force")
    for path in (root, logs, plots):
        path.mkdir(parents=True, exist_ok=True)


def _proxy_scales(model: Mapping[str, Any]) -> ProxyScales:
    # Persistent-proxy calibration deliberately extends the shared scale
    # mapping with `persistent_ig_q05/q95`.  Communication-cost replay only
    # needs the legacy link-quality scales, so ignore those additive fields
    # rather than making the old cost path incompatible with the new model.
    allowed = {field.name for field in fields(ProxyScales)}
    scales = {
        key: float(value)
        for key, value in dict(model["scales"]).items()
        if key in allowed
    }
    missing = allowed.difference(scales)
    if missing:
        raise ValueError(f"proxy model is missing communication scales: {sorted(missing)}")
    return ProxyScales(**scales)


def _candidate_cost_map(
    scale_root: Path,
    seed: int,
    split: str,
    proxy_model: Mapping[str, Any],
    fixed_fraction: float,
) -> dict[tuple[str, float, str, str], float]:
    from ..learning.measurement_cost import candidate_communication_cost
    raw: list[dict[str, Any]] = []
    for episode in sorted(path for path in (scale_root / split).iterdir() if path.is_dir()):
        raw.extend(_candidate_rows_from_episode(
            episode, base_seed=seed, treatment=BASELINE_BRANCH, split=split,
            include_offline_label=False,
            age_variance_coefficient=float(proxy_model.get("age_variance_coefficient", 1.0)),
        ))
    rows = _materialize_proxy_rows(
        raw, _proxy_scales(proxy_model), float(proxy_model.get("age_variance_coefficient", 1.0))
    )
    return {
        (str(row["episode_id"]), _timestamp(row["timestamp"]), str(row["uav_id"]), str(row["source_uav_id"])):
        float(candidate_communication_cost(1.0 - _finite(row["link_quality_q"], 0.0), fixed_fraction))
        for row in rows
    }


def fit_measurement_communication_cost_targets(
    baseline_scale: Path,
    *,
    seed: int,
    proxy_model: Mapping[str, Any],
    fixed_packet_cost_fraction: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    costs = _candidate_cost_map(
        baseline_scale, seed, "train", proxy_model, fixed_packet_cost_fraction
    )
    values: dict[str, list[float]] = {f"uav_{index:02d}": [] for index in range(3)}
    selected_counts = {uav: 0 for uav in values}
    missing_visible_cost_counts = {uav: 0 for uav in values}
    for episode in sorted(path for path in (baseline_scale / "train").iterdir() if path.is_dir()):
        for row in _read_csv(episode / "actions.csv"):
            uav = str(row["uav_id"])
            selected = str(row.get("measurement_uav_id", "none"))
            value = 0.0
            if selected != "none":
                key = (str(row["episode_id"]), _timestamp(row["timestamp"]), uav, selected)
                # Historical baseline logs can contain a selected action whose
                # actor-visible candidate record is absent. Match the live
                # proxy path: missing visible link quality receives maximal
                # finite risk/cost rather than any truth-based fallback.
                if key not in costs:
                    value = 1.0
                    missing_visible_cost_counts[uav] += 1
                else:
                    value = costs[key]
                selected_counts[uav] += 1
            values[uav].append(float(value))
    targets = {uav: float(np.mean(items)) if items else 0.0 for uav, items in values.items()}
    rows = [{
        "base_seed": int(seed), "uav_id": uav, "source_branch": BASELINE_BRANCH,
        "source_split": "train", "decision_windows": len(values[uav]),
        "selected_measurements": selected_counts[uav],
        "selected_measurements_with_missing_visible_cost": missing_visible_cost_counts[uav],
        "fixed_packet_cost_fraction": float(fixed_packet_cost_fraction),
        "communication_cost_target": targets[uav],
        "validation_used_for_fitting": 0, "test_used_for_fitting": 0,
    } for uav in sorted(values)]
    return targets, rows


def _base_config(config: CommunicationConstrainedProxyMAPPOConfig) -> ProxyGuidedAdaptiveMAPPOConfig:
    return ProxyGuidedAdaptiveMAPPOConfig(
        output_dir=config.output_dir, seeds=config.seeds, max_steps=config.max_steps,
        warmup_episodes=config.warmup_episodes, train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes, test_episodes=config.test_episodes,
        scenario_profile_path=config.scenario_profile_path, ekf_config_path=config.ekf_config_path,
        logs_dir=config.logs_dir, plots_dir=config.plots_dir,
        bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.bootstrap_seed,
        device=config.device, measurement_quality_logit_penalty=config.measurement_quality_logit_penalty,
        visual_odometry_quality_calibration_enabled=(
            bool(config.visual_odometry_quality_calibration_enabled)
        ),
        proxy_teacher_coef=config.proxy_teacher_coef, proxy_temperature=config.proxy_temperature,
        communication_cost_weight=config.communication_cost_weight, force=False, progress=config.progress,
        branch_invariant_initialization=bool(config.branch_invariant_initialization),
    )


def _initialization_hash_audit(seed_root: Path) -> dict[str, Any]:
    """Compare construction-time hashes for modules shared by all branches."""

    details: dict[str, Any] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for branch in BRANCHES:
        path = seed_root / branch / "scale_3" / "diagnostics" / "initial_parameter_hashes.json"
        if not path.is_file():
            details[branch] = {"available": False, "path": str(path), "modules": {}}
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, TypeError):
            payload = {}
        modules = payload.get("modules", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(modules, Mapping):
            modules = {}
        payloads[branch] = payload if isinstance(payload, Mapping) else {}
        details[branch] = {
            "available": bool(set(SHARED_INITIALIZATION_MODULES).issubset(modules)),
            "path": str(path),
            "branch_invariant_initialization": bool(
                payload.get("branch_invariant_initialization", False)
            ),
            "parameter_initialization_seed": payload.get(
                "parameter_initialization_seed"
            ),
            "modules": {str(key): str(value) for key, value in modules.items()},
        }
    shared_equal = bool(payloads) and all(
        details.get(branch, {}).get("available", False) for branch in BRANCHES
    )
    if shared_equal:
        reference = payloads[BRANCHES[0]].get("modules", {})
        shared_equal = all(
            all(
                payloads[branch].get("modules", {}).get(module)
                == reference.get(module)
                for module in SHARED_INITIALIZATION_MODULES
            )
            for branch in BRANCHES[1:]
        )
    return {"equal": bool(shared_equal), "details": details}


def _communication_formal(
    config: CommunicationConstrainedProxyMAPPOConfig, *, seed: int, output_dir: Path,
    ekf: Mapping[str, float], budgets: Mapping[str, int], proxy_model: Mapping[str, Any],
    targets: Mapping[str, float], metric_scales: Mapping[str, float],
):
    proxy = _formal_config(
        _base_config(config), seed=seed, output_dir=output_dir, ekf_parameters=ekf,
        treatment=PROXY_BRANCH, budgets=budgets, targets={uav: 0.0 for uav in budgets},
        proxy_model=proxy_model, metric_scales=metric_scales,
    )
    return replace(
        proxy, mappo_variant=COMMUNICATION_BRANCH,
        measurement_rate_constraint_enabled=False, measurement_rate_target_by_uav=None,
        measurement_rate_dual_enabled=False,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav=dict(targets),
        measurement_communication_cost_tolerance=config.communication_cost_tolerance,
        measurement_communication_dual_initial=config.communication_dual_initial,
        measurement_communication_dual_step_size=config.communication_dual_step_size,
        measurement_communication_dual_max=config.communication_dual_max,
        measurement_communication_fixed_packet_fraction=config.fixed_packet_cost_fraction,
        # The real communication constraint still evaluates the full cost;
        # only this detached teacher charges cost above the baseline-train
        # target so the fixed packet floor is not penalized twice.
        measurement_proxy_target_relative_cost=True,
        command_line="run-communication-constrained-proxy-mappo",
    )


def _run_seed(config: CommunicationConstrainedProxyMAPPOConfig, seed: int, root: Path) -> dict[str, Any]:
    ekf = load_selected_ekf_config(config.ekf_config_path)
    seed_root = root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    base = _base_config(config)
    template = _formal_config(base, seed=seed, output_dir=seed_root / "metric_probe", ekf_parameters=ekf, treatment=BASELINE_BRANCH)
    metric_scales = fit_shared_training_metric_scales(template, (3,))
    baseline_formal = replace(template, output_dir=seed_root / BASELINE_BRANCH, metric_scales=metric_scales)
    if config.progress: print(f"[communication-constrained] seed {seed}: {BASELINE_BRANCH}", flush=True)
    run_preliminary(baseline_formal)
    baseline_scale = seed_root / BASELINE_BRANCH / "scale_3"
    baseline_report = validate_episode_benchmark(baseline_scale)
    if not baseline_report.get("ok", False): raise RuntimeError(f"baseline validation failed: {baseline_report}")
    quality_model = _read_json(baseline_scale / "calibration" / "measurement_quality_model.json")
    budgets = _fit_measurement_budgets(baseline_scale, uav_count=3)
    rate_targets, _rate_rows = _fit_measurement_rate_targets(baseline_scale, uav_count=3, base_seed=seed)
    proxy_model = _train_proxy_model(baseline_scale, seed, 1.0)
    communication_targets, target_rows = fit_measurement_communication_cost_targets(
        baseline_scale, seed=seed, proxy_model=proxy_model,
        fixed_packet_cost_fraction=config.fixed_packet_cost_fraction,
    )
    reports = {BASELINE_BRANCH: baseline_report}
    proxy_formal = _formal_config(
        base, seed=seed, output_dir=seed_root / PROXY_BRANCH, ekf_parameters=ekf,
        treatment=PROXY_BRANCH, budgets=budgets, targets=rate_targets,
        proxy_model=proxy_model, metric_scales=metric_scales,
    )
    communication_formal = _communication_formal(
        config, seed=seed, output_dir=seed_root / COMMUNICATION_BRANCH, ekf=ekf,
        budgets=budgets, proxy_model=proxy_model, targets=communication_targets,
        metric_scales=metric_scales,
    )
    for branch, formal in ((PROXY_BRANCH, proxy_formal), (COMMUNICATION_BRANCH, communication_formal)):
        if config.progress: print(f"[communication-constrained] seed {seed}: {branch}", flush=True)
        run_preliminary(formal, measurement_quality_model=quality_model, measurement_proxy_model=proxy_model)
        report = validate_episode_benchmark(seed_root / branch / "scale_3")
        if not report.get("ok", False): raise RuntimeError(f"{branch} validation failed: {report}")
        reports[branch] = report
    initialization_audit = _initialization_hash_audit(seed_root)
    return {
        "base_seed": seed, "budgets": budgets, "communication_targets": communication_targets,
        "communication_target_rows": target_rows, "proxy_model": proxy_model,
        "episode_reports": reports, "test_data_used_for_selection": False,
        "shared_initialization_hashes_equal": bool(initialization_audit["equal"]),
        "initialization_hash_audit": initialization_audit["details"],
    }


def _seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in rows}):
        for branch in BRANCHES:
            selected = [row for row in rows if int(row["base_seed"]) == seed and row["split"] == "validation" and row["treatment"] == branch]
            output.append({"base_seed": seed, "split": "validation", "treatment": branch,
                           "episode_count": len(selected), **{metric: _mean([row.get(metric) for row in selected]) for metric in METRICS}})
    return output


def _validation_episode_rows(branch_scale: Path, treatment: str, base_seed: int) -> list[dict[str, Any]]:
    """Read only split-specific validation diagnostics; test stays sealed."""

    metrics = _read_csv(branch_scale / "diagnostics" / "validation_episode_metrics.csv")
    events = _read_jsonl(branch_scale / "diagnostics" / "validation_cooperative_updates.jsonl")
    quality = _read_jsonl(branch_scale / "diagnostics" / "validation_measurement_quality.jsonl")
    output: list[dict[str, Any]] = []
    for row in metrics:
        episode_id = str(row["episode_id"])
        episode_root = branch_scale / "validation" / episode_id
        event_rows = [item for item in events if str(item.get("episode_id")) == episode_id]
        quality_rows = [item for item in quality if str(item.get("episode_id")) == episode_id]
        position_metrics = position_error_metrics(
            _read_csv(episode_root / "ground_truth.csv"),
            _read_csv(episode_root / "estimates.csv"),
        )
        value = dict(row)
        value.update({
            "base_seed": int(base_seed), "treatment": treatment,
            "position_error_p95_m": _position_error_p95(episode_root),
            "position_rmse_norm_m": _finite(
                row.get("position_rmse_norm_m"),
                position_metrics.get("rmse_norm_m", 0.0),
            ),
            "position_p95_norm_m": _finite(
                row.get("position_p95_norm_m"),
                position_metrics.get("p95_norm_m", 0.0),
            ),
            **_event_summary(event_rows), **_quality_summary(quality_rows),
            **_measurement_budget_summary(episode_root),
        })
        # Backfill metrics introduced after the historical formal runs.  The
        # derivation is read-only and uses only this episode's rewards/events.
        reward_rows = _read_csv(episode_root / "rewards.csv")
        accepted = max(0.0, _finite(value.get("accepted_update_count"), 0.0))
        uav_count = max(1, int(_finite(value.get("uav_count"), 3.0)))
        decision_windows = max(1.0, round(len(reward_rows) / uav_count))
        uncertainty = np.asarray(
            [_finite(item.get("normalized_uncertainty"), float("nan")) for item in reward_rows],
            dtype=float,
        )
        uncertainty = uncertainty[np.isfinite(uncertainty)]
        value.setdefault(
            "j_loc_saturation_fraction",
            float(np.mean(uncertainty >= 1.0 - 1e-12)) if uncertainty.size else 0.0,
        )
        value.setdefault("update_coverage_any", float(accepted > 0.0))
        value.setdefault("accepted_update_rate", accepted / max(1.0, decision_windows * uav_count))
        value.setdefault("accepted_updates_per_window", accepted / decision_windows)
        value.setdefault(
            "beneficial_per_window",
            max(0.0, _finite(value.get("beneficial_fraction"), 0.0) * accepted) / decision_windows,
        )
        value.setdefault(
            "harmful_per_window",
            max(0.0, _finite(value.get("harmful_fraction"), 0.0) * accepted) / decision_windows,
        )
        value.setdefault("no_update_episode_count", float(accepted <= 0.0))
        output.append(value)
    return output


def _contrasts(seed_rows: Sequence[Mapping[str, Any]], config: CommunicationConstrainedProxyMAPPOConfig) -> list[dict[str, Any]]:
    index = {(int(row["base_seed"]), str(row["treatment"])): row for row in seed_rows}
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for branch in BRANCHES[1:]:
        for metric in METRICS:
            diffs = np.asarray([
                _finite(index[(seed, branch)][metric]) - _finite(index[(seed, BASELINE_BRANCH)][metric])
                for seed in config.seeds
            ], dtype=float)
            lower, upper = _bootstrap_interval(diffs, rng=rng, samples=config.bootstrap_samples)
            output.append({
                "comparison": f"{branch}_minus_{BASELINE_BRANCH}", "metric": metric,
                "seed_count": len(diffs), "mean_difference": float(np.mean(diffs)),
                "ci95_lower": lower, "ci95_upper": upper,
                "improved_seed_count": int(np.sum(diffs < 0.0)) if metric not in {"mean_reward", "beneficial_fraction", "connectivity"} else int(np.sum(diffs > 0.0)),
            })
    return output


def _validation_cost_rows(root: Path, runs: Sequence[Mapping[str, Any]], fixed: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        seed = int(run["base_seed"]); model = run["proxy_model"]
        for branch in BRANCHES:
            scale = root / f"seed_{seed}" / branch / "scale_3"
            costs = _candidate_cost_map(scale, seed, "validation", model, fixed)
            logs = _read_jsonl(scale / "diagnostics" / "validation_measurement_quality.jsonl")
            action_maps: dict[str, dict[tuple[float, str], Mapping[str, Any]]] = {}
            for episode in sorted(path for path in (scale / "validation").iterdir() if path.is_dir()):
                action_maps[episode.name] = {
                    (_timestamp(item["timestamp"]), str(item["uav_id"])): item
                    for item in _read_csv(episode / "actions.csv")
                }
            by_uav: dict[str, list[tuple[float, float]]] = {f"uav_{i:02d}": [] for i in range(3)}
            for row in logs:
                if str(row.get("split")) != "validation": continue
                episode = str(row["episode_id"]); time = _timestamp(row["timestamp"]); uav = str(row["uav_id"])
                logits = np.asarray(row.get("measurement_logits_adjusted", []), dtype=float)
                action = action_maps[episode][(time, uav)]
                mask_obj = json.loads(action["action_mask_json"]); mask = np.asarray(mask_obj["measurement"], dtype=bool)
                masked = np.where(mask, logits, -1e9); probs = _softmax(masked)
                candidate_costs = np.asarray([costs.get((episode, time, uav, f"uav_{i:02d}"), 1.0) for i in range(3)])
                expected = float(np.sum(probs[:3] * candidate_costs))
                selected = str(action.get("measurement_uav_id", "none"))
                actual = 0.0 if selected == "none" else costs.get((episode, time, uav, selected), 1.0)
                by_uav[uav].append((expected, actual))
            for uav, values in by_uav.items():
                target = float(run["communication_targets"][uav])
                output.append({"base_seed": seed, "treatment": branch, "split": "validation", "uav_id": uav,
                               "decision_windows": len(values), "target": target,
                                "expected_communication_cost": _mean([item[0] for item in values]),
                                "actual_communication_cost": _mean([item[1] for item in values]),
                                "expected_cost_excess": _mean([item[0] for item in values]) - target,
                                "actual_cost_excess": _mean([item[1] for item in values]) - target,
                                # Historical name retained for consumers; it
                                # means the expected-proxy check only.
                                "within_target_plus_tolerance": int(_mean([item[0] for item in values]) <= target + 0.05 + 1e-12),
                                "within_expected_target_plus_tolerance": int(_mean([item[0] for item in values]) <= target + 0.05 + 1e-12),
                                "within_actual_target_plus_tolerance": int(_mean([item[1] for item in values]) <= target + 0.05 + 1e-12)})
    return output


def _plot(seed_rows: Sequence[Mapping[str, Any]], cost_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {BASELINE_BRANCH: "#356AE6", PROXY_BRANCH: "#18A558", COMMUNICATION_BRANCH: "#D62728"}
    labels = {
        BASELINE_BRANCH: "Local-EKF-GAT-MAPPO (Proposed Method)",
        PROXY_BRANCH: "Proxy-guided historical ablation",
        COMMUNICATION_BRANCH: "Deprecated communication-constrained branch",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, metric, title in zip(
        axes.flat[:5],
        ("position_rmse_norm_m", "j_com", "mean_reward", "harmful_fraction", "measurement_count"),
        ("Position RMSE (vector norm)", "$J_{com}$", "Mean team reward", "Harmful update fraction", "Measurement count"),
    ):
        for branch in BRANCHES:
            selected = sorted([row for row in seed_rows if row["treatment"] == branch], key=lambda row: int(row["base_seed"]))
            # Seed identifiers are nominal, not an ordered time axis.  Use
            # offset points rather than connecting them with a line.
            positions = np.arange(len(selected), dtype=float) + {
                BASELINE_BRANCH: -0.22,
                PROXY_BRANCH: 0.0,
                COMMUNICATION_BRANCH: 0.22,
            }[branch]
            axis.scatter(positions, [row[metric] for row in selected], color=colors[branch], label=labels[branch])
        axis.set_title(title); axis.set_xlabel("Training seed"); axis.set_xticks(np.arange(len(config_seeds := sorted({int(row["base_seed"]) for row in seed_rows})))); axis.set_xticklabels([str(seed) for seed in config_seeds]); axis.grid(alpha=0.25)
    axis = axes.flat[5]
    selected = [row for row in cost_rows if row["treatment"] == COMMUNICATION_BRANCH]
    axis.scatter([row["target"] for row in selected], [row["expected_communication_cost"] for row in selected], color=colors[COMMUNICATION_BRANCH])
    line = np.linspace(0, 1, 100); axis.plot(line, line + 0.05, "k--", linewidth=1, label="target + 0.05")
    axis.set_title("Expected communication cost constraint"); axis.set_xlabel("Baseline-train target"); axis.set_ylabel("Validation expected cost"); axis.grid(alpha=0.25)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=200); plt.close(fig)


def run_communication_constrained_proxy_mappo(config: CommunicationConstrainedProxyMAPPOConfig) -> dict[str, Any]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir else Path("logs/preliminary") / root.name
    plots = Path(config.plots_dir) if config.plots_dir else Path("plots/preliminary") / root.name
    _prepare(root, logs, plots, config.force)
    runs = [_run_seed(config, int(seed), root) for seed in config.seeds]
    episode_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in BRANCHES:
            episode_rows.extend(_validation_episode_rows(root / f"seed_{seed}" / branch / "scale_3", branch, seed))
    validation_episode_rows = episode_rows
    seed_rows = _seed_metrics(validation_episode_rows)
    contrasts = _contrasts(seed_rows, config)
    cost_rows = _validation_cost_rows(root, runs, config.fixed_packet_cost_fraction)
    target_rows = [row for run in runs for row in run["communication_target_rows"]]
    training_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        path = root / f"seed_{seed}" / COMMUNICATION_BRANCH / "scale_3" / "diagnostics" / "training_metrics.jsonl"
        training_rows.extend({"base_seed": seed, **row} for row in _read_jsonl(path))
    contrast_index = {(row["comparison"], row["metric"]): row for row in contrasts}
    comparison = f"{COMMUNICATION_BRANCH}_minus_{BASELINE_BRANCH}"
    rmse = contrast_index[(comparison, "position_rmse_norm_m")]
    jcom = contrast_index[(comparison, "j_com")]
    harmful = contrast_index[(comparison, "harmful_fraction")]
    reward = contrast_index[(comparison, "mean_reward")]
    expected_seed_constraint_ok = 0
    actual_seed_constraint_ok = 0
    for seed in config.seeds:
        selected = [row for row in cost_rows if row["base_seed"] == seed and row["treatment"] == COMMUNICATION_BRANCH]
        expected_seed_constraint_ok += int(
            bool(selected)
            and all(row["within_expected_target_plus_tolerance"] for row in selected)
        )
        actual_seed_constraint_ok += int(
            bool(selected)
            and all(row["within_actual_target_plus_tolerance"] for row in selected)
        )
    initialization_ok = all(
        (not config.branch_invariant_initialization)
        or bool(run.get("shared_initialization_hashes_equal", False))
        for run in runs
    )
    structure_ok = all(run["episode_reports"][branch].get("ok", False) for run in runs for branch in BRANCHES)
    decision = bool(
        actual_seed_constraint_ok >= 4 and rmse["improved_seed_count"] >= 4 and rmse["ci95_upper"] < 0.0
        and jcom["ci95_upper"] <= 0.005 and harmful["ci95_upper"] <= 0.02
        and reward["mean_difference"] >= -0.005
        and initialization_ok
    )
    audit = {
        "experiment": "communication_constrained_proxy_mappo", "seed_set": list(config.seeds),
        "independent_unit": "training seed", "analysis_split": "validation", "test_data_used": False,
        "checks": {"episode_v1_3_all_branches": structure_ok, "train_only_targets": True,
                    "expected_communication_constraint_seed_count": expected_seed_constraint_ok,
                    "actual_communication_constraint_seed_count": actual_seed_constraint_ok,
                    "communication_constraint_seed_count": actual_seed_constraint_ok,
                    "shared_initialization_hashes_equal": initialization_ok,
                   "fixed_packet_cost_fraction": config.fixed_packet_cost_fraction,
                   "dual_bounds": [0.0, config.communication_dual_max], "none_cost": 0.0,
                    "measurement_payload_bytes": 512,
                    "initialization_hashes": [
                        run.get("initialization_hash_audit", {}) for run in runs
                    ]},
        "confirmatory_metrics": {"rmse": rmse, "j_com": jcom, "harmful_fraction": harmful, "mean_reward": reward},
         "audit_ok": bool(structure_ok and initialization_ok), "decision_supported": decision,
        "decision": "communication_constraint_supported" if decision else "communication_constraint_not_supported",
    }
    _write_table(logs / "measurement_communication_cost_targets.csv", target_rows)
    _write_table(logs / "communication_dual_training.csv", training_rows)
    _write_table(logs / "validation_communication_cost.csv", cost_rows)
    _write_table(logs / "episode_metrics.csv", validation_episode_rows)
    _write_table(logs / "seed_metrics.csv", seed_rows)
    _write_table(logs / "paired_contrasts.csv", contrasts)
    _write_json(logs / "audit_report.json", audit)
    plot = plots / "communication_constrained_proxy_mappo.png"; _plot(seed_rows, cost_rows, plot)
    _write_json(
        root / "manifest.json",
        {
            "dataset_id": "communication_constrained_proxy_mappo",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "branches": list(BRANCHES),
            "seed_set": list(config.seeds),
            "branch_invariant_initialization": bool(
                config.branch_invariant_initialization
            ),
            "test_data_used_for_selection": False,
            "audit": audit,
        },
    )
    return {"output_dir": str(root), "logs_dir": str(logs), "plots_dir": str(plots), "audit_ok": structure_ok, "decision_supported": decision, "audit_report": str(logs / "audit_report.json"), "plot": str(plot)}


def run_measurement_communication_cost_audit(config: MeasurementCommunicationCostAuditConfig) -> dict[str, Any]:
    logs = Path(config.output_dir); plots = Path(config.plots_dir) if config.plots_dir else logs.parent / (logs.name + "_plots")
    if config.force:
        for path in (logs, plots):
            if path.exists(): shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("audit output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, Any]] = []; candidate_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        scale = Path(config.reference_root) / f"seed_{seed}" / BASELINE_BRANCH / "scale_3"
        model_path = scale / "calibration" / "measurement_proxy_model.json"
        model = _read_json(model_path) if model_path.is_file() else _train_proxy_model(scale, seed, 1.0)
        fitted, rows = fit_measurement_communication_cost_targets(scale, seed=seed, proxy_model=model, fixed_packet_cost_fraction=config.fixed_packet_cost_fraction)
        targets.extend(rows)
        costs = _candidate_cost_map(scale, seed, "train", model, config.fixed_packet_cost_fraction)
        candidate_rows.extend({"base_seed": seed, "candidate_cost": value, "none_cost": 0.0} for value in costs.values())
    values = np.asarray([row["candidate_cost"] for row in candidate_rows], dtype=float)
    finite = bool(values.size and np.isfinite(values).all())
    lower = float(config.fixed_packet_cost_fraction); bounded = bool(finite and np.all(values >= lower - 1e-12) and np.all(values <= 1.0 + 1e-12))
    audit = {"experiment": "measurement_communication_cost_audit", "reference_root": str(config.reference_root), "seeds": list(config.seeds), "source_split": "baseline_mappo/train", "validation_used_for_fitting": False, "test_data_used": False, "candidate_count": len(candidate_rows), "checks": {"cost_finite": finite, "cost_in_fixed_fraction_to_one": bounded, "none_cost_zero": True, "target_recomputation_complete": len(targets) == 3 * len(config.seeds), "fixed_packet_cost_fraction": lower, "payload_bytes": 512}, "audit_ok": bool(finite and bounded and len(targets) == 3 * len(config.seeds))}
    _write_table(logs / "candidate_communication_cost.csv", candidate_rows); _write_table(logs / "measurement_communication_cost_targets.csv", targets); _write_json(logs / "audit_report.json", audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True); axis.hist(values, bins=30, color="#356AE6", alpha=0.85); axis.axvline(lower, color="black", linestyle="--", label="fixed packet floor"); axis.set(xlabel="Normalized communication cost", ylabel="Candidate count", title="Train-only execution-time communication cost"); axis.grid(alpha=0.25); axis.legend(); figure = plots / "measurement_communication_cost_audit.png"; fig.savefig(figure, dpi=200); plt.close(fig)
    return {"output_dir": str(logs), "plots_dir": str(plots), "audit_ok": audit["audit_ok"], "audit_report": str(logs / "audit_report.json"), "plot": str(figure)}


__all__ = [
    "BRANCHES", "COMMUNICATION_BRANCH", "CommunicationConstrainedProxyMAPPOConfig",
    "MeasurementCommunicationCostAuditConfig", "fit_measurement_communication_cost_targets",
    "run_communication_constrained_proxy_mappo", "run_measurement_communication_cost_audit",
]
