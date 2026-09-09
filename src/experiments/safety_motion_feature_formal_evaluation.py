"""Five-policy-seed safety evaluation for the 37/41-dimensional ablation.

This evaluator uses the ten previously frozen safety scenes as a hold-out
population.  The five policy seeds are averaged within each scene; all
inferential contrasts therefore use the scene seed as the independent unit.
No training update is called and the formal training ``test`` directories are
never opened.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _jsonable
from .formal import _make_environment
from .policy_mechanism_audit import _formal_config, _load_trainer
from .safety_mappo_evaluation import (
    BASELINE_BRANCH,
    DEFAULT_SCENARIO_SEEDS,
    _actor_observation_truth_free,
    _bootstrap_ci,
    _load_scenario,
    _state_and_covariance_finite,
)
from .safety_motion_feature_evaluation import _episode as _motion_episode
from .safety_motion_feature_formal_training import (
    DEFAULT_POLICY_SEEDS,
    FORMAL_BRANCHES,
)


FULL_BRANCH = "budgeted_quality_aware_41d"
EXPECTED_DIM = {
    "baseline_37d": 37,
    "baseline_41d": 41,
    "budgeted_quality_aware_41d": 41,
}
EXPECTED_DYNAMIC = {
    "baseline_37d": False,
    "baseline_41d": True,
    "budgeted_quality_aware_41d": True,
}
EXPECTED_VARIANT = {
    "baseline_37d": "baseline_mappo",
    "baseline_41d": "baseline_mappo",
    "budgeted_quality_aware_41d": "budgeted_quality_aware_mappo",
}


@dataclass(frozen=True)
class SafetyMotionFeatureFormalEvaluationConfig:
    """Configuration for the formal hold-out safety evaluation."""

    checkpoint_root: Path
    scenario_root: Path
    output_dir: Path = Path(
        "data/synthetic/safety_motion_feature_formal_evaluation_seed2026"
    )
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    branches: tuple[str, ...] = FORMAL_BRANCHES
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    measurement_overhead_enabled: bool = True
    quality_logit_penalty: float = 1.0
    # ``None`` reuses the value frozen in each checkpoint manifest.  A
    # corrected evaluation can set this explicitly to True without changing
    # the legacy evaluator's defaults.
    dynamic_obstacle_swept_projection_enabled: bool | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True
    command_line: str = ""

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("formal safety evaluation is fixed to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("formal safety evaluation supports CPU only")
        if tuple(self.branches) != FORMAL_BRANCHES:
            raise ValueError("formal evaluation requires all three fixed branches")
        if not self.policy_seeds:
            raise ValueError("policy_seeds must be non-empty")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if not self.scenario_seeds:
            raise ValueError("scenario_seeds must be non-empty")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("formal safety evaluation requires measurement overhead")
        if not math.isfinite(float(self.quality_logit_penalty)) or self.quality_logit_penalty < 0.0:
            raise ValueError("quality_logit_penalty must be finite and non-negative")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                columns.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _progress(config: SafetyMotionFeatureFormalEvaluationConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-motion-formal-eval] {message}", flush=True)


def _prepare_outputs(config: SafetyMotionFeatureFormalEvaluationConfig) -> tuple[Path, Path, Path]:
    output = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / output.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / output.name
    paths = (output, logs, plots)
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("output, logs, and plots directories must be distinct")
    nonempty = [path for path in paths if path.exists() and any(path.iterdir())]
    if nonempty and not config.force:
        raise FileExistsError(
            "formal evaluation output is not empty; use --force to replace these exact directories: "
            + ", ".join(str(path) for path in nonempty)
        )
    if config.force:
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_formal_bundle(
    config: SafetyMotionFeatureFormalEvaluationConfig,
    policy_seed: int,
    branch: str,
) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]:
    scale_root = (
        Path(config.checkpoint_root)
        / f"seed_{int(policy_seed)}"
        / branch
        / f"scale_{config.uav_count}"
    )
    manifest_path = scale_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing formal checkpoint manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    raw_config = dict(manifest.get("config", {}))
    expected_dim = EXPECTED_DIM[branch]
    if int(raw_config.get("node_feature_dim", expected_dim)) != expected_dim:
        raise ValueError(f"node-feature dimension mismatch for {branch}: {manifest_path}")
    if bool(raw_config.get("dynamic_obstacle_motion_features_enabled", False)) != EXPECTED_DYNAMIC[branch]:
        raise ValueError(f"dynamic-obstacle flag mismatch for {branch}: {manifest_path}")
    if not bool(raw_config.get("measurement_overhead_enabled", False)):
        raise ValueError(f"measurement overhead is disabled in {manifest_path}")
    internal_variant = EXPECTED_VARIANT[branch]
    base_config = _formal_config(manifest, internal_variant)
    eval_config = type(base_config)(
        **{
            **asdict(base_config),
            "max_steps": int(config.max_steps),
            "device": config.device,
            "progress": False,
            "force": False,
            "mappo_variant": internal_variant,
            "dynamic_obstacle_swept_projection_enabled": (
                bool(raw_config.get("dynamic_obstacle_swept_projection_enabled", False))
                if config.dynamic_obstacle_swept_projection_enabled is None
                else bool(config.dynamic_obstacle_swept_projection_enabled)
            ),
        }
    )
    environment = _make_environment(
        eval_config,
        seed=1_000_000 + int(policy_seed),
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=True,
        measurement_budget_enabled=eval_config.measurement_budget_enabled,
        measurement_budget_by_uav=eval_config.measurement_budget_by_uav,
    )
    trainer, quality_model = _load_trainer(
        eval_config,
        scale_root,
        manifest,
        environment,
        seed=int(policy_seed),
    )
    trainer.policy.eval()
    metadata = {
        "branch": branch,
        "internal_variant": internal_variant,
        "node_feature_dim": int(getattr(trainer.config, "node_feature_dim", -1)),
        # Older formal-training manifests did not serialize this derived
        # property.  The loader already validates the expected dimension via
        # the branch contract above, so use that validated value for the
        # audit metadata while remaining compatible with those manifests.
        "manifest_node_feature_dim": int(raw_config.get("node_feature_dim", expected_dim)),
        "dynamic_obstacle_motion_features_enabled": int(
            bool(raw_config.get("dynamic_obstacle_motion_features_enabled", False))
        ),
        "quality_features_enabled": int(
            bool(getattr(trainer.config, "use_measurement_quality_features", False))
        ),
        "budget_features_enabled": int(
            bool(getattr(trainer.config, "use_measurement_budget_features", False))
        ),
        "measurement_budget_enabled": int(bool(eval_config.measurement_budget_enabled)),
        "measurement_overhead_enabled": int(bool(eval_config.measurement_overhead_enabled)),
        "quality_model_source_split": str(quality_model.get("source_split", "")),
        "checkpoint": str((scale_root / "checkpoints" / "final.pt").resolve()),
        "test_data_read": 0,
    }
    return trainer, eval_config, manifest, scale_root, metadata


def _mean(values: Sequence[Any]) -> float:
    numbers = np.asarray([_finite(value) for value in values], dtype=float)
    return float(np.mean(numbers)) if numbers.size else 0.0


def _scenario_means(
    episode_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    metric_keys = (
        "collision_episode",
        "pre_projection_collision_episode",
        "minimum_clearance_m",
        "pre_projection_minimum_clearance_m",
        "safety_projection_count",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "joint_objective",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "measurement_count_total",
        "measurement_overhead_packets_total",
    )
    output: list[dict[str, Any]] = []
    for scenario_seed in scenario_seeds:
        for branch in branches:
            matching = [
                row
                for row in episode_rows
                if int(row["scenario_seed"]) == int(scenario_seed)
                and str(row["branch"]) == branch
            ]
            if not matching:
                raise ValueError(f"missing episode rows for {branch}/{scenario_seed}")
            row = {
                "scenario_seed": int(scenario_seed),
                "branch": branch,
                "policy_seed_count": len(matching),
                "scenario_hash": str(matching[0]["scenario_hash"]),
                **{key: _mean([item.get(key, 0.0) for item in matching]) for key in metric_keys},
                "initial_collision_any": int(any(int(item.get("initial_collision", 0)) for item in matching)),
                "all_state_finite": int(all(int(item.get("state_finite", 0)) for item in matching)),
                "all_covariance_finite": int(all(int(item.get("covariance_finite", 0)) for item in matching)),
                "all_observations_truth_free": int(
                    all(int(item.get("actor_observation_truth_free", 0)) for item in matching)
                ),
                "all_trainer_updates_zero": int(
                    all(int(item.get("trainer_update_called", 1)) == 0 for item in matching)
                ),
                "all_expected_node_dimensions": int(
                    all(int(item.get("actor_node_feature_dim", -1)) == EXPECTED_DIM[branch] for item in matching)
                ),
                "all_dynamic_feature_finite": int(
                    all(
                        int(item.get("dynamic_motion_feature_finite", 0)) == 1
                        for item in matching
                    )
                    if EXPECTED_DYNAMIC[branch]
                    else 1
                ),
                "collision_censoring_rate": _mean(
                    [int(item.get("collision_censored", 0)) for item in matching]
                ),
            }
            output.append(row)
    return output


def _paired_contrasts(
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    comparisons = (
        ("baseline_41d", "baseline_37d", "baseline_41d - baseline_37d"),
        (FULL_BRANCH, "baseline_41d", f"{FULL_BRANCH} - baseline_41d"),
        (FULL_BRANCH, "baseline_37d", f"{FULL_BRANCH} - baseline_37d"),
    )
    metrics = (
        "collision_episode",
        "minimum_clearance_m",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "position_error_p95_m",
        "safety_projection_count",
        "j_loc",
        "j_com",
        "mean_reward",
        "collision_censoring_rate",
    )
    output: list[dict[str, Any]] = []
    for comparison_index, (treatment, reference, label) in enumerate(comparisons):
        for metric_index, metric in enumerate(metrics):
            differences: list[float] = []
            for seed in scenario_seeds:
                treatment_row = next(
                    row
                    for row in scenario_rows
                    if int(row["scenario_seed"]) == int(seed)
                    and str(row["branch"]) == treatment
                )
                reference_row = next(
                    row
                    for row in scenario_rows
                    if int(row["scenario_seed"]) == int(seed)
                    and str(row["branch"]) == reference
                )
                differences.append(
                    _finite(treatment_row.get(metric)) - _finite(reference_row.get(metric))
                )
            low, high = _bootstrap_ci(
                differences,
                int(bootstrap_samples),
                int(bootstrap_seed + comparison_index * 100 + metric_index),
            )
            output.append(
                {
                    "comparison": label,
                    "treatment": treatment,
                    "reference": reference,
                    "metric": metric,
                    "independent_unit": "scenario_seed",
                    "scenario_count": len(differences),
                    "mean_difference": float(np.mean(differences)) if differences else 0.0,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )
    return output


def _find_contrast(
    rows: Sequence[Mapping[str, Any]],
    comparison: str,
    metric: str,
) -> dict[str, Any]:
    for row in rows:
        if str(row["comparison"]) == comparison and str(row["metric"]) == metric:
            return dict(row)
    raise KeyError(f"missing contrast {comparison}/{metric}")


def _plot(
    path: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_37d": "#0072B2",
        "baseline_41d": "#009E73",
        FULL_BRANCH: "#D55E00",
    }
    labels = {
        "baseline_37d": "baseline 37d",
        "baseline_41d": "baseline 41d",
        FULL_BRANCH: "budgeted quality-aware 41d",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metrics = (
        ("collision_episode", "Collision rate", "fraction"),
        ("minimum_clearance_m", "Minimum clearance", "m"),
        ("energy_proxy", "Energy proxy", "normalized"),
    )
    x = np.arange(len(scenario_seeds), dtype=float)
    for axis, (metric, title, ylabel) in zip(axes[0], metrics):
        for branch in FORMAL_BRANCHES:
            values = [
                _finite(
                    next(
                        row[metric]
                        for row in scenario_rows
                        if int(row["scenario_seed"]) == int(seed)
                        and str(row["branch"]) == branch
                    )
                )
                for seed in scenario_seeds
            ]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color=colors[branch],
                label=labels[branch],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    comparison_labels = [
        "41d−37d",
        "full−41d",
        "full−37d",
    ]
    comparison_names = [
        "baseline_41d - baseline_37d",
        f"{FULL_BRANCH} - baseline_41d",
        f"{FULL_BRANCH} - baseline_37d",
    ]
    for axis, (metric, title, ylabel) in zip(
        axes[1],
        (
            ("collision_episode", "Paired collision contrast", "difference"),
            ("minimum_clearance_m", "Paired clearance contrast", "difference (m)"),
            ("energy_proxy", "Paired energy contrast", "difference"),
        ),
    ):
        means = []
        lows = []
        highs = []
        for name in comparison_names:
            row = _find_contrast(contrasts, name, metric)
            mean = _finite(row["mean_difference"])
            low = _finite(row["bootstrap_95_ci_low"])
            high = _finite(row["bootstrap_95_ci_high"])
            means.append(mean)
            lows.append(max(0.0, mean - low))
            highs.append(max(0.0, high - mean))
        positions = np.arange(len(means), dtype=float)
        axis.errorbar(
            positions,
            means,
            yerr=np.asarray([lows, highs]),
            fmt="o",
            capsize=4,
            color="#333333",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, comparison_labels)
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Formal safety evaluation: 37/41-dimensional dynamic-obstacle observation ablation",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Five policy seeds averaged within each frozen scenario; scenario seed is the independent unit; test data sealed",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_motion_feature_formal_evaluation(
    config: SafetyMotionFeatureFormalEvaluationConfig,
) -> dict[str, Any]:
    """Evaluate all three formal branches on the frozen ten-scene hold-out."""

    output, logs, plots = _prepare_outputs(config)
    scenario_payloads = {
        int(seed): _load_scenario(Path(config.scenario_root), int(seed), config.uav_count)
        for seed in config.scenario_seeds
    }
    scenario_hashes = {
        str(seed): scenario_payloads[int(seed)][1] for seed in config.scenario_seeds
    }
    bundles: dict[str, dict[int, tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]]] = {}
    bundle_checks: dict[str, dict[str, Any]] = {}
    for branch in FORMAL_BRANCHES:
        bundles[branch] = {}
        for policy_seed in config.policy_seeds:
            _progress(config, f"loading branch={branch} policy_seed={policy_seed}")
            bundle = _load_formal_bundle(config, int(policy_seed), branch)
            bundles[branch][int(policy_seed)] = bundle
            bundle_checks[f"{branch}/seed_{int(policy_seed)}"] = bundle[4]

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for branch in FORMAL_BRANCHES:
        for policy_seed in config.policy_seeds:
            trainer, eval_config, _manifest, _scale_root, _metadata = bundles[branch][int(policy_seed)]
            for scenario_seed in config.scenario_seeds:
                _progress(
                    config,
                    f"branch={branch} policy_seed={policy_seed} scenario_seed={scenario_seed} start",
                )
                scenario, scenario_hash = scenario_payloads[int(scenario_seed)]
                # The existing motion-feature replay is intentionally reused
                # here: it records per-UAV dynamic features and physical action
                # closure while never calling trainer.update().  It also works
                # for the 37-dimensional branch; absent dynamic features are
                # treated as an expected branch condition below.
                rows, summary = _motion_episode(
                    config,
                    trainer,
                    eval_config,
                    scenario,
                    int(scenario_seed),
                    scenario_hash,
                    int(policy_seed),
                    branch,
                )
                for row in rows:
                    row["expected_node_feature_dim"] = EXPECTED_DIM[branch]
                    row["dynamic_features_expected"] = int(EXPECTED_DYNAMIC[branch])
                    row["formal_holdout_evaluation"] = 1
                summary["expected_node_feature_dim"] = EXPECTED_DIM[branch]
                summary["dynamic_features_expected"] = int(EXPECTED_DYNAMIC[branch])
                summary["formal_holdout_evaluation"] = 1
                episode_rows.append(summary)
                step_rows.extend(rows)
                if int(summary.get("initial_collision", 0)) != 0:
                    errors.append(f"initial collision: {branch}/{policy_seed}/{scenario_seed}")
                if int(summary.get("collision_episode", 0)) and not int(summary.get("terminated", 0)):
                    errors.append(f"collision not terminated: {branch}/{policy_seed}/{scenario_seed}")
                if int(summary.get("task_complete", 0)):
                    errors.append(f"collision marked complete: {branch}/{policy_seed}/{scenario_seed}")

    scenario_rows = _scenario_means(episode_rows, config.scenario_seeds, FORMAL_BRANCHES)
    contrasts = _paired_contrasts(
        scenario_rows,
        config.scenario_seeds,
        config.bootstrap_samples,
        config.bootstrap_seed,
    )
    expected_episode_count = len(FORMAL_BRANCHES) * len(config.policy_seeds) * len(config.scenario_seeds)
    expected_scenario_count = len(FORMAL_BRANCHES) * len(config.scenario_seeds)
    audit_ok = bool(
        len(episode_rows) == expected_episode_count
        and len(scenario_rows) == expected_scenario_count
        and not errors
        and all(int(row.get("initial_collision", 0)) == 0 for row in episode_rows)
        and all(int(row.get("state_finite", 0)) == 1 for row in episode_rows)
        and all(int(row.get("covariance_finite", 0)) == 1 for row in episode_rows)
        and all(int(row.get("actor_observation_truth_free", 0)) == 1 for row in episode_rows)
        and all(int(row.get("trainer_update_called", 1)) == 0 for row in episode_rows)
        and all(int(row.get("test_data_read", 1)) == 0 for row in episode_rows)
        and all(
            int(row.get("actor_node_feature_dim_41", 0)) == 1
            if branch in {"baseline_41d", FULL_BRANCH}
            else int(row.get("actor_node_feature_dim_41", 0)) == 0
            for row in episode_rows
            for branch in [str(row.get("branch", ""))]
        )
        and all(
            int(row.get("actor_node_feature_dim", -1)) == EXPECTED_DIM[str(row["branch"])]
            for row in step_rows
        )
        and all(
            int(metadata.get("node_feature_dim", -1)) == EXPECTED_DIM[branch]
            and int(metadata.get("manifest_node_feature_dim", -1)) == EXPECTED_DIM[branch]
            and int(metadata.get("dynamic_obstacle_motion_features_enabled", -1))
            == int(EXPECTED_DYNAMIC[branch])
            and int(metadata.get("measurement_overhead_enabled", 0)) == 1
            and str(metadata.get("test_data_read", 1)) == "0"
            for branch in FORMAL_BRANCHES
            for metadata in [
                bundle_checks[f"{branch}/seed_{int(config.policy_seeds[0])}"]
            ]
        )
        and all(
            str(row.get("scenario_hash", "")) == scenario_hashes[str(int(row["scenario_seed"]))]
            for row in episode_rows
        )
        and not any("test" in str(path).lower() for path in (config.checkpoint_root, config.scenario_root))
    )

    def decision_for(treatment: str, reference: str) -> dict[str, Any]:
        prefix = f"{treatment} - {reference}"
        collision = _find_contrast(contrasts, prefix, "collision_episode")
        clearance = _find_contrast(contrasts, prefix, "minimum_clearance_m")
        energy = _find_contrast(contrasts, prefix, "energy_proxy")
        censoring = _find_contrast(contrasts, prefix, "collision_censoring_rate")
        return {
            "comparison": prefix,
            "collision_rate_improved_supported": bool(_finite(collision["bootstrap_95_ci_high"]) < 0.0),
            "minimum_clearance_not_worse_supported": bool(_finite(clearance["bootstrap_95_ci_low"]) >= 0.0),
            "energy_not_increased_supported": bool(_finite(energy["bootstrap_95_ci_high"]) <= 0.0),
            "collision_censoring_comparable": bool(
                _finite(censoring["bootstrap_95_ci_low"]) <= 0.0
                <= _finite(censoring["bootstrap_95_ci_high"])
            ),
            "collision_rate": collision,
            "minimum_clearance": clearance,
            "energy": energy,
            "collision_censoring": censoring,
        }

    dynamic_effect = decision_for("baseline_41d", "baseline_37d")
    quality_budget_effect = decision_for(FULL_BRANCH, "baseline_41d")
    end_to_end = decision_for(FULL_BRANCH, "baseline_37d")
    decision_supported = bool(
        audit_ok
        and end_to_end["collision_rate_improved_supported"]
        and end_to_end["minimum_clearance_not_worse_supported"]
        and end_to_end["energy_not_increased_supported"]
        and end_to_end["collision_censoring_comparable"]
    )
    if not audit_ok:
        decision = "structural_audit_failed"
    elif decision_supported:
        decision = "full_method_initial_safety_improvement_supported"
    elif dynamic_effect["collision_rate_improved_supported"]:
        decision = "dynamic_motion_features_help_but_full_branch_not_supported"
    elif quality_budget_effect["collision_rate_improved_supported"]:
        decision = "quality_budget_effect_requires_further_safety_analysis"
    else:
        decision = "no_formal_safety_improvement_supported"

    report = {
        "schema_version": "safety-motion-feature-formal-evaluation-v1",
        "audit_ok": audit_ok,
        "decision_supported": decision_supported,
        "decision": decision,
        "design": {
            "uav_count": config.uav_count,
            "gnss_mode": "denied",
            "max_steps": config.max_steps,
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "branches": list(FORMAL_BRANCHES),
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "mean of five policy seeds within each scene",
            "bootstrap_samples": config.bootstrap_samples,
            "deterministic_cpu_inference": True,
            "trainer_update_called": False,
            "measurement_overhead_enabled": True,
            "test_data_read": False,
            "test_data_role": "formal training test directories sealed and never opened",
        },
        "source": {
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "scenario_hashes": scenario_hashes,
            "bundle_checks": bundle_checks,
        },
        "counts": {
            "episode_rows": len(episode_rows),
            "expected_episode_rows": expected_episode_count,
            "step_rows": len(step_rows),
            "scenario_rows": len(scenario_rows),
            "expected_scenario_rows": expected_scenario_count,
            "initial_collision_count": int(sum(int(row.get("initial_collision", 0)) for row in episode_rows)),
            "collision_episode_counts": {
                branch: int(
                    sum(
                        int(row.get("collision_episode", 0))
                        for row in episode_rows
                        if str(row.get("branch")) == branch
                    )
                )
                for branch in FORMAL_BRANCHES
            },
        },
        "comparisons": {
            "baseline_41d_minus_baseline_37d": dynamic_effect,
            f"{FULL_BRANCH}_minus_baseline_41d": quality_budget_effect,
            f"{FULL_BRANCH}_minus_baseline_37d": end_to_end,
        },
        "audit_errors": errors,
        "interpretation": {
            "collision_censoring": "early collision termination censors energy, path length, and RMSE; these are not claimed as advantages when censoring differs",
            "feature_source": "idealized simulator aggregate of current dynamic-obstacle motion fields; not a real-sensor performance claim",
            "next_step_if_dynamic_effect_only": "retain 41-dimensional observation and inspect quality/budget policy under the same observation",
            "next_step_if_not_supported": "do not alter EKF/reward/channel; inspect safety action and observation mechanisms before retraining",
        },
    }
    _write_csv(logs / "episode_metrics.csv", episode_rows)
    _write_csv(logs / "step_metrics.csv", step_rows)
    _write_csv(logs / "scenario_summary.csv", scenario_rows)
    _write_csv(logs / "paired_contrasts.csv", contrasts)
    _write_json(logs / "audit_report.json", report)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "safety-motion-feature-formal-evaluation-v1",
            "output_root": str(output.resolve()),
            "logs_root": str(logs.resolve()),
            "plots_root": str(plots.resolve()),
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "branches": list(FORMAL_BRANCHES),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "test_data_read": False,
            "independent_unit": "scenario_seed",
        },
    )
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "output_dir": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
        },
    )
    plot_path = plots / "safety_motion_feature_formal_comparison.png"
    _plot(plot_path, scenario_rows, contrasts, config.scenario_seeds)
    return {
        "audit_ok": audit_ok,
        "decision_supported": decision_supported,
        "decision": decision,
        "data": str(output.resolve()),
        "logs": str(logs.resolve()),
        "plots": str(plots.resolve()),
        "audit_report": str((logs / "audit_report.json").resolve()),
        "plot": str(plot_path.resolve()),
    }


__all__ = [
    "FULL_BRANCH",
    "SafetyMotionFeatureFormalEvaluationConfig",
    "run_safety_motion_feature_formal_evaluation",
]
