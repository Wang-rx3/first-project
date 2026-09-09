"""Formal three-branch training for the dynamic-obstacle safety study.

The module deliberately keeps the safety-scene generation, training splits,
and branch definitions explicit.  It is an experiment orchestrator around the
existing :mod:`formal` runner; the environment, EKF, reward, channel, and
MAPPO implementation remain unchanged.

The statistical unit of the later safety evaluation is the frozen scenario
seed.  Training uses a separate scenario pool so that the ten historical
hold-out scenes cannot leak into fitting or model selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.obstacle_model import collision_event
from ..simulation.scenario import generate_safety_risk_scenario
from ..synthetic.episode_generator import _jsonable
from .baseline import _read_csv, _read_json
from .formal import (
    FormalExperimentConfig,
    _make_environment,
    fit_shared_training_metric_scales,
    load_selected_ekf_config,
    run_preliminary,
)


DEFAULT_POLICY_SEEDS = (7071, 8080, 9099, 10118, 11127)
DEFAULT_TRAIN_SCENARIO_SEEDS = tuple(range(12026, 12046))
DEFAULT_VALIDATION_SCENARIO_SEEDS = tuple(range(13026, 13031))
DEFAULT_TEST_SCENARIO_SEEDS = tuple(range(14026, 14031))
FORMAL_BRANCHES = (
    "baseline_37d",
    "baseline_41d",
    "budgeted_quality_aware_41d",
)
INTERNAL_VARIANT = {
    "baseline_37d": "baseline_mappo",
    "baseline_41d": "baseline_mappo",
    "budgeted_quality_aware_41d": "budgeted_quality_aware_mappo",
}
EXPECTED_DYNAMIC = {
    "baseline_37d": False,
    "baseline_41d": True,
    "budgeted_quality_aware_41d": True,
}
EXPECTED_QUALITY = {
    "baseline_37d": False,
    "baseline_41d": False,
    "budgeted_quality_aware_41d": True,
}
EXPECTED_BUDGET = {
    "baseline_37d": False,
    "baseline_41d": False,
    "budgeted_quality_aware_41d": True,
}


@dataclass(frozen=True)
class SafetyMotionFeatureFormalTrainingConfig:
    """Configuration for the formal safety-motion training batch."""

    scenario_pool_root: Path
    output_dir: Path = Path(
        "data/synthetic/safety_motion_feature_formal_mappo_seedset_7071_11127"
    )
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    warmup_episodes: int = 20
    train_episodes: int = 100
    validation_episodes: int = 20
    test_episodes: int = 20
    scenario_profile_path: Path = Path(
        "data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"
    )
    ekf_config_path: Path = Path(
        "data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"
    )
    measurement_overhead_enabled: bool = True
    quality_logit_penalty: float = 1.0
    dynamic_obstacle_swept_projection_enabled: bool = False
    device: str = "cpu"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True
    command_line: str = ""

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("formal safety-motion training is fixed to three UAVs")
        if not self.policy_seeds:
            raise ValueError("policy_seeds must be non-empty")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if min(
            self.warmup_episodes,
            self.train_episodes,
            self.validation_episodes,
            self.test_episodes,
        ) < 1:
            raise ValueError("all episode counts must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("formal safety-motion training supports CPU only")
        if not self.measurement_overhead_enabled:
            raise ValueError("formal safety-motion training requires measurement overhead")
        if not math.isfinite(float(self.quality_logit_penalty)) or self.quality_logit_penalty < 0.0:
            raise ValueError("quality_logit_penalty must be finite and non-negative")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        expected = tuple(int(seed) for seed in self.policy_seeds)
        if len(expected) != len(set(expected)):
            raise ValueError("policy_seeds must be unique")


def _progress(config: SafetyMotionFeatureFormalTrainingConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-motion-formal] {message}", flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_pool() -> dict[str, tuple[int, ...]]:
    return {
        "train": DEFAULT_TRAIN_SCENARIO_SEEDS,
        "validation": DEFAULT_VALIDATION_SCENARIO_SEEDS,
        "test": DEFAULT_TEST_SCENARIO_SEEDS,
    }


def _scenario_file(root: Path, split: str, seed: int) -> Path:
    return Path(root) / split / f"scenario_{int(seed)}.json"


def _validate_initial_safety(
    scenario: Mapping[str, Any],
    *,
    profile_path: Path,
) -> float:
    """Validate initial safety through the same obstacle parser as the env."""

    config = FormalExperimentConfig(
        output_dir=Path(".") / "_scenario_pool_probe",
        uav_count=3,
        seed=int(scenario.get("seed", 0)),
        max_steps=1,
        warmup_episodes=1,
        train_episodes=1,
        validation_episodes=1,
        test_episodes=1,
        scenario_profile_path=profile_path,
        progress=False,
        measurement_overhead_enabled=True,
    )
    environment = _make_environment(config, seed=int(scenario.get("seed", 0)))
    environment.reset(
        seed=int(scenario.get("seed", 0)),
        scenario_config=dict(scenario),
    )
    positions = {
        uav_id: np.asarray(state.position_m, dtype=float).copy()
        for uav_id, state in environment.states.items()
    }
    collided, clearance = collision_event(
        positions,
        environment.obstacles,
        environment.config.motion.safety_distance_m,
    )
    if collided:
        raise ValueError(f"scenario {scenario.get('seed')} is initially unsafe")
    if not np.isfinite(float(clearance)):
        raise ValueError(f"scenario {scenario.get('seed')} has non-finite initial clearance")
    return float(clearance)


def ensure_formal_scenario_pool(
    root: str | Path,
    *,
    scenario_profile_path: str | Path,
    progress: bool = True,
) -> dict[str, Any]:
    """Create or validate the independent formal safety scenario pool.

    Existing pool files are never overwritten by ``--force`` from the training
    batch.  A changed or incomplete pool fails closed instead of silently
    changing the experimental population.
    """

    root_path = Path(root)
    profile_path = Path(scenario_profile_path)
    expected = _expected_pool()
    root_path.mkdir(parents=True, exist_ok=True)
    manifest_path = root_path / "manifest.json"
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, dict):
            raise ValueError(f"invalid scenario pool manifest: {manifest_path}")
        actual_seed_map = {
            str(split): tuple(int(seed) for seed in payload.get("seeds", []))
            for split, payload in dict(existing_manifest.get("splits", {})).items()
            if isinstance(payload, Mapping)
        }
        if actual_seed_map != expected:
            raise ValueError(
                f"existing scenario pool seed assignment differs from the frozen plan: "
                f"{manifest_path}"
            )

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, seeds in expected.items():
        split_dir = root_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            path = _scenario_file(root_path, split, seed)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError(f"scenario file is not an object: {path}")
                scenario = dict(payload)
                if int(scenario.get("seed", -1)) != int(seed) or int(
                    scenario.get("uav_count", -1)
                ) != 3:
                    raise ValueError(f"scenario identity mismatch: {path}")
            else:
                scenario = generate_safety_risk_scenario(int(seed), uav_count=3)
                _write_json(path, scenario)
            clearance = _validate_initial_safety(scenario, profile_path=profile_path)
            rows.append(
                {
                    "seed": int(seed),
                    "path": str(path.resolve()),
                    "sha256": _canonical_hash(scenario),
                    "initial_clearance_m": clearance,
                    "initial_state_safe": True,
                }
            )
        split_rows[split] = rows

    pool_manifest = {
        "schema_version": "safety-motion-feature-formal-scenario-pool-v1",
        "uav_count": 3,
        "generation_rule": (
            "safety-risk-screening-v1; independent train/validation/test seeds; "
            "same-altitude dynamic sphere crossing corridor"
        ),
        "scenario_profile": str(profile_path.resolve()),
        "splits": {
            split: {
                "seeds": [int(seed) for seed in expected[split]],
                "count": len(expected[split]),
                "scenarios": rows,
            }
            for split, rows in split_rows.items()
        },
        "overlap_check": {
            "train_validation": False,
            "train_test": False,
            "validation_test": False,
        },
        "existing_pool_preserved": existing_manifest is not None,
    }
    _write_json(manifest_path, pool_manifest)
    if progress:
        print(
            f"[safety-motion-formal] scenario pool ready: {root_path.resolve()} "
            f"train={len(expected['train'])} validation={len(expected['validation'])} "
            f"test={len(expected['test'])}",
            flush=True,
        )
    return pool_manifest


def _prepare_batch_outputs(config: SafetyMotionFeatureFormalTrainingConfig) -> tuple[Path, Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / root.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / root.name
    paths = (root, logs, plots)
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("output, logs, and plots directories must be distinct")
    nonempty = [path for path in paths if path.exists() and any(path.iterdir())]
    if nonempty and not config.force:
        raise FileExistsError(
            "formal safety output is not empty; use --force to replace these exact directories: "
            + ", ".join(str(path) for path in nonempty)
        )
    if config.force:
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _fit_baseline_measurement_budgets(branch_scale: Path, uav_count: int) -> dict[str, int]:
    train_root = branch_scale / "train"
    episode_dirs = sorted(path for path in train_root.glob("episode_*") if path.is_dir())
    if len(episode_dirs) == 0:
        raise RuntimeError(f"baseline train episodes are missing: {train_root}")
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


def _base_config(
    config: SafetyMotionFeatureFormalTrainingConfig,
    seed: int,
    output_dir: Path,
    ekf_parameters: Mapping[str, float],
) -> FormalExperimentConfig:
    return FormalExperimentConfig(
        output_dir=output_dir,
        uav_count=config.uav_count,
        seed=int(seed),
        gnss_mode="denied",
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
        scenario_pool_root=Path(config.scenario_pool_root),
        ekf_config_source=str(Path(config.ekf_config_path).resolve()),
        command_line=config.command_line,
        progress=config.progress,
        measurement_overhead_enabled=True,
        measurement_quality_logit_penalty=float(config.quality_logit_penalty),
        dynamic_obstacle_swept_projection_enabled=(
            config.dynamic_obstacle_swept_projection_enabled
        ),
    )


def _branch_config(
    template: FormalExperimentConfig,
    *,
    branch: str,
    output_dir: Path,
    metric_scales: Mapping[str, float],
    measurement_budget_by_uav: Mapping[str, int] | None,
) -> FormalExperimentConfig:
    quality_enabled = EXPECTED_QUALITY[branch]
    budget_enabled = EXPECTED_BUDGET[branch]
    return FormalExperimentConfig(
        **{
            **asdict(template),
            "output_dir": output_dir,
            "dynamic_obstacle_motion_features_enabled": EXPECTED_DYNAMIC[branch],
            "metric_scales": dict(metric_scales),
            "metric_scale_source": "baseline_37d_train_warmup_p95",
            "mappo_variant": INTERNAL_VARIANT[branch],
            "measurement_quality_logit_penalty": (
                float(template.measurement_quality_logit_penalty) if quality_enabled else 0.0
            ),
            "measurement_budget_enabled": budget_enabled,
            "measurement_budget_by_uav": (
                dict(measurement_budget_by_uav) if budget_enabled else None
            ),
        }
    )


def _seed_training(
    config: SafetyMotionFeatureFormalTrainingConfig,
    seed: int,
    batch_root: Path,
) -> dict[str, Any]:
    ekf_parameters = load_selected_ekf_config(config.ekf_config_path)
    seed_root = batch_root / f"seed_{int(seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    template = _base_config(config, seed, seed_root / "metric_scale_probe", ekf_parameters)
    # The common scale is fitted from training warm-up probes only.  All
    # branches use it so branch differences cannot be caused by KPI scaling.
    metric_scales = fit_shared_training_metric_scales(template, (config.uav_count,))
    budgets: dict[str, int] | None = None
    summaries: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    branch_records: dict[str, Any] = {}
    for branch in FORMAL_BRANCHES:
        _progress(config, f"seed={seed} branch={branch} start")
        branch_root = seed_root / branch
        branch_config = _branch_config(
            template,
            branch=branch,
            output_dir=branch_root,
            metric_scales=metric_scales,
            measurement_budget_by_uav=budgets,
        )
        summary = run_preliminary(branch_config)
        scale_root = branch_root / "scale_3"
        report = validate_episode_benchmark(scale_root)
        if not bool(report.get("ok", False)):
            raise RuntimeError(f"episode-v1.3 failed for seed={seed}, branch={branch}: {report}")
        summaries[branch] = summary
        reports[branch] = report
        if branch == "baseline_37d":
            budgets = _fit_baseline_measurement_budgets(scale_root, config.uav_count)
            _write_json(seed_root / "measurement_budgets.json", budgets)
        branch_records[branch] = {
            "branch": branch,
            "internal_mappo_variant": INTERNAL_VARIANT[branch],
            "scale_root": str(scale_root.resolve()),
            "manifest": str((scale_root / "manifest.json").resolve()),
            "checkpoint": str((scale_root / "checkpoints" / "final.pt").resolve()),
            "node_feature_dim": int(summary["node_feature_dim"]),
            "dynamic_obstacle_motion_features_enabled": bool(
                summary["dynamic_obstacle_motion_features_enabled"]
            ),
            "quality_features_enabled": bool(summary["measurement_quality"]["enabled"]),
            "measurement_budget_enabled": bool(summary["measurement_budget_enabled"]),
            "measurement_overhead_enabled": bool(summary["measurement_overhead_enabled"]),
            "measurement_quality_logit_penalty": float(
                summary["measurement_quality"]["logit_penalty"]
            ),
            "episode_v1_3_ok": True,
        }
        _progress(config, f"seed={seed} branch={branch} complete")
    return {
        "policy_seed": int(seed),
        "seed_root": str(seed_root.resolve()),
        "branches": branch_records,
        "measurement_budgets": dict(budgets or {}),
        "common_metric_scales": {str(key): float(value) for key, value in metric_scales.items()},
        "episode_v1_3": reports,
        "test_data_generated_but_sealed": True,
    }


def run_safety_motion_feature_formal_training(
    config: SafetyMotionFeatureFormalTrainingConfig,
) -> dict[str, Any]:
    """Run the independent scenario-pool training batch."""

    pool_manifest = ensure_formal_scenario_pool(
        config.scenario_pool_root,
        scenario_profile_path=config.scenario_profile_path,
        progress=config.progress,
    )
    output_root, logs_root, plots_root = _prepare_batch_outputs(config)
    seed_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for seed in config.policy_seeds:
        try:
            seed_records.append(_seed_training(config, int(seed), output_root))
        except Exception as exc:
            errors.append(f"seed={seed}: {exc}")
            raise

    manifest = {
        "schema_version": "safety-motion-feature-formal-training-v1",
        "output_root": str(output_root.resolve()),
        "logs_root": str(logs_root.resolve()),
        "plots_root": str(plots_root.resolve()),
        "scenario_pool_root": str(Path(config.scenario_pool_root).resolve()),
        "scenario_pool_manifest": str(
            (Path(config.scenario_pool_root) / "manifest.json").resolve()
        ),
        "policy_seeds": [int(seed) for seed in config.policy_seeds],
        "branches": list(FORMAL_BRANCHES),
        "branch_mapping": {
            branch: {
                "dynamic_obstacle_motion_features_enabled": EXPECTED_DYNAMIC[branch],
                "quality_features_enabled": EXPECTED_QUALITY[branch],
                "measurement_budget_enabled": EXPECTED_BUDGET[branch],
                "node_feature_dim": 41 if EXPECTED_DYNAMIC[branch] else 37,
                "measurement_overhead_enabled": True,
            }
            for branch in FORMAL_BRANCHES
        },
        "fixed_conditions": {
            "uav_count": int(config.uav_count),
            "gnss_mode": "denied",
            "max_steps": int(config.max_steps),
            "warmup_episodes": int(config.warmup_episodes),
            "train_episodes": int(config.train_episodes),
            "validation_episodes": int(config.validation_episodes),
            "test_episodes": int(config.test_episodes),
            "quality_logit_penalty": float(config.quality_logit_penalty),
            "dynamic_obstacle_swept_projection_enabled": bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
            "device": config.device,
            "test_role": "generated and sealed; no model selection or fitting",
        },
        "seed_records": seed_records,
        "test_data_used_for_fitting_or_selection": False,
        "errors": errors,
        "command_line": config.command_line,
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(logs_root / "audit_report.json", {
        "schema_version": "safety-motion-feature-formal-training-audit-v1",
        "audit_ok": not errors and len(seed_records) == len(config.policy_seeds),
        "episode_v1_3_all_ok": all(
            all(bool(report.get("ok", False)) for report in record["episode_v1_3"].values())
            for record in seed_records
        ),
        "policy_seed_count": len(seed_records),
        "branch_count_per_seed": len(FORMAL_BRANCHES),
        "test_data_used_for_fitting_or_selection": False,
        "scenario_pool_manifest": manifest["scenario_pool_manifest"],
    })
    _write_json(logs_root / "run_manifest.json", manifest)
    _write_json(
        logs_root / "measurement_budgets.json",
        {
            str(record["policy_seed"]): record["measurement_budgets"]
            for record in seed_records
        },
    )
    _write_json(
        output_root / "sealed_test_notice.json",
        {
            "sealed": True,
            "message": "Formal test episodes were generated for final reporting only; they were not read by this training orchestrator for fitting or selection.",
        },
    )
    return {
        "audit_ok": not errors and len(seed_records) == len(config.policy_seeds),
        "data": str(output_root.resolve()),
        "logs": str(logs_root.resolve()),
        "plots": str(plots_root.resolve()),
        "scenario_pool": str(Path(config.scenario_pool_root).resolve()),
        "policy_seeds": [int(seed) for seed in config.policy_seeds],
        "branches": list(FORMAL_BRANCHES),
        "seed_records": seed_records,
        "audit_report": str((logs_root / "audit_report.json").resolve()),
    }


__all__ = [
    "DEFAULT_POLICY_SEEDS",
    "DEFAULT_TEST_SCENARIO_SEEDS",
    "DEFAULT_TRAIN_SCENARIO_SEEDS",
    "DEFAULT_VALIDATION_SCENARIO_SEEDS",
    "FORMAL_BRANCHES",
    "SafetyMotionFeatureFormalTrainingConfig",
    "ensure_formal_scenario_pool",
    "run_safety_motion_feature_formal_training",
]
