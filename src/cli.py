"""Command-line entry points for dataset registration and benchmark generation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .dataset_registry import registry_as_dicts, render_registry
from .schema import validate_dataset_dir, validate_episode_benchmark
from .symbol_audit import audit_symbol_table
from .synthetic.episode_generator import (
    EpisodeBenchmarkConfig,
    generate_episode_benchmark,
    preset_config,
)
from .synthetic.flightmare_adapter import import_flightmare_episode
from .synthetic.generator import SyntheticConfig, generate_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research dataset registry, validation, and synthetic benchmark tools."
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-datasets", help="list the planned dataset roles")
    list_parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")

    generate_parser = subparsers.add_parser(
        "generate-synthetic", help="generate a deterministic schema-compatible benchmark"
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_flat_benchmark"),
        help="output directory",
    )
    generate_parser.add_argument("--seed", type=int, default=2026)
    generate_parser.add_argument("--scenarios", type=int, default=9)
    generate_parser.add_argument("--steps", type=int, default=120)
    generate_parser.add_argument("--dt", type=float, default=0.2)
    generate_parser.add_argument("--force", action="store_true", help="allow writing into a non-empty output directory")

    validate_parser = subparsers.add_parser(
        "validate", help="validate trajectory, link, and obstacle CSV files"
    )
    validate_parser.add_argument("path", type=Path)

    episode_parser = subparsers.add_parser(
        "generate-episodes",
        help="generate hierarchical scenario/state/sensor/link/action/reward episodes",
    )
    episode_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_benchmark"),
        help="benchmark root directory",
    )
    episode_parser.add_argument(
        "--preset",
        choices=("smoke", "pilot", "thesis"),
        default="smoke",
        help="smoke=3 short episodes, pilot=30x30s, thesis=420x60s",
    )
    episode_parser.add_argument("--seed", type=int, default=2026)
    episode_parser.add_argument("--force", action="store_true", help="allow writing into an existing hierarchical benchmark")

    episode_validate_parser = subparsers.add_parser(
        "validate-episodes", help="validate a hierarchical episode benchmark"
    )
    episode_validate_parser.add_argument("path", type=Path)

    pilot_parser = subparsers.add_parser(
        "run-pilot",
        help="run the reproducible three-UAV Python GAT/MAPPO closed-loop pilot",
    )
    pilot_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_pilot"),
        help="pilot output directory",
    )
    pilot_parser.add_argument("--seed", type=int, default=2026)
    pilot_parser.add_argument("--steps", type=int, default=20, dest="max_steps")
    pilot_parser.add_argument("--warmup-episodes", type=int, default=2)
    pilot_parser.add_argument("--integration-steps", type=int, default=5)
    pilot_parser.add_argument("--train-episodes", type=int, default=10)
    pilot_parser.add_argument("--validation-episodes", type=int, default=2)
    pilot_parser.add_argument("--device", choices=("cpu",), default="cpu")
    pilot_parser.add_argument("--force", action="store_true", help="replace this pilot output directory")

    calibration_parser = subparsers.add_parser(
        "calibrate-ekf",
        help="run a training-only VO/process/NLOS EKF parameter sweep",
    )
    calibration_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026"),
        help="calibration output directory",
    )
    calibration_parser.add_argument("--seed", type=int, default=2026)
    calibration_parser.add_argument("--uav-count", type=int, choices=(3, 5, 7), default=3)
    calibration_parser.add_argument("--episodes", type=int, default=3)
    calibration_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    calibration_parser.add_argument("--vo-std", type=float, nargs="+", default=(0.015, 0.03, 0.05))
    calibration_parser.add_argument("--process-position-std", type=float, nargs="+", default=(0.03, 0.08, 0.15))
    calibration_parser.add_argument("--process-velocity-std", type=float, nargs="+", default=(0.08, 0.15, 0.25))
    calibration_parser.add_argument("--nlos-variance-multiplier", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    calibration_parser.add_argument("--force", action="store_true", help="replace this calibration output directory")

    preliminary_parser = subparsers.add_parser(
        "run-preliminary",
        help="run the single-UAV sanity or 3/5/7-UAV MAPPO experiment and create plots",
    )
    preliminary_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_preliminary"),
        help="root directory for independent scale runs",
    )
    preliminary_parser.add_argument(
        "--uav-counts",
        type=int,
        nargs="+",
        choices=(1, 3, 5, 7),
        default=(3, 5, 7),
        help="one or more independent scales; 1 is a single-agent sanity case",
    )
    preliminary_parser.add_argument("--seed", type=int, default=2026)
    preliminary_parser.add_argument(
        "--scenario-seed",
        type=int,
        default=None,
        help="hold generated scene geometry fixed across episodes; useful for the single-UAV sanity case",
    )
    preliminary_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    preliminary_parser.add_argument("--warmup-episodes", type=int, default=5)
    preliminary_parser.add_argument(
        "--train-episodes",
        type=int,
        default=100,
        help="MAPPO training episodes/updates per UAV scale (default: 100)",
    )
    preliminary_parser.add_argument("--validation-episodes", type=int, default=10)
    preliminary_parser.add_argument("--test-episodes", type=int, default=10)
    preliminary_parser.add_argument("--learning-rate", type=float, default=1e-4)
    preliminary_parser.add_argument("--policy-epochs", type=int, default=4)
    preliminary_parser.add_argument("--entropy-coef", type=float, default=0.005)
    preliminary_parser.add_argument("--initial-delta-std", type=float, default=0.20)
    preliminary_parser.add_argument(
        "--single-uav-learning-mode",
        choices=("off", "loc_goal_v1"),
        default="off",
        help=(
            "single-UAV PPO reward channel; loc_goal_v1 uses warm-up-fitted "
            "EKF localization and goal-progress signals"
        ),
    )
    preliminary_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=None,
        help="selected_ekf_config.json; its four EKF parameters become the run defaults",
    )
    preliminary_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=None,
        help="validated frozen environment profile, e.g. selected_profiles/moderate.json",
    )
    # These remain available for controlled overrides.  ``None`` is
    # intentional: it lets --ekf-config distinguish an omitted argument from
    # an explicit conflicting value.
    preliminary_parser.add_argument("--vo-std", type=float, default=None)
    preliminary_parser.add_argument("--process-position-std", type=float, default=None)
    preliminary_parser.add_argument("--process-velocity-std", type=float, default=None)
    preliminary_parser.add_argument("--nlos-variance-multiplier", type=float, default=None)
    preliminary_parser.add_argument("--device", choices=("cpu",), default="cpu")
    preliminary_parser.add_argument(
        "--mappo-variant",
        choices=("baseline_mappo", "quality_aware_mappo", "budgeted_quality_aware_mappo"),
        default="baseline_mappo",
        help="baseline actor or warm-up-fitted quality-aware measurement actor",
    )
    preliminary_parser.add_argument(
        "--vo-quality-calibration-enabled",
        action="store_true",
        help=(
            "fit an execution-time visual-odometry covariance map from warm-up "
            "tracking quality and innovation diagnostics"
        ),
    )
    preliminary_parser.add_argument(
        "--encoder-type",
        choices=("gat", "mlp_flat"),
        default="gat",
        help="local actor encoder; mlp_flat is the fixed-slot no-GAT architecture ablation",
    )
    preliminary_parser.add_argument("--force", action="store_true", help="replace selected scale directories")
    preliminary_parser.add_argument("--quiet", action="store_true", help="disable per-episode progress output")

    safety_motion_parser = subparsers.add_parser(
        "run-safety-motion-feature-smoke",
        help=(
            "train a small three-UAV smoke experiment with execution-visible "
            "dynamic-obstacle motion features"
        ),
    )
    safety_motion_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="frozen explicit safety scenario JSON",
    )
    safety_motion_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_smoke_seed2026"),
        help="new smoke data output root",
    )
    safety_motion_parser.add_argument(
        "--logs-output",
        type=Path,
        default=Path("logs/preliminary/safety_motion_feature_smoke_seed2026"),
        help="new smoke log output root",
    )
    safety_motion_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path("plots/preliminary/safety_motion_feature_smoke_seed2026"),
        help="new smoke plot output root",
    )
    safety_motion_parser.add_argument("--uav-count", type=int, choices=(3,), default=3)
    safety_motion_parser.add_argument("--seed", type=int, default=2026)
    safety_motion_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_motion_parser.add_argument("--warmup-episodes", type=int, default=2)
    safety_motion_parser.add_argument("--train-episodes", type=int, default=5)
    safety_motion_parser.add_argument("--validation-episodes", type=int, default=5)
    safety_motion_parser.add_argument("--test-episodes", type=int, default=5)
    safety_motion_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        default=True,
        help="include the existing 512-byte measurement queue overhead (default: enabled)",
    )
    safety_motion_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    safety_motion_parser.add_argument(
        "--dynamic-obstacle-swept-projection-enabled",
        action="store_true",
        help="enable the corrected one-step moving-obstacle safety envelope",
    )
    safety_motion_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_motion_parser.add_argument("--force", action="store_true", help="replace this smoke output")
    safety_motion_parser.add_argument("--quiet", action="store_true", help="disable per-episode progress output")

    baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="replay a three-UAV MAPPO validation/test split with the independent fixed baseline",
    )
    baseline_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="reference MAPPO scale_3 directory",
    )
    baseline_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_independent_baseline_seed2026"),
        help="separate baseline output root",
    )
    baseline_parser.add_argument(
        "--controller",
        choices=("independent-fixed",),
        default="independent-fixed",
    )
    baseline_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="frozen reference split to replay (default: test)",
    )
    baseline_parser.add_argument("--cruise-speed", type=float, default=3.0, dest="cruise_speed_mps")
    baseline_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    baseline_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    baseline_parser.add_argument("--force", action="store_true", help="replace the baseline scale_3 output")
    baseline_parser.add_argument("--quiet", action="store_true", help="disable per-episode progress output")

    causal_audit_parser = subparsers.add_parser(
        "audit-causal-chain",
        help="audit the paired action-to-KPI-to-objective chain without changing experiments",
    )
    causal_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="MAPPO scale directory containing the paired split",
    )
    causal_audit_parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="independent fixed-baseline scale directory with matching seeds",
    )
    causal_audit_parser.add_argument("--split", choices=("validation", "test"), default="test")
    causal_audit_parser.add_argument("--logs-output", type=Path, default=None)
    causal_audit_parser.add_argument("--plots-output", type=Path, default=None)
    causal_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    causal_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)

    objective_sensitivity_parser = subparsers.add_parser(
        "run-objective-sensitivity-audit",
        help="audit validation-only consistency among RMSE, J_loc, J_com, and team reward",
    )
    objective_sensitivity_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen five-seed quality-aware MAPPO output root",
    )
    objective_sensitivity_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are sealed",
    )
    objective_sensitivity_parser.add_argument("--logs-output", type=Path, default=None)
    objective_sensitivity_parser.add_argument("--plots-output", type=Path, default=None)
    objective_sensitivity_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    objective_sensitivity_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    objective_sensitivity_parser.add_argument(
        "--force", action="store_true", help="replace only this audit's logs and plot directories"
    )

    execution_uncertainty_proxy_parser = subparsers.add_parser(
        "run-execution-uncertainty-proxy-audit",
        help="fit a train-only execution-time uncertainty proxy and audit validation candidates",
    )
    execution_uncertainty_proxy_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen five-seed quality-aware MAPPO output root",
    )
    execution_uncertainty_proxy_parser.add_argument(
        "--train-split",
        choices=("train",),
        default="train",
        help="training split used only to fit per-seed proxy scales",
    )
    execution_uncertainty_proxy_parser.add_argument(
        "--eval-split",
        choices=("validation",),
        default="validation",
        help="validation split used for the read-only audit; test is sealed",
    )
    execution_uncertainty_proxy_parser.add_argument("--logs-output", type=Path, default=None)
    execution_uncertainty_proxy_parser.add_argument("--plots-output", type=Path, default=None)
    execution_uncertainty_proxy_parser.add_argument(
        "--age-variance-coefficient",
        type=float,
        default=1.0,
        help="diagnostic age^2 variance coefficient used by the proxy",
    )
    execution_uncertainty_proxy_parser.add_argument(
        "--force", action="store_true", help="replace only this audit's logs and plot directories"
    )

    execution_uncertainty_proxy_alignment_parser = subparsers.add_parser(
        "run-execution-uncertainty-proxy-alignment-audit",
        help="align the execution-time uncertainty proxy with validation linear covariance reduction",
    )
    execution_uncertainty_proxy_alignment_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen five-seed quality-aware MAPPO output root",
    )
    execution_uncertainty_proxy_alignment_parser.add_argument(
        "--train-split",
        choices=("train",),
        default="train",
        help="training split used only to fit per-seed proxy scales",
    )
    execution_uncertainty_proxy_alignment_parser.add_argument(
        "--eval-split",
        choices=("validation",),
        default="validation",
        help="validation split used for the read-only alignment audit; test is sealed",
    )
    execution_uncertainty_proxy_alignment_parser.add_argument("--logs-output", type=Path, default=None)
    execution_uncertainty_proxy_alignment_parser.add_argument("--plots-output", type=Path, default=None)
    execution_uncertainty_proxy_alignment_parser.add_argument(
        "--age-variance-coefficient",
        type=float,
        default=1.0,
        help="diagnostic age^2 variance coefficient used by the proxy",
    )
    execution_uncertainty_proxy_alignment_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    execution_uncertainty_proxy_alignment_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    execution_uncertainty_proxy_alignment_parser.add_argument(
        "--force", action="store_true", help="replace only this audit's logs and plot directories"
    )

    bandwidth_intervention_parser = subparsers.add_parser(
        "run-bandwidth-intervention",
        help="replay the frozen three-UAV validation set under controlled bandwidth allocations",
    )
    bandwidth_intervention_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    bandwidth_intervention_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_bandwidth_intervention_moderate_validation_seed2026"),
        help="separate intervention dataset root",
    )
    bandwidth_intervention_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally unavailable",
    )
    bandwidth_intervention_parser.add_argument("--logs-output", type=Path, default=None)
    bandwidth_intervention_parser.add_argument("--plots-output", type=Path, default=None)
    bandwidth_intervention_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    bandwidth_intervention_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    bandwidth_intervention_parser.add_argument(
        "--minimum-informative-fraction",
        type=float,
        default=0.20,
        help="minimum fraction of global windows with an effective bandwidth intervention",
    )
    bandwidth_intervention_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    bandwidth_intervention_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    localization_coupling_parser = subparsers.add_parser(
        "run-localization-coupling-intervention",
        help="isolate received-neighbor-message consumption by the cooperative EKF",
    )
    localization_coupling_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    localization_coupling_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_localization_coupling_moderate_validation_seed2026"),
        help="separate paired intervention dataset root",
    )
    localization_coupling_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    localization_coupling_parser.add_argument("--logs-output", type=Path, default=None)
    localization_coupling_parser.add_argument("--plots-output", type=Path, default=None)
    localization_coupling_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    localization_coupling_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    localization_coupling_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    localization_coupling_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    ekf_strength_parser = subparsers.add_parser(
        "run-ekf-update-strength-audit",
        help="replay validation episodes and audit cooperative EKF update strength without changing the filter",
    )
    ekf_strength_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    ekf_strength_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_ekf_update_strength_audit_seed2026"),
        help="separate read-only audit replay dataset root",
    )
    ekf_strength_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    ekf_strength_parser.add_argument("--logs-output", type=Path, default=None)
    ekf_strength_parser.add_argument("--plots-output", type=Path, default=None)
    ekf_strength_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    ekf_strength_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    measurement_density_parser = subparsers.add_parser(
        "run-measurement-density-intervention",
        help="compare recorded MAPPO measurement choices with an actor-visible round-robin baseline",
    )
    measurement_density_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    measurement_density_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_measurement_density_moderate_validation_seed2026"),
        help="separate paired intervention dataset root",
    )
    measurement_density_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    measurement_density_parser.add_argument("--logs-output", type=Path, default=None)
    measurement_density_parser.add_argument("--plots-output", type=Path, default=None)
    measurement_density_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    measurement_density_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    measurement_density_parser.add_argument(
        "--minimum-selection-fraction",
        type=float,
        default=0.80,
        help="minimum non-none selection fraction among actor rows with candidates",
    )
    measurement_density_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    measurement_density_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    update_quality_parser = subparsers.add_parser(
        "run-cooperative-update-quality-audit",
        help="audit whether individual cooperative EKF updates are beneficial, harmful, or neutral",
    )
    update_quality_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    update_quality_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_cooperative_update_quality_audit_seed2026"),
        help="separate paired read-only audit dataset root",
    )
    update_quality_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    update_quality_parser.add_argument("--logs-output", type=Path, default=None)
    update_quality_parser.add_argument("--plots-output", type=Path, default=None)
    update_quality_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    update_quality_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    update_quality_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    update_quality_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    quality_measurement_parser = subparsers.add_parser(
        "run-quality-aware-measurement-intervention",
        help="compare MAPPO, round-robin, and train-fitted actor-visible quality-aware measurement choices",
    )
    quality_measurement_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    quality_measurement_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_quality_aware_measurement_seed2026"),
        help="separate paired intervention dataset root",
    )
    quality_measurement_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    quality_measurement_parser.add_argument("--logs-output", type=Path, default=None)
    quality_measurement_parser.add_argument("--plots-output", type=Path, default=None)
    quality_measurement_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    quality_measurement_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    quality_measurement_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    quality_measurement_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    soft_quality_parser = subparsers.add_parser(
        "run-soft-quality-measurement-intervention",
        help="compare MAPPO, round-robin, hard gates, and a train-fitted continuous quality score",
    )
    soft_quality_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="frozen three-UAV Moderate MAPPO scale_3 directory",
    )
    soft_quality_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_soft_quality_measurement_seed2026"),
        help="separate paired soft-quality intervention dataset root",
    )
    soft_quality_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are prohibited",
    )
    soft_quality_parser.add_argument("--logs-output", type=Path, default=None)
    soft_quality_parser.add_argument("--plots-output", type=Path, default=None)
    soft_quality_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    soft_quality_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    soft_quality_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    soft_quality_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    multiseed_soft_quality_parser = subparsers.add_parser(
        "run-multiseed-soft-quality-validation",
        help="train independent 3-UAV Moderate seeds and validate the frozen soft selector",
    )
    multiseed_soft_quality_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_multiseed_soft_quality_seed2026"),
        help="batch output root containing per-seed references and interventions",
    )
    multiseed_soft_quality_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062),
        help="independent base training seeds",
    )
    multiseed_soft_quality_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    multiseed_soft_quality_parser.add_argument("--warmup-episodes", type=int, default=20)
    multiseed_soft_quality_parser.add_argument("--train-episodes", type=int, default=100)
    multiseed_soft_quality_parser.add_argument("--validation-episodes", type=int, default=20)
    multiseed_soft_quality_parser.add_argument("--test-episodes", type=int, default=20)
    multiseed_soft_quality_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    multiseed_soft_quality_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    multiseed_soft_quality_parser.add_argument("--logs-output", type=Path, default=None)
    multiseed_soft_quality_parser.add_argument("--plots-output", type=Path, default=None)
    multiseed_soft_quality_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    multiseed_soft_quality_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    multiseed_soft_quality_parser.add_argument("--device", choices=("cpu",), default="cpu")
    multiseed_soft_quality_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    multiseed_soft_quality_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    quality_mappo_parser = subparsers.add_parser(
        "run-quality-aware-mappo",
        help="train baseline and quality-aware MAPPO branches over independent 3-UAV seeds",
    )
    quality_mappo_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_quality_aware_mappo_seed2026"),
        help="batch output root containing baseline and quality-aware runs",
    )
    quality_mappo_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(2026, 3035, 4044, 5053, 6062)
    )
    quality_mappo_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    quality_mappo_parser.add_argument("--warmup-episodes", type=int, default=20)
    quality_mappo_parser.add_argument("--train-episodes", type=int, default=100)
    quality_mappo_parser.add_argument("--validation-episodes", type=int, default=20)
    quality_mappo_parser.add_argument("--test-episodes", type=int, default=20)
    quality_mappo_parser.add_argument(
        "--scenario-profile", type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    quality_mappo_parser.add_argument(
        "--ekf-config", type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    quality_mappo_parser.add_argument("--logs-output", type=Path, default=None)
    quality_mappo_parser.add_argument("--plots-output", type=Path, default=None)
    quality_mappo_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    quality_mappo_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    quality_mappo_parser.add_argument("--device", choices=("cpu",), default="cpu")
    quality_mappo_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        help="inject one equivalent 512-byte packet per selected measurement into the communication queue",
    )
    quality_mappo_parser.add_argument(
        "--quality-logit-penalty",
        type=float,
        default=1.0,
        help="non-negative fixed penalty applied to quality-aware candidate measurement logits",
    )
    quality_mappo_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    quality_mappo_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    rate_controlled_mappo_parser = subparsers.add_parser(
        "run-rate-controlled-quality-mappo",
        help="train baseline, budgeted, and train-rate-controlled quality-aware MAPPO branches",
    )
    rate_controlled_mappo_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_rate_controlled_quality_mappo_seedset_7071_11127"),
        help="separate batch output root",
    )
    rate_controlled_mappo_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    rate_controlled_mappo_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    rate_controlled_mappo_parser.add_argument("--warmup-episodes", type=int, default=20)
    rate_controlled_mappo_parser.add_argument("--train-episodes", type=int, default=100)
    rate_controlled_mappo_parser.add_argument("--validation-episodes", type=int, default=20)
    rate_controlled_mappo_parser.add_argument("--test-episodes", type=int, default=20)
    rate_controlled_mappo_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    rate_controlled_mappo_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    rate_controlled_mappo_parser.add_argument("--logs-output", type=Path, default=None)
    rate_controlled_mappo_parser.add_argument("--plots-output", type=Path, default=None)
    rate_controlled_mappo_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    rate_controlled_mappo_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    rate_controlled_mappo_parser.add_argument("--device", choices=("cpu",), default="cpu")
    rate_controlled_mappo_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        help="record the fixed 512-byte equivalent packet load (enabled by this experiment)",
    )
    rate_controlled_mappo_parser.add_argument(
        "--quality-logit-penalty",
        type=float,
        default=1.0,
        help="non-negative fixed penalty applied to quality-aware candidate measurement logits",
    )
    rate_controlled_mappo_parser.add_argument(
        "--measurement-rate-coef",
        type=float,
        default=0.5,
        help="fixed coefficient for the train-only expected measurement-rate constraint",
    )
    rate_controlled_mappo_parser.add_argument(
        "--measurement-rate-tolerance",
        type=float,
        default=0.05,
        help="fixed tolerance around each baseline train measurement-rate target",
    )
    rate_controlled_mappo_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    rate_controlled_mappo_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    adaptive_rate_mappo_parser = subparsers.add_parser(
        "run-adaptive-budgeted-quality-mappo",
        help="train baseline, budgeted, and adaptive per-UAV rate-controlled MAPPO branches",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/adaptive_budgeted_quality_aware_mappo_seedset_7071_11127"),
        help="separate batch output root",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    adaptive_rate_mappo_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    adaptive_rate_mappo_parser.add_argument("--warmup-episodes", type=int, default=20)
    adaptive_rate_mappo_parser.add_argument("--train-episodes", type=int, default=100)
    adaptive_rate_mappo_parser.add_argument("--validation-episodes", type=int, default=20)
    adaptive_rate_mappo_parser.add_argument("--test-episodes", type=int, default=20)
    adaptive_rate_mappo_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    adaptive_rate_mappo_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    adaptive_rate_mappo_parser.add_argument("--logs-output", type=Path, default=None)
    adaptive_rate_mappo_parser.add_argument("--plots-output", type=Path, default=None)
    adaptive_rate_mappo_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    adaptive_rate_mappo_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    adaptive_rate_mappo_parser.add_argument("--device", choices=("cpu",), default="cpu")
    adaptive_rate_mappo_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        help="enable the fixed 512-byte equivalent packet load (required)",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--quality-logit-penalty",
        type=float,
        default=1.0,
        help="non-negative fixed penalty applied to quality-aware candidate measurement logits",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--dual-initial",
        type=float,
        default=0.0,
        help="initial per-UAV measurement-rate dual multiplier",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--dual-step-size",
        type=float,
        default=0.25,
        help="per-update multiplier step size",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--dual-max",
        type=float,
        default=5.0,
        help="maximum per-UAV measurement-rate dual multiplier",
    )
    adaptive_rate_mappo_parser.add_argument(
        "--measurement-rate-tolerance",
        type=float,
        default=0.05,
        help="tolerance around each baseline train measurement-rate target",
    )
    adaptive_rate_mappo_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    adaptive_rate_mappo_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    proxy_guided_mappo_parser = subparsers.add_parser(
        "run-proxy-guided-adaptive-quality-mappo",
        help="train baseline, adaptive-rate, and proxy-guided adaptive MAPPO branches",
    )
    proxy_guided_mappo_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/proxy_guided_adaptive_quality_mappo_seedset_7071_11127"),
        help="separate batch output root",
    )
    proxy_guided_mappo_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    proxy_guided_mappo_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    proxy_guided_mappo_parser.add_argument("--warmup-episodes", type=int, default=20)
    proxy_guided_mappo_parser.add_argument("--train-episodes", type=int, default=100)
    proxy_guided_mappo_parser.add_argument("--validation-episodes", type=int, default=20)
    proxy_guided_mappo_parser.add_argument("--test-episodes", type=int, default=20)
    proxy_guided_mappo_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    proxy_guided_mappo_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    proxy_guided_mappo_parser.add_argument("--logs-output", type=Path, default=None)
    proxy_guided_mappo_parser.add_argument("--plots-output", type=Path, default=None)
    proxy_guided_mappo_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    proxy_guided_mappo_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    proxy_guided_mappo_parser.add_argument("--device", choices=("cpu",), default="cpu")
    proxy_guided_mappo_parser.add_argument(
        "--measurement-overhead-enabled", action="store_true",
        help="enable the fixed 512-byte equivalent measurement packet load (required)",
    )
    proxy_guided_mappo_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    proxy_guided_mappo_parser.add_argument("--measurement-rate-tolerance", type=float, default=0.05)
    proxy_guided_mappo_parser.add_argument("--dual-initial", type=float, default=0.0)
    proxy_guided_mappo_parser.add_argument("--dual-step-size", type=float, default=0.25)
    proxy_guided_mappo_parser.add_argument("--dual-max", type=float, default=5.0)
    proxy_guided_mappo_parser.add_argument("--proxy-age-variance-coefficient", type=float, default=1.0)
    proxy_guided_mappo_parser.add_argument("--proxy-teacher-coef", type=float, default=0.25)
    proxy_guided_mappo_parser.add_argument("--proxy-temperature", type=float, default=0.25)
    proxy_guided_mappo_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    proxy_guided_mappo_parser.add_argument(
        "--branch-invariant-initialization",
        action="store_true",
        help="use identical isolated RNG streams for shared MAPPO modules across branches",
    )
    proxy_guided_mappo_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    proxy_guided_mappo_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    communication_constrained_parser = subparsers.add_parser(
        "run-communication-constrained-proxy-mappo",
        help="train baseline, proxy-guided, and execution-cost-constrained proxy MAPPO branches",
    )
    communication_constrained_parser.add_argument("--output", type=Path, default=Path("data/synthetic/communication_constrained_proxy_mappo_seedset_12136_16172"))
    communication_constrained_parser.add_argument("--seeds", type=int, nargs="+", default=(12136, 13145, 14154, 15163, 16172))
    communication_constrained_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    communication_constrained_parser.add_argument("--warmup-episodes", type=int, default=20)
    communication_constrained_parser.add_argument("--train-episodes", type=int, default=100)
    communication_constrained_parser.add_argument("--validation-episodes", type=int, default=20)
    communication_constrained_parser.add_argument("--test-episodes", type=int, default=20)
    communication_constrained_parser.add_argument("--scenario-profile", type=Path, default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"))
    communication_constrained_parser.add_argument("--ekf-config", type=Path, default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"))
    communication_constrained_parser.add_argument("--logs-output", type=Path, default=None)
    communication_constrained_parser.add_argument("--plots-output", type=Path, default=None)
    communication_constrained_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    communication_constrained_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    communication_constrained_parser.add_argument("--device", choices=("cpu",), default="cpu")
    communication_constrained_parser.add_argument(
        "--vo-quality-calibration-enabled",
        action="store_true",
        help="fit and freeze a conservative VO covariance map from each seed's train warm-up",
    )
    communication_constrained_parser.add_argument("--measurement-overhead-enabled", action="store_true")
    communication_constrained_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    communication_constrained_parser.add_argument("--proxy-teacher-coef", type=float, default=0.25)
    communication_constrained_parser.add_argument("--proxy-temperature", type=float, default=0.25)
    communication_constrained_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    communication_constrained_parser.add_argument("--communication-cost-tolerance", type=float, default=0.05)
    communication_constrained_parser.add_argument("--communication-dual-initial", type=float, default=0.0)
    communication_constrained_parser.add_argument("--communication-dual-step-size", type=float, default=0.25)
    communication_constrained_parser.add_argument("--communication-dual-max", type=float, default=5.0)
    communication_constrained_parser.add_argument("--fixed-packet-cost-fraction", type=float, default=0.5)
    communication_constrained_parser.add_argument(
        "--branch-invariant-initialization",
        action="store_true",
        help="use identical isolated RNG streams for shared MAPPO modules across branches",
    )
    communication_constrained_parser.add_argument("--force", action="store_true")
    communication_constrained_parser.add_argument("--quiet", action="store_true")

    persistent_proxy_parser = subparsers.add_parser(
        "run-persistent-proxy-guided-quality-mappo",
        help="train baseline and fixed-horizon persistent-proxy MAPPO branches",
    )
    persistent_proxy_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/persistent_proxy_guided_quality_mappo_seedset_17181_21227"),
        help="separate batch output root",
    )
    persistent_proxy_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(17181, 18190, 19209, 20218, 21227)
    )
    persistent_proxy_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    persistent_proxy_parser.add_argument("--warmup-episodes", type=int, default=20)
    persistent_proxy_parser.add_argument("--train-episodes", type=int, default=100)
    persistent_proxy_parser.add_argument("--validation-episodes", type=int, default=20)
    persistent_proxy_parser.add_argument("--test-episodes", type=int, default=20)
    persistent_proxy_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    persistent_proxy_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    persistent_proxy_parser.add_argument("--logs-output", type=Path, default=None)
    persistent_proxy_parser.add_argument("--plots-output", type=Path, default=None)
    persistent_proxy_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    persistent_proxy_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    persistent_proxy_parser.add_argument("--device", choices=("cpu",), default="cpu")
    persistent_proxy_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        help="enable one equivalent 512-byte packet per selected measurement (required)",
    )
    persistent_proxy_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    persistent_proxy_parser.add_argument("--persistent-horizon", type=int, default=3)
    persistent_proxy_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    persistent_proxy_parser.add_argument("--proxy-teacher-coef", type=float, default=0.25)
    persistent_proxy_parser.add_argument("--proxy-temperature", type=float, default=0.25)
    persistent_proxy_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    persistent_proxy_parser.add_argument("--communication-cost-tolerance", type=float, default=0.05)
    persistent_proxy_parser.add_argument("--communication-dual-initial", type=float, default=0.0)
    persistent_proxy_parser.add_argument("--communication-dual-step-size", type=float, default=0.25)
    persistent_proxy_parser.add_argument("--communication-dual-max", type=float, default=5.0)
    persistent_proxy_parser.add_argument("--fixed-packet-cost-fraction", type=float, default=0.5)
    persistent_proxy_parser.add_argument("--force", action="store_true", help="replace batch outputs")
    persistent_proxy_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    persistent_proxy_audit_parser = subparsers.add_parser(
        "run-persistent-proxy-audit",
        help="audit the fixed-horizon persistent proxy using train-only fitting and validation diagnostics",
    )
    persistent_proxy_audit_parser.add_argument("--reference", type=Path, required=True)
    persistent_proxy_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227/persistent_proxy_audit"),
    )
    persistent_proxy_audit_parser.add_argument("--plots-output", type=Path, default=None)
    persistent_proxy_audit_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(17181, 18190, 19209, 20218, 21227)
    )
    persistent_proxy_audit_parser.add_argument("--persistent-horizon", type=int, default=3)
    persistent_proxy_audit_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    persistent_proxy_audit_parser.add_argument("--fixed-packet-cost-fraction", type=float, default=0.5)
    persistent_proxy_audit_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    persistent_proxy_audit_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    persistent_proxy_mechanism_audit_parser = subparsers.add_parser(
        "run-persistent-proxy-mechanism-audit",
        help="audit persistent-proxy teacher utility, policy coverage, and measurement collapse",
    )
    persistent_proxy_mechanism_audit_parser.add_argument(
        "--reference", type=Path, required=True,
        help="completed persistent-proxy MAPPO batch root",
    )
    persistent_proxy_mechanism_audit_parser.add_argument(
        "--output", type=Path,
        default=Path(
            "logs/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227/"
            "persistent_proxy_mechanism_audit"
        ),
    )
    persistent_proxy_mechanism_audit_parser.add_argument("--plots-output", type=Path, default=None)
    persistent_proxy_mechanism_audit_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(17181, 18190, 19209, 20218, 21227)
    )
    persistent_proxy_mechanism_audit_parser.add_argument(
        "--split", choices=("validation",), default="validation",
        help="validation-only audit; test data remain sealed",
    )
    persistent_proxy_mechanism_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    persistent_proxy_mechanism_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    persistent_proxy_mechanism_audit_parser.add_argument("--device", choices=("cpu",), default="cpu")
    persistent_proxy_mechanism_audit_parser.add_argument("--force", action="store_true")
    persistent_proxy_mechanism_audit_parser.add_argument("--quiet", action="store_true")

    communication_cost_audit_parser = subparsers.add_parser(
        "run-measurement-communication-cost-audit",
        help="audit train-only fixed-packet plus visible-link communication costs",
    )
    communication_cost_audit_parser.add_argument("--reference", type=Path, required=True)
    communication_cost_audit_parser.add_argument("--output", type=Path, default=Path("logs/preliminary/measurement_communication_cost_audit"))
    communication_cost_audit_parser.add_argument("--plots-output", type=Path, default=None)
    communication_cost_audit_parser.add_argument("--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127))
    communication_cost_audit_parser.add_argument("--fixed-packet-cost-fraction", type=float, default=0.5)
    communication_cost_audit_parser.add_argument("--force", action="store_true")

    proxy_teacher_audit_parser = subparsers.add_parser(
        "run-proxy-teacher-audit",
        help="audit the execution-visible information-gain/communication-cost teacher before training",
    )
    proxy_teacher_audit_parser.add_argument("--reference", type=Path, required=True, help="completed multi-seed baseline reference root")
    proxy_teacher_audit_parser.add_argument(
        "--output", type=Path,
        default=Path("logs/preliminary/proxy_teacher_audit"),
        help="audit output directory",
    )
    proxy_teacher_audit_parser.add_argument("--plots-output", type=Path, default=None)
    proxy_teacher_audit_parser.add_argument("--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127))
    proxy_teacher_audit_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    proxy_teacher_audit_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    proxy_teacher_audit_parser.add_argument("--temperature", type=float, default=0.25)
    proxy_teacher_audit_parser.add_argument("--force", action="store_true", help="replace audit output")
    proxy_teacher_audit_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    measurement_rate_constraint_audit_parser = subparsers.add_parser(
        "run-measurement-rate-constraint-audit",
        help="audit train-only measurement-rate targets, constraint diagnostics, and validation rate stability",
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed rate-controlled MAPPO batch root",
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/"
            "rate_constraint_audit"
        ),
        help="audit CSV/JSON output directory",
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults beside the reference batch",
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--source-logs",
        type=Path,
        default=None,
        help="source formal logs; defaults to logs/preliminary/<reference-name>",
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    measurement_rate_constraint_audit_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    measurement_rate_constraint_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    measurement_rate_constraint_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    measurement_rate_constraint_audit_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    measurement_rate_constraint_audit_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    measurement_rate_sampling_audit_parser = subparsers.add_parser(
        "run-measurement-rate-sampling-audit",
        help="audit expected non-none probabilities against sampled measurement actions",
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed rate-controlled MAPPO batch root",
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/"
            "rate_sampling_audit"
        ),
        help="audit CSV/JSON output directory",
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults beside the reference batch",
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--source-logs",
        type=Path,
        default=None,
        help="source formal logs; defaults to logs/preliminary/<reference-name>",
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    measurement_rate_sampling_audit_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    measurement_rate_sampling_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    measurement_rate_sampling_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    measurement_rate_sampling_audit_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    measurement_rate_sampling_audit_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    measurement_rate_constraint_mechanism_audit_parser = subparsers.add_parser(
        "run-measurement-rate-constraint-mechanism-audit",
        help="audit per-UAV measurement-rate targets, aggregation, train/validation shift, and gradient flow",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed rate-controlled MAPPO batch root",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/"
            "rate_constraint_mechanism_audit"
        ),
        help="audit CSV/JSON output directory",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults to plots/preliminary/<reference-name>/rate_constraint_mechanism_audit",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--source-logs",
        type=Path,
        default=None,
        help="source formal logs; defaults to logs/preliminary/<reference-name>",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation-only mechanism audit; test data are intentionally sealed",
    )
    measurement_rate_constraint_mechanism_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    measurement_rate_constraint_mechanism_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    measurement_rate_constraint_mechanism_audit_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    measurement_rate_constraint_mechanism_audit_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    information_gain_budget_parser = subparsers.add_parser(
        "run-information-gain-budget-intervention",
        help="compare recorded, unconstrained, and budgeted execution-time information-gain measurement choices",
    )
    information_gain_budget_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="quality-aware MAPPO batch root containing the five baseline_mappo seed branches",
    )
    information_gain_budget_parser.add_argument(
        "--reference-branch",
        choices=("baseline_mappo",),
        default="baseline_mappo",
        help="common reference branch; the intervention is intentionally baseline-only",
    )
    information_gain_budget_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_information_gain_budget_intervention_seed2026"),
        help="separate validation-only intervention dataset root",
    )
    information_gain_budget_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally unavailable",
    )
    information_gain_budget_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="exactly five seed IDs; when omitted, discover the five seed_* branches under --reference",
    )
    information_gain_budget_parser.add_argument("--logs-output", type=Path, default=None)
    information_gain_budget_parser.add_argument("--plots-output", type=Path, default=None)
    information_gain_budget_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    information_gain_budget_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    information_gain_budget_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    information_gain_budget_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        help="inject one equivalent 512-byte packet per non-none measurement into the communication queue",
    )
    information_gain_budget_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    information_gain_budget_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    measurement_overhead_parser = subparsers.add_parser(
        "run-measurement-overhead-intervention",
        help="validate measurement-message communication overhead with paired five-seed replays",
    )
    measurement_overhead_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="quality-aware MAPPO batch root containing the five baseline_mappo seed branches",
    )
    measurement_overhead_parser.add_argument(
        "--reference-branch",
        choices=("baseline_mappo",),
        default="baseline_mappo",
        help="common baseline branch used for all paired replays",
    )
    measurement_overhead_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/python_sim_measurement_overhead_intervention_seed2026"),
        help="separate validation-only intervention dataset root",
    )
    measurement_overhead_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally unavailable",
    )
    measurement_overhead_parser.add_argument("--logs-output", type=Path, default=None)
    measurement_overhead_parser.add_argument("--plots-output", type=Path, default=None)
    measurement_overhead_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    measurement_overhead_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    measurement_overhead_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    measurement_overhead_parser.add_argument("--force", action="store_true", help="replace intervention outputs")
    measurement_overhead_parser.add_argument("--quiet", action="store_true", help="disable episode progress output")

    policy_mechanism_audit_parser = subparsers.add_parser(
        "run-policy-mechanism-audit",
        help="audit quality-aware MAPPO measurement probabilities and communication overhead without retraining",
    )
    policy_mechanism_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed quality-aware MAPPO overhead batch root",
    )
    policy_mechanism_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/"
            "policy_mechanism_audit"
        ),
        help="diagnostic log output directory",
    )
    policy_mechanism_audit_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="diagnostic plot directory; defaults beside the reference batch",
    )
    policy_mechanism_audit_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062),
    )
    policy_mechanism_audit_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "quality_aware_mappo",
            "budgeted_quality_aware_mappo",
        ),
        default=("baseline_mappo", "quality_aware_mappo"),
        help="branches to audit; defaults to the historical two-branch audit",
    )
    policy_mechanism_audit_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    policy_mechanism_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    policy_mechanism_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    policy_mechanism_audit_parser.add_argument("--device", choices=("cpu",), default="cpu")
    policy_mechanism_audit_parser.add_argument("--force", action="store_true", help="replace diagnostic outputs")
    policy_mechanism_audit_parser.add_argument("--quiet", action="store_true", help="disable replay progress output")

    proxy_update_quality_audit_parser = subparsers.add_parser(
        "run-proxy-guided-update-quality-audit",
        help="attribute proxy-guided measurement choices to accepted cooperative EKF update quality",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed proxy-guided adaptive MAPPO batch root",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/proxy_guided_adaptive_quality_mappo_seedset_7071_11127/"
            "proxy_update_quality_audit"
        ),
        help="validation-only audit CSV/JSON output directory",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults beside the reference batch",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
        help="training seed IDs to audit",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "adaptive_budgeted_quality_aware_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        ),
        default=(
            "baseline_mappo",
            "adaptive_budgeted_quality_aware_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        ),
        help="frozen branches to replay",
    )
    proxy_update_quality_audit_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    proxy_update_quality_audit_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    proxy_update_quality_audit_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    proxy_update_quality_audit_parser.add_argument("--age-variance-coefficient", type=float, default=1.0)
    proxy_update_quality_audit_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    proxy_update_quality_audit_parser.add_argument("--temperature", type=float, default=0.25)
    proxy_update_quality_audit_parser.add_argument("--device", choices=("cpu",), default="cpu")
    proxy_update_quality_audit_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    proxy_update_quality_audit_parser.add_argument("--quiet", action="store_true", help="disable replay progress output")

    metric_consistency_parser = subparsers.add_parser(
        "run-metric-consistency-audit",
        help="audit legacy coordinate-wise versus canonical vector-norm localization metrics",
    )
    metric_consistency_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed MAPPO batch root containing frozen scale_3 diagnostics",
    )
    metric_consistency_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "communication_constrained_proxy_mappo_seedset_12136_16172/"
            "metric_consistency_audit"
        ),
        help="validation-only metric audit CSV/JSON output directory",
    )
    metric_consistency_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="metric audit plot directory; defaults beside the reference batch",
    )
    metric_consistency_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(12136, 13145, 14154, 15163, 16172),
        help="training seeds to audit",
    )
    metric_consistency_parser.add_argument(
        "--branches",
        nargs="+",
        default=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        help="frozen branch names to audit",
    )
    metric_consistency_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    metric_consistency_parser.add_argument("--expected-episodes", type=int, default=20)
    metric_consistency_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    metric_consistency_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    metric_consistency_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    metric_consistency_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    final_mainline_parser = subparsers.add_parser(
        "run-final-mainline-audit",
        help="run the validation-only final communication-localization audit and figures",
    )
    final_mainline_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed multi-seed MAPPO batch root containing frozen scale_3 diagnostics",
    )
    final_mainline_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
            "final_mainline_audit"
        ),
        help="validation-only final audit output directory",
    )
    final_mainline_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/"
            "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
            "final_mainline_figures"
        ),
        help="final figure output directory",
    )
    final_mainline_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(27281, 28290, 29309, 30318, 31327),
        help="training seeds to audit",
    )
    final_mainline_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        default=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        help="frozen branches to include; baseline and proposed are required",
    )
    final_mainline_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    final_mainline_parser.add_argument("--expected-episodes", type=int, default=20)
    final_mainline_parser.add_argument("--expected-training-updates", type=int, default=100)
    final_mainline_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    final_mainline_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    final_mainline_parser.add_argument("--force", action="store_true", help="replace final audit outputs")
    final_mainline_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")
    final_mainline_parser.add_argument("--figure-profile", choices=("legacy", "mainline-v2"), default="legacy",
                                      help="mainline-v2 adds paired five-seed figures and exact training reward decomposition")

    curve_cause_parser = subparsers.add_parser(
        "run-mainline-curve-cause-audit",
        help="attribute 3-UAV training/validation curve anomalies without retraining",
    )
    curve_cause_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed five-seed 3-UAV experiment root",
    )
    curve_cause_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
            "curve_cause_audit"
        ),
        help="derived read-only attribution tables and report",
    )
    curve_cause_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/"
            "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
            "curve_cause_audit"
        ),
        help="diagnostic-only attribution figures",
    )
    curve_cause_parser.add_argument(
        "--source-logs",
        type=Path,
        default=None,
        help="optional parent logs directory; inferred from the reference name by default",
    )
    curve_cause_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(27281, 28290, 29309, 30318, 31327),
        help="training seeds treated as independent statistical units",
    )
    curve_cause_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        default=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
    )
    curve_cause_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test remains sealed",
    )
    curve_cause_parser.add_argument("--expected-training-updates", type=int, default=100)
    curve_cause_parser.add_argument("--expected-validation-episodes", type=int, default=20)
    curve_cause_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    curve_cause_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    curve_cause_parser.add_argument("--force", action="store_true", help="replace only this audit's outputs")
    curve_cause_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    ekf_persistence_parser = subparsers.add_parser(
        "run-ekf-update-persistence-audit",
        help="audit whether accepted cooperative EKF update benefits persist over later validation windows",
    )
    ekf_persistence_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed MAPPO batch root containing frozen scale_3 diagnostics",
    )
    ekf_persistence_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "communication_constrained_proxy_mappo_seedset_12136_16172/"
            "ekf_update_persistence_audit"
        ),
        help="validation-only audit CSV/JSON output directory",
    )
    ekf_persistence_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults to plots/preliminary/<reference-name>/ekf_update_persistence_audit",
    )
    ekf_persistence_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(12136, 13145, 14154, 15163, 16172),
        help="training seeds to audit",
    )
    ekf_persistence_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        default=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        help="frozen branches to audit",
    )
    ekf_persistence_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    ekf_persistence_parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=(1, 2, 3, 5),
        help="future decision-window horizons for persistence diagnostics",
    )
    ekf_persistence_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    ekf_persistence_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    ekf_persistence_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    ekf_persistence_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    ekf_sensor_interaction_parser = subparsers.add_parser(
        "run-ekf-sensor-interaction-audit",
        help="audit accepted cooperative UWB updates against subsequent EKF sensor windows",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed MAPPO batch root containing frozen validation diagnostics",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "communication_constrained_proxy_mappo_seedset_12136_16172/"
            "ekf_sensor_interaction_audit"
        ),
        help="validation-only audit CSV/JSON output directory",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="audit plot directory; defaults beside the reference batch",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(12136, 13145, 14154, 15163, 16172),
        help="training seeds to audit",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--branches",
        nargs="+",
        default=(
            "baseline_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
        ),
        help="frozen branches to audit in contrast order",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    ekf_sensor_interaction_parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=(1, 2, 3, 5),
        help="future diagnostic-window horizons",
    )
    ekf_sensor_interaction_parser.add_argument("--expected-episodes", type=int, default=20)
    ekf_sensor_interaction_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    ekf_sensor_interaction_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    ekf_sensor_interaction_parser.add_argument("--force", action="store_true", help="replace audit outputs")
    ekf_sensor_interaction_parser.add_argument("--quiet", action="store_true", help="disable audit progress output")

    measurement_policy_stability_parser = subparsers.add_parser(
        "run-measurement-policy-stability-audit",
        help="diagnose cross-seed measurement-policy and training-dynamics stability without retraining",
    )
    measurement_policy_stability_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed MAPPO batch root containing seed/branch/scale_3 diagnostics",
    )
    measurement_policy_stability_parser.add_argument(
        "--mechanism-audit",
        type=Path,
        default=None,
        help="validation-only policy mechanism audit directory; defaults beside the reference batch",
    )
    measurement_policy_stability_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/"
            "measurement_policy_stability_audit"
        ),
        help="diagnostic CSV/JSON output directory",
    )
    measurement_policy_stability_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="diagnostic plot directory; defaults beside the reference batch",
    )
    measurement_policy_stability_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
    )
    measurement_policy_stability_parser.add_argument(
        "--branches",
        nargs="+",
        choices=(
            "baseline_mappo",
            "quality_aware_mappo",
            "budgeted_quality_aware_mappo",
        ),
        default=("baseline_mappo", "quality_aware_mappo", "budgeted_quality_aware_mappo"),
    )
    measurement_policy_stability_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    measurement_policy_stability_parser.add_argument("--force", action="store_true", help="replace diagnostic outputs")
    measurement_policy_stability_parser.add_argument("--quiet", action="store_true", help="disable summary progress output")

    quality_feature_sensitivity_parser = subparsers.add_parser(
        "run-quality-feature-sensitivity-audit",
        help="audit quality-risk semantics and MAPPO measurement-head counterfactual response",
    )
    quality_feature_sensitivity_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="completed quality-aware MAPPO overhead batch root",
    )
    quality_feature_sensitivity_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/"
            "quality_feature_sensitivity"
        ),
        help="validation-only diagnostic log output directory",
    )
    quality_feature_sensitivity_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="diagnostic plot directory; defaults beside the reference batch",
    )
    quality_feature_sensitivity_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062),
    )
    quality_feature_sensitivity_parser.add_argument(
        "--split",
        choices=("validation",),
        default="validation",
        help="validation only; test data are intentionally sealed",
    )
    quality_feature_sensitivity_parser.add_argument("--device", choices=("cpu",), default="cpu")
    quality_feature_sensitivity_parser.add_argument("--force", action="store_true", help="replace diagnostic outputs")
    quality_feature_sensitivity_parser.add_argument("--quiet", action="store_true", help="disable replay progress output")

    screening_parser = subparsers.add_parser(
        "screen-scenarios",
        help="screen three-UAV communication parameters without training MAPPO",
    )
    screening_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026"),
        help="screening manifest and frozen-profile directory",
    )
    screening_parser.add_argument("--logs-output", type=Path, default=None)
    screening_parser.add_argument("--plots-output", type=Path, default=None)
    screening_parser.add_argument("--seed", type=int, default=2026, dest="design_seed")
    screening_parser.add_argument("--design-points", type=int, default=36)
    screening_parser.add_argument("--episodes-per-point", type=int, default=10)
    screening_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    screening_parser.add_argument("--cruise-speed", type=float, default=3.0, dest="cruise_speed_mps")
    screening_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=None,
        help="optional selected_ekf_config.json used consistently for every screening point",
    )
    screening_parser.add_argument("--force", action="store_true", help="replace screening outputs")
    screening_parser.add_argument("--quiet", action="store_true", help="disable per-run progress output")

    safety_screening_parser = subparsers.add_parser(
        "run-safety-risk-screening",
        help="screen a fixed three-UAV safety-risk scene with an independent controller",
    )
    safety_screening_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026"),
        help="scenario JSON and screening manifest output directory",
    )
    safety_screening_parser.add_argument(
        "--logs-output",
        type=Path,
        default=None,
        help="screening CSV/JSON log directory; defaults under logs/preliminary",
    )
    safety_screening_parser.add_argument(
        "--plots-output",
        type=Path,
        default=None,
        help="screening plot directory; defaults under plots/preliminary",
    )
    safety_screening_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="independent fixed scenario seeds",
    )
    safety_screening_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_screening_parser.add_argument("--cruise-speed", type=float, default=3.0, dest="cruise_speed_mps")
    safety_screening_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
        help="frozen Moderate communication profile",
    )
    safety_screening_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
        help="frozen training-only EKF configuration",
    )
    safety_screening_parser.add_argument(
        "--no-projection-diagnostic",
        action="store_true",
        help="omit the diagnostic replay with safety projection disabled",
    )
    safety_screening_parser.add_argument(
        "--dynamic-obstacle-swept-projection-enabled",
        action="store_true",
        help="include the one-step dynamic-obstacle swept envelope in safety projection",
    )
    safety_screening_parser.add_argument("--force", action="store_true", help="replace screening outputs")
    safety_screening_parser.add_argument("--quiet", action="store_true", help="disable per-scenario progress output")

    safety_mappo_parser = subparsers.add_parser(
        "run-safety-mappo-evaluation",
        help="evaluate frozen MAPPO checkpoints on paired safety-risk scenarios",
    )
    safety_mappo_parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("data/synthetic/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"),
        help="root containing frozen policy-seed/branch/scale_3 checkpoints",
    )
    safety_mappo_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="directory containing frozen scenario_<seed>.json files",
    )
    safety_mappo_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_mappo_evaluation_seed2026"),
        help="read-only evaluation manifest/data directory",
    )
    safety_mappo_parser.add_argument("--logs-output", type=Path, default=None)
    safety_mappo_parser.add_argument("--plots-output", type=Path, default=None)
    safety_mappo_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
        help="frozen policy checkpoint seeds",
    )
    safety_mappo_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="frozen safety-scene seeds",
    )
    safety_mappo_parser.add_argument(
        "--branches",
        nargs="+",
        default=("baseline_mappo", "budgeted_quality_aware_mappo"),
        help="checkpoint branches to evaluate; both comparison branches are required",
    )
    safety_mappo_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_mappo_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_mappo_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_mappo_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_mappo_parser.add_argument("--force", action="store_true", help="replace these exact outputs")
    safety_mappo_parser.add_argument("--quiet", action="store_true", help="disable per-run progress output")

    safety_motion_eval_parser = subparsers.add_parser(
        "run-safety-motion-feature-evaluation",
        help=(
            "evaluate the direct single-seed 41-dimensional dynamic-obstacle "
            "Smoke checkpoints on frozen safety scenarios"
        ),
    )
    safety_motion_eval_parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_smoke_seed2026"),
        help="direct Smoke root containing branch/scale_3 checkpoints",
    )
    safety_motion_eval_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="directory containing frozen scenario_<seed>.json files",
    )
    safety_motion_eval_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_evaluation_seed2026"),
        help="new diagnostic manifest/data directory",
    )
    safety_motion_eval_parser.add_argument(
        "--logs-output",
        type=Path,
        default=Path("logs/preliminary/safety_motion_feature_evaluation_seed2026"),
        help="diagnostic CSV/JSON output directory",
    )
    safety_motion_eval_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path("plots/preliminary/safety_motion_feature_evaluation_seed2026"),
        help="diagnostic plot output directory",
    )
    safety_motion_eval_parser.add_argument(
        "--branches",
        nargs="+",
        choices=("baseline_mappo", "budgeted_quality_aware_mappo"),
        default=("baseline_mappo", "budgeted_quality_aware_mappo"),
        help="the two frozen Smoke branches; both are required",
    )
    safety_motion_eval_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(2026,),
        help="fixed Smoke policy seed; this diagnostic accepts only 2026",
    )
    safety_motion_eval_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="frozen safety-scene seeds",
    )
    safety_motion_eval_parser.add_argument("--uav-count", type=int, choices=(3,), default=3)
    safety_motion_eval_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_motion_eval_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        default=True,
        help="require the Smoke checkpoints to use the 512-byte overhead mechanism",
    )
    safety_motion_eval_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    safety_motion_eval_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_motion_eval_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_motion_eval_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_motion_eval_parser.add_argument("--force", action="store_true", help="replace these exact outputs")
    safety_motion_eval_parser.add_argument("--quiet", action="store_true", help="disable per-scenario progress output")

    safety_motion_formal_train_parser = subparsers.add_parser(
        "run-safety-motion-feature-formal-training",
        help="train the three formal 37d/41d safety-motion ablation branches",
    )
    safety_motion_formal_train_parser.add_argument(
        "--scenario-pool-root",
        type=Path,
        required=True,
        help="independent train/validation/test safety scenario pool root",
    )
    safety_motion_formal_train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_formal_mappo_seedset_7071_11127"),
    )
    safety_motion_formal_train_parser.add_argument("--logs-output", type=Path, default=None)
    safety_motion_formal_train_parser.add_argument("--plots-output", type=Path, default=None)
    safety_motion_formal_train_parser.add_argument(
        "--policy-seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    safety_motion_formal_train_parser.add_argument("--uav-count", type=int, choices=(3,), default=3)
    safety_motion_formal_train_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_motion_formal_train_parser.add_argument("--warmup-episodes", type=int, default=20)
    safety_motion_formal_train_parser.add_argument("--train-episodes", type=int, default=100)
    safety_motion_formal_train_parser.add_argument("--validation-episodes", type=int, default=20)
    safety_motion_formal_train_parser.add_argument("--test-episodes", type=int, default=20)
    safety_motion_formal_train_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    safety_motion_formal_train_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    safety_motion_formal_train_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        default=True,
        help="enable one equivalent 512-byte packet per selected measurement",
    )
    safety_motion_formal_train_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    safety_motion_formal_train_parser.add_argument(
        "--dynamic-obstacle-swept-projection-enabled",
        action="store_true",
        help="enable the corrected one-step moving-obstacle safety envelope",
    )
    safety_motion_formal_train_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_motion_formal_train_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_motion_formal_train_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_motion_formal_train_parser.add_argument("--force", action="store_true")
    safety_motion_formal_train_parser.add_argument("--quiet", action="store_true")

    safety_motion_formal_eval_parser = subparsers.add_parser(
        "run-safety-motion-feature-formal-evaluation",
        help="evaluate formal 37d/41d safety branches on frozen hold-out scenes",
    )
    safety_motion_formal_eval_parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="formal training root containing seed_*/branch/scale_3 checkpoints",
    )
    safety_motion_formal_eval_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
    )
    safety_motion_formal_eval_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_formal_evaluation_seed2026"),
    )
    safety_motion_formal_eval_parser.add_argument("--logs-output", type=Path, default=None)
    safety_motion_formal_eval_parser.add_argument("--plots-output", type=Path, default=None)
    safety_motion_formal_eval_parser.add_argument(
        "--branches",
        nargs="+",
        choices=("baseline_37d", "baseline_41d", "budgeted_quality_aware_41d"),
        default=("baseline_37d", "baseline_41d", "budgeted_quality_aware_41d"),
    )
    safety_motion_formal_eval_parser.add_argument(
        "--policy-seeds", type=int, nargs="+", default=(7071, 8080, 9099, 10118, 11127)
    )
    safety_motion_formal_eval_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
    )
    safety_motion_formal_eval_parser.add_argument("--uav-count", type=int, choices=(3,), default=3)
    safety_motion_formal_eval_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    safety_motion_formal_eval_parser.add_argument("--measurement-overhead-enabled", action="store_true", default=True)
    safety_motion_formal_eval_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    safety_motion_formal_eval_parser.add_argument(
        "--dynamic-obstacle-swept-projection-enabled",
        action="store_true",
        default=None,
        help="override the checkpoint setting and enable the one-step swept envelope",
    )
    safety_motion_formal_eval_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_motion_formal_eval_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_motion_formal_eval_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_motion_formal_eval_parser.add_argument("--force", action="store_true")
    safety_motion_formal_eval_parser.add_argument("--quiet", action="store_true")

    safety_motion_formal_mechanism_parser = subparsers.add_parser(
        "run-safety-motion-feature-formal-mechanism-audit",
        help=(
            "attribute formal 37d/41d safety differences to dynamic observations, "
            "nominal actions, projection coverage, or quality/budget policy"
        ),
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data/synthetic/safety_motion_feature_formal_mappo_seedset_7071_11127"),
        help="formal training root containing seed_*/branch/scale_3 checkpoints",
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="frozen ten-scene safety hold-out directory",
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/"
            "safety_motion_feature_formal_mechanism_audit_seed2026"
        ),
        help="validation-only mechanism-audit output directory",
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/"
            "safety_motion_feature_formal_mechanism_audit_seed2026"
        ),
        help="mechanism-audit plot directory",
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--branches",
        nargs="+",
        choices=("baseline_37d", "baseline_41d", "budgeted_quality_aware_41d"),
        default=("baseline_37d", "baseline_41d", "budgeted_quality_aware_41d"),
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--uav-count", type=int, choices=(3,), default=3
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--steps", type=int, default=50, dest="max_steps"
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--measurement-overhead-enabled",
        action="store_true",
        default=True,
        help="require the frozen measurement-overhead configuration",
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--quality-logit-penalty", type=float, default=1.0
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--bootstrap-samples", type=int, default=10_000
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--bootstrap-seed", type=int, default=2026
    )
    safety_motion_formal_mechanism_parser.add_argument(
        "--device", choices=("cpu",), default="cpu"
    )
    safety_motion_formal_mechanism_parser.add_argument("--force", action="store_true")
    safety_motion_formal_mechanism_parser.add_argument("--quiet", action="store_true")

    safety_mechanism_parser = subparsers.add_parser(
        "run-safety-mechanism-audit",
        help=(
            "attribute safety collisions to actor observations, nominal actions, "
            "projection, and dynamic-obstacle timing"
        ),
    )
    safety_mechanism_parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "data/synthetic/"
            "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
        ),
        help="root containing frozen policy-seed/branch/scale_3 checkpoints",
    )
    safety_mechanism_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="directory containing frozen scenario_<seed>.json files",
    )
    safety_mechanism_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_mechanism_audit"
        ),
        help="diagnostic log directory; source episodes are never modified",
    )
    safety_mechanism_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_mechanism_audit"
        ),
        help="diagnostic plot directory",
    )
    safety_mechanism_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
        help="frozen policy checkpoint seeds",
    )
    safety_mechanism_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="frozen safety-scene seeds",
    )
    safety_mechanism_parser.add_argument(
        "--branches",
        nargs="+",
        default=("baseline_mappo", "budgeted_quality_aware_mappo"),
        choices=("baseline_mappo", "budgeted_quality_aware_mappo"),
        help="frozen policy branches to compare",
    )
    safety_mechanism_parser.add_argument(
        "--steps", type=int, default=50, dest="max_steps"
    )
    safety_mechanism_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_mechanism_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_mechanism_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_mechanism_parser.add_argument(
        "--force", action="store_true", help="replace these exact diagnostic outputs"
    )
    safety_mechanism_parser.add_argument(
        "--quiet", action="store_true", help="disable per-episode progress output"
    )

    safety_timing_parser = subparsers.add_parser(
        "run-safety-timing-counterfactual-audit",
        help=(
            "audit dynamic-obstacle collision timing with an offline one-step "
            "predicted-obstacle counterfactual"
        ),
    )
    safety_timing_parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "data/synthetic/"
            "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
        ),
        help="root containing frozen policy-seed/branch/scale_3 checkpoints",
    )
    safety_timing_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="directory containing frozen scenario_<seed>.json files",
    )
    safety_timing_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_timing_counterfactual"
        ),
        help="timing-audit CSV/JSON directory; source data is never modified",
    )
    safety_timing_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_timing_counterfactual"
        ),
        help="timing-audit plot directory",
    )
    safety_timing_parser.add_argument(
        "--reference-audit",
        type=Path,
        default=Path(
            "logs/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_mechanism_audit"
        ),
        help="prior safety mechanism audit used for replay signature checking",
    )
    safety_timing_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
        help="frozen policy checkpoint seeds",
    )
    safety_timing_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="frozen safety-scene seeds",
    )
    safety_timing_parser.add_argument(
        "--branches",
        nargs="+",
        default=("baseline_mappo", "budgeted_quality_aware_mappo"),
        choices=("baseline_mappo", "budgeted_quality_aware_mappo"),
        help="frozen policy branches to compare",
    )
    safety_timing_parser.add_argument(
        "--steps", type=int, default=50, dest="max_steps"
    )
    safety_timing_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    safety_timing_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_timing_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_timing_parser.add_argument(
        "--force", action="store_true", help="replace these exact diagnostic outputs"
    )
    safety_timing_parser.add_argument(
        "--quiet", action="store_true", help="disable per-episode progress output"
    )

    safety_observability_parser = subparsers.add_parser(
        "run-safety-observability-audit",
        help=(
            "audit dynamic-obstacle observability and multi-horizon safety "
            "envelope coverage from the frozen timing replay"
        ),
    )
    safety_observability_parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "data/synthetic/"
            "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
        ),
        help="root containing the frozen policy checkpoints (provenance only)",
    )
    safety_observability_parser.add_argument(
        "--scenario-root",
        type=Path,
        default=Path("data/synthetic/safety_risk_screening_seed2026/scenarios"),
        help="directory containing the frozen safety scenarios (provenance only)",
    )
    safety_observability_parser.add_argument(
        "--source-timing-audit",
        type=Path,
        default=Path(
            "logs/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_timing_counterfactual"
        ),
        dest="source_timing_audit_dir",
        help="frozen validation-only timing audit to read",
    )
    safety_observability_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_observability_audit"
        ),
        help="observability audit CSV/JSON directory",
    )
    safety_observability_parser.add_argument(
        "--plots-output",
        type=Path,
        default=Path(
            "plots/preliminary/safety_mappo_evaluation_seed2026/"
            "safety_observability_audit"
        ),
        help="observability audit plot directory",
    )
    safety_observability_parser.add_argument(
        "--policy-seeds",
        type=int,
        nargs="+",
        default=(7071, 8080, 9099, 10118, 11127),
        help="frozen policy checkpoint seeds",
    )
    safety_observability_parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(2026, 3035, 4044, 5053, 6062, 7071, 8080, 9089, 10098, 11107),
        help="frozen safety-scene seeds",
    )
    safety_observability_parser.add_argument(
        "--branches",
        nargs="+",
        default=("baseline_mappo", "budgeted_quality_aware_mappo"),
        choices=("baseline_mappo", "budgeted_quality_aware_mappo"),
        help="frozen policy branches to compare",
    )
    safety_observability_parser.add_argument(
        "--steps", type=int, default=50, dest="max_steps"
    )
    safety_observability_parser.add_argument(
        "--bootstrap-samples", type=int, default=10_000
    )
    safety_observability_parser.add_argument("--bootstrap-seed", type=int, default=2026)
    safety_observability_parser.add_argument("--device", choices=("cpu",), default="cpu")
    safety_observability_parser.add_argument(
        "--force", action="store_true", help="replace these exact diagnostic outputs"
    )
    safety_observability_parser.add_argument(
        "--quiet", action="store_true", help="disable audit progress output"
    )

    trend_parser = subparsers.add_parser("run-training-trend-diagnostic", help="frozen gradient coupling audit and gated smoke only")
    trend_parser.add_argument("--seed", type=int, default=42)
    trend_parser.add_argument("--stage", choices=("frozen-audit", "smoke"), default="frozen-audit")
    trend_parser.add_argument(
        "--audit-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves the historical audit; v2 adds read-only execution evidence",
    )

    ordered_diagnostic_parser = subparsers.add_parser(
        "run-ordered-v4-diagnostic",
        help="run the read-only P0-P3 diagnostic for the completed 3-UAV V4 development arms",
    )
    ordered_diagnostic_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/synthetic/mainline_core_framework_v4_corrected_seed42"),
    )
    ordered_diagnostic_parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs/preliminary/mainline_core_framework_v4_corrected_seed42"),
    )
    ordered_diagnostic_parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("plots/preliminary/mainline_core_framework_v4_corrected_seed42"),
    )
    ordered_diagnostic_parser.add_argument(
        "--output-name", default="ordered_diagnostic",
        help="new child directory name; a non-empty destination is never overwritten",
    )

    vo_interaction_parser = subparsers.add_parser(
        "run-vo-interaction-audit",
        help="run a validation-only audit of cooperative UWB benefit persistence and subsequent VO updates",
    )
    vo_interaction_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/synthetic/mainline_core_framework_v4_delta_scale_seed42"),
    )
    vo_interaction_parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs/preliminary/mainline_core_framework_v4_delta_scale_seed42"),
    )
    vo_interaction_parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("plots/preliminary/mainline_core_framework_v4_delta_scale_seed42"),
    )
    vo_interaction_parser.add_argument(
        "--output-name",
        default="vo_interaction_audit",
        help="new child directory name; a non-empty destination is never overwritten",
    )

    mainline_comparison_parser = subparsers.add_parser(
        "run-mainline-baseline-comparison",
        help="run the seed-42 weak/strong baseline comparison and validation figures",
    )
    mainline_comparison_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="single development seed; this entry intentionally accepts only 42",
    )
    mainline_comparison_parser.add_argument(
        "--stage",
        choices=("smoke", "development", "plot-only", "replay-audit"),
        default="smoke",
        help="smoke=2/5/5/5, development=20/100/20/20, plot-only=replot an existing root",
    )
    mainline_comparison_parser.add_argument("--output", type=Path, default=None)
    mainline_comparison_parser.add_argument(
        "--protocol",
        choices=("core-framework-v1", "core-framework-v2", "core-framework-v3", "core-framework-v4", "core-framework-v4-delta-scale", "core-framework-v5", "core-framework-v6", "core-framework-v7", "core-framework-v8", "core-framework-v9-action-selection", "core-framework-v10-safety", "core-framework-v11-comm-occluder", "legacy", "corrected-v2", "stabilized-v3", "stabilized-v4", "stabilized-v4-corrected"),
        default="core-framework-v3",
        help="core-framework-v8 first tests report-direct finite-horizon learning and then loc-progress-v2; older versioned choices are retained for historical reproduction",
    )
    mainline_comparison_parser.add_argument(
        "--experiment-arm",
        choices=(
            "reference",
            "actor-lr-half",
            "feedback-aligned",
            "proxy-aligned",
            "rollout4",
            "loc-progress",
            "actual-load",
            "loc-progress-actual-load",
            "loc-progress-v2",
            "report-clean",
            "bounded-collision",
            "action-linked-feedback",
            "action-selection-control",
            "action-head-v3",
            "action-linked-feedback-v2",
        ),
        default="reference",
    )
    mainline_comparison_parser.add_argument(
        "--figure-profile",
        choices=("legacy", "compact-mainline", "compact-proposed-v3", "compact-proposed-v4"),
        default="legacy",
    )
    mainline_comparison_parser.add_argument("--logs-output", type=Path, default=None)
    mainline_comparison_parser.add_argument("--plots-output", type=Path, default=None)
    mainline_comparison_parser.add_argument("--steps", type=int, default=50, dest="max_steps")
    mainline_comparison_parser.add_argument("--warmup-episodes", type=int, default=None)
    mainline_comparison_parser.add_argument("--train-episodes", type=int, default=None)
    mainline_comparison_parser.add_argument("--validation-episodes", type=int, default=None)
    mainline_comparison_parser.add_argument("--test-episodes", type=int, default=None)
    mainline_comparison_parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"),
    )
    mainline_comparison_parser.add_argument(
        "--ekf-config",
        type=Path,
        default=Path("data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"),
    )
    mainline_comparison_parser.add_argument(
        "--vo-quality-calibration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the warm-up-fitted VO quality calibration for all learned branches",
    )
    mainline_comparison_parser.add_argument(
        "--measurement-overhead-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain the existing 512-byte measurement overhead",
    )
    mainline_comparison_parser.add_argument(
        "--branch-invariant-initialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use paired module-level initialization streams",
    )
    mainline_comparison_parser.add_argument(
        "--measurement-update-eligibility-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "mask a source UAV until its per-link UWB update period elapses; "
            "off by default for historical compatibility"
        ),
    )
    mainline_comparison_parser.add_argument("--quality-logit-penalty", type=float, default=1.0)
    mainline_comparison_parser.add_argument("--proxy-teacher-coef", type=float, default=0.25)
    mainline_comparison_parser.add_argument("--proxy-temperature", type=float, default=0.25)
    mainline_comparison_parser.add_argument("--communication-cost-weight", type=float, default=1.0)
    mainline_comparison_parser.add_argument("--communication-cost-tolerance", type=float, default=0.05)
    mainline_comparison_parser.add_argument("--communication-dual-initial", type=float, default=0.0)
    mainline_comparison_parser.add_argument("--communication-dual-step-size", type=float, default=0.25)
    mainline_comparison_parser.add_argument("--communication-dual-max", type=float, default=5.0)
    mainline_comparison_parser.add_argument("--fixed-packet-cost-fraction", type=float, default=0.5)
    mainline_comparison_parser.add_argument(
        "--persistent-horizon",
        type=int,
        default=3,
        help="fixed prediction horizon for the stabilized-v4 proposed branch",
    )
    mainline_comparison_parser.add_argument("--force", action="store_true", help="replace only these new comparison roots")
    mainline_comparison_parser.add_argument("--quiet", action="store_true", help="disable progress output")

    preliminary_summary_parser = subparsers.add_parser(
        "summarize-preliminary",
        help="aggregate preliminary metrics and create scale or training-trend PNG plots",
    )
    preliminary_summary_parser.add_argument("path", type=Path)
    preliminary_summary_parser.add_argument("--logs-output", type=Path, default=None)
    preliminary_summary_parser.add_argument("--plots-output", type=Path, default=None)

    main_plot_parser = subparsers.add_parser(
        "plot-main",
        help="create the eight paper-oriented main figures from preliminary episodes",
    )
    main_plot_parser.add_argument("path", type=Path, help="preliminary output root")
    main_plot_parser.add_argument("--logs-output", type=Path, default=None)
    main_plot_parser.add_argument("--plots-output", type=Path, default=None)
    main_plot_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="omit validation records from the comparison and CDF figures",
    )
    main_plot_parser.add_argument(
        "--no-test",
        action="store_true",
        help="omit test records from the figures",
    )

    single_uav_diagnostics_parser = subparsers.add_parser(
        "plot-single-uav-diagnostics",
        help="create time-series and XY/XYZ diagnostic plots for one recorded single-UAV episode",
    )
    single_uav_diagnostics_parser.add_argument(
        "episode",
        type=Path,
        help="episode directory containing scenario.json/trajectory.csv and canonical episode files",
    )
    single_uav_diagnostics_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="diagnostic PNG output directory; defaults beside the experiment plots",
    )
    single_uav_diagnostics_parser.add_argument(
        "--force",
        action="store_true",
        help="allow writing into a non-empty diagnostic output directory",
    )

    preliminary_validate_parser = subparsers.add_parser(
        "validate-preliminary",
        help="validate every preliminary scale directory",
    )
    preliminary_validate_parser.add_argument("path", type=Path)

    import_parser = subparsers.add_parser(
        "import-flightmare", help="import a canonical Flightmare episode export"
    )
    import_parser.add_argument("source", type=Path, help="source episode directory")
    import_parser.add_argument("--output", type=Path, default=Path("data/synthetic/flightmare_legacy"))
    import_parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    import_parser.add_argument("--episode-id", required=True)
    import_parser.add_argument("--force", action="store_true")

    symbol_parser = subparsers.add_parser(
        "audit-symbols", help="check the canonical notation registry for collisions"
    )
    symbol_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("sections/2_system_model.md"),
        help="notation registry Markdown file",
    )

    return parser


def _resolve_preliminary_ekf_parameters(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, float], str]:
    from .experiments.formal import default_ekf_parameters, load_selected_ekf_config

    defaults = default_ekf_parameters()
    if args.ekf_config is not None:
        config_path = Path(args.ekf_config)
        try:
            selected = load_selected_ekf_config(config_path)
        except ValueError as exc:
            parser.error(str(exc))
        values = dict(selected)
        source = str(config_path.resolve())
    else:
        values = dict(defaults)
        source = "cli/defaults"

    cli_to_config = {
        "vo_std": "vo_position_std_m",
        "process_position_std": "process_position_std_m",
        "process_velocity_std": "process_velocity_std_mps",
        "nlos_variance_multiplier": "nlos_variance_multiplier",
    }
    for argument_name, parameter_name in cli_to_config.items():
        explicit = getattr(args, argument_name)
        if explicit is None:
            continue
        explicit = float(explicit)
        if not math.isfinite(explicit) or explicit <= 0.0:
            parser.error(f"--{argument_name.replace('_', '-')} must be finite and positive")
        if args.ekf_config is not None and not math.isclose(
            explicit,
            float(values[parameter_name]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            parser.error(
                f"--{argument_name.replace('_', '-')}={explicit} conflicts with "
                f"{parameter_name}={values[parameter_name]} in --ekf-config"
            )
        values[parameter_name] = explicit
    return values, source


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    if args.command == "list-datasets":
        if args.json:
            print(json.dumps(registry_as_dicts(), ensure_ascii=False, indent=2))
        else:
            print(render_registry())
        return 0
    if args.command == "generate-synthetic":
        if args.scenarios < 3 or args.steps < 2 or args.dt <= 0:
            parser.error("--scenarios must be >= 3, --steps >= 2, and --dt > 0")
        summary = generate_benchmark(
            SyntheticConfig(
                output_dir=args.output,
                seed=args.seed,
                scenarios=args.scenarios,
                steps=args.steps,
                dt=args.dt,
            ),
            allow_overwrite=args.force,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        report = validate_dataset_dir(args.path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "generate-episodes":
        config = preset_config(args.output, args.preset, seed=args.seed)
        summary = generate_episode_benchmark(config, allow_overwrite=args.force)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-episodes":
        report = validate_episode_benchmark(args.path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "run-pilot":
        from .experiments.pilot import PilotConfig, run_pilot

        summary = run_pilot(
            PilotConfig(
                output_dir=args.output,
                seed=args.seed,
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                integration_steps=args.integration_steps,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                device=args.device,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "calibrate-ekf":
        from .experiments.calibration import EKFCalibrationConfig, run_ekf_parameter_scan

        summary = run_ekf_parameter_scan(
            EKFCalibrationConfig(
                output_dir=args.output,
                seed=args.seed,
                uav_count=args.uav_count,
                episodes=args.episodes,
                max_steps=args.max_steps,
                vo_stds_m=tuple(args.vo_std),
                process_position_stds_m=tuple(args.process_position_std),
                process_velocity_stds_mps=tuple(args.process_velocity_std),
                nlos_variance_multipliers=tuple(args.nlos_variance_multiplier),
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-safety-motion-feature-smoke":
        from .experiments.safety_motion_feature_smoke import (
            SafetyMotionFeatureSmokeConfig,
            run_safety_motion_feature_smoke,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        summary = run_safety_motion_feature_smoke(
            SafetyMotionFeatureSmokeConfig(
                scenario_path=args.scenario,
                output_dir=args.output,
                logs_output=args.logs_output,
                plots_output=args.plots_output,
                uav_count=args.uav_count,
                seed=args.seed,
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                quality_logit_penalty=args.quality_logit_penalty,
                dynamic_obstacle_swept_projection_enabled=(
                    args.dynamic_obstacle_swept_projection_enabled
                ),
                device=args.device,
                force=args.force,
                progress=not args.quiet,
                command_line=command_line,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-training-trend-diagnostic":
        if args.audit_version == "v2":
            from .experiments.training_trend_diagnostic_v2 import run
        else:
            from .experiments.training_trend_diagnostic import run
        result = run(args.stage, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("audit_ok") else 1
    if args.command == "run-ordered-v4-diagnostic":
        from .experiments.ordered_v4_diagnostic import run

        result = run(
            data_root=args.data_root,
            logs_root=args.logs_root,
            plots_root=args.plots_root,
            output_name=args.output_name,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("audit_ok") else 1
    if args.command == "run-vo-interaction-audit":
        from .experiments.vo_interaction_audit import run

        result = run(
            data_root=args.data_root,
            logs_root=args.logs_root,
            plots_root=args.plots_root,
            output_name=args.output_name,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("audit_ok") else 1
    if args.command == "run-mainline-baseline-comparison":
        from .experiments.mainline_baseline_comparison import (
            MainlineBaselineComparisonConfig,
            run_mainline_baseline_comparison,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        counts = (
            args.warmup_episodes,
            args.train_episodes,
            args.validation_episodes,
            args.test_episodes,
        )
        if any(value is not None and int(value) < 1 for value in counts):
            parser.error("episode counts must be positive")
        comparison_config = MainlineBaselineComparisonConfig(
            output_dir=args.output,
            protocol=args.protocol,
            experiment_arm=args.experiment_arm,
            figure_profile=args.figure_profile,
            seed=args.seed,
            stage=args.stage,
            max_steps=args.max_steps,
            warmup_episodes=args.warmup_episodes,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
            scenario_profile_path=args.scenario_profile,
            ekf_config_path=args.ekf_config,
            logs_dir=args.logs_output,
            plots_dir=args.plots_output,
            visual_odometry_quality_calibration_enabled=bool(
                args.vo_quality_calibration_enabled
            ),
            measurement_overhead_enabled=bool(args.measurement_overhead_enabled),
            branch_invariant_initialization=bool(args.branch_invariant_initialization),
            measurement_update_eligibility_enabled=bool(
                args.measurement_update_eligibility_enabled
            ),
            quality_logit_penalty=args.quality_logit_penalty,
            proxy_teacher_coef=args.proxy_teacher_coef,
            proxy_temperature=args.proxy_temperature,
            communication_cost_weight=args.communication_cost_weight,
            communication_cost_tolerance=args.communication_cost_tolerance,
            communication_dual_initial=args.communication_dual_initial,
            communication_dual_step_size=args.communication_dual_step_size,
            communication_dual_max=args.communication_dual_max,
            fixed_packet_cost_fraction=args.fixed_packet_cost_fraction,
            persistent_horizon=args.persistent_horizon,
            force=args.force,
            progress=not args.quiet,
        )
        if comparison_config.stage == "development" and comparison_config.protocol in {
            "legacy",
            "core-framework-v2",
            "core-framework-v3",
            "core-framework-v4",
            "core-framework-v4-delta-scale",
            "core-framework-v5",
            "core-framework-v6",
            "core-framework-v7",
            "core-framework-v8",
            "core-framework-v9-action-selection",
            "core-framework-v10-safety",
            "core-framework-v11-comm-occluder",
        }:
            from .experiments.mainline_corrected_stages import find_compatible_smoke_report

            if comparison_config.protocol in {"core-framework-v2", "core-framework-v3", "core-framework-v4", "core-framework-v4-delta-scale", "core-framework-v5", "core-framework-v6", "core-framework-v7", "core-framework-v8", "core-framework-v9-action-selection", "core-framework-v10-safety", "core-framework-v11-comm-occluder"}:
                smoke_match = find_compatible_smoke_report(comparison_config)
            else:
                smoke_root = comparison_config.resolved_output_dir.parent / (
                    "mainline_baseline_comparison_smoke_seed42"
                )
                smoke_logs = Path("logs/preliminary") / smoke_root.name
                smoke_report_path = smoke_logs / "audit_report.json"
                smoke_match = None
                if smoke_report_path.is_file():
                    try:
                        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
                        smoke_match = (smoke_report_path, smoke_report) if bool(smoke_report.get("audit_ok", False)) else None
                    except (OSError, ValueError):
                        smoke_match = None
            if smoke_match is None:
                parser.error(
                    "development requires a completed seed=42 smoke report at "
                    f"{comparison_config.resolved_logs_dir.parent.parent / 'smoke' / comparison_config.experiment_arm}"
                )
        summary = run_mainline_baseline_comparison(comparison_config)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-preliminary":
        from .experiments.formal import (
            FormalExperimentConfig,
            fit_shared_training_metric_scales,
            run_preliminary,
            summarize_preliminary,
        )
        from .simulation.scenario_profile import load_scenario_profile

        ekf_parameters, ekf_source = _resolve_preliminary_ekf_parameters(args, parser)
        if args.scenario_profile is not None:
            if len(args.uav_counts) != 1 or int(args.uav_counts[0]) != 3:
                parser.error("--scenario-profile currently requires --uav-counts 3 only")
            try:
                load_scenario_profile(args.scenario_profile, expected_uav_count=3)
            except ValueError as exc:
                parser.error(str(exc))
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        base_config = FormalExperimentConfig(
            output_dir=args.output,
            uav_count=int(args.uav_counts[0]),
            seed=args.seed,
            scenario_seed=args.scenario_seed,
            max_steps=args.max_steps,
            warmup_episodes=args.warmup_episodes,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
            device=args.device,
            force=args.force,
            policy_epochs=args.policy_epochs,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            initial_delta_std=args.initial_delta_std,
            single_uav_learning_mode=args.single_uav_learning_mode,
            vo_position_std_m=ekf_parameters["vo_position_std_m"],
            process_position_std_m=ekf_parameters["process_position_std_m"],
            process_velocity_std_mps=ekf_parameters["process_velocity_std_mps"],
            nlos_variance_multiplier=ekf_parameters["nlos_variance_multiplier"],
            scenario_profile_path=args.scenario_profile,
            ekf_config_source=ekf_source,
            command_line=command_line,
            progress=not args.quiet,
            mappo_variant=args.mappo_variant,
            encoder_type=args.encoder_type,
            visual_odometry_quality_calibration_enabled=(
                bool(args.vo_quality_calibration_enabled)
            ),
        )
        shared_metric_scales = fit_shared_training_metric_scales(base_config, args.uav_counts)
        summaries = []
        for uav_count in args.uav_counts:
            summaries.append(
                run_preliminary(
                    replace(
                        base_config,
                        uav_count=uav_count,
                        metric_scales=shared_metric_scales,
                        metric_scale_source="pooled_training_warmup_p95",
                    )
                )
            )
        aggregate = summarize_preliminary(args.output)
        print(json.dumps({"scales": summaries, "aggregate": aggregate}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-baseline":
        from .experiments.baseline import IndependentBaselineConfig, run_independent_baseline

        summary = run_independent_baseline(
            IndependentBaselineConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                controller=args.controller,
                cruise_speed_mps=args.cruise_speed_mps,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "audit-causal-chain":
        from .experiments.causal_audit import CausalAuditConfig, run_causal_audit

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_causal_audit(
            CausalAuditConfig(
                reference_scale=args.reference,
                baseline_scale=args.baseline,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-objective-sensitivity-audit":
        from .experiments.objective_sensitivity_audit import (
            ObjectiveSensitivityAuditConfig,
            run_objective_sensitivity_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_objective_sensitivity_audit(
            ObjectiveSensitivityAuditConfig(
                reference_root=args.reference,
                logs_output=args.logs_output,
                plots_output=args.plots_output,
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-execution-uncertainty-proxy-audit":
        from .experiments.execution_uncertainty_proxy_audit import (
            ExecutionUncertaintyProxyAuditConfig,
            run_execution_uncertainty_proxy_audit,
        )

        summary = run_execution_uncertainty_proxy_audit(
            ExecutionUncertaintyProxyAuditConfig(
                reference_root=args.reference,
                logs_output=args.logs_output,
                plots_output=args.plots_output,
                train_split=args.train_split,
                eval_split=args.eval_split,
                age_variance_coefficient=args.age_variance_coefficient,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-execution-uncertainty-proxy-alignment-audit":
        from .experiments.execution_uncertainty_proxy_alignment_audit import (
            ExecutionUncertaintyProxyAlignmentAuditConfig,
            run_execution_uncertainty_proxy_alignment_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_execution_uncertainty_proxy_alignment_audit(
            ExecutionUncertaintyProxyAlignmentAuditConfig(
                reference_root=args.reference,
                logs_output=args.logs_output,
                plots_output=args.plots_output,
                train_split=args.train_split,
                eval_split=args.eval_split,
                age_variance_coefficient=args.age_variance_coefficient,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-bandwidth-intervention":
        from .experiments.bandwidth_intervention import (
            BandwidthInterventionConfig,
            run_bandwidth_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not 0.0 <= args.minimum_informative_fraction <= 1.0:
            parser.error("--minimum-informative-fraction must lie in [0, 1]")
        summary = run_bandwidth_intervention(
            BandwidthInterventionConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                minimum_informative_fraction=args.minimum_informative_fraction,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-localization-coupling-intervention":
        from .experiments.localization_coupling_intervention import (
            LocalizationCouplingInterventionConfig,
            run_localization_coupling_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_localization_coupling_intervention(
            LocalizationCouplingInterventionConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-ekf-update-strength-audit":
        from .experiments.ekf_update_strength_audit import (
            EKFUpdateStrengthAuditConfig,
            run_ekf_update_strength_audit,
        )

        summary = run_ekf_update_strength_audit(
            EKFUpdateStrengthAuditConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-measurement-density-intervention":
        from .experiments.measurement_density_intervention import (
            MeasurementDensityInterventionConfig,
            run_measurement_density_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not 0.0 <= args.minimum_selection_fraction <= 1.0:
            parser.error("--minimum-selection-fraction must lie in [0, 1]")
        summary = run_measurement_density_intervention(
            MeasurementDensityInterventionConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                minimum_selection_fraction=args.minimum_selection_fraction,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-cooperative-update-quality-audit":
        from .experiments.cooperative_update_quality_audit import (
            CooperativeUpdateQualityAuditConfig,
            run_cooperative_update_quality_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_cooperative_update_quality_audit(
            CooperativeUpdateQualityAuditConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-quality-aware-measurement-intervention":
        from .experiments.quality_aware_measurement_intervention import (
            QualityAwareMeasurementInterventionConfig,
            run_quality_aware_measurement_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_quality_aware_measurement_intervention(
            QualityAwareMeasurementInterventionConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-soft-quality-measurement-intervention":
        from .experiments.soft_quality_measurement_intervention import (
            SoftQualityMeasurementInterventionConfig,
            run_soft_quality_measurement_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_soft_quality_measurement_intervention(
            SoftQualityMeasurementInterventionConfig(
                reference_scale=args.reference,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-multiseed-soft-quality-validation":
        from .experiments.multiseed_soft_quality_validation import (
            MultiSeedSoftQualityValidationConfig,
            run_multiseed_soft_quality_validation,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_multiseed_soft_quality_validation(
            MultiSeedSoftQualityValidationConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-quality-aware-mappo":
        from .experiments.quality_aware_mappo import (
            QualityAwareMAPPOConfig,
            run_quality_aware_mappo,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_quality_aware_mappo(
            QualityAwareMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-rate-controlled-quality-mappo":
        from .experiments.measurement_rate_controlled_mappo import (
            MeasurementRateControlledMAPPOConfig,
            run_measurement_rate_controlled_mappo,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_measurement_rate_controlled_mappo(
            MeasurementRateControlledMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                measurement_overhead_enabled=True,
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                measurement_rate_constraint_coef=args.measurement_rate_coef,
                measurement_rate_tolerance=args.measurement_rate_tolerance,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-adaptive-budgeted-quality-mappo":
        from .experiments.adaptive_budgeted_quality_aware_mappo import (
            AdaptiveMeasurementRateMAPPOConfig,
            run_adaptive_budgeted_quality_aware_mappo,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not args.measurement_overhead_enabled:
            parser.error("--measurement-overhead-enabled is required for adaptive measurement-rate MAPPO")
        summary = run_adaptive_budgeted_quality_aware_mappo(
            AdaptiveMeasurementRateMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                measurement_overhead_enabled=True,
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                measurement_rate_tolerance=args.measurement_rate_tolerance,
                measurement_rate_dual_initial=args.dual_initial,
                measurement_rate_dual_step_size=args.dual_step_size,
                measurement_rate_dual_max=args.dual_max,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-proxy-guided-adaptive-quality-mappo":
        from .experiments.proxy_guided_adaptive_budgeted_quality_aware_mappo import (
            ProxyGuidedAdaptiveMAPPOConfig,
            run_proxy_guided_adaptive_budgeted_quality_aware_mappo,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not args.measurement_overhead_enabled:
            parser.error("--measurement-overhead-enabled is required for proxy-guided MAPPO")
        summary = run_proxy_guided_adaptive_budgeted_quality_aware_mappo(
            ProxyGuidedAdaptiveMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                measurement_overhead_enabled=True,
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                measurement_rate_tolerance=args.measurement_rate_tolerance,
                measurement_rate_dual_initial=args.dual_initial,
                measurement_rate_dual_step_size=args.dual_step_size,
                measurement_rate_dual_max=args.dual_max,
                proxy_age_variance_coefficient=args.proxy_age_variance_coefficient,
                proxy_teacher_coef=args.proxy_teacher_coef,
                proxy_temperature=args.proxy_temperature,
                 communication_cost_weight=args.communication_cost_weight,
                 branch_invariant_initialization=bool(args.branch_invariant_initialization),
                 force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-communication-constrained-proxy-mappo":
        from .experiments.communication_constrained_proxy_mappo import (
            CommunicationConstrainedProxyMAPPOConfig,
            run_communication_constrained_proxy_mappo,
        )

        if not args.measurement_overhead_enabled:
            parser.error("--measurement-overhead-enabled is required")
        summary = run_communication_constrained_proxy_mappo(
            CommunicationConstrainedProxyMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                visual_odometry_quality_calibration_enabled=(
                    bool(args.vo_quality_calibration_enabled)
                ),
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                proxy_teacher_coef=args.proxy_teacher_coef,
                proxy_temperature=args.proxy_temperature,
                communication_cost_weight=args.communication_cost_weight,
                communication_cost_tolerance=args.communication_cost_tolerance,
                communication_dual_initial=args.communication_dual_initial,
                communication_dual_step_size=args.communication_dual_step_size,
                communication_dual_max=args.communication_dual_max,
                 fixed_packet_cost_fraction=args.fixed_packet_cost_fraction,
                 branch_invariant_initialization=bool(args.branch_invariant_initialization),
                 force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-persistent-proxy-guided-quality-mappo":
        from .experiments.persistent_proxy_guided_quality_mappo import (
            PersistentProxyMAPPOConfig,
            run_persistent_proxy_guided_quality_mappo,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not args.measurement_overhead_enabled:
            parser.error("--measurement-overhead-enabled is required for persistent-proxy MAPPO")
        if not math.isfinite(args.age_variance_coefficient) or args.age_variance_coefficient < 0.0:
            parser.error("--age-variance-coefficient must be finite and non-negative")
        if args.persistent_horizon < 1:
            parser.error("--persistent-horizon must be positive")
        summary = run_persistent_proxy_guided_quality_mappo(
            PersistentProxyMAPPOConfig(
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                measurement_overhead_enabled=True,
                measurement_quality_logit_penalty=args.quality_logit_penalty,
                persistent_horizon=args.persistent_horizon,
                proxy_age_variance_coefficient=args.age_variance_coefficient,
                proxy_teacher_coef=args.proxy_teacher_coef,
                proxy_temperature=args.proxy_temperature,
                communication_cost_weight=args.communication_cost_weight,
                communication_cost_tolerance=args.communication_cost_tolerance,
                communication_dual_initial=args.communication_dual_initial,
                communication_dual_step_size=args.communication_dual_step_size,
                communication_dual_max=args.communication_dual_max,
                fixed_packet_cost_fraction=args.fixed_packet_cost_fraction,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-communication-cost-audit":
        from .experiments.communication_constrained_proxy_mappo import (
            MeasurementCommunicationCostAuditConfig,
            run_measurement_communication_cost_audit,
        )

        summary = run_measurement_communication_cost_audit(
            MeasurementCommunicationCostAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                fixed_packet_cost_fraction=args.fixed_packet_cost_fraction,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-persistent-proxy-audit":
        from .experiments.persistent_proxy_guided_quality_mappo import (
            PersistentProxyAuditConfig,
            run_persistent_proxy_audit,
        )

        if args.persistent_horizon < 1:
            parser.error("--persistent-horizon must be positive")
        if not math.isfinite(args.age_variance_coefficient) or args.age_variance_coefficient < 0.0:
            parser.error("--age-variance-coefficient must be finite and non-negative")
        if not 0.0 <= args.fixed_packet_cost_fraction <= 1.0:
            parser.error("--fixed-packet-cost-fraction must be in [0, 1]")
        summary = run_persistent_proxy_audit(
            PersistentProxyAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                persistent_horizon=args.persistent_horizon,
                age_variance_coefficient=args.age_variance_coefficient,
                fixed_packet_cost_fraction=args.fixed_packet_cost_fraction,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-persistent-proxy-mechanism-audit":
        from .experiments.persistent_proxy_mechanism_audit import (
            PersistentProxyMechanismAuditConfig,
            run_persistent_proxy_mechanism_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_persistent_proxy_mechanism_audit(
            PersistentProxyMechanismAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-proxy-teacher-audit":
        from .experiments.proxy_guided_adaptive_budgeted_quality_aware_mappo import (
            ProxyTeacherAuditConfig,
            run_proxy_teacher_audit,
        )

        if args.temperature <= 0.0:
            parser.error("--temperature must be positive")
        summary = run_proxy_teacher_audit(
            ProxyTeacherAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                age_variance_coefficient=args.age_variance_coefficient,
                communication_cost_weight=args.communication_cost_weight,
                temperature=args.temperature,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-rate-constraint-audit":
        from .experiments.measurement_rate_constraint_audit import (
            MeasurementRateConstraintAuditConfig,
            run_measurement_rate_constraint_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_measurement_rate_constraint_audit(
            MeasurementRateConstraintAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                source_logs_dir=args.source_logs,
                seeds=tuple(int(seed) for seed in args.seeds),
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-rate-sampling-audit":
        from .experiments.measurement_rate_sampling_audit import (
            MeasurementRateSamplingAuditConfig,
            run_measurement_rate_sampling_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_measurement_rate_sampling_audit(
            MeasurementRateSamplingAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                source_logs_dir=args.source_logs,
                seeds=tuple(int(seed) for seed in args.seeds),
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-rate-constraint-mechanism-audit":
        from .experiments.measurement_rate_constraint_mechanism_audit import (
            MeasurementRateConstraintMechanismAuditConfig,
            run_measurement_rate_constraint_mechanism_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_measurement_rate_constraint_mechanism_audit(
            MeasurementRateConstraintMechanismAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                source_logs_dir=args.source_logs,
                seeds=tuple(int(seed) for seed in args.seeds),
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-information-gain-budget-intervention":
        from .experiments.information_gain_budget_intervention import (
            InformationGainBudgetInterventionConfig,
            run_information_gain_budget_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not math.isfinite(args.age_variance_coefficient) or args.age_variance_coefficient < 0.0:
            parser.error("--age-variance-coefficient must be finite and non-negative")
        summary = run_information_gain_budget_intervention(
            InformationGainBudgetInterventionConfig(
                reference_root=args.reference,
                reference_branch=args.reference_branch,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                age_variance_coefficient=args.age_variance_coefficient,
                seeds=tuple(args.seeds) if args.seeds is not None else None,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-overhead-intervention":
        from .experiments.measurement_overhead_intervention import (
            MeasurementOverheadInterventionConfig,
            run_measurement_overhead_intervention,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not math.isfinite(args.age_variance_coefficient) or args.age_variance_coefficient < 0.0:
            parser.error("--age-variance-coefficient must be finite and non-negative")
        summary = run_measurement_overhead_intervention(
            MeasurementOverheadInterventionConfig(
                reference_root=args.reference,
                reference_branch=args.reference_branch,
                output_dir=args.output,
                split=args.split,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                age_variance_coefficient=args.age_variance_coefficient,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-policy-mechanism-audit":
        from .experiments.policy_mechanism_audit import (
            PolicyMechanismAuditConfig,
            run_policy_mechanism_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_policy_mechanism_audit(
            PolicyMechanismAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                plots_dir=args.plots_output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                device=args.device,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-proxy-guided-update-quality-audit":
        from .experiments.proxy_update_quality_audit import (
            ProxyUpdateQualityAuditConfig,
            run_proxy_guided_update_quality_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if not math.isfinite(args.age_variance_coefficient) or args.age_variance_coefficient < 0.0:
            parser.error("--age-variance-coefficient must be finite and non-negative")
        if not math.isfinite(args.communication_cost_weight) or args.communication_cost_weight < 0.0:
            parser.error("--communication-cost-weight must be finite and non-negative")
        if not math.isfinite(args.temperature) or args.temperature <= 0.0:
            parser.error("--temperature must be finite and positive")
        summary = run_proxy_guided_update_quality_audit(
            ProxyUpdateQualityAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                age_variance_coefficient=args.age_variance_coefficient,
                communication_cost_weight=args.communication_cost_weight,
                temperature=args.temperature,
                device=args.device,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-metric-consistency-audit":
        from .experiments.metric_consistency_audit import (
            MetricConsistencyAuditConfig,
            run_metric_consistency_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if args.expected_episodes < 1:
            parser.error("--expected-episodes must be positive")
        summary = run_metric_consistency_audit(
            MetricConsistencyAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                expected_episodes=args.expected_episodes,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-final-mainline-audit":
        from .experiments.final_mainline_audit import (
            FinalMainlineAuditConfig,
            run_final_mainline_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if args.expected_episodes < 1:
            parser.error("--expected-episodes must be positive")
        if args.expected_training_updates < 1:
            parser.error("--expected-training-updates must be positive")
        summary = run_final_mainline_audit(
            FinalMainlineAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                expected_episodes=args.expected_episodes,
                expected_training_updates=args.expected_training_updates,
                figure_profile=args.figure_profile,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-mainline-curve-cause-audit":
        from .experiments.mainline_curve_cause_audit import (
            MainlineCurveCauseAuditConfig,
            run_mainline_curve_cause_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        if args.expected_training_updates < 1 or args.expected_validation_episodes < 1:
            parser.error("expected update and validation episode counts must be positive")
        summary = run_mainline_curve_cause_audit(
            MainlineCurveCauseAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                source_logs_dir=args.source_logs,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                expected_training_updates=args.expected_training_updates,
                expected_validation_episodes=args.expected_validation_episodes,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-ekf-update-persistence-audit":
        from .experiments.ekf_update_persistence_audit import (
            EKFUpdatePersistenceAuditConfig,
            run_ekf_update_persistence_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_ekf_update_persistence_audit(
            EKFUpdatePersistenceAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                horizons=tuple(int(horizon) for horizon in args.horizons),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-ekf-sensor-interaction-audit":
        from .experiments.ekf_sensor_interaction_audit import (
            EKFSensorInteractionAuditConfig,
            run_ekf_sensor_interaction_audit,
        )

        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_ekf_sensor_interaction_audit(
            EKFSensorInteractionAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                horizons=tuple(int(horizon) for horizon in args.horizons),
                expected_episodes=args.expected_episodes,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-measurement-policy-stability-audit":
        from .experiments.measurement_policy_stability_audit import (
            MeasurementPolicyStabilityAuditConfig,
            run_measurement_policy_stability_audit,
        )

        summary = run_measurement_policy_stability_audit(
            MeasurementPolicyStabilityAuditConfig(
                reference_root=args.reference,
                mechanism_audit_root=args.mechanism_audit,
                output_dir=args.output,
                plots_dir=args.plots_output,
                seeds=tuple(int(seed) for seed in args.seeds),
                branches=tuple(str(branch) for branch in args.branches),
                split=args.split,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-quality-feature-sensitivity-audit":
        from .experiments.quality_feature_sensitivity_audit import (
            QualityFeatureSensitivityAuditConfig,
            run_quality_feature_sensitivity_audit,
        )

        summary = run_quality_feature_sensitivity_audit(
            QualityFeatureSensitivityAuditConfig(
                reference_root=args.reference,
                output_dir=args.output,
                seeds=tuple(int(seed) for seed in args.seeds),
                split=args.split,
                plots_dir=args.plots_output,
                device=args.device,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "screen-scenarios":
        from .experiments.formal import default_ekf_parameters, load_selected_ekf_config
        from .experiments.scenario_screening import ScenarioScreeningConfig, run_scenario_screening

        if args.design_points < 2 or args.episodes_per_point < 1 or args.max_steps < 1:
            parser.error("--design-points must be >= 2, --episodes-per-point >= 1, and --steps >= 1")
        ekf_parameters = (
            load_selected_ekf_config(args.ekf_config)
            if args.ekf_config is not None
            else default_ekf_parameters()
        )
        scenario_seeds = tuple(
            int(args.design_seed + 1009 * index)
            for index in range(args.episodes_per_point)
        )
        summary = run_scenario_screening(
            ScenarioScreeningConfig(
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                design_seed=args.design_seed,
                scenario_seeds=scenario_seeds,
                design_points=args.design_points,
                max_steps=args.max_steps,
                cruise_speed_mps=args.cruise_speed_mps,
                force=args.force,
                progress=not args.quiet,
                vo_position_std_m=ekf_parameters["vo_position_std_m"],
                process_position_std_m=ekf_parameters["process_position_std_m"],
                process_velocity_std_mps=ekf_parameters["process_velocity_std_mps"],
                nlos_variance_multiplier=ekf_parameters["nlos_variance_multiplier"],
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "run-safety-risk-screening":
        from .experiments.safety_risk_screening import (
            SafetyRiskScreeningConfig,
            run_safety_risk_screening,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        summary = run_safety_risk_screening(
            SafetyRiskScreeningConfig(
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                max_steps=args.max_steps,
                cruise_speed_mps=args.cruise_speed_mps,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                include_unprotected_diagnostic=not args.no_projection_diagnostic,
                dynamic_obstacle_swept_projection_enabled=(
                    args.dynamic_obstacle_swept_projection_enabled
                ),
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        # ``audit_ok`` is the structural execution check.  A screening gate
        # is reported in the audit JSON and may intentionally fail while the
        # experiment remains a valid diagnostic run.
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-mappo-evaluation":
        from .experiments.safety_mappo_evaluation import (
            SafetyMAPPOEvaluationConfig,
            run_safety_mappo_evaluation,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_safety_mappo_evaluation(
            SafetyMAPPOEvaluationConfig(
                model_root=args.model_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                branches=tuple(str(branch) for branch in args.branches),
                max_steps=args.max_steps,
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-motion-feature-evaluation":
        from .experiments.safety_motion_feature_evaluation import (
            SafetyMotionFeatureEvaluationConfig,
            run_safety_motion_feature_evaluation,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        summary = run_safety_motion_feature_evaluation(
            SafetyMotionFeatureEvaluationConfig(
                checkpoint_root=args.checkpoint_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                branches=tuple(str(branch) for branch in args.branches),
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                uav_count=args.uav_count,
                max_steps=args.max_steps,
                device=args.device,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                quality_logit_penalty=args.quality_logit_penalty,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
                command_line=command_line,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-motion-feature-formal-training":
        from .experiments.safety_motion_feature_formal_training import (
            SafetyMotionFeatureFormalTrainingConfig,
            run_safety_motion_feature_formal_training,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if min(
            args.warmup_episodes,
            args.train_episodes,
            args.validation_episodes,
            args.test_episodes,
        ) < 1:
            parser.error("all episode counts must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        summary = run_safety_motion_feature_formal_training(
            SafetyMotionFeatureFormalTrainingConfig(
                scenario_pool_root=args.scenario_pool_root,
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                uav_count=args.uav_count,
                max_steps=args.max_steps,
                warmup_episodes=args.warmup_episodes,
                train_episodes=args.train_episodes,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                scenario_profile_path=args.scenario_profile,
                ekf_config_path=args.ekf_config,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                quality_logit_penalty=args.quality_logit_penalty,
                dynamic_obstacle_swept_projection_enabled=(
                    args.dynamic_obstacle_swept_projection_enabled
                ),
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
                command_line=command_line,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-motion-feature-formal-evaluation":
        from .experiments.safety_motion_feature_formal_evaluation import (
            SafetyMotionFeatureFormalEvaluationConfig,
            run_safety_motion_feature_formal_evaluation,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        summary = run_safety_motion_feature_formal_evaluation(
            SafetyMotionFeatureFormalEvaluationConfig(
                checkpoint_root=args.checkpoint_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                logs_dir=args.logs_output,
                plots_dir=args.plots_output,
                branches=tuple(str(branch) for branch in args.branches),
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                uav_count=args.uav_count,
                max_steps=args.max_steps,
                device=args.device,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                quality_logit_penalty=args.quality_logit_penalty,
                dynamic_obstacle_swept_projection_enabled=(
                    args.dynamic_obstacle_swept_projection_enabled
                ),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
                command_line=command_line,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-motion-feature-formal-mechanism-audit":
        from .experiments.safety_motion_feature_formal_mechanism_audit import (
            SafetyMotionFeatureFormalMechanismAuditConfig,
            run_safety_motion_feature_formal_mechanism_audit,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        command_line = subprocess.list2cmdline([sys.executable, "main.py", *raw_argv])
        summary = run_safety_motion_feature_formal_mechanism_audit(
            SafetyMotionFeatureFormalMechanismAuditConfig(
                checkpoint_root=args.checkpoint_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                plots_dir=args.plots_output,
                branches=tuple(str(branch) for branch in args.branches),
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                uav_count=args.uav_count,
                max_steps=args.max_steps,
                device=args.device,
                measurement_overhead_enabled=args.measurement_overhead_enabled,
                quality_logit_penalty=args.quality_logit_penalty,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
                command_line=command_line,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-mechanism-audit":
        from .experiments.safety_mechanism_audit import (
            SafetyMechanismAuditConfig,
            run_safety_mechanism_audit,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_safety_mechanism_audit(
            SafetyMechanismAuditConfig(
                model_root=args.model_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                plots_dir=args.plots_output,
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                branches=tuple(str(branch) for branch in args.branches),
                max_steps=args.max_steps,
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-timing-counterfactual-audit":
        from .experiments.safety_timing_counterfactual import (
            SafetyTimingCounterfactualConfig,
            run_safety_timing_counterfactual,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_safety_timing_counterfactual(
            SafetyTimingCounterfactualConfig(
                model_root=args.model_root,
                scenario_root=args.scenario_root,
                output_dir=args.output,
                plots_dir=args.plots_output,
                reference_audit_dir=args.reference_audit,
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                branches=tuple(str(branch) for branch in args.branches),
                max_steps=args.max_steps,
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "run-safety-observability-audit":
        from .experiments.safety_observability_audit import (
            SafetyObservabilityAuditConfig,
            run_safety_observability_audit,
        )

        if args.max_steps < 1:
            parser.error("--steps must be positive")
        if args.bootstrap_samples < 100:
            parser.error("--bootstrap-samples must be at least 100")
        summary = run_safety_observability_audit(
            SafetyObservabilityAuditConfig(
                model_root=args.model_root,
                scenario_root=args.scenario_root,
                source_timing_audit_dir=args.source_timing_audit_dir,
                output_dir=args.output,
                plots_dir=args.plots_output,
                policy_seeds=tuple(int(seed) for seed in args.policy_seeds),
                scenario_seeds=tuple(int(seed) for seed in args.scenario_seeds),
                branches=tuple(str(branch) for branch in args.branches),
                max_steps=args.max_steps,
                device=args.device,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                force=args.force,
                progress=not args.quiet,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("audit_ok", False) else 1
    if args.command == "summarize-preliminary":
        from .experiments.formal import summarize_preliminary

        summary = summarize_preliminary(
            args.path,
            logs_output=args.logs_output,
            plots_output=args.plots_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plot-main":
        from .experiments.plotting import plot_main_figures

        summary = plot_main_figures(
            args.path,
            output_dir=args.plots_output,
            logs_output=args.logs_output,
            include_validation=not args.no_validation,
            include_test=not args.no_test,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plot-single-uav-diagnostics":
        from .experiments.single_uav_diagnostics import (
            SingleUAVDiagnosticsConfig,
            plot_single_uav_diagnostics,
        )

        episode_path = Path(args.episode)
        output_dir = args.output
        if output_dir is None:
            experiment_root = episode_path.parent.parent.parent
            output_dir = (
                Path("plots/preliminary")
                / experiment_root.name
                / f"single_uav_diagnostics_{episode_path.name}"
            )
        summary = plot_single_uav_diagnostics(
            SingleUAVDiagnosticsConfig(
                episode_path=episode_path,
                output_dir=output_dir,
                force=args.force,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-preliminary":
        from .experiments.formal import validate_preliminary

        report = validate_preliminary(args.path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "import-flightmare":
        summary = import_flightmare_episode(
            source_episode=args.source,
            output_root=args.output,
            split=args.split,
            episode_id=args.episode_id,
            allow_overwrite=args.force,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-symbols":
        report = audit_symbol_table(args.path)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1
    parser.print_help()
    return 0
