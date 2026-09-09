"""Bounded R/P/T training and validation, with no implicit coefficient search."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

from .mainline_baseline_comparison import (
    ACTIVE_LEARNED_METHODS,
    ACTIVE_MAINLINE_METHODS,
    DEPRECATED_METHODS,
    METHODS,
    FIXED_CONTROL_METHODS,
    BASELINE_BRANCH,
    COMMUNICATION_BRANCH,
    PAPER_BRANCH_LABELS,
)
from .baseline import _read_json, _read_csv, _sha256, _reference_formal_config
from .mainline_corrected import HISTORICAL_ROOT
from .formal import (
    FormalExperimentConfig,
    run_preliminary,
    fit_shared_training_metric_scales,
    _episode_metrics,
    load_selected_ekf_config,
)
from .mainline_baseline_comparison import _write_table, _load_scenario_episodes
from ..synthetic.episode_generator import _write_json

LEARNERS = (BASELINE_BRANCH, "mappo_mlp", COMMUNICATION_BRANCH)
CORE_METHODS = ACTIVE_MAINLINE_METHODS
CORE_LEARNERS = ACTIVE_LEARNED_METHODS
V3_METHODS = tuple(FIXED_CONTROL_METHODS) + (BASELINE_BRANCH,)
V3_LEARNERS = (BASELINE_BRANCH,)
V4_METHODS = V3_METHODS
V4_LEARNERS = V3_LEARNERS
V5_METHODS = V4_METHODS
V5_LEARNERS = V4_LEARNERS
V6_METHODS = V5_METHODS
V6_LEARNERS = V5_LEARNERS
V7_METHODS = V6_METHODS
V7_LEARNERS = V6_LEARNERS
V8_METHODS = V7_METHODS
V8_LEARNERS = V7_LEARNERS
V9_METHODS = V3_METHODS
V9_LEARNERS = V3_LEARNERS
V10_METHODS = V3_METHODS
V10_LEARNERS = V3_LEARNERS
# v11 keeps the v10 joint-safety contract but gives the moving LOS screen a
# communication-only semantic role.  It is intentionally a new protocol so
# historical v10 artifacts remain reproducible.
V11_METHODS = V3_METHODS
V11_LEARNERS = V3_LEARNERS
LOG_ROOT = Path("logs/preliminary/mainline_baseline_comparison_v2_seed42")
DATA_ROOT = Path("data/synthetic/mainline_baseline_comparison_v2_seed42")
V3_REFERENCE_ROOT = Path("data/synthetic/mainline_core_framework_v3_seed42/development/reference")


def _methods_for(config):
    """Return the methods belonging to the selected paper protocol.

    Historical protocols retain their complete method registry.  The core
    framework protocol deliberately excludes the deprecated proxy/constraint
    branch and treats ``baseline_mappo`` as the proposed Local-EKF-GAT-MAPPO
    method.
    """

    if config.protocol == "core-framework-v3":
        return V3_METHODS
    if config.protocol in {"core-framework-v4", "core-framework-v4-delta-scale"}:
        return V4_METHODS
    if config.protocol == "core-framework-v5":
        return V5_METHODS
    if config.protocol == "core-framework-v6":
        return V6_METHODS
    if config.protocol == "core-framework-v7":
        return V7_METHODS
    if config.protocol == "core-framework-v8":
        return V8_METHODS
    if config.protocol == "core-framework-v9-action-selection":
        return V9_METHODS
    if config.protocol == "core-framework-v10-safety":
        return V10_METHODS
    if config.protocol == "core-framework-v11-comm-occluder":
        return V11_METHODS
    return CORE_METHODS if config.protocol in {"core-framework-v1", "core-framework-v2"} else METHODS


def _learners_for(config):
    if config.protocol in {"core-framework-v3", "core-framework-v4", "core-framework-v4-delta-scale"}:
        return V3_LEARNERS
    if config.protocol == "core-framework-v5":
        return V5_LEARNERS
    if config.protocol == "core-framework-v6":
        return V6_LEARNERS
    if config.protocol == "core-framework-v7":
        return V7_LEARNERS
    if config.protocol == "core-framework-v8":
        return V8_LEARNERS
    if config.protocol == "core-framework-v9-action-selection":
        return V9_LEARNERS
    if config.protocol == "core-framework-v10-safety":
        return V10_LEARNERS
    if config.protocol == "core-framework-v11-comm-occluder":
        return V11_LEARNERS
    return CORE_LEARNERS if config.protocol in {"core-framework-v1", "core-framework-v2"} else LEARNERS


def _roots(config):
    """Resolve versioned roots without changing the corrected-v2 archive."""

    if config.protocol == "core-framework-v1":
        return (
            Path("logs/preliminary/mainline_core_framework_v1_seed42"),
            Path("data/synthetic/mainline_core_framework_v1_seed42"),
        )
    if config.protocol == "core-framework-v2":
        return (
            Path("logs/preliminary/mainline_core_framework_v2_seed42"),
            Path("data/synthetic/mainline_core_framework_v2_seed42"),
        )
    if config.protocol == "core-framework-v3":
        return (
            Path("logs/preliminary/mainline_core_framework_v3_seed42"),
            Path("data/synthetic/mainline_core_framework_v3_seed42"),
        )
    if config.protocol == "core-framework-v4":
        return (
            Path("logs/preliminary/mainline_core_framework_v4_seed42"),
            Path("data/synthetic/mainline_core_framework_v4_seed42"),
        )
    if config.protocol == "core-framework-v4-delta-scale":
        return (
            Path("logs/preliminary/mainline_core_framework_v4_delta_scale_seed42"),
            Path("data/synthetic/mainline_core_framework_v4_delta_scale_seed42"),
        )
    if config.protocol == "core-framework-v5":
        return (
            Path("logs/preliminary/mainline_core_framework_v5_seed42"),
            Path("data/synthetic/mainline_core_framework_v5_seed42"),
        )
    if config.protocol == "core-framework-v6":
        return (
            Path("logs/preliminary/mainline_reward_rebalanced_seed42"),
            Path("data/synthetic/mainline_reward_rebalanced_seed42"),
        )
    if config.protocol == "core-framework-v7":
        return (
            Path("logs/preliminary/mainline_core_framework_v7_seed42"),
            Path("data/synthetic/mainline_core_framework_v7_seed42"),
        )
    if config.protocol == "core-framework-v8":
        return (
            Path("logs/preliminary/mainline_core_framework_v8_seed42"),
            Path("data/synthetic/mainline_core_framework_v8_seed42"),
        )
    if config.protocol == "core-framework-v9-action-selection":
        return (
            Path("logs/preliminary/mainline_core_framework_v9_action_selection_seed42"),
            Path("data/synthetic/mainline_core_framework_v9_action_selection_seed42"),
        )
    if config.protocol == "core-framework-v10-safety":
        return (
            Path("logs/preliminary/mainline_core_framework_v10_safety_seed42"),
            Path("data/synthetic/mainline_core_framework_v10_safety_seed42"),
        )
    if config.protocol == "core-framework-v11-comm-occluder":
        return (
            Path("logs/preliminary/mainline_core_framework_v11_comm_occluder_seed42"),
            Path("data/synthetic/mainline_core_framework_v11_comm_occluder_seed42"),
        )
    if config.protocol == "stabilized-v3":
        return (
            Path("logs/preliminary/mainline_baseline_comparison_v3_seed42"),
            Path("data/synthetic/mainline_baseline_comparison_v3_seed42"),
        )
    if config.protocol in {"stabilized-v4", "stabilized-v4-corrected"}:
        return (
            Path("logs/preliminary/mainline_baseline_comparison_v4_corrected_seed42")
            if config.protocol == "stabilized-v4-corrected"
            else Path("logs/preliminary/mainline_baseline_comparison_v4_seed42"),
            Path("data/synthetic/mainline_baseline_comparison_v4_corrected_seed42")
            if config.protocol == "stabilized-v4-corrected"
            else Path("data/synthetic/mainline_baseline_comparison_v4_seed42"),
        )
    return LOG_ROOT, DATA_ROOT


def _replay_log_root(config):
    """Use the already verified historical replay as a read-only prerequisite."""

    log_root, _data_root = _roots(config)
    candidate = log_root / "replay-audit/reference/audit_report.json"
    if candidate.is_file():
        return log_root
    # The stable-v3 run changes only learning targets and PPO construction;
    # its physical replay prerequisite is the immutable corrected-v2 audit.
    return LOG_ROOT


def configuration_signature(config):
    payload = asdict(config)
    payload.pop("rule_reference_scale", None)
    for key in ("stage", "output_dir", "logs_dir", "plots_dir", "force", "progress", "warmup_episodes", "train_episodes", "validation_episodes", "test_episodes"):
        payload.pop(key, None)
    payload["vo_mode"] = "fixed_nominal"
    payload["implementation_hashes"] = {str(p):_sha256(p) for p in [
        Path("src/learning/mappo.py"), Path("src/simulation/environment.py"),
        Path("src/simulation/topology.py"), Path("src/experiments/formal.py"),
        Path("src/experiments/mainline_corrected_stages.py"),
        Path("src/experiments/mainline_baseline_comparison.py"),
        Path("src/experiments/action_selection_diagnostic.py"),
    ]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def find_compatible_smoke_report(config):
    """Find a completed Smoke for an explicitly named rerun root.

    A Smoke report is tied to the implementation/configuration signature.  A
    source edit after an earlier Smoke must not be silently reused, but an
    already completed report should also not be overwritten merely because a
    retry needs a distinct output directory.  Prefer the canonical
    ``smoke/<arm>`` location and then accept a same-arm suffixed retry whose
    signature matches the current configuration exactly.
    """

    expected_signature = configuration_signature(config)
    smoke_root = config.resolved_logs_dir.parent.parent / "smoke"
    candidates = []
    canonical = smoke_root / config.experiment_arm / "audit_report.json"
    if canonical.is_file():
        candidates.append(canonical)
    if smoke_root.is_dir():
        prefix = f"{config.experiment_arm}-"
        for directory in sorted(smoke_root.iterdir(), key=lambda path: path.name):
            if directory.is_dir() and directory.name.startswith(prefix):
                report = directory / "audit_report.json"
                if report.is_file() and report not in candidates:
                    candidates.append(report)
    for report_path in candidates:
        try:
            report = _read_json(report_path)
        except (OSError, ValueError):
            continue
        if bool(report.get("audit_ok")) and report.get("configuration_signature") == expected_signature:
            return report_path, report
    return None


def find_stage_report(log_root, stage, arm):
    """Return an auditable report for ``stage/arm`` or a suffixed retry."""

    stage_root = Path(log_root) / stage
    candidates = []
    canonical = stage_root / arm / "audit_report.json"
    if canonical.is_file():
        candidates.append(canonical)
    if stage_root.is_dir():
        prefix = f"{arm}-"
        for directory in sorted(stage_root.iterdir(), key=lambda path: path.name):
            if directory.is_dir() and directory.name.startswith(prefix):
                report = directory / "audit_report.json"
                if report.is_file() and report not in candidates:
                    candidates.append(report)
    for report_path in candidates:
        try:
            report = _read_json(report_path)
        except (OSError, ValueError):
            continue
        if (
            bool(report.get("audit_ok"))
            and str(report.get("stage")) == str(stage)
            and str(report.get("arm")) == str(arm)
        ):
            return report_path, report
    return None


def _fresh(*paths):
    for path in paths:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"corrected-v2 output must be new: {path}")
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _template(config, method, target, *, persistent_model=None):
    source_root = (
        V3_REFERENCE_ROOT
        if config.protocol in {
            "core-framework-v4",
            "core-framework-v4-delta-scale",
            "core-framework-v5",
            "core-framework-v6",
            "core-framework-v7",
            "core-framework-v8",
            "core-framework-v9-action-selection",
            "core-framework-v10-safety",
            "core-framework-v11-comm-occluder",
        }
        else HISTORICAL_ROOT
    )
    old = source_root / "seed_42" / (COMMUNICATION_BRANCH if method == COMMUNICATION_BRANCH else BASELINE_BRANCH) / "scale_3"
    manifest = old / "manifest.json"
    if manifest.is_file():
        # Historical runs remain the preferred source of exact template
        # values when their manifests are available.  The manifests are
        # experiment artifacts, however, and may be cleaned independently of
        # the executable code.  Do not make future runs depend on a deleted
        # result directory.
        formal = _reference_formal_config(
            _read_json(manifest), target, force=False, progress=config.progress
        )
    else:
        # Keep the protocol constructors runnable after an experiment-output
        # cleanup.  These defaults mirror the explicit three-UAV preliminary
        # contract; the protocol-specific ``replace`` calls below still apply
        # the versioned settings (PPO mode, feedback, horizon, and so on).
        fallback_cost_targets = (
            {f"uav_{index:02d}": 0.0 for index in range(3)}
            if config.protocol in {"stabilized-v4", "stabilized-v4-corrected"}
            and method == COMMUNICATION_BRANCH
            else None
        )
        formal = FormalExperimentConfig(
            output_dir=target,
            uav_count=3,
            seed=42,
            gnss_mode="denied",
            max_steps=config.max_steps,
            device="cpu",
            force=False,
            progress=config.progress,
            measurement_communication_cost_target_by_uav=fallback_cost_targets,
        )
    warmup, train, validation, test = config.counts
    stable = config.protocol in {
        "stabilized-v3",
        "stabilized-v4",
        "stabilized-v4-corrected",
        "core-framework-v1",
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
        }
    v2 = config.protocol == "core-framework-v2"
    v3 = config.protocol == "core-framework-v3"
    v4 = config.protocol in {"core-framework-v4", "core-framework-v4-delta-scale"}
    v5 = config.protocol == "core-framework-v5"
    v6 = config.protocol == "core-framework-v6"
    v7 = config.protocol == "core-framework-v7"
    v8 = config.protocol == "core-framework-v8"
    v9 = config.protocol == "core-framework-v9-action-selection"
    v10 = config.protocol == "core-framework-v10-safety"
    v11 = config.protocol == "core-framework-v11-comm-occluder"
    persistent = config.protocol in {"stabilized-v4", "stabilized-v4-corrected"} and method == COMMUNICATION_BRANCH
    legacy_proxy = method == COMMUNICATION_BRANCH and not persistent
    result = replace(formal, output_dir=target, seed=42, max_steps=config.max_steps,
                   warmup_episodes=warmup, train_episodes=train, validation_episodes=validation, test_episodes=test,
                   visual_odometry_quality_calibration_enabled=False, visual_odometry_quality_model=None,
                   random_stream_mode="indexed-v1", rollout_episodes_per_update=(
                       8 if v7 else 4 if (stable or config.experiment_arm == "rollout4") else 1
                   ),
                   actor_learning_rate_scale=(
                       0.5 if (v7 or (v3 and config.experiment_arm == "actor-lr-half")) else 1.0
                   ),
                   critic_loss_type="huber" if stable else formal.critic_loss_type,
                   value_clip_enabled=True if stable else formal.value_clip_enabled,
                   value_clip_epsilon=0.2 if stable else formal.value_clip_epsilon,
                   target_kl=0.05 if stable else formal.target_kl,
                   ppo_action_logprob_mode=(
                       "sampled" if (v3 or v4 or v5 or v6 or v7 or v8 or v9 or v10 or v11) else formal.ppo_action_logprob_mode
                   ),
                   bootstrap_time_limits=(
                       False if (v7 or v8 or v9) else True if stable else formal.bootstrap_time_limits
                   ),
                   episode_objective=(
                       "finite_horizon" if (v7 or v8 or v9) else formal.episode_objective
                   ),
                   include_remaining_time=(
                       True if (v7 or v8 or v9) else formal.include_remaining_time
                   ),
                   max_grad_norm=1.0 if (v6 or v7) else formal.max_grad_norm,
                   collision_penalty=2.0 if (v6 or v7) else formal.collision_penalty,
                   uncertainty_normalization=(
                       "warmup_q05_q95" if (v6 or v7) else formal.uncertainty_normalization
                   ),
                   learning_reward_clip=1.0 if stable else formal.learning_reward_clip,
                   learning_collision_penalty_cap=(
                       0.5 if v5 and config.experiment_arm == "bounded-collision" else
                       10.0 if v8 else
                       1.0 if v7 else
                       2.0 if v6 else
                       1.0 if stable else formal.learning_collision_penalty_cap
                   ),
                   learning_potential_coef=(
                       0.0 if (v5 or v6 or v7 or v8) else 0.25 if stable else formal.learning_potential_coef
                   ),
                   learning_gamma=0.99 if stable else formal.learning_gamma,
                   potential_scale_source="warmup" if (v2 or v3 or v4 or v5) else formal.potential_scale_source,
                   potential_statistics=None,
                   metric_scale_source=(
                       "training_warmup_uncertainty_q05_q95"
                       if (v6 or v7) else formal.metric_scale_source
                   ),
                   branch_invariant_initialization=True,
                   measurement_update_eligibility_enabled=bool(
                       config.measurement_update_eligibility_enabled
                   ),
                   encoder_type="mlp_slots_v2" if method == "mappo_mlp" else "gat",
                   measurement_head_mode=(
                       "action_conditioned_v2" if (v2 or v3 or v4 or v5 or v6 or v7 or v8 or v9 or v10 or v11) else formal.measurement_head_mode
                   ),
                   gradient_clip_scope=(
                       "actor_critic" if (v2 or v3 or v4 or v5 or v6 or v7 or v8 or v9 or v10 or v11) else formal.gradient_clip_scope
                   ),
                   learning_reward_mode=(
                       ("report_direct" if config.experiment_arm == "reference" else "report_plus_feedback")
                       if v8 else "bounded_potential" if v7 else "report_direct" if v6 else (
                           (
                           "report_direct" if config.experiment_arm == "report-clean" else
                           "legacy" if config.experiment_arm in {
                               "report-clean", "loc-progress-v2", "actual-load", "loc-progress-actual-load"
                           } else "bounded_potential"
                           ) if v5 else
                       "progress_potential_v2" if (v2 or v3 or v4) else (
                           "bounded_potential" if stable else formal.learning_reward_mode
                       ))
                   ),
                   # v4's proposed branch uses the same full-packet cost as
                   # the execution-time communication constraint.  The old
                   # protocols retain their historical teacher semantics.
                   measurement_proxy_full_packet_cost=(
                       True if persistent else config.experiment_arm != "reference"
                   ),
                   measurement_proxy_features_enabled=legacy_proxy,
                   measurement_proxy_guidance_enabled=legacy_proxy,
                   measurement_proxy_model=None if persistent else formal.measurement_proxy_model,
                   measurement_proxy_target_relative_cost=legacy_proxy,
                   measurement_persistent_proxy_features_enabled=persistent,
                   measurement_persistent_proxy_guidance_enabled=persistent,
                   measurement_persistent_proxy_horizon=int(config.persistent_horizon),
                   measurement_persistent_proxy_model=(
                       persistent_model if persistent else None
                   ),
                   # The target-relative flag is enabled only after the
                   # baseline-train target has been fitted in run_stage;
                   # constructing a FormalExperimentConfig must not depend on
                   # a target that does not yet exist.
                   measurement_persistent_proxy_target_relative_cost=False,
                   measurement_communication_constraint_enabled=method == COMMUNICATION_BRANCH,
                   auxiliary_gradient_diagnostics=True, development_monitor_enabled=True,
                   sealed_test_generation=True, metric_scales=None, frozen_normalization_path=None,
                   learning_feedback_mode=(
                        {
                            "loc-progress": "loc_progress_v1",
                            "actual-load": "actual_load_v1",
                            "loc-progress-actual-load": "loc_progress_actual_load_v1",
                            "loc-progress-v2": "loc_progress_v2",
                        }.get(config.experiment_arm, "v3")
                       if v4 else
                       {
                           "loc-progress-v2": "loc_progress_v2",
                           "actual-load": "actual_load_v1",
                           "loc-progress-actual-load": "loc_progress_actual_load_v1",
                       }.get(config.experiment_arm, "v3")
                       if v5 else
                       {
                           "loc-progress-v2": "loc_progress_v2",
                           "action-linked-feedback": "accepted_update_progress_v1",
                       }.get(config.experiment_arm, "v3")
                       if v8 else "v3"
                   ),
                   loc_progress_coef=(
                        0.10
                        if (v4 or v5 or v8)
                        and config.experiment_arm
                        in {"loc-progress", "loc-progress-actual-load", "loc-progress-v2"}
                        else 0.0
                    ),
                   actual_measurement_cost_enabled=bool(
                       (v4 or v5) and config.experiment_arm in {"actual-load", "loc-progress-actual-load"}
                   ),
                   actual_measurement_cost_target=(
                       0.0
                       if (v4 or v5) and config.experiment_arm in {"actual-load", "loc-progress-actual-load"}
                       else None
                   ),
                   actual_measurement_cost_coef=(0.10 if (v4 or v5) and config.experiment_arm in {"actual-load", "loc-progress-actual-load"} else 0.0),
                   accepted_update_feedback_coef=(
                       float(config.accepted_update_feedback_coef)
                       if v8 and config.experiment_arm == "action-linked-feedback"
                       else 0.0
                   ),
                   accepted_update_feedback_delay_steps=(
                       int(config.accepted_update_feedback_delay_steps)
                       if v8 and config.experiment_arm == "action-linked-feedback"
                       else 1
                   ),
                   command_line=f"run-mainline-baseline-comparison --protocol {config.protocol} --stage {config.stage} --experiment-arm {config.experiment_arm}")

    if v9:
        action_head = (
            "action_conditioned_v2"
            if config.experiment_arm == "action-selection-control"
            else "action_conditioned_v3"
        )
        feedback_mode = (
            "accepted_update_progress_v2"
            if config.experiment_arm == "action-linked-feedback-v2"
            else "v3"
        )
        result = replace(
            result,
            visual_odometry_quality_calibration_enabled=False,
            ppo_action_logprob_mode="sampled",
            rollout_episodes_per_update=4,
            critic_loss_type="huber",
            value_clip_enabled=True,
            target_kl=0.05,
            bootstrap_time_limits=False,
            episode_objective="finite_horizon",
            include_remaining_time=True,
            learning_reward_mode=(
                "report_plus_feedback"
                if config.experiment_arm == "action-linked-feedback-v2"
                else "report_direct"
            ),
            learning_feedback_mode=feedback_mode,
            learning_potential_coef=0.0,
            loc_progress_coef=0.0,
            actual_measurement_cost_enabled=False,
            actual_measurement_cost_target=None,
            actual_measurement_cost_coef=0.0,
            accepted_update_feedback_coef=(
                float(config.accepted_update_feedback_coef)
                if config.experiment_arm == "action-linked-feedback-v2"
                else 0.0
            ),
            accepted_update_feedback_delay_steps=(
                int(config.accepted_update_feedback_delay_steps)
                if config.experiment_arm == "action-linked-feedback-v2"
                else 1
            ),
            accepted_update_feedback_location=0.0,
            accepted_update_feedback_scale=1.0,
            accepted_update_feedback_none_baseline=0.0,
            collision_penalty=10.0,
            max_grad_norm=0.5,
            uncertainty_normalization="p95",
            actor_learning_rate_scale=1.0,
            measurement_head_mode=action_head,
            gradient_clip_scope="actor_critic",
            encoder_type="gat",
        )

    if v10 or v11:
        result = replace(
            result,
            learning_reward_mode="report_direct",
            learning_feedback_mode="v3",
            learning_potential_coef=0.0,
            actual_measurement_cost_enabled=False,
            actual_measurement_cost_target=None,
            actual_measurement_cost_coef=0.0,
            dynamic_obstacle_swept_projection_enabled=True,
            safety_execution_mode="joint_swept_v1",
            dynamic_obstacle_motion_features_enabled=bool(v11),
            communication_occluder_physical_collision_enabled=bool(not v11),
        )


    if config.finite_horizon_reward is not None:
        if config.finite_horizon_reward not in {"existing", "report_direct"}:
            raise ValueError("unknown finite horizon reward")
        result = replace(result, episode_objective="finite_horizon",
                         include_remaining_time=True, bootstrap_time_limits=False)
        result = replace(result, learning_reward_mode="progress_potential_v2",
                         learning_feedback_mode="loc_progress_v2", loc_progress_coef=0.1,
                         learning_potential_coef=0.25)
        if config.finite_horizon_reward == "report_direct":
            result = replace(result, learning_reward_mode="report_direct",
                             learning_feedback_mode="v3", learning_potential_coef=0.0,
                             loc_progress_coef=0.0, actual_measurement_cost_enabled=False,
                             actual_measurement_cost_coef=0.0)
    return result


def _fit_actual_measurement_target(
    scale: Path,
    *,
    source_protocol: str,
) -> dict[str, Any]:
    """Read one frozen baseline *train* split for a fixed load target.

    The target is a provenance input, not a validation-derived tuning
    parameter.  We deliberately read only baseline train episodes and
    summarize each episode before taking an equal-weighted mean, matching the
    experiment's episode-level aggregation rule.
    """

    manifest_path = scale / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"v3 baseline manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    train_episodes = [
        item for item in manifest.get("episodes", [])
        if isinstance(item, dict) and str(item.get("split")) == "train"
    ]
    if not train_episodes:
        raise ValueError("v3 baseline has no train episodes for actual-load target")
    episode_rates: list[float] = []
    selected_total = 0
    window_total = 0
    for episode in train_episodes:
        episode_root = scale / "train" / str(episode["episode_id"])
        actions_path = episode_root / "actions.csv"
        if not actions_path.is_file():
            raise FileNotFoundError(actions_path)
        actions = _read_csv(actions_path)
        if not actions:
            raise ValueError(f"empty v3 baseline actions: {actions_path}")
        selected = sum(
            str(row.get("measurement_uav_id", "none")) not in {"none", ""}
            for row in actions
        )
        uav_count = 3
        windows = len(actions) / uav_count
        if windows <= 0.0:
            raise ValueError(f"invalid action-window count: {actions_path}")
        # ``environment._rewards`` defines ``b_t`` as bytes per UAV per
        # decision window, i.e. ``B_t / (512 * N_u)``.  The action trace has
        # one row per UAV and window, so ``selected / windows`` is the total
        # number of measurements per window, not the same quantity.  Divide
        # by the UAV count here to keep the train-only target in the exact
        # units used by the feedback term (and bounded by one).
        episode_rates.append(float(selected / (windows * uav_count)))
        selected_total += int(selected)
        window_total += int(round(windows))
    target = float(np.mean(episode_rates))
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("fitted v4 actual measurement target is not finite/non-negative")
    return {
        "target": target,
        "target_definition": "mean_episode_measurement_rate_per_uav_window",
        "packet_size_bytes": 512,
        "uav_count": 3,
        "source_split": "train",
        "source_protocol": source_protocol,
        "source_scale": str(scale),
        "source_episode_count": len(episode_rates),
        "selected_measurements_total": int(selected_total),
        "windows_total": int(window_total),
    }


def _fit_v4_actual_measurement_target() -> dict[str, Any]:
    """Read the historical v3 baseline *train* split for v4 compatibility."""

    return _fit_actual_measurement_target(
        V3_REFERENCE_ROOT / "seed_42" / BASELINE_BRANCH / "scale_3",
        source_protocol="core-framework-v3",
    )


def _schema(scale, destination):
    subprocess.run([sys.executable, "-m", "src.experiments.corrected_schema_validation", str(scale), str(destination)], check=True)
    return _read_json(destination)


def strict_episode_metrics(scale, episode):
    """Read only explicit validation files; reconcile events/counters and loads."""
    if episode["split"] != "validation":
        raise ValueError("performance collector only accepts validation")
    path = scale / "validation" / episode["episode_id"]
    for name in ("actions.csv", "ground_truth.csv", "estimates.csv", "rewards.csv", "links.csv", "measurement_overhead.csv", "scenario.json"):
        if not (path/name).is_file():
            raise FileNotFoundError(path/name)
    actions, estimates = _read_csv(path/"actions.csv"), _read_csv(path/"estimates.csv")
    events_path = scale / "diagnostics/validation_cooperative_updates.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    events = [json.loads(s) for s in events_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    events = [e for e in events if e["episode_id"] == episode["episode_id"]]
    for event in events:
        for key in ("true_position_error_before_m", "true_position_error_after_m", "kalman_gain_norm", "position_correction_norm_m"):
            if not np.isfinite(float(event[key])):
                raise ValueError(f"nonfinite cooperative diagnostic {key}")
    identities = [e["event_id"] for e in events]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate cooperative event IDs")
    for uav in ("uav_00", "uav_01", "uav_02"):
        last = [r for r in estimates if r["uav_id"] == uav][-1]
        count = sum(bool(e["accepted"]) and e["target_uav_id"] == uav for e in events)
        if count != int(last["valid_neighbor_count"]):
            raise ValueError(f"episode accepted counter mismatch: {path}/{uav}")
    selected = sum(r["measurement_uav_id"] != "none" for r in actions)
    packets = sum(float(r["load_packets_applied"]) for r in _read_csv(path/"measurement_overhead.csv"))
    if selected != packets:
        raise ValueError(f"measurement/payload mismatch: {path}")
    scenario = _read_json(path/"scenario.json")
    limits = scenario.get("measurement_budget_limits", {})
    if scenario.get("measurement_budget_enabled", False):
        for uav, limit in limits.items():
            if sum(r["uav_id"] == uav and r["measurement_uav_id"] != "none" for r in actions) > int(limit):
                raise ValueError(f"budget exceeded: {path}/{uav}")
    row = _episode_metrics(path, episode)
    for field in ("position_rmse_norm_m", "position_p95_norm_m", "mean_reward", "j_com", "j_loc", "accepted_updates_per_window"):
        if field not in row or not np.isfinite(float(row[field])):
            raise ValueError(f"nonfinite/missing {field}: {path}")
    if int(row["position_error_sample_count"]) != len(estimates):
        raise ValueError(f"incomplete position-error pairing: {path}")
    if not row["accepted_update_count"]:
        for key in ("beneficial_fraction", "neutral_fraction", "harmful_fraction"):
            row[key] = "NA"
    row.update(measurement_count=selected, actual_measurement_bytes=512*selected,
               windows=len(actions)//3, source_scale=str(scale),
               scenario_generation_hash=hashlib.sha256(json.dumps(scenario["scenario_generation"], sort_keys=True).encode()).hexdigest())
    return row


def summarize(paths, expected_count, methods=None):
    rows = []
    methods = tuple(METHODS if methods is None else methods)
    from .mainline_figures_v2 import SourceGuard
    with SourceGuard([Path(p) for p in paths.values()], []) as guard:
        for method in methods:
            scale = Path(paths[method])
            episodes = _load_scenario_episodes(scale)
            if len(episodes) != expected_count:
                raise ValueError(f"incomplete validation for {method}")
            for episode in episodes:
                row = strict_episode_metrics(scale, episode)
                if method == "local_only_fixed" and (row["measurement_count"] or row["accepted_update_count"]):
                    raise ValueError("local-only method used cooperative information")
                rows.append({"method": method, **row})
        for episode in {row["episode_id"] for row in rows}:
            matched = [r for r in rows if r["episode_id"] == episode]
            if len({r["scenario_generation_hash"] for r in matched}) != 1:
                raise ValueError(f"scenario mismatch: {episode}")
        hashes = guard.verify()
    if guard.test_attempts or not all(r["unchanged"] for r in hashes):
        raise ValueError("source guard failed")
    return rows, {"test_read_attempts": guard.test_attempts, "read_files": hashes}


def descriptive_comparison(rows, methods=None, proposed_method=None):
    methods = tuple(METHODS if methods is None else methods)
    fields = ("position_rmse_norm_m", "position_p95_norm_m", "actual_measurement_bytes", "j_com",
              "mean_reward", "harmful_fraction", "update_coverage_any", "accepted_updates_per_window",
              "beneficial_per_window", "harmful_per_window", "windows", "packet_loss", "delay_s", "aoi_s")
    means=[]
    for method in methods:
        group=[r for r in rows if r["method"]==method]
        result={"method":method,"episode_count":len(group)}
        for key in fields:
            values=[float(r[key]) for r in group if r[key] != "NA"]
            result[key]=float(np.mean(values)) if values else "NA"
            result[key+"_n"]=len(values)
        means.append(result)
    index={r["method"]:r for r in means}
    # The historical protocols compare a candidate branch against the GAT
    # baseline.  The core-framework protocol has no candidate branch: the
    # integrated Local-EKF-GAT-MAPPO method *is* the proposed reference.  Its
    # eligibility is therefore deliberately not applicable rather than being
    # manufactured from a self-comparison.
    proposed = proposed_method or COMMUNICATION_BRANCH
    if proposed == BASELINE_BRANCH:
        qualified = None
    else:
        proposed_row=index[proposed]; baseline=index[BASELINE_BRANCH]
        qualified = (proposed_row["position_rmse_norm_m"] <= baseline["position_rmse_norm_m"]
                     and proposed_row["position_p95_norm_m"] <= baseline["position_p95_norm_m"]
                     and proposed_row["actual_measurement_bytes"] <= baseline["actual_measurement_bytes"]
                     and proposed_row["j_com"]-baseline["j_com"] <= .005
                     and proposed_row["harmful_fraction"] != "NA" and baseline["harmful_fraction"] != "NA"
                     and proposed_row["harmful_fraction"]-baseline["harmful_fraction"] <= .02
                     and proposed_row["update_coverage_any"] >= baseline["update_coverage_any"]-.05)
    pairs=[]
    for reference in methods:
        if reference==proposed: continue
        for episode in sorted({r["episode_id"] for r in rows}):
            a=next(r for r in rows if r["method"]==proposed and r["episode_id"]==episode)
            b=next(r for r in rows if r["method"]==reference and r["episode_id"]==episode)
            pairs.append({"reference":reference,"episode_id":episode, **{
                key: float(a[key])-float(b[key]) if a[key]!="NA" and b[key]!="NA" else "NA" for key in fields}})
    # ``None`` is intentional for the core framework protocol: the proposed
    # method is the GAT-MAPPO reference itself, so there is no historical
    # candidate-selection decision to manufacture from a self-comparison.
    return means, pairs, qualified


def common_prefix_sensitivity(paths, methods=None):
    from .metrics import position_error_metrics
    from .mainline_figures_v2 import SourceGuard
    methods = tuple(METHODS if methods is None else methods)
    rows=[]
    with SourceGuard([Path(p) for p in paths.values()], []):
        for episode in _load_scenario_episodes(Path(paths[BASELINE_BRANCH])):
            eid=episode["episode_id"]
            for other in methods:
                if other==BASELINE_BRANCH: continue
                sources=[]
                for method in (BASELINE_BRANCH,other):
                    folder=Path(paths[method])/"validation"/eid
                    sources.append((_read_csv(folder/"ground_truth.csv"),_read_csv(folder/"estimates.csv")))
                end=min(max(float(r["timestamp"]) for r in estimates) for truth, estimates in sources)
                values=[position_error_metrics([r for r in truth if float(r["timestamp"])<=end],
                                               [r for r in estimates if float(r["timestamp"])<=end])["rmse_norm_m"] for truth,estimates in sources]
                rows.append({"method":other,"episode_id":eid,"common_end_s":end,"baseline_rmse":values[0],"method_rmse":values[1],"paired_difference":values[1]-values[0]})
    return rows


def audit_training(paths, counts, rollout_size, learners=None):
    import math
    import torch
    learners = tuple(LEARNERS if learners is None else learners)
    details={}
    shared=[]
    for method in learners:
        scale=Path(paths[method]); raw=_read_json(scale/"manifest.json")["config"]
        rows=[json.loads(s) for s in (scale/"diagnostics/training_metrics.jsonl").read_text().splitlines() if s.strip()]
        if len(rows)!=math.ceil(counts[1]/rollout_size):
            raise ValueError(f"update count mismatch {method}")
        episodes=[eid for r in rows for eid in r["episode_ids"]]
        if episodes!=[f"episode_{i:06d}" for i in range(1,counts[1]+1)]:
            raise ValueError(f"rollout episode binding mismatch {method}")
        for row in rows:
            required_training_keys = (
                "loss","actor_loss","critic_loss","entropy","clip_fraction","gradient_norm","approximate_kl",
                "measurement_head_ppo_gradient_norm","measurement_head_teacher_gradient_norm","measurement_head_communication_gradient_norm",
            )
            if raw.get("gradient_clip_scope") == "actor_critic":
                required_training_keys += (
                    "actor_gradient_norm",
                    "critic_gradient_norm",
                    "actor_clip_scale",
                    "critic_clip_scale",
                    "gradient_clip_scope_actor_critic",
                )
            for key in required_training_keys:
                if not np.isfinite(float(row[key])):
                    raise ValueError(f"nonfinite/missing training diagnostic {method}/{key}")
            for key,value in row.items():
                if key.startswith("measurement_communication_dual_uav_") and not 0 <= float(value) <= 5:
                    raise ValueError("dual outside [0,5]")
        checkpoint=torch.load(scale/"checkpoints/final.pt",map_location="cpu",weights_only=False)
        final=_read_json(scale/"checks/integration_summary.json")
        if checkpoint["measurement_communication_dual_state"]!=final["measurement_communication_dual_state"]:
            raise ValueError("validation/test changed communication multipliers")
        if raw["visual_odometry_quality_calibration_enabled"] or raw["random_stream_mode"]!="indexed-v1":
            raise ValueError("incorrect effective VO/RNG configuration")
        if raw["rollout_episodes_per_update"]!=rollout_size:
            raise ValueError("incorrect rollout configuration")
        stats=_read_json(scale/"normalization_stats.json")
        shared.append({key:stats[key] for key in ("normalizers","central_normalizer","metric_scales")})
        needs_potential_statistics = (
            raw.get("learning_reward_mode") == "progress_potential_v2"
            or raw.get("learning_feedback_mode") == "loc_progress_v2"
        )
        if needs_potential_statistics:
            if not isinstance(stats.get("potential_statistics"), dict):
                raise ValueError(f"missing frozen potential statistics for {method}")
            shared[-1]["potential_statistics"] = stats["potential_statistics"]
            if stats.get("potential_scale_source") not in {"warmup", "frozen"}:
                raise ValueError(f"invalid potential scale provenance for {method}")
        details[method]={"updates":len(rows),"episodes":len(episodes),"optimizer_steps":sum(r["optimizer_steps"] for r in rows),
                         "communication_multiplier_updates":sum(r["communication_multiplier_updates"] for r in rows),
                         "active_parameter_count":sum(p.numel() for name,p in checkpoint["policy_state_dict"].items() if not (method=="mappo_mlp" and name.startswith("gat."))),
                         "effective_config":raw}
    if not all(s==shared[0] for s in shared):
        raise ValueError("shared observation/reward normalization differs across branches")
    keys=("max_steps","vo_position_std_m","process_position_std_m","process_velocity_std_mps","nlos_variance_multiplier",
          "learning_rate","policy_epochs","minibatch_size","gamma","gae_lambda","entropy_coef","clip_epsilon",
          "scenario_profile_path","measurement_overhead_enabled","branch_invariant_initialization")
    if any(len({str(v["effective_config"][key]) for v in details.values()})!=1 for key in keys):
        raise ValueError("effective common experimental configuration differs")
    return details


def run_stage(config):
    if config.max_steps != 50 or config.counts != ((2,5,5,5) if config.stage=="smoke" else (20,100,20,20)):
        raise ValueError("the stabilized comparison fixes the preregistered step and episode counts")
    fixed={"quality_logit_penalty":1.,"proxy_teacher_coef":.25,"proxy_temperature":.25,
           "communication_cost_weight":1.,"communication_cost_tolerance":.05,"communication_dual_initial":0.,
           "communication_dual_step_size":.25,"communication_dual_max":5.,"fixed_packet_cost_fraction":.5}
    if any(float(getattr(config,k))!=v for k,v in fixed.items()) or not config.measurement_overhead_enabled or not config.branch_invariant_initialization:
        raise ValueError("stabilized comparison fixes coefficients, overhead and shared initialization")
    if config.protocol in {"stabilized-v4", "stabilized-v4-corrected"} and config.experiment_arm != "reference":
        raise ValueError("stabilized-v4-corrected uses the reference arm; its proposed branch is persistent-proxy aligned")
    methods = tuple(_methods_for(config))
    learners = tuple(_learners_for(config))
    core_scope = config.protocol in {
        "core-framework-v1",
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
    }
    v4_scope = config.protocol in {"core-framework-v4", "core-framework-v4-delta-scale"}
    v5_scope = config.protocol == "core-framework-v5"
    v6_scope = config.protocol == "core-framework-v6"
    v7_scope = config.protocol == "core-framework-v7"
    v8_scope = config.protocol == "core-framework-v8"
    v9_scope = config.protocol == "core-framework-v9-action-selection"
    v10_scope = config.protocol == "core-framework-v10-safety"
    v11_scope = config.protocol == "core-framework-v11-comm-occluder"
    root, logs, plots = config.resolved_output_dir, config.resolved_logs_dir, config.resolved_plots_dir
    # Derive the protocol roots from the resolved artifact directories.  This
    # matters for an explicitly versioned rerun: dependencies and the
    # same-configuration Smoke must be read from that rerun's own root rather
    # than silently falling back to the already-completed v4 archive.
    log_root = logs.parent.parent
    data_root = root.parent.parent
    signature = configuration_signature(config)
    if v4_scope or v5_scope or v8_scope:
        dependency_arms = (
            {
                "actual-load": ("loc-progress",),
                "loc-progress-actual-load": ("loc-progress", "actual-load"),
            }
            if v4_scope
            else {
                "bounded-collision": ("report-clean",),
                "loc-progress-v2": ("bounded-collision",),
                "actual-load": ("loc-progress-v2",),
                "loc-progress-actual-load": ("loc-progress-v2", "actual-load"),
            }
            if v5_scope
            else {
                "loc-progress-v2": ("reference",),
            }
        ).get(config.experiment_arm, ())
        for dependency in dependency_arms:
            dependency_match = find_stage_report(log_root, config.stage, dependency)
            if dependency_match is None:
                raise RuntimeError(
                    f"{config.protocol} {config.experiment_arm} requires a completed "
                    f"{config.stage}/{dependency} report under {log_root / config.stage}"
                )
            dependency_report, dependency_payload = dependency_match
            if not bool(dependency_payload.get("audit_ok")):
                raise RuntimeError(
                    f"{config.protocol} dependency failed: {dependency_report}"
                )
    replay = _read_json(_replay_log_root(config) / "replay-audit/reference/audit_report.json")
    if not replay.get("audit_ok"):
        raise RuntimeError("frozen historical replay must pass before training")
    if config.stage == "development":
        smoke_match = find_compatible_smoke_report(config)
        if smoke_match is None:
            raise RuntimeError("same-configuration smoke is missing or failed")
    _fresh(root, logs, plots)
    run_config = json.loads(json.dumps(asdict(config), default=str))
    if config.protocol in {"core-framework-v3", "core-framework-v4", "core-framework-v4-delta-scale"}:
        # ``MainlineBaselineComparisonConfig`` keeps permissive historical
        # defaults for direct Python callers.  Persist the effective v3
        # protocol settings as well, so the top-level provenance file cannot
        # be mistaken for the per-branch training manifest.
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": True,
                "learning_reward_mode": "progress_potential_v2",
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v9_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": False,
                "episode_objective": "finite_horizon",
                "include_remaining_time": True,
                "learning_reward_mode": (
                    "report_plus_feedback"
                    if config.experiment_arm == "action-linked-feedback-v2"
                    else "report_direct"
                ),
                "learning_feedback_mode": (
                    "accepted_update_progress_v2"
                    if config.experiment_arm == "action-linked-feedback-v2"
                    else "v3"
                ),
                "learning_potential_coef": 0.0,
                "accepted_update_feedback_coef": (
                    float(config.accepted_update_feedback_coef)
                    if config.experiment_arm == "action-linked-feedback-v2"
                    else 0.0
                ),
                "accepted_update_feedback_delay_steps": (
                    int(config.accepted_update_feedback_delay_steps)
                    if config.experiment_arm == "action-linked-feedback-v2"
                    else 1
                ),
                "measurement_head_mode": (
                    "action_conditioned_v2"
                    if config.experiment_arm == "action-selection-control"
                    else "action_conditioned_v3"
                ),
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v5_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": True,
                "learning_reward_mode": (
                    "bounded_potential"
                    if config.experiment_arm == "bounded-collision"
                    else "report_direct" if config.experiment_arm == "report-clean"
                    else "legacy"
                ),
                "learning_collision_penalty_cap": (
                    0.5 if config.experiment_arm == "bounded-collision" else 1.0
                ),
                "learning_potential_coef": 0.0,
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v6_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": True,
                "learning_reward_mode": "report_direct",
                "learning_feedback_mode": "v3",
                "learning_potential_coef": 0.0,
                "collision_penalty": 2.0,
                "max_grad_norm": 1.0,
                "uncertainty_normalization": "warmup_q05_q95",
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v7_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 8,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": False,
                "episode_objective": "finite_horizon",
                "include_remaining_time": True,
                "learning_reward_mode": "bounded_potential",
                "learning_feedback_mode": "v3",
                "learning_potential_coef": 0.0,
                "learning_collision_penalty_cap": 1.0,
                "collision_penalty": 2.0,
                "max_grad_norm": 1.0,
                "uncertainty_normalization": "warmup_q05_q95",
                "actor_learning_rate_scale": 0.5,
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v8_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": False,
                "episode_objective": "finite_horizon",
                "include_remaining_time": True,
                "learning_reward_mode": (
                    "report_direct"
                    if config.experiment_arm == "reference"
                    else "report_plus_feedback"
                ),
                "learning_feedback_mode": (
                    "v3"
                    if config.experiment_arm == "reference"
                    else "accepted_update_progress_v1"
                    if config.experiment_arm == "action-linked-feedback"
                    else "loc_progress_v2"
                ),
                "learning_potential_coef": 0.0,
                "learning_collision_penalty_cap": 10.0,
                "accepted_update_feedback_coef": (
                    float(config.accepted_update_feedback_coef)
                    if config.experiment_arm == "action-linked-feedback"
                    else 0.0
                ),
                "accepted_update_feedback_delay_steps": (
                    int(config.accepted_update_feedback_delay_steps)
                    if config.experiment_arm == "action-linked-feedback"
                    else 1
                ),
                "collision_penalty": 10.0,
                "max_grad_norm": 0.5,
                "uncertainty_normalization": "p95",
                "actor_learning_rate_scale": 1.0,
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    if v10_scope or v11_scope:
        run_config.update(
            {
                "visual_odometry_quality_calibration_enabled": False,
                "ppo_action_logprob_mode": "sampled",
                "rollout_episodes_per_update": 4,
                "critic_loss_type": "huber",
                "value_clip_enabled": True,
                "target_kl": 0.05,
                "bootstrap_time_limits": True,
                "learning_reward_mode": "report_direct",
                "learning_feedback_mode": "v3",
                "learning_potential_coef": 0.0,
                "measurement_head_mode": "action_conditioned_v2",
                "gradient_clip_scope": "actor_critic",
                "encoder_type": "gat",
                "safety_execution_mode": "joint_swept_v1",
                "dynamic_obstacle_swept_projection_enabled": True,
                "dynamic_obstacle_motion_features_enabled": bool(v11_scope),
                "communication_occluder_physical_collision_enabled": bool(not v11_scope),
                "active_methods": list(_methods_for(config)),
                "active_learned_methods": list(_learners_for(config)),
            }
        )
    feedback_target = None
    if v4_scope:
        feedback_target = _fit_v4_actual_measurement_target()
    elif v5_scope and config.experiment_arm in {
        "actual-load",
        "loc-progress-actual-load",
    }:
        target_scale = (
            data_root
            / config.stage
            / "report-clean"
            / "seed_42"
            / BASELINE_BRANCH
            / "scale_3"
        )
        feedback_target = _fit_actual_measurement_target(
            target_scale,
            source_protocol="core-framework-v5/report-clean",
        )
    feedback_scope = v4_scope or v5_scope or v8_scope
    if feedback_scope:
        feedback_mode = (
            {
                "loc-progress": "loc_progress_v1",
                "actual-load": "actual_load_v1",
                "loc-progress-actual-load": "loc_progress_actual_load_v1",
                "loc-progress-v2": "loc_progress_v2",
                "action-linked-feedback": "accepted_update_progress_v1",
            }
            if v4_scope
            else {
                "report-clean": "v3",
                "bounded-collision": "v3",
                "loc-progress-v2": "loc_progress_v2",
                "actual-load": "actual_load_v1",
                "loc-progress-actual-load": "loc_progress_actual_load_v1",
            }
            if v5_scope
            else {
                "reference": "v3",
                "loc-progress-v2": "loc_progress_v2",
                "action-linked-feedback": "accepted_update_progress_v1",
            }
        )[config.experiment_arm]
        actual_enabled = config.experiment_arm in {
            "actual-load",
            "loc-progress-actual-load",
        }
        run_config.update(
            {
                "learning_feedback_mode": feedback_mode,
                "loc_progress_coef": (
                    0.10
                    if config.experiment_arm in {
                        "loc-progress",
                        "loc-progress-actual-load",
                        "loc-progress-v2",
                    }
                    else 0.0
                ),
                "actual_measurement_cost_enabled": actual_enabled,
                "actual_measurement_cost_target": (
                    float(feedback_target["target"])
                    if actual_enabled and feedback_target is not None
                    else None
                ),
                "actual_measurement_cost_coef": (
                    0.10
                    if actual_enabled
                    else 0.0
                ),
                "accepted_update_feedback_coef": (
                    float(config.accepted_update_feedback_coef)
                    if config.experiment_arm == "action-linked-feedback"
                    else 0.0
                ),
                "accepted_update_feedback_delay_steps": (
                    int(config.accepted_update_feedback_delay_steps)
                    if config.experiment_arm == "action-linked-feedback"
                    else 1
                ),
                "actual_measurement_cost_target_source": (
                    feedback_target if actual_enabled else None
                ),
            }
        )
    _write_json(logs / "run_config.json", run_config)
    report = {"protocol": config.protocol, "stage": config.stage, "arm": config.experiment_arm,
              "configuration_signature": signature, "audit_ok": False, "status": "running",
              "cross_seed_stability": "not_evaluated", "candidate_selected": False,
              "paper_scope": {
                  "proposed_method_internal_id": BASELINE_BRANCH,
                  "proposed_method_label": PAPER_BRANCH_LABELS[BASELINE_BRANCH],
                  "active_methods": list(methods),
                  "active_learned_methods": list(learners),
                  "deprecated_methods_excluded": list(DEPRECATED_METHODS),
              }}
    _write_json(
        logs / "method_definitions.json",
        {
            "protocol": config.protocol,
            "paper_scope": report["paper_scope"],
            "methods": {
                method: {
                    "label": PAPER_BRANCH_LABELS.get(method, method),
                    "trained": method in learners,
                    "active": method in methods,
                }
                for method in methods
            },
        },
    )
    if config.finite_horizon_reward is not None:
        run_config.update(episode_objective="finite_horizon", include_remaining_time=True,
                          bootstrap_time_limits=False, finite_horizon_reward=config.finite_horizon_reward)
        _write_json(logs / "run_config.json", run_config)
    _write_json(logs/"audit_report.json", report)
    paths = {}
    persistent_model = None
    persistent_model_rows = 0
    rule_proxy = None
    try:
        reference_root = data_root / config.stage / "reference"
        if config.experiment_arm == "proxy-aligned":
            reference = _read_json(log_root / config.stage / "reference/audit_report.json")
            if not reference.get("audit_ok"):
                raise RuntimeError("reference arm must pass before proxy alignment")
            paths = dict(reference["method_paths"])
            baseline = Path(paths[BASELINE_BRANCH])
        else:
            baseline_output = root/"seed_42"/BASELINE_BRANCH
            formal = _template(config, BASELINE_BRANCH, baseline_output)
            if feedback_target is not None:
                formal = replace(
                    formal,
                    actual_measurement_cost_target=float(feedback_target["target"]),
                )
            if config.experiment_arm == "rollout4":
                reference = _read_json(log_root/config.stage/"reference/audit_report.json")
                shared = _read_json(Path(reference["method_paths"][BASELINE_BRANCH])/"normalization_stats.json")["metric_scales"]
            else:
                shared = fit_shared_training_metric_scales(formal, (3,))
            run_preliminary(replace(formal, metric_scales=shared))
            baseline = baseline_output/"scale_3"
            paths[BASELINE_BRANCH] = str(baseline)
        stats = _read_json(baseline/"normalization_stats.json")
        from .measurement_rate_controlled_mappo import _fit_measurement_budgets
        from .communication_constrained_proxy_mappo import _train_proxy_model, fit_measurement_communication_cost_targets
        from .mainline_figures_v2 import SourceGuard
        with SourceGuard([baseline], []) as fitting_guard:
            budgets = _fit_measurement_budgets(baseline, uav_count=3)
            proxy = _train_proxy_model(baseline, 42, 1.0)
            # The greedy rule baseline must use the same frozen, execution-time
            # proxy fitted from baseline/train.  Persist its provenance so that
            # a later replay cannot silently fall back to an all-none policy.
            rule_proxy = proxy
            if core_scope:
                targets, target_rows = {}, []
            else:
                targets, target_rows = fit_measurement_communication_cost_targets(
                    baseline,
                    seed=42,
                    proxy_model=proxy,
                    fixed_packet_cost_fraction=.5,
                )
            if config.protocol in {"stabilized-v4", "stabilized-v4-corrected"}:
                from .persistent_proxy_guided_quality_mappo import (
                    _fit_persistent_model,
                    PersistentProxyMAPPOConfig,
                )
                persistent_model, persistent_rows = _fit_persistent_model(
                    baseline,
                    seed=42,
                    config=PersistentProxyMAPPOConfig(
                        output_dir=root / "persistent_proxy_fit_metadata",
                        seeds=(42,),
                        uav_count=3,
                        max_steps=config.max_steps,
                        warmup_episodes=config.counts[0],
                        train_episodes=config.counts[1],
                        validation_episodes=config.counts[2],
                        test_episodes=config.counts[3],
                        scenario_profile_path=config.scenario_profile_path,
                        ekf_config_path=config.ekf_config_path,
                        measurement_overhead_enabled=True,
                        persistent_horizon=int(config.persistent_horizon),
                        progress=False,
                    ),
                    ekf_parameters=load_selected_ekf_config(config.ekf_config_path),
                )
                persistent_model_rows = len(persistent_rows)
        fitting_audit={"test_read_attempts":fitting_guard.test_attempts,"source_files":fitting_guard.verify()}
        quality = _read_json(baseline/"calibration/measurement_quality_model.json")
        if core_scope:
            _write_table(
                logs/"communication_targets.csv",
                [{
                    "enabled": False,
                    "reason": "communication-constrained branch is deprecated in the core framework protocol",
                }],
            )
        else:
            _write_table(logs/"communication_targets.csv", target_rows)
        _write_json(
            logs / "rule_measurement_proxy_model.json",
            rule_proxy,
        )
        if persistent_model is not None:
            _write_json(logs / "persistent_proxy_model.json", persistent_model)
        if config.experiment_arm != "proxy-aligned" and config.protocol not in {
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
            formal = _template(config, "mappo_mlp", root/"seed_42/mappo_mlp")
            run_preliminary(replace(formal, metric_scales=stats["metric_scales"], frozen_normalization_path=str(baseline/"normalization_stats.json")))
            paths["mappo_mlp"] = str(root/"seed_42/mappo_mlp/scale_3")
        if not core_scope:
            formal = _template(
                config,
                COMMUNICATION_BRANCH,
                root / "seed_42" / COMMUNICATION_BRANCH,
                persistent_model=persistent_model,
            )
            formal = replace(
                formal,
                metric_scales=stats["metric_scales"],
                frozen_normalization_path=str(baseline/"normalization_stats.json"),
                measurement_budget_by_uav=budgets,
                measurement_proxy_model=(proxy if formal.measurement_proxy_features_enabled else None),
                measurement_persistent_proxy_target_relative_cost=(
                    bool(formal.measurement_persistent_proxy_features_enabled)
                ),
                measurement_communication_cost_target_by_uav=targets,
                measurement_persistent_proxy_model=(
                    persistent_model
                    if formal.measurement_persistent_proxy_features_enabled
                    else None
                ),
            )
            run_preliminary(
                formal,
                measurement_quality_model=quality,
                measurement_proxy_model=(proxy if formal.measurement_proxy_features_enabled else None),
                measurement_persistent_proxy_model=(
                    persistent_model
                    if formal.measurement_persistent_proxy_features_enabled
                    else None
                ),
            )
            paths[COMMUNICATION_BRANCH] = str(root/"seed_42"/COMMUNICATION_BRANCH/"scale_3")
        from .mainline_baseline_comparison import _run_rule_method, _run_fixed_control_learned
        rule_source = Path(config.rule_reference_scale) if config.rule_reference_scale else baseline
        if config.rule_reference_scale:
            proxy = _train_proxy_model(rule_source, 42, 1.0)
        if config.experiment_arm != "proxy-aligned":
            for method in FIXED_CONTROL_METHODS:
                _run_rule_method(config, root, method, baseline_source=rule_source, rule_proxy_model=proxy)
                paths[method] = str(root/"seed_42/rules"/method)
        schema = {method: _schema(Path(path), logs/f"schema_{method}.json") for method, path in paths.items()}
        for method, result in schema.items():
            expected = sum(config.counts[1:]) if method in learners else config.counts[2]
            if result["episode_count"] != expected:
                raise ValueError(f"schema episode count mismatch: {method}")
        # Schema for rule branches lacks learned manifests; validate each episode
        # with a conventional scale manifest generated from their frozen formal config.
        rows, isolation = summarize(paths, config.counts[2], methods=methods)
        _write_table(logs/"validation_episode_metrics.csv", rows)
        means, pairs, eligible = descriptive_comparison(
            rows,
            methods=methods,
            proposed_method=BASELINE_BRANCH if core_scope else None,
        )
        _write_table(logs/"method_means.csv",means)
        _write_table(logs/"paired_episode_differences.csv",pairs)
        _write_table(
            logs/"common_prefix_rmse.csv",
            common_prefix_sensitivity(paths, methods=methods),
        )
        checkpoint_hashes = {
            m: _sha256(Path(paths[m])/"checkpoints/final.pt") for m in learners
        }
        b_paths = {m: paths[m] for m in FIXED_CONTROL_METHODS}
        for method in learners:
            _run_fixed_control_learned(config, root, method, baseline_source=baseline, branch_source=Path(paths[method]))
            b_paths[method] = str(root/"seed_42/fixed_control_diagnostic"/method)
        b_methods = tuple(FIXED_CONTROL_METHODS) + tuple(learners)
        b_rows, b_isolation = summarize(b_paths, config.counts[2], methods=b_methods)
        _write_table(logs/"fixed_control_B_episode_metrics.csv", b_rows)
        if any(_sha256(Path(paths[m])/"checkpoints/final.pt") != checkpoint_hashes[m] for m in learners):
            raise ValueError("B replay changed checkpoint")
        modules = [_read_json(Path(paths[m])/"diagnostics/initial_parameter_hashes.json")["modules"] for m in learners]
        shared_keys = ("actor_body", "bandwidth_head", "delta_mean_head", "delta_logstd", "critic")
        initialization_ok = all(len({item[k] for item in modules}) == 1 for k in shared_keys)
        initialization_ok &= len({item["gat"] for item in modules}) == 1
        if not initialization_ok:
            raise ValueError("shared module initialization mismatch")
        training_audit=audit_training(
            paths,
            config.counts,
            8 if config.protocol == "core-framework-v7" else 4 if (
                config.protocol in {
                    "stabilized-v3",
                    "stabilized-v4",
                    "stabilized-v4-corrected",
                    "core-framework-v1",
                    "core-framework-v2",
                    "core-framework-v3",
                    "core-framework-v4",
                    "core-framework-v4-delta-scale",
                    "core-framework-v5",
                    "core-framework-v6",
                    "core-framework-v8",
                    "core-framework-v9-action-selection",
                    "core-framework-v10-safety",
                    "core-framework-v11-comm-occluder",
                }
                or config.experiment_arm == "rollout4"
            ) else 1,
            learners=learners,
        )
        report.update(method_paths=paths, schema_reports=schema, initialization_ok=initialization_ok,
                      checkpoint_hashes=checkpoint_hashes, fixed_control_B_paths=b_paths,
                      fixed_control_B_isolation=b_isolation, counts=list(config.counts),
                      development_candidate_eligible=eligible, method_means=means,
                      fitting_audit=fitting_audit,
                      persistent_proxy_fit={
                          "enabled": bool(persistent_model is not None),
                          "horizon": int(config.persistent_horizon),
                          "source_split": "baseline_mappo/train" if persistent_model is not None else None,
                          "candidate_row_count": int(persistent_model_rows),
                          "validation_used_for_fitting": False,
                          "test_used_for_fitting": False,
                      },
                      rule_proxy_fit={
                          "enabled": True,
                          "source_split": "baseline_mappo/train",
                          "source_branch": BASELINE_BRANCH,
                          "model_file": "rule_measurement_proxy_model.json",
                          "candidate_row_count": int((rule_proxy or {}).get("candidate_row_count", 0)),
                          "validation_used_for_fitting": False,
                          "test_used_for_fitting": False,
                      },
                      training_audit=training_audit,
                      validation_isolation=isolation, active_methods=list(methods),
                      active_learned_methods=list(learners),
                      deprecated_methods=list(DEPRECATED_METHODS),
                      communication_constraint_enabled=not core_scope,
                      audit_ok=all(r["ok"] for r in schema.values()),
                      status="complete", decision_scope="seed42_descriptive_only")
        _write_json(logs/"audit_report.json", report)
        from .mainline_compact_figures import generate_compact_figures
        figure_profile = (
            "compact-proposed-v4"
            if config.protocol in {"core-framework-v4", "core-framework-v4-delta-scale", "core-framework-v5", "core-framework-v6", "core-framework-v7", "core-framework-v8", "core-framework-v9-action-selection", "core-framework-v10-safety", "core-framework-v11-comm-occluder"}
            else "compact-proposed-v3"
            if config.protocol == "core-framework-v3"
            else config.figure_profile
        )
        report["figures"] = generate_compact_figures(
            paths,
            rows,
            logs/"figures",
            plots,
            arm=config.experiment_arm,
            figure_profile=figure_profile,
        )
        # The action -> EKF -> feedback audit is deliberately read-only and
        # split-scoped.  It is kept beside the main figures so the training
        # curve cannot be interpreted without checking whether actions led to
        # eligible, accepted, and persistent updates.  The diagnostic itself
        # never opens the test split.
        from .action_feedback_diagnostic import run_action_feedback_diagnostic

        report["action_feedback_diagnostic"] = run_action_feedback_diagnostic(
            paths,
            logs / "action_feedback_diagnostic",
            plots / "action_feedback_diagnostic",
            methods=methods,
        )
        if config.protocol == "core-framework-v9-action-selection":
            from .action_selection_diagnostic import run_action_selection_diagnostic

            report["action_selection_diagnostic"] = run_action_selection_diagnostic(
                paths,
                logs / "action_selection_diagnostic",
                plots / "action_selection_diagnostic",
                methods=learners,
            )
        _write_json(logs/"audit_report.json", report)
        _write_json(root/"comparison_manifest.json", report)
        return {"audit_ok":report["audit_ok"], "audit_report":str(logs/"audit_report.json"),
                "status":report["status"], "stage":config.stage, "arm":config.experiment_arm,
                "development_candidate_eligible":eligible, "figures":report["figures"]}
    except Exception as exc:
        report.update(audit_ok=False, status="failed", error={"type":type(exc).__name__, "message":str(exc)}, method_paths=paths)
        _write_json(logs/"audit_report.json", report)
        raise


def finalize_development():
    """Apply the fixed candidate rule once all three bounded arms exist."""
    reports={arm:_read_json(LOG_ROOT/"development"/arm/"audit_report.json") for arm in ("reference","proxy-aligned","rollout4")}
    if not all(r.get("audit_ok") and r.get("status")=="complete" for r in reports.values()):
        raise RuntimeError("R/P/T must all finish structural acceptance")
    order={"reference":0,"proxy-aligned":1,"rollout4":2}
    def key(arm):
        row=next(r for r in reports[arm]["method_means"] if r["method"]==COMMUNICATION_BRANCH)
        return row["position_rmse_norm_m"],row["j_com"],row["actual_measurement_bytes"],order[arm]
    qualified=[arm for arm,r in reports.items() if r["development_candidate_eligible"]]
    selected=min(qualified,key=key) if qualified else "reference"
    logs=LOG_ROOT/"final"; plots=Path("plots/preliminary/mainline_baseline_comparison_v2_seed42/final")
    _fresh(logs,plots)
    rows,isolation=summarize(reports[selected]["method_paths"],20)
    _write_table(logs/"R_P_T_comparison.csv",[{"arm":arm,"eligible":r["development_candidate_eligible"],**row} for arm,r in reports.items() for row in r["method_means"]])
    from .mainline_compact_figures import generate_compact_figures
    figures=generate_compact_figures(reports[selected]["method_paths"],rows,logs/"figures",plots,arm=selected)
    report={"audit_ok":True,"selected_arm":selected,"optimization_success":bool(qualified),
            "selection_scope":"seed42 development only; selection bias; not cross-seed confirmation",
            "reason":"fixed eligibility and ranking rule" if qualified else "no eligible candidate; retain reference without claiming success",
            "cross_seed_stability":"not_evaluated","figures":figures,"isolation":isolation}
    _write_json(logs/"audit_report.json",report)
    return report
