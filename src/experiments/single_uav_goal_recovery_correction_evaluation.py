"""Frozen BC Actor evaluation with Goal Recovery Correction OFF versus ON.

This entry point is intentionally evaluation-only.  It does not call the
external-method study, the DAgger/BC runner, a Teacher controller, or any
training/update method.  OFF and ON use one ordered static-holdout manifest.
"""

from __future__ import annotations

if __package__ in {None, ""}:  # PyCharm "Run File" support.
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "src.experiments"

import argparse
import copy
import csv
import json
import math
import subprocess
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..control.goal_recovery_correction import (
    GoalRecoveryCorrection,
    GoalRecoveryCorrectionConfig,
    decompose_goal_relative_motion,
    fuse_actor_action,
)
from ..control.lateral_exit_braking_assistance import (
    LATERAL_EXIT_ASSIST_CAP_MPS,
    MinimalLateralExitBrakingAssistance,
)
from ..control.pass_commit_non_regression import (
    PASS_COMMIT_HARD_CAP_MPS,
    PassCommitNonRegression,
)
from ..control.pass_commit_action_decomposition import (
    SingleCallActionTransform,
    compose_pre_transform_action,
    post_safety_execution_residual_mps,
    pre_transform_decomposition_residual_mps,
)
from ..simulation.motion_model import apply_velocity_increment
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import FormalExperimentConfig, _make_environment
from .pilot import _assert_action_feasible, _assert_observation_finite, _seed_everything
from .single_uav_actor_constrained_oracle import actor_action_transform
from .single_uav_external_recovery_method_study import (
    SOFT_RECOVERY_BUFFER_M,
    _soft_pass_progress,
)
from .single_uav_obstacle_dagger_stage2b import (
    STATIC_SAFE_DISTANCE_M,
    _obstacle_extent,
    _physical_obstacles,
    point_obstacle_distance,
    select_relevant_obstacle,
)
from .single_uav_obstacle_dagger_stage2br import _actor_decision
from .single_uav_static_obstacle_dagger_recovery_goal import (
    _build_pool,
    _canonical_hash,
    _load_checkpoint,
    _visible_any,
    _write_rows_csv,
)
from .single_uav_checkpoint_evaluation import _state_dict_hash
from .single_uav_signed_bypass_observability import (
    DEFAULT_SIGN_DEADBAND_M,
    DEFAULT_SIGN_DEADBAND_MPS,
    SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION,
    build_horizontal_goal_frame,
    classify_signed_failure,
    compare_identity_components,
    decompose_signed_vector,
    passed_geometry_fields,
    pipeline_signed_components,
    sign_change_count,
    sign_value,
    signed_side,
    update_side_run,
)
from .single_uav_goal_recovery_stable_contract import (
    STABLE_LOG_SCHEMA_VERSION,
    aggregate_actual_stable_validation,
    aggregate_episode_stable_fields,
    assert_previous_formal_identity,
    episode_identity,
    extract_canonical_stable_fields,
    load_previous_formal_summary,
)


STAGE_NAME = "single_uav_goal_recovery_correction_evaluation"
DEFAULT_CHECKPOINT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_goal_recovery_correction_eval_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_goal_recovery_correction_eval_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_goal_recovery_correction_eval_seed2026"
)
ACTUAL_STABLE_DATA_ROOT = Path(
    "data/synthetic/single_uav_goal_recovery_actual_stable_validation_seed2026"
)
ACTUAL_STABLE_LOG_ROOT = Path(
    "logs/preliminary/single_uav_goal_recovery_actual_stable_validation_seed2026"
)
ACTUAL_STABLE_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_goal_recovery_actual_stable_validation_seed2026"
)
TERMINAL_CAPTURE_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_validation_seed2026"
)
TERMINAL_CAPTURE_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_validation_seed2026"
)
TERMINAL_CAPTURE_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_validation_seed2026"
)
TERMINAL_CAPTURE_V2_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v2_validation_seed2026"
)
TERMINAL_CAPTURE_V2_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v2_validation_seed2026"
)
TERMINAL_CAPTURE_V2_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v2_validation_seed2026"
)
TERMINAL_CAPTURE_V2_HORIZON_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v2_horizon120_diagnostic_seed2026"
)
TERMINAL_CAPTURE_V2_HORIZON_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v2_horizon120_diagnostic_seed2026"
)
TERMINAL_CAPTURE_V2_HORIZON_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v2_horizon120_diagnostic_seed2026"
)
TERMINAL_CAPTURE_V21_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v21_validation_seed2026"
)
TERMINAL_CAPTURE_V21_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v21_validation_seed2026"
)
TERMINAL_CAPTURE_V21_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v21_validation_seed2026"
)
TERMINAL_CAPTURE_V22_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v22_validation_seed2026"
)
TERMINAL_CAPTURE_V22_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v22_validation_seed2026"
)
TERMINAL_CAPTURE_V22_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v22_validation_seed2026"
)
TERMINAL_CAPTURE_V23_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v23_validation_seed2026"
)
TERMINAL_CAPTURE_V23_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v23_validation_seed2026"
)
TERMINAL_CAPTURE_V23_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v23_validation_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_DATA_ROOT = Path(
    "data/synthetic/single_uav_terminal_capture_v23_authority_repair_validation_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT = Path(
    "logs/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_DATA_ROOT = Path(
    "data/synthetic/single_uav_near_goal_capture_horizon120_diagnostic_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT = Path(
    "logs/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026"
)
TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026"
)
SIGNED_BYPASS_OBSERVABILITY_DATA_ROOT = Path(
    "data/synthetic/single_uav_signed_bypass_observability_diagnostic_seed2026"
)
SIGNED_BYPASS_OBSERVABILITY_LOG_ROOT = Path(
    "logs/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026"
)
SIGNED_BYPASS_OBSERVABILITY_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026"
)
LATERAL_EXIT_BRAKING_DATA_ROOT = Path(
    "data/synthetic/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
LATERAL_EXIT_BRAKING_LOG_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
LATERAL_EXIT_BRAKING_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
RADIAL_INTENT_PERSISTENCE_DATA_ROOT = Path(
    "data/synthetic/single_uav_radial_intent_persistence_diagnostic_seed2026"
)
RADIAL_INTENT_PERSISTENCE_LOG_ROOT = Path(
    "logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/run"
)
RADIAL_INTENT_PERSISTENCE_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026"
)
PASS_COMMIT_DATA_ROOT = Path(
    "data/synthetic/single_uav_pass_commit_diagnostic_seed2026_action_decomposition_fix_v1"
)
PASS_COMMIT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run_action_decomposition_fix_v1"
)
PASS_COMMIT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_pass_commit_diagnostic_seed2026_action_decomposition_fix_v1"
)
TERMINAL_CAPTURE_V2_FORMAL_BASELINE_ROOT = TERMINAL_CAPTURE_V2_LOG_ROOT
TERMINAL_CAPTURE_V1_BASELINE_ROOT = TERMINAL_CAPTURE_LOG_ROOT
TERMINAL_CAPTURE_V21_FORMAL_BASELINE_ROOT = TERMINAL_CAPTURE_V21_LOG_ROOT
TERMINAL_CAPTURE_V22_FORMAL_BASELINE_ROOT = TERMINAL_CAPTURE_V22_LOG_ROOT
TERMINAL_CAPTURE_V23_FORMAL_BASELINE_ROOT = TERMINAL_CAPTURE_V23_LOG_ROOT
ACTUAL_STABLE_BASELINE_ROOT = ACTUAL_STABLE_LOG_ROOT
PREVIOUS_FORMAL_LOG_ROOT = Path(
    "logs/preliminary/single_uav_goal_recovery_correction_eval_formal20_seed2026"
)
STATIC_HOLDOUT_SEED_BASE = 4_020_000
STATIC_HOLDOUT_ROUND = 99
FORMAL_EPISODES = 20
DEFAULT_SMOKE_EPISODES = 2
HORIZON_BASELINE_MAX_STEPS = 100
HORIZON_DIAGNOSTIC_MAX_STEPS = 120
HORIZON_FOCUS_EPISODE_IDS = (
    "test_r99_episode_0001",
    "test_r99_episode_0008",
    "test_r99_episode_0017",
)

EVALUATION_ONLY_FLAGS: dict[str, Any] = {
    "training_enabled": False,
    "training_performed": False,
    "ppo_updates": 0,
    "mappo_updates": 0,
    "bc_retraining_executed": False,
    "dagger_collection_executed": False,
    "teacher_bootstrap_executed": False,
    "teacher_constructed": False,
    "teacher_queried": False,
    "teacher_rollout_steps": 0,
    "optimizer_created": False,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "actor_updates": 0,
    "buffer_updates": 0,
    "checkpoint_writes": 0,
    "formal_training_experiment_executed": False,
    "reward_modified": False,
    "dynamics_modified": False,
    "observation_modified": False,
    "actor_modified": False,
    "critic_modified": False,
    "gat_modified": False,
    "control_algorithm_modified": False,
    "stable_goal_criterion_modified": False,
}


def _condition_names(condition: str) -> tuple[str, ...]:
    if condition == "paired":
        return ("off", "on")
    if condition in {"off", "on"}:
        return (condition,)
    raise ValueError("condition must be paired, off, or on")


def _validate_fresh_output_layout(
    output_root: Path,
    condition: str,
    checkpoint: Path,
) -> dict[str, Path]:
    """Resolve a fresh output layout without creating any directory."""

    requested_root = Path(output_root)
    if requested_root.is_symlink():
        raise ValueError("output root must not be a symbolic link")
    base = requested_root.resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if base.exists():
        raise FileExistsError(f"refusing to reuse existing output root: {base}")
    if base in {checkpoint_path, checkpoint_path.parent}:
        raise ValueError("output root cannot be the checkpoint or checkpoint directory")
    result = {"base": base}
    if condition == "paired":
        result.update({"off": base / "off", "on": base / "on"})
        if result["off"] == result["on"]:
            raise ValueError("paired OFF and ON roots must be distinct")
    else:
        result[condition] = base
    return result


def _prepare_output_layouts(
    *,
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    condition: str,
    checkpoint: Path,
) -> dict[str, dict[str, Path]]:
    layouts = {
        "data": _validate_fresh_output_layout(data_root, condition, checkpoint),
        "logs": _validate_fresh_output_layout(logs_root, condition, checkpoint),
        "plots": _validate_fresh_output_layout(plots_root, condition, checkpoint),
    }
    bases = [layout["base"] for layout in layouts.values()]
    if len(set(bases)) != len(bases):
        raise ValueError("data, logs, and plots roots must be distinct")
    for left in bases:
        for right in bases:
            if left != right and (left in right.parents or right in left.parents):
                raise ValueError("data, logs, and plots roots must not be nested")
    for layout in layouts.values():
        layout["base"].mkdir(parents=True, exist_ok=False)
        if condition == "paired":
            layout["off"].mkdir()
            layout["on"].mkdir()
    return layouts


def _ordered_pool_identity(
    pool: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str, int]]:
    return [
        (
            int(item["episode_index"]),
            str(item["episode_id"]),
            int(item["noise_seed"]),
        )
        for item in pool
    ]


def _assert_paired_identity(
    off_rows: Sequence[Mapping[str, Any]],
    on_rows: Sequence[Mapping[str, Any]],
) -> None:
    def identity(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, int]]:
        return [
            (
                int(row["episode_index"]),
                str(row["episode_id"]),
                int(row["noise_seed"]),
            )
            for row in rows
        ]

    if identity(off_rows) != identity(on_rows):
        raise AssertionError("paired OFF/ON episode identifiers or seeds differ")


def _git_identity() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"available": True, "revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "revision": None, "dirty": None}


def _runtime_identity_jsonable(
    value: Any,
    *,
    _active_object_ids: set[int] | None = None,
) -> Any:
    """Build a stable, read-only JSON-safe snapshot of runtime component state."""

    active = set() if _active_object_ids is None else _active_object_ids
    if isinstance(value, np.generic):
        return _runtime_identity_jsonable(
            value.item(), _active_object_ids=active
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        if math.isinf(value):
            return {"__nonfinite_float__": "+inf" if value > 0.0 else "-inf"}
        return value
    if callable(value):
        raise TypeError("callable runtime identity state is not serializable")
    object_id = id(value)
    if object_id in active:
        raise ValueError("RUNTIME_IDENTITY_OBJECT_CYCLE")
    active.add(object_id)
    try:
        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "values": _runtime_identity_jsonable(
                        value.tolist(), _active_object_ids=active
                    ),
                }
            }
        if isinstance(value, Mapping):
            items = [
                (key, item)
                for key, item in value.items()
                if not callable(item)
            ]
            if all(isinstance(key, str) for key, _ in items):
                return {
                    key: _runtime_identity_jsonable(
                        item, _active_object_ids=active
                    )
                    for key, item in items
                }
            encoded_items = [
                {
                    "key": _runtime_identity_jsonable(
                        key, _active_object_ids=active
                    ),
                    "value": _runtime_identity_jsonable(
                        item, _active_object_ids=active
                    ),
                }
                for key, item in items
            ]
            encoded_items.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            return {"__mapping_items__": encoded_items}
        if isinstance(value, (list, tuple)):
            return {
                "__sequence_type__": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    _runtime_identity_jsonable(item, _active_object_ids=active)
                    for item in value
                ],
            }
        if is_dataclass(value) and not isinstance(value, type):
            state = {
                field.name: getattr(value, field.name)
                for field in fields(value)
                if not callable(getattr(value, field.name))
            }
        elif hasattr(value, "__dict__"):
            state = {
                key: item
                for key, item in vars(value).items()
                if not callable(item)
            }
        else:
            raise TypeError(
                "unsupported runtime identity state type: "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        return {
            "__class__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "state": _runtime_identity_jsonable(
                state, _active_object_ids=active
            ),
        }
    finally:
        active.remove(object_id)


def _frozen_runtime_component_hashes(trainer: Any) -> dict[str, str]:
    """Hash frozen Actor/Critic/GAT/normalizer state at condition boundaries."""

    policy_parts = trainer.policy.parameter_initialization_hashes()
    actor_parts = {
        key: value for key, value in policy_parts.items()
        if key not in {"critic", "gat"}
    }
    normalizer_state = _runtime_identity_jsonable(trainer.normalizer)
    return {
        "actor": _canonical_hash(actor_parts),
        "critic": str(policy_parts.get("critic", "NOT_SEPARATELY_AVAILABLE")),
        "gat": str(policy_parts.get("gat", "NOT_SEPARATELY_AVAILABLE")),
        "normalizer": _canonical_hash(normalizer_state),
        "policy_combined": _state_dict_hash(trainer.policy.state_dict()),
    }


def _manifest_payload(
    pool: Sequence[Mapping[str, Any]],
    config: FormalExperimentConfig,
    checkpoint_identity: Mapping[str, Any],
    *,
    paired: bool,
) -> dict[str, Any]:
    ordered = [
        {
            "episode_index": int(item["episode_index"]),
            "episode_id": str(item["episode_id"]),
            "scenario_seed": int(item["scenario_seed"]),
            "noise_seed": int(item["noise_seed"]),
            "pair_id": str(item["pair_id"]),
            "scenario_hash": _canonical_hash(item["scenario"]),
            "initial_position": item["scenario"]["initial_positions"]["uav_00"],
            "goal": item["scenario"]["goals"]["uav_00"],
            "obstacle_specs": item["scenario"].get("obstacle_specs", []),
        }
        for item in pool
    ]
    identity = {
        "static_holdout_source": (
            "single_uav_static_obstacle_dagger_recovery_goal._build_pool"
        ),
        "static_holdout_seed_base": STATIC_HOLDOUT_SEED_BASE,
        "static_holdout_round": STATIC_HOLDOUT_ROUND,
        "ordered_scenarios": ordered,
        "max_steps": int(config.max_steps),
        "node_feature_dim": int(config.node_feature_dim),
        "safety_projection_enabled": True,
        "environment_factory": "src.experiments.formal._make_environment",
        "environment_config_identity": {
            "uav_count": int(config.uav_count),
            "max_steps": int(config.max_steps),
            "metric_scales": _jsonable(config.metric_scales),
            "single_uav_braking_safety_projection_enabled": bool(
                config.single_uav_braking_safety_projection_enabled
            ),
            "goal_recovery_radial_gain": float(config.goal_recovery_radial_gain),
            "goal_recovery_tangential_gain": float(
                config.goal_recovery_tangential_gain
            ),
            "goal_recovery_max_delta_v_mps": float(
                config.goal_recovery_max_delta_v_mps
            ),
        },
        "goal_threshold_m": float(config.single_uav_goal_threshold_m),
        "stable_hold_steps": int(config.single_uav_stable_hold_steps),
        "soft_recovery_buffer_m": float(SOFT_RECOVERY_BUFFER_M),
        "static_safe_distance_m": float(STATIC_SAFE_DISTANCE_M),
    }
    return {
        "stage": STAGE_NAME,
        "scenario_manifest_hash": _canonical_hash(identity),
        "identity": identity,
        "checkpoint": dict(checkpoint_identity),
        "git": _git_identity(),
        "paired_same_scenarios": bool(paired),
        **EVALUATION_ONLY_FLAGS,
    }


class _EstimatedRecoveryTracker:
    """Teacher-free replica of the existing estimated-state release gate."""

    def __init__(self) -> None:
        self.active_obstacle_id: str | None = None
        self.release_latched = False

    def update(
        self,
        *,
        visible: Mapping[str, Any],
        goal_position_m: np.ndarray,
        static_obstacles: Sequence[Any],
    ) -> dict[str, Any]:
        position = np.asarray(visible["position_m"], dtype=float).reshape(3)
        goal = np.asarray(goal_position_m, dtype=float).reshape(3)
        goal_error = goal - position
        goal_norm = float(np.linalg.norm(goal_error))
        goal_direction = (
            goal_error / goal_norm
            if np.isfinite(goal_norm) and goal_norm > 1.0e-12
            else np.zeros(3, dtype=float)
        )
        obstacle = select_relevant_obstacle(position, static_obstacles)
        obstacle_id = str(obstacle.obstacle_id) if obstacle is not None else None
        if obstacle_id != self.active_obstacle_id:
            self.active_obstacle_id = obstacle_id
            self.release_latched = False
        clearance = (
            float(point_obstacle_distance(position, obstacle))
            if obstacle is not None
            else float("inf")
        )
        along = float("nan")
        extent = 0.0
        pass_progress = 0.0
        if obstacle is not None and goal_norm > 1.0e-12:
            relative = np.asarray(obstacle.center_m, dtype=float).reshape(3) - position
            along = float(np.dot(relative, goal_direction))
            extent = float(_obstacle_extent(obstacle, goal_direction))
            pass_progress = _soft_pass_progress(along, extent)
            if (
                along < -(extent + SOFT_RECOVERY_BUFFER_M)
                and clearance >= STATIC_SAFE_DISTANCE_M + 0.5
            ):
                self.release_latched = True
        return {
            "release_latched": bool(self.release_latched),
            "post_bypass_recovery": bool(self.release_latched),
            "pass_progress": float(pass_progress),
            "relevant_obstacle_id": obstacle_id or "",
            "relevant_obstacle_clearance_m": float(clearance),
            "along_obstacle_m": float(along),
            "obstacle_extent_m": float(extent),
            "state_source": "pre_step_ekf_visible_state_plus_static_map",
        }


def _correction_config(
    config: FormalExperimentConfig,
    environment: Any,
    *,
    enabled: bool,
) -> GoalRecoveryCorrectionConfig:
    return GoalRecoveryCorrectionConfig(
        enabled=enabled,
        radial_gain=float(config.goal_recovery_radial_gain),
        tangential_gain=float(config.goal_recovery_tangential_gain),
        max_delta_v_mps=float(config.goal_recovery_max_delta_v_mps),
        activation_mode=str(config.goal_recovery_activation_mode),
        alpha_ramp_enabled=bool(config.goal_recovery_alpha_ramp_enabled),
        max_acceleration_mps2=float(environment.config.motion.max_acceleration_mps2),
        goal_threshold_m=float(config.single_uav_goal_threshold_m),
        minimum_goal_speed_mps=0.35,
        cruise_speed_mps=float(
            min(
                environment.config.motion.max_speed_mps,
                config.single_uav_target_speed_far_mps,
            )
        ),
        terminal_capture_enabled=bool(
            config.goal_recovery_terminal_capture_enabled
        ),
        terminal_capture_radius_m=float(
            config.goal_recovery_terminal_capture_radius_m
        ),
        terminal_radial_speed_floor_mps=float(
            config.goal_recovery_terminal_radial_speed_floor_mps
        ),
        terminal_radial_brake_gain=float(
            config.goal_recovery_terminal_radial_brake_gain
        ),
        terminal_tangential_gain=float(
            config.goal_recovery_terminal_tangential_gain
        ),
        terminal_propulsion_fade_enabled=bool(
            config.goal_recovery_terminal_propulsion_fade_enabled
        ),
        terminal_smooth_blend=bool(
            config.goal_recovery_terminal_smooth_blend
        ),
        terminal_capture_v2_enabled=bool(
            config.goal_recovery_terminal_capture_v2_enabled
        ),
        terminal_effective_deceleration_mps2=float(
            config.goal_recovery_terminal_effective_deceleration_mps2
        ),
        terminal_stopping_margin_m=float(
            config.goal_recovery_terminal_stopping_margin_m
        ),
        terminal_hold_inside_goal_enabled=bool(
            config.goal_recovery_terminal_hold_inside_goal_enabled
        ),
        terminal_radial_conflict_arbitration_enabled=bool(
            config.goal_recovery_terminal_radial_conflict_arbitration_enabled
        ),
        terminal_radial_conflict_compensation_gain=float(
            config.goal_recovery_terminal_radial_conflict_compensation_gain
        ),
        terminal_capture_v22_enabled=bool(
            config.goal_recovery_terminal_capture_v22_enabled
        ),
        terminal_hold_enter_radius_m=float(
            config.goal_recovery_terminal_hold_enter_radius_m
        ),
        terminal_hold_release_radius_m=float(
            config.goal_recovery_terminal_hold_release_radius_m
        ),
        terminal_hold_restorative_enabled=bool(
            config.goal_recovery_terminal_hold_restorative_enabled
        ),
        terminal_capture_v23_enabled=bool(
            config.goal_recovery_terminal_capture_v23_enabled
        ),
        terminal_hold_target_radius_m=float(
            config.goal_recovery_terminal_hold_target_radius_m
        ),
    )


def _correction_result_value(
    result: Any,
    name: str,
    default: Any,
) -> Any:
    """Read an optional controller result field without changing old schemas."""

    return default if result is None else getattr(result, name, default)


def _correction_result_vector(
    result: Any,
    name: str,
    default: np.ndarray,
) -> list[float]:
    value = _correction_result_value(result, name, default)
    if value is None:
        value = default
    return np.asarray(value, dtype=float).reshape(3).tolist()


def _build_encounter_horizontal_assist_frame(
    *,
    goal_position_m: Sequence[float],
    estimated_position_m: Sequence[float],
) -> tuple[Any | None, str, np.ndarray]:
    """Project the encounter-time EKF goal ray to XY before frame construction."""

    estimated_goal_ray = (
        np.asarray(goal_position_m, dtype=float).reshape(3)
        - np.asarray(estimated_position_m, dtype=float).reshape(3)
    )
    horizontal_goal_ray = estimated_goal_ray.copy()
    horizontal_goal_ray[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal_goal_ray))
    if not np.isfinite(horizontal_norm) or horizontal_norm <= 1.0e-9:
        return None, "horizontal_goal_frame_degenerate", horizontal_goal_ray
    return build_horizontal_goal_frame(horizontal_goal_ray), "", horizontal_goal_ray


def _rollout_frozen_actor_episode(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    item: Mapping[str, Any],
    correction_enabled: bool,
    signed_observability: bool = False,
    lateral_exit_assistance_enabled: bool | None = None,
    radial_intent_assistance: Any | None = None,
    pass_commit_assistance: Any | None = None,
    condition_label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one deterministic frozen Actor episode without constructing Teacher."""

    import torch

    reset_seed = int(item["noise_seed"])
    _seed_everything(reset_seed)
    scenario = copy.deepcopy(dict(item["scenario"]))
    environment = _make_environment(
        config,
        reset_seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, info = environment.reset(
        seed=reset_seed,
        scenario_config=scenario,
    )
    static_obstacles = tuple(_physical_obstacles(environment))
    obstacle_by_id = {
        str(getattr(obstacle, "obstacle_id", "")): obstacle
        for obstacle in static_obstacles
    }
    goal = np.asarray(scenario["goals"]["uav_00"], dtype=float).reshape(3)
    signed_base_frame = None
    if signed_observability:
        signed_base_frame = build_horizontal_goal_frame(
            goal
            - np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
        )
    tracker = _EstimatedRecoveryTracker()
    correction = (
        GoalRecoveryCorrection(
            _correction_config(config, environment, enabled=True)
        )
        if correction_enabled
        else None
    )
    # Reset the controller's hysteresis state after environment/controller
    # construction and before the first policy decision.  This is essential
    # for independent deterministic episodes, especially for v2.2 hold state.
    if correction is not None:
        correction.reset()
    lateral_exit_assistance = (
        MinimalLateralExitBrakingAssistance(
            enabled=bool(lateral_exit_assistance_enabled),
            cap_mps=LATERAL_EXIT_ASSIST_CAP_MPS,
        )
        if lateral_exit_assistance_enabled is not None
        else None
    )
    if lateral_exit_assistance is not None and not signed_observability:
        raise ValueError("lateral-exit assistance requires signed observability")
    rows: list[dict[str, Any]] = []
    episode_return = 0.0
    terminated_flag = False
    truncated_flag = False
    first_1m_step: int | None = None
    stable_step: int | None = None
    # Signed observability is a read-only side channel.  These variables are
    # intentionally kept out of the controller/observation path.
    signed_frame = None
    signed_encounter_step: int | None = None
    signed_encounter_estimated_position: np.ndarray | None = None
    signed_encounter_actual_position: np.ndarray | None = None
    signed_anchor_actor = 0
    signed_anchor_actual = 0
    signed_actor_lateral_values: list[float] = []
    signed_actual_lateral_values: list[float] = []
    signed_prev_clearance = float("nan")
    signed_min_clearance = float("inf")
    signed_actor_run = 0
    signed_actual_run = 0
    signed_actor_sign_changes = 0
    signed_actual_sign_changes = 0
    signed_side_comparable_steps = 0
    signed_side_agreement_steps = 0
    assist_frame = None
    assist_frame_unavailable_reason = ""
    assist_horizontal_goal_ray = np.full(3, np.nan, dtype=float)
    assist_obstacle_id = ""
    assist_outcome_actual_position_anchor = None
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        visible = _visible_any(observations["uav_00"], config)
        pre_step_estimated_speed = float(
            np.linalg.norm(np.asarray(visible["velocity_mps"], dtype=float))
        )
        with torch.inference_mode():
            _decision, actor_raw, _actor_observation = _actor_decision(
                trainer, observations, info
            )
        actor_raw = np.asarray(actor_raw, dtype=float).reshape(3)
        context = tracker.update(
            visible=visible,
            goal_position_m=goal,
            static_obstacles=static_obstacles,
        )
        actual_pre_position = np.asarray(
            environment.states["uav_00"].position_m, dtype=float
        ).reshape(3).copy()
        actual_pre_velocity = np.asarray(
            environment.states["uav_00"].velocity_mps, dtype=float
        ).reshape(3).copy()
        signed_step: dict[str, Any] = {}
        if signed_observability:
            current_frame = signed_base_frame
            obstacle_id = str(context["relevant_obstacle_id"])
            obstacle = obstacle_by_id.get(obstacle_id)
            along = float(context["along_obstacle_m"])
            extent = float(context["obstacle_extent_m"])
            # The first tracker-visible entry at the existing extent+buffer
            # boundary freezes the frame and encounter anchor.  No new
            # threshold is introduced; it reuses the release-gate geometry.
            if (
                signed_frame is None
                and obstacle is not None
                and np.isfinite(along)
                and along <= extent + SOFT_RECOVERY_BUFFER_M
            ):
                signed_frame = current_frame
                signed_encounter_step = int(step)
                signed_encounter_estimated_position = np.asarray(
                    visible["position_m"], dtype=float
                ).reshape(3).copy()
                signed_encounter_actual_position = actual_pre_position.copy()
                if lateral_exit_assistance is not None:
                    # The treatment frame is frozen only from pre-step EKF-visible
                    # position, the static goal, and the selected static obstacle.
                    (
                        assist_frame,
                        assist_frame_unavailable_reason,
                        assist_horizontal_goal_ray,
                    ) = _build_encounter_horizontal_assist_frame(
                        goal_position_m=goal,
                        estimated_position_m=visible["position_m"],
                    )
                    assist_obstacle_id = obstacle_id
                    if assist_frame is not None:
                        # Outcome-only anchor: never passed to the assistance helper.
                        assist_outcome_actual_position_anchor = actual_pre_position.copy()
            axis = signed_frame if signed_frame is not None else current_frame
            if obstacle is not None:
                obstacle_relative = np.asarray(obstacle.center_m, dtype=float).reshape(3) - np.asarray(visible["position_m"], dtype=float).reshape(3)
                obstacle_components = decompose_signed_vector(obstacle_relative, axis)
                obstacle_side = signed_side(
                    obstacle_components["lateral"], deadband=DEFAULT_SIGN_DEADBAND_M
                )
            else:
                obstacle_relative = np.full(3, np.nan, dtype=float)
                obstacle_components = {
                    "radial": float("nan"),
                    "lateral": float("nan"),
                    "vertical": float("nan"),
                    "reconstruction_residual": float("nan"),
                }
                obstacle_side = "AMBIGUOUS"
            signed_step.update(
                {
                    "signed_observability_schema_version": SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION,
                    "signed_axis_source": axis.axis_source,
                    "signed_axis_sign_convention": axis.sign_convention,
                    "signed_axis_radial_unit": axis.radial_unit.tolist(),
                    "signed_axis_lateral_unit": axis.lateral_unit.tolist(),
                    "signed_axis_vertical_unit": axis.vertical_unit.tolist(),
                    "signed_axis_frozen": int(signed_frame is not None),
                    "signed_encounter_step": "" if signed_encounter_step is None else int(signed_encounter_step),
                    "signed_encounter_estimated_position_m": (
                        signed_encounter_estimated_position.tolist()
                        if signed_encounter_estimated_position is not None else []
                    ),
                    "signed_encounter_actual_position_m": (
                        signed_encounter_actual_position.tolist()
                        if signed_encounter_actual_position is not None else []
                    ),
                    "obstacle_center_m": (
                        np.asarray(obstacle.center_m, dtype=float).reshape(3).tolist()
                        if obstacle is not None else []
                    ),
                    "obstacle_relative_radial_m": obstacle_components["radial"],
                    "obstacle_relative_lateral_m": obstacle_components["lateral"],
                    "obstacle_relative_vertical_m": obstacle_components["vertical"],
                    "obstacle_relative_vector_m": obstacle_relative.tolist(),
                    "signed_obstacle_side": obstacle_side,
                    "recommended_geometric_bypass_side": "AMBIGUOUS",
                    "recommended_geometric_bypass_side_source": "not_inferred_without_teacher_or_oracle",
                }
            )
            signed_step.update(
                passed_geometry_fields(
                    along,
                    extent,
                    float(context["relevant_obstacle_clearance_m"]),
                    buffer_m=SOFT_RECOVERY_BUFFER_M,
                    safe_distance_m=STATIC_SAFE_DISTANCE_M,
                )
            )
        relative = decompose_goal_relative_motion(
            visible["position_m"], visible["velocity_mps"], goal
        )
        zero = np.zeros(3, dtype=float)
        proposed = zero.copy()
        applied_correction = zero.copy()
        alpha = 0.0
        eligible = False
        active = False
        reason = "disabled"
        corrected_pre_transform = actor_raw.copy()
        pre_safety = actor_raw.copy()
        transform_applied = False
        radial_delta = zero.copy()
        tangential_delta = zero.copy()
        desired_radial_velocity = 0.0
        correction_result = None
        assist_decision = None
        if correction is not None:
            result = correction.compute(
                estimated_position_m=visible["position_m"],
                estimated_velocity_mps=visible["velocity_mps"],
                goal_position_m=goal,
                release_latched=bool(context["release_latched"]),
                post_bypass_recovery=bool(context["post_bypass_recovery"]),
                pass_progress=float(context["pass_progress"]),
                action_provenance="evaluation",
                actor_raw_delta_v_mps=actor_raw,
            )
            proposed = result.correction_delta_v_mps.copy()
            alpha = float(result.correction_alpha)
            applied_correction = alpha * proposed
            eligible = bool(result.eligible)
            active = bool(result.active)
            reason = str(result.activation_reason)
            radial_delta = result.radial_correction_delta_v_mps.copy()
            tangential_delta = result.tangential_correction_delta_v_mps.copy()
            desired_radial_velocity = float(result.desired_radial_velocity_mps)
            correction_result = result
        action_limit_mps = float(
            environment.config.motion.max_acceleration_mps2
            * environment.config.motion.dt_s
        )
        pass_commit_action_transform = (
            SingleCallActionTransform(actor_action_transform, action_limit_mps)
            if pass_commit_assistance is not None
            else None
        )

        def apply_action_transform_once(value: Any) -> Any:
            if pass_commit_action_transform is not None:
                return pass_commit_action_transform(value)
            return actor_action_transform(value, action_limit_mps)

        frozen_obstacle_valid = False
        assist_clearance = float("nan")
        if lateral_exit_assistance is not None:
            if abs(action_limit_mps - 0.6) > 1.0e-12:
                raise ValueError("lateral-exit diagnostic requires the frozen 0.6 m/s action cap")
            frozen_obstacle = obstacle_by_id.get(assist_obstacle_id)
            assist_clearance = (
                float(
                    point_obstacle_distance(
                        np.asarray(visible["position_m"], dtype=float).reshape(3),
                        frozen_obstacle,
                    )
                )
                if assist_frame is not None and frozen_obstacle is not None
                else float("nan")
            )
            frozen_obstacle_valid = bool(
                assist_frame is not None
                and frozen_obstacle is not None
                and str(context["relevant_obstacle_id"]) == assist_obstacle_id
            )
            if assist_frame is not None:
                assist_decision = lateral_exit_assistance.compute(
                    step=int(step),
                    encounter_established=bool(frozen_obstacle is not None),
                    frozen_obstacle_valid=frozen_obstacle_valid,
                    estimated_obstacle_clearance_m=assist_clearance,
                    estimated_velocity_mps=visible["velocity_mps"],
                    actor_delta_v_mps=actor_raw,
                    encounter_lateral_unit=assist_frame.lateral_unit,
                    goal_recovery_active=bool(
                        context["post_bypass_recovery"]
                        or (correction_result is not None and correction_result.active)
                    ),
                )
        radial_intent_decision = None
        radial_intent_frame = assist_frame
        if radial_intent_assistance is not None:
            if signed_base_frame is None:
                raise AssertionError("radial-intent diagnostics require signed estimated observability")
            radial_unit = (
                radial_intent_frame.radial_unit
                if radial_intent_frame is not None
                else signed_base_frame.radial_unit
            )
            estimated_lateral_velocity = (
                float(np.dot(
                    np.asarray(visible["velocity_mps"], dtype=float).reshape(3),
                    radial_intent_frame.lateral_unit,
                ))
                if radial_intent_frame is not None
                else float("nan")
            )
            radial_intent_decision = radial_intent_assistance.evaluate(
                step=int(step),
                actor_raw_delta_v_mps=actor_raw,
                radial_unit=radial_unit,
                estimated_lateral_velocity_mps=estimated_lateral_velocity,
                frame_frozen=bool(radial_intent_frame is not None),
                relevant_obstacle_match=bool(
                    frozen_obstacle_valid
                    and assist_obstacle_id
                    and str(context["relevant_obstacle_id"]) == assist_obstacle_id
                ),
                release_latched=bool(context["release_latched"]),
                post_bypass_recovery=bool(context["post_bypass_recovery"]),
                passed_obstacle=bool(
                    context["release_latched"]
                    or signed_step.get("passed_condition_now", False)
                ),
                goal_recovery_active=bool(
                    correction_result is not None and correction_result.active
                ),
                lateral_assist_active=bool(
                    assist_decision is not None and assist_decision.active
                ),
                done=bool(terminated_flag or truncated_flag),
            )
        pass_commit_decision = None
        if pass_commit_assistance is not None:
            if radial_intent_assistance is not None:
                raise AssertionError("PASS_COMMIT requires radial-intent assistance OFF")
            frozen_obstacle = obstacle_by_id.get(assist_obstacle_id)
            pass_lateral_velocity = (
                float(assist_decision.estimated_lateral_velocity_mps)
                if assist_decision is not None
                else (
                    float(np.dot(
                        np.asarray(visible["velocity_mps"], dtype=float).reshape(3),
                        assist_frame.lateral_unit,
                    ))
                    if assist_frame is not None else float("nan")
                )
            )
            pass_commit_decision = pass_commit_assistance.evaluate(
                step=int(step),
                estimated_position_m=visible["position_m"],
                goal_position_m=goal,
                obstacle_center_m=(
                    np.asarray(frozen_obstacle.center_m, dtype=float).reshape(3)
                    if frozen_obstacle is not None
                    else np.asarray(goal, dtype=float).reshape(3)
                ),
                obstacle_extent_m=float(context["obstacle_extent_m"]),
                pass_buffer_m=SOFT_RECOVERY_BUFFER_M,
                logged_along_obstacle_m=float(context["along_obstacle_m"]),
                logged_passed_margin_m=float(
                    signed_step.get("passed_margin_m", float("nan"))
                ),
                estimated_velocity_mps=visible["velocity_mps"],
                baseline_candidate_delta_v_mps=actor_raw,
                estimated_lateral_velocity_mps=pass_lateral_velocity,
                encounter_context_established=bool(
                    assist_frame is not None and bool(assist_obstacle_id)
                ),
                frozen_obstacle_matches=bool(
                    frozen_obstacle_valid
                    and assist_obstacle_id
                    and str(context["relevant_obstacle_id"]) == assist_obstacle_id
                ),
                relevant_obstacle_available=frozen_obstacle is not None,
                relevant_obstacle_clearance_m=float(
                    context["relevant_obstacle_clearance_m"]
                ),
                passed_obstacle=bool(context["release_latched"]),
                passed_condition_now=bool(
                    signed_step.get("passed_condition_now", False)
                ),
                release_latched=bool(context["release_latched"]),
                post_bypass_recovery=bool(context["post_bypass_recovery"]),
                goal_recovery_active=bool(
                    (correction_result is not None and correction_result.active)
                    or context["post_bypass_recovery"]
                ),
                lateral_exit_assist_active=bool(
                    assist_decision is not None and assist_decision.active
                ),
                past_closest_approach=bool(
                    assist_decision is not None
                    and assist_decision.closest_approach_gate
                ),
                clearance_increasing=bool(
                    assist_decision is not None
                    and assist_decision.clearance_increasing_gate
                ),
                episode_done=bool(terminated_flag or truncated_flag),
            )
        if radial_intent_decision is not None and radial_intent_decision.active:
            from .single_uav_radial_intent_persistence_diagnostic import (
                assert_radial_intent_exclusive as _assert_radial_intent_exclusive,
                transform_radial_assisted_delta_v_once as _transform_radial_intent_once,
            )

            _assert_radial_intent_exclusive(
                radial_intent_active=True,
                goal_recovery_active=bool(
                    correction_result is not None and correction_result.active
                ),
                lateral_assist_active=bool(
                    assist_decision is not None and assist_decision.active
                ),
            )
            if abs(action_limit_mps - 0.6) > 1.0e-12:
                raise ValueError("radial-intent diagnostic requires the frozen 0.6 m/s action cap")
            base_pre_transform = (
                assist_decision.combined_pre_transform_delta_v_mps.copy()
                if assist_decision is not None and assist_decision.active
                else actor_raw.copy()
            )
            combined_values, constrained_values = _transform_radial_intent_once(
                base_delta_v_mps=base_pre_transform,
                radial_assist_delta_v_mps=(
                    radial_intent_decision.applied_radial_assist_delta_v_mps
                ),
                action_transform=(
                    pass_commit_action_transform
                    if pass_commit_action_transform is not None
                    else actor_action_transform
                ),
                action_limit_mps=action_limit_mps,
            )
            combined_pre_transform = np.asarray(combined_values, dtype=float).reshape(3)
            corrected_pre_transform = combined_pre_transform.copy()
            pre_safety = np.asarray(constrained_values, dtype=float).reshape(3)
            transform_applied = True
        elif assist_decision is not None and assist_decision.active:
            if correction_result is not None and correction_result.active:
                raise AssertionError("lateral-exit assistance overlapped Goal Recovery")
            transformed = apply_action_transform_once(
                assist_decision.combined_pre_transform_delta_v_mps
            )
            corrected_pre_transform = (
                assist_decision.combined_pre_transform_delta_v_mps.copy()
            )
            pre_safety = np.asarray(
                transformed["constrained_delta_v_mps"], dtype=float
            ).reshape(3)
            transform_applied = True
        elif pass_commit_decision is not None and pass_commit_decision.active:
            if correction_result is not None and correction_result.active:
                raise AssertionError("PASS_COMMIT overlapped Goal Recovery")
            if radial_intent_decision is not None and radial_intent_decision.active:
                raise AssertionError("PASS_COMMIT overlapped radial-intent assistance")
            combined = np.asarray(
                compose_pre_transform_action(
                    actor_base_candidate_mps=actor_raw,
                    pass_commit_applied_mps=pass_commit_decision.applied_vector_mps,
                    pass_commit_enabled=pass_commit_decision.enabled,
                    pass_commit_active=pass_commit_decision.active,
                    goal_recovery_active=bool(
                        correction_result is not None and correction_result.active
                    ),
                    lateral_exit_active=bool(
                        assist_decision is not None and assist_decision.active
                    ),
                    radial_intent_active=bool(
                        radial_intent_decision is not None
                        and radial_intent_decision.active
                    ),
                ),
                dtype=float,
            ).reshape(3)
            transformed = apply_action_transform_once(combined)
            corrected_pre_transform = combined.copy()
            pre_safety = np.asarray(
                transformed["constrained_delta_v_mps"], dtype=float
            ).reshape(3)
            transform_applied = True
        elif correction_result is not None:
            # This is the pre-existing path.  In particular, assistance OFF
            # leaves an inactive correction as a raw-Actor/no-transform bypass.
            fusion = fuse_actor_action(
                actor_raw,
                correction_result,
                apply_action_transform_once,
            )
            corrected_pre_transform = (
                fusion.corrected_delta_v_pre_action_transform_mps.copy()
            )
            pre_safety = fusion.corrected_delta_v_pre_safety_mps.copy()
            transform_applied = bool(fusion.transform_applied)
        if pass_commit_action_transform is not None:
            pass_commit_action_transform.assert_call_count(int(transform_applied))
        _assert_action_feasible(
            environment, {"uav_00": {"delta_v": pre_safety.tolist()}}
        )
        _nominal_state, nominal_execution = apply_velocity_increment(
            environment.states["uav_00"],
            pre_safety,
            environment.config.motion,
            safety_projector=None,
        )
        next_observations, rewards, terminated, truncated, next_info = environment.step(
            {
                "uav_00": {
                    "delta_v": pre_safety.tolist(),
                    "measurement_uav_id": "none",
                    "bandwidth": {},
                }
            }
        )
        stable_fields = extract_canonical_stable_fields(next_info)
        actual_relative = decompose_goal_relative_motion(
            environment.states["uav_00"].position_m,
            environment.states["uav_00"].velocity_mps,
            goal,
        )
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        applied = np.asarray(
            execution.get("delta_v_applied", pre_safety), dtype=float
        ).reshape(3)
        next_visible = _visible_any(next_observations["uav_00"], config)
        next_distance = float(next_visible["distance_m"])
        if signed_observability:
            # Everything below is a projection of values already produced by
            # the frozen rollout.  It is deliberately after ``environment.step``
            # and cannot feed back into the action or RNG state.
            axis = signed_frame or signed_base_frame
            actual_post_position = np.asarray(
                environment.states["uav_00"].position_m, dtype=float
            ).reshape(3).copy()
            actual_post_velocity = np.asarray(
                environment.states["uav_00"].velocity_mps, dtype=float
            ).reshape(3).copy()
            nominal_applied = np.asarray(
                nominal_execution.get("delta_v_applied", pre_safety), dtype=float
            ).reshape(3)
            motion_boundary_delta = nominal_applied - pre_safety
            safety_delta = applied - nominal_applied
            signed_step.update(
                pipeline_signed_components(
                    {
                        "actor_raw_delta_v_mps": actor_raw,
                        "pre_transform_delta_v_mps": corrected_pre_transform,
                        "post_transform_delta_v_mps": pre_safety,
                        "pre_safety_delta_v_mps": pre_safety,
                        # ``apply_velocity_increment`` applies the existing
                        # boundary stage before the safety projector; the
                        # names below make that order explicit.
                        "post_boundary_delta_v_mps": nominal_applied,
                        "post_safety_delta_v_mps": applied,
                        "final_applied_delta_v_mps": applied,
                    },
                    axis,
                )
            )
            actual_velocity_components = decompose_signed_vector(
                actual_post_velocity, axis
            )
            estimated_velocity_components = decompose_signed_vector(
                np.asarray(next_visible["velocity_mps"], dtype=float), axis
            )
            actual_step_displacement = actual_post_position - actual_pre_position
            estimated_step_displacement = np.asarray(
                next_visible["position_m"], dtype=float
            ).reshape(3) - np.asarray(visible["position_m"], dtype=float).reshape(3)
            actual_displacement_components = decompose_signed_vector(
                actual_step_displacement, axis
            )
            estimated_displacement_components = decompose_signed_vector(
                estimated_step_displacement, axis
            )
            clearance = float(context["relevant_obstacle_clearance_m"])
            clearance_increasing = bool(
                np.isfinite(clearance)
                and np.isfinite(signed_prev_clearance)
                and clearance > signed_prev_clearance + 1.0e-9
            )
            past_closest = bool(
                np.isfinite(clearance)
                and np.isfinite(signed_min_clearance)
                and signed_min_clearance < float("inf")
                and clearance > signed_min_clearance + 1.0e-9
            )
            if np.isfinite(clearance):
                signed_min_clearance = min(signed_min_clearance, clearance)
            signed_prev_clearance = clearance
            actor_components = decompose_signed_vector(actor_raw, axis)
            actor_lateral = float(actor_components["lateral"])
            actual_lateral = float(actual_velocity_components["lateral"])
            current_actor_sign = sign_value(
                actor_lateral, deadband=DEFAULT_SIGN_DEADBAND_MPS
            )
            current_actual_sign = sign_value(
                actual_lateral, deadband=DEFAULT_SIGN_DEADBAND_MPS
            )
            side_agree_now = bool(
                current_actor_sign
                and current_actual_sign
                and current_actor_sign == current_actual_sign
            )
            if current_actor_sign and current_actual_sign:
                signed_side_comparable_steps += 1
                signed_side_agreement_steps += int(side_agree_now)
            if signed_anchor_actor == 0 and current_actor_sign:
                signed_anchor_actor = current_actor_sign
            if signed_anchor_actual == 0 and current_actual_sign:
                signed_anchor_actual = current_actual_sign
            if signed_actor_lateral_values:
                previous_sign = sign_value(
                    signed_actor_lateral_values[-1], deadband=DEFAULT_SIGN_DEADBAND_MPS
                )
                if previous_sign and current_actor_sign and previous_sign != current_actor_sign:
                    signed_actor_sign_changes += 1
            if signed_actual_lateral_values:
                previous_sign = sign_value(
                    signed_actual_lateral_values[-1], deadband=DEFAULT_SIGN_DEADBAND_MPS
                )
                if previous_sign and current_actual_sign and previous_sign != current_actual_sign:
                    signed_actual_sign_changes += 1
            signed_actor_lateral_values.append(actor_lateral)
            signed_actual_lateral_values.append(actual_lateral)
            signed_actor_run = update_side_run(
                signed_actor_run,
                actor_lateral,
                signed_anchor_actor,
                deadband=DEFAULT_SIGN_DEADBAND_MPS,
            ) if signed_anchor_actor else 0
            signed_actual_run = update_side_run(
                signed_actual_run,
                actual_lateral,
                signed_anchor_actual,
                deadband=DEFAULT_SIGN_DEADBAND_MPS,
            ) if signed_anchor_actual else 0
            actual_from_encounter = float("nan")
            estimated_from_encounter = float("nan")
            if signed_encounter_actual_position is not None:
                actual_from_encounter = float(
                    np.dot(actual_post_position - signed_encounter_actual_position, axis.lateral_unit)
                )
            if signed_encounter_estimated_position is not None:
                estimated_from_encounter = float(
                    np.dot(
                        np.asarray(next_visible["position_m"], dtype=float).reshape(3)
                        - signed_encounter_estimated_position,
                        axis.lateral_unit,
                    )
                )
            signed_step.update(
                {
                    "actor_raw_radial_delta_v_mps": actor_components["radial"],
                    "actor_raw_lateral_delta_v_mps": actor_components["lateral"],
                    "actor_raw_vertical_delta_v_mps": actor_components["vertical"],
                    "post_transform_radial_delta_v_mps": signed_step["post_transform_delta_v_mps_radial"],
                    "post_transform_lateral_delta_v_mps": signed_step["post_transform_delta_v_mps_lateral"],
                    "post_transform_vertical_delta_v_mps": signed_step["post_transform_delta_v_mps_vertical"],
                    "pre_safety_radial_delta_v_mps": signed_step["pre_safety_delta_v_mps_radial"],
                    "pre_safety_lateral_delta_v_mps": signed_step["pre_safety_delta_v_mps_lateral"],
                    "pre_safety_vertical_delta_v_mps": signed_step["pre_safety_delta_v_mps_vertical"],
                    "post_boundary_radial_delta_v_mps": signed_step["post_boundary_delta_v_mps_radial"],
                    "post_boundary_lateral_delta_v_mps": signed_step["post_boundary_delta_v_mps_lateral"],
                    "post_boundary_vertical_delta_v_mps": signed_step["post_boundary_delta_v_mps_vertical"],
                    "post_safety_radial_delta_v_mps": signed_step["post_safety_delta_v_mps_radial"],
                    "post_safety_lateral_delta_v_mps": signed_step["post_safety_delta_v_mps_lateral"],
                    "post_safety_vertical_delta_v_mps": signed_step["post_safety_delta_v_mps_vertical"],
                    "final_applied_radial_delta_v_mps": signed_step["final_applied_delta_v_mps_radial"],
                    "final_applied_lateral_delta_v_mps": signed_step["final_applied_delta_v_mps_lateral"],
                    "final_applied_vertical_delta_v_mps": signed_step["final_applied_delta_v_mps_vertical"],
                    "motion_boundary_radial_delta_v_mps": float(np.dot(motion_boundary_delta, axis.radial_unit)),
                    "motion_boundary_lateral_delta_v_mps": float(np.dot(motion_boundary_delta, axis.lateral_unit)),
                    "motion_boundary_vertical_delta_v_mps": float(np.dot(motion_boundary_delta, axis.vertical_unit)),
                    "safety_projection_radial_delta_v_mps": float(np.dot(safety_delta, axis.radial_unit)),
                    "safety_projection_lateral_delta_v_mps": float(np.dot(safety_delta, axis.lateral_unit)),
                    "safety_projection_vertical_delta_v_mps": float(np.dot(safety_delta, axis.vertical_unit)),
                    "estimated_radial_velocity_mps": estimated_velocity_components["radial"],
                    "estimated_lateral_velocity_mps": estimated_velocity_components["lateral"],
                    "estimated_vertical_velocity_mps": estimated_velocity_components["vertical"],
                    "actual_radial_velocity_mps": actual_velocity_components["radial"],
                    "actual_lateral_velocity_mps": actual_velocity_components["lateral"],
                    "actual_vertical_velocity_mps": actual_velocity_components["vertical"],
                    "estimated_radial_displacement_step_m": estimated_displacement_components["radial"],
                    "estimated_lateral_displacement_step_m": estimated_displacement_components["lateral"],
                    "estimated_vertical_displacement_step_m": estimated_displacement_components["vertical"],
                    "actual_radial_displacement_step_m": actual_displacement_components["radial"],
                    "actual_lateral_displacement_step_m": actual_displacement_components["lateral"],
                    "actual_vertical_displacement_step_m": actual_displacement_components["vertical"],
                    "estimated_lateral_displacement_from_encounter_m": estimated_from_encounter,
                    "actual_lateral_displacement_from_encounter_m": actual_from_encounter,
                    "actor_lateral_sign": int(current_actor_sign),
                    "actual_lateral_sign": int(current_actual_sign),
                    "actor_lateral_sign_changes": int(signed_actor_sign_changes),
                    "actual_lateral_sign_changes": int(signed_actual_sign_changes),
                    "actor_same_side_run_length": int(signed_actor_run),
                    "actual_same_side_run_length": int(signed_actual_run),
                    "actor_still_same_side": bool(
                        signed_anchor_actor != 0 and current_actor_sign == signed_anchor_actor
                    ),
                    "actual_still_same_side": bool(
                        signed_anchor_actual != 0 and current_actual_sign == signed_anchor_actual
                    ),
                    "actor_anchor_lateral_sign": int(signed_anchor_actor),
                    "actual_anchor_lateral_sign": int(signed_anchor_actual),
                    "actor_actual_lateral_side_agree_now": side_agree_now,
                    "actor_actual_lateral_side_agreement_fraction": (
                        float(signed_side_agreement_steps / signed_side_comparable_steps)
                        if signed_side_comparable_steps else float("nan")
                    ),
                    "clearance_increasing": clearance_increasing,
                    "past_closest_approach": past_closest,
                    "signed_obstacle_along_goal_m": float(context["along_obstacle_m"]),
                    "signed_obstacle_extent_m": float(context["obstacle_extent_m"]),
                    "signed_obstacle_clearance_m": clearance,
                }
            )
        if first_1m_step is None and int(stable_fields["goal_distance_ok"]):
            first_1m_step = step
        if stable_step is None and bool(next_info.get("single_uav_goal_stable", False)):
            stable_step = step
        reward = float(rewards.get("uav_00", 0.0))
        episode_return += reward
        projection_delta = applied - pre_safety
        assist_applied = (
            assist_decision.applied_assist_delta_v_mps
            if assist_decision is not None else np.zeros(3, dtype=float)
        )
        radial_intent_applied = (
            np.asarray(
                radial_intent_decision.applied_radial_assist_delta_v_mps,
                dtype=float,
            ).reshape(3)
            if radial_intent_decision is not None
            else np.zeros(3, dtype=float)
        )
        pass_commit_applied = (
            np.asarray(pass_commit_decision.applied_vector_mps, dtype=float).reshape(3)
            if pass_commit_decision is not None
            else np.zeros(3, dtype=float)
        )
        reconstruction_residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=corrected_pre_transform,
            actor_base_candidate_mps=actor_raw,
            goal_recovery_applied_mps=applied_correction,
            lateral_exit_applied_mps=assist_applied,
            pass_commit_applied_mps=pass_commit_applied,
            radial_intent_applied_mps=radial_intent_applied,
            pass_commit_enabled=bool(
                pass_commit_decision is not None and pass_commit_decision.enabled
            ),
            pass_commit_active=bool(
                pass_commit_decision is not None and pass_commit_decision.active
            ),
            goal_recovery_active=bool(
                correction_result is not None and correction_result.active
            ),
            lateral_exit_active=bool(
                assist_decision is not None and assist_decision.active
            ),
            radial_intent_active=bool(
                radial_intent_decision is not None and radial_intent_decision.active
            ),
        )
        final_residual = post_safety_execution_residual_mps(
            pre_safety_action_mps=pre_safety,
            projection_delta_mps=projection_delta,
            final_applied_action_mps=applied,
        )
        row = {
            "condition": condition_label or ("on" if correction_enabled else "off"),
            "episode_index": int(item["episode_index"]),
            "episode_id": str(item["episode_id"]),
            "scenario_id": str(item["episode_id"]),
            "scenario_seed": int(item["scenario_seed"]),
            "noise_seed": reset_seed,
            "step": int(step),
            "pre_step_estimated_speed_mps": pre_step_estimated_speed,
            "actor_raw_delta_v_mps": actor_raw.tolist(),
            "actor_raw_delta_v_norm_mps": float(np.linalg.norm(actor_raw)),
            "goal_recovery_correction_enabled": int(correction_enabled),
            "goal_recovery_correction_eligible": int(eligible),
            "goal_recovery_correction_active": int(active),
            "goal_recovery_correction_reason": reason,
            "goal_recovery_correction_alpha": float(alpha),
            "goal_recovery_correction_delta_v_mps": proposed.tolist(),
            "goal_recovery_correction_norm_mps": float(np.linalg.norm(proposed)),
            "goal_recovery_radial_delta_v_mps": radial_delta.tolist(),
            "goal_recovery_tangential_delta_v_mps": tangential_delta.tolist(),
            "goal_recovery_applied_delta_v_mps": applied_correction.tolist(),
            "terminal_capture_enabled": int(
                bool(correction_result and correction_result.terminal_capture_enabled)
            ),
            "terminal_capture_active": int(
                bool(correction_result and correction_result.terminal_capture_active)
            ),
            "terminal_capture_radius_m": float(
                correction_result.terminal_capture_radius_m if correction_result else 0.0
            ),
            "terminal_blend": float(
                correction_result.terminal_blend if correction_result else 0.0
            ),
            "base_desired_radial_velocity_mps": float(
                correction_result.base_desired_radial_velocity_mps
                if correction_result else 0.0
            ),
            "terminal_desired_radial_velocity_mps": float(
                correction_result.terminal_desired_radial_velocity_mps
                if correction_result else 0.0
            ),
            "radial_velocity_error_mps": float(
                correction_result.radial_velocity_error_mps
                if correction_result else 0.0
            ),
            "radial_propulsion_delta_v_mps": (
                correction_result.radial_propulsion_delta_v_mps.tolist()
                if correction_result is not None else zero.tolist()
            ),
            "radial_propulsion_scale": float(
                correction_result.radial_propulsion_scale
                if correction_result else 1.0
            ),
            "radial_braking_delta_v_mps": (
                correction_result.radial_braking_delta_v_mps.tolist()
                if correction_result is not None else zero.tolist()
            ),
            "base_tangential_gain": float(
                correction_result.base_tangential_gain if correction_result else 0.0
            ),
            "effective_terminal_tangential_gain": float(
                correction_result.effective_terminal_tangential_gain
                if correction_result else 0.0
            ),
            "terminal_tangential_damping_delta_v_mps": (
                correction_result.terminal_tangential_damping_delta_v_mps.tolist()
                if correction_result is not None else zero.tolist()
            ),
            "terminal_raw_correction_delta_v_mps": (
                correction_result.terminal_raw_correction_delta_v_mps.tolist()
                if correction_result is not None else zero.tolist()
            ),
            "terminal_raw_correction_norm_mps": float(
                correction_result.terminal_raw_correction_norm_mps
                if correction_result else 0.0
            ),
            "terminal_correction_clipped": int(
                bool(correction_result and correction_result.terminal_correction_clipped)
            ),
            "terminal_correction_clip_ratio": float(
                correction_result.terminal_correction_clip_ratio
                if correction_result else 1.0
            ),
            "terminal_v2_enabled": int(
                bool(correction_result and correction_result.terminal_v2_enabled)
            ),
            "terminal_v2_governor_eligible": int(
                bool(correction_result and correction_result.terminal_v2_governor_eligible)
            ),
            "terminal_v2_governor_active": int(
                bool(correction_result and correction_result.terminal_v2_governor_active)
            ),
            "terminal_v2_a_eff_mps2": float(
                correction_result.terminal_v2_a_eff_mps2 if correction_result else 0.0
            ),
            "terminal_v2_stopping_margin_m": float(
                correction_result.terminal_v2_stopping_margin_m if correction_result else 0.0
            ),
            "terminal_v2_distance_to_threshold_m": float(
                correction_result.terminal_v2_distance_to_threshold_m if correction_result else 0.0
            ),
            "terminal_v2_usable_braking_distance_m": float(
                correction_result.terminal_v2_usable_braking_distance_m if correction_result else 0.0
            ),
            "terminal_v2_base_desired_radial_speed_mps": float(
                correction_result.terminal_v2_base_desired_radial_speed_mps if correction_result else 0.0
            ),
            "terminal_v2_envelope_speed_mps": float(
                correction_result.terminal_v2_envelope_speed_mps if correction_result else 0.0
            ),
            "terminal_v2_governed_desired_radial_speed_mps": float(
                correction_result.terminal_v2_governed_desired_radial_speed_mps if correction_result else 0.0
            ),
            "terminal_v2_radial_overspeed_mps": float(
                correction_result.terminal_v2_radial_overspeed_mps if correction_result else 0.0
            ),
            "terminal_v2_braking_required": int(
                bool(correction_result and correction_result.terminal_v2_braking_required)
            ),
            "terminal_v2_inside_goal_hold": int(
                bool(correction_result and correction_result.terminal_v2_inside_goal_hold)
            ),
            "terminal_v2_anti_stall_active": int(
                bool(correction_result and correction_result.terminal_v2_anti_stall_active)
            ),
            "terminal_v21_radial_conflict_enabled": int(bool(
                correction_result
                and correction_result.terminal_v21_radial_conflict_enabled
            )),
            "terminal_v21_radial_conflict_eligible": int(bool(
                correction_result
                and correction_result.terminal_v21_radial_conflict_eligible
            )),
            "terminal_v21_radial_conflict_active": int(bool(
                correction_result
                and correction_result.terminal_v21_radial_conflict_active
            )),
            "terminal_v21_inactive_reason": (
                correction_result.terminal_v21_inactive_reason
                if correction_result else "disabled"
            ),
            "terminal_v21_goal_direction_source": (
                correction_result.terminal_v21_goal_direction_source
                if correction_result else "unavailable"
            ),
            "terminal_v21_actor_radial_delta_v_mps": float(
                correction_result.terminal_v21_actor_radial_delta_v_mps
                if correction_result else 0.0
            ),
            "terminal_v21_actor_opposing_radial_mps": float(
                correction_result.terminal_v21_actor_opposing_radial_mps
                if correction_result else 0.0
            ),
            "terminal_v21_compensation_gain": float(
                correction_result.terminal_v21_compensation_gain
                if correction_result else 0.0
            ),
            "terminal_v21_conflict_compensation_delta_v_mps": (
                correction_result.terminal_v21_conflict_compensation_delta_v_mps.tolist()
                if correction_result else zero.tolist()
            ),
            "terminal_v21_conflict_compensation_norm_mps": float(
                correction_result.terminal_v21_conflict_compensation_norm_mps
                if correction_result else 0.0
            ),
            "terminal_v21_conflict_compensation_radial_mps": float(
                correction_result.terminal_v21_conflict_compensation_radial_mps
                if correction_result else 0.0
            ),
            "terminal_v21_raw_correction_before_conflict_mps": (
                correction_result.terminal_v21_raw_correction_before_conflict_mps.tolist()
                if correction_result else zero.tolist()
            ),
            "terminal_v21_raw_correction_after_conflict_mps": (
                correction_result.terminal_v21_raw_correction_after_conflict_mps.tolist()
                if correction_result else zero.tolist()
            ),
            "terminal_v21_correction_clip_ratio": float(
                correction_result.terminal_v21_correction_clip_ratio
                if correction_result else 1.0
            ),
            "terminal_v21_compensation_post_cap_radial_mps": float(
                correction_result.terminal_v21_compensation_post_cap_radial_mps
                if correction_result else 0.0
            ),
            "terminal_v21_net_radial_pre_transform_mps": float(
                np.dot(
                    corrected_pre_transform,
                    correction_result.goal_direction if correction_result else zero,
                )
            ),
            "terminal_v21_net_radial_post_transform_mps": float(
                np.dot(
                    pre_safety,
                    correction_result.goal_direction if correction_result else zero,
                )
            ),
            "terminal_v21_actor_opposition_cancel_fraction": float(
                correction_result.terminal_v21_compensation_post_cap_radial_mps
                / correction_result.terminal_v21_actor_opposing_radial_mps
                if correction_result
                and correction_result.terminal_v21_actor_opposing_radial_mps > 0.0
                else 0.0
            ),
            # v2.2 fields are copied from the pre-step controller decision.
            # Canonical stable fields below remain the post-environment-step
            # contract extracted from ``next_info``.
            "terminal_v22_enabled": int(bool(_correction_result_value(
                correction_result, "terminal_v22_enabled", False
            ))),
            "terminal_v22_hold_latched_before": int(bool(_correction_result_value(
                correction_result, "terminal_v22_hold_latched_before", False
            ))),
            "terminal_v22_hold_latched": int(bool(_correction_result_value(
                correction_result, "terminal_v22_hold_latched", False
            ))),
            "terminal_v22_hold_enter_radius_m": float(_correction_result_value(
                correction_result, "terminal_v22_hold_enter_radius_m", 0.98
            )),
            "terminal_v22_hold_release_radius_m": float(_correction_result_value(
                correction_result, "terminal_v22_hold_release_radius_m", 1.05
            )),
            "terminal_v22_hold_enter_event": int(bool(_correction_result_value(
                correction_result, "terminal_v22_hold_enter_event", False
            ))),
            "terminal_v22_hold_release_event": int(bool(_correction_result_value(
                correction_result, "terminal_v22_hold_release_event", False
            ))),
            "terminal_v22_hold_mode": str(_correction_result_value(
                correction_result, "terminal_v22_hold_mode", "disabled"
            )),
            "terminal_v22_restorative_active": int(bool(_correction_result_value(
                correction_result, "terminal_v22_restorative_active", False
            ))),
            "terminal_v22_distance_outside_stable_m": float(_correction_result_value(
                correction_result, "terminal_v22_distance_outside_stable_m", 0.0
            )),
            "terminal_v22_restore_fraction": float(_correction_result_value(
                correction_result, "terminal_v22_restore_fraction", 0.0
            )),
            "terminal_v22_restorative_desired_radial_speed_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v22_restorative_desired_radial_speed_mps",
                    0.0,
                )
            ),
            "terminal_v22_restorative_radial_error_mps": float(
                _correction_result_value(
                    correction_result, "terminal_v22_restorative_radial_error_mps", 0.0
                )
            ),
            "terminal_v22_restorative_delta_v_mps": _correction_result_vector(
                correction_result,
                "terminal_v22_restorative_delta_v_mps",
                zero,
            ),
            "terminal_v22_inside_canonical_goal": int(bool(_correction_result_value(
                correction_result, "terminal_v22_inside_canonical_goal", False
            ))),
            "terminal_v23_enabled": bool(_correction_result_value(
                correction_result, "terminal_v23_enabled", False
            )),
            "terminal_v23_hold_target_radius_m": float(_correction_result_value(
                correction_result, "terminal_v23_hold_target_radius_m", 0.96
            )),
            "terminal_v23_guard_active": bool(_correction_result_value(
                correction_result, "terminal_v23_guard_active", False
            )),
            "terminal_v23_guard_fraction": float(_correction_result_value(
                correction_result, "terminal_v23_guard_fraction", 0.0
            )),
            "terminal_v23_desired_radial_speed_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_desired_radial_speed_mps",
                    0.0,
                )
            ),
            "terminal_v23_preboundary_active": bool(_correction_result_value(
                correction_result, "terminal_v23_preboundary_active", False
            )),
            "terminal_v23_postboundary_active": bool(_correction_result_value(
                correction_result, "terminal_v23_postboundary_active", False
            )),
            "terminal_v23_distance_to_target_band_m": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_distance_to_target_band_m",
                    0.0,
                )
            ),
            "terminal_v23_radial_error_mps": float(_correction_result_value(
                correction_result, "terminal_v23_radial_error_mps", 0.0
            )),
            "terminal_v23_positive_radial_error": bool(_correction_result_value(
                correction_result, "terminal_v23_positive_radial_error", False
            )),
            "terminal_v23_radial_propulsion_allowed": bool(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_radial_propulsion_allowed",
                    False,
                )
            ),
            "terminal_v23_radial_propulsion_suppressed": bool(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_radial_propulsion_suppressed",
                    False,
                )
            ),
            "terminal_v23_radial_propulsion_suppression_reason": str(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_radial_propulsion_suppression_reason",
                    "not_v23_guard",
                )
            ),
            "terminal_v23_requested_propulsion_radial_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_requested_propulsion_radial_mps",
                    0.0,
                )
            ),
            "terminal_v23_executed_propulsion_radial_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_executed_propulsion_radial_mps",
                    0.0,
                )
            ),
            "terminal_v23_authority_active": bool(_correction_result_value(
                correction_result, "terminal_v23_authority_active", False
            )),
            "terminal_v23_preboundary_authority_repair_active": bool(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_preboundary_authority_repair_active",
                    False,
                )
            ),
            "terminal_v23_net_tracking_radial_before_v21_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_net_tracking_radial_before_v21_mps",
                    0.0,
                )
            ),
            "terminal_v23_net_tracking_radial_after_v21_mps": float(
                _correction_result_value(
                    correction_result,
                    "terminal_v23_net_tracking_radial_after_v21_mps",
                    0.0,
                )
            ),
            "goal_distance_m": float(relative.goal_distance_m),
            "goal_distance_after_m": next_distance,
            "current_radial_velocity_mps": float(relative.current_radial_velocity_mps),
            "desired_radial_velocity_mps": float(desired_radial_velocity),
            "current_tangential_velocity_mps": float(relative.current_tangential_velocity_mps),
            "corrected_delta_v_pre_transform_mps": corrected_pre_transform.tolist(),
            "corrected_delta_v_pre_safety_mps": pre_safety.tolist(),
            "applied_delta_v_mps": applied.tolist(),
            "motion_boundary_delta_v_mps": (
                np.asarray(
                    nominal_execution.get("delta_v_applied", pre_safety), dtype=float
                ).reshape(3)
                - pre_safety
            ).tolist(),
            "safety_projection_delta_v_mps": (
                applied
                - np.asarray(
                    nominal_execution.get("delta_v_applied", pre_safety), dtype=float
                ).reshape(3)
            ).tolist(),
            "safety_projection_applied": int(bool(execution.get("safety_projected", False))),
            "boundary_projection_applied": int(bool(execution.get("boundary_projected", False))),
            "action_clipped": int(bool(execution.get("action_clipped", False))),
            "action_transform_applied": int(transform_applied),
            "passed_obstacle": int(bool(context["release_latched"])),
            "post_bypass_recovery": int(bool(context["post_bypass_recovery"])),
            "soft_pass_progress": float(context["pass_progress"]),
            "release_state_source": str(context["state_source"]),
            "relevant_obstacle_id": str(context["relevant_obstacle_id"]),
            "relevant_obstacle_clearance_m": float(context["relevant_obstacle_clearance_m"]),
            "along_obstacle_m": float(context["along_obstacle_m"]),
            "obstacle_extent_m": float(context["obstacle_extent_m"]),
            "reward": reward,
            "collision": int(bool(next_info.get("collision", False))),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
            "termination_reason": str(next_info.get("termination_reason", "running")),
            "stable_goal": int(bool(next_info.get("single_uav_goal_stable", False))),
            "post_step_actual_radial_velocity_mps": float(
                actual_relative.current_radial_velocity_mps
            ),
            "post_step_actual_tangential_velocity_mps": float(
                actual_relative.current_tangential_velocity_mps
            ),
            "minimum_clearance_m": float(next_info.get("minimum_clearance", float("nan"))),
            "pre_transform_decomposition_residual_mps": reconstruction_residual,
            "final_execution_decomposition_residual_mps": final_residual,
            **stable_fields,
        }
        if pass_commit_decision is not None:
            row.update(
                {
                    "pass_commit_enabled": int(pass_commit_decision.enabled),
                    "pass_commit_eligible": int(pass_commit_decision.eligible),
                    "pass_commit_triggered": int(pass_commit_decision.triggered),
                    "pass_commit_latched": int(pass_commit_decision.latched),
                    "pass_commit_active": int(pass_commit_decision.active),
                    "pass_commit_inactive_reason": pass_commit_decision.inactive_reason,
                    "pass_commit_margin_m": float(pass_commit_decision.passed_margin_m),
                    "pass_commit_margin_running_best_m": float(
                        pass_commit_decision.running_best_margin_m
                    ),
                    "pass_commit_margin_delta_m": float(pass_commit_decision.margin_delta_m),
                    "pass_commit_margin_regression_run": int(
                        pass_commit_decision.regression_run
                    ),
                    "pass_commit_direction_source": pass_commit_decision.direction_source,
                    "pass_commit_unit": list(pass_commit_decision.direction_unit),
                    "pass_commit_estimated_pass_progress_velocity_before_mps": float(
                        pass_commit_decision.pass_progress_velocity_before_mps
                    ),
                    "pass_commit_estimated_pass_progress_velocity_after_mps": float(
                        pass_commit_decision.pass_progress_velocity_after_mps
                    ),
                    "pass_commit_requested_delta_v_mps": float(
                        pass_commit_decision.requested_delta_v_mps
                    ),
                    "pass_commit_applied_delta_v_mps": float(
                        pass_commit_decision.applied_delta_v_mps
                    ),
                    "pass_commit_requested_vector_mps": list(
                        pass_commit_decision.requested_vector_mps
                    ),
                    "pass_commit_applied_vector_mps": list(
                        pass_commit_decision.applied_vector_mps
                    ),
                    "pass_commit_baseline_candidate_delta_v_mps": actor_raw.tolist(),
                    "pass_commit_combined_pre_transform_delta_v_mps": (
                        corrected_pre_transform.tolist()
                    ),
                    "pass_commit_action_transform_called": int(transform_applied),
                    "pass_commit_action_transform_input_delta_v_mps": (
                        corrected_pre_transform.tolist() if transform_applied else []
                    ),
                    "pass_commit_action_transform_output_delta_v_mps": (
                        pre_safety.tolist() if transform_applied else []
                    ),
                    "pass_commit_safety_input_delta_v_mps": pre_safety.tolist(),
                    "pass_commit_boundary_input_delta_v_mps": pre_safety.tolist(),
                    "pass_commit_final_applied_delta_v_mps": applied.tolist(),
                    "pass_commit_pre_transform_decomposition_residual_mps": (
                        reconstruction_residual
                    ),
                    "pass_commit_post_safety_execution_residual_mps": final_residual,
                    "pass_commit_estimated_lateral_exit_ready": int(
                        pass_commit_decision.estimated_lateral_exit_ready
                    ),
                    "lateral_exit_assist_clearance_increasing_gate": int(
                        assist_decision is not None
                        and assist_decision.clearance_increasing_gate
                    ),
                    "lateral_exit_assist_active": int(
                        assist_decision is not None and assist_decision.active
                    ),
                    "pass_commit_radial_intent_assist_active": int(
                        radial_intent_decision is not None
                        and radial_intent_decision.active
                    ),
                    "pass_commit_goal_recovery_active": int(
                        correction_result is not None and correction_result.active
                    ),
                    "pass_commit_permanently_ended": int(
                        pass_commit_decision.permanently_ended
                    ),
                }
            )
        if assist_decision is not None:
            assist_axis = assist_frame
            if assist_axis is None:
                raise AssertionError("assistance decision requires a frozen horizontal frame")
            row.update(
                {
                    "lateral_exit_assist_enabled": int(assist_decision.enabled),
                    "lateral_exit_assist_eligible": int(assist_decision.eligible),
                    "lateral_exit_assist_active": int(assist_decision.active),
                    "lateral_exit_assist_inactive_reason": assist_decision.inactive_reason,
                    "lateral_exit_assist_encounter_step": (
                        "" if assist_decision.encounter_step is None
                        else int(assist_decision.encounter_step)
                    ),
                    "lateral_exit_assist_frame_source": (
                        "encounter_time_ekf_estimated_goal_ray_projected_to_xy"
                    ),
                    "lateral_exit_assist_horizontal_goal_ray_m": (
                        assist_horizontal_goal_ray.tolist()
                    ),
                    "lateral_exit_assist_lateral_unit": assist_axis.lateral_unit.tolist(),
                    "lateral_exit_assist_frozen_obstacle_id": assist_obstacle_id,
                    "lateral_exit_assist_frozen_obstacle_available": int(
                        assist_decision.frozen_obstacle_gate
                    ),
                    "lateral_exit_assist_frozen_obstacle_gate": int(
                        assist_decision.frozen_obstacle_gate
                    ),
                    "lateral_exit_assist_closest_approach_gate": int(
                        assist_decision.closest_approach_gate
                    ),
                    "lateral_exit_assist_clearance_increasing_gate": int(
                        assist_decision.clearance_increasing_gate
                    ),
                    "lateral_exit_assist_sustained_countersteer_gate": int(
                        assist_decision.sustained_countersteer_gate
                    ),
                    "lateral_exit_assist_initial_lateral_sign": int(
                        assist_decision.initial_lateral_sign
                    ),
                    "lateral_exit_assist_minimum_clearance_m": float(
                        assist_decision.minimum_clearance_m
                    ),
                    "lateral_exit_assist_clearance_increase_run": int(
                        assist_decision.clearance_increase_run
                    ),
                    "lateral_exit_assist_actor_sign_window": list(
                        assist_decision.actor_sign_window
                    ),
                    "lateral_exit_assist_countersteer_count": int(
                        assist_decision.countersteer_count
                    ),
                    "estimated_lateral_velocity_mps": float(
                        assist_decision.estimated_lateral_velocity_mps
                    ),
                    "actor_lateral_delta_v_mps": float(
                        assist_decision.actor_lateral_delta_v_mps
                    ),
                    "requested_assist_lateral_delta_v_mps": float(
                        assist_decision.requested_assist_lateral_delta_v_mps
                    ),
                    "post_cap_assist_lateral_delta_v_mps": float(
                        assist_decision.post_cap_assist_lateral_delta_v_mps
                    ),
                    "requested_assist_delta_v_mps": (
                        assist_decision.requested_assist_delta_v_mps.tolist()
                    ),
                    "applied_assist_delta_v_mps": (
                        assist_decision.applied_assist_delta_v_mps.tolist()
                    ),
                    "combined_pre_transform_delta_v_mps": (
                        assist_decision.combined_pre_transform_delta_v_mps.tolist()
                    ),
                    "final_applied_lateral_delta_v_mps": float(
                        np.dot(applied, assist_axis.lateral_unit)
                    ),
                    "lateral_exit_outcome_actual_lateral_velocity_mps": float(
                        np.dot(actual_post_velocity, assist_axis.lateral_unit)
                    ),
                    "lateral_exit_outcome_actual_lateral_displacement_from_encounter_m": (
                        float(
                            np.dot(
                                actual_post_position
                                - assist_outcome_actual_position_anchor,
                                assist_axis.lateral_unit,
                            )
                        )
                        if assist_outcome_actual_position_anchor is not None
                        else float("nan")
                    ),
                }
            )
        elif lateral_exit_assistance is not None:
            row.update(
                {
                    "lateral_exit_assist_enabled": int(
                        lateral_exit_assistance.enabled
                    ),
                    "lateral_exit_assist_eligible": 0,
                    "lateral_exit_assist_active": 0,
                    "lateral_exit_assist_inactive_reason": (
                        assist_frame_unavailable_reason
                        or "encounter_not_established"
                    ),
                    "lateral_exit_assist_encounter_step": "",
                    "lateral_exit_assist_frame_source": (
                        "encounter_time_ekf_estimated_goal_ray_projected_to_xy"
                    ),
                    "lateral_exit_assist_horizontal_goal_ray_m": (
                        assist_horizontal_goal_ray.tolist()
                    ),
                    "lateral_exit_assist_lateral_unit": [],
                    "lateral_exit_assist_frozen_obstacle_id": assist_obstacle_id,
                    "lateral_exit_assist_frozen_obstacle_available": 0,
                    "lateral_exit_assist_frozen_obstacle_gate": 0,
                    "lateral_exit_assist_closest_approach_gate": 0,
                    "lateral_exit_assist_clearance_increasing_gate": 0,
                    "lateral_exit_assist_sustained_countersteer_gate": 0,
                    "lateral_exit_assist_initial_lateral_sign": 0,
                    "lateral_exit_assist_minimum_clearance_m": float("nan"),
                    "lateral_exit_assist_clearance_increase_run": 0,
                    "lateral_exit_assist_actor_sign_window": [],
                    "lateral_exit_assist_countersteer_count": 0,
                    "estimated_lateral_velocity_mps": float("nan"),
                    "actor_lateral_delta_v_mps": float("nan"),
                    "requested_assist_lateral_delta_v_mps": 0.0,
                    "post_cap_assist_lateral_delta_v_mps": 0.0,
                    "requested_assist_delta_v_mps": [0.0, 0.0, 0.0],
                    "applied_assist_delta_v_mps": [0.0, 0.0, 0.0],
                    "combined_pre_transform_delta_v_mps": actor_raw.tolist(),
                    "final_applied_lateral_delta_v_mps": float("nan"),
                    "lateral_exit_outcome_actual_lateral_velocity_mps": float(
                        "nan"
                    ),
                    "lateral_exit_outcome_actual_lateral_displacement_from_encounter_m": float(
                        "nan"
                    ),
                }
            )
        if radial_intent_decision is not None:
            radial_unit_for_row = (
                radial_intent_frame.radial_unit.tolist()
                if radial_intent_frame is not None
                else []
            )
            row.update(
                {
                    "radial_intent_assist_enabled": int(radial_intent_decision.enabled),
                    "radial_intent_assist_eligible": int(radial_intent_decision.eligible),
                    "radial_intent_assist_active": int(radial_intent_decision.active),
                    "radial_intent_assist_inactive_reason": radial_intent_decision.inactive_reason,
                    "radial_intent_assist_frame_source": radial_intent_decision.frame_source,
                    "radial_intent_assist_radial_unit": radial_unit_for_row,
                    "estimated_lateral_exit_run": int(
                        radial_intent_decision.estimated_lateral_exit_run
                    ),
                    "estimated_lateral_exit_ready": int(
                        radial_intent_decision.estimated_lateral_exit_ready
                    ),
                    "actor_radial_delta_v_mps": float(
                        radial_intent_decision.actor_radial_delta_v_mps
                    ),
                    "actor_goalward_history_5": list(
                        radial_intent_decision.actor_goalward_history_5
                    ),
                    "actor_goalward_count_5": int(
                        radial_intent_decision.actor_goalward_count_5
                    ),
                    "actor_radial_persistence_deficient": int(
                        radial_intent_decision.actor_radial_persistence_deficient
                    ),
                    "target_radial_floor_mps": float(
                        radial_intent_assistance.target_floor_mps
                    ),
                    "requested_radial_assist_mps": float(
                        radial_intent_decision.requested_radial_assist_mps
                    ),
                    "applied_radial_assist_mps": float(
                        radial_intent_decision.applied_radial_assist_mps
                    ),
                    "requested_radial_assist_delta_v_mps": list(
                        radial_intent_decision.requested_radial_assist_delta_v_mps
                    ),
                    "applied_radial_assist_delta_v_mps": list(
                        radial_intent_decision.applied_radial_assist_delta_v_mps
                    ),
                    "radial_intent_final_applied_radial_delta_v_mps": (
                        float(np.dot(applied, radial_intent_frame.radial_unit))
                        if radial_intent_frame is not None
                        else float("nan")
                    ),
                    "radial_intent_goal_recovery_active": int(
                        bool(row.get("goal_recovery_correction_active", 0))
                    ),
                    "radial_intent_lateral_exit_assist_active": int(
                        bool(assist_decision is not None and assist_decision.active)
                    ),
                }
            )
            if radial_intent_decision.enabled and radial_intent_frame is not None:
                row["final_applied_radial_delta_v_mps"] = float(
                    np.dot(applied, radial_intent_frame.radial_unit)
                )
            if radial_intent_decision.active:
                row["combined_pre_transform_delta_v_mps"] = corrected_pre_transform.tolist()
        if signed_observability:
            row.update(signed_step)
        if reconstruction_residual > 1.0e-9 or final_residual > 1.0e-9:
            raise AssertionError("action decomposition residual exceeded tolerance")
        rows.append(row)
        observations, info = next_observations, next_info
        terminated_flag = bool(terminated)
        truncated_flag = bool(truncated)
        if terminated or truncated:
            break
    post = [row for row in rows if int(row["post_bypass_recovery"])]
    correction_norms = [
        float(row["goal_recovery_correction_norm_mps"])
        for row in rows
        if int(row["goal_recovery_correction_active"])
    ]
    actor_norms = [float(row["actor_raw_delta_v_norm_mps"]) for row in rows]
    ratios = [
        float(row["goal_recovery_correction_norm_mps"])
        / max(float(row["actor_raw_delta_v_norm_mps"]), 1.0e-12)
        for row in rows
    ]
    distances = [float(row["goal_distance_after_m"]) for row in rows]
    collision = int(any(int(row["collision"]) for row in rows))
    stable = int(any(int(row["stable_goal"]) for row in rows))
    timeout = int(
        not collision
        and not stable
        and (
            len(rows) >= int(config.max_steps)
            or (rows and str(rows[-1]["termination_reason"]) == "time_limit")
        )
    )
    stable_episode = aggregate_episode_stable_fields(rows)
    terminal_rows = [row for row in rows if int(row["terminal_capture_active"])]
    braking_rows = [
        row for row in terminal_rows
        if float(np.linalg.norm(row["radial_braking_delta_v_mps"])) > 1.0e-12
    ]
    first_terminal = terminal_rows[0] if terminal_rows else None
    first_braking = braking_rows[0] if braking_rows else None
    governor_eligible_rows = [
        row for row in rows if int(row["terminal_v2_governor_eligible"])
    ]
    governor_active_rows = [
        row for row in governor_eligible_rows
        if int(row["terminal_v2_governor_active"])
    ]
    v2_braking_rows = [
        row for row in governor_eligible_rows
        if int(row["terminal_v2_braking_required"])
    ]
    anti_stall_rows = [
        row for row in governor_eligible_rows
        if int(row["terminal_v2_anti_stall_active"])
    ]
    v21_eligible_rows = [
        row for row in rows
        if int(row["terminal_v21_radial_conflict_eligible"])
    ]
    v21_active_rows = [
        row for row in v21_eligible_rows
        if int(row["terminal_v21_radial_conflict_active"])
    ]
    first_v21_conflict = v21_active_rows[0] if v21_active_rows else None
    total_actor_opposition = sum(
        float(row["terminal_v21_actor_opposing_radial_mps"])
        for row in v21_active_rows
    )
    total_post_cap_compensation = sum(
        float(row["terminal_v21_compensation_post_cap_radial_mps"])
        for row in v21_active_rows
    )
    first_governor = governor_active_rows[0] if governor_active_rows else None
    first_v2_braking = v2_braking_rows[0] if v2_braking_rows else None
    radial_overspeeds = [
        float(row["terminal_v2_radial_overspeed_mps"])
        for row in governor_eligible_rows
    ]
    summary = {
        "condition": condition_label or ("on" if correction_enabled else "off"),
        "episode_index": int(item["episode_index"]),
        "episode_id": str(item["episode_id"]),
        "scenario_seed": int(item["scenario_seed"]),
        "noise_seed": reset_seed,
        "scenario_hash": _canonical_hash(scenario),
        "steps": len(rows),
        "episode_return": float(episode_return),
        "passed_obstacle": int(any(int(row["passed_obstacle"]) for row in rows)),
        "post_bypass_recovery": int(bool(post)),
        "entered_8m": int(any(distance <= 8.0 for distance in distances)),
        "entered_5m": int(any(distance <= 5.0 for distance in distances)),
        "entered_3m": int(any(distance <= 3.0 for distance in distances)),
        "entered_1m": int(any(distance <= 1.0 for distance in distances)),
        "stable_goal": stable,
        "timeout": timeout,
        "collision": collision,
        "terminated": int(terminated_flag),
        "truncated": int(truncated_flag),
        "termination_reason": str(rows[-1]["termination_reason"]) if rows else "empty",
        "first_entered_1m_step": "" if first_1m_step is None else int(first_1m_step),
        "stable_goal_step": "" if stable_step is None else int(stable_step),
        "safety_projection_steps": int(sum(int(row["safety_projection_applied"]) for row in rows)),
        "boundary_projection_steps": int(sum(int(row["boundary_projection_applied"]) for row in rows)),
        "minimum_goal_distance_m": min(distances, default=float("nan")),
        "mean_post_recovery_radial_velocity_mps": float(np.mean([float(row["current_radial_velocity_mps"]) for row in post])) if post else float("nan"),
        "mean_post_recovery_tangential_velocity_mps": float(np.mean([float(row["current_tangential_velocity_mps"]) for row in post])) if post else float("nan"),
        "goal_correction_active_steps": int(sum(int(row["goal_recovery_correction_active"]) for row in rows)),
        "terminal_capture_active_steps": len(terminal_rows),
        "first_terminal_capture_step": int(first_terminal["step"]) if first_terminal else "",
        "goal_distance_at_first_terminal_capture_m": float(first_terminal["goal_distance_m"]) if first_terminal else float("nan"),
        "actual_speed_at_first_terminal_capture_mps": float(first_terminal["post_step_actual_speed_mps"]) if first_terminal else float("nan"),
        "first_radial_braking_step": int(first_braking["step"]) if first_braking else "",
        "goal_distance_at_first_radial_braking_m": float(first_braking["goal_distance_m"]) if first_braking else float("nan"),
        "actual_speed_at_first_radial_braking_mps": float(first_braking["post_step_actual_speed_mps"]) if first_braking else float("nan"),
        "mean_terminal_desired_radial_speed": _mean([row["terminal_desired_radial_velocity_mps"] for row in terminal_rows]),
        "mean_radial_braking_norm": _mean([np.linalg.norm(row["radial_braking_delta_v_mps"]) for row in terminal_rows]),
        "mean_radial_propulsion_norm": _mean([np.linalg.norm(row["radial_propulsion_delta_v_mps"]) for row in terminal_rows]),
        "mean_terminal_tangential_damping_norm": _mean([np.linalg.norm(row["terminal_tangential_damping_delta_v_mps"]) for row in terminal_rows]),
        "terminal_correction_saturation_fraction": (
            sum(int(row["terminal_correction_clipped"]) for row in terminal_rows) / len(terminal_rows)
            if terminal_rows else 0.0
        ),
        "terminal_v2_governor_eligible_steps": len(governor_eligible_rows),
        "terminal_v2_governor_active_steps": len(governor_active_rows),
        "terminal_v2_first_governor_limiting_step": (
            int(first_governor["step"]) if first_governor else ""
        ),
        "terminal_v2_goal_distance_at_first_governor_limiting_m": (
            float(first_governor["goal_distance_m"])
            if first_governor else float("nan")
        ),
        "terminal_v2_actual_speed_at_first_governor_limiting_mps": (
            float(first_governor["post_step_actual_speed_mps"])
            if first_governor else float("nan")
        ),
        "terminal_v2_first_radial_braking_step": (
            int(first_v2_braking["step"]) if first_v2_braking else ""
        ),
        "terminal_v2_goal_distance_at_first_radial_braking_m": (
            float(first_v2_braking["goal_distance_m"])
            if first_v2_braking else float("nan")
        ),
        "terminal_v2_actual_speed_at_first_radial_braking_mps": (
            float(first_v2_braking["post_step_actual_speed_mps"])
            if first_v2_braking else float("nan")
        ),
        "terminal_v2_mean_radial_overspeed_mps": _mean(radial_overspeeds),
        "terminal_v2_max_radial_overspeed_mps": max(radial_overspeeds, default=0.0),
        "terminal_v2_governor_limiting_fraction": (
            len(governor_active_rows) / len(governor_eligible_rows)
            if governor_eligible_rows else 0.0
        ),
        "terminal_v2_anti_stall_active_steps": len(anti_stall_rows),
        "terminal_v21_conflict_eligible_steps": len(v21_eligible_rows),
        "terminal_v21_conflict_active_steps": len(v21_active_rows),
        "mean_actor_opposing_radial_mps": (
            float(np.mean([
                float(row["terminal_v21_actor_opposing_radial_mps"])
                for row in v21_active_rows
            ])) if v21_active_rows else 0.0
        ),
        "median_actor_opposing_radial_mps": (
            float(np.median([
                float(row["terminal_v21_actor_opposing_radial_mps"])
                for row in v21_active_rows
            ])) if v21_active_rows else 0.0
        ),
        "mean_conflict_compensation_requested_mps": (
            float(np.mean([
                float(row["terminal_v21_conflict_compensation_radial_mps"])
                for row in v21_active_rows
            ])) if v21_active_rows else 0.0
        ),
        "mean_conflict_compensation_post_cap_mps": (
            float(np.mean([
                float(row["terminal_v21_compensation_post_cap_radial_mps"])
                for row in v21_active_rows
            ])) if v21_active_rows else 0.0
        ),
        "mean_net_radial_pre_transform_mps": (
            float(np.mean([
                float(row["terminal_v21_net_radial_pre_transform_mps"])
                for row in v21_eligible_rows
            ])) if v21_eligible_rows else 0.0
        ),
        "mean_net_radial_post_transform_mps": (
            float(np.mean([
                float(row["terminal_v21_net_radial_post_transform_mps"])
                for row in v21_eligible_rows
            ])) if v21_eligible_rows else 0.0
        ),
        "fraction_opposing_actor_cancelled": (
            total_post_cap_compensation / total_actor_opposition
            if total_actor_opposition > 0.0 else 0.0
        ),
        "terminal_v21_first_conflict_active_step": (
            int(first_v21_conflict["step"]) if first_v21_conflict else ""
        ),
        "terminal_v21_goal_distance_at_first_conflict_active_m": (
            float(first_v21_conflict["goal_distance_m"])
            if first_v21_conflict else float("nan")
        ),
        "terminal_v21_correction_saturation_fraction": (
            sum(
                float(row["terminal_v21_correction_clip_ratio"]) < 1.0
                for row in v21_eligible_rows
            ) / len(v21_eligible_rows)
            if v21_eligible_rows else 0.0
        ),
        "mean_goal_correction_norm_mps": float(np.mean(correction_norms)) if correction_norms else 0.0,
        "max_goal_correction_norm_mps": max(correction_norms, default=0.0),
        "mean_actor_raw_action_norm_mps": float(np.mean(actor_norms)) if actor_norms else 0.0,
        "mean_correction_to_actor_norm_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "goal_threshold_m": float(config.single_uav_goal_threshold_m),
        "stable_hold_steps": int(config.single_uav_stable_hold_steps),
        "stable_speed_threshold_mps": float(config.single_uav_goal_speed_threshold_mps),
        "stable_log_schema_version": STABLE_LOG_SCHEMA_VERSION,
        "max_steps": int(config.max_steps),
        **stable_episode,
        **EVALUATION_ONLY_FLAGS,
    }
    summary.update(_aggregate_v22_episode_fields(rows))
    summary.update(_aggregate_v23_episode_fields(rows))
    if signed_observability:
        signed_rows = [row for row in rows if "actor_raw_lateral_delta_v_mps" in row]
        encounter_steps = [
            int(row["signed_encounter_step"])
            for row in signed_rows
            if str(row.get("signed_encounter_step", "")) not in {"", "None"}
        ]
        actor_values = [float(row["actor_raw_lateral_delta_v_mps"]) for row in signed_rows]
        actual_values = [float(row["actual_lateral_velocity_mps"]) for row in signed_rows]
        summary.update(
            {
                "signed_bypass_observability_logged": True,
                "signed_observability_schema_version": SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION,
                "signed_encounter_step": min(encounter_steps) if encounter_steps else "",
                "signed_actor_lateral_sign_changes": sign_change_count(
                    actor_values, deadband=DEFAULT_SIGN_DEADBAND_MPS
                ),
                "signed_actual_lateral_sign_changes": sign_change_count(
                    actual_values, deadband=DEFAULT_SIGN_DEADBAND_MPS
                ),
                "signed_actor_actual_lateral_side_agreement_fraction": (
                    float(
                        np.mean(
                            [
                                float(row["actor_actual_lateral_side_agreement_fraction"])
                                for row in signed_rows
                                if np.isfinite(
                                    float(row.get("actor_actual_lateral_side_agreement_fraction", float("nan")))
                                )
                            ]
                        )
                    )
                    if any(
                        np.isfinite(float(row.get("actor_actual_lateral_side_agreement_fraction", float("nan"))))
                        for row in signed_rows
                    )
                    else float("nan")
                ),
                "signed_actor_max_same_side_run_length": max(
                    [int(row.get("actor_same_side_run_length", 0)) for row in signed_rows],
                    default=0,
                ),
                "signed_actual_max_same_side_run_length": max(
                    [int(row.get("actual_same_side_run_length", 0)) for row in signed_rows],
                    default=0,
                ),
                "signed_clearance_increasing_steps": sum(
                    int(bool(row.get("clearance_increasing", False))) for row in signed_rows
                ),
                "signed_past_closest_approach_steps": sum(
                    int(bool(row.get("past_closest_approach", False))) for row in signed_rows
                ),
            }
        )
    if lateral_exit_assistance is not None:
        assist_rows = [
            row for row in rows if "lateral_exit_assist_enabled" in row
        ]
        eligible_rows = [
            row for row in assist_rows
            if int(row.get("lateral_exit_assist_eligible", 0))
        ]
        active_rows = [
            row for row in assist_rows
            if int(row.get("lateral_exit_assist_active", 0))
        ]
        summary.update(
            {
                "lateral_exit_assist_enabled": int(
                    bool(lateral_exit_assistance_enabled)
                ),
                "lateral_exit_assist_cap_mps": LATERAL_EXIT_ASSIST_CAP_MPS,
                "lateral_exit_assist_first_eligible_step": (
                    int(eligible_rows[0]["step"]) if eligible_rows else ""
                ),
                "lateral_exit_assist_first_active_step": (
                    int(active_rows[0]["step"]) if active_rows else ""
                ),
                "lateral_exit_assist_active_steps": len(active_rows),
                "lateral_exit_assist_cumulative_abs_delta_v_mps": float(
                    sum(
                        abs(float(row["post_cap_assist_lateral_delta_v_mps"]))
                        for row in active_rows
                    )
                ),
                "lateral_exit_assist_max_abs_delta_v_mps": max(
                    (
                        abs(float(row["post_cap_assist_lateral_delta_v_mps"]))
                        for row in active_rows
                    ),
                    default=0.0,
                ),
                "lateral_exit_assist_goal_recovery_overlap_steps": int(
                    sum(
                        int(row.get("goal_recovery_correction_active", 0))
                        and int(row.get("lateral_exit_assist_active", 0))
                        for row in assist_rows
                    )
                ),
            }
        )
    if rows and "pass_commit_active" in rows[0]:
        pass_rows = [row for row in rows if "pass_commit_active" in row]
        first_trigger = next(
            (row for row in pass_rows if int(row["pass_commit_triggered"])), None
        )
        first_active = next(
            (row for row in pass_rows if int(row["pass_commit_active"])), None
        )
        first_exit = next(
            (row for row in pass_rows if int(row["pass_commit_estimated_lateral_exit_ready"])), None
        )
        first_pass = next(
            (
                row for row in pass_rows
                if int(row.get("passed_obstacle", 0))
                or bool(row.get("passed_condition_now", False))
            ),
            None,
        )
        pre_pass_rows = [
            row for row in pass_rows
            if first_pass is None or int(row["step"]) <= int(first_pass["step"])
        ]
        finite_margins = [
            float(row["pass_commit_margin_m"])
            for row in pre_pass_rows
            if math.isfinite(float(row["pass_commit_margin_m"]))
        ]
        regression_rows = [
            row for row in pass_rows
            if math.isfinite(float(row["pass_commit_margin_delta_m"]))
            and float(row["pass_commit_margin_delta_m"]) < -1.0e-8
        ]
        summary.update(
            {
                "pass_commit_enabled": bool(any(int(row["pass_commit_enabled"]) for row in pass_rows)),
                "pass_commit_triggered": bool(first_trigger),
                "pass_commit_latched": bool(any(int(row["pass_commit_latched"]) for row in pass_rows)),
                "first_pass_step": int(first_pass["step"]) if first_pass else "",
                "first_pass_commit_trigger_step": int(first_trigger["step"]) if first_trigger else "",
                "first_pass_commit_active_step": int(first_active["step"]) if first_active else "",
                "pass_commit_active_steps": int(sum(int(row["pass_commit_active"]) for row in pass_rows)),
                "best_passed_margin_m": max(finite_margins, default=float("nan")),
                "final_pre_pass_passed_margin_m": finite_margins[-1] if finite_margins else float("nan"),
                "pass_margin_regression_count": len(regression_rows),
                "longest_pass_margin_regression_run": max(
                    (int(row["pass_commit_margin_regression_run"]) for row in pass_rows),
                    default=0,
                ),
                "time_lateral_exit_to_pass_steps": (
                    int(first_pass["step"]) - int(first_exit["step"])
                    if first_pass and first_exit else ""
                ),
                "time_commit_to_pass_steps": (
                    int(first_pass["step"]) - int(first_active["step"])
                    if first_pass and first_active else ""
                ),
            }
        )
    return summary, rows


def _mean(values: Sequence[Any]) -> float:
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            finite.append(number)
    return float(np.mean(finite)) if finite else float("nan")


def _finite_max(values: Sequence[Any]) -> float:
    """Return the maximum finite value, or NaN when no value is valid."""

    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            finite.append(number)
    return max(finite, default=float("nan"))


def _episode_summary_values(
    summaries: Sequence[Mapping[str, Any]],
    field: str,
    *aliases: str,
) -> list[Any]:
    """Read one episode field, accepting explicit unit aliases as fallback."""

    values: list[Any] = []
    for summary in summaries:
        if field in summary:
            values.append(summary[field])
            continue
        for alias in aliases:
            if alias in summary:
                values.append(summary[alias])
                break
        else:
            values.append(None)
    return values


def _row_flag(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _v22_inside_pre_step(row: Mapping[str, Any]) -> bool:
    """Return the v2.2 inside/outside state from pre-step estimated data."""

    if "terminal_v22_inside_canonical_goal" in row:
        return _row_flag(row, "terminal_v22_inside_canonical_goal")
    if "terminal_v22_inside_goal_canonical" in row:
        return _row_flag(row, "terminal_v22_inside_goal_canonical")
    if "terminal_v22_inside_canonical" in row:
        return _row_flag(row, "terminal_v22_inside_canonical")
    try:
        distance = float(row.get("goal_distance_m", float("nan")))
        threshold = float(row.get("goal_threshold_m", 1.0))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(distance) and np.isfinite(threshold) and distance <= threshold)


def _aggregate_v22_episode_fields(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute v2.2 hold/restoration metrics from already logged rows.

    Hold fields are decisions made before ``environment.step``.  The actual
    speed at hold entry is intentionally read from the same row's canonical
    post-step field and is therefore documented as post-step timing.
    """

    ordered = sorted(rows, key=lambda row: int(row.get("step", 0)))
    entries: list[Mapping[str, Any]] = []
    releases: list[Mapping[str, Any]] = []
    previous_latched = False
    for row in ordered:
        latched_before = _row_flag(row, "terminal_v22_hold_latched_before", previous_latched)
        latched_after = _row_flag(row, "terminal_v22_hold_latched", latched_before)
        entered = _row_flag(row, "terminal_v22_hold_enter_event") or (
            latched_after and not latched_before
        )
        released = _row_flag(row, "terminal_v22_hold_release_event") or (
            latched_before and not latched_after
        )
        if entered:
            entries.append(row)
        if released:
            releases.append(row)
        previous_latched = latched_after

    first_entry = entries[0] if entries else None
    first_entry_step = int(first_entry.get("step", 0)) if first_entry else ""
    first_entry_index = ordered.index(first_entry) if first_entry in ordered else -1
    reentry_count = max(len(entries) - 1, 0)
    # A release-boundary decision starts with the latch still active.  Include
    # both pre- and post-decision latch states when measuring its distance.
    latched_rows = [
        row for row in ordered
        if _row_flag(row, "terminal_v22_hold_latched")
        or _row_flag(row, "terminal_v22_hold_latched_before")
    ]
    restorative_rows = [
        row for row in ordered if _row_flag(row, "terminal_v22_restorative_active")
    ]
    desired = [
        float(row.get("terminal_v22_restorative_desired_radial_speed_mps", 0.0))
        for row in restorative_rows
    ]
    desired = [value for value in desired if np.isfinite(value)]

    crossings = 0
    if first_entry_index >= 0:
        hold_tail = ordered[first_entry_index:]
        previous_inside: bool | None = None
        for index, row in enumerate(hold_tail):
            inside = _v22_inside_pre_step(row)
            if previous_inside is True and index > 0 and not inside:
                crossings += 1
            previous_inside = inside

    counter_resets = 0
    distance_only_resets = speed_only_resets = both_resets = 0
    previous_counter: int | None = None
    previous_distance_ok: bool | None = None
    previous_speed_ok: bool | None = None
    for row in ordered:
        try:
            counter = int(float(row.get("stable_hold_counter", 0)))
        except (TypeError, ValueError):
            counter = 0
        distance_ok = _row_flag(row, "goal_distance_ok")
        speed_ok = _row_flag(row, "goal_speed_ok")
        if previous_counter is not None and previous_counter > 0 and counter == 0:
            counter_resets += 1
            if not distance_ok and speed_ok:
                distance_only_resets += 1
            elif distance_ok and not speed_ok:
                speed_only_resets += 1
            elif not distance_ok and not speed_ok:
                both_resets += 1
        previous_counter = counter
        previous_distance_ok = distance_ok
        previous_speed_ok = speed_ok

    distances = []
    for row in latched_rows:
        try:
            value = float(row.get("goal_distance_m", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            distances.append(value)
    mean_restorative_speed = _mean(desired)
    max_restorative_speed = max(desired, default=0.0)
    return {
        "v22_hold_entered": int(bool(first_entry)),
        "first_hold_enter_step": first_entry_step,
        "goal_distance_at_hold_enter": (
            float(first_entry.get("goal_distance_m", float("nan")))
            if first_entry else float("nan")
        ),
        "actual_speed_at_hold_enter": (
            float(first_entry.get("post_step_actual_speed_mps", float("nan")))
            if first_entry else float("nan")
        ),
        "actual_speed_at_hold_enter_timing": (
            "post_step_actual_speed_of_same_logged_decision_step"
            if first_entry else "not_entered"
        ),
        "hold_release_count": len(releases),
        "hold_reentry_count": reentry_count,
        "restorative_active_steps": len(restorative_rows),
        # Preserve both the requirement-level schema and explicit-unit aliases.
        "mean_restorative_desired_radial_speed": mean_restorative_speed,
        "max_restorative_desired_radial_speed": max_restorative_speed,
        "mean_restorative_desired_radial_speed_mps": mean_restorative_speed,
        "max_restorative_desired_radial_speed_mps": max_restorative_speed,
        "maximum_distance_while_hold_latched": max(distances, default=float("nan")),
        "inside_outside_crossings_after_hold": crossings,
        "stable_counter_reset_count": counter_resets,
        "distance_only_reset_count": distance_only_resets,
        "speed_only_reset_count": speed_only_resets,
        "both_reset_count": both_resets,
        "stable_distance_reset_count": distance_only_resets + both_resets,
        "stable_speed_reset_count": speed_only_resets + both_resets,
        "distance_reset_count": distance_only_resets + both_resets,
        "speed_reset_count": speed_only_resets + both_resets,
    }


def _augment_v22_episode_summary(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an episode summary augmented with pure v2.2 log metrics."""

    result = dict(summary)
    result.update(_aggregate_v22_episode_fields(rows))
    return result


# Descriptive alias for tests and downstream offline analysis.
_aggregate_terminal_capture_v22_episode = _aggregate_v22_episode_fields


def _aggregate_v23_episode_fields(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute v2.3 target-radius guard metrics from logged decision rows."""

    ordered = sorted(rows, key=lambda row: int(row.get("step", 0)))
    active = [row for row in ordered if _row_flag(row, "terminal_v23_guard_active")]
    preboundary = [
        row for row in ordered if _row_flag(row, "terminal_v23_preboundary_active")
    ]
    postboundary = [
        row for row in ordered if _row_flag(row, "terminal_v23_postboundary_active")
    ]
    positive_error = [
        row for row in ordered
        if _row_flag(row, "terminal_v23_positive_radial_error")
    ]
    propulsion_allowed = [
        row for row in ordered
        if _row_flag(row, "terminal_v23_radial_propulsion_allowed")
    ]
    propulsion_suppressed = [
        row for row in ordered
        if _row_flag(row, "terminal_v23_radial_propulsion_suppressed")
    ]
    authority_active = [
        row for row in ordered if _row_flag(row, "terminal_v23_authority_active")
    ]
    preboundary_repair_active = [
        row for row in ordered
        if _row_flag(row, "terminal_v23_preboundary_authority_repair_active")
    ]
    desired: list[float] = []
    for row in active:
        try:
            value = float(row.get("terminal_v23_desired_radial_speed_mps", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0.0:
            desired.append(value)
    first = active[0] if active else None
    maximum = max(desired, default=0.0)
    requested = [
        float(row.get("terminal_v23_requested_propulsion_radial_mps", 0.0))
        for row in propulsion_allowed
    ]
    executed = [
        float(row.get("terminal_v23_executed_propulsion_radial_mps", 0.0))
        for row in propulsion_allowed
    ]
    return {
        "v23_guard_active_steps": len(active),
        "v23_preboundary_guard_steps": len(preboundary),
        "v23_postboundary_restore_steps": len(postboundary),
        "mean_v23_desired_radial_speed": _mean(desired),
        "global_max_v23_desired_radial_speed": maximum,
        "first_v23_guard_step": int(first["step"]) if first else "",
        "distance_at_first_v23_guard": (
            float(first.get("goal_distance_m", float("nan")))
            if first else float("nan")
        ),
        "v23_positive_radial_error_steps": len(positive_error),
        "v23_radial_propulsion_allowed_steps": len(propulsion_allowed),
        "v23_radial_propulsion_suppressed_steps": len(propulsion_suppressed),
        "v23_authority_active_steps": len(authority_active),
        "v23_preboundary_authority_repair_active_steps": len(
            preboundary_repair_active
        ),
        "mean_v23_requested_propulsion_radial_mps": _mean(requested),
        "mean_v23_executed_propulsion_radial_mps": _mean(executed),
    }


_aggregate_terminal_capture_v23_episode = _aggregate_v23_episode_fields


def _augment_v23_episode_summary(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an episode summary augmented with pure v2.3 log metrics."""

    result = dict(summary)
    result.update(_aggregate_v23_episode_fields(rows))
    return result


COMPARISON_METRICS = (
    "passed_obstacle",
    "post_bypass_recovery",
    "entered_8m",
    "entered_5m",
    "entered_3m",
    "entered_1m",
    "stable_goal",
    "boundary_projection_steps",
    "safety_projection_steps",
    "mean_post_recovery_radial_velocity_mps",
    "mean_post_recovery_tangential_velocity_mps",
    "mean_goal_correction_norm_mps",
)


def _aggregate_episode_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count_metrics = {
        "passed_obstacle",
        "post_bypass_recovery",
        "entered_8m",
        "entered_5m",
        "entered_3m",
        "entered_1m",
        "stable_goal",
        "timeout",
        "collision",
        "boundary_projection_steps",
        "safety_projection_steps",
        "goal_correction_active_steps",
        "terminal_capture_active_steps",
        "terminal_v2_governor_eligible_steps",
        "terminal_v2_governor_active_steps",
        "terminal_v2_anti_stall_active_steps",
        "terminal_v21_conflict_eligible_steps",
        "terminal_v21_conflict_active_steps",
        "v22_hold_entered",
        "hold_release_count",
        "hold_reentry_count",
        "restorative_active_steps",
        "inside_outside_crossings_after_hold",
        "stable_counter_reset_count",
        "stable_distance_reset_count",
        "stable_speed_reset_count",
        "distance_reset_count",
        "speed_reset_count",
        "distance_only_reset_count",
        "speed_only_reset_count",
        "both_reset_count",
        "v23_guard_active_steps",
        "v23_preboundary_guard_steps",
        "v23_postboundary_restore_steps",
    }
    output: dict[str, Any] = {"episodes": len(summaries)}
    for metric in count_metrics:
        output[metric] = int(sum(int(row.get(metric, 0) or 0) for row in summaries))
    for metric in (
        "minimum_goal_distance_m",
        "mean_post_recovery_radial_velocity_mps",
        "mean_post_recovery_tangential_velocity_mps",
        "mean_goal_correction_norm_mps",
        "max_goal_correction_norm_mps",
        "mean_actor_raw_action_norm_mps",
        "mean_correction_to_actor_norm_ratio",
        "mean_terminal_desired_radial_speed",
        "mean_radial_braking_norm",
        "mean_radial_propulsion_norm",
        "mean_terminal_tangential_damping_norm",
        "terminal_correction_saturation_fraction",
        "actual_speed_at_first_terminal_capture_mps",
        "actual_speed_at_first_radial_braking_mps",
        "terminal_v2_actual_speed_at_first_governor_limiting_mps",
        "terminal_v2_actual_speed_at_first_radial_braking_mps",
        "terminal_v2_mean_radial_overspeed_mps",
        "terminal_v2_max_radial_overspeed_mps",
        "terminal_v2_governor_limiting_fraction",
        "mean_actor_opposing_radial_mps",
        "median_actor_opposing_radial_mps",
        "mean_conflict_compensation_requested_mps",
        "mean_conflict_compensation_post_cap_mps",
        "mean_net_radial_pre_transform_mps",
        "mean_net_radial_post_transform_mps",
        "fraction_opposing_actor_cancelled",
        "terminal_v21_correction_saturation_fraction",
        "goal_distance_at_hold_enter",
        "actual_speed_at_hold_enter",
        "mean_v23_desired_radial_speed",
        "distance_at_first_v23_guard",
    ):
        output[metric] = _mean([row.get(metric) for row in summaries])

    v23_maxima = _episode_summary_values(
        summaries, "global_max_v23_desired_radial_speed"
    )
    output["global_max_v23_desired_radial_speed"] = _finite_max(v23_maxima)
    first_guard_steps = []
    for summary in summaries:
        try:
            value = int(summary.get("first_v23_guard_step", ""))
        except (TypeError, ValueError):
            continue
        if value >= 0:
            first_guard_steps.append(value)
    output["first_v23_guard_step"] = min(first_guard_steps, default="")

    # Keep episode-level names in the episode records, but make the
    # evaluation-level reductions explicit.  In particular, an aggregate
    # ``max_*`` field must not silently mean the mean of episode maxima.
    episode_mean_restorative_speed = _episode_summary_values(
        summaries,
        "mean_restorative_desired_radial_speed",
        "mean_restorative_desired_radial_speed_mps",
    )
    episode_max_restorative_speed = _episode_summary_values(
        summaries,
        "max_restorative_desired_radial_speed",
        "max_restorative_desired_radial_speed_mps",
    )
    episode_max_hold_distance = _episode_summary_values(
        summaries,
        "maximum_distance_while_hold_latched",
    )
    mean_restorative_speed = _mean(episode_mean_restorative_speed)
    mean_episode_max_restorative_speed = _mean(episode_max_restorative_speed)
    global_max_restorative_speed = _finite_max(episode_max_restorative_speed)
    mean_episode_max_hold_distance = _mean(episode_max_hold_distance)
    global_max_hold_distance = _finite_max(episode_max_hold_distance)
    output.update(
        {
            "mean_restorative_desired_radial_speed": mean_restorative_speed,
            "mean_restorative_desired_radial_speed_mps": mean_restorative_speed,
            "global_max_restorative_desired_radial_speed": global_max_restorative_speed,
            "global_max_restorative_desired_radial_speed_mps": global_max_restorative_speed,
            "mean_episode_max_restorative_desired_radial_speed": mean_episode_max_restorative_speed,
            "mean_episode_max_restorative_desired_radial_speed_mps": mean_episode_max_restorative_speed,
            "global_maximum_distance_while_hold_latched": global_max_hold_distance,
            "mean_episode_maximum_distance_while_hold_latched": mean_episode_max_hold_distance,
        }
    )
    return {**output, **EVALUATION_ONLY_FLAGS}


def _comparison_rows(
    off_summary: Mapping[str, Any],
    on_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in COMPARISON_METRICS:
        off = float(off_summary.get(metric, 0.0) or 0.0)
        on = float(on_summary.get(metric, 0.0) or 0.0)
        rows.append({"metric": metric, "OFF": off, "ON": on, "delta": on - off})
    return rows


def _paired_episode_rows(
    off: Sequence[Mapping[str, Any]], on: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    _assert_paired_identity(off, on)
    rows = []
    for off_row, on_row in zip(off, on):
        row: dict[str, Any] = {
            "episode_index": int(off_row["episode_index"]),
            "episode_id": str(off_row["episode_id"]),
            "noise_seed": int(off_row["noise_seed"]),
        }
        for metric in COMPARISON_METRICS:
            off_value = float(off_row.get(metric, 0.0) or 0.0)
            on_value = float(on_row.get(metric, 0.0) or 0.0)
            row[f"{metric}_OFF"] = off_value
            row[f"{metric}_ON"] = on_value
            row[f"{metric}_delta"] = on_value - off_value
        rows.append(row)
    return rows


def _write_condition_outputs(
    *,
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    condition: str,
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    _write_rows_csv(logs_root / "steps.csv", step_rows)
    _write_rows_csv(logs_root / "episodes.csv", episode_rows)
    payload = {
        "stage": STAGE_NAME,
        "condition": condition,
        "aggregate": dict(aggregate),
        "provenance": dict(provenance),
        **EVALUATION_ONLY_FLAGS,
    }
    _write_json(logs_root / "summary.json", _jsonable(payload))
    _write_json(data_root / "summary.json", _jsonable(payload))
    _write_json(
        plots_root / "manifest.json",
        _jsonable({"plots_generated": False, "evaluation_only": True}),
    )


_SIGNED_TRACE_BEHAVIOR_KEYS = (
    "episode_index",
    "episode_id",
    "step",
    "goal_distance_m",
    "goal_distance_after_m",
    "current_radial_velocity_mps",
    "current_tangential_velocity_mps",
    "actor_raw_delta_v_mps",
    "corrected_delta_v_pre_transform_mps",
    "corrected_delta_v_pre_safety_mps",
    "applied_delta_v_mps",
    "motion_boundary_delta_v_mps",
    "safety_projection_delta_v_mps",
    "passed_obstacle",
    "post_bypass_recovery",
    "safety_projection_applied",
    "boundary_projection_applied",
    "action_clipped",
    "collision",
    "terminated",
    "truncated",
    "termination_reason",
    "stable_goal",
)


def _find_output_artifact(root: Path, name: str) -> Path:
    for candidate in (Path(root) / name, Path(root) / "on" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(Path(root) / name)


def _normalize_trace_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in {
        "episode_index", "step", "passed_obstacle", "post_bypass_recovery",
        "safety_projection_applied", "boundary_projection_applied", "action_clipped",
        "collision", "terminated", "truncated", "stable_goal",
    }:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if key.endswith("_mps") or key.endswith("_m") or key in {
        "goal_distance_m", "goal_distance_after_m", "current_radial_velocity_mps",
        "current_tangential_velocity_mps",
    }:
        try:
            number = float(value)
            return None if not np.isfinite(number) else round(number, 12)
        except (TypeError, ValueError):
            pass
    if key.endswith("_mps") and isinstance(value, str) and value.startswith("["):
        try:
            return [round(float(item), 12) for item in json.loads(value)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    if isinstance(value, (np.ndarray, tuple, list)):
        return [
            round(float(item), 12) if isinstance(item, (float, int, np.floating, np.integer)) else item
            for item in value
        ]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _behavior_trace_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: _normalize_trace_value(key, row.get(key))
            for key in _SIGNED_TRACE_BEHAVIOR_KEYS
        }
        for row in rows
    ]


def _signed_episode_diagnostics(
    episode_summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actor = [float(row["actor_raw_lateral_delta_v_mps"]) for row in rows if "actor_raw_lateral_delta_v_mps" in row]
    actual = [float(row["actual_lateral_velocity_mps"]) for row in rows if "actual_lateral_velocity_mps" in row]
    radial = [float(row["actual_radial_velocity_mps"]) for row in rows if "actual_radial_velocity_mps" in row]
    comparable = [
        (sign_value(a, deadband=DEFAULT_SIGN_DEADBAND_MPS), sign_value(b, deadband=DEFAULT_SIGN_DEADBAND_MPS))
        for a, b in zip(actor, actual)
    ]
    comparable = [(a, b) for a, b in comparable if a and b]
    agreement = float(sum(a == b for a, b in comparable) / len(comparable)) if comparable else float("nan")
    first_side = str(rows[0].get("signed_obstacle_side", "AMBIGUOUS")) if rows else "AMBIGUOUS"
    assessable = bool(
        rows
        and any(bool(row.get("signed_axis_frozen", False)) for row in rows)
        and any(np.isfinite(value) and abs(value) > DEFAULT_SIGN_DEADBAND_M for value in actor)
    )
    label = classify_signed_failure(
        assessable=assessable,
        initial_obstacle_side=first_side,
        recommended_side=str(rows[0].get("recommended_geometric_bypass_side", "AMBIGUOUS")) if rows else "AMBIGUOUS",
        passed=bool(episode_summary.get("passed_obstacle", 0)),
        lateral_sign_changes=sign_change_count(actor, deadband=DEFAULT_SIGN_DEADBAND_MPS),
        actor_motion_agreement=agreement,
        radial_progress_mean=float(np.mean(radial)) if radial else float("nan"),
        lateral_valid=bool(actor),
    )
    return {
        "episode_index": int(episode_summary.get("episode_index", 0)),
        "episode_id": str(episode_summary.get("episode_id", "")),
        "passed_obstacle": int(episode_summary.get("passed_obstacle", 0)),
        "post_bypass_recovery": int(episode_summary.get("post_bypass_recovery", 0)),
        "signed_failure_label": label,
        "signed_assessable": int(assessable),
        "signed_initial_obstacle_side": first_side,
        "signed_actor_motion_agreement": agreement,
        "signed_actor_lateral_sign_changes": sign_change_count(actor, deadband=DEFAULT_SIGN_DEADBAND_MPS),
        "signed_actual_lateral_sign_changes": sign_change_count(actual, deadband=DEFAULT_SIGN_DEADBAND_MPS),
        "signed_mean_actual_radial_velocity_mps": float(np.mean(radial)) if radial else float("nan"),
        "signed_encounter_step": episode_summary.get("signed_encounter_step", ""),
    }


def _write_signed_bypass_outputs(
    *,
    output_root: Path,
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    baseline_evaluation: Mapping[str, Any],
    baseline_step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the independent signed-observability artifact family."""

    root = Path(output_root)
    rows_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        rows_by_episode.setdefault(str(row.get("episode_id", "")), []).append(row)
    focus = [
        _signed_episode_diagnostics(summary, rows_by_episode.get(str(summary.get("episode_id", "")), []))
        for summary in episode_rows
        if not int(summary.get("passed_obstacle", 0)) or not int(summary.get("post_bypass_recovery", 0))
    ]
    transitions = []
    for row in step_rows:
        if int(row.get("actor_lateral_sign_changes", 0)) or int(row.get("actual_lateral_sign_changes", 0)):
            transitions.append(
                {
                    "episode_index": row.get("episode_index"),
                    "episode_id": row.get("episode_id"),
                    "step": row.get("step"),
                    "actor_lateral_sign": row.get("actor_lateral_sign"),
                    "actual_lateral_sign": row.get("actual_lateral_sign"),
                    "actor_lateral_sign_changes": row.get("actor_lateral_sign_changes"),
                    "actual_lateral_sign_changes": row.get("actual_lateral_sign_changes"),
                    "actor_same_side_run_length": row.get("actor_same_side_run_length"),
                    "actual_same_side_run_length": row.get("actual_same_side_run_length"),
                    "actor_actual_lateral_side_agree_now": row.get("actor_actual_lateral_side_agree_now"),
                    "actor_actual_lateral_side_agreement_fraction": row.get("actor_actual_lateral_side_agreement_fraction"),
                    "clearance_increasing": row.get("clearance_increasing"),
                    "past_closest_approach": row.get("past_closest_approach"),
                }
            )
    baseline_results = baseline_evaluation.get("results", {}).get("on", {})
    baseline_episodes = baseline_results.get("episodes", [])
    behavior_fields = (
        "passed_obstacle", "post_bypass_recovery", "entered_1m", "entered_3m", "entered_5m",
        "entered_8m", "stable_goal", "collision", "safety_projection_steps",
        "boundary_projection_steps", "termination_reason", "timeout",
    )
    def _identity_json(value: Any) -> str:
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))

    episode_identity_match = bool(
        evaluation.get("scenario_manifest_hash") == baseline_evaluation.get("scenario_manifest_hash")
        and evaluation.get("checkpoint", {}).get("sha256") == baseline_evaluation.get("checkpoint", {}).get("sha256")
        and _identity_json(evaluation.get("ordered_pool_identity")) == _identity_json(baseline_evaluation.get("ordered_pool_identity"))
        and len(episode_rows) == len(baseline_episodes)
    )
    behavior_match = episode_identity_match
    behavior_differences: list[dict[str, Any]] = []
    for current, baseline in zip(episode_rows, baseline_episodes):
        for field in behavior_fields:
            if current.get(field) != baseline.get(field):
                behavior_match = False
                behavior_differences.append({"episode_id": current.get("episode_id"), "field": field, "current": current.get(field), "baseline": baseline.get(field)})
    current_trace = _behavior_trace_rows(step_rows)
    baseline_trace = _behavior_trace_rows(baseline_step_rows)
    trajectory_match = current_trace == baseline_trace
    if not trajectory_match:
        raise RuntimeError("LOGGING_PERTURBED_TRAJECTORY")
    current_max_steps = {
        int(row.get("max_steps", 0))
        for row in episode_rows
        if str(row.get("max_steps", "")) not in {"", "None"}
    }
    baseline_max_steps = {
        int(float(row.get("max_steps", 0)))
        for row in baseline_episodes
        if str(row.get("max_steps", "")) not in {"", "None"}
    }
    identity_lock = {
        "same_checkpoint_sha256": bool(evaluation.get("checkpoint", {}).get("sha256") == baseline_evaluation.get("checkpoint", {}).get("sha256")),
        "same_policy_hash": bool(evaluation.get("checkpoint", {}).get("policy_hash_after") == baseline_evaluation.get("checkpoint", {}).get("policy_hash_after")),
        "same_observation_dimension": bool(evaluation.get("checkpoint", {}).get("observation_dimension") == baseline_evaluation.get("checkpoint", {}).get("observation_dimension")),
        "same_ordered_identity": bool(_identity_json(evaluation.get("ordered_pool_identity")) == _identity_json(baseline_evaluation.get("ordered_pool_identity"))),
        "same_scenario_manifest": bool(evaluation.get("scenario_manifest_hash") == baseline_evaluation.get("scenario_manifest_hash")),
        "same_max_steps": bool(
            current_max_steps == {120} and baseline_max_steps == {120}
        ),
        "only_logging_changed": bool(
            episode_identity_match
            and behavior_match
            and trajectory_match
        ),
        "logging_perturbed_trajectory": bool(not trajectory_match),
    }
    authority_checks = evaluation.get("previous_formal_identity_checks", {})
    identity_lock["same_frozen_controller_identity"] = bool(
        authority_checks and all(bool(value) for value in authority_checks.values())
    )
    component_identity = compare_identity_components(
        evaluation,
        baseline_evaluation,
    )
    for component_name, component_result in component_identity.items():
        identity_lock[component_name] = component_result["status"]
    identity_lock["same_reward_dynamics_action_safety_boundary"] = {
        "reward": component_identity["reward_identity"]["status"],
        "dynamics": component_identity["dynamics_identity"]["status"],
        "action_transform": component_identity["action_transform_identity"]["status"],
        "safety_projection": component_identity["safety_projection_identity"]["status"],
        "boundary_projection": component_identity["boundary_projection_identity"]["status"],
    }
    identity_lock["component_identity"] = component_identity
    identity_lock["same_goal_recovery_activation"] = bool(
        evaluation.get("effective_goal_recovery_correction_config", {}).get(
            "activation_mode", "post_bypass_only"
        )
        == "post_bypass_only"
    )
    identity_lock["only_logging_changed"] = bool(
        episode_identity_match
        and behavior_match
        and trajectory_match
        and identity_lock["same_frozen_controller_identity"]
        and identity_lock["same_goal_recovery_activation"]
    )
    identity_lock["trajectory_fingerprint_behavior_protection"] = {
        "status": "ACTIVE",
        "source": "step_level_behavior_trace",
        "fail_fast": "LOGGING_PERTURBED_TRAJECTORY",
        "meaning": (
            "The step-level trajectory fingerprint remains the behavioral "
            "equivalence guard when a component identity field is unavailable."
        ),
    }
    payload = {
        "stage": "single_uav_signed_bypass_observability_diagnostic",
        "schema_version": SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION,
        "episode_count": len(episode_rows),
        "step_count": len(step_rows),
        "axis_source": "goal_projected_xy_left",
        "sign_convention": "+1=goal-frame-left,-1=goal-frame-right",
        "teacher_or_oracle_called": False,
        "only_logging_changed": identity_lock["only_logging_changed"],
        "identity_lock": identity_lock,
        "behavior_differences": behavior_differences,
        "trajectory_fingerprint_checked": True,
        "trajectory_fingerprint_behavior_protection": identity_lock[
            "trajectory_fingerprint_behavior_protection"
        ],
        "episode_failure_label_counts": {
            label: sum(1 for row in focus if row["signed_failure_label"] == label)
            for label in sorted({str(row["signed_failure_label"]) for row in focus})
        },
        "evaluation": dict(evaluation),
        "flags": {
            **EVALUATION_ONLY_FLAGS,
            "signed_bypass_observability_prepared": True,
            "behavior_algorithm_modified": False,
            "controller_modified": False,
            "observation_modified": False,
            "rollout_executed": True,
        },
    }
    _write_rows_csv(root / "signed_bypass_steps.csv", step_rows)
    _write_rows_csv(root / "signed_bypass_episodes.csv", episode_rows)
    _write_rows_csv(root / "signed_bypass_focus_failures.csv", focus)
    _write_rows_csv(root / "signed_bypass_direction_transitions.csv", transitions)
    _write_json(root / "signed_bypass_summary.json", _jsonable(payload))
    return payload


def _load_terminal_capture_baseline(root: Path) -> tuple[dict[str, Any], Path, str]:
    """Load the frozen actual-stable baseline without constructing runtime state."""

    path = Path(root).resolve() / "evaluation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("stage") != STAGE_NAME
        or payload.get("condition") != "on"
        or payload.get("formal") is not True
        or int(payload.get("episode_count", 0)) != FORMAL_EPISODES
        or payload.get("actual_stable_validation") is not True
    ):
        raise ValueError("terminal baseline must be the ON formal20 actual-stable validation")
    return payload, path, _sha256(path)


def _load_terminal_capture_v1_baseline(
    root: Path,
) -> tuple[dict[str, Any], Path, str]:
    """Load the canonical v1 terminal-capture formal20 identity source."""

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    terminal_path = resolved / "terminal_capture_validation_summary.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    if not terminal_path.is_file():
        raise FileNotFoundError(terminal_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_validation") is not True
        or terminal.get("stage") != "single_uav_terminal_capture_validation"
        or terminal.get("candidate_canonical_defaults") is not True
    ):
        raise ValueError("v2 baseline must be the canonical v1 terminal-capture formal20 result")
    if terminal.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError("v1 terminal-capture baseline config hashes disagree")
    return evaluation, terminal_path, _sha256(terminal_path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _load_terminal_capture_v2_baseline(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], Path, str]:
    """Load the completed v2 formal20 result used by the horizon diagnostic."""

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    summary_path = resolved / "terminal_capture_v2_validation_summary.json"
    episodes_path = resolved / "episodes.csv"
    for path in (evaluation_path, summary_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = _read_csv_rows(episodes_path)
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_v2_validation") is not True
        or summary.get("stage") != "single_uav_terminal_capture_v2_validation"
        or int(summary.get("episodes", 0)) != FORMAL_EPISODES
    ):
        raise ValueError("horizon baseline must be the completed v2 formal20 validation")
    if len(episodes) != FORMAL_EPISODES:
        raise ValueError("horizon baseline episodes.csv must contain exactly 20 rows")
    if any(int(float(row.get("max_steps", 0))) != HORIZON_BASELINE_MAX_STEPS for row in episodes):
        raise ValueError("horizon baseline max_steps must be exactly 100")
    if episode_identity(episodes) != episode_identity(
        evaluation.get("results", {}).get("on", {}).get("episodes", [])
    ):
        raise ValueError("horizon baseline episode CSV identity disagrees with summary")
    if summary.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError("horizon baseline controller config hashes disagree")
    candidate = summary.get("candidate_config", {})
    expected_frozen = {
        "enabled": True,
        "terminal_capture_enabled": True,
        "terminal_capture_v2_enabled": True,
        "max_delta_v_mps": 0.6,
        "terminal_effective_deceleration_mps2": 1.697726,
        "terminal_stopping_margin_m": 1.328009,
        "terminal_tangential_gain": 1.0,
    }
    if any(candidate.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError("horizon baseline does not use the frozen v2 controller config")
    if (
        float(summary.get("stable_distance_threshold", float("nan"))) != 1.0
        or float(summary.get("stable_speed_threshold", float("nan"))) != 0.5
        or int(summary.get("stable_hold_steps", -1)) != 5
    ):
        raise ValueError("horizon baseline stable criterion is not canonical 1m/0.5mps/5")
    recovery_count = sum(
        int(float(row.get("passed_obstacle", 0)))
        and int(float(row.get("post_bypass_recovery", 0)))
        for row in episodes
    )
    if recovery_count != 11:
        raise ValueError("horizon baseline must contain the expected 11 recovery episodes")
    return evaluation, summary, episodes, summary_path, _sha256(summary_path)


def _load_terminal_capture_v21_baseline(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], Path, str]:
    """Load the completed v2.1 formal20/max100 baseline for v2.2.

    The v2.2 comparison is intentionally anchored to all three v2.1
    artifacts: the evaluation summary, the dedicated validation summary, and
    the ordered episode CSV.  This prevents a stale summary from silently
    becoming the behavioral baseline.
    """

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    summary_path = resolved / "terminal_capture_v21_validation_summary.json"
    episodes_path = resolved / "terminal_capture_v21_validation_episodes.csv"
    for path in (evaluation_path, summary_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = _read_csv_rows(episodes_path)
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_v21_validation") is not True
        or summary.get("stage") != "single_uav_terminal_capture_v21_validation"
        or int(summary.get("episodes", 0)) != FORMAL_EPISODES
    ):
        raise ValueError("v2.1 baseline must be the completed ON formal20 validation")
    if len(episodes) != FORMAL_EPISODES:
        raise ValueError("v2.1 baseline episode CSV must contain exactly 20 rows")
    if any(
        int(float(row.get("max_steps", 0))) != HORIZON_BASELINE_MAX_STEPS
        for row in episodes
    ):
        raise ValueError("v2.1 baseline max_steps must be exactly 100")
    if episode_identity(episodes) != episode_identity(
        evaluation.get("results", {}).get("on", {}).get("episodes", [])
    ):
        raise ValueError("v2.1 baseline episode CSV identity disagrees with summary")
    if summary.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError("v2.1 baseline controller config hashes disagree")
    candidate = summary.get("candidate_config", {})
    expected_frozen = {
        "enabled": True,
        "terminal_capture_enabled": True,
        "terminal_capture_v2_enabled": True,
        "terminal_radial_conflict_arbitration_enabled": True,
        "max_delta_v_mps": 0.6,
        "terminal_radial_speed_floor_mps": 0.25,
        "terminal_effective_deceleration_mps2": 1.697726,
        "terminal_stopping_margin_m": 1.328009,
        "terminal_tangential_gain": 1.0,
    }
    if any(candidate.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError("v2.1 baseline does not use the frozen v2/v2.1 config")
    if (
        float(summary.get("stable_distance_threshold", float("nan"))) != 1.0
        or float(summary.get("stable_speed_threshold", float("nan"))) != 0.5
        or int(summary.get("stable_hold_steps", -1)) != 5
    ):
        raise ValueError("v2.1 baseline stable criterion is not canonical 1m/0.5mps/5")
    return evaluation, summary, episodes, summary_path, _sha256(summary_path)


def _load_terminal_capture_v22_baseline(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], Path, str]:
    """Load the frozen v2.2 formal20/max100 artifacts for v2.3 identity."""

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    summary_path = resolved / "terminal_capture_v22_validation_summary.json"
    episodes_path = resolved / "terminal_capture_v22_validation_episodes.csv"
    for path in (evaluation_path, summary_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = _read_csv_rows(episodes_path)
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_v22_validation") is not True
        or summary.get("stage") != "single_uav_terminal_capture_v22_validation"
        or int(summary.get("episodes", 0)) != FORMAL_EPISODES
    ):
        raise ValueError("v2.2 baseline must be the completed ON formal20 validation")
    if len(episodes) != FORMAL_EPISODES:
        raise ValueError("v2.2 baseline episode CSV must contain exactly 20 rows")
    if any(
        int(float(row.get("max_steps", 0))) != HORIZON_BASELINE_MAX_STEPS
        for row in episodes
    ):
        raise ValueError("v2.2 baseline max_steps must be exactly 100")
    if episode_identity(episodes) != episode_identity(
        evaluation.get("results", {}).get("on", {}).get("episodes", [])
    ):
        raise ValueError("v2.2 baseline episode CSV identity disagrees with summary")
    if summary.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError("v2.2 baseline controller config hashes disagree")
    candidate = summary.get("candidate_config", {})
    expected_frozen = {
        "enabled": True,
        "terminal_capture_enabled": True,
        "terminal_capture_v2_enabled": True,
        "terminal_radial_conflict_arbitration_enabled": True,
        "terminal_capture_v22_enabled": True,
        "terminal_hold_enter_radius_m": 0.98,
        "terminal_hold_release_radius_m": 1.05,
        "terminal_hold_restorative_enabled": True,
        "max_delta_v_mps": 0.6,
        "terminal_radial_speed_floor_mps": 0.25,
        "terminal_effective_deceleration_mps2": 1.697726,
        "terminal_stopping_margin_m": 1.328009,
        "terminal_tangential_gain": 1.0,
    }
    if any(candidate.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError("v2.2 baseline does not use the frozen v2/v2.1/v2.2 config")
    if (
        float(summary.get("stable_distance_threshold", float("nan"))) != 1.0
        or float(summary.get("stable_speed_threshold", float("nan"))) != 0.5
        or int(summary.get("stable_hold_steps", -1)) != 5
    ):
        raise ValueError("v2.2 baseline stable criterion is not canonical 1m/0.5mps/5")
    return evaluation, summary, episodes, summary_path, _sha256(summary_path)


def _load_terminal_capture_v23_baseline(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], Path, str]:
    """Load the frozen v2.3 formal20/max100 artifacts for repair identity."""

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    summary_path = resolved / "terminal_capture_v23_validation_summary.json"
    episodes_path = resolved / "terminal_capture_v23_validation_episodes.csv"
    for path in (evaluation_path, summary_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = _read_csv_rows(episodes_path)
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_v23_validation") is not True
        or summary.get("stage") != "single_uav_terminal_capture_v23_validation"
        or int(summary.get("episodes", 0)) != FORMAL_EPISODES
    ):
        raise ValueError("v2.3 repair baseline must be the completed ON formal20 validation")
    if len(episodes) != FORMAL_EPISODES:
        raise ValueError("v2.3 repair baseline episode CSV must contain exactly 20 rows")
    if any(
        int(float(row.get("max_steps", 0))) != HORIZON_BASELINE_MAX_STEPS
        for row in episodes
    ):
        raise ValueError("v2.3 repair baseline max_steps must be exactly 100")
    if episode_identity(episodes) != episode_identity(
        evaluation.get("results", {}).get("on", {}).get("episodes", [])
    ):
        raise ValueError("v2.3 repair baseline episode CSV identity disagrees with summary")
    if summary.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError("v2.3 repair baseline controller config hashes disagree")
    candidate = summary.get("candidate_config", {})
    expected_frozen = {
        "enabled": True,
        "terminal_capture_enabled": True,
        "terminal_capture_v2_enabled": True,
        "terminal_radial_conflict_arbitration_enabled": True,
        "terminal_radial_conflict_compensation_gain": 1.0,
        "terminal_capture_v22_enabled": True,
        "terminal_hold_enter_radius_m": 0.98,
        "terminal_hold_release_radius_m": 1.05,
        "terminal_capture_v23_enabled": True,
        "terminal_hold_target_radius_m": 0.96,
        "max_delta_v_mps": 0.6,
        "terminal_radial_speed_floor_mps": 0.25,
        "terminal_effective_deceleration_mps2": 1.697726,
        "terminal_stopping_margin_m": 1.328009,
        "terminal_tangential_gain": 1.0,
    }
    if any(candidate.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError("v2.3 repair baseline does not use the frozen v2.3 config")
    if (
        float(summary.get("stable_distance_threshold", float("nan"))) != 1.0
        or float(summary.get("stable_speed_threshold", float("nan"))) != 0.5
        or int(summary.get("stable_hold_steps", -1)) != 5
    ):
        raise ValueError("v2.3 repair baseline stable criterion is not canonical 1m/0.5mps/5")
    return evaluation, summary, episodes, summary_path, _sha256(summary_path)


def _load_terminal_capture_v23_authority_repair_baseline(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], Path, str]:
    """Load the frozen authority-repaired v2.3 formal100 baseline.

    This loader is deliberately separate from the original v2.3 loader.  The
    horizon diagnostic must anchor to the already repaired formal100 artifacts,
    never silently fall back to the pre-repair v2.3 validation directory.
    """

    resolved = Path(root).resolve()
    evaluation_path = resolved / "evaluation_summary.json"
    summary_path = resolved / (
        "terminal_capture_v23_authority_repair_validation_summary.json"
    )
    episodes_path = resolved / (
        "terminal_capture_v23_authority_repair_validation_episodes.csv"
    )
    for path in (evaluation_path, summary_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = _read_csv_rows(episodes_path)
    if (
        evaluation.get("stage") != STAGE_NAME
        or evaluation.get("condition") != "on"
        or evaluation.get("formal") is not True
        or int(evaluation.get("episode_count", 0)) != FORMAL_EPISODES
        or evaluation.get("terminal_capture_v23_authority_repair_validation") is not True
        or summary.get("stage")
        != "single_uav_terminal_capture_v23_authority_repair_validation"
        or int(summary.get("episodes", 0)) != FORMAL_EPISODES
        or summary.get("terminal_capture_v23_authority_repaired") is not True
        or summary.get("only_v23_authority_repair_changed") is not True
    ):
        raise ValueError(
            "v2.3 authority-repair horizon baseline must be the completed repaired formal100 validation"
        )
    if len(episodes) != FORMAL_EPISODES:
        raise ValueError(
            "v2.3 authority-repair horizon baseline episode CSV must contain exactly 20 rows"
        )
    if any(
        int(float(row.get("max_steps", 0))) != HORIZON_BASELINE_MAX_STEPS
        for row in episodes
    ):
        raise ValueError(
            "v2.3 authority-repair horizon baseline max_steps must be exactly 100"
        )
    if episode_identity(episodes) != episode_identity(
        evaluation.get("results", {}).get("on", {}).get("episodes", [])
    ):
        raise ValueError(
            "v2.3 authority-repair horizon baseline episode identity disagrees with evaluation summary"
        )
    if summary.get("candidate_config_hash") != evaluation.get(
        "effective_goal_recovery_correction_config_hash"
    ):
        raise ValueError(
            "v2.3 authority-repair horizon baseline controller config hashes disagree"
        )
    candidate = summary.get("candidate_config", {})
    expected_frozen = {
        "enabled": True,
        "radial_gain": 0.8,
        "tangential_gain": 0.7,
        "max_delta_v_mps": 0.6,
        "terminal_capture_enabled": True,
        "terminal_radial_speed_floor_mps": 0.25,
        "terminal_radial_brake_gain": 0.8,
        "terminal_tangential_gain": 1.0,
        "terminal_capture_v2_enabled": True,
        "terminal_effective_deceleration_mps2": 1.697726,
        "terminal_stopping_margin_m": 1.328009,
        "terminal_radial_conflict_arbitration_enabled": True,
        "terminal_radial_conflict_compensation_gain": 1.0,
        "terminal_capture_v22_enabled": True,
        "terminal_hold_enter_radius_m": 0.98,
        "terminal_hold_release_radius_m": 1.05,
        "terminal_hold_restorative_enabled": True,
        "terminal_capture_v23_enabled": True,
        "terminal_hold_target_radius_m": 0.96,
    }
    if any(candidate.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError(
            "v2.3 authority-repair horizon baseline does not use the frozen repaired config"
        )
    if (
        float(summary.get("stable_distance_threshold", float("nan"))) != 1.0
        or float(summary.get("stable_speed_threshold", float("nan"))) != 0.5
        or int(summary.get("stable_hold_steps", -1)) != 5
    ):
        raise ValueError(
            "v2.3 authority-repair horizon baseline stable criterion is not canonical 1m/0.5mps/5"
        )
    return evaluation, summary, episodes, summary_path, _sha256(summary_path)


def _effective_correction_config(
    config: FormalExperimentConfig,
    *,
    enabled: bool,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "radial_gain": float(config.goal_recovery_radial_gain),
        "tangential_gain": float(config.goal_recovery_tangential_gain),
        "max_delta_v_mps": float(config.goal_recovery_max_delta_v_mps),
        "activation_mode": str(config.goal_recovery_activation_mode),
        "alpha_ramp_enabled": bool(config.goal_recovery_alpha_ramp_enabled),
        "terminal_capture_enabled": bool(config.goal_recovery_terminal_capture_enabled),
        "terminal_capture_radius_m": float(config.goal_recovery_terminal_capture_radius_m),
        "terminal_radial_speed_floor_mps": float(config.goal_recovery_terminal_radial_speed_floor_mps),
        "terminal_radial_brake_gain": float(config.goal_recovery_terminal_radial_brake_gain),
        "terminal_tangential_gain": float(config.goal_recovery_terminal_tangential_gain),
        "terminal_propulsion_fade_enabled": bool(config.goal_recovery_terminal_propulsion_fade_enabled),
        "terminal_smooth_blend": bool(config.goal_recovery_terminal_smooth_blend),
        "terminal_capture_v2_enabled": bool(config.goal_recovery_terminal_capture_v2_enabled),
        "terminal_effective_deceleration_mps2": float(
            config.goal_recovery_terminal_effective_deceleration_mps2
        ),
        "terminal_stopping_margin_m": float(config.goal_recovery_terminal_stopping_margin_m),
        "terminal_hold_inside_goal_enabled": bool(
            config.goal_recovery_terminal_hold_inside_goal_enabled
        ),
        "terminal_radial_conflict_arbitration_enabled": bool(
            config.goal_recovery_terminal_radial_conflict_arbitration_enabled
        ),
        "terminal_radial_conflict_compensation_gain": float(
            config.goal_recovery_terminal_radial_conflict_compensation_gain
        ),
        "terminal_capture_v22_enabled": bool(
            config.goal_recovery_terminal_capture_v22_enabled
        ),
        "terminal_hold_enter_radius_m": float(
            config.goal_recovery_terminal_hold_enter_radius_m
        ),
        "terminal_hold_release_radius_m": float(
            config.goal_recovery_terminal_hold_release_radius_m
        ),
        "terminal_hold_restorative_enabled": bool(
            config.goal_recovery_terminal_hold_restorative_enabled
        ),
        "terminal_capture_v23_enabled": bool(
            config.goal_recovery_terminal_capture_v23_enabled
        ),
        "terminal_hold_target_radius_m": float(
            config.goal_recovery_terminal_hold_target_radius_m
        ),
    }


def _behavior_config_snapshot(config: FormalExperimentConfig) -> dict[str, Any]:
    """Canonical runtime config excluding output location and episode horizon."""

    return {
        field.name: _jsonable(getattr(config, field.name))
        for field in fields(config)
        if field.name not in {"output_dir", "max_steps"}
    }


def _horizon_only_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    diagnostic_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    baseline_normalized_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    checkpoint = baseline_evaluation["checkpoint"]
    baseline_episodes = baseline_evaluation["results"]["on"]["episodes"]
    controller_config = _effective_correction_config(diagnostic_config, enabled=True)
    frozen_config_same = _behavior_config_snapshot(baseline_config) == _behavior_config_snapshot(
        diagnostic_config
    )
    checks: dict[str, bool | int] = {
        "same_checkpoint_as_v2_formal": checkpoint_path.resolve()
        == Path(checkpoint.get("resolved_path", checkpoint.get("path", ""))).resolve(),
        "same_checkpoint_sha256": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest": baseline_normalized_manifest_hash
        == str(baseline_evaluation.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity": episode_identity(current_episodes)
        == episode_identity(baseline_episodes),
        "same_controller_config": _canonical_hash(controller_config)
        == str(baseline_summary.get("candidate_config_hash", "")),
        "same_reward": frozen_config_same,
        "same_dynamics": frozen_config_same,
        "same_stable_criterion": (
            frozen_config_same
            and float(diagnostic_config.single_uav_goal_threshold_m)
            == float(baseline_summary.get("stable_distance_threshold", float("nan")))
            and float(diagnostic_config.single_uav_goal_speed_threshold_mps)
            == float(baseline_summary.get("stable_speed_threshold", float("nan")))
            and int(diagnostic_config.single_uav_stable_hold_steps)
            == int(baseline_summary.get("stable_hold_steps", -1))
        ),
        "same_action_transform": frozen_config_same,
        "same_max_correction": (
            frozen_config_same
            and float(diagnostic_config.goal_recovery_max_delta_v_mps) == 0.6
        ),
        "baseline_max_steps": int(baseline_config.max_steps),
        "diagnostic_max_steps": int(diagnostic_config.max_steps),
    }
    invariant_keys = [key for key in checks if key.startswith("same_")]
    checks["only_horizon_changed"] = bool(
        int(baseline_config.max_steps) == HORIZON_BASELINE_MAX_STEPS
        and int(diagnostic_config.max_steps) == HORIZON_DIAGNOSTIC_MAX_STEPS
        and int(diagnostic_config.max_steps) - int(baseline_config.max_steps) == 20
        and all(bool(checks[key]) for key in invariant_keys)
    )
    failed = [key for key in invariant_keys if not bool(checks[key])]
    if failed or not bool(checks["only_horizon_changed"]):
        raise ValueError("horizon-only identity mismatch: " + ", ".join(failed or ["max_steps"]))
    return checks


def _v23_authority_repair_horizon_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    diagnostic_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    baseline_normalized_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    """Lock a v2.3 authority-repaired run to a horizon-only difference."""

    checks = dict(
        _horizon_only_identity_checks(
            baseline_evaluation=baseline_evaluation,
            baseline_summary=baseline_summary,
            baseline_config=baseline_config,
            diagnostic_config=diagnostic_config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            policy_hash=policy_hash,
            baseline_normalized_manifest_hash=baseline_normalized_manifest_hash,
            current_episodes=current_episodes,
        )
    )
    baseline_checkpoint = baseline_evaluation.get("checkpoint", {})
    behavior_same = _behavior_config_snapshot(baseline_config) == _behavior_config_snapshot(
        diagnostic_config
    )
    baseline_controller = dict(baseline_summary.get("candidate_config", {}))
    diagnostic_controller = _effective_correction_config(diagnostic_config, enabled=True)
    checks.update(
        {
            "same_checkpoint": bool(checks.get("same_checkpoint_as_v2_formal")),
            "same_observation": (
                behavior_same
                and int(getattr(baseline_config, "node_feature_dim", -1))
                == int(getattr(diagnostic_config, "node_feature_dim", -2))
                and int(baseline_checkpoint.get("observation_dimension", -3))
                == int(getattr(diagnostic_config, "node_feature_dim", -4))
            ),
            # The frozen checkpoint/policy identity is the authoritative lock
            # for Actor, Critic and GAT; no separate network mutation is
            # allowed in this evaluation-only mode.
            "same_actor": bool(checks.get("same_policy_hash")),
            "same_critic": bool(checks.get("same_checkpoint_sha256")),
            "same_gat": bool(checks.get("same_checkpoint_sha256")),
            "same_radial_gain": (
                behavior_same
                and float(diagnostic_config.goal_recovery_radial_gain) == 0.8
            ),
            "same_tangential_gain": (
                behavior_same
                and float(diagnostic_config.goal_recovery_tangential_gain) == 0.7
            ),
            "same_terminal_radial_speed_floor": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_radial_speed_floor_mps)
                == 0.25
            ),
            "same_terminal_radial_brake_gain": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_radial_brake_gain)
                == 0.8
            ),
            "same_terminal_tangential_gain": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_tangential_gain)
                == 1.0
            ),
            "same_v2_governor": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_effective_deceleration_mps2)
                == 1.697726
                and float(diagnostic_config.goal_recovery_terminal_stopping_margin_m)
                == 1.328009
            ),
            "same_a_eff": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_effective_deceleration_mps2)
                == 1.697726
            ),
            "same_stopping_margin": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_stopping_margin_m)
                == 1.328009
            ),
            "same_v21_arbitration": (
                behavior_same
                and bool(diagnostic_config.goal_recovery_terminal_radial_conflict_arbitration_enabled)
                and float(diagnostic_config.goal_recovery_terminal_radial_conflict_compensation_gain)
                == 1.0
            ),
            "same_v21_gain": (
                behavior_same
                and float(diagnostic_config.goal_recovery_terminal_radial_conflict_compensation_gain)
                == 1.0
            ),
            "same_obstacle_logic": behavior_same,
            "same_safety_projection": behavior_same,
            "same_boundary_projection": behavior_same,
            "same_v22_parameters": (
                behavior_same
                and bool(diagnostic_config.goal_recovery_terminal_capture_v22_enabled)
                and float(diagnostic_config.goal_recovery_terminal_hold_enter_radius_m) == 0.98
                and float(diagnostic_config.goal_recovery_terminal_hold_release_radius_m) == 1.05
                and bool(diagnostic_config.goal_recovery_terminal_hold_restorative_enabled)
            ),
            "same_v23_parameters": (
                behavior_same
                and bool(diagnostic_config.goal_recovery_terminal_capture_v23_enabled)
                and float(diagnostic_config.goal_recovery_terminal_hold_target_radius_m) == 0.96
            ),
            "same_authority_repair_behavior": (
                baseline_summary.get("terminal_capture_v23_authority_repaired") is True
                and baseline_summary.get("only_v23_authority_repair_changed") is True
                and baseline_controller == diagnostic_controller
            ),
            "same_controller_config": (
                baseline_controller == diagnostic_controller
                and bool(checks.get("same_controller_config"))
            ),
        }
    )
    identity_keys = (
        "same_checkpoint",
        "same_checkpoint_sha256",
        "same_policy_hash",
        "same_scenario_manifest",
        "same_ordered_episode_identity",
        "same_observation",
        "same_actor",
        "same_critic",
        "same_gat",
        "same_radial_gain",
        "same_tangential_gain",
        "same_terminal_radial_speed_floor",
        "same_terminal_radial_brake_gain",
        "same_terminal_tangential_gain",
        "same_reward",
        "same_dynamics",
        "same_action_transform",
        "same_obstacle_logic",
        "same_safety_projection",
        "same_boundary_projection",
        "same_stable_criterion",
        "same_max_correction",
        "same_a_eff",
        "same_stopping_margin",
        "same_v21_gain",
        "same_v2_governor",
        "same_v21_arbitration",
        "same_v22_parameters",
        "same_v23_parameters",
        "same_authority_repair_behavior",
        "same_controller_config",
    )
    checks["only_horizon_changed"] = bool(
        int(baseline_config.max_steps) == HORIZON_BASELINE_MAX_STEPS
        and int(diagnostic_config.max_steps) == HORIZON_DIAGNOSTIC_MAX_STEPS
        and all(bool(checks.get(key)) for key in identity_keys)
    )
    failed = [key for key in identity_keys if not bool(checks.get(key))]
    if failed or not bool(checks["only_horizon_changed"]):
        raise ValueError(
            "v2.3 authority-repair horizon identity mismatch: "
            + ", ".join(failed or ["only_horizon_changed"])
        )
    return checks


def _v21_only_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    candidate_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    scenario_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    """Prove that v2.1 changes only the opt-in arbitration enable flag."""

    checkpoint = baseline_evaluation["checkpoint"]
    baseline_episodes = baseline_evaluation["results"]["on"]["episodes"]
    baseline_behavior = _behavior_config_snapshot(baseline_config)
    candidate_behavior = _behavior_config_snapshot(candidate_config)
    enable_field = "goal_recovery_terminal_radial_conflict_arbitration_enabled"
    gain_field = "goal_recovery_terminal_radial_conflict_compensation_gain"
    baseline_behavior.setdefault(enable_field, False)
    baseline_behavior.setdefault(gain_field, 1.0)
    normalized_candidate_behavior = dict(candidate_behavior)
    normalized_candidate_behavior[enable_field] = False

    baseline_controller = dict(baseline_summary.get("candidate_config", {}))
    baseline_controller.setdefault("terminal_radial_conflict_arbitration_enabled", False)
    baseline_controller.setdefault("terminal_radial_conflict_compensation_gain", 1.0)
    candidate_controller = _effective_correction_config(candidate_config, enabled=True)
    normalized_candidate_controller = dict(candidate_controller)
    normalized_candidate_controller["terminal_radial_conflict_arbitration_enabled"] = False

    checks: dict[str, bool | int] = {
        "same_checkpoint_as_v2_formal": checkpoint_path.resolve()
        == Path(checkpoint.get("resolved_path", checkpoint.get("path", ""))).resolve(),
        "same_checkpoint_sha256": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest": scenario_manifest_hash
        == str(baseline_evaluation.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity": episode_identity(current_episodes)
        == episode_identity(baseline_episodes),
        "same_reward": normalized_candidate_behavior == baseline_behavior,
        "same_dynamics": normalized_candidate_behavior == baseline_behavior,
        "same_action_transform": normalized_candidate_behavior == baseline_behavior,
        "same_stable_criterion": (
            normalized_candidate_behavior == baseline_behavior
            and float(candidate_config.single_uav_goal_threshold_m)
            == float(baseline_summary.get("stable_distance_threshold", float("nan")))
            and float(candidate_config.single_uav_goal_speed_threshold_mps)
            == float(baseline_summary.get("stable_speed_threshold", float("nan")))
            and int(candidate_config.single_uav_stable_hold_steps)
            == int(baseline_summary.get("stable_hold_steps", -1))
        ),
        "same_max_correction": float(candidate_config.goal_recovery_max_delta_v_mps) == 0.6,
        "same_a_eff": float(candidate_config.goal_recovery_terminal_effective_deceleration_mps2)
        == 1.697726,
        "same_stopping_margin": float(candidate_config.goal_recovery_terminal_stopping_margin_m)
        == 1.328009,
        "same_terminal_tangential_gain": float(
            candidate_config.goal_recovery_terminal_tangential_gain
        ) == 1.0,
        "same_max_steps": int(baseline_config.max_steps) == int(candidate_config.max_steps) == 100,
        "same_controller_except_v21_enable": normalized_candidate_controller
        == baseline_controller,
        "v21_baseline_disabled": bool(baseline_behavior[enable_field]) is False,
        "v21_candidate_enabled": bool(candidate_behavior.get(enable_field)) is True,
        "same_v21_compensation_gain": (
            float(baseline_behavior[gain_field])
            == float(candidate_behavior.get(gain_field, float("nan")))
            == 1.0
        ),
        "baseline_max_steps": int(baseline_config.max_steps),
        "candidate_max_steps": int(candidate_config.max_steps),
    }
    invariant_keys = [
        key for key, value in checks.items()
        if isinstance(value, bool) and key not in {"v21_baseline_disabled", "v21_candidate_enabled"}
    ]
    checks["only_v21_arbitration_enabled_changed"] = bool(
        all(bool(checks[key]) for key in invariant_keys)
        and checks["v21_baseline_disabled"]
        and checks["v21_candidate_enabled"]
    )
    failed = [key for key in invariant_keys if not bool(checks[key])]
    if failed or not bool(checks["only_v21_arbitration_enabled_changed"]):
        raise ValueError(
            "v2.1-only identity mismatch: "
            + ", ".join(failed or ["arbitration_enable_transition"])
        )
    return checks


def _v22_only_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    candidate_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    scenario_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    """Prove v2.2 differs from v2.1 only in terminal-hold fields."""

    checkpoint = baseline_evaluation["checkpoint"]
    baseline_episodes = baseline_evaluation["results"]["on"]["episodes"]
    v22_config_fields = {
        "goal_recovery_terminal_capture_v22_enabled",
        "goal_recovery_terminal_hold_enter_radius_m",
        "goal_recovery_terminal_hold_release_radius_m",
        "goal_recovery_terminal_hold_restorative_enabled",
    }
    baseline_behavior = _behavior_config_snapshot(baseline_config)
    candidate_behavior = _behavior_config_snapshot(candidate_config)
    normalized_candidate_behavior = dict(candidate_behavior)
    for field_name in v22_config_fields:
        normalized_candidate_behavior[field_name] = baseline_behavior[field_name]

    baseline_controller = dict(baseline_summary.get("candidate_config", {}))
    controller_defaults = {
        "terminal_capture_v22_enabled": False,
        "terminal_hold_enter_radius_m": 0.98,
        "terminal_hold_release_radius_m": 1.05,
        "terminal_hold_restorative_enabled": True,
    }
    for key, value in controller_defaults.items():
        baseline_controller.setdefault(key, value)
    candidate_controller = _effective_correction_config(candidate_config, enabled=True)
    normalized_candidate_controller = dict(candidate_controller)
    for key in controller_defaults:
        normalized_candidate_controller[key] = baseline_controller[key]

    checks: dict[str, bool | int] = {
        "same_checkpoint_as_v21_formal": checkpoint_path.resolve()
        == Path(checkpoint.get("resolved_path", checkpoint.get("path", ""))).resolve(),
        "same_checkpoint_sha256": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest": scenario_manifest_hash
        == str(baseline_evaluation.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity": episode_identity(current_episodes)
        == episode_identity(baseline_episodes),
        "same_reward": normalized_candidate_behavior == baseline_behavior,
        "same_dynamics": normalized_candidate_behavior == baseline_behavior,
        "same_action_transform": normalized_candidate_behavior == baseline_behavior,
        "same_stable_criterion": (
            normalized_candidate_behavior == baseline_behavior
            and float(candidate_config.single_uav_goal_threshold_m)
            == float(baseline_summary.get("stable_distance_threshold", float("nan")))
            and float(candidate_config.single_uav_goal_speed_threshold_mps)
            == float(baseline_summary.get("stable_speed_threshold", float("nan")))
            and int(candidate_config.single_uav_stable_hold_steps)
            == int(baseline_summary.get("stable_hold_steps", -1))
        ),
        "same_max_correction": float(candidate_config.goal_recovery_max_delta_v_mps) == 0.6,
        "same_terminal_radial_speed_floor": float(
            candidate_config.goal_recovery_terminal_radial_speed_floor_mps
        ) == 0.25,
        "same_a_eff": float(
            candidate_config.goal_recovery_terminal_effective_deceleration_mps2
        ) == 1.697726,
        "same_stopping_margin": float(
            candidate_config.goal_recovery_terminal_stopping_margin_m
        ) == 1.328009,
        "same_v21_compensation_gain": float(
            candidate_config.goal_recovery_terminal_radial_conflict_compensation_gain
        ) == 1.0,
        "same_max_steps": int(baseline_config.max_steps)
        == int(candidate_config.max_steps)
        == HORIZON_BASELINE_MAX_STEPS,
        "same_controller_except_v22": normalized_candidate_controller
        == baseline_controller,
        "v21_enabled": bool(candidate_config.goal_recovery_terminal_radial_conflict_arbitration_enabled),
        "v22_baseline_disabled": not bool(
            baseline_behavior["goal_recovery_terminal_capture_v22_enabled"]
        ),
        "v22_candidate_enabled": bool(
            candidate_config.goal_recovery_terminal_capture_v22_enabled
        ),
        "v22_enter_radius_locked": float(
            candidate_config.goal_recovery_terminal_hold_enter_radius_m
        )
        == 0.98,
        "v22_release_radius_locked": float(
            candidate_config.goal_recovery_terminal_hold_release_radius_m
        )
        == 1.05,
        "v22_restorative_locked": bool(
            candidate_config.goal_recovery_terminal_hold_restorative_enabled
        ),
    }
    v22_transition_keys = {
        "v22_baseline_disabled",
        "v22_candidate_enabled",
        "v22_enter_radius_locked",
        "v22_release_radius_locked",
        "v22_restorative_locked",
    }
    invariant_keys = [
        key for key, value in checks.items()
        if isinstance(value, bool) and key not in v22_transition_keys
    ]
    checks["only_v22_fields_changed"] = bool(
        all(bool(checks[key]) for key in invariant_keys)
        and checks["v22_baseline_disabled"]
        and checks["v22_candidate_enabled"]
        and checks["v22_enter_radius_locked"]
        and checks["v22_release_radius_locked"]
        and checks["v22_restorative_locked"]
    )
    failed = [key for key in invariant_keys if not bool(checks[key])]
    if failed or not bool(checks["only_v22_fields_changed"]):
        raise ValueError(
            "v2.2-only identity mismatch: "
            + ", ".join(failed or ["v22_transition"])
        )
    return checks


def _v23_only_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    candidate_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    scenario_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    """Prove v2.3 differs from v2.2 only in the target-radius guard fields."""

    checkpoint = baseline_evaluation["checkpoint"]
    baseline_episodes = baseline_evaluation["results"]["on"]["episodes"]
    v23_fields = {
        "goal_recovery_terminal_capture_v23_enabled",
        "goal_recovery_terminal_hold_target_radius_m",
    }
    baseline_behavior = _behavior_config_snapshot(baseline_config)
    candidate_behavior = _behavior_config_snapshot(candidate_config)
    normalized_candidate_behavior = dict(candidate_behavior)
    for field_name in v23_fields:
        normalized_candidate_behavior[field_name] = baseline_behavior[field_name]

    baseline_controller = dict(baseline_summary.get("candidate_config", {}))
    baseline_controller.setdefault("terminal_capture_v23_enabled", False)
    baseline_controller.setdefault("terminal_hold_target_radius_m", 0.96)
    candidate_controller = _effective_correction_config(candidate_config, enabled=True)
    normalized_candidate_controller = dict(candidate_controller)
    normalized_candidate_controller["terminal_capture_v23_enabled"] = False
    normalized_candidate_controller["terminal_hold_target_radius_m"] = baseline_controller[
        "terminal_hold_target_radius_m"
    ]

    checks: dict[str, bool | int] = {
        "same_checkpoint_as_v22_formal": checkpoint_path.resolve()
        == Path(checkpoint.get("resolved_path", checkpoint.get("path", ""))).resolve(),
        "same_checkpoint_sha256": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest": scenario_manifest_hash
        == str(baseline_evaluation.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity": episode_identity(current_episodes)
        == episode_identity(baseline_episodes),
        "same_reward": normalized_candidate_behavior == baseline_behavior,
        "same_dynamics": normalized_candidate_behavior == baseline_behavior,
        "same_action_transform": normalized_candidate_behavior == baseline_behavior,
        "same_stable_criterion": (
            normalized_candidate_behavior == baseline_behavior
            and float(candidate_config.single_uav_goal_threshold_m)
            == float(baseline_summary.get("stable_distance_threshold", float("nan")))
            and float(candidate_config.single_uav_goal_speed_threshold_mps)
            == float(baseline_summary.get("stable_speed_threshold", float("nan")))
            and int(candidate_config.single_uav_stable_hold_steps)
            == int(baseline_summary.get("stable_hold_steps", -1))
        ),
        "same_max_correction": float(candidate_config.goal_recovery_max_delta_v_mps) == 0.6,
        "same_a_eff": float(candidate_config.goal_recovery_terminal_effective_deceleration_mps2) == 1.697726,
        "same_stopping_margin": float(candidate_config.goal_recovery_terminal_stopping_margin_m) == 1.328009,
        "same_terminal_tangential_gain": float(candidate_config.goal_recovery_terminal_tangential_gain) == 1.0,
        "same_v22_enabled": bool(candidate_config.goal_recovery_terminal_capture_v22_enabled),
        "same_v22_enter_radius": float(candidate_config.goal_recovery_terminal_hold_enter_radius_m) == 0.98,
        "same_v22_release_radius": float(candidate_config.goal_recovery_terminal_hold_release_radius_m) == 1.05,
        "same_v22_restorative": bool(candidate_config.goal_recovery_terminal_hold_restorative_enabled),
        "same_max_steps": int(baseline_config.max_steps) == int(candidate_config.max_steps) == HORIZON_BASELINE_MAX_STEPS,
        "same_controller_except_v23": normalized_candidate_controller == baseline_controller,
        "v23_baseline_disabled": not bool(baseline_behavior["goal_recovery_terminal_capture_v23_enabled"]),
        "v23_candidate_enabled": bool(candidate_config.goal_recovery_terminal_capture_v23_enabled),
        "v23_target_locked": float(candidate_config.goal_recovery_terminal_hold_target_radius_m) == 0.96,
    }
    transition = {"v23_baseline_disabled", "v23_candidate_enabled", "v23_target_locked"}
    invariant_keys = [
        key for key, value in checks.items()
        if isinstance(value, bool) and key not in transition
    ]
    checks["only_v23_fields_changed"] = bool(
        all(bool(checks[key]) for key in invariant_keys)
        and checks["v23_baseline_disabled"]
        and checks["v23_candidate_enabled"]
        and checks["v23_target_locked"]
    )
    failed = [key for key in invariant_keys if not bool(checks[key])]
    if failed or not bool(checks["only_v23_fields_changed"]):
        raise ValueError(
            "v2.3-only identity mismatch: "
            + ", ".join(failed or ["v23_transition"])
        )
    return checks


def _v23_authority_repair_identity_checks(
    *,
    baseline_evaluation: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    baseline_config: FormalExperimentConfig,
    candidate_config: FormalExperimentConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    policy_hash: str,
    scenario_manifest_hash: str,
    current_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, bool | int]:
    """Prove the repaired run changes source behavior, not frozen identity."""

    checkpoint = baseline_evaluation["checkpoint"]
    baseline_episodes = baseline_evaluation["results"]["on"]["episodes"]
    baseline_behavior = _behavior_config_snapshot(baseline_config)
    candidate_behavior = _behavior_config_snapshot(candidate_config)
    baseline_controller = dict(baseline_summary.get("candidate_config", {}))
    candidate_controller = _effective_correction_config(candidate_config, enabled=True)
    checks: dict[str, bool | int] = {
        "same_checkpoint_as_v23_formal": checkpoint_path.resolve()
        == Path(checkpoint.get("resolved_path", checkpoint.get("path", ""))).resolve(),
        "same_checkpoint_sha256": checkpoint_sha256 == str(checkpoint.get("sha256", "")),
        "same_policy_hash": policy_hash == str(checkpoint.get("policy_hash", "")),
        "same_scenario_manifest": scenario_manifest_hash
        == str(baseline_evaluation.get("scenario_manifest_hash", "")),
        "same_ordered_episode_identity": episode_identity(current_episodes)
        == episode_identity(baseline_episodes),
        "same_reward": candidate_behavior == baseline_behavior,
        "same_dynamics": candidate_behavior == baseline_behavior,
        "same_action_transform": candidate_behavior == baseline_behavior,
        "same_stable_criterion": (
            candidate_behavior == baseline_behavior
            and float(candidate_config.single_uav_goal_threshold_m)
            == float(baseline_summary.get("stable_distance_threshold", float("nan")))
            and float(candidate_config.single_uav_goal_speed_threshold_mps)
            == float(baseline_summary.get("stable_speed_threshold", float("nan")))
            and int(candidate_config.single_uav_stable_hold_steps)
            == int(baseline_summary.get("stable_hold_steps", -1))
        ),
        "same_max_correction": float(candidate_config.goal_recovery_max_delta_v_mps) == 0.6,
        "same_a_eff": float(candidate_config.goal_recovery_terminal_effective_deceleration_mps2) == 1.697726,
        "same_stopping_margin": float(candidate_config.goal_recovery_terminal_stopping_margin_m) == 1.328009,
        "same_v21_gain": float(candidate_config.goal_recovery_terminal_radial_conflict_compensation_gain) == 1.0,
        "same_v22_parameters": (
            bool(candidate_config.goal_recovery_terminal_capture_v22_enabled)
            and float(candidate_config.goal_recovery_terminal_hold_enter_radius_m) == 0.98
            and float(candidate_config.goal_recovery_terminal_hold_release_radius_m) == 1.05
            and bool(candidate_config.goal_recovery_terminal_hold_restorative_enabled)
        ),
        "same_v23_target": (
            bool(candidate_config.goal_recovery_terminal_capture_v23_enabled)
            and float(candidate_config.goal_recovery_terminal_hold_target_radius_m) == 0.96
        ),
        "same_max_steps": int(baseline_config.max_steps)
        == int(candidate_config.max_steps)
        == HORIZON_BASELINE_MAX_STEPS,
        "same_controller_config": candidate_controller == baseline_controller,
    }
    invariant_keys = [key for key, value in checks.items() if isinstance(value, bool)]
    checks["only_v23_authority_repair_changed"] = all(
        bool(checks[key]) for key in invariant_keys
    )
    failed = [key for key in invariant_keys if not bool(checks[key])]
    if failed or not bool(checks["only_v23_authority_repair_changed"]):
        raise ValueError(
            "v2.3 authority-repair identity mismatch: "
            + ", ".join(failed or ["authority_repair_transition"])
        )
    return checks


def _augment_horizon_episode_summary(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add observational horizon-boundary fields without changing control."""

    ordered = sorted(rows, key=lambda row: int(row["step"]))

    def first(predicate: Any) -> Mapping[str, Any] | None:
        return next((row for row in ordered if predicate(row)), None)

    def snapshot(executed_step: int) -> Mapping[str, Any] | None:
        logged_index = executed_step - 1
        return next((row for row in ordered if int(row["step"]) == logged_index), None)

    first_entry = first(lambda row: int(row.get("goal_distance_ok", 0)))
    first_speed = first(lambda row: int(row.get("goal_speed_ok", 0)))
    first_candidate = first(lambda row: int(row.get("stable_candidate_step", 0)))
    first_stable = first(lambda row: int(row.get("stable_goal", 0)))
    first_terminal = first(
        lambda row: int(row.get("terminated", 0)) or int(row.get("truncated", 0))
    )
    after_100_entry = (
        first_entry
        if first_entry is not None
        and int(first_entry["step"]) >= HORIZON_BASELINE_MAX_STEPS
        else None
    )
    after_100_stable = first(
        lambda row: int(row["step"]) >= HORIZON_BASELINE_MAX_STEPS
        and int(row.get("stable_goal", 0))
    )
    baseline_rows = [
        row for row in ordered
        if 0 <= int(row["step"]) < HORIZON_BASELINE_MAX_STEPS
    ]
    extension_rows = [
        row for row in ordered
        if HORIZON_BASELINE_MAX_STEPS
        <= int(row["step"])
        < HORIZON_DIAGNOSTIC_MAX_STEPS
    ]

    def finite_mean(rows_to_measure: Sequence[Mapping[str, Any]], field: str) -> float:
        values = []
        for row in rows_to_measure:
            try:
                value = float(row.get(field, float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return _mean(values)

    def vector_norm(row: Mapping[str, Any], field: str) -> float:
        value = row.get(field, (0.0, 0.0, 0.0))
        try:
            return float(np.linalg.norm(np.asarray(value, dtype=float).reshape(3)))
        except (TypeError, ValueError):
            try:
                return float(np.linalg.norm(np.asarray(json.loads(str(value)), dtype=float).reshape(3)))
            except (TypeError, ValueError, json.JSONDecodeError):
                return 0.0

    def switch_count(rows_to_measure: Sequence[Mapping[str, Any]]) -> int:
        states = [
            (
                vector_norm(row, "radial_propulsion_delta_v_mps") > 1.0e-12,
                vector_norm(row, "radial_braking_delta_v_mps") > 1.0e-12,
            )
            for row in rows_to_measure
        ]
        return sum(left != right for left, right in zip(states, states[1:]))

    def count(rows_to_measure: Sequence[Mapping[str, Any]], field: str) -> int:
        return sum(int(bool(row.get(field, 0))) for row in rows_to_measure)

    baseline_boundary_row = next(
        (row for row in ordered if int(row["step"]) == HORIZON_BASELINE_MAX_STEPS - 1),
        None,
    )
    diagnostic_boundary_row = next(
        (row for row in ordered if int(row["step"]) == HORIZON_DIAGNOSTIC_MAX_STEPS - 1),
        None,
    )
    extension_start_distance = (
        float(baseline_boundary_row.get("goal_distance_after_m", float("nan")))
        if baseline_boundary_row else float("nan")
    )
    extension_end_distance = (
        float(diagnostic_boundary_row.get("goal_distance_after_m", float("nan")))
        if diagnostic_boundary_row else float("nan")
    )
    extension_distance_reduction = (
        extension_start_distance - extension_end_distance
        if np.isfinite(extension_start_distance) and np.isfinite(extension_end_distance)
        else float("nan")
    )
    extension_actor_rows = [
        row for row in extension_rows
        if float(row.get("terminal_v21_actor_opposing_radial_mps", 0.0)) > 1.0e-9
    ]
    extension_conflict_rows = [
        row for row in extension_rows
        if int(bool(row.get("terminal_v21_radial_conflict_active", 0)))
    ]
    first_entry_in_extension = (
        after_100_entry is not None
        and int(after_100_entry["step"]) < HORIZON_DIAGNOSTIC_MAX_STEPS
    )
    stable_in_extension = (
        after_100_stable is not None
        and int(after_100_stable["step"]) < HORIZON_DIAGNOSTIC_MAX_STEPS
    )
    result = dict(summary)
    executed_steps = len(ordered)
    last_step = int(ordered[-1]["step"]) if ordered else None
    result.update({
        "configured_max_steps": HORIZON_DIAGNOSTIC_MAX_STEPS,
        "baseline_max_steps": HORIZON_BASELINE_MAX_STEPS,
        "horizon_delta_steps": HORIZON_DIAGNOSTIC_MAX_STEPS - HORIZON_BASELINE_MAX_STEPS,
        "executed_steps": executed_steps,
        "last_executed_step_index": last_step if last_step is not None else "",
        "reached_baseline_horizon_boundary": int(executed_steps >= HORIZON_BASELINE_MAX_STEPS),
        "entered_horizon_extension": int(executed_steps > HORIZON_BASELINE_MAX_STEPS),
        "extension_steps_executed": max(executed_steps - HORIZON_BASELINE_MAX_STEPS, 0),
        "reached_diagnostic_horizon_boundary": int(
            executed_steps >= HORIZON_DIAGNOSTIC_MAX_STEPS
        ),
        "termination_in_extension": int(
            first_terminal is not None
            and HORIZON_BASELINE_MAX_STEPS
            <= int(first_terminal["step"])
            < HORIZON_DIAGNOSTIC_MAX_STEPS
        ),
        "first_entered_1m_step": int(first_entry["step"]) if first_entry else "",
        "stable_goal_step": int(first_stable["step"]) if first_stable else "",
        "time_of_first_speed_ok": int(first_speed["step"]) if first_speed else "",
        "time_of_first_stable_candidate": int(first_candidate["step"]) if first_candidate else "",
        "entered_1m_by_step_100": int(any(
            int(row["step"]) <= 99 and int(row.get("goal_distance_ok", 0)) for row in ordered
        )),
        "stable_by_step_100": int(any(
            int(row["step"]) <= 99 and int(row.get("stable_goal", 0)) for row in ordered
        )),
        "entered_1m_after_step_100": int(after_100_entry is not None),
        "stable_after_step_100": int(after_100_stable is not None),
        "first_entered_1m_step_if_after_100": (
            int(after_100_entry["step"]) if after_100_entry else ""
        ),
        "stable_goal_step_if_after_100": (
            int(after_100_stable["step"]) if after_100_stable else ""
        ),
        "post_100_steps_used": sum(
            HORIZON_BASELINE_MAX_STEPS
            <= int(row["step"])
            < HORIZON_DIAGNOSTIC_MAX_STEPS
            for row in ordered
        ),
        "baseline_valid_step_index_start": 0,
        "baseline_valid_step_index_end": HORIZON_BASELINE_MAX_STEPS - 1,
        "extension_valid_step_index_start": HORIZON_BASELINE_MAX_STEPS,
        "extension_valid_step_index_end": HORIZON_DIAGNOSTIC_MAX_STEPS - 1,
        "entered1_before_or_at_100": int(bool(first_entry and int(first_entry["step"]) < HORIZON_BASELINE_MAX_STEPS)),
        "entered1_in_extension_101_120": int(first_entry_in_extension),
        "never_entered1_by_120": int(first_entry is None),
        "stable_before_or_at_100": int(bool(first_stable and int(first_stable["step"]) < HORIZON_BASELINE_MAX_STEPS)),
        "stable_in_extension_101_120": int(stable_in_extension),
        "never_stable_by_120": int(first_stable is None),
        "extension_steps_observed": len(extension_rows),
        "distance_at_extension_start_m": extension_start_distance,
        "distance_at_extension_end_m": extension_end_distance,
        "distance_reduction_during_extension_m": extension_distance_reduction,
        "mean_extension_radial_velocity_mps": finite_mean(
            extension_rows, "post_step_actual_radial_velocity_mps"
        ),
        "mean_extension_tangential_velocity_mps": finite_mean(
            extension_rows, "post_step_actual_tangential_velocity_mps"
        ),
        "extension_actor_opposing_fraction": (
            len(extension_actor_rows) / len(extension_rows)
            if extension_rows else 0.0
        ),
        "extension_v21_conflict_fraction": (
            len(extension_conflict_rows) / len(extension_rows)
            if extension_rows else 0.0
        ),
        "extension_propulsion_braking_switch_count": switch_count(extension_rows),
        "first100_collision_count": count(baseline_rows, "collision"),
        "extension_collision_count": count(extension_rows, "collision"),
        "first100_safety_projection_steps": count(baseline_rows, "safety_projection_applied"),
        "extension_safety_projection_steps": count(extension_rows, "safety_projection_applied"),
        "first100_boundary_projection_steps": count(baseline_rows, "boundary_projection_applied"),
        "extension_boundary_projection_steps": count(extension_rows, "boundary_projection_applied"),
        "first100_action_clipping_steps": count(baseline_rows, "action_clipped"),
        "extension_action_clipping_steps": count(extension_rows, "action_clipped"),
        "first100_terminal_saturation_steps": count(baseline_rows, "terminal_correction_clipped"),
        "extension_terminal_saturation_steps": count(extension_rows, "terminal_correction_clipped"),
    })
    for executed_step in (100, 110, 120):
        row = snapshot(executed_step)
        suffix = str(executed_step)
        result[f"goal_distance_at_step_{suffix}"] = (
            float(
                row.get(
                    "goal_distance_for_stable_m",
                    row.get("goal_distance_after_m", float("nan")),
                )
            )
            if row
            else float("nan")
        )
        result[f"actual_speed_at_step_{suffix}"] = (
            float(row.get("post_step_actual_speed_mps", float("nan")))
            if row
            else float("nan")
        )
        result[f"stable_counter_at_step_{suffix}"] = (
            int(row.get("stable_hold_counter", 0)) if row else ""
        )
        result[f"radial_velocity_at_step_{suffix}"] = (
            float(row.get("post_step_actual_radial_velocity_mps", float("nan")))
            if row
            else float("nan")
        )
        result[f"tangential_velocity_at_step_{suffix}"] = (
            float(row.get("post_step_actual_tangential_velocity_mps", float("nan")))
            if row
            else float("nan")
        )
    return result


def _build_horizon_recovery_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the baseline recovery subset to diagnostic episode summaries."""

    baseline_identity = episode_identity(baseline_rows)
    diagnostic_identity = episode_identity(diagnostic_rows)
    if baseline_identity != diagnostic_identity:
        raise ValueError("horizon recovery comparison requires identical ordered episodes")
    output: list[dict[str, Any]] = []
    for baseline, diagnostic in zip(baseline_rows, diagnostic_rows):
        if not (
            int(float(baseline.get("passed_obstacle", 0)))
            and int(float(baseline.get("post_bypass_recovery", 0)))
        ):
            continue
        output.append({
            "episode_id": str(baseline["episode_id"]),
            "formal100_baseline_min_distance_m": float(baseline["minimum_goal_distance_m"]),
            "formal100_min_distance_m": float(baseline["minimum_goal_distance_m"]),
            "v23_authority_repair_baseline_min_distance_m": float(
                baseline["minimum_goal_distance_m"]
            ),
            "v23_authority_repair_horizon120_min_distance_m": float(
                diagnostic["minimum_goal_distance_m"]
            ),
            "formal120_min_distance_m": float(diagnostic["minimum_goal_distance_m"]),
            # Preserve the older generic aliases for downstream readers that
            # already consume the v2 horizon comparison shape.
            "v2_baseline_min_distance_m": float(baseline["minimum_goal_distance_m"]),
            "diagnostic_min_distance_m": float(diagnostic["minimum_goal_distance_m"]),
            "distance_at_baseline_horizon_m": diagnostic["goal_distance_at_step_100"],
            "entered_1m_by_100": int(diagnostic["entered_1m_by_step_100"]),
            "entered_1m_by_120": int(diagnostic["entered_1m"]),
            "entered1_by120": int(diagnostic["entered_1m"]),
            "first_entered_1m_step": diagnostic["first_entered_1m_step"],
            "entered1_before_or_at_100": int(
                diagnostic.get("entered1_before_or_at_100", diagnostic.get("entered_1m_by_step_100", 0))
            ),
            "entered1_in_extension_101_120": int(
                diagnostic.get("entered1_in_extension_101_120", diagnostic.get("entered_1m_after_step_100", 0))
            ),
            "entered1_occurred_in_extension": int(
                diagnostic.get("entered1_in_extension_101_120", diagnostic.get("entered_1m_after_step_100", 0))
            ),
            "never_entered1_by_120": int(
                diagnostic.get("never_entered1_by_120", not int(diagnostic.get("entered_1m", 0)))
            ),
            "stable_counter_at_100": diagnostic["stable_counter_at_step_100"],
            "max_stable_counter_by_120": int(diagnostic["max_stable_hold_counter"]),
            "stable_goal_by_120": int(diagnostic["stable_goal"]),
            "stable_by_120": int(diagnostic["stable_goal"]),
            "stable_goal_step": diagnostic["stable_goal_step"],
            "stable_before_or_at_100": int(
                diagnostic.get("stable_before_or_at_100", diagnostic.get("stable_by_step_100", 0))
            ),
            "stable_in_extension_101_120": int(
                diagnostic.get("stable_in_extension_101_120", diagnostic.get("stable_after_step_100", 0))
            ),
            "stable_occurred_in_extension": int(
                diagnostic.get("stable_in_extension_101_120", diagnostic.get("stable_after_step_100", 0))
            ),
            "never_stable_by_120": int(
                diagnostic.get("never_stable_by_120", not int(diagnostic.get("stable_goal", 0)))
            ),
            "goal_distance_at_step_100": diagnostic.get("goal_distance_at_step_100", float("nan")),
            "goal_distance_at_step_110": diagnostic.get("goal_distance_at_step_110", float("nan")),
            "goal_distance_at_step_120": diagnostic.get("goal_distance_at_step_120", float("nan")),
            "distance_reduction_during_extension_m": diagnostic.get("distance_reduction_during_extension_m", float("nan")),
            "mean_extension_radial_velocity_mps": diagnostic.get("mean_extension_radial_velocity_mps", float("nan")),
            "mean_extension_tangential_velocity_mps": diagnostic.get("mean_extension_tangential_velocity_mps", float("nan")),
            "extension_actor_opposing_fraction": diagnostic.get("extension_actor_opposing_fraction", 0.0),
            "actor_opposing_fraction_in_extension": diagnostic.get("extension_actor_opposing_fraction", 0.0),
            "extension_v21_conflict_fraction": diagnostic.get("extension_v21_conflict_fraction", 0.0),
            "v21_conflict_fraction_in_extension": diagnostic.get("extension_v21_conflict_fraction", 0.0),
            "extension_propulsion_braking_switch_count": diagnostic.get("extension_propulsion_braking_switch_count", 0),
            "propulsion_braking_switch_count_in_extension": diagnostic.get("extension_propulsion_braking_switch_count", 0),
            "first100_collision_count": diagnostic.get("first100_collision_count", diagnostic.get("collision", 0)),
            "extension_collision_count": diagnostic.get("extension_collision_count", 0),
            "first100_safety_projection_steps": diagnostic.get("first100_safety_projection_steps", diagnostic.get("safety_projection_steps", 0)),
            "extension_safety_projection_steps": diagnostic.get("extension_safety_projection_steps", 0),
            "first100_boundary_projection_steps": diagnostic.get("first100_boundary_projection_steps", diagnostic.get("boundary_projection_steps", 0)),
            "extension_boundary_projection_steps": diagnostic.get("extension_boundary_projection_steps", 0),
            "first100_action_clipping_steps": diagnostic.get("first100_action_clipping_steps", 0),
            "extension_action_clipping_steps": diagnostic.get("extension_action_clipping_steps", 0),
            "first100_terminal_saturation_steps": diagnostic.get("first100_terminal_saturation_steps", 0),
            "extension_terminal_saturation_steps": diagnostic.get("extension_terminal_saturation_steps", 0),
            "actual_speed_at_step_100": diagnostic.get("actual_speed_at_step_100", float("nan")),
            "actual_speed_at_step_110": diagnostic.get("actual_speed_at_step_110", float("nan")),
            "actual_speed_at_step_120": diagnostic.get("actual_speed_at_step_120", float("nan")),
            "radial_velocity_at_step_100": diagnostic.get("radial_velocity_at_step_100", float("nan")),
            "radial_velocity_at_step_110": diagnostic.get("radial_velocity_at_step_110", float("nan")),
            "radial_velocity_at_step_120": diagnostic.get("radial_velocity_at_step_120", float("nan")),
            "tangential_velocity_at_step_100": diagnostic.get("tangential_velocity_at_step_100", float("nan")),
            "tangential_velocity_at_step_110": diagnostic.get("tangential_velocity_at_step_110", float("nan")),
            "tangential_velocity_at_step_120": diagnostic.get("tangential_velocity_at_step_120", float("nan")),
        })
    return output


def run_goal_recovery_correction_evaluation(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    condition: str = "paired",
    formal: bool = False,
    smoke_count: int = DEFAULT_SMOKE_EPISODES,
    progress: bool = True,
    actual_stable_validation: bool = False,
    previous_formal_root: str | Path = PREVIOUS_FORMAL_LOG_ROOT,
    terminal_capture_validation: bool = False,
    terminal_baseline_root: str | Path = ACTUAL_STABLE_BASELINE_ROOT,
    terminal_capture_v2_validation: bool = False,
    terminal_capture_v1_baseline_root: str | Path = TERMINAL_CAPTURE_V1_BASELINE_ROOT,
    terminal_capture_v2_horizon_diagnostic: bool = False,
    terminal_capture_v2_formal_baseline_root: str | Path = TERMINAL_CAPTURE_V2_FORMAL_BASELINE_ROOT,
    terminal_capture_v23_authority_repair_horizon_diagnostic: bool = False,
    terminal_capture_v21_validation: bool = False,
    terminal_capture_v22_validation: bool = False,
    terminal_capture_v23_validation: bool = False,
    terminal_capture_v23_authority_repair_validation: bool = False,
    signed_bypass_observability_diagnostic: bool = False,
    lateral_exit_braking_assistance_diagnostic: bool = False,
    radial_intent_persistence_diagnostic: bool = False,
    pass_commit_diagnostic: bool = False,
    terminal_capture_v21_baseline_root: str | Path = TERMINAL_CAPTURE_V21_FORMAL_BASELINE_ROOT,
    terminal_capture_v22_baseline_root: str | Path = TERMINAL_CAPTURE_V22_FORMAL_BASELINE_ROOT,
    terminal_capture_v23_baseline_root: str | Path = TERMINAL_CAPTURE_V23_FORMAL_BASELINE_ROOT,
    terminal_capture_radius_m: float = 8.0,
    terminal_radial_speed_floor_mps: float = 0.25,
    terminal_radial_brake_gain: float = 0.8,
    terminal_tangential_gain: float = 1.0,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not 1 <= int(smoke_count) <= 3:
        raise ValueError("smoke_count must be in [1, 3]")
    if signed_bypass_observability_diagnostic:
        if condition != "on" or not formal:
            raise ValueError(
                "signed bypass observability diagnostic requires formal ON-only mode"
            )
        terminal_capture_v23_authority_repair_horizon_diagnostic = True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = SIGNED_BYPASS_OBSERVABILITY_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = SIGNED_BYPASS_OBSERVABILITY_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = SIGNED_BYPASS_OBSERVABILITY_PLOT_ROOT
    if lateral_exit_braking_assistance_diagnostic:
        if signed_bypass_observability_diagnostic:
            raise ValueError("lateral-exit assistance and signed logging modes are exclusive")
        if not formal or condition not in {"paired", "off", "on"}:
            raise ValueError(
                "lateral-exit braking assistance requires locked formal OFF/ON/paired mode"
            )
        terminal_capture_v23_authority_repair_horizon_diagnostic = True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = LATERAL_EXIT_BRAKING_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = LATERAL_EXIT_BRAKING_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = LATERAL_EXIT_BRAKING_PLOT_ROOT
    radial_intent_baseline_contract = None
    radial_intent_cap_derivation = None
    radial_intent_source_lock = None
    radial_intent_assistance_type = None
    pass_commit_baseline_contract = None
    pass_commit_cap_derivation = None
    pass_commit_source_lock = None
    pass_commit_preregistration = None
    if radial_intent_persistence_diagnostic:
        if lateral_exit_braking_assistance_diagnostic or signed_bypass_observability_diagnostic:
            raise ValueError("radial-intent persistence is exclusive with legacy lateral and signed-only diagnostics")
        if not formal or condition != "paired":
            raise ValueError("radial-intent persistence requires locked formal paired20 mode")
        if any((
            actual_stable_validation,
            terminal_capture_validation,
            terminal_capture_v2_validation,
            terminal_capture_v2_horizon_diagnostic,
            terminal_capture_v21_validation,
            terminal_capture_v22_validation,
            terminal_capture_v23_validation,
            terminal_capture_v23_authority_repair_validation,
        )):
            raise ValueError("radial-intent persistence is exclusive with other evaluation modes")
        terminal_capture_v23_authority_repair_horizon_diagnostic = True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = RADIAL_INTENT_PERSISTENCE_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = RADIAL_INTENT_PERSISTENCE_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = RADIAL_INTENT_PERSISTENCE_PLOT_ROOT
        from .single_uav_radial_intent_persistence_diagnostic import (
            RadialIntentPersistenceAssistance as _RadialIntentPersistenceAssistance,
            load_runtime_locks as _load_radial_intent_runtime_locks,
        )

        radial_intent_assistance_type = _RadialIntentPersistenceAssistance

        (
            radial_intent_baseline_contract,
            radial_intent_cap_derivation,
            radial_intent_source_lock,
        ) = _load_radial_intent_runtime_locks()
        if float(radial_intent_cap_derivation.get("max_radial_intent_assist_mps", 0.0)) <= 0.0:
            raise ValueError("RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED")
    if pass_commit_diagnostic:
        if (
            lateral_exit_braking_assistance_diagnostic
            or radial_intent_persistence_diagnostic
            or signed_bypass_observability_diagnostic
        ):
            raise ValueError("PASS_COMMIT owns its frozen base diagnostic modes")
        if not formal or condition != "paired":
            raise ValueError("PASS_COMMIT requires locked formal paired20 mode")
        if any((
            actual_stable_validation,
            terminal_capture_validation,
            terminal_capture_v2_validation,
            terminal_capture_v2_horizon_diagnostic,
            terminal_capture_v21_validation,
            terminal_capture_v22_validation,
            terminal_capture_v23_validation,
            terminal_capture_v23_authority_repair_validation,
        )):
            raise ValueError("PASS_COMMIT is exclusive with other evaluation modes")
        terminal_capture_v23_authority_repair_horizon_diagnostic = True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = PASS_COMMIT_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = PASS_COMMIT_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = PASS_COMMIT_PLOT_ROOT
        from .single_uav_pass_commit_diagnostic import (
            load_runtime_locks as _load_pass_commit_runtime_locks,
        )

        (
            pass_commit_baseline_contract,
            pass_commit_cap_derivation,
            pass_commit_source_lock,
            pass_commit_preregistration,
        ) = _load_pass_commit_runtime_locks()
        if not 0.0 < float(
            pass_commit_cap_derivation.get("pass_commit_max_delta_v_mps", 0.0)
        ) <= PASS_COMMIT_HARD_CAP_MPS:
            raise ValueError("PASS_COMMIT_CAP_DERIVATION_FAIL_CLOSED")
    if actual_stable_validation and (condition != "on" or not formal):
        raise ValueError("actual stable validation requires formal ON-only mode")
    if terminal_capture_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture validation requires formal ON-only mode")
    if terminal_capture_v2_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture v2 validation requires formal ON-only mode")
    if terminal_capture_v2_horizon_diagnostic and (condition != "on" or not formal):
        raise ValueError("terminal capture v2 horizon diagnostic requires formal ON-only mode")
    if (
        terminal_capture_v23_authority_repair_horizon_diagnostic
        and not lateral_exit_braking_assistance_diagnostic
        and not radial_intent_persistence_diagnostic
        and not pass_commit_diagnostic
        and (condition != "on" or not formal)
    ):
        raise ValueError(
            "terminal capture v2.3 authority-repair horizon diagnostic requires formal ON-only mode"
        )
    if terminal_capture_v21_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture v2.1 validation requires formal ON-only mode")
    if terminal_capture_v22_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture v2.2 validation requires formal ON-only mode")
    if terminal_capture_v23_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture v2.3 validation requires formal ON-only mode")
    if terminal_capture_v23_authority_repair_validation and (condition != "on" or not formal):
        raise ValueError("terminal capture v2.3 authority repair validation requires formal ON-only mode")
    if (
        terminal_capture_v22_validation
        or terminal_capture_v23_authority_repair_horizon_diagnostic
        or terminal_capture_v23_validation
        or terminal_capture_v23_authority_repair_validation
    ):
        locked_terminal_parameters = {
            "terminal_capture_radius_m": 8.0,
            "terminal_radial_speed_floor_mps": 0.25,
            "terminal_radial_brake_gain": 0.8,
            "terminal_tangential_gain": 1.0,
        }
        actual_terminal_parameters = {
            "terminal_capture_radius_m": float(terminal_capture_radius_m),
            "terminal_radial_speed_floor_mps": float(terminal_radial_speed_floor_mps),
            "terminal_radial_brake_gain": float(terminal_radial_brake_gain),
            "terminal_tangential_gain": float(terminal_tangential_gain),
        }
        if actual_terminal_parameters != locked_terminal_parameters:
            raise ValueError("terminal capture v2.2/v2.3 validation uses locked terminal parameters")
    if sum(bool(value) for value in (
        actual_stable_validation,
        terminal_capture_validation,
        terminal_capture_v2_validation,
        terminal_capture_v2_horizon_diagnostic,
        terminal_capture_v23_authority_repair_horizon_diagnostic,
        terminal_capture_v21_validation,
        terminal_capture_v22_validation,
        terminal_capture_v23_validation,
        terminal_capture_v23_authority_repair_validation,
    )) > 1:
        raise ValueError("validation modes are mutually exclusive")
    terminal_parameter_config = GoalRecoveryCorrectionConfig(
        enabled=True,
        terminal_capture_enabled=True,
        terminal_capture_radius_m=float(terminal_capture_radius_m),
        terminal_radial_speed_floor_mps=float(terminal_radial_speed_floor_mps),
        terminal_radial_brake_gain=float(terminal_radial_brake_gain),
        terminal_tangential_gain=float(terminal_tangential_gain),
    ) if (
        terminal_capture_validation
        or terminal_capture_v2_validation
        or terminal_capture_v2_horizon_diagnostic
        or terminal_capture_v23_authority_repair_horizon_diagnostic
        or terminal_capture_v21_validation
        or terminal_capture_v22_validation
        or terminal_capture_v23_validation
        or terminal_capture_v23_authority_repair_validation
    ) else None
    modes = _condition_names(condition)
    planned_data = _validate_fresh_output_layout(Path(data_root), condition, checkpoint_path)
    _validate_fresh_output_layout(Path(logs_root), condition, checkpoint_path)
    _validate_fresh_output_layout(Path(plots_root), condition, checkpoint_path)
    previous_formal = (
        load_previous_formal_summary(Path(previous_formal_root))
        if actual_stable_validation else None
    )
    terminal_baseline = None
    terminal_baseline_path = None
    terminal_baseline_sha256 = None
    if terminal_capture_validation:
        (
            terminal_baseline,
            terminal_baseline_path,
            terminal_baseline_sha256,
        ) = _load_terminal_capture_baseline(Path(terminal_baseline_root))
    terminal_v1_baseline = None
    terminal_v1_baseline_path = None
    terminal_v1_baseline_sha256 = None
    if terminal_capture_v2_validation:
        (
            terminal_v1_baseline,
            terminal_v1_baseline_path,
            terminal_v1_baseline_sha256,
        ) = _load_terminal_capture_v1_baseline(Path(terminal_capture_v1_baseline_root))
    horizon_baseline = None
    horizon_baseline_summary = None
    horizon_baseline_episodes = None
    horizon_baseline_path = None
    horizon_baseline_sha256 = None
    if terminal_capture_v2_horizon_diagnostic:
        (
            horizon_baseline,
            horizon_baseline_summary,
            horizon_baseline_episodes,
            horizon_baseline_path,
            horizon_baseline_sha256,
        ) = _load_terminal_capture_v2_baseline(
            Path(terminal_capture_v2_formal_baseline_root)
        )
    v21_baseline = None
    v21_baseline_summary = None
    v21_baseline_episodes = None
    v21_baseline_path = None
    v21_baseline_sha256 = None
    if terminal_capture_v21_validation:
        (
            v21_baseline,
            v21_baseline_summary,
            v21_baseline_episodes,
            v21_baseline_path,
            v21_baseline_sha256,
        ) = _load_terminal_capture_v2_baseline(TERMINAL_CAPTURE_V2_FORMAL_BASELINE_ROOT)
    v22_baseline = None
    v22_baseline_summary = None
    v22_baseline_episodes = None
    v22_baseline_path = None
    v22_baseline_sha256 = None
    if terminal_capture_v22_validation:
        (
            v22_baseline,
            v22_baseline_summary,
            v22_baseline_episodes,
            v22_baseline_path,
            v22_baseline_sha256,
        ) = _load_terminal_capture_v21_baseline(
            Path(terminal_capture_v21_baseline_root)
        )
    v23_baseline = None
    v23_baseline_summary = None
    v23_baseline_episodes = None
    v23_baseline_path = None
    v23_baseline_sha256 = None
    if terminal_capture_v23_validation:
        (
            v23_baseline,
            v23_baseline_summary,
            v23_baseline_episodes,
            v23_baseline_path,
            v23_baseline_sha256,
        ) = _load_terminal_capture_v22_baseline(
            Path(terminal_capture_v22_baseline_root)
        )
    v23_repair_baseline = None
    v23_repair_baseline_summary = None
    v23_repair_baseline_episodes = None
    v23_repair_baseline_path = None
    v23_repair_baseline_sha256 = None
    if terminal_capture_v23_authority_repair_validation:
        (
            v23_repair_baseline,
            v23_repair_baseline_summary,
            v23_repair_baseline_episodes,
            v23_repair_baseline_path,
            v23_repair_baseline_sha256,
        ) = _load_terminal_capture_v23_baseline(
            Path(terminal_capture_v23_baseline_root)
        )
    v23_repair_horizon_baseline = None
    v23_repair_horizon_baseline_summary = None
    v23_repair_horizon_baseline_episodes = None
    v23_repair_horizon_baseline_path = None
    v23_repair_horizon_baseline_sha256 = None
    if terminal_capture_v23_authority_repair_horizon_diagnostic:
        (
            v23_repair_horizon_baseline,
            v23_repair_horizon_baseline_summary,
            v23_repair_horizon_baseline_episodes,
            v23_repair_horizon_baseline_path,
            v23_repair_horizon_baseline_sha256,
        ) = _load_terminal_capture_v23_authority_repair_baseline(
            TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT
        )
    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_reference = (
        v23_repair_horizon_baseline
        if terminal_capture_v23_authority_repair_horizon_diagnostic
        else
        v23_repair_baseline
        if terminal_capture_v23_authority_repair_validation
        else v21_baseline
        if terminal_capture_v21_validation
        else v22_baseline
        if terminal_capture_v22_validation
        else v23_baseline
        if terminal_capture_v23_validation
        else horizon_baseline
        if terminal_capture_v2_horizon_diagnostic
        else terminal_v1_baseline
        if terminal_capture_v2_validation
        else terminal_baseline if terminal_capture_validation else previous_formal
    )
    if checkpoint_reference is not None:
        previous_checkpoint = checkpoint_reference["checkpoint"]
        previous_path = Path(
            previous_checkpoint.get("resolved_path", previous_checkpoint.get("path", ""))
        ).resolve()
        if checkpoint_path != previous_path or checkpoint_sha256 != str(previous_checkpoint.get("sha256", "")):
            raise ValueError("checkpoint path or SHA-256 differs from previous formal evaluation")
    trainer, config, payload, identity = _load_checkpoint(
        checkpoint_path, planned_data["base"] / "checkpoint_source"
    )
    if terminal_parameter_config is not None:
        config = replace(
            config,
            goal_recovery_terminal_capture_enabled=True,
            goal_recovery_terminal_capture_radius_m=terminal_parameter_config.terminal_capture_radius_m,
            goal_recovery_terminal_radial_speed_floor_mps=terminal_parameter_config.terminal_radial_speed_floor_mps,
            goal_recovery_terminal_radial_brake_gain=terminal_parameter_config.terminal_radial_brake_gain,
            goal_recovery_terminal_tangential_gain=terminal_parameter_config.terminal_tangential_gain,
            goal_recovery_terminal_capture_v2_enabled=bool(
                terminal_capture_v2_validation
                or terminal_capture_v2_horizon_diagnostic
                or terminal_capture_v23_authority_repair_horizon_diagnostic
                or terminal_capture_v21_validation
                or terminal_capture_v22_validation
                or terminal_capture_v23_validation
                or terminal_capture_v23_authority_repair_validation
            ),
            goal_recovery_terminal_radial_conflict_arbitration_enabled=bool(
                terminal_capture_v23_authority_repair_horizon_diagnostic
                or terminal_capture_v21_validation
                or terminal_capture_v22_validation
                or terminal_capture_v23_validation
                or terminal_capture_v23_authority_repair_validation
            ),
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
            # A v2.3 comparison is normalized against the completed v2.2
            # treatment, so its baseline runtime must keep v2.2 enabled while
            # leaving only v2.3 disabled.
            goal_recovery_terminal_capture_v22_enabled=bool(
                terminal_capture_v23_authority_repair_horizon_diagnostic
                or terminal_capture_v23_validation
                or terminal_capture_v23_authority_repair_validation
            ),
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
            goal_recovery_terminal_capture_v23_enabled=False,
            goal_recovery_terminal_hold_target_radius_m=0.96,
        )
    baseline_runtime_config = config
    if terminal_capture_v23_validation:
        baseline_runtime_config = replace(
            baseline_runtime_config,
            goal_recovery_max_delta_v_mps=0.6,
            goal_recovery_terminal_effective_deceleration_mps2=1.697726,
            goal_recovery_terminal_stopping_margin_m=1.328009,
            goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
            goal_recovery_terminal_capture_v22_enabled=True,
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
            goal_recovery_terminal_capture_v23_enabled=False,
            goal_recovery_terminal_hold_target_radius_m=0.96,
        )
        config = baseline_runtime_config
    elif terminal_capture_v23_authority_repair_validation:
        baseline_runtime_config = replace(
            baseline_runtime_config,
            goal_recovery_max_delta_v_mps=0.6,
            goal_recovery_terminal_effective_deceleration_mps2=1.697726,
            goal_recovery_terminal_stopping_margin_m=1.328009,
            goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
            goal_recovery_terminal_capture_v22_enabled=True,
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
            goal_recovery_terminal_capture_v23_enabled=True,
            goal_recovery_terminal_hold_target_radius_m=0.96,
        )
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError("v2.3 authority repair baseline runtime max_steps must be 100")
        config = baseline_runtime_config
    elif terminal_capture_v23_authority_repair_horizon_diagnostic:
        baseline_runtime_config = replace(
            baseline_runtime_config,
            goal_recovery_max_delta_v_mps=0.6,
            goal_recovery_terminal_effective_deceleration_mps2=1.697726,
            goal_recovery_terminal_stopping_margin_m=1.328009,
            goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
            goal_recovery_terminal_capture_v22_enabled=True,
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
            goal_recovery_terminal_capture_v23_enabled=True,
            goal_recovery_terminal_hold_target_radius_m=0.96,
        )
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError(
                "v2.3 authority-repair horizon baseline runtime max_steps must be 100"
            )
        config = replace(
            baseline_runtime_config,
            max_steps=HORIZON_DIAGNOSTIC_MAX_STEPS,
        )
    if terminal_capture_v2_horizon_diagnostic:
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError("v2 formal baseline runtime max_steps must be 100")
        config = replace(
            baseline_runtime_config,
            max_steps=HORIZON_DIAGNOSTIC_MAX_STEPS,
        )
    elif terminal_capture_v21_validation:
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError("v2.1 formal baseline runtime max_steps must be 100")
        config = replace(
            baseline_runtime_config,
            goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
        )
    elif terminal_capture_v22_validation:
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError("v2.2 formal baseline runtime max_steps must be 100")
        config = replace(
            baseline_runtime_config,
            goal_recovery_max_delta_v_mps=0.6,
            goal_recovery_terminal_effective_deceleration_mps2=1.697726,
            goal_recovery_terminal_stopping_margin_m=1.328009,
            goal_recovery_terminal_capture_v22_enabled=True,
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
        )
    elif terminal_capture_v23_validation:
        if int(baseline_runtime_config.max_steps) != HORIZON_BASELINE_MAX_STEPS:
            raise ValueError("v2.3 formal baseline runtime max_steps must be 100")
        config = replace(
            baseline_runtime_config,
            goal_recovery_max_delta_v_mps=0.6,
            goal_recovery_terminal_effective_deceleration_mps2=1.697726,
            goal_recovery_terminal_stopping_margin_m=1.328009,
            goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
            goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
            goal_recovery_terminal_capture_v22_enabled=True,
            goal_recovery_terminal_hold_enter_radius_m=0.98,
            goal_recovery_terminal_hold_release_radius_m=1.05,
            goal_recovery_terminal_hold_restorative_enabled=True,
            goal_recovery_terminal_capture_v23_enabled=True,
            goal_recovery_terminal_hold_target_radius_m=0.96,
        )
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    policy_hash_before = _state_dict_hash(trainer.policy.state_dict())
    checkpoint_identity = {
        **identity,
        "resolved_path": str(checkpoint_path),
        "sha256": checkpoint_sha256,
        "size_bytes": checkpoint_path.stat().st_size,
        "policy_class": f"{type(trainer.policy).__module__}.{type(trainer.policy).__qualname__}",
        "policy_hash_before": policy_hash_before,
        "observation_dimension": int(config.node_feature_dim),
        "normalizer_source": "checkpoint_payload",
        "normalizer_payload_keys": sorted(
            str(key) for key in payload if "normalizer" in str(key).lower()
        ),
        "checkpoint_payload_keys": sorted(str(key) for key in payload),
    }
    requested = FORMAL_EPISODES if formal else int(smoke_count)
    canonical_pool = _build_pool(
        baseline_runtime_config
        if (
            terminal_capture_v2_horizon_diagnostic
            or terminal_capture_v23_authority_repair_horizon_diagnostic
            or terminal_capture_v21_validation
            or terminal_capture_v22_validation
            or terminal_capture_v23_validation
            or terminal_capture_v23_authority_repair_validation
        )
        else config,
        split="test",
        normal_count=0,
        static_count=FORMAL_EPISODES,
        seed_base=STATIC_HOLDOUT_SEED_BASE,
        round_index=STATIC_HOLDOUT_ROUND,
    )
    pool = list(canonical_pool if formal else canonical_pool[:requested])
    if len(pool) != requested:
        raise AssertionError("canonical static holdout did not provide requested episodes")
    manifest = _manifest_payload(
        pool,
        config,
        checkpoint_identity,
        paired=condition == "paired",
    )
    baseline_normalized_manifest = (
        _manifest_payload(
            pool,
            baseline_runtime_config,
            checkpoint_identity,
            paired=False,
        )
        if terminal_capture_v2_horizon_diagnostic
        or terminal_capture_v23_authority_repair_horizon_diagnostic
        else None
    )
    current_episode_identity = [
        {
            "episode_index": int(item["episode_index"]),
            "episode_id": str(item["episode_id"]),
            "scenario_seed": int(item["scenario_seed"]),
            "noise_seed": int(item["noise_seed"]),
            "scenario_hash": _canonical_hash(item["scenario"]),
        }
        for item in pool
    ]
    identity_checks: dict[str, bool] = {}
    identity_reference = (
        v23_repair_horizon_baseline
        if terminal_capture_v23_authority_repair_horizon_diagnostic
        else
        v23_repair_baseline
        if terminal_capture_v23_authority_repair_validation
        else v23_baseline
        if terminal_capture_v23_validation
        else
        v21_baseline
        if terminal_capture_v21_validation
        else v22_baseline
        if terminal_capture_v22_validation
        else horizon_baseline
        if terminal_capture_v2_horizon_diagnostic
        else terminal_v1_baseline
        if terminal_capture_v2_validation
        else terminal_baseline if terminal_capture_validation else previous_formal
    )
    if terminal_capture_v23_authority_repair_horizon_diagnostic:
        identity_checks = _v23_authority_repair_horizon_identity_checks(
            baseline_evaluation=v23_repair_horizon_baseline,
            baseline_summary=v23_repair_horizon_baseline_summary,
            baseline_config=baseline_runtime_config,
            diagnostic_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            baseline_normalized_manifest_hash=baseline_normalized_manifest[
                "scenario_manifest_hash"
            ],
            current_episodes=current_episode_identity,
        )
    elif terminal_capture_v23_authority_repair_validation:
        identity_checks = _v23_authority_repair_identity_checks(
            baseline_evaluation=v23_repair_baseline,
            baseline_summary=v23_repair_baseline_summary,
            baseline_config=baseline_runtime_config,
            candidate_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            scenario_manifest_hash=manifest["scenario_manifest_hash"],
            current_episodes=current_episode_identity,
        )
    elif terminal_capture_v23_validation:
        identity_checks = _v23_only_identity_checks(
            baseline_evaluation=v23_baseline,
            baseline_summary=v23_baseline_summary,
            baseline_config=baseline_runtime_config,
            candidate_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            scenario_manifest_hash=manifest["scenario_manifest_hash"],
            current_episodes=current_episode_identity,
        )
    elif terminal_capture_v21_validation:
        identity_checks = _v21_only_identity_checks(
            baseline_evaluation=v21_baseline,
            baseline_summary=v21_baseline_summary,
            baseline_config=baseline_runtime_config,
            candidate_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            scenario_manifest_hash=manifest["scenario_manifest_hash"],
            current_episodes=current_episode_identity,
        )
    elif terminal_capture_v22_validation:
        identity_checks = _v22_only_identity_checks(
            baseline_evaluation=v22_baseline,
            baseline_summary=v22_baseline_summary,
            baseline_config=baseline_runtime_config,
            candidate_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            scenario_manifest_hash=manifest["scenario_manifest_hash"],
            current_episodes=current_episode_identity,
        )
    elif terminal_capture_v2_horizon_diagnostic:
        identity_checks = _horizon_only_identity_checks(
            baseline_evaluation=horizon_baseline,
            baseline_summary=horizon_baseline_summary,
            baseline_config=baseline_runtime_config,
            diagnostic_config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            baseline_normalized_manifest_hash=baseline_normalized_manifest[
                "scenario_manifest_hash"
            ],
            current_episodes=current_episode_identity,
        )
    elif identity_reference is not None:
        identity_checks = assert_previous_formal_identity(
            identity_reference,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_identity["sha256"],
            policy_hash=checkpoint_identity["policy_hash"],
            scenario_manifest_hash=manifest["scenario_manifest_hash"],
            current_episodes=current_episode_identity,
        )
    lateral_exit_baseline = None
    if lateral_exit_braking_assistance_diagnostic:
        from .single_uav_lateral_exit_braking_assistance import load_frozen_baseline

        lateral_exit_baseline = load_frozen_baseline()
    # In canonical paired mode, do not create even the ON directory until the
    # newly executed OFF trace has passed its immutable-baseline check.
    layouts = None if (
        (
            lateral_exit_braking_assistance_diagnostic
            or radial_intent_persistence_diagnostic
            or pass_commit_diagnostic
        )
        and "off" in modes
    ) else _prepare_output_layouts(
        data_root=Path(data_root), logs_root=Path(logs_root),
        plots_root=Path(plots_root), condition=condition, checkpoint=checkpoint_path,
    )
    results: dict[str, Any] = {}
    lateral_exit_controller_config_hash = _canonical_hash(
        {
            field.name: _jsonable(getattr(config, field.name))
            for field in fields(config)
            if field.name != "output_dir"
        }
    )
    lateral_exit_goal_recovery_config_hash = _canonical_hash(
        _effective_correction_config(config, enabled=True)
    )
    runtime_component_hashes = {
        "before_off_or_first_condition": _frozen_runtime_component_hashes(trainer)
    }
    for mode in modes:
        episode_summaries: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        for item in pool:
            radial_intent_controller = (
                radial_intent_assistance_type(
                    enabled=mode == "on",
                    cap_mps=float(
                        radial_intent_cap_derivation[
                            "max_radial_intent_assist_mps"
                        ]
                    ),
                )
                if radial_intent_persistence_diagnostic
                else None
            )
            pass_commit_controller = (
                PassCommitNonRegression(
                    enabled=mode == "on",
                    cap_mps=float(
                        pass_commit_cap_derivation["pass_commit_max_delta_v_mps"]
                    ),
                )
                if pass_commit_diagnostic
                else None
            )
            summary, rows = _rollout_frozen_actor_episode(
                trainer=trainer,
                config=config,
                item=item,
                correction_enabled=(
                    True
                    if lateral_exit_braking_assistance_diagnostic
                    or radial_intent_persistence_diagnostic
                    or pass_commit_diagnostic
                    else mode == "on"
                ),
                signed_observability=bool(signed_bypass_observability_diagnostic)
                or bool(lateral_exit_braking_assistance_diagnostic)
                or bool(radial_intent_persistence_diagnostic)
                or bool(pass_commit_diagnostic),
                lateral_exit_assistance_enabled=(
                    (mode == "on")
                    if lateral_exit_braking_assistance_diagnostic
                    else True if radial_intent_persistence_diagnostic
                    else True if pass_commit_diagnostic
                    else None
                ),
                radial_intent_assistance=radial_intent_controller,
                pass_commit_assistance=pass_commit_controller,
                condition_label=(
                    mode
                    if lateral_exit_braking_assistance_diagnostic
                    or radial_intent_persistence_diagnostic
                    or pass_commit_diagnostic
                    else None
                ),
            )
            if terminal_capture_v22_validation:
                summary = _augment_v22_episode_summary(summary, rows)
            if (
                terminal_capture_v23_validation
                or terminal_capture_v23_authority_repair_horizon_diagnostic
                or terminal_capture_v23_authority_repair_validation
            ):
                summary = _augment_v23_episode_summary(summary, rows)
            if (
                terminal_capture_v2_horizon_diagnostic
                or terminal_capture_v23_authority_repair_horizon_diagnostic
            ):
                summary = _augment_horizon_episode_summary(summary, rows)
            episode_summaries.append(summary)
            step_rows.extend(rows)
            if progress:
                print(
                    f"[{STAGE_NAME}] {mode} {item['episode_index']}/{len(pool)}",
                    flush=True,
                )
        aggregate = _aggregate_episode_summaries(episode_summaries)
        off_baseline_check = None
        if lateral_exit_braking_assistance_diagnostic and mode == "off":
            from .single_uav_lateral_exit_braking_assistance import (
                validate_off_against_frozen_baseline,
            )

            off_baseline_check = validate_off_against_frozen_baseline(
                current_episodes=episode_summaries,
                current_steps=step_rows,
                baseline=lateral_exit_baseline,
            )
            if layouts is None:
                layouts = _prepare_output_layouts(
                    data_root=Path(data_root), logs_root=Path(logs_root),
                    plots_root=Path(plots_root), condition=condition,
                    checkpoint=checkpoint_path,
                )
        radial_intent_off_baseline_check = None
        if radial_intent_persistence_diagnostic and mode == "off":
            from .single_uav_radial_intent_persistence_diagnostic import (
                validate_off_against_frozen_baseline as _validate_radial_intent_off,
            )

            radial_intent_off_baseline_check = _validate_radial_intent_off(
                current_rows=step_rows,
                baseline_contract=radial_intent_baseline_contract,
            )
            if layouts is None:
                layouts = _prepare_output_layouts(
                    data_root=Path(data_root), logs_root=Path(logs_root),
                    plots_root=Path(plots_root), condition=condition,
                    checkpoint=checkpoint_path,
                )
        pass_commit_off_baseline_check = None
        if pass_commit_diagnostic and mode == "off":
            from .single_uav_pass_commit_diagnostic import (
                validate_off_against_baseline as _validate_pass_commit_off,
            )

            pass_commit_off_baseline_check = _validate_pass_commit_off(
                current_rows=step_rows,
                current_episodes=episode_summaries,
                current_aggregate=aggregate,
                contract=pass_commit_baseline_contract,
            )
            if layouts is None:
                layouts = _prepare_output_layouts(
                    data_root=Path(data_root), logs_root=Path(logs_root),
                    plots_root=Path(plots_root), condition=condition,
                    checkpoint=checkpoint_path,
                )
        condition_provenance = {
            "scenario_manifest_hash": manifest["scenario_manifest_hash"],
            "ordered_pool_identity": _ordered_pool_identity(pool),
            "checkpoint": checkpoint_identity,
            "policy_hash": policy_hash_before,
            "observation_dimension": int(config.node_feature_dim),
            "controller_config_hash": lateral_exit_controller_config_hash,
            "goal_recovery_config_hash": lateral_exit_goal_recovery_config_hash,
            "correction_enabled": (
                True
                if lateral_exit_braking_assistance_diagnostic
                or radial_intent_persistence_diagnostic
                or pass_commit_diagnostic
                else mode == "on"
            ),
            "lateral_exit_braking_assistance_enabled": bool(
                (lateral_exit_braking_assistance_diagnostic and mode == "on")
                or radial_intent_persistence_diagnostic
                or pass_commit_diagnostic
            ),
        }
        if lateral_exit_braking_assistance_diagnostic:
            condition_provenance["lateral_exit_treatment_config"] = {
                "enabled": mode == "on",
                "assist_cap_mps": LATERAL_EXIT_ASSIST_CAP_MPS,
            }
        if radial_intent_persistence_diagnostic:
            condition_provenance.update(
                {
                    "radial_intent_assistance_enabled": bool(mode == "on"),
                    "radial_intent_assistance_config": {
                        "enabled": bool(mode == "on"),
                        "target_floor_mps": float(
                            radial_intent_cap_derivation["target_floor_mps"]
                        ),
                        "deadband_mps": float(
                            radial_intent_cap_derivation["deadband_mps"]
                        ),
                        "cap_mps": float(
                            radial_intent_cap_derivation[
                                "max_radial_intent_assist_mps"
                            ]
                        ),
                        "frame_source": radial_intent_cap_derivation[
                            "frozen_frame_source"
                        ],
                    },
                    "radial_intent_preregistration_hash": radial_intent_source_lock[
                        "new_radial_intent_preregistration_hash"
                    ],
                    "radial_intent_off_baseline_fingerprint": radial_intent_baseline_contract[
                        "trace_sha256"
                    ],
                }
            )
        if pass_commit_diagnostic:
            condition_provenance.update(
                {
                    "pass_commit_attempt_id": pass_commit_source_lock["attempt_id"],
                    "pass_commit_enabled": bool(mode == "on"),
                    "pass_commit_radial_intent_assistance_enabled": False,
                    "pass_commit_lateral_exit_braking_assistance_enabled": True,
                    "pass_commit_cap_mps": float(
                        pass_commit_cap_derivation["pass_commit_max_delta_v_mps"]
                    ),
                    "pass_commit_preregistration_hash": pass_commit_source_lock[
                        "active_pass_commit_preregistration_hash"
                    ],
                    "pass_commit_off_baseline_fingerprint": pass_commit_baseline_contract[
                        "fingerprint"
                    ],
                }
            )
            if pass_commit_off_baseline_check is not None:
                condition_provenance["pass_commit_off_baseline_check"] = (
                    pass_commit_off_baseline_check
                )
        if radial_intent_off_baseline_check is not None:
            condition_provenance["radial_intent_off_baseline_check"] = (
                radial_intent_off_baseline_check
            )
        if off_baseline_check is not None:
            condition_provenance["off_baseline_check"] = off_baseline_check
        if layouts is None:
            raise AssertionError("output layout not initialized after OFF validation")
        _write_condition_outputs(
            data_root=layouts["data"][mode],
            logs_root=layouts["logs"][mode],
            plots_root=layouts["plots"][mode],
            condition=mode,
            episode_rows=episode_summaries,
            step_rows=step_rows,
            aggregate=aggregate,
            provenance=condition_provenance,
        )
        results[mode] = {
            "episodes": episode_summaries,
            "aggregate": aggregate,
            "provenance": condition_provenance,
        }
        if (
            signed_bypass_observability_diagnostic
            or lateral_exit_braking_assistance_diagnostic
            or radial_intent_persistence_diagnostic
            or pass_commit_diagnostic
        ):
            results[mode]["step_rows"] = step_rows
        current_runtime_hashes = _frozen_runtime_component_hashes(trainer)
        runtime_component_hashes[f"after_{mode}"] = current_runtime_hashes
        if (
            (
                lateral_exit_braking_assistance_diagnostic
                or radial_intent_persistence_diagnostic
                or pass_commit_diagnostic
            )
            and current_runtime_hashes
            != runtime_component_hashes["before_off_or_first_condition"]
        ):
            raise AssertionError("frozen Actor/Critic/GAT/normalizer state changed")
    radial_intent_identity_checks: dict[str, Any] = {}
    if radial_intent_persistence_diagnostic:
        from .single_uav_radial_intent_persistence_diagnostic import (
            assert_paired_identity as _assert_radial_intent_paired_identity,
        )

        _assert_radial_intent_paired_identity(
            off_rows=results["off"]["episodes"],
            on_rows=results["on"]["episodes"],
            off_component_hashes=runtime_component_hashes["after_off"],
            on_component_hashes=runtime_component_hashes["after_on"],
            off_provenance=results["off"]["provenance"],
            on_provenance=results["on"]["provenance"],
        )
        radial_intent_identity_checks = {
            "off_baseline_fingerprint_match": bool(
                results["off"]["provenance"].get(
                    "radial_intent_off_baseline_check", {}
                ).get("status") == "PASS"
            ),
            "same_actor_critic_gat_normalizer_hashes": (
                runtime_component_hashes["after_off"]
                == runtime_component_hashes["after_on"]
            ),
            "same_ordered_episode_seed_checkpoint_runtime_identity": True,
            "both_arms_goal_recovery_enabled": True,
            "both_arms_lateral_exit_assistance_enabled": True,
            "only_radial_intent_treatment_differs": True,
        }
    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    if policy_hash_after != policy_hash_before:
        raise AssertionError("frozen Actor parameters changed during evaluation")
    if trainer.policy.training or any(
        parameter.requires_grad for parameter in trainer.policy.parameters()
    ):
        raise AssertionError("frozen Actor left evaluation/frozen mode")
    radial_intent_itt_summary = None
    if radial_intent_persistence_diagnostic:
        from .single_uav_radial_intent_persistence_diagnostic import (
            paired_itt_endpoint_summary as _radial_intent_paired_itt_endpoint_summary,
        )

        radial_intent_itt_summary = _radial_intent_paired_itt_endpoint_summary(
            off_episodes=results["off"]["episodes"],
            on_episodes=results["on"]["episodes"],
            off_steps=results["off"]["step_rows"],
            on_steps=results["on"]["step_rows"],
        )
    effective_correction_config = _effective_correction_config(
        config,
        enabled=(
            True
            if lateral_exit_braking_assistance_diagnostic
            or radial_intent_persistence_diagnostic
            or pass_commit_diagnostic
            else condition == "on"
        ),
    )
    output: dict[str, Any] = {
        "stage": STAGE_NAME,
        "condition": condition,
        "formal": bool(formal),
        "episode_count": len(pool),
        "checkpoint": {**checkpoint_identity, "policy_hash_after": policy_hash_after},
        "scenario_manifest_hash": manifest["scenario_manifest_hash"],
        "ordered_pool_identity": _ordered_pool_identity(pool),
        "actual_stable_validation": bool(actual_stable_validation),
        "terminal_capture_validation": bool(terminal_capture_validation),
        "terminal_capture_v2_validation": bool(terminal_capture_v2_validation),
        "terminal_capture_v2_horizon_diagnostic": bool(
            terminal_capture_v2_horizon_diagnostic
        ),
        "terminal_capture_v23_authority_repair_horizon_diagnostic": bool(
            terminal_capture_v23_authority_repair_horizon_diagnostic
        ),
        "terminal_capture_v21_validation": bool(terminal_capture_v21_validation),
        "terminal_capture_v22_validation": bool(terminal_capture_v22_validation),
        "terminal_capture_v23_validation": bool(terminal_capture_v23_validation),
        "terminal_capture_v23_authority_repair_validation": bool(
            terminal_capture_v23_authority_repair_validation
        ),
        "signed_bypass_observability_diagnostic": bool(
            signed_bypass_observability_diagnostic
        ),
        "lateral_exit_braking_assistance_diagnostic": bool(
            lateral_exit_braking_assistance_diagnostic
        ),
        "radial_intent_persistence_diagnostic": bool(
            radial_intent_persistence_diagnostic
        ),
        "pass_commit_diagnostic": bool(pass_commit_diagnostic),
        "pass_commit_diagnostic_prepared": bool(pass_commit_diagnostic),
        "pass_commit_diagnostic_executed": bool(pass_commit_diagnostic),
        "pass_commit_parent_radial_intent_preregistration_hash_identity": (
            pass_commit_source_lock["parent_radial_intent_diagnostic_identity"]
            if pass_commit_source_lock is not None else None
        ),
        "pass_commit_new_preregistration_hash": (
            pass_commit_source_lock["active_pass_commit_preregistration_hash"]
            if pass_commit_source_lock is not None else None
        ),
        "pass_commit_off_baseline_fingerprint": (
            pass_commit_baseline_contract["fingerprint"]
            if pass_commit_baseline_contract is not None else None
        ),
        "pass_commit_cap_derivation": pass_commit_cap_derivation,
        "pass_commit_runtime_source_hashes": (
            pass_commit_source_lock["source_hashes"]
            if pass_commit_source_lock is not None else None
        ),
        "radial_intent_diagnostic_prepared": bool(
            radial_intent_persistence_diagnostic
        ),
        "radial_intent_diagnostic_executed": bool(
            radial_intent_persistence_diagnostic
        ),
        "radial_intent_parent_preregistration_hash": (
            radial_intent_source_lock["parent_lateral_exit_preregistration_hash"]
            if radial_intent_source_lock is not None else None
        ),
        "radial_intent_new_preregistration_hash": (
            radial_intent_source_lock["new_radial_intent_preregistration_hash"]
            if radial_intent_source_lock is not None else None
        ),
        "radial_intent_off_baseline_fingerprint": (
            radial_intent_baseline_contract["trace_sha256"]
            if radial_intent_baseline_contract is not None else None
        ),
        "radial_intent_cap_derivation": radial_intent_cap_derivation,
        "radial_intent_runtime_source_hashes": (
            radial_intent_source_lock["source_hashes"]
            if radial_intent_source_lock is not None else None
        ),
        "radial_intent_identity_checks": radial_intent_identity_checks,
        "radial_intent_paired_itt_endpoint_summary": radial_intent_itt_summary,
        "frozen_runtime_component_hashes": runtime_component_hashes,
        "previous_formal_identity_checks": identity_checks,
        "same_checkpoint_as_previous_formal": bool(identity_checks) and all(
            identity_checks[key] for key in identity_checks if "checkpoint" in key or "policy" in key
        ),
        "same_scenario_manifest_as_previous_formal": bool(identity_checks) and bool(
            (
                identity_checks.get("same_scenario_manifest_as_previous_formal")
                or identity_checks.get("same_scenario_manifest")
            )
            and (
                identity_checks.get("same_ordered_episode_identity_as_previous_formal")
                or identity_checks.get("same_ordered_episode_identity")
            )
        ),
        "previous_formal_checkpoint": (
            previous_formal.get("checkpoint") if previous_formal is not None else None
        ),
        "previous_formal_scenario_manifest_hash": (
            previous_formal.get("scenario_manifest_hash")
            if previous_formal is not None else None
        ),
        "effective_goal_recovery_correction_config": effective_correction_config,
        "effective_goal_recovery_correction_config_hash": _canonical_hash(
            effective_correction_config
        ),
        "results": results,
        **EVALUATION_ONLY_FLAGS,
    }
    _write_json(layouts["data"]["base"] / "scenario_manifest.json", _jsonable(manifest))
    if condition == "paired":
        off_episodes = results["off"]["episodes"]
        on_episodes = results["on"]["episodes"]
        _assert_paired_identity(off_episodes, on_episodes)
        if (
            results["off"]["provenance"]["scenario_manifest_hash"]
            != results["on"]["provenance"]["scenario_manifest_hash"]
        ):
            raise AssertionError("paired scenario manifest hashes differ")
        comparison = _comparison_rows(
            results["off"]["aggregate"], results["on"]["aggregate"]
        )
        paired_episodes = _paired_episode_rows(off_episodes, on_episodes)
        _write_rows_csv(
            layouts["logs"]["base"] / "goal_recovery_correction_comparison.csv",
            comparison,
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "comparison_episodes.csv",
            paired_episodes,
        )
        comparison_payload = {
            **output,
            "paired_same_scenarios": True,
            "comparison": comparison,
            "off_checkpoint_sha256": checkpoint_identity["sha256"],
            "on_checkpoint_sha256": checkpoint_identity["sha256"],
        }
        _write_json(
            layouts["logs"]["base"] / "goal_recovery_correction_comparison.json",
            _jsonable(comparison_payload),
        )
        output["paired_same_scenarios"] = True
        output["comparison"] = comparison
    if actual_stable_validation:
        episodes = results["on"]["episodes"]
        validation_summary = {
            "stage": "single_uav_goal_recovery_actual_stable_validation",
            **aggregate_actual_stable_validation(episodes),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "policy_hash": checkpoint_identity["policy_hash"],
            "scenario_manifest_hash": manifest["scenario_manifest_hash"],
            "previous_scenario_manifest_hash": previous_formal.get(
                "scenario_manifest_hash"
            ),
            "effective_goal_recovery_correction_config": effective_correction_config,
            "effective_goal_recovery_correction_config_hash": _canonical_hash(
                effective_correction_config
            ),
            "identity_checks": identity_checks,
            "same_checkpoint_as_previous_formal": output["same_checkpoint_as_previous_formal"],
            "same_scenario_manifest_as_previous_formal": output["same_scenario_manifest_as_previous_formal"],
            "stable_log_schema_version": STABLE_LOG_SCHEMA_VERSION,
            **EVALUATION_ONLY_FLAGS,
        }
        _write_json(
            layouts["logs"]["base"] / "actual_stable_validation_summary.json",
            _jsonable(validation_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "actual_stable_validation_episodes.csv",
            episodes,
        )
        output["actual_stable_validation_summary"] = validation_summary
    if terminal_capture_validation:
        episodes = results["on"]["episodes"]
        canonical_candidate = (
            float(config.goal_recovery_terminal_capture_radius_m) == 8.0
            and float(config.goal_recovery_terminal_radial_speed_floor_mps) == 0.25
            and float(config.goal_recovery_terminal_radial_brake_gain) == 0.8
            and float(config.goal_recovery_terminal_tangential_gain) == 1.0
        )
        terminal_summary = {
            "stage": "single_uav_terminal_capture_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            "baseline_reference_path": str(terminal_baseline_path),
            "baseline_reference_sha256": terminal_baseline_sha256,
            "baseline_checkpoint_sha256": terminal_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": terminal_baseline["scenario_manifest_hash"],
            "identity_checks": identity_checks,
            "identity_match": bool(identity_checks) and all(identity_checks.values()),
            "candidate_canonical_defaults": canonical_candidate,
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "control_algorithm_modified": True,
            "terminal_capture_added": True,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_validation_summary.json",
            _jsonable(terminal_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_validation_summary"] = terminal_summary
    if terminal_capture_v2_validation:
        episodes = results["on"]["episodes"]
        terminal_v2_summary = {
            "stage": "single_uav_terminal_capture_v2_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            "baseline_reference_path": str(terminal_v1_baseline_path),
            "baseline_reference_sha256": terminal_v1_baseline_sha256,
            "baseline_checkpoint_sha256": terminal_v1_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": terminal_v1_baseline["scenario_manifest_hash"],
            "identity_checks": identity_checks,
            "identity_match": bool(identity_checks) and all(identity_checks.values()),
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "source_control_algorithm_modified": True,
            "control_algorithm_modified": True,
            "terminal_capture_v2_added": True,
            "actor_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "action_transform_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "stable_goal_criterion_modified": False,
            "obstacle_logic_modified": False,
            "max_correction_modified": False,
            "training_performed": False,
            "ppo_updates": 0,
            "mappo_updates": 0,
            "bc_retraining_executed": False,
            "dagger_collection_executed": False,
            "teacher_rollout_executed": False,
            "formal_evaluation_executed_by_codex": False,
            "formal_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed": False,
            "rollout_executed_by_codex": False,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_v2_validation_summary.json",
            _jsonable(terminal_v2_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_v2_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_v2_validation_summary"] = terminal_v2_summary
    if terminal_capture_v21_validation:
        episodes = results["on"]["episodes"]
        v21_summary = {
            "stage": "single_uav_terminal_capture_v21_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "baseline_reference_path": str(v21_baseline_path),
            "baseline_reference_sha256": v21_baseline_sha256,
            "baseline_checkpoint_sha256": v21_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": v21_baseline["scenario_manifest_hash"],
            "candidate_manifest_hash": manifest["scenario_manifest_hash"],
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "source_control_algorithm_modified": True,
            "terminal_capture_v21_added": True,
            "goal_recovery_controller_modified": True,
            "actor_network_modified": False,
            "actor_parameters_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "actor_raw_action_rewritten": False,
            "action_transform_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "stable_goal_criterion_modified": False,
            "obstacle_logic_modified": False,
            "max_correction_modified": False,
            "terminal_v2_governor_modified": False,
            "a_eff_modified": False,
            "stopping_margin_modified": False,
            "training_performed": False,
            "ppo_updates": 0,
            "mappo_updates": 0,
            "bc_retraining_executed": False,
            "dagger_collection_executed": False,
            "teacher_rollout_executed": False,
            "formal_evaluation_executed_by_codex": False,
            "formal_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed": False,
            "rollout_executed_by_codex": False,
            "offline_conflict_audit_executed": True,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_v21_validation_summary.json",
            _jsonable(v21_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_v21_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_v21_validation_summary"] = v21_summary
    if terminal_capture_v22_validation:
        episodes = results["on"]["episodes"]
        v22_summary = {
            "stage": "single_uav_terminal_capture_v22_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "baseline_reference_path": str(v22_baseline_path),
            "baseline_reference_sha256": v22_baseline_sha256,
            "baseline_checkpoint_sha256": v22_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": v22_baseline["scenario_manifest_hash"],
            "candidate_manifest_hash": manifest["scenario_manifest_hash"],
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "source_control_algorithm_modified": True,
            "terminal_capture_v22_added": True,
            "terminal_capture_v21_enabled": True,
            "terminal_capture_v2_enabled": True,
            "terminal_v21_modified": False,
            "terminal_v2_governor_modified": False,
            "goal_recovery_controller_modified": True,
            "terminal_hold_enter_radius_m": 0.98,
            "terminal_hold_release_radius_m": 1.05,
            "terminal_hold_restorative_enabled": True,
            "actor_modified": False,
            "actor_network_modified": False,
            "actor_parameters_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "action_transform_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "stable_goal_criterion_modified": False,
            "obstacle_logic_modified": False,
            "max_correction_modified": False,
            "a_eff_modified": False,
            "stopping_margin_modified": False,
            "training_performed": False,
            "ppo_updates": 0,
            "mappo_updates": 0,
            "bc_retraining_executed": False,
            "dagger_collection_executed": False,
            "teacher_rollout_executed": False,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed": False,
            "offline_conflict_audit_executed": True,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_v22_validation_summary.json",
            _jsonable(v22_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_v22_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_v22_validation_summary"] = v22_summary
    if terminal_capture_v23_validation:
        episodes = results["on"]["episodes"]
        v23_summary = {
            "stage": "single_uav_terminal_capture_v23_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "baseline_reference_path": str(v23_baseline_path),
            "baseline_reference_sha256": v23_baseline_sha256,
            "baseline_checkpoint_sha256": v23_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": v23_baseline["scenario_manifest_hash"],
            "candidate_manifest_hash": manifest["scenario_manifest_hash"],
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "source_control_algorithm_modified": True,
            "terminal_capture_v23_added": True,
            "terminal_capture_v22_enabled": True,
            "terminal_capture_v21_enabled": True,
            "terminal_capture_v2_enabled": True,
            "terminal_v22_modified": False,
            "terminal_v21_modified": False,
            "terminal_v2_governor_modified": False,
            "goal_recovery_controller_modified": True,
            "terminal_hold_target_radius_m": 0.96,
            "actor_modified": False,
            "actor_network_modified": False,
            "actor_parameters_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "action_transform_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "stable_goal_criterion_modified": False,
            "obstacle_logic_modified": False,
            "max_correction_modified": False,
            "a_eff_modified": False,
            "stopping_margin_modified": False,
            "training_performed": False,
            "ppo_updates": 0,
            "mappo_updates": 0,
            "bc_retraining_executed": False,
            "dagger_collection_executed": False,
            "teacher_rollout_executed": False,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed": False,
            "offline_target_guard_audit_executed": True,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_v23_validation_summary.json",
            _jsonable(v23_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_v23_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_v23_validation_summary"] = v23_summary
    if terminal_capture_v23_authority_repair_validation:
        episodes = results["on"]["episodes"]
        repair_summary = {
            "stage": "single_uav_terminal_capture_v23_authority_repair_validation",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "baseline_reference_path": str(v23_repair_baseline_path),
            "baseline_reference_sha256": v23_repair_baseline_sha256,
            "baseline_checkpoint_sha256": v23_repair_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": v23_repair_baseline["scenario_manifest_hash"],
            "candidate_manifest_hash": manifest["scenario_manifest_hash"],
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "source_control_algorithm_modified": True,
            "terminal_capture_v23_authority_repaired": True,
            "terminal_capture_v24_added": False,
            "terminal_v23_target_modified": False,
            "terminal_v22_modified": False,
            "terminal_v21_modified": False,
            "terminal_v2_governor_modified": False,
            "actor_network_modified": False,
            "actor_parameters_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "obstacle_logic_modified": False,
            "stable_goal_criterion_modified": False,
            "action_transform_modified": False,
            "max_correction_modified": False,
            "a_eff_modified": False,
            "stopping_margin_modified": False,
            "training_performed": False,
            "rollout_executed": True,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "offline_authority_audit_executed": True,
        }
        _write_json(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_validation_summary.json",
            _jsonable(repair_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_validation_episodes.csv",
            episodes,
        )
        output["terminal_capture_v23_authority_repair_validation_summary"] = (
            repair_summary
        )
    if terminal_capture_v2_horizon_diagnostic:
        episodes = results["on"]["episodes"]
        recovery_rows = _build_horizon_recovery_rows(
            horizon_baseline_episodes,
            episodes,
        )
        horizon_summary = {
            "stage": "single_uav_terminal_capture_v2_horizon120_diagnostic",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "diagnostic_horizon_only": True,
            "baseline_reference_path": str(horizon_baseline_path),
            "baseline_reference_sha256": horizon_baseline_sha256,
            "baseline_checkpoint_sha256": horizon_baseline["checkpoint"]["sha256"],
            "baseline_manifest_hash": horizon_baseline["scenario_manifest_hash"],
            "diagnostic_manifest_hash": manifest["scenario_manifest_hash"],
            "baseline_normalized_manifest_hash": baseline_normalized_manifest[
                "scenario_manifest_hash"
            ],
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "recovery_episode_count": len(recovery_rows),
            "matched_recovery_episode_count": len(recovery_rows),
            "recovery_episodes_entering_extension": sum(
                int(row["post_100_steps_used"]) > 0 for row in episodes
                if int(row["passed_obstacle"]) and int(row["post_bypass_recovery"])
            ),
            "recovery_episodes_entered_1m_after_step_100": sum(
                int(row["entered_1m_after_step_100"]) for row in episodes
                if int(row["passed_obstacle"]) and int(row["post_bypass_recovery"])
            ),
            "recovery_episodes_stable_by_120": sum(
                int(row["stable_goal"]) for row in episodes
                if int(row["passed_obstacle"]) and int(row["post_bypass_recovery"])
            ),
            "recovery_episodes_unstable_by_120": sum(
                not int(row["stable_goal"]) for row in episodes
                if int(row["passed_obstacle"]) and int(row["post_bypass_recovery"])
            ),
            "recovery_episode_comparison": recovery_rows,
            "control_algorithm_modified": False,
            "goal_recovery_controller_modified": False,
            "terminal_capture_v2_controller_modified": False,
            "actor_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "stable_goal_criterion_modified": False,
            "action_transform_modified": False,
            "max_correction_modified": False,
            "training_performed": False,
            "ppo_updates": 0,
            "mappo_updates": 0,
            "bc_retraining_executed": False,
            "dagger_collection_executed": False,
            "teacher_rollout_executed": False,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed": False,
        }
        _write_json(
            layouts["logs"]["base"] / "terminal_capture_v2_horizon120_summary.json",
            _jsonable(horizon_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"] / "terminal_capture_v2_horizon120_episodes.csv",
            episodes,
        )
        _write_rows_csv(
            layouts["logs"]["base"]
            / "terminal_capture_v2_horizon120_recovery_episodes.csv",
            recovery_rows,
        )
        output["terminal_capture_v2_horizon120_summary"] = horizon_summary
    if (
        terminal_capture_v23_authority_repair_horizon_diagnostic
        and not lateral_exit_braking_assistance_diagnostic
    ):
        episodes = results["on"]["episodes"]
        recovery_rows = _build_horizon_recovery_rows(
            v23_repair_horizon_baseline_episodes,
            episodes,
        )
        focus_rows = [
            row for row in recovery_rows
            if str(row["episode_id"]) in HORIZON_FOCUS_EPISODE_IDS
        ]
        recovery_episode_ids = {
            str(row["episode_id"]) for row in recovery_rows
        }
        extension_rows = [
            row for row in episodes
            if int(row.get("extension_steps_observed", 0)) > 0
        ]
        repair_horizon_summary = {
            "stage": "single_uav_terminal_capture_v23_authority_repair_horizon120_diagnostic",
            **EVALUATION_ONLY_FLAGS,
            **results["on"]["aggregate"],
            **aggregate_actual_stable_validation(episodes),
            **identity_checks,
            "horizon120_diagnostic_prepared": True,
            "diagnostic_horizon_only": True,
            "baseline_max_steps": HORIZON_BASELINE_MAX_STEPS,
            "candidate_max_steps": HORIZON_DIAGNOSTIC_MAX_STEPS,
            "max_steps_baseline": HORIZON_BASELINE_MAX_STEPS,
            "max_steps_candidate": HORIZON_DIAGNOSTIC_MAX_STEPS,
            "only_horizon_changed": bool(identity_checks["only_horizon_changed"]),
            "baseline_reference_path": str(v23_repair_horizon_baseline_path),
            "baseline_reference_sha256": v23_repair_horizon_baseline_sha256,
            "baseline_summary_path": str(v23_repair_horizon_baseline_path),
            "baseline_summary_sha256": v23_repair_horizon_baseline_sha256,
            "baseline_checkpoint_sha256": v23_repair_horizon_baseline["checkpoint"]["sha256"],
            "baseline_policy_hash": v23_repair_horizon_baseline["checkpoint"]["policy_hash"],
            "baseline_manifest_hash": v23_repair_horizon_baseline["scenario_manifest_hash"],
            "baseline_normalized_manifest_hash": baseline_normalized_manifest[
                "scenario_manifest_hash"
            ],
            "diagnostic_manifest_hash": manifest["scenario_manifest_hash"],
            "baseline_ordered_episode_identity": v23_repair_horizon_baseline.get(
                "ordered_pool_identity", []
            ),
            "diagnostic_ordered_episode_identity": _ordered_pool_identity(pool),
            "candidate_config": effective_correction_config,
            "candidate_config_hash": _canonical_hash(effective_correction_config),
            "stable_distance_threshold": float(config.single_uav_goal_threshold_m),
            "stable_speed_threshold": float(config.single_uav_goal_speed_threshold_mps),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "recovery_episode_count": len(recovery_rows),
            "focus_episode_ids": list(HORIZON_FOCUS_EPISODE_IDS),
            "focus_recovery_episode_count": len(focus_rows),
            "focus_recovery_episode_comparison": focus_rows,
            "recovery_episode_comparison": recovery_rows,
            "recovery_episode_ids": sorted(recovery_episode_ids),
            "recovery_episodes_entered_extension": sum(
                int(row.get("extension_steps_observed", 0)) > 0
                for row in episodes
                if int(row.get("passed_obstacle", 0))
                and int(row.get("post_bypass_recovery", 0))
            ),
            "recovery_episodes_entered_1m_in_extension": sum(
                int(row.get("entered1_in_extension_101_120", 0))
                for row in episodes
                if int(row.get("passed_obstacle", 0))
                and int(row.get("post_bypass_recovery", 0))
            ),
            "recovery_episodes_stable_in_extension": sum(
                int(row.get("stable_in_extension_101_120", 0))
                for row in episodes
                if int(row.get("passed_obstacle", 0))
                and int(row.get("post_bypass_recovery", 0))
            ),
            "extension_observed_episode_count": len(extension_rows),
            "extension_total_collision_count": sum(
                int(row.get("extension_collision_count", 0)) for row in episodes
            ),
            "extension_total_safety_projection_steps": sum(
                int(row.get("extension_safety_projection_steps", 0)) for row in episodes
            ),
            "extension_total_boundary_projection_steps": sum(
                int(row.get("extension_boundary_projection_steps", 0)) for row in episodes
            ),
            "extension_total_action_clipping_steps": sum(
                int(row.get("extension_action_clipping_steps", 0)) for row in episodes
            ),
            "extension_total_terminal_saturation_steps": sum(
                int(row.get("extension_terminal_saturation_steps", 0)) for row in episodes
            ),
            "controller_code_modified": False,
            "terminal_capture_v23_modified": False,
            "authority_repair_modified": False,
            "actor_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "action_transform_modified": False,
            "stable_criterion_modified": False,
            "training_performed": False,
            "formal_evaluation_executed_by_codex": False,
            "smoke_evaluation_executed_by_codex": False,
            "rollout_executed_by_codex": False,
            "formal_executed_by_codex": False,
            "smoke_executed_by_codex": False,
            "source_logs_modified": False,
        }
        _write_json(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_horizon120_summary.json",
            _jsonable(repair_horizon_summary),
        )
        _write_rows_csv(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_horizon120_episodes.csv",
            episodes,
        )
        _write_rows_csv(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_horizon120_recovery_episodes.csv",
            recovery_rows,
        )
        _write_rows_csv(
            layouts["logs"]["base"]
            / "terminal_capture_v23_authority_repair_horizon120_focus_recovery_episodes.csv",
            focus_rows,
        )
        output[
            "terminal_capture_v23_authority_repair_horizon120_summary"
        ] = repair_horizon_summary
        output.update(
            {
                "horizon120_diagnostic_prepared": True,
                "max_steps_baseline": HORIZON_BASELINE_MAX_STEPS,
                "max_steps_candidate": HORIZON_DIAGNOSTIC_MAX_STEPS,
                "only_horizon_changed": True,
                "controller_code_modified": False,
                "terminal_capture_v23_modified": False,
                "authority_repair_modified": False,
                "rollout_executed_by_codex": False,
                "source_logs_modified": False,
            }
        )
    if signed_bypass_observability_diagnostic:
        baseline_root = Path(TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT)
        baseline_steps_path = _find_output_artifact(baseline_root, "steps.csv")
        baseline_evaluation_path = _find_output_artifact(
            baseline_root, "evaluation_summary.json"
        )
        signed_payload = _write_signed_bypass_outputs(
            output_root=Path(logs_root).resolve(),
            episode_rows=results["on"]["episodes"],
            step_rows=results["on"]["step_rows"],
            evaluation=output,
            baseline_evaluation=json.loads(
                baseline_evaluation_path.read_text(encoding="utf-8")
            ),
            baseline_step_rows=_read_csv_rows(baseline_steps_path),
        )
        output["signed_bypass_observability_summary"] = signed_payload
        output.update(
            {
                "signed_bypass_observability_prepared": True,
                "behavior_algorithm_modified": False,
                "controller_modified": False,
                "observation_modified": False,
                "reward_modified": False,
                "dynamics_modified": False,
                "action_transform_modified": False,
                "safety_projection_modified": False,
                "boundary_projection_modified": False,
                "goal_recovery_modified": False,
                "only_logging_changed": bool(
                    signed_payload.get("only_logging_changed", False)
                ),
            }
        )
    if lateral_exit_braking_assistance_diagnostic:
        from .single_uav_lateral_exit_braking_assistance import (
            NOT_INDEPENDENTLY_VERIFIED,
            assert_paired_identity_report,
            build_paired_identity_report,
            write_diagnostic_summary,
        )

        if "off" in results and "on" in results:
            identity_report = build_paired_identity_report(
                off_result=results["off"],
                on_result=results["on"],
                runtime_hashes=runtime_component_hashes,
                expected_observation_dimension=int(config.node_feature_dim),
                expected_max_steps=HORIZON_DIAGNOSTIC_MAX_STEPS,
            )
            assert_paired_identity_report(identity_report)
        else:
            identity_report = {
                "identity_status": {
                    "paired_identity": NOT_INDEPENDENTLY_VERIFIED,
                    "reason": "paired OFF and ON results are both required",
                },
                "fail_fast_boolean_guards": {},
                "only_lateral_exit_braking_assistance_changed": False,
            }
        if layouts is None:
            raise AssertionError("lateral-exit output layout was not initialized")
        lateral_summary = write_diagnostic_summary(
            output_root=layouts["logs"]["base"],
            results=results,
            identity_report=identity_report,
            runtime_hashes=runtime_component_hashes,
        )
        output["lateral_exit_braking_assistance_summary"] = lateral_summary
        output.update(
            {
                "lateral_exit_braking_diagnostic_prepared": True,
                "lateral_exit_assist_cap_mps": LATERAL_EXIT_ASSIST_CAP_MPS,
                "only_lateral_exit_braking_assistance_changed": bool(
                    identity_report.get(
                        "only_lateral_exit_braking_assistance_changed", False
                    )
                ),
                "actor_modified": False,
                "reward_modified": False,
                "dynamics_modified": False,
                "goal_recovery_modified": False,
                "terminal_controller_modified": False,
                "safety_modified": False,
                "boundary_modified": False,
            }
        )
    if pass_commit_diagnostic:
        if layouts is None or "off" not in results or "on" not in results:
            raise AssertionError("PASS_COMMIT requires a validated paired OFF/ON result")
        from ..pass_commit_finalization import build_pass_commit_paired_summary

        off_baseline_check = results["off"].get("provenance", {}).get(
            "pass_commit_off_baseline_check", {}
        )
        if off_baseline_check.get("status") != "PASS":
            raise AssertionError("PASS_COMMIT_OFF_BASELINE_MISMATCH")
        output["pass_commit_paired_itt_summary"] = build_pass_commit_paired_summary(
            results,
            preregistration=pass_commit_preregistration,
            attempt_id=pass_commit_source_lock["attempt_id"],
            off_baseline_fingerprint_match=True,
        )
    _write_json(
        layouts["logs"]["base"] / "evaluation_summary.json", _jsonable(output)
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--previous-formal-root", type=Path, default=PREVIOUS_FORMAL_LOG_ROOT)
    parser.add_argument(
        "--actual-stable-validation",
        action="store_true",
        help="locked formal20 correction-ON validation with canonical stable logging",
    )
    parser.add_argument(
        "--terminal-capture-validation",
        action="store_true",
        help="locked formal20 correction-ON validation with terminal capture enabled",
    )
    parser.add_argument(
        "--terminal-capture-v2-validation",
        action="store_true",
        help="locked formal20 correction-ON terminal-capture v2 validation",
    )
    parser.add_argument(
        "--terminal-capture-v2-horizon-diagnostic",
        action="store_true",
        help="locked formal20 v2 diagnostic with max_steps extended from 100 to 120",
    )
    parser.add_argument(
        "--terminal-capture-v23-authority-repair-horizon120-diagnostic",
        action="store_true",
        help=(
            "locked formal20 authority-repaired v2.3 diagnostic with max_steps "
            "extended from 100 to 120"
        ),
    )
    parser.add_argument(
        "--single-uav-signed-bypass-observability-diagnostic",
        action="store_true",
        help=(
            "prepare/run the locked max120 authority-repaired evaluation with "
            "signed bypass observability logging only"
        ),
    )
    parser.add_argument(
        "--single-uav-pass-commit-diagnostic",
        action="store_true",
        help="locked formal paired20 PASS_COMMIT diagnostic; requires --formal",
    )
    parser.add_argument(
        "--terminal-capture-v21-validation",
        action="store_true",
        help="locked formal20/max100 validation of v2.1 radial conflict arbitration",
    )
    parser.add_argument(
        "--terminal-capture-v22-validation",
        action="store_true",
        help="locked formal20/max100 validation of v2.2 hysteretic restorative hold",
    )
    parser.add_argument(
        "--terminal-capture-v23-validation",
        action="store_true",
        help="locked formal20/max100 validation of v2.3 target-radius guard",
    )
    parser.add_argument(
        "--terminal-capture-v23-authority-repair-validation",
        action="store_true",
        help="locked formal20/max100 validation of the v2.3 radial-authority repair",
    )
    parser.add_argument("--terminal-baseline-root", type=Path, default=ACTUAL_STABLE_BASELINE_ROOT)
    parser.add_argument(
        "--terminal-capture-v1-baseline-root",
        type=Path,
        default=TERMINAL_CAPTURE_V1_BASELINE_ROOT,
    )
    parser.add_argument(
        "--terminal-capture-v2-formal-baseline-root",
        type=Path,
        default=TERMINAL_CAPTURE_V2_FORMAL_BASELINE_ROOT,
    )
    parser.add_argument(
        "--terminal-capture-v21-baseline-root",
        type=Path,
        default=TERMINAL_CAPTURE_V21_FORMAL_BASELINE_ROOT,
    )
    parser.add_argument(
        "--terminal-capture-v22-baseline-root",
        type=Path,
        default=TERMINAL_CAPTURE_V22_FORMAL_BASELINE_ROOT,
    )
    parser.add_argument(
        "--terminal-capture-v23-formal-baseline-root",
        type=Path,
        default=TERMINAL_CAPTURE_V23_FORMAL_BASELINE_ROOT,
    )
    parser.add_argument("--terminal-capture-radius", type=float, default=8.0)
    parser.add_argument("--terminal-radial-speed-floor", type=float, default=0.25)
    parser.add_argument("--terminal-radial-brake-gain", type=float, default=0.8)
    parser.add_argument("--terminal-tangential-gain", type=float, default=1.0)
    parser.add_argument(
        "--condition", choices=("paired", "off", "on"), default="paired"
    )
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--formal", action="store_true")
    scale.add_argument(
        "--smoke",
        action="store_true",
        help="explicitly select the default two-episode smoke evaluation",
    )
    scale.add_argument(
        "--smoke-count", type=int, choices=(1, 2, 3), default=DEFAULT_SMOKE_EPISODES
    )
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument(
        "--paired",
        dest="legacy_condition",
        action="store_const",
        const="paired",
        help="explicit alias for --condition paired",
    )
    correction.add_argument(
        "--goal-recovery-correction",
        dest="legacy_condition",
        action="store_const",
        const="on",
        help="single-condition alias for --condition on",
    )
    correction.add_argument(
        "--no-goal-recovery-correction",
        dest="legacy_condition",
        action="store_const",
        const="off",
        help="single-condition alias for --condition off",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_cli_mode(args: argparse.Namespace, raw_args: Sequence[str]) -> dict[str, Any]:
    condition = args.legacy_condition or args.condition
    formal = bool(args.formal)
    data_root, logs_root, plots_root = args.data_root, args.logs_root, args.plots_root
    validation_modes = (
        args.actual_stable_validation,
        args.terminal_capture_validation,
        args.terminal_capture_v2_validation,
        args.terminal_capture_v2_horizon_diagnostic,
        args.terminal_capture_v23_authority_repair_horizon120_diagnostic,
        args.terminal_capture_v21_validation,
        args.terminal_capture_v22_validation,
        args.terminal_capture_v23_validation,
        args.terminal_capture_v23_authority_repair_validation,
        args.single_uav_signed_bypass_observability_diagnostic,
    )
    if sum(bool(value) for value in validation_modes) > 1:
        raise ValueError("validation modes are mutually exclusive")
    if args.single_uav_signed_bypass_observability_diagnostic:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--terminal-capture-v23-authority-repair-horizon120-diagnostic",
            "--terminal-capture-v21-validation", "--terminal-capture-v22-validation",
            "--terminal-capture-v23-validation", "--terminal-capture-v23-authority-repair-validation",
            "--checkpoint", "--data-root", "--logs-root", "--plots-root",
            "--previous-formal-root", "--terminal-baseline-root",
            "--terminal-capture-radius", "--terminal-radial-speed-floor",
            "--terminal-radial-brake-gain", "--terminal-tangential-gain",
            "--terminal-capture-v1-baseline-root", "--terminal-capture-v2-formal-baseline-root",
            "--terminal-capture-v21-baseline-root", "--terminal-capture-v22-baseline-root",
            "--terminal-capture-v23-formal-baseline-root",
        }
        conflicts = sorted(token for token in raw_args if token.split("=", 1)[0] in forbidden)
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--single-uav-signed-bypass-observability-diagnostic requires locked formal ON-only mode"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = SIGNED_BYPASS_OBSERVABILITY_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = SIGNED_BYPASS_OBSERVABILITY_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = SIGNED_BYPASS_OBSERVABILITY_PLOT_ROOT
    if args.actual_stable_validation:
        forbidden = {"--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count"}
        conflicts = sorted(token for token in raw_args if token.split("=", 1)[0] in forbidden)
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1] for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError("--actual-stable-validation conflicts with OFF/paired/smoke options")
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = ACTUAL_STABLE_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = ACTUAL_STABLE_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = ACTUAL_STABLE_PLOT_ROOT
    if args.terminal_capture_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation",
        }
        conflicts = sorted(token for token in raw_args if token.split("=", 1)[0] in forbidden)
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1] for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError("--terminal-capture-validation conflicts with OFF/paired/smoke/actual-stable options")
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_PLOT_ROOT
    if args.terminal_capture_v2_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--checkpoint", "--data-root", "--logs-root", "--plots-root",
            "--terminal-capture-radius", "--terminal-radial-speed-floor",
            "--terminal-radial-brake-gain", "--terminal-tangential-gain",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v2-validation conflicts with OFF/paired/smoke/other validation options"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V2_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V2_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V2_PLOT_ROOT
    if args.terminal_capture_v2_horizon_diagnostic:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--checkpoint", "--data-root",
            "--logs-root", "--plots-root", "--terminal-capture-radius",
            "--terminal-radial-speed-floor", "--terminal-radial-brake-gain",
            "--terminal-tangential-gain", "--terminal-capture-v2-formal-baseline-root",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v2-horizon-diagnostic conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V2_HORIZON_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V2_HORIZON_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V2_HORIZON_PLOT_ROOT
    if args.terminal_capture_v23_authority_repair_horizon120_diagnostic:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--terminal-capture-v21-validation", "--terminal-capture-v22-validation",
            "--terminal-capture-v23-validation",
            "--terminal-capture-v23-authority-repair-validation",
            "--checkpoint", "--data-root", "--logs-root", "--plots-root",
            "--previous-formal-root", "--terminal-baseline-root",
            "--terminal-capture-radius", "--terminal-radial-speed-floor",
            "--terminal-radial-brake-gain", "--terminal-tangential-gain",
            "--terminal-capture-v1-baseline-root",
            "--terminal-capture-v2-formal-baseline-root",
            "--terminal-capture-v21-baseline-root",
            "--terminal-capture-v22-baseline-root",
            "--terminal-capture-v23-formal-baseline-root",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v23-authority-repair-horizon120-diagnostic "
                "conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_PLOT_ROOT
    if args.terminal_capture_v21_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--checkpoint", "--data-root", "--logs-root", "--plots-root",
            "--terminal-capture-radius", "--terminal-radial-speed-floor",
            "--terminal-radial-brake-gain", "--terminal-tangential-gain",
            "--terminal-capture-v2-formal-baseline-root",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v21-validation conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V21_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V21_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V21_PLOT_ROOT
    if args.terminal_capture_v22_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--terminal-capture-v21-validation", "--checkpoint", "--data-root",
            "--logs-root", "--plots-root", "--terminal-capture-radius",
            "--terminal-radial-speed-floor", "--terminal-radial-brake-gain",
            "--terminal-tangential-gain", "--terminal-capture-v1-baseline-root",
            "--terminal-capture-v2-formal-baseline-root",
            "--terminal-capture-v21-baseline-root",
            "--terminal-capture-v23-validation",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v22-validation conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V22_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V22_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V22_PLOT_ROOT
    if args.terminal_capture_v23_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--terminal-capture-v21-validation", "--terminal-capture-v22-validation",
            "--checkpoint", "--data-root", "--logs-root", "--plots-root",
            "--terminal-capture-radius", "--terminal-radial-speed-floor",
            "--terminal-radial-brake-gain", "--terminal-tangential-gain",
            "--terminal-capture-v1-baseline-root", "--terminal-capture-v2-formal-baseline-root",
            "--terminal-capture-v21-baseline-root", "--terminal-capture-v22-baseline-root",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v23-validation conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V23_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V23_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V23_PLOT_ROOT
    if args.terminal_capture_v23_authority_repair_validation:
        forbidden = {
            "--paired", "--no-goal-recovery-correction", "--smoke", "--smoke-count",
            "--actual-stable-validation", "--terminal-capture-validation",
            "--terminal-capture-v2-validation", "--terminal-capture-v2-horizon-diagnostic",
            "--terminal-capture-v21-validation", "--terminal-capture-v22-validation",
            "--terminal-capture-v23-validation", "--checkpoint", "--data-root",
            "--logs-root", "--plots-root", "--terminal-capture-radius",
            "--terminal-radial-speed-floor", "--terminal-radial-brake-gain",
            "--terminal-tangential-gain", "--terminal-capture-v1-baseline-root",
            "--terminal-capture-v2-formal-baseline-root",
            "--terminal-capture-v21-baseline-root", "--terminal-capture-v22-baseline-root",
            "--terminal-capture-v23-formal-baseline-root",
        }
        conflicts = sorted(
            token for token in raw_args if token.split("=", 1)[0] in forbidden
        )
        explicit_conditions = [
            raw_args[index + 1]
            for index, token in enumerate(raw_args[:-1]) if token == "--condition"
        ]
        explicit_conditions += [
            token.split("=", 1)[1]
            for token in raw_args if token.startswith("--condition=")
        ]
        if conflicts or any(value != "on" for value in explicit_conditions):
            raise ValueError(
                "--terminal-capture-v23-authority-repair-validation conflicts with overrides or other modes"
            )
        condition, formal = "on", True
        if data_root == DEFAULT_DATA_ROOT:
            data_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_DATA_ROOT
        if logs_root == DEFAULT_LOG_ROOT:
            logs_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT
        if plots_root == DEFAULT_PLOT_ROOT:
            plots_root = TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_PLOT_ROOT
    return {
        "condition": condition, "formal": formal,
        "data_root": data_root, "logs_root": logs_root, "plots_root": plots_root,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        import sys
        raw_args = sys.argv[1:]
    else:
        raw_args = list(argv)
    args = parser.parse_args(raw_args)
    try:
        resolved = _resolve_cli_mode(args, raw_args)
    except ValueError as exc:
        parser.error(str(exc))
    condition = resolved["condition"]
    if args.legacy_condition is not None and args.condition != "paired":
        parser.error("use either --condition or the ON/OFF alias, not both")
    result = run_goal_recovery_correction_evaluation(
        checkpoint=args.checkpoint,
        data_root=resolved["data_root"],
        logs_root=resolved["logs_root"],
        plots_root=resolved["plots_root"],
        condition=condition,
        formal=resolved["formal"],
        smoke_count=int(args.smoke_count),
        progress=not args.quiet,
        actual_stable_validation=bool(args.actual_stable_validation),
        previous_formal_root=args.previous_formal_root,
        terminal_capture_validation=bool(args.terminal_capture_validation),
        terminal_baseline_root=args.terminal_baseline_root,
        terminal_capture_v2_validation=bool(args.terminal_capture_v2_validation),
        terminal_capture_v1_baseline_root=args.terminal_capture_v1_baseline_root,
        terminal_capture_v2_horizon_diagnostic=bool(
            args.terminal_capture_v2_horizon_diagnostic
        ),
        terminal_capture_v2_formal_baseline_root=(
            args.terminal_capture_v2_formal_baseline_root
        ),
        terminal_capture_v23_authority_repair_horizon_diagnostic=bool(
            args.terminal_capture_v23_authority_repair_horizon120_diagnostic
        ),
        terminal_capture_v21_validation=bool(args.terminal_capture_v21_validation),
        terminal_capture_v22_validation=bool(args.terminal_capture_v22_validation),
        terminal_capture_v23_validation=bool(args.terminal_capture_v23_validation),
        terminal_capture_v23_authority_repair_validation=bool(
            args.terminal_capture_v23_authority_repair_validation
        ),
        signed_bypass_observability_diagnostic=bool(
            args.single_uav_signed_bypass_observability_diagnostic
        ),
        pass_commit_diagnostic=bool(args.single_uav_pass_commit_diagnostic),
        terminal_capture_v21_baseline_root=args.terminal_capture_v21_baseline_root,
        terminal_capture_v22_baseline_root=args.terminal_capture_v22_baseline_root,
        terminal_capture_v23_baseline_root=(
            args.terminal_capture_v23_formal_baseline_root
        ),
        terminal_capture_radius_m=float(args.terminal_capture_radius),
        terminal_radial_speed_floor_mps=float(args.terminal_radial_speed_floor),
        terminal_radial_brake_gain=float(args.terminal_radial_brake_gain),
        terminal_tangential_gain=float(args.terminal_tangential_gain),
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "ACTUAL_STABLE_DATA_ROOT",
    "ACTUAL_STABLE_LOG_ROOT",
    "ACTUAL_STABLE_PLOT_ROOT",
    "TERMINAL_CAPTURE_DATA_ROOT",
    "TERMINAL_CAPTURE_LOG_ROOT",
    "TERMINAL_CAPTURE_PLOT_ROOT",
    "TERMINAL_CAPTURE_V2_DATA_ROOT",
    "TERMINAL_CAPTURE_V2_LOG_ROOT",
    "TERMINAL_CAPTURE_V2_PLOT_ROOT",
    "TERMINAL_CAPTURE_V2_HORIZON_DATA_ROOT",
    "TERMINAL_CAPTURE_V2_HORIZON_LOG_ROOT",
    "TERMINAL_CAPTURE_V2_HORIZON_PLOT_ROOT",
    "TERMINAL_CAPTURE_V21_DATA_ROOT",
    "TERMINAL_CAPTURE_V21_LOG_ROOT",
    "TERMINAL_CAPTURE_V21_PLOT_ROOT",
    "TERMINAL_CAPTURE_V22_DATA_ROOT",
    "TERMINAL_CAPTURE_V22_LOG_ROOT",
    "TERMINAL_CAPTURE_V22_PLOT_ROOT",
    "TERMINAL_CAPTURE_V23_DATA_ROOT",
    "TERMINAL_CAPTURE_V23_LOG_ROOT",
    "TERMINAL_CAPTURE_V23_PLOT_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_DATA_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_PLOT_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_DATA_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT",
    "TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_PLOT_ROOT",
    "TERMINAL_CAPTURE_V21_FORMAL_BASELINE_ROOT",
    "TERMINAL_CAPTURE_V22_FORMAL_BASELINE_ROOT",
    "TERMINAL_CAPTURE_V23_FORMAL_BASELINE_ROOT",
    "PREVIOUS_FORMAL_LOG_ROOT",
    "EVALUATION_ONLY_FLAGS",
    "build_parser",
    "_aggregate_v22_episode_fields",
    "_augment_v22_episode_summary",
    "_aggregate_v23_episode_fields",
    "_augment_v23_episode_summary",
    "_load_terminal_capture_v21_baseline",
    "_load_terminal_capture_v22_baseline",
    "_load_terminal_capture_v23_baseline",
    "_load_terminal_capture_v23_authority_repair_baseline",
    "_v22_only_identity_checks",
    "_v23_only_identity_checks",
    "_v23_authority_repair_identity_checks",
    "_v23_authority_repair_horizon_identity_checks",
    "_augment_horizon_episode_summary",
    "_build_horizon_recovery_rows",
    "run_goal_recovery_correction_evaluation",
]
