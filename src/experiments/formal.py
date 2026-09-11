"""Reproducible single-UAV sanity and 3/5/7-UAV engineering experiments.

This module is deliberately separate from :mod:`src.experiments.pilot`.
The pilot proves that the environment, EKF, GAT, MAPPO and episode writer are
connected.  This module runs the first scale-up experiment with independent
policies for 3, 5 and 7 UAVs.  Its outputs are marked as preliminary because
formal claims still require repeated seeds, baselines, and sensitivity checks.
"""

from __future__ import annotations

import csv
import copy
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from ..simulation.ekf import EKFConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.sensor_model import SensorConfig
from ..simulation.scenario import (
    add_moving_los_screen,
    generate_scenario,
    normalize_communication_occluder_semantics,
)
from ..simulation.scenario_profile import load_scenario_profile
from ..simulation.topology import TopologyConfig
from ..synthetic.channel_model import ChannelConfig
from ..synthetic.episode_generator import _write_csv, _write_json
from ..learning.normalization import FeatureNormalizer, MetricNormalizer, ObservationNormalizer
from ..learning.measurement_quality import fit_warmup_quality_model
from ..learning.visual_odometry_quality import fit_visual_odometry_quality_model
from .pilot import (
    _EpisodeTraceWriter,
    _append_jsonl,
    _assert_action_feasible,
    _assert_attention_trace,
    _assert_graph_trace,
    _assert_observation_finite,
    _parameter_vector,
    _probe_actions,
    _seed_everything,
)
from .metrics import aggregate_candidate_link_metrics, metric_clip_fractions
from .metrics import position_error_metrics
from .cooperative_event_logging import accepted_counts, collect_window_events, require_new_episode
from ..learning.measurement_cost import candidate_communication_cost


_ALLOWED_UAV_COUNTS = (1, 3, 5, 7)
# Keep the single-UAV sanity stream away from the historical 3/5/7 streams.
_SCALE_OFFSETS = {1: 600_000, 3: 0, 5: 100_000, 7: 200_000}
_PHASE_OFFSETS = {
    "warmup": 0,
    "train": 300_000,
    "validation": 400_000,
    "test": 500_000,
}
_EPISODE_RE = re.compile(r"^episode_(\d+)$")
# The braking-distance features are bounded physical features, but a zero-
# motion warm-up probe can still make their empirical variance almost zero.
# Keep their normalization numerically meaningful until the policy has seen
# actual motion during training.
_SINGLE_UAV_BRAKING_FEATURE_SCALE_FLOOR = 0.25
_SINGLE_UAV_VELOCITY_SCALE_FLOOR_FRACTION = 0.5
_SINGLE_UAV_GOAL_DELTA_SCALE_FLOOR_M = 10.0
_DEFAULT_EKF_PARAMETERS = {
    "vo_position_std_m": 0.03,
    "process_position_std_m": 0.08,
    "process_velocity_std_mps": 0.15,
    "nlos_variance_multiplier": 2.0,
}


def load_selected_ekf_config(path: str | Path) -> dict[str, float]:
    """Load the selected training-only EKF parameters from a scan result.

    Calibration files have historically used ``vo_std_m`` in the CSV-like
    selected row and ``vo_position_std_m`` in ``selected_parameters``.  Both
    spellings are accepted, but the returned mapping always uses the names
    consumed by :class:`FormalExperimentConfig`.
    """

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read EKF configuration {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"EKF configuration must contain a JSON object: {config_path}")
    selected_parameters = payload.get("selected_parameters")
    selected_row = payload.get("selected")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(selected_parameters, Mapping):
        candidates.append(selected_parameters)
    if isinstance(selected_row, Mapping):
        candidates.append(selected_row)
    if not candidates:
        raise ValueError(
            f"EKF configuration has neither selected_parameters nor selected: {config_path}"
        )
    aliases = {
        "vo_position_std_m": ("vo_position_std_m", "vo_std_m"),
        "process_position_std_m": ("process_position_std_m",),
        "process_velocity_std_mps": ("process_velocity_std_mps",),
        "nlos_variance_multiplier": ("nlos_variance_multiplier",),
    }
    values: dict[str, float] = {}
    for canonical, names in aliases.items():
        raw_value: Any = None
        for candidate in candidates:
            for name in names:
                if name in candidate:
                    raw_value = candidate[name]
                    break
            if raw_value is not None:
                break
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid {canonical} in {config_path}") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{canonical} must be finite and positive in {config_path}")
        values[canonical] = value
    return values


def default_ekf_parameters() -> dict[str, float]:
    """Return the explicit preliminary defaults used without a config file."""

    return dict(_DEFAULT_EKF_PARAMETERS)


@dataclass(frozen=True)
class FormalExperimentConfig:
    """One scale of the preliminary experiment.

    ``uav_count=1`` is intentionally supported as a training-pipeline sanity
    case.  It is not a cooperative-localization result because no inter-UAV
    link or neighbor measurement exists in that setting.
    """

    output_dir: Path
    uav_count: int = 3
    seed: int = 2026
    gnss_mode: str = "denied"
    max_steps: int = 50
    warmup_episodes: int = 5
    # The scalable preliminary run uses 100 MAPPO updates by default.  Keep
    # this separate from the short ``run-pilot`` connectivity check.
    train_episodes: int = 100
    validation_episodes: int = 10
    test_episodes: int = 10
    device: str = "cpu"
    force: bool = False
    gat_hidden_dim: int = 32
    gat_output_dim: int = 32
    gat_heads: int = 1
    gat_layers: int = 2
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64
    max_obstacles: int = 8
    policy_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-4
    # PPO's gradient norm threshold is explicit in the formal manifest so a
    # stability intervention is reproducible without changing the historical
    # MAPPO default.
    max_grad_norm: float = 0.5
    # New stability protocols may reduce only the actor learning rate while
    # keeping the centralized critic at the historical rate.
    actor_learning_rate_scale: float = 1.0
    entropy_coef: float = 0.005
    initial_delta_std: float = 0.20
    clip_epsilon: float = 0.20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Opt-in PPO stabilization.  Legacy experiments retain MSE, joint
    # clipping, and terminal treatment of time limits for reproducibility.
    critic_loss_type: str = "mse"
    value_clip_enabled: bool = False
    value_clip_epsilon: float = 0.2
    target_kl: float | None = None
    bootstrap_time_limits: bool = False
    episode_objective: str = "legacy"
    include_remaining_time: bool = False
    # ``executed`` is the historical compatibility mode.  In ``sampled``
    # mode PPO keeps the policy-sampled action/log-probability pair intact;
    # environment projections are recorded separately.
    ppo_action_logprob_mode: str = "executed"
    # Reporting reward weights remain explicit in the manifest.  The mapping
    # is copied into RewardConfig and is never inferred from plotted fields.
    reward_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "uncertainty": 0.35,
            "loss": 0.20,
            "aoi": 0.15,
            "delay": 0.15,
            "connectivity": 0.15,
        }
    )
    collision_penalty: float = 10.0
    energy_weight: float = 0.01
    # Training-only local boundary shaping.  Zero preserves historical
    # experiments; the physical boundary projection remains unchanged.
    boundary_penalty_weight: float = 0.0
    # Explicit single-UAV learning channel.  ``off`` preserves the historical
    # PPO reward; ``loc_goal_v1`` is the existing localization-first baseline;
    # ``loc_goal_v2`` adds dense goal navigation, near-goal braking feedback,
    # and a one-shot stable-arrival bonus.
    # Both optional channels are fitted from warm-up-only EKF quantities and
    # are active only for the one-UAV scale.
    single_uav_learning_mode: str = "off"
    single_uav_learning_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "localization_progress": 0.8,
            "goal_progress": 0.2,
        }
    )
    single_uav_goal_progress_coef: float = 1.0
    single_uav_localization_cost_weight: float = 0.2
    single_uav_goal_bonus: float = 5.0
    single_uav_goal_threshold_m: float = 1.0
    single_uav_goal_speed_threshold_mps: float = 0.5
    single_uav_near_goal_radius_m: float = 5.0
    single_uav_near_goal_speed_penalty_weight: float = 0.1
    single_uav_speed_control_mode: str = "fixed_near_goal"
    single_uav_target_speed_far_mps: float = 6.0
    single_uav_braking_acceleration_mps2: float = 2.0
    single_uav_braking_margin_m: float = 2.0
    single_uav_stable_hold_steps: int = 1
    single_uav_goal_hold_reward: float = 0.0
    single_uav_speed_deficit_penalty_weight: float = 0.05
    single_uav_goal_direction_coef: float = 0.75
    single_uav_braking_action_reward_coef: float = 1.0
    single_uav_target_velocity_action_reward_coef: float = 0.0
    single_uav_braking_safety_projection_enabled: bool = False
    # Optional training-only curriculum for the local braking skill.  It
    # changes neither the generated geometry nor validation/test starts; a
    # fraction of training episodes starts at 3/5/8 m with a controlled
    # forward speed so the policy observes timely-braking states often enough.
    single_uav_braking_curriculum_fraction: float = 0.0
    learning_reward_mode: str = "legacy"
    learning_reward_clip: float = 1.0
    learning_collision_penalty_cap: float = 1.0
    learning_potential_coef: float = 0.25
    learning_gamma: float = 0.99
    # v4 training-only residual feedback.  The report reward and its physical
    # objective remain unchanged; these fields only affect PPO's learning
    # channel after a transition has been executed.
    learning_feedback_mode: str = "v3"
    loc_progress_coef: float = 0.0
    actual_measurement_cost_enabled: bool = False
    actual_measurement_cost_target: float | None = None
    actual_measurement_cost_coef: float = 0.0
    accepted_update_feedback_coef: float = 0.0
    accepted_update_feedback_delay_steps: int = 1
    # v2 action-linked feedback is normalized against warm-up-only robust
    # statistics and compared with an explicit none-action baseline.
    accepted_update_feedback_location: float = 0.0
    accepted_update_feedback_scale: float = 1.0
    accepted_update_feedback_none_baseline: float = 0.0
    # ``progress_potential_v2`` is a training-only shaping channel.  Its
    # robust scale statistics are fitted from warm-up and frozen thereafter.
    potential_scale_source: str = "warmup"
    potential_statistics: Mapping[str, Any] | None = None
    vo_position_std_m: float = 0.03
    # Optional execution-time VO quality calibration.  The mapping is fitted
    # from warm-up sensor/innovation diagnostics and then frozen for the
    # train/validation/test phases of this run.
    visual_odometry_quality_calibration_enabled: bool = False
    visual_odometry_quality_model: Mapping[str, Any] | None = None
    process_position_std_m: float = 0.08
    process_velocity_std_mps: float = 0.15
    nlos_variance_multiplier: float = 2.0
    scenario_profile_path: Path | None = None
    # A frozen explicit scenario is used by the safety motion-feature smoke.
    # It is mutually exclusive with the communication-profile generator.
    scenario_path: Path | None = None
    # Optional fixed geometry seed for the first single-UAV sanity run.  The
    # transition/sensor seed remains phase-specific; only the generated scene
    # layout is held constant when this is supplied.
    scenario_seed: int | None = None
    # A deterministic split-aware pool of explicit scenarios.  This is used
    # by the formal safety experiment; legacy runs continue to use either
    # scenario_path or the seeded generator above.
    scenario_pool_root: Path | None = None
    # The legacy actor observation has 4 obstacle aggregate values (37 node
    # features). The safety smoke opts into four additional dynamic-motion
    # values, producing a 41-dimensional node feature.
    dynamic_obstacle_motion_features_enabled: bool = False
    # Corrected safety-study option: include a one-step swept envelope of
    # moving obstacles in the execution-time safety projection.  It remains
    # disabled by default so historical checkpoints remain reproducible.
    dynamic_obstacle_swept_projection_enabled: bool = False
    # Versioned safety execution contract.  The v10 safety protocol opts into
    # the joint swept-motion implementation; historical configurations remain
    # on the legacy per-UAV projector.
    safety_execution_mode: str = "legacy"
    # A screening geometry may affect link visibility without being a physical
    # obstacle.  Historical protocols keep the old physical interpretation;
    # the versioned communication-occluder protocol sets this to False.
    communication_occluder_physical_collision_enabled: bool = True
    # Set by the multi-scale CLI after pooling all training warm-up rows.
    metric_scales: Mapping[str, float] | None = None
    metric_scale_source: str = "training_warmup_p95"
    uncertainty_normalization: str = "p95"
    ekf_config_source: str = "cli/defaults"
    command_line: str = ""
    progress: bool = True
    # When enabled, the MAPPO policy uses isolated module-level RNG streams
    # so paired branches share identical initialization for common modules.
    branch_invariant_initialization: bool = False
    # Optional actor-visible/physical guard for the per-link UWB update
    # period.  Disabled by default to preserve historical runs.
    measurement_update_eligibility_enabled: bool = False
    # Local actor encoder.  ``gat`` preserves the historical architecture;
    # ``mlp_flat`` is a role-aware fixed-slot architecture ablation.
    encoder_type: str = "gat"
    measurement_head_mode: str = "global_mean"
    gradient_clip_scope: str = "joint"
    random_stream_mode: str = "legacy"
    rollout_episodes_per_update: int = 1
    development_monitor_enabled: bool = False
    measurement_proxy_full_packet_cost: bool = False
    auxiliary_gradient_diagnostics: bool = False
    frozen_normalization_path: str | None = None
    sealed_test_generation: bool = False
    # ``baseline_mappo`` preserves the historical actor.  The quality-aware
    # branch changes only the measurement head input.
    mappo_variant: str = "baseline_mappo"
    # When enabled, a selected neighbour-measurement action injects one
    # equivalent L_pkt-sized packet into the communication queue. Keep the
    # default disabled so historical runs remain reproducible.
    measurement_overhead_enabled: bool = False
    # Quality-aware runs may apply a fixed monotone penalty to candidate
    # measurement logits. The baseline path is forced to zero at trainer
    # construction, regardless of this template value.
    measurement_quality_logit_penalty: float = 1.0
    # The budgeted quality-aware branch receives and enforces a fixed
    # train-derived per-UAV measurement budget.
    measurement_budget_enabled: bool = False
    measurement_budget_by_uav: Mapping[str, int] | None = None
    # The rate-controlled quality-aware branch adds a train-only, differentiable
    # penalty that keeps expected measurement rates near baseline train targets.
    measurement_rate_constraint_enabled: bool = False
    measurement_rate_constraint_coef: float = 0.5
    measurement_rate_tolerance: float = 0.05
    measurement_rate_target_by_uav: Mapping[str, float] | None = None
    # Optional adaptive per-UAV multiplier for the train-only measurement-rate
    # constraint.  Defaults preserve all historical branches/checkpoints.
    measurement_rate_dual_enabled: bool = False
    measurement_rate_dual_initial: float = 0.0
    measurement_rate_dual_step_size: float = 0.25
    measurement_rate_dual_max: float = 5.0
    # Optional execution-time information-gain/communication-cost proxy for
    # the measurement head.  The proxy model is fitted by the multi-seed
    # runner from baseline training observations only.
    measurement_proxy_features_enabled: bool = False
    measurement_proxy_guidance_enabled: bool = False
    measurement_proxy_teacher_coef: float = 0.25
    measurement_proxy_temperature: float = 0.25
    measurement_proxy_communication_cost_weight: float = 1.0
    measurement_proxy_model: Mapping[str, Any] | None = None
    # Fixed-horizon, execution-time persistent information-gain proxy.  This
    # is a separate block so legacy proxy checkpoints retain their dimensions.
    measurement_persistent_proxy_features_enabled: bool = False
    measurement_persistent_proxy_guidance_enabled: bool = False
    measurement_persistent_proxy_horizon: int = 3
    measurement_persistent_proxy_model: Mapping[str, Any] | None = None
    measurement_persistent_proxy_target_relative_cost: bool = False
    # Communication-constrained legacy proxy teachers can charge only the
    # execution-time cost above the baseline-train target.  The real queue
    # and communication objective still uses the full cost.
    measurement_proxy_target_relative_cost: bool = False
    # One-sided train-only expected communication-cost constraint. The target
    # is fitted from baseline train and includes the fixed measurement packet.
    measurement_communication_constraint_enabled: bool = False
    measurement_communication_cost_target_by_uav: Mapping[str, float] | None = None
    measurement_communication_cost_tolerance: float = 0.05
    measurement_communication_dual_initial: float = 0.0
    measurement_communication_dual_step_size: float = 0.25
    measurement_communication_dual_max: float = 5.0
    measurement_communication_fixed_packet_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.uav_count not in _ALLOWED_UAV_COUNTS:
            raise ValueError(f"uav_count must be one of {_ALLOWED_UAV_COUNTS}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if min(self.warmup_episodes, self.train_episodes, self.validation_episodes, self.test_episodes) < 1:
            raise ValueError("warmup, train, validation, and test episode counts must be positive")
        if self.device != "cpu":
            raise ValueError("the preliminary experiment currently supports device='cpu' only")
        if self.critic_loss_type not in {"mse", "huber"}:
            raise ValueError("critic_loss_type must be mse or huber")
        if not np.isfinite(float(self.value_clip_epsilon)) or float(self.value_clip_epsilon) <= 0.0:
            raise ValueError("value_clip_epsilon must be finite and positive")
        if self.target_kl is not None and (
            not np.isfinite(float(self.target_kl)) or float(self.target_kl) <= 0.0
        ):
            raise ValueError("target_kl must be None or finite and positive")
        if not np.isfinite(float(self.actor_learning_rate_scale)) or float(self.actor_learning_rate_scale) <= 0.0:
            raise ValueError("actor_learning_rate_scale must be finite and positive")
        if self.ppo_action_logprob_mode not in {"sampled", "executed"}:
            raise ValueError("ppo_action_logprob_mode must be sampled or executed")
        if self.single_uav_learning_mode not in {"off", "loc_goal_v1", "loc_goal_v2"}:
            raise ValueError(
                "single_uav_learning_mode must be off, loc_goal_v1, or loc_goal_v2"
            )
        if not isinstance(self.single_uav_learning_weights, Mapping):
            raise ValueError("single_uav_learning_weights must be a mapping")
        expected_single_uav_weight_keys = {
            "localization_progress",
            "goal_progress",
        }
        if set(self.single_uav_learning_weights) != expected_single_uav_weight_keys:
            raise ValueError(
                "single_uav_learning_weights must contain exactly "
                "localization_progress and goal_progress"
            )
        single_uav_weight_values = [
            float(self.single_uav_learning_weights[key])
            for key in sorted(expected_single_uav_weight_keys)
        ]
        if any(
            not np.isfinite(value) or value < 0.0
            for value in single_uav_weight_values
        ):
            raise ValueError(
                "single_uav_learning_weights must contain finite non-negative values"
            )
        if not np.isclose(sum(single_uav_weight_values), 1.0, atol=1e-8):
            raise ValueError("single_uav_learning_weights must sum to one")
        for name, value in (
            ("single_uav_goal_progress_coef", self.single_uav_goal_progress_coef),
            (
                "single_uav_localization_cost_weight",
                self.single_uav_localization_cost_weight,
            ),
            ("single_uav_goal_bonus", self.single_uav_goal_bonus),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(float(self.single_uav_goal_threshold_m))
            or float(self.single_uav_goal_threshold_m) <= 0.0
        ):
            raise ValueError("single_uav_goal_threshold_m must be finite and positive")
        if (
            not np.isfinite(float(self.single_uav_goal_speed_threshold_mps))
            or float(self.single_uav_goal_speed_threshold_mps) <= 0.0
        ):
            raise ValueError(
                "single_uav_goal_speed_threshold_mps must be finite and positive"
            )
        if (
            not np.isfinite(float(self.single_uav_near_goal_radius_m))
            or float(self.single_uav_near_goal_radius_m)
            < float(self.single_uav_goal_threshold_m)
        ):
            raise ValueError(
                "single_uav_near_goal_radius_m must be at least the goal threshold"
            )
        if (
            not np.isfinite(float(self.single_uav_near_goal_speed_penalty_weight))
            or float(self.single_uav_near_goal_speed_penalty_weight) < 0.0
        ):
            raise ValueError(
                "single_uav_near_goal_speed_penalty_weight must be finite and non-negative"
            )
        if self.single_uav_speed_control_mode not in {
            "fixed_near_goal",
            "distance_target_speed",
            "braking_distance",
        }:
            raise ValueError(
                "single_uav_speed_control_mode must be fixed_near_goal, "
                "distance_target_speed, or braking_distance"
            )
        if (
            not np.isfinite(float(self.single_uav_target_speed_far_mps))
            or float(self.single_uav_target_speed_far_mps)
            < float(self.single_uav_goal_speed_threshold_mps)
        ):
            raise ValueError(
                "single_uav_target_speed_far_mps must be finite and at least "
                "single_uav_goal_speed_threshold_mps"
            )
        if (
            not np.isfinite(float(self.single_uav_braking_acceleration_mps2))
            or float(self.single_uav_braking_acceleration_mps2) <= 0.0
        ):
            raise ValueError(
                "single_uav_braking_acceleration_mps2 must be finite and positive"
            )
        if (
            not np.isfinite(float(self.single_uav_braking_margin_m))
            or float(self.single_uav_braking_margin_m) < 0.0
        ):
            raise ValueError(
                "single_uav_braking_margin_m must be finite and non-negative"
            )
        if (
            isinstance(self.single_uav_stable_hold_steps, bool)
            or int(self.single_uav_stable_hold_steps)
            != self.single_uav_stable_hold_steps
            or int(self.single_uav_stable_hold_steps) < 1
        ):
            raise ValueError("single_uav_stable_hold_steps must be a positive integer")
        if (
            not np.isfinite(float(self.single_uav_goal_hold_reward))
            or float(self.single_uav_goal_hold_reward) < 0.0
        ):
            raise ValueError(
                "single_uav_goal_hold_reward must be finite and non-negative"
            )
        if (
            not np.isfinite(float(self.single_uav_speed_deficit_penalty_weight))
            or float(self.single_uav_speed_deficit_penalty_weight) < 0.0
        ):
            raise ValueError(
                "single_uav_speed_deficit_penalty_weight must be finite and non-negative"
            )
        if (
            not np.isfinite(float(self.single_uav_goal_direction_coef))
            or float(self.single_uav_goal_direction_coef) < 0.0
        ):
            raise ValueError(
                "single_uav_goal_direction_coef must be finite and non-negative"
            )
        if (
            not np.isfinite(float(self.single_uav_braking_action_reward_coef))
            or float(self.single_uav_braking_action_reward_coef) < 0.0
        ):
            raise ValueError(
                "single_uav_braking_action_reward_coef must be finite and non-negative"
            )
        if (
            not np.isfinite(float(self.single_uav_target_velocity_action_reward_coef))
            or float(self.single_uav_target_velocity_action_reward_coef) < 0.0
        ):
            raise ValueError(
                "single_uav_target_velocity_action_reward_coef must be finite and non-negative"
            )
        if not isinstance(
            self.single_uav_braking_safety_projection_enabled, bool
        ):
            raise ValueError(
                "single_uav_braking_safety_projection_enabled must be boolean"
            )
        if (
            not np.isfinite(float(self.single_uav_braking_curriculum_fraction))
            or not 0.0 <= float(self.single_uav_braking_curriculum_fraction) <= 1.0
        ):
            raise ValueError(
                "single_uav_braking_curriculum_fraction must be finite and in [0, 1]"
            )
        if self.learning_reward_mode not in {
            "legacy",
            "bounded_potential",
            "progress_potential_v2",
            "report_direct",
            "report_plus_feedback",
        }:
            raise ValueError(
                "learning_reward_mode must be legacy, bounded_potential, "
                "progress_potential_v2, report_direct, or report_plus_feedback"
            )
        if self.learning_reward_mode == "report_direct" and (
            self.learning_feedback_mode != "v3"
            or float(self.learning_potential_coef) != 0.0
            or float(self.loc_progress_coef) != 0.0
            or bool(self.actual_measurement_cost_enabled)
            or float(self.actual_measurement_cost_coef) != 0.0
            or float(self.accepted_update_feedback_coef) != 0.0
        ):
            raise ValueError("report_direct forbids additional learning feedback")
        if self.learning_feedback_mode not in {
            "v3",
            "loc_progress_v1",
            "loc_progress_v2",
            "actual_load_v1",
            "loc_progress_actual_load_v1",
            "accepted_update_progress_v1",
            "accepted_update_progress_v2",
        }:
            raise ValueError(
                "learning_feedback_mode must be v3, loc_progress_v1, loc_progress_v2, "
                "actual_load_v1, loc_progress_actual_load_v1, accepted_update_progress_v1, "
                "or accepted_update_progress_v2"
            )
        if self.potential_scale_source not in {"warmup", "frozen"}:
            raise ValueError("potential_scale_source must be warmup or frozen")
        if self.potential_statistics is not None:
            if not isinstance(self.potential_statistics, Mapping):
                raise ValueError("potential_statistics must be a mapping when supplied")
            for key in ("u_center", "u_scale", "c_center", "c_scale"):
                if key not in self.potential_statistics:
                    raise ValueError(f"potential_statistics is missing {key}")
            if self.learning_feedback_mode == "loc_progress_v2":
                if "delta_u_scale" not in self.potential_statistics:
                    raise ValueError(
                        "loc_progress_v2 requires potential_statistics[delta_u_scale]"
                    )
                if not np.isfinite(float(self.potential_statistics["delta_u_scale"])) or float(
                    self.potential_statistics["delta_u_scale"]
                ) <= 0.0:
                    raise ValueError("potential_statistics[delta_u_scale] must be positive")
            if self.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}:
                for key in ("delta_u_scale", "delta_goal_scale"):
                    if key not in self.potential_statistics:
                        raise ValueError(
                            "single-UAV goal-progress mode requires "
                            f"potential_statistics[{key}]"
                        )
                    if not np.isfinite(float(self.potential_statistics[key])) or float(
                        self.potential_statistics[key]
                    ) <= 0.0:
                        raise ValueError(
                            f"potential_statistics[{key}] must be positive"
                        )
        for name, value in (
            ("collision_penalty", self.collision_penalty),
            ("energy_weight", self.energy_weight),
            ("boundary_penalty_weight", self.boundary_penalty_weight),
            ("learning_reward_clip", self.learning_reward_clip),
            ("learning_collision_penalty_cap", self.learning_collision_penalty_cap),
            ("learning_potential_coef", self.learning_potential_coef),
            ("learning_gamma", self.learning_gamma),
            ("loc_progress_coef", self.loc_progress_coef),
            ("actual_measurement_cost_coef", self.actual_measurement_cost_coef),
            ("accepted_update_feedback_coef", self.accepted_update_feedback_coef),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.accepted_update_feedback_delay_steps, bool) or int(
            self.accepted_update_feedback_delay_steps
        ) < 0:
            raise ValueError(
                "accepted_update_feedback_delay_steps must be a non-negative integer"
            )
        if self.actual_measurement_cost_target is not None and (
            not np.isfinite(float(self.actual_measurement_cost_target))
            or float(self.actual_measurement_cost_target) < 0.0
        ):
            raise ValueError("actual_measurement_cost_target must be finite and non-negative")
        if not isinstance(self.actual_measurement_cost_enabled, bool):
            raise ValueError("actual_measurement_cost_enabled must be boolean")
        if self.learning_feedback_mode in {"actual_load_v1", "loc_progress_actual_load_v1"}:
            if not self.actual_measurement_cost_enabled:
                raise ValueError(
                    "actual measurement cost must be enabled for the selected feedback mode"
                )
            if self.actual_measurement_cost_target is None:
                raise ValueError(
                    "actual_measurement_cost_target is required for actual-load feedback"
                )
        if self.learning_feedback_mode in {
            "accepted_update_progress_v1",
            "accepted_update_progress_v2",
        }:
            if self.learning_reward_mode != "report_plus_feedback":
                raise ValueError(
                    "accepted_update_progress feedback requires report_plus_feedback"
                )
            if float(self.accepted_update_feedback_coef) <= 0.0:
                raise ValueError(
                    "accepted_update_progress feedback requires a positive feedback coefficient"
                )
            for name, value in (
                ("accepted_update_feedback_location", self.accepted_update_feedback_location),
                ("accepted_update_feedback_scale", self.accepted_update_feedback_scale),
                ("accepted_update_feedback_none_baseline", self.accepted_update_feedback_none_baseline),
            ):
                if not np.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite")
            if self.learning_feedback_mode == "accepted_update_progress_v2" and float(
                self.accepted_update_feedback_scale
            ) <= 0.0:
                raise ValueError(
                    "accepted_update_progress_v2 requires a positive warm-up feedback scale"
                )
        if not 0.0 < float(self.learning_gamma) <= 1.0:
            raise ValueError("learning_gamma must be in (0, 1]")
        if not isinstance(self.reward_weights, Mapping):
            raise ValueError("reward_weights must be a mapping")
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in self.reward_weights.values()
        ):
            raise ValueError("reward_weights must contain finite non-negative values")
        if not isinstance(self.branch_invariant_initialization, bool):
            raise ValueError("branch_invariant_initialization must be boolean")
        if self.random_stream_mode not in {"legacy", "indexed-v1"}:
            raise ValueError("unknown random stream mode")
        if self.safety_execution_mode not in {"legacy", "joint_swept_v1"}:
            raise ValueError("safety_execution_mode must be legacy or joint_swept_v1")
        if not isinstance(self.communication_occluder_physical_collision_enabled, bool):
            raise ValueError(
                "communication_occluder_physical_collision_enabled must be boolean"
            )
        if self.rollout_episodes_per_update not in {1, 4, 8}:
            raise ValueError("rollout_episodes_per_update must be 1, 4, or 8")
        if self.encoder_type not in {"gat", "mlp_flat", "mlp_slots_v2"}:
            raise ValueError("encoder_type must be 'gat', 'mlp_flat', or 'mlp_slots_v2'")
        if self.measurement_head_mode not in {
            "global_mean",
            "action_conditioned_v2",
            "action_conditioned_v3",
        }:
            raise ValueError(
                "measurement_head_mode must be global_mean, action_conditioned_v2, or action_conditioned_v3"
            )
        if self.gradient_clip_scope not in {"joint", "actor_critic"}:
            raise ValueError("gradient_clip_scope must be joint or actor_critic")
        if self.gat_hidden_dim < 1 or self.gat_output_dim < 1 or self.gat_layers < 1:
            raise ValueError("GAT dimensions and layer count must be positive")
        if self.learning_rate <= 0 or self.initial_delta_std <= 0:
            raise ValueError("learning_rate and initial_delta_std must be positive")
        if not np.isfinite(float(self.max_grad_norm)) or float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.uncertainty_normalization not in {"p95", "warmup_q05_q95"}:
            raise ValueError(
                "uncertainty_normalization must be p95 or warmup_q05_q95"
            )
        if self.vo_position_std_m <= 0 or self.process_position_std_m <= 0 or self.process_velocity_std_mps <= 0:
            raise ValueError("sensor and process standard deviations must be positive")
        if self.visual_odometry_quality_model is not None and not isinstance(
            self.visual_odometry_quality_model, Mapping
        ):
            raise ValueError("visual_odometry_quality_model must be a mapping when supplied")
        if self.nlos_variance_multiplier <= 0:
            raise ValueError("nlos_variance_multiplier must be positive")
        if self.scenario_profile_path is not None and self.scenario_path is not None:
            raise ValueError("scenario_profile_path and scenario_path are mutually exclusive")
        if self.scenario_path is not None and self.scenario_pool_root is not None:
            raise ValueError("scenario_path and scenario_pool_root are mutually exclusive")
        if self.scenario_seed is not None and (
            isinstance(self.scenario_seed, bool)
            or not isinstance(self.scenario_seed, (int, np.integer))
        ):
            raise ValueError("scenario_seed must be an integer or None")
        if self.scenario_path is not None and not Path(self.scenario_path).is_file():
            raise ValueError(f"scenario_path does not exist or is not a file: {self.scenario_path}")
        if self.mappo_variant not in {
            "baseline_mappo",
            "quality_aware_mappo",
            "budgeted_quality_aware_mappo",
            "rate_controlled_quality_aware_mappo",
            "adaptive_budgeted_quality_aware_mappo",
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
            "communication_constrained_proxy_guided_mappo",
            "persistent_proxy_guided_quality_mappo",
        }:
            raise ValueError(
                "mappo_variant must be baseline_mappo, quality_aware_mappo, "
                "budgeted_quality_aware_mappo, rate_controlled_quality_aware_mappo, "
                "adaptive_budgeted_quality_aware_mappo, or "
                "proxy_guided_adaptive_budgeted_quality_aware_mappo, or "
                "communication_constrained_proxy_guided_mappo, or "
                "persistent_proxy_guided_quality_mappo"
            )
        if not np.isfinite(self.measurement_quality_logit_penalty) or self.measurement_quality_logit_penalty < 0.0:
            raise ValueError("measurement_quality_logit_penalty must be finite and non-negative")
        if self.measurement_budget_enabled and self.measurement_budget_by_uav is None:
            raise ValueError(
                "measurement_budget_by_uav is required for budgeted MAPPO"
            )
        if self.measurement_budget_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            if {str(key) for key in self.measurement_budget_by_uav} != expected:
                raise ValueError(
                    "measurement_budget_by_uav must contain exactly all UAV identifiers"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in self.measurement_budget_by_uav.values()
            ):
                raise ValueError("measurement budgets must be non-negative integers")
        if (
            not np.isfinite(self.measurement_rate_constraint_coef)
            or self.measurement_rate_constraint_coef < 0.0
        ):
            raise ValueError("measurement_rate_constraint_coef must be finite and non-negative")
        if (
            not np.isfinite(self.measurement_rate_tolerance)
            or not 0.0 <= self.measurement_rate_tolerance <= 1.0
        ):
            raise ValueError("measurement_rate_tolerance must be finite and in [0, 1]")
        if self.measurement_rate_constraint_enabled and self.measurement_rate_target_by_uav is None:
            raise ValueError(
                "measurement_rate_target_by_uav is required when the rate constraint is enabled"
            )
        if self.measurement_rate_target_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            if {str(key) for key in self.measurement_rate_target_by_uav} != expected:
                raise ValueError(
                    "measurement_rate_target_by_uav must contain exactly all UAV identifiers"
                )
            if any(
                not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.measurement_rate_target_by_uav.values()
            ):
                raise ValueError("measurement-rate targets must be finite values in [0, 1]")
        for name, value in (
            ("measurement_rate_dual_initial", self.measurement_rate_dual_initial),
            ("measurement_rate_dual_step_size", self.measurement_rate_dual_step_size),
            ("measurement_rate_dual_max", self.measurement_rate_dual_max),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.measurement_rate_dual_initial) > float(self.measurement_rate_dual_max):
            raise ValueError("measurement_rate_dual_initial must not exceed measurement_rate_dual_max")
        if self.measurement_rate_dual_enabled:
            if not self.measurement_rate_constraint_enabled:
                raise ValueError(
                    "measurement_rate_dual_enabled requires measurement_rate_constraint_enabled"
                )
            if float(self.measurement_rate_dual_step_size) <= 0.0:
                raise ValueError("measurement_rate_dual_step_size must be positive when dual control is enabled")
            if float(self.measurement_rate_dual_max) <= 0.0:
                raise ValueError("measurement_rate_dual_max must be positive when dual control is enabled")
        if self.measurement_proxy_guidance_enabled and not self.measurement_proxy_features_enabled:
            raise ValueError(
                "measurement_proxy_guidance_enabled requires measurement_proxy_features_enabled"
            )
        if self.measurement_proxy_features_enabled and not isinstance(self.measurement_proxy_model, Mapping):
            raise ValueError(
                "measurement_proxy_model is required when measurement proxy features are enabled"
            )
        if self.measurement_persistent_proxy_guidance_enabled and not self.measurement_persistent_proxy_features_enabled:
            raise ValueError(
                "measurement_persistent_proxy_guidance_enabled requires "
                "measurement_persistent_proxy_features_enabled"
            )
        if self.measurement_proxy_features_enabled and self.measurement_persistent_proxy_features_enabled:
            raise ValueError("legacy and persistent proxy feature blocks are mutually exclusive")
        if self.measurement_persistent_proxy_features_enabled and not isinstance(
            self.measurement_persistent_proxy_model, Mapping
        ):
            raise ValueError(
                "measurement_persistent_proxy_model is required when persistent proxy features are enabled"
            )
        if self.measurement_persistent_proxy_target_relative_cost:
            if not self.measurement_persistent_proxy_guidance_enabled:
                raise ValueError(
                    "target-relative persistent proxy cost requires persistent proxy guidance"
                )
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError(
                    "measurement communication-cost targets are required for target-relative persistent proxy cost"
                )
        if self.measurement_proxy_target_relative_cost:
            if not self.measurement_proxy_guidance_enabled:
                raise ValueError("target-relative proxy cost requires proxy guidance")
            if not self.measurement_communication_constraint_enabled:
                raise ValueError(
                    "target-relative proxy cost requires the communication constraint"
                )
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError(
                    "measurement communication-cost targets are required for target-relative proxy cost"
                )
        if (
            isinstance(self.measurement_persistent_proxy_horizon, bool)
            or int(self.measurement_persistent_proxy_horizon) < 1
        ):
            raise ValueError("measurement_persistent_proxy_horizon must be a positive integer")
        if (
            not np.isfinite(float(self.measurement_proxy_teacher_coef))
            or float(self.measurement_proxy_teacher_coef) < 0.0
        ):
            raise ValueError("measurement_proxy_teacher_coef must be finite and non-negative")
        if (
            not np.isfinite(float(self.measurement_proxy_temperature))
            or float(self.measurement_proxy_temperature) <= 0.0
        ):
            raise ValueError("measurement_proxy_temperature must be finite and positive")
        if (
            not np.isfinite(float(self.measurement_proxy_communication_cost_weight))
            or float(self.measurement_proxy_communication_cost_weight) < 0.0
        ):
            raise ValueError(
                "measurement_proxy_communication_cost_weight must be finite and non-negative"
            )
        if self.measurement_communication_constraint_enabled:
            if not (
                self.measurement_proxy_features_enabled
                or self.measurement_persistent_proxy_features_enabled
            ):
                raise ValueError("measurement communication constraint requires proxy features")
            if self.measurement_rate_constraint_enabled or self.measurement_rate_dual_enabled:
                raise ValueError("communication and measurement-rate constraints are mutually exclusive")
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError("measurement communication-cost targets are required")
        if self.measurement_communication_cost_target_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            if {str(key) for key in self.measurement_communication_cost_target_by_uav} != expected:
                raise ValueError("measurement communication-cost targets must contain all UAV identifiers")
            if any(
                not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.measurement_communication_cost_target_by_uav.values()
            ):
                raise ValueError("measurement communication-cost targets must be in [0, 1]")
        for name, value in (
            ("measurement_communication_cost_tolerance", self.measurement_communication_cost_tolerance),
            ("measurement_communication_dual_initial", self.measurement_communication_dual_initial),
            ("measurement_communication_dual_step_size", self.measurement_communication_dual_step_size),
            ("measurement_communication_dual_max", self.measurement_communication_dual_max),
            ("measurement_communication_fixed_packet_fraction", self.measurement_communication_fixed_packet_fraction),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.measurement_communication_cost_tolerance) <= 1.0:
            raise ValueError("measurement communication-cost tolerance must be in [0, 1]")
        if not 0.0 <= float(self.measurement_communication_fixed_packet_fraction) <= 1.0:
            raise ValueError("fixed packet fraction must be in [0, 1]")
        if float(self.measurement_communication_dual_initial) > float(self.measurement_communication_dual_max):
            raise ValueError("measurement communication dual initial must not exceed dual max")
        if self.measurement_communication_constraint_enabled and (
            float(self.measurement_communication_dual_step_size) <= 0.0
            or float(self.measurement_communication_dual_max) <= 0.0
        ):
            raise ValueError("measurement communication dual step size and max must be positive")

    @property
    def node_feature_dim(self) -> int:
        """Return the actor/critic node feature dimension for this run."""

        base = 41 if self.dynamic_obstacle_motion_features_enabled else 37
        if self.uav_count == 1 and self.single_uav_speed_control_mode == "braking_distance":
            base += 2
        return base


def _scale_seed(base_seed: int, uav_count: int, phase: str, index: int = 0) -> int:
    return int(base_seed + _SCALE_OFFSETS[uav_count] + _PHASE_OFFSETS[phase] + 1009 * index)


def _progress(config: FormalExperimentConfig, message: str) -> None:
    if config.progress:
        print(f"[preliminary] {message}", flush=True)


def _scenario_profile(config: FormalExperimentConfig) -> dict[str, Any] | None:
    """Resolve the optional frozen environment profile for one scale."""

    if config.scenario_profile_path is None:
        return None
    return load_scenario_profile(
        config.scenario_profile_path,
        expected_uav_count=config.uav_count,
    )


def _effective_channel_config(config: FormalExperimentConfig) -> ChannelConfig:
    profile = _scenario_profile(config)
    if profile is None:
        return ChannelConfig()
    parameters = profile["parameters"]
    return replace(
        ChannelConfig(),
        total_bandwidth_hz=float(parameters["total_bandwidth_hz"]),
        nlos_extra_loss_db=float(parameters["nlos_extra_loss_db"]),
    )


def _effective_topology_config(config: FormalExperimentConfig) -> TopologyConfig:
    profile = _scenario_profile(config)
    if profile is None:
        return TopologyConfig(
            max_comm_range_m=75.0,
            uwb_range_m=100.0,
            message_arrivals_per_window=1.0,
        )
    parameters = profile["parameters"]
    return TopologyConfig(
        max_comm_range_m=float(parameters["max_comm_range_m"]),
        uwb_range_m=100.0,
        message_arrivals_per_window=float(parameters["message_arrivals_per_window"]),
    )


def _make_environment(
    config: FormalExperimentConfig,
    seed: int,
    *,
    metric_scales: Mapping[str, float] | None = None,
    cooperative_ekf_enabled: bool = True,
    measurement_overhead_enabled: bool | None = None,
    measurement_budget_enabled: bool | None = None,
    measurement_budget_by_uav: Mapping[str, int] | None = None,
    visual_odometry_quality_model: Mapping[str, Any] | None = None,
) -> MultiUAVJointEnv:
    """Create the same pure-Python environment used by the pilot."""

    if measurement_overhead_enabled is None:
        measurement_overhead_enabled = config.measurement_overhead_enabled
    if measurement_budget_enabled is None:
        measurement_budget_enabled = config.measurement_budget_enabled
    if measurement_budget_by_uav is None:
        measurement_budget_by_uav = config.measurement_budget_by_uav
    if visual_odometry_quality_model is None:
        visual_odometry_quality_model = config.visual_odometry_quality_model
    visual_odometry_quality_enabled = bool(
        config.visual_odometry_quality_calibration_enabled
        and isinstance(visual_odometry_quality_model, Mapping)
    )

    motion = MotionConfig(
        dt_s=0.2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    sensor = SensorConfig(
        uwb_max_range_m=100.0,
        vo_position_std_m=config.vo_position_std_m,
        uwb_nlos_variance_multiplier=config.nlos_variance_multiplier,
    )
    ekf = EKFConfig(
        process_position_std_m=config.process_position_std_m,
        process_velocity_std_mps=config.process_velocity_std_mps,
        vo_position_std_m=config.vo_position_std_m,
    )
    channel = _effective_channel_config(config)
    topology = _effective_topology_config(config)
    return MultiUAVJointEnv(
        SimulationConfig(
            random_stream_mode=config.random_stream_mode,
            uav_count=config.uav_count,
            max_steps=config.max_steps,
            seed=seed,
            motion=motion,
            sensor=sensor,
            channel=channel,
            topology=topology,
            reward=RewardConfig(
                collision_penalty=float(config.collision_penalty),
                energy_weight=float(config.energy_weight),
                boundary_penalty_weight=float(config.boundary_penalty_weight),
                weights=dict(config.reward_weights),
                metric_scales=dict(metric_scales) if metric_scales is not None else None,
                metric_scale_source=(config.metric_scale_source if metric_scales is not None else "default"),
                uncertainty_normalization=str(config.uncertainty_normalization),
                single_uav_learning_mode=str(config.single_uav_learning_mode),
                single_uav_learning_weights=dict(config.single_uav_learning_weights),
                single_uav_goal_progress_coef=float(config.single_uav_goal_progress_coef),
                single_uav_localization_cost_weight=float(
                    config.single_uav_localization_cost_weight
                ),
                single_uav_goal_bonus=float(config.single_uav_goal_bonus),
                single_uav_goal_threshold_m=float(config.single_uav_goal_threshold_m),
                single_uav_goal_speed_threshold_mps=float(
                    config.single_uav_goal_speed_threshold_mps
                ),
                single_uav_near_goal_radius_m=float(
                    config.single_uav_near_goal_radius_m
                ),
                single_uav_near_goal_speed_penalty_weight=float(
                    config.single_uav_near_goal_speed_penalty_weight
                ),
                single_uav_speed_control_mode=str(
                    config.single_uav_speed_control_mode
                ),
                single_uav_target_speed_far_mps=float(
                    config.single_uav_target_speed_far_mps
                ),
                single_uav_braking_acceleration_mps2=float(
                    config.single_uav_braking_acceleration_mps2
                ),
                single_uav_braking_margin_m=float(
                    config.single_uav_braking_margin_m
                ),
                single_uav_stable_hold_steps=int(
                    config.single_uav_stable_hold_steps
                ),
                single_uav_goal_hold_reward=float(
                    config.single_uav_goal_hold_reward
                ),
                single_uav_speed_deficit_penalty_weight=float(
                    config.single_uav_speed_deficit_penalty_weight
                ),
                single_uav_goal_direction_coef=float(
                    config.single_uav_goal_direction_coef
                ),
                single_uav_braking_action_reward_coef=float(
                    config.single_uav_braking_action_reward_coef
                ),
                single_uav_target_velocity_action_reward_coef=float(
                    config.single_uav_target_velocity_action_reward_coef
                ),
                single_uav_braking_safety_projection_enabled=bool(
                    config.single_uav_braking_safety_projection_enabled
                ),
                learning_reward_mode=config.learning_reward_mode,
                learning_reward_clip=float(config.learning_reward_clip),
                learning_collision_penalty_cap=float(config.learning_collision_penalty_cap),
                learning_potential_coef=float(config.learning_potential_coef),
                learning_gamma=float(config.learning_gamma),
                learning_feedback_mode=str(config.learning_feedback_mode),
                loc_progress_coef=float(config.loc_progress_coef),
                actual_measurement_cost_enabled=bool(
                    config.actual_measurement_cost_enabled
                ),
                actual_measurement_cost_target=(
                    float(config.actual_measurement_cost_target)
                    if config.actual_measurement_cost_target is not None
                    else None
                ),
                actual_measurement_cost_coef=float(
                    config.actual_measurement_cost_coef
                ),
                accepted_update_feedback_coef=float(
                    config.accepted_update_feedback_coef
                ),
                accepted_update_feedback_delay_steps=int(
                    config.accepted_update_feedback_delay_steps
                ),
                accepted_update_feedback_location=float(
                    config.accepted_update_feedback_location
                ),
                accepted_update_feedback_scale=float(
                    config.accepted_update_feedback_scale
                ),
                accepted_update_feedback_none_baseline=float(
                    config.accepted_update_feedback_none_baseline
                ),
                potential_statistics=(
                    copy.deepcopy(dict(config.potential_statistics))
                    if config.potential_statistics is not None
                    else None
                ),
                potential_scale_source=str(config.potential_scale_source),
            ),
            ekf=ekf,
            gnss_mode=config.gnss_mode,
            cooperative_ekf_enabled=cooperative_ekf_enabled,
            measurement_overhead_enabled=measurement_overhead_enabled,
            measurement_budget_enabled=measurement_budget_enabled,
            measurement_budget_by_uav=measurement_budget_by_uav,
            measurement_update_eligibility_enabled=bool(
                config.measurement_update_eligibility_enabled
            ),
            dynamic_obstacle_motion_features_enabled=(
                config.dynamic_obstacle_motion_features_enabled
            ),
            dynamic_obstacle_swept_projection_enabled=(
                config.dynamic_obstacle_swept_projection_enabled
            ),
            safety_execution_mode=str(config.safety_execution_mode),
            visual_odometry_quality_calibration_enabled=visual_odometry_quality_enabled,
            visual_odometry_quality_model=(
                copy.deepcopy(dict(visual_odometry_quality_model))
                if visual_odometry_quality_enabled
                else None
            ),
        )
    )


def _prepare_scale_output(config: FormalExperimentConfig) -> Path:
    root = Path(config.output_dir)
    scale_root = root / f"scale_{config.uav_count}"
    if scale_root.exists() and any(scale_root.iterdir()) and not config.force:
        raise FileExistsError(
            f"preliminary output is not empty: {scale_root}; use --force to replace this scale"
        )
    if config.force and scale_root.exists():
        managed_directories = (
            "train",
            "validation",
            "test",
            "calibration",
            "checkpoints",
            "diagnostics",
            "checks",
        )
        managed_files = (
            "manifest.json",
            "normalization_stats.json",
        )
        for name in managed_directories:
            target = scale_root / name
            if target.exists():
                shutil.rmtree(target)
        for name in managed_files:
            target = scale_root / name
            if target.exists():
                target.unlink()
    scale_root.mkdir(parents=True, exist_ok=True)
    return scale_root


def _copy_calibration_files(scale_root: Path, config: FormalExperimentConfig) -> None:
    target = scale_root / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    profile = _scenario_profile(config)
    channel = _effective_channel_config(config)
    topology = _effective_topology_config(config)
    _write_json(
        target / "localization_noise.json",
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "calibration_status": "synthetic_configured; no_external_dataset",
            "external_dataset_required": False,
            "parameter_sources": ["Python SensorConfig", "Python EKFConfig", "Python scenario generator"],
            "parameters": asdict(SensorConfig(
                uwb_max_range_m=100.0,
                vo_position_std_m=config.vo_position_std_m,
                uwb_nlos_variance_multiplier=config.nlos_variance_multiplier,
            )),
            "ekf_parameters": asdict(EKFConfig(
                process_position_std_m=config.process_position_std_m,
                process_velocity_std_mps=config.process_velocity_std_mps,
                vo_position_std_m=config.vo_position_std_m,
            )),
            "calibration_policy": "controlled parameter sweeps, repeated seeds, and ablation checks",
            "scenario_profile": profile,
        },
    )
    _write_json(
        target / "communication_model.json",
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "calibration_status": "synthetic_configured; no_external_dataset",
            "external_dataset_required": False,
            "parameter_sources": ["Python ChannelConfig", "Python queue and packet model"],
            "model": "log_distance_los_nlos_effective_link_with_queue_and_aoi",
            "parameters": asdict(channel),
            "topology_parameters": asdict(topology),
            "scenario_profile": profile,
            "calibration_policy": "controlled parameter sweeps, repeated seeds, and ablation checks",
        },
    )


def _scenario_pool_files(config: FormalExperimentConfig, split: str) -> list[Path]:
    if config.scenario_pool_root is None:
        return []
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported scenario-pool split: {split}")
    root = Path(config.scenario_pool_root) / split
    files = sorted(root.glob("scenario_*.json"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"scenario pool split is empty: {root}")
    return files


def _validate_scenario_pool(config: FormalExperimentConfig) -> None:
    if config.scenario_pool_root is None:
        return
    root = Path(config.scenario_pool_root)
    if not root.is_dir():
        raise FileNotFoundError(f"scenario_pool_root does not exist: {root}")
    for split in ("train", "validation", "test"):
        files = _scenario_pool_files(config, split)
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read scenario pool file {path}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"scenario pool file must contain an object: {path}")
            declared_count = payload.get("uav_count")
            if declared_count is not None and int(declared_count) != config.uav_count:
                raise ValueError(
                    f"scenario pool UAV count={declared_count} does not match "
                    f"config UAV count={config.uav_count}: {path}"
                )


def _apply_single_uav_braking_curriculum(
    config: FormalExperimentConfig,
    scenario: Mapping[str, Any],
    *,
    split: str | None,
    episode_index: int,
) -> dict[str, Any]:
    """Add training-only local braking starts without changing scene geometry.

    The ordinary generated scenario remains the source of the obstacle layout
    and goal.  A deterministic fraction of training episodes is interleaved
    throughout the training stream; in those episodes the UAV is placed on
    the original start-to-goal ray at a prescribed distance and is given a
    forward initial velocity.  Validation and test calls never enter this
    branch, so their initial state remains the original fixed-scene state.
    """

    result = copy.deepcopy(dict(scenario))
    fraction = float(config.single_uav_braking_curriculum_fraction)
    if (
        config.uav_count != 1
        or split != "train"
        or fraction <= 0.0
        or config.single_uav_learning_mode != "loc_goal_v2"
        or config.single_uav_speed_control_mode != "braking_distance"
        or not _single_uav_braking_curriculum_episode_selected(
            int(episode_index), fraction
        )
    ):
        return result

    positions = result.get("initial_positions")
    velocities = result.get("initial_velocities")
    goals = result.get("goals")
    uav_id = "uav_00"
    if not isinstance(positions, Mapping) or not isinstance(goals, Mapping):
        return result
    if uav_id not in positions or uav_id not in goals:
        return result

    start = np.asarray(positions[uav_id], dtype=float).reshape(3)
    goal = np.asarray(goals[uav_id], dtype=float).reshape(3)
    if isinstance(velocities, Mapping) and uav_id in velocities:
        source_velocity = np.asarray(velocities[uav_id], dtype=float).reshape(3)
    else:
        source_velocity = np.zeros(3, dtype=float)
    displacement = goal - start
    initial_distance = float(np.linalg.norm(displacement))
    if not np.isfinite(initial_distance) or initial_distance <= 1.5:
        return result
    direction = displacement / initial_distance
    distance_levels = (3.0, 5.0, 8.0)
    requested_distance = float(distance_levels[int(episode_index) % len(distance_levels)])
    target_distance = min(requested_distance, max(1.5, initial_distance - 1.0))
    curriculum_position = goal - direction * target_distance

    bounds_min = np.asarray(result.get("bounds_min_m", (0.0, 0.0, 2.0)), dtype=float)
    bounds_max = np.asarray(result.get("bounds_max_m", (200.0, 200.0, 60.0)), dtype=float)
    curriculum_position = np.clip(curriculum_position, bounds_min, bounds_max)
    actual_distance = float(np.linalg.norm(goal - curriculum_position))
    if not np.isfinite(actual_distance) or actual_distance <= 1.0:
        return result

    # Start above the stopping-distance target, but remain below the physical
    # speed limit.  This makes the curriculum a genuine braking example rather
    # than a second copy of the far-start navigation task.
    usable_distance = max(
        0.0,
        actual_distance
        - float(config.single_uav_goal_threshold_m)
        - float(config.single_uav_braking_margin_m),
    )
    conservative_target = float(
        np.sqrt(
            max(
                0.0,
                float(config.single_uav_goal_speed_threshold_mps) ** 2
                + 2.0
                * float(config.single_uav_braking_acceleration_mps2)
                * usable_distance,
            )
        )
    )
    initial_speed = float(
        np.clip(
            max(
                float(config.single_uav_goal_speed_threshold_mps) + 1.5,
                conservative_target + 1.0,
            ),
            float(config.single_uav_goal_speed_threshold_mps) + 1.0,
            float(MotionConfig().max_speed_mps),
        )
    )
    initial_direction = goal - curriculum_position
    initial_direction_norm = float(np.linalg.norm(initial_direction))
    if initial_direction_norm <= 1e-9:
        return result
    initial_velocity = initial_direction / initial_direction_norm * initial_speed

    updated_positions = dict(positions)
    updated_positions[uav_id] = curriculum_position.tolist()
    result["initial_positions"] = updated_positions
    if isinstance(velocities, Mapping):
        updated_velocities = dict(velocities)
    else:
        updated_velocities = {}
    updated_velocities[uav_id] = initial_velocity.tolist()
    result["initial_velocities"] = updated_velocities
    result["single_uav_braking_curriculum"] = {
        "enabled": True,
        "split": "train",
        "episode_index": int(episode_index),
        "fraction": fraction,
        "requested_distance_m": requested_distance,
        "actual_distance_m": actual_distance,
        "initial_speed_mps": initial_speed,
        "source_initial_position_m": start.tolist(),
        "source_initial_velocity_mps": source_velocity.tolist(),
        "goal_m": goal.tolist(),
        "geometry_unchanged": True,
    }
    return result


def _single_uav_braking_curriculum_episode_selected(
    episode_index: int, fraction: float
) -> bool:
    """Select a repeatable, interleaved subset of training episodes.

    A short five-episode cycle prevents the local braking examples from being
    confined to the beginning of training, while keeping the schedule fully
    reproducible and independent of the random policy samples.
    """

    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    cycle_length = 5
    selected_per_cycle = int(math.ceil(cycle_length * fraction))
    return int(episode_index) % cycle_length < selected_per_cycle


def _scenario_for(
    config: FormalExperimentConfig,
    seed: int,
    *,
    split: str | None = None,
    episode_index: int = 0,
) -> dict[str, Any]:
    if config.scenario_path is not None:
        path = Path(config.scenario_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read scenario_path {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"scenario_path must contain a JSON object: {path}")
        declared_count = payload.get("uav_count")
        if declared_count is not None and int(declared_count) != config.uav_count:
            raise ValueError(
                f"scenario_path uav_count={declared_count} does not match "
                f"config.uav_count={config.uav_count}"
            )
        scenario = normalize_communication_occluder_semantics(
            dict(payload),
            physical_collision_enabled=config.communication_occluder_physical_collision_enabled,
        )
        return _apply_single_uav_braking_curriculum(
            config, scenario, split=split, episode_index=episode_index
        )
    if config.scenario_pool_root is not None:
        if split is None:
            split = "train"
        files = _scenario_pool_files(config, split)
        path = files[int(episode_index) % len(files)]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read scenario pool file {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"scenario pool file must contain a JSON object: {path}")
        declared_count = payload.get("uav_count")
        if declared_count is not None and int(declared_count) != config.uav_count:
            raise ValueError(
                f"scenario pool uav_count={declared_count} does not match "
                f"config.uav_count={config.uav_count}: {path}"
            )
        scenario = normalize_communication_occluder_semantics(
            dict(payload),
            physical_collision_enabled=config.communication_occluder_physical_collision_enabled,
        )
        profile = _scenario_profile(config)
        if profile is not None:
            scenario["scenario_profile"] = profile
            scenario.setdefault("screening_geometry", profile.get("screening_geometry"))
        return _apply_single_uav_braking_curriculum(
            config, scenario, split=split, episode_index=episode_index
        )
    scenario = generate_scenario(
        int(config.scenario_seed) if config.scenario_seed is not None else seed,
        config.uav_count,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    profile = _scenario_profile(config)
    if profile is not None:
        scenario = add_moving_los_screen(
            scenario,
            physical_collision_enabled=(
                config.communication_occluder_physical_collision_enabled
            ),
        )
        scenario["scenario_profile"] = profile
        scenario["screening_geometry"] = profile["screening_geometry"]
    return _apply_single_uav_braking_curriculum(
        config, scenario, split=split, episode_index=episode_index
    )


def _position_logdet_from_central_state(central_state: Mapping[str, Any] | None) -> float:
    """Compute the actor-independent local position log-determinant statistic."""

    state = central_state if isinstance(central_state, Mapping) else {}
    covariances = state.get("covariances", {})
    values: list[float] = []
    if isinstance(covariances, Mapping):
        for covariance in covariances.values():
            matrix = np.asarray(covariance, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
                continue
            matrix = 0.5 * matrix[:3, :3] + 0.5 * matrix[:3, :3].T + 1e-6 * np.eye(3)
            sign, logdet = np.linalg.slogdet(matrix)
            if sign <= 0.0 or not np.isfinite(logdet):
                diagonal = np.maximum(np.diag(matrix), 1e-6)
                logdet = float(np.log(diagonal).sum())
            if np.isfinite(logdet):
                values.append(float(logdet))
    return float(np.mean(values)) if values else 0.0


def _robust_potential_center_scale(values: Sequence[float]) -> tuple[float, float]:
    """Return finite median/MAD statistics with a deterministic fallback."""

    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return 0.0, 1.0
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return center, scale


def _fit_learning_potential_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit the progress-potential scales using warm-up rows only."""

    uncertainty_values = [float(row["u"]) for row in rows if "u" in row]
    communication_values = [float(row["c"]) for row in rows if "c" in row]
    u_center, u_scale = _robust_potential_center_scale(uncertainty_values)
    c_center, c_scale = _robust_potential_center_scale(communication_values)
    return {
        "source_split": "warmup",
        "source": "warmup_only",
        "u_center": u_center,
        "u_scale": u_scale,
        "c_center": c_center,
        "c_scale": c_scale,
        "clip_range": [-3.0, 3.0],
        "epsilon": 1e-6,
    }


def _fit_adjacent_uncertainty_delta_statistics(
    values: Sequence[float],
) -> dict[str, Any]:
    """Fit the scale for same-episode adjacent uncertainty changes.

    The legacy ``u_scale`` describes the absolute level of the covariance
    statistic.  It is not a valid denominator for the one-step difference
    used by the v2 learning feedback.  This helper therefore fits a separate
    robust MAD scale from warm-up transitions only.  Episode boundaries are
    handled by the caller, which supplies only within-episode differences.
    """

    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return {
            "delta_u_scale": 1e-3,
            "delta_u_median": 0.0,
            "delta_u_mad": 0.0,
            "delta_u_samples": 0,
            "delta_u_scale_floor_applied": True,
            "delta_u_scale_source": "warmup_adjacent_window_mad",
            "delta_u_definition": "u_previous_minus_u_current_same_episode",
        }
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_scale = float(1.4826 * mad)
    scale = max(robust_scale, 1e-3)
    return {
        "delta_u_scale": float(scale),
        "delta_u_median": median,
        "delta_u_mad": mad,
        "delta_u_samples": int(array.size),
        "delta_u_scale_floor_applied": bool(robust_scale < 1e-3),
        "delta_u_scale_source": "warmup_adjacent_window_mad",
        "delta_u_definition": "u_previous_minus_u_current_same_episode",
    }


def _fit_adjacent_goal_distance_delta_statistics(
    values: Sequence[float],
) -> dict[str, Any]:
    """Fit a warm-up-only robust scale for goal-distance progress."""

    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return {
            "delta_goal_scale": 1e-3,
            "delta_goal_median": 0.0,
            "delta_goal_mad": 0.0,
            "delta_goal_samples": 0,
            "delta_goal_scale_floor_applied": True,
            "delta_goal_scale_source": "warmup_adjacent_window_mad",
            "delta_goal_definition": "goal_distance_previous_minus_current_same_episode",
        }
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_scale = float(1.4826 * mad)
    scale = max(robust_scale, 1e-3)
    return {
        "delta_goal_scale": float(scale),
        "delta_goal_median": median,
        "delta_goal_mad": mad,
        "delta_goal_samples": int(array.size),
        "delta_goal_scale_floor_applied": bool(robust_scale < 1e-3),
        "delta_goal_scale_source": "warmup_adjacent_window_mad",
        "delta_goal_definition": "goal_distance_previous_minus_current_same_episode",
    }


def _fit_accepted_update_feedback_statistics(
    values: Sequence[float],
) -> dict[str, Any]:
    """Fit a robust, warm-up-only scale for accepted-update credit.

    The samples are pre-restoration, actor-visible update-quality proxies
    collected from accepted events.  The none-action baseline is explicit and
    is not inferred from validation or test data.
    """

    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return {
            "accepted_update_feedback_location": 0.0,
            "accepted_update_feedback_scale": 0.25,
            "accepted_update_feedback_mad": 0.0,
            "accepted_update_feedback_samples": 0,
            "accepted_update_feedback_scale_floor_applied": True,
            "accepted_update_feedback_none_baseline": 0.0,
            "accepted_update_feedback_scale_source": "warmup_only_accepted_events",
        }
    location = float(np.median(array))
    mad = float(1.4826 * np.median(np.abs(array - location)))
    scale = mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(array))
    floor = 0.05
    floor_applied = not np.isfinite(scale) or scale < floor
    scale = max(float(scale) if np.isfinite(scale) else floor, floor)
    return {
        "accepted_update_feedback_location": location,
        "accepted_update_feedback_scale": scale,
        "accepted_update_feedback_mad": mad,
        "accepted_update_feedback_samples": int(array.size),
        "accepted_update_feedback_scale_floor_applied": bool(floor_applied),
        "accepted_update_feedback_none_baseline": 0.0,
        "accepted_update_feedback_scale_source": "warmup_only_accepted_events",
    }


def _within_episode_uncertainty_deltas(values: Sequence[float]) -> list[float]:
    """Return ``u_previous - u_current`` only for adjacent values in one episode.

    Keeping this operation separate makes the episode-boundary rule explicit:
    callers must pass one episode at a time and concatenate the returned
    within-episode transitions, never the raw episode endpoints.
    """

    deltas: list[float] = []
    for previous, current in zip(values, values[1:]):
        previous_value = float(previous)
        current_value = float(current)
        delta = previous_value - current_value
        if np.isfinite(delta):
            deltas.append(float(delta))
    return deltas


def _fit_normalizer(
    config: FormalExperimentConfig,
    *,
    measurement_quality_model: Mapping[str, Any] | None = None,
) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, float], dict[str, Any]]:
    from ..learning.mappo import central_state_to_vector
    samples: list[dict[str, dict[str, Any]]] = []
    metric_rows: list[Mapping[str, Any]] = []
    central_vectors: list[np.ndarray] = []
    visual_odometry_rows: list[dict[str, Any]] = []
    potential_rows: list[dict[str, float]] = []
    adjacent_delta_u_values: list[float] = []
    adjacent_delta_goal_values: list[float] = []
    accepted_update_feedback_values: list[float] = []
    transition_steps = 0
    for index in range(config.warmup_episodes):
        seed = _scale_seed(config.seed, config.uav_count, "warmup", index)
        _progress(
            config,
            f"[{config.uav_count} UAV][warmup] episode {index + 1}/{config.warmup_episodes} start",
        )
        # v2 is fitted from raw warm-up transitions.  Do not let an unfitted
        # feedback channel affect the probe trajectory or require a statistic
        # before the statistic exists.
        warmup_config = (
            replace(
                config,
                learning_feedback_mode="v3",
                loc_progress_coef=0.0,
                actual_measurement_cost_enabled=False,
                actual_measurement_cost_target=None,
                actual_measurement_cost_coef=0.0,
                accepted_update_feedback_coef=0.0,
                accepted_update_feedback_delay_steps=1,
                single_uav_learning_mode="off",
                uncertainty_normalization=(
                    "p95"
                    if config.uncertainty_normalization == "warmup_q05_q95"
                    and config.metric_scales is None
                    else config.uncertainty_normalization
                ),
            )
            if config.learning_feedback_mode in {
                "loc_progress_v2",
                "accepted_update_progress_v1",
                "accepted_update_progress_v2",
            }
            or config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
            or (
                config.uncertainty_normalization == "warmup_q05_q95"
                and config.metric_scales is None
            )
            else config
        )
        environment = _make_environment(
            warmup_config,
            seed,
            metric_scales=config.metric_scales,
        )
        observations, info = environment.reset(
            seed=seed,
            scenario_config=_scenario_for(config, seed, split="train", episode_index=index),
        )
        samples.append(observations)
        episode_u_values = [
            _position_logdet_from_central_state(info.get("central_state"))
        ]
        episode_goal_distance_values = [environment.estimated_goal_distance()]
        central_vectors.append(
            central_state_to_vector(
                info.get("central_state"),
                uav_count=config.uav_count,
                node_feature_dim=config.node_feature_dim,
                edge_feature_dim=11,
                max_obstacles=config.max_obstacles,
                include_remaining_time=config.include_remaining_time,
            )
        )
        for _step in range(config.max_steps):
            actions = _probe_actions(environment)
            next_observations, _rewards, terminated, truncated, next_info = environment.step(actions)
            if config.learning_feedback_mode == "accepted_update_progress_v2":
                for event in list(getattr(environment, "relative_range_update_events", [])):
                    if not bool(event.get("accepted", False)):
                        continue
                    quality, _gain, _consistency, _direction = environment._accepted_update_feedback_quality(event)
                    if np.isfinite(float(quality)):
                        accepted_update_feedback_values.append(float(quality))
            metric_rows.append(dict(next_info.get("global_metrics", {})))
            current_u = _position_logdet_from_central_state(
                next_info.get("central_state")
            )
            episode_u_values.append(current_u)
            episode_goal_distance_values.append(environment.estimated_goal_distance())
            objective_components = next_info.get("objective_components", {})
            if isinstance(objective_components, Mapping):
                potential_rows.append(
                    {
                        "u": current_u,
                        "c": float(objective_components.get("j_com", 0.0)),
                    }
                )
            for uav_id, packet in dict(next_info.get("sensor_packets", {})).items():
                if not isinstance(packet, Mapping):
                    continue
                vo = packet.get("vo", {})
                if not isinstance(vo, Mapping):
                    continue
                diagnostics = dict(next_info.get("ekf_diagnostics", {})).get(uav_id, {})
                if not isinstance(diagnostics, Mapping):
                    diagnostics = {}
                visual_odometry_rows.append(
                    {
                        "uav_id": str(uav_id),
                        "valid": int(vo.get("valid", 0)),
                        "tracking_quality": vo.get("quality"),
                        "nis_vo": diagnostics.get("nis_vo", -1.0),
                    }
                )
            central_vectors.append(
                central_state_to_vector(
                    next_info.get("central_state"),
                    uav_count=config.uav_count,
                    node_feature_dim=config.node_feature_dim,
                    edge_feature_dim=11,
                    max_obstacles=config.max_obstacles,
                    include_remaining_time=config.include_remaining_time,
                )
            )
            samples.append(next_observations)
            transition_steps += 1
            observations = next_observations
            if terminated or truncated:
                break
        if config.learning_feedback_mode == "loc_progress_v2" or (
            config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
        ):
            adjacent_delta_u_values.extend(
                _within_episode_uncertainty_deltas(episode_u_values)
            )
        if config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}:
            adjacent_delta_goal_values.extend(
                _within_episode_uncertainty_deltas(episode_goal_distance_values)
            )
        _progress(
            config,
            f"[{config.uav_count} UAV][warmup] episode {index + 1}/{config.warmup_episodes} complete",
        )
    normalizer = ObservationNormalizer().fit(samples)
    central_normalizer = FeatureNormalizer().fit(np.asarray(central_vectors, dtype=np.float64))
    braking_normalization_floors: dict[str, float] = {}
    braking_feature_scale_adjustments: dict[str, Any] = {}
    if (
        config.uav_count == 1
        and config.single_uav_speed_control_mode == "braking_distance"
    ):
        braking_normalization_floors = {
                "velocity_mps": max(
                    1.0,
                    float(MotionConfig().max_speed_mps)
                    * _SINGLE_UAV_VELOCITY_SCALE_FLOOR_FRACTION,
                ),
            "goal_delta_m": max(
                _SINGLE_UAV_GOAL_DELTA_SCALE_FLOOR_M,
                2.0 * float(config.single_uav_target_speed_far_mps),
            ),
            "braking_features": _SINGLE_UAV_BRAKING_FEATURE_SCALE_FLOOR,
        }
        feature_slices = {
            "velocity_mps": slice(3, 6),
            # The 15-state EKF estimate and 15 covariance-diagonal entries
            # precede the three goal-delta values in the node schema.
            "goal_delta_m": slice(30, 33),
            "braking_features": slice(config.node_feature_dim - 2, config.node_feature_dim),
        }

        def _apply_node_floors(
            target: FeatureNormalizer,
            *,
            label: str,
            offset: int = 0,
        ) -> None:
            if target.scale is None:
                return
            applied: dict[str, list[float]] = {}
            before_values: dict[str, list[float]] = {}
            for name, feature_slice in feature_slices.items():
                start = offset + feature_slice.start
                stop = offset + feature_slice.stop
                before = target.scale[start:stop].copy()
                target.scale[start:stop] = np.maximum(
                    target.scale[start:stop],
                    braking_normalization_floors[name],
                )
                before_values[name] = [float(value) for value in before]
                applied[name] = [
                    float(value) for value in target.scale[start:stop]
                ]
            braking_feature_scale_adjustments[f"{label}_before"] = before_values
            braking_feature_scale_adjustments[label] = applied

        node_normalizer = normalizer.normalizers.get("node_features")
        if node_normalizer is not None:
            _apply_node_floors(node_normalizer, label="actor_node_features")
        # central_state_to_vector stores the single-UAV node table first, so
        # the critic sees the same three feature groups at offset zero.
        _apply_node_floors(central_normalizer, label="critic_node_features")
    if config.metric_scales is None:
        metric_normalizer = MetricNormalizer().fit(metric_rows)
        metric_scales = dict(metric_normalizer.scales or {})
        scale_source = "training_warmup_p95"
        if config.uncertainty_normalization == "warmup_q05_q95":
            uncertainty_values = np.asarray(
                [float(row["uncertainty"]) for row in metric_rows
                 if "uncertainty" in row and np.isfinite(float(row["uncertainty"]))],
                dtype=float,
            )
            if uncertainty_values.size < 2:
                raise RuntimeError(
                    "warmup_q05_q95 requires at least two finite uncertainty samples"
                )
            metric_scales["uncertainty_q05"] = float(np.quantile(uncertainty_values, 0.05))
            metric_scales["uncertainty_q95"] = float(np.quantile(uncertainty_values, 0.95))
            if metric_scales["uncertainty_q95"] <= metric_scales["uncertainty_q05"]:
                raise RuntimeError("warm-up uncertainty Q05/Q95 interval is degenerate")
            scale_source = "training_warmup_uncertainty_q05_q95"
    else:
        metric_scales = {str(key): float(value) for key, value in config.metric_scales.items()}
        scale_source = config.metric_scale_source
    quality_model = (
        dict(measurement_quality_model)
        if measurement_quality_model is not None
        else fit_warmup_quality_model(
            samples,
            tuple(f"uav_{index:02d}" for index in range(config.uav_count)),
        )
    )
    visual_model = (
        copy.deepcopy(dict(config.visual_odometry_quality_model))
        if config.visual_odometry_quality_model is not None
        else (
            fit_visual_odometry_quality_model(
                visual_odometry_rows,
                base_position_std_m=config.vo_position_std_m,
            )
            if config.visual_odometry_quality_calibration_enabled
            else None
        )
    )
    potential_statistics = (
        copy.deepcopy(dict(config.potential_statistics))
        if config.potential_statistics is not None
        else _fit_learning_potential_statistics(potential_rows)
    )
    if config.learning_feedback_mode == "loc_progress_v2":
        if config.potential_statistics is not None:
            potential_statistics = copy.deepcopy(dict(potential_statistics))
            if "delta_u_scale" not in potential_statistics:
                raise RuntimeError(
                    "loc_progress_v2 frozen statistics are missing delta_u_scale"
                )
        else:
            potential_statistics = dict(potential_statistics)
            potential_statistics.update(
                _fit_adjacent_uncertainty_delta_statistics(adjacent_delta_u_values)
            )
    if config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}:
        if config.potential_statistics is not None:
            potential_statistics = copy.deepcopy(dict(potential_statistics))
            for key in ("delta_u_scale", "delta_goal_scale"):
                if key not in potential_statistics:
                    raise RuntimeError(
                        "single-UAV goal-progress mode frozen statistics are missing "
                        f"{key}"
                    )
        else:
            potential_statistics = dict(potential_statistics)
            if "delta_u_scale" not in potential_statistics:
                potential_statistics.update(
                    _fit_adjacent_uncertainty_delta_statistics(
                        adjacent_delta_u_values
                    )
                )
            potential_statistics.update(
                _fit_adjacent_goal_distance_delta_statistics(
                    adjacent_delta_goal_values
                )
            )
    accepted_feedback_statistics = _fit_accepted_update_feedback_statistics(
        accepted_update_feedback_values
    )
    return normalizer, central_normalizer, metric_scales, {
        "episodes": config.warmup_episodes,
        "steps": transition_steps,
        "observation_snapshots": len(samples),
        "fitted_keys": sorted(normalizer.normalizers),
        "central_feature_dim": central_normalizer.feature_dim,
        "central_normalization_fitted": central_normalizer.fitted,
        "single_uav_braking_normalization_floors": braking_normalization_floors,
        "single_uav_braking_feature_scale_adjustments": braking_feature_scale_adjustments,
        "source": "warmup_only_pooled_across_scales" if config.metric_scales is not None else "warmup_only",
        "metric_scales": metric_scales,
        "metric_scale_source": scale_source,
        "metric_clip_fractions": metric_clip_fractions(metric_rows, metric_scales),
        "measurement_quality_model": quality_model,
        "visual_odometry_quality_model": visual_model,
        "visual_odometry_quality_rows": int(len(visual_odometry_rows)),
        "potential_statistics": potential_statistics,
        "potential_statistics_rows": int(len(potential_rows)),
        "adjacent_delta_u_rows": int(len(adjacent_delta_u_values)),
        "adjacent_delta_goal_rows": int(len(adjacent_delta_goal_values)),
        "adjacent_delta_u_scale_source": (
            str(potential_statistics.get("delta_u_scale_source"))
            if isinstance(potential_statistics, Mapping)
            else None
        ),
        "adjacent_delta_goal_scale_source": (
            str(potential_statistics.get("delta_goal_scale_source"))
            if isinstance(potential_statistics, Mapping)
            else None
        ),
        "accepted_update_feedback_statistics": accepted_feedback_statistics,
        "accepted_update_feedback_rows": int(len(accepted_update_feedback_values)),
        "potential_scale_source": (
            str(config.potential_scale_source)
            if config.potential_statistics is not None
            else "warmup"
        ),
    }


def fit_shared_training_metric_scales(
    base_config: FormalExperimentConfig,
    uav_counts: Sequence[int],
) -> dict[str, float]:
    """Fit common p95 metric scales from pooled training warm-up probes."""

    from ..learning.mappo import central_state_to_vector  # noqa: F401  # shape contract marker

    rows: list[Mapping[str, Any]] = []
    for count in sorted(set(int(value) for value in uav_counts)):
        config = replace(base_config, uav_count=count, metric_scales=None)
        # Shared metric scales are fitted before the per-run warm-up can fit
        # the v2 adjacent-delta scale.  The probe must therefore use the
        # unchanged physical/reporting path with training-only feedback off;
        # this does not weaken the fail-closed validation of loc_progress_v2
        # for the actual experiment environment.
        probe_config = (
            replace(
                config,
                learning_feedback_mode="v3",
                loc_progress_coef=0.0,
                actual_measurement_cost_enabled=False,
                actual_measurement_cost_target=None,
                actual_measurement_cost_coef=0.0,
                single_uav_learning_mode="off",
                uncertainty_normalization=(
                    "p95"
                    if config.uncertainty_normalization == "warmup_q05_q95"
                    else config.uncertainty_normalization
                ),
            )
            if config.learning_feedback_mode == "loc_progress_v2"
            or config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
            or config.uncertainty_normalization == "warmup_q05_q95"
            else config
        )
        for index in range(config.warmup_episodes):
            seed = _scale_seed(config.seed, count, "warmup", index)
            environment = _make_environment(probe_config, seed)
            observations, _info = environment.reset(
                seed=seed,
                scenario_config=_scenario_for(
                    probe_config, seed, split="train", episode_index=index
                ),
            )
            for _step in range(config.max_steps):
                actions = _probe_actions(environment)
                _next_observations, _rewards, terminated, truncated, next_info = environment.step(actions)
                rows.append(dict(next_info.get("global_metrics", {})))
                observations = _next_observations
                if terminated or truncated:
                    break
    normalizer = MetricNormalizer().fit(rows)
    scales = dict(normalizer.scales or {})
    if base_config.uncertainty_normalization == "warmup_q05_q95":
        uncertainty_values = np.asarray(
            [float(row["uncertainty"]) for row in rows
             if "uncertainty" in row and np.isfinite(float(row["uncertainty"]))],
            dtype=float,
        )
        if uncertainty_values.size < 2:
            raise RuntimeError("shared warm-up uncertainty quantile fit has too few samples")
        scales["uncertainty_q05"] = float(np.quantile(uncertainty_values, 0.05))
        scales["uncertainty_q95"] = float(np.quantile(uncertainty_values, 0.95))
        if scales["uncertainty_q95"] <= scales["uncertainty_q05"]:
            raise RuntimeError("shared warm-up uncertainty Q05/Q95 interval is degenerate")
    return scales


def _attention_rows(
    *,
    episode_id: str,
    phase: str,
    step: int,
    timestamp: float,
    uav_id: str,
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    physical_edges = np.asarray(trace.get("physical_edge_index", np.empty((2, 0))), dtype=np.int64)
    physical_edge_count = int(physical_edges.shape[1]) if physical_edges.ndim == 2 else 0
    rows: list[dict[str, Any]] = []
    for layer in trace.get("attention_maps", []):
        attention = np.asarray(layer.get("attention", np.empty((0, 1))), dtype=np.float64).reshape(-1)
        finite = attention[np.isfinite(attention)]
        positive = finite[finite > 0.0]
        entropy = float(-np.mean(positive * np.log(np.maximum(positive, 1e-12)))) if positive.size else 0.0
        edge_index = np.asarray(layer.get("edge_index", np.empty((2, 0))), dtype=np.int64)
        rows.append(
            {
                "episode_id": episode_id,
                "phase": phase,
                "step": step,
                "timestamp": timestamp,
                "uav_id": uav_id,
                "layer": int(layer.get("layer", 0)),
                "physical_edge_count": physical_edge_count,
                "attention_edge_count": int(edge_index.shape[1]) if edge_index.ndim == 2 else 0,
                "layer_count": int(trace.get("layer_count", 0)),
                "mean_attention": float(np.mean(finite)) if finite.size else 0.0,
                "max_attention": float(np.max(finite)) if finite.size else 0.0,
                "attention_entropy": entropy,
                "virtual_self_loops_added": int(bool(trace.get("virtual_self_loops_added", False))),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _finite_mean(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else default


def _read_episode_update_rows(episode_path: Path) -> list[dict[str, Any]]:
    """Read only this episode's cooperative-update diagnostics, if present."""

    diagnostics = episode_path.parent.parent / "diagnostics"
    path = diagnostics / f"{episode_path.parent.name}_cooperative_updates.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    episode_id = episode_path.name
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and str(value.get("episode_id", "")) == episode_id:
                rows.append(dict(value))
    return rows


def _episode_update_metrics(
    episode_path: Path,
    *,
    rewards: Sequence[Mapping[str, Any]],
    uav_count: int,
) -> dict[str, Any]:
    """Return update coverage/intensity fields with explicit no-update status."""

    events = _read_episode_update_rows(episode_path)
    accepted = [
        row for row in events
        if bool(row.get("accepted", False))
    ]
    deltas = np.asarray(
        [
            _number(row, "true_position_error_after_m")
            - _number(row, "true_position_error_before_m")
            for row in accepted
        ],
        dtype=float,
    )
    deltas = deltas[np.isfinite(deltas)]
    beneficial = int(np.sum(deltas < -0.001))
    harmful = int(np.sum(deltas > 0.001))
    neutral = int(max(0, deltas.size - beneficial - harmful))
    decision_windows = max(1, int(round(len(rewards) / max(1, uav_count))))
    opportunity_count = max(1, decision_windows * max(1, uav_count))
    no_update = int(deltas.size == 0)
    return {
        "accepted_update_count": float(deltas.size),
        "update_coverage_any": float(not no_update),
        "accepted_update_rate": float(deltas.size / opportunity_count),
        "accepted_updates_per_window": float(deltas.size / decision_windows),
        "beneficial_per_window": float(beneficial / decision_windows),
        "harmful_per_window": float(harmful / decision_windows),
        "no_update_episode_count": float(no_update),
        "update_quality_status": "NA" if no_update else "available",
        "update_quality_event_count": float(deltas.size),
        "beneficial_fraction": float(beneficial / max(1, deltas.size)),
        "harmful_fraction": float(harmful / max(1, deltas.size)),
        "neutral_fraction": float(neutral / max(1, deltas.size)),
        "nlos_update_fraction": _finite_mean(
            1.0 - float(bool(row.get("los", True))) for row in accepted
        ),
        "stale_update_fraction": _finite_mean(
            _number(row, "message_age_s", 0.0) > float(EKFConfig().stale_message_limit_s)
            for row in accepted
        ),
        "mean_update_gain_norm": _finite_mean(
            _number(row, "kalman_gain_norm") for row in accepted
        ),
        "mean_position_correction_m": _finite_mean(
            _number(row, "position_correction_norm_m") for row in accepted
        ),
    }


def _episode_metrics(episode_path: Path, episode_summary: Mapping[str, Any]) -> dict[str, Any]:
    ground_truth = _read_csv(episode_path / "ground_truth.csv")
    estimates = _read_csv(episode_path / "estimates.csv")
    links = _read_csv(episode_path / "links.csv")
    rewards = _read_csv(episode_path / "rewards.csv")
    uav_ids = sorted({row.get("uav_id", "") for row in ground_truth if row.get("uav_id")})
    uav_count = len(uav_ids)

    truth_by_key: dict[tuple[str, float], np.ndarray] = {}
    for row in ground_truth:
        truth_by_key[(row.get("uav_id", ""), round(_number(row, "timestamp"), 6))] = np.asarray(
            [_number(row, "x"), _number(row, "y"), _number(row, "z")], dtype=float
        )
    position_errors: list[float] = []
    covariance_traces: list[float] = []
    for row in estimates:
        key = (row.get("uav_id", ""), round(_number(row, "timestamp"), 6))
        if key in truth_by_key:
            estimate = np.asarray([_number(row, "x"), _number(row, "y"), _number(row, "z")], dtype=float)
            if np.isfinite(estimate).all() and np.isfinite(truth_by_key[key]).all():
                position_errors.extend((estimate - truth_by_key[key]).tolist())
        try:
            covariance = np.asarray(json.loads(row.get("covariance_flat_json", "[]")), dtype=float).reshape(-1)
            dimension = int(round(math.sqrt(covariance.size)))
            if dimension * dimension == covariance.size and dimension >= 3:
                covariance_traces.append(float(covariance[0] + covariance[dimension + 1] + covariance[2 * dimension + 2]))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    squared_position_errors = np.asarray(position_errors, dtype=float) ** 2
    position_rmse = float(np.sqrt(squared_position_errors.mean())) if squared_position_errors.size else 0.0
    canonical_position = position_error_metrics(ground_truth, estimates)
    position_rmse_norm = float(canonical_position.get("rmse_norm_m", float("nan")))
    position_p95_norm = float(canonical_position.get("p95_norm_m", float("nan")))
    position_sample_count = int(canonical_position.get("sample_count", 0.0))
    if not np.isfinite(position_rmse_norm):
        position_rmse_norm = 0.0
    if not np.isfinite(position_p95_norm):
        position_p95_norm = 0.0
    reward_values = [_number(row, "reward_total") for row in rewards]
    learning_reward_values = [
        _number(row, "learning_reward_total") for row in rewards
    ]
    reward_mean = _finite_mean(reward_values)
    episode_return = float(sum(value for value in reward_values if np.isfinite(value)) / max(1, uav_count))
    learning_reward_mean = _finite_mean(learning_reward_values)
    learning_episode_return = float(
        sum(value for value in learning_reward_values if np.isfinite(value))
        / max(1, uav_count)
    )
    boundary_projected_values = [
        _number(row, "boundary_projected", 0.0) for row in rewards
    ]
    boundary_reward_values = [
        _number(row, "boundary_reward", 0.0) for row in rewards
    ]
    boundary_events = int(
        round(sum(value for value in boundary_projected_values if np.isfinite(value)))
    )
    boundary_penalty_sum = float(
        sum(value for value in boundary_reward_values if np.isfinite(value))
    )
    scenario = json.loads((episode_path / "scenario.json").read_text(encoding="utf-8"))
    reward_weights = dict(scenario.get("reward_config", {}).get("weights", {}))
    reward_config = scenario.get("reward_config", {})
    reward_config = reward_config if isinstance(reward_config, Mapping) else {}
    uncertainty_weight = float(
        reward_weights.get("uncertainty", reward_weights.get("unc", 0.35))
    )
    joint_objective = _finite_mean(_number(row, "joint_objective") for row in rewards)
    j_loc = float(
        uncertainty_weight
        * _finite_mean(_number(row, "normalized_uncertainty") for row in rewards)
    )
    j_com = float(joint_objective - j_loc)
    normalized_uncertainty = np.asarray(
        [_number(row, "normalized_uncertainty") for row in rewards],
        dtype=float,
    )
    normalized_uncertainty = normalized_uncertainty[np.isfinite(normalized_uncertainty)]
    j_loc_saturation_fraction = float(
        np.mean(normalized_uncertainty >= 1.0 - 1e-12)
    ) if normalized_uncertainty.size else 0.0
    collision_values = [_number(row, "collision") for row in rewards]
    task_values = [_number(row, "task_complete") for row in rewards]
    return {
        "episode_id": episode_summary["episode_id"],
        "scenario_id": episode_summary["scenario_id"],
        "split": episode_summary["split"],
        "phase": episode_summary["phase"],
        "seed": episode_summary["seed"],
        "uav_count": uav_count,
        "mean_reward": reward_mean,
        "episode_return": episode_return,
        "mean_learning_reward": learning_reward_mean,
        "learning_reward_total": float(
            episode_summary.get("learning_reward_total", learning_episode_return)
        ),
        "boundary_events": int(
            episode_summary.get("boundary_events", boundary_events)
        ),
        "boundary_penalty_sum": float(
            episode_summary.get("boundary_penalty_sum", boundary_penalty_sum)
        ),
        "reporting_reward_total": float(
            episode_summary.get("reporting_reward_total", episode_return)
        ),
        "communication_applicable": bool(uav_count > 1),
        "single_uav_learning_mode": str(
            reward_config.get("single_uav_learning_mode", "off")
        ),
        "localization_progress_score": _finite_mean(
            _number(row, "localization_progress_score") for row in rewards
        ),
        "goal_progress_score": _finite_mean(
            _number(row, "goal_progress_score") for row in rewards
        ),
        "goal_progress_reward": _finite_mean(
            _number(row, "single_uav_goal_progress_reward") for row in rewards
        ),
        "goal_bonus_total": float(
            sum(
                value
                for value in (
                    _number(row, "single_uav_goal_bonus") for row in rewards
                )
                if np.isfinite(value)
            )
        ),
        "goal_bonus_triggered": max(
            [
                _number(row, "single_uav_goal_bonus_triggered")
                for row in rewards
                if np.isfinite(_number(row, "single_uav_goal_bonus_triggered"))
            ]
            or [0.0]
        ),
        "goal_reached": max(
            [
                _number(row, "single_uav_goal_reached")
                for row in rewards
                if np.isfinite(_number(row, "single_uav_goal_reached"))
            ]
            or [0.0]
        ),
        "goal_in_radius": max(
            [
                _number(row, "single_uav_goal_in_radius")
                for row in rewards
                if np.isfinite(_number(row, "single_uav_goal_in_radius"))
            ]
            or [0.0]
        ),
        "goal_speed_ok": max(
            [
                _number(row, "single_uav_goal_speed_ok")
                for row in rewards
                if np.isfinite(_number(row, "single_uav_goal_speed_ok"))
            ]
            or [0.0]
        ),
        "goal_stable": max(
            [
                _number(row, "single_uav_goal_stable")
                for row in rewards
                if np.isfinite(_number(row, "single_uav_goal_stable"))
            ]
            or [0.0]
        ),
        "speed_mps": _finite_mean(
            _number(row, "single_uav_speed_mps") for row in rewards
        ),
        "near_goal_speed_cost": _finite_mean(
            _number(row, "single_uav_near_goal_speed_cost") for row in rewards
        ),
        "speed_deficit_cost": _finite_mean(
            _number(row, "single_uav_speed_deficit_cost") for row in rewards
        ),
        "goal_direction_signal": _finite_mean(
            _number(row, "single_uav_goal_direction_signal") for row in rewards
        ),
        "goal_direction_reward": _finite_mean(
            _number(row, "single_uav_goal_direction_reward") for row in rewards
        ),
        "braking_action_reward": _finite_mean(
            _number(row, "single_uav_braking_action_reward") for row in rewards
        ),
        "target_velocity_action_error": _finite_mean(
            _number(row, "single_uav_target_velocity_action_error")
            for row in rewards
        ),
        "target_velocity_action_reward": _finite_mean(
            _number(row, "single_uav_target_velocity_action_reward")
            for row in rewards
        ),
        "speed_control_mode": str(
            reward_config.get("single_uav_speed_control_mode", "fixed_near_goal")
        ),
        "target_speed_far_mps": float(
            reward_config.get("single_uav_target_speed_far_mps", 0.0)
        ),
        "mean_target_speed_mps": _finite_mean(
            _number(row, "single_uav_target_speed_mps") for row in rewards
        ),
        "mean_speed_excess_mps": _finite_mean(
            _number(row, "single_uav_speed_excess_mps") for row in rewards
        ),
        "localization_cost": _finite_mean(
            _number(row, "single_uav_localization_cost") for row in rewards
        ),
        "goal_distance_initial": _finite_mean(
            _number(row, "previous_goal_distance") for row in rewards[:1]
        ),
        "goal_distance_final": _finite_mean(
            _number(row, "current_goal_distance") for row in rewards[-1:]
        ),
        # ``episode_return`` is intentionally retained as the per-episode
        # diagnostic used by the existing schema.  This companion field
        # removes the confounding effect of variable early termination when
        # inspecting training trends; it is not a replacement for the
        # physical report reward.
        "return_per_window": float(
            episode_return / max(1, len(reward_values) // max(1, uav_count))
        ),
        "j_loc": j_loc,
        "j_com": j_com,
        "joint_objective": joint_objective,
        "position_rmse_m": position_rmse,
        "position_rmse_norm_m": position_rmse_norm,
        "position_p95_norm_m": position_p95_norm,
        "position_error_sample_count": position_sample_count,
        "covariance_position_trace": _finite_mean(covariance_traces),
        "j_loc_saturation_fraction": j_loc_saturation_fraction,
        **_episode_update_metrics(episode_path, rewards=rewards, uav_count=uav_count),
        **aggregate_candidate_link_metrics(links),
        "connectivity": _finite_mean(_number(row, "normalized_connectivity") for row in rewards),
        "energy_proxy": _finite_mean(-_number(row, "reward_energy") for row in rewards),
        "collision_rate": _finite_mean(collision_values),
        "task_complete": max([value for value in task_values if np.isfinite(value)] or [0.0]),
        "effective_edge_rate": _finite_mean(_number(row, "effective_edge") for row in links),
        "effective_edges_seen": episode_summary.get("effective_edges_seen", 0),
        "max_neighbor_updates": episode_summary.get("max_neighbor_updates", 0),
        "window_count": episode_summary.get("window_count", len(rewards) // max(1, uav_count)),
        "max_steps": episode_summary.get("max_steps", 0),
        "collision_occurred": episode_summary.get("collision_occurred", False),
        "collision_step": episode_summary.get("collision_step", -1),
        "physical_collision_steps": json.dumps(
            episode_summary.get("physical_collision_steps", []), ensure_ascii=False
        ),
        "clearance_violation_steps": json.dumps(
            episode_summary.get("clearance_violation_steps", []), ensure_ascii=False
        ),
        "termination_reason": episode_summary.get("termination_reason", "running"),
        "outcome_class": episode_summary.get("outcome_class", "unknown_short"),
        "terminated": episode_summary.get("terminated", False),
        "truncated": episode_summary.get("truncated", False),
        "minimum_clearance": episode_summary.get("minimum_clearance", float("nan")),
        "pre_projection_minimum_clearance": episode_summary.get(
            "pre_projection_minimum_clearance", float("nan")
        ),
        "collision_types": json.dumps(
            episode_summary.get("collision_types", []), ensure_ascii=False
        ),
        "collision_objects": json.dumps(
            episode_summary.get("collision_objects", []), ensure_ascii=False
        ),
        "physical_obstacle_ids": json.dumps(
            episode_summary.get("physical_obstacle_ids", []), ensure_ascii=False
        ),
        "communication_occluder_ids": json.dumps(
            episode_summary.get("communication_occluder_ids", []), ensure_ascii=False
        ),
        "ignored_nonphysical_obstacle_ids": json.dumps(
            episode_summary.get("ignored_nonphysical_obstacle_ids", []), ensure_ascii=False
        ),
        "safety_execution_mode": episode_summary.get("safety_execution_mode", "legacy"),
        "mean_action_saturation": episode_summary.get("mean_action_saturation", 0.0),
        "max_action_saturation": episode_summary.get("max_action_saturation", 0.0),
        "safety_projection_count": episode_summary.get("safety_projection_count", 0),
        "boundary_projection_count": episode_summary.get("boundary_projection_count", 0),
    }


def _write_episode_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "episode_id",
        "scenario_id",
        "split",
        "phase",
        "seed",
        "uav_count",
        "mean_reward",
        "episode_return",
        "mean_learning_reward",
        "learning_reward_total",
        "boundary_events",
        "boundary_penalty_sum",
        "reporting_reward_total",
        "return_per_window",
        "communication_applicable",
        "single_uav_learning_mode",
        "localization_progress_score",
        "goal_progress_score",
        "goal_progress_reward",
        "goal_bonus_total",
        "goal_bonus_triggered",
        "goal_reached",
        "goal_in_radius",
        "goal_speed_ok",
        "goal_stable",
         "speed_mps",
         "near_goal_speed_cost",
         "speed_deficit_cost",
         "goal_direction_signal",
         "goal_direction_reward",
         "braking_action_reward",
         "target_velocity_action_error",
         "target_velocity_action_reward",
         "localization_cost",
        "goal_distance_initial",
        "goal_distance_final",
        "j_loc",
        "j_com",
        "joint_objective",
        "position_rmse_m",
        "position_rmse_norm_m",
        "position_p95_norm_m",
        "position_error_sample_count",
        "covariance_position_trace",
        "j_loc_saturation_fraction",
        "accepted_update_count",
        "update_coverage_any",
        "accepted_update_rate",
        "accepted_updates_per_window",
        "beneficial_per_window",
        "harmful_per_window",
        "no_update_episode_count",
        "update_quality_status",
        "update_quality_event_count",
        "beneficial_fraction",
        "neutral_fraction",
        "harmful_fraction",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "collision_rate",
        "task_complete",
        "effective_edge_rate",
        "effective_edges_seen",
        "max_neighbor_updates",
        "window_count",
        "max_steps",
        "collision_occurred",
        "collision_step",
        "physical_collision_steps",
        "clearance_violation_steps",
        "termination_reason",
        "outcome_class",
        "terminated",
        "truncated",
        "minimum_clearance",
        "pre_projection_minimum_clearance",
        "collision_types",
        "collision_objects",
        "physical_obstacle_ids",
        "communication_occluder_ids",
        "ignored_nonphysical_obstacle_ids",
        "safety_execution_mode",
        "mean_action_saturation",
        "max_action_saturation",
        "safety_projection_count",
        "boundary_projection_count",
    ]
    _write_csv(path, columns, [dict(row) for row in rows])


def _write_attention_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "episode_id",
        "phase",
        "step",
        "timestamp",
        "uav_id",
        "layer",
        "physical_edge_count",
        "attention_edge_count",
        "layer_count",
        "mean_attention",
        "max_attention",
        "attention_entropy",
        "virtual_self_loops_added",
    ]
    _write_csv(path, columns, [dict(row) for row in rows])


def _save_checkpoint(path: Path, trainer: Any, config: FormalExperimentConfig, episode: int) -> None:
    import torch

    trainer_config = asdict(trainer.config)
    payload = {
        "checkpoint_version": "mappo-preliminary-v1",
        "uav_count": config.uav_count,
        "episode": config.train_episodes,
        "update_count": episode,
        "experiment_config": {**asdict(config), "output_dir": str(config.output_dir)},
        "trainer_config": trainer_config,
        "policy_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "normalization_stats": trainer.normalizer.state_dict(),
        "central_normalization_stats": trainer.central_normalizer.state_dict(),
        "potential_statistics": copy.deepcopy(config.potential_statistics),
        "initial_parameter_hashes": trainer.policy.initial_parameter_initialization_hashes(),
        # Old checkpoints remain loadable because consumers may ignore this
        # optional field; adaptive runs persist the per-UAV dual state here.
        "measurement_rate_dual_state": trainer.measurement_rate_dual_state_dict(),
        "measurement_communication_dual_state": trainer.measurement_communication_dual_state_dict(),
        # Proxy-guided runs persist the frozen train-derived proxy parameters
        # alongside the actor so deterministic evaluation can reproduce the
        # exact measurement-head inputs without refitting on validation/test.
        "measurement_proxy_model": copy.deepcopy(
            getattr(trainer, "measurement_proxy_model", None)
        ),
        "measurement_persistent_proxy_model": copy.deepcopy(
            getattr(trainer, "measurement_persistent_proxy_model", None)
        ),
        "visual_odometry_quality_model": copy.deepcopy(
            config.visual_odometry_quality_model
            if config.visual_odometry_quality_calibration_enabled
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_scale_manifest(
    *,
    scale_root: Path,
    config: FormalExperimentConfig,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profile = _scenario_profile(config)
    config_payload = {**asdict(config), "output_dir": str(config.output_dir)}
    # ``node_feature_dim`` is a derived configuration property rather than a
    # dataclass field.  Persist the effective value so downstream audits can
    # distinguish the 37-dimensional legacy branch from the 41-dimensional
    # dynamic-obstacle branch without relying on a fallback.
    config_payload["node_feature_dim"] = int(config.node_feature_dim)
    initial_hashes: dict[str, Any] = {}
    initial_hash_path = scale_root / "diagnostics" / "initial_parameter_hashes.json"
    if initial_hash_path.is_file():
        try:
            loaded_hashes = json.loads(initial_hash_path.read_text(encoding="utf-8"))
            if isinstance(loaded_hashes, Mapping):
                initial_hashes = dict(loaded_hashes)
        except (OSError, UnicodeError, json.JSONDecodeError):
            initial_hashes = {}
    if config.scenario_profile_path is not None:
        config_payload["scenario_profile_path"] = str(config.scenario_profile_path.resolve())
    if config.scenario_path is not None:
        config_payload["scenario_path"] = str(config.scenario_path.resolve())
    if config.scenario_pool_root is not None:
        config_payload["scenario_pool_root"] = str(config.scenario_pool_root.resolve())
    manifest = {
        "dataset_id": "python_multi_uav_mappo_preliminary",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "engine_status": "pure_python_action_driven_environment_with_mappo",
        "generation_status": "engineering_preliminary; Python configuration recorded; repeated-seed and sensitivity checks required",
        "result_status": (
            "single_uav_sanity_only" if config.uav_count == 1 else "preliminary_only"
        ),
        "policy_source": "mappo",
        "uav_count": config.uav_count,
        "algorithm_role": (
            "single_agent_ppo_sanity"
            if config.uav_count == 1
            else "shared_policy_mappo"
        ),
        "cooperative_links_expected": bool(config.uav_count > 1),
        "communication_applicable": bool(config.uav_count > 1),
        "single_uav_learning_mode": str(config.single_uav_learning_mode),
        "single_uav_learning_active": bool(
            config.uav_count == 1
            and config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
        ),
        "single_uav_learning_weights": {
            str(key): float(value)
            for key, value in config.single_uav_learning_weights.items()
        },
        "single_uav_navigation_reward": {
            "goal_progress_coef": float(config.single_uav_goal_progress_coef),
            "localization_cost_weight": float(
                config.single_uav_localization_cost_weight
            ),
            "goal_bonus": float(config.single_uav_goal_bonus),
            "goal_threshold_m": float(config.single_uav_goal_threshold_m),
            "goal_speed_threshold_mps": float(
                config.single_uav_goal_speed_threshold_mps
            ),
            "near_goal_radius_m": float(config.single_uav_near_goal_radius_m),
            "near_goal_speed_penalty_weight": float(
                config.single_uav_near_goal_speed_penalty_weight
            ),
            "speed_control_mode": str(config.single_uav_speed_control_mode),
            "target_speed_far_mps": float(config.single_uav_target_speed_far_mps),
            "braking_acceleration_mps2": float(
                config.single_uav_braking_acceleration_mps2
            ),
            "braking_margin_m": float(config.single_uav_braking_margin_m),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "goal_hold_reward": float(config.single_uav_goal_hold_reward),
            "speed_deficit_penalty_weight": float(
                config.single_uav_speed_deficit_penalty_weight
            ),
            "goal_direction_coef": float(config.single_uav_goal_direction_coef),
            "braking_action_reward_coef": float(
                config.single_uav_braking_action_reward_coef
            ),
            "target_velocity_action_reward_coef": float(
                config.single_uav_target_velocity_action_reward_coef
            ),
            "boundary_penalty_weight": float(config.boundary_penalty_weight),
            "braking_safety_projection_enabled": bool(
                config.single_uav_braking_safety_projection_enabled
            ),
            "braking_curriculum_fraction": float(
                config.single_uav_braking_curriculum_fraction
            ),
            "goal_progress_definition": "estimated_goal_distance_previous_minus_current",
            "stable_goal_definition": (
                "estimated_goal_distance_at_or_below_goal_threshold_and_actual_speed_at_or_below_speed_threshold_for_configured_consecutive_hold_steps"
            ),
            "goal_bonus_definition": "one_shot_first_stable_entry_into_estimated_goal_radius",
            "near_goal_speed_penalty_definition": (
                "fixed_mode_weight_times_actual_speed_or_distance_mode_weight_times_positive_speed_excess_above_distance_target_or_braking_distance_target"
            ),
            "goal_termination": "loc_goal_v2_terminates_when_stable_goal_is_reached",
        },
        "motion_action_contract": MotionConfig().action_contract(),
        "learning_reward_contract": {
            "reporting_reward_preserved": True,
            "boundary_penalty_in_reporting_reward": False,
            "boundary_penalty_local_per_uav": True,
            "single_uav_communication_reward_included": bool(
                config.uav_count > 1
            ),
            "single_uav_progress_scales": (
                "warmup_only_adjacent_EKF_logdet_and_estimated_goal_distance_deltas"
                if config.uav_count == 1
                and config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
                else "not_applicable"
            ),
        },
        "scenario_mode": (
            "fixed_generated_geometry_seed"
            if config.scenario_seed is not None
            else "phase_seeded_generated_geometry"
        ),
        "scenario_seed": (
            int(config.scenario_seed) if config.scenario_seed is not None
            else None
        ),
        "parameter_initialization": {
            "branch_invariant": bool(config.branch_invariant_initialization),
            "seed": (
                int(config.seed)
                if config.branch_invariant_initialization
                else None
            ),
            "module_hashes": initial_hashes.get("modules", {}),
            "shared_modules": initial_hashes.get("shared_modules", []),
        },
        "config": config_payload,
        "scenario_profile": profile,
        "screening_geometry": profile["screening_geometry"] if profile else None,
        "communication_occluder_semantics": {
            "screen_id": "screening_moving_nlos_screen",
            "physical_collision_enabled": bool(
                config.communication_occluder_physical_collision_enabled
            ),
            "affects_los": True,
            "historical_default": "physical_obstacle",
            "current_role": (
                "communication_occluder"
                if not config.communication_occluder_physical_collision_enabled
                else "physical_obstacle"
            ),
        },
        "profile_allowed_for_training": bool(profile and profile["allowed_for_training"]),
        "communication_parameters": asdict(_effective_channel_config(config)),
        "topology_parameters": asdict(_effective_topology_config(config)),
        "ekf_config": {
            "source": config.ekf_config_source,
            "parameters": {
                "vo_position_std_m": float(config.vo_position_std_m),
                "process_position_std_m": float(config.process_position_std_m),
                "process_velocity_std_mps": float(config.process_velocity_std_mps),
                "nlos_variance_multiplier": float(config.nlos_variance_multiplier),
            },
        },
        "sensor_config": asdict(SensorConfig(
            uwb_max_range_m=100.0,
            vo_position_std_m=config.vo_position_std_m,
            uwb_nlos_variance_multiplier=config.nlos_variance_multiplier,
        )),
        "command_line": config.command_line,
        "calibration_status": "synthetic_configured; no_external_dataset; repeated_seed_and_sensitivity_checks_required",
        "coordinate_frame": "ENU",
        "units": {"position": "m", "time": "s", "angles": "rad"},
        "calibration": {
            "localization": "calibration/localization_noise.json",
            "communication": "calibration/communication_model.json",
            "visual_odometry_quality": (
                "calibration/visual_odometry_quality_model.json"
                if config.visual_odometry_quality_calibration_enabled
                else None
            ),
        },
        "files_per_episode": (
            list(EPISODE_FILE_SPECS)
            + ["scenario.json"]
            + (["measurement_overhead.csv"] if config.measurement_overhead_enabled else [])
            + (["measurement_budget.csv"] if config.measurement_budget_enabled else [])
        ),
        "episodes": list(episodes),
        "notes": [
            "This is an engineering preliminary run, not a calibrated scientific result.",
            "Training, validation, and test scenes use disjoint deterministic seeds.",
            (
                "For uav_count=1 this run is a single-agent PPO/simulation-pipeline sanity check; "
                "there are no inter-UAV communication or cooperative-localization edges."
                if config.uav_count == 1
                else "The multi-UAV run contains inter-UAV communication and cooperative-localization edges."
            ),
            (
                "loc_goal_v2 uses a separate EKF-estimated dense goal-progress reward, "
                "near-goal speed penalty, one-shot stable-arrival bonus, localization cost, "
                "collision cost, and energy cost; a stable arrival terminates the episode; "
                "reporting_reward remains unchanged and single-UAV communication metrics are diagnostic only."
                if config.uav_count == 1
                and config.single_uav_learning_mode == "loc_goal_v2"
                else (
                    "loc_goal_v1 uses a separate EKF-estimated localization/goal-progress learning reward; "
                    "reporting_reward remains unchanged and single-UAV communication metrics are diagnostic only."
                    if config.uav_count == 1
                    and config.single_uav_learning_mode == "loc_goal_v1"
                    else "The single-UAV localization-first learning channel is disabled; historical learning reward is retained."
                )
            ),
            (
                "Generated scene geometry is fixed by scenario_seed; transition and sensor randomness "
                "remain phase-specific."
                if config.scenario_seed is not None
                else "Generated scene geometry is phase-seeded and therefore varies across episodes."
            ),
            "Validation and test policies are deterministic and never call MAPPO update().",
            "UWB measurements are logged at the decision rate, while local EKF range updates obey sensor.uwb_update_period_s.",
            "The existing python_sim_benchmark and python_sim_pilot directories are not overwritten.",
            "When a scenario profile is supplied, train/validation/test use the same frozen communication regime and screening geometry.",
            (
                "Visual-odometry covariance calibration is fitted from warm-up tracking quality and "
                "execution-time VO innovation diagnostics, then frozen for train/validation/test."
                if config.visual_odometry_quality_calibration_enabled
                else "Visual-odometry quality covariance calibration is disabled; fixed SensorConfig covariance is used."
            ),
        ],
    }
    _write_json(scale_root / "manifest.json", manifest)
    return manifest


def _update_root_manifest(root: Path, scale_manifest: Mapping[str, Any]) -> None:
    path = root / "manifest.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
    scales = dict(payload.get("scales", {}))
    count = str(scale_manifest["uav_count"])
    scales[count] = {
        "uav_count": scale_manifest["uav_count"],
        "path": f"scale_{count}",
        "manifest": f"scale_{count}/manifest.json",
        "result_status": scale_manifest["result_status"],
        "episode_count": len(scale_manifest.get("episodes", [])),
    }
    payload.update(
        {
            "dataset_id": "python_multi_uav_mappo_preliminary",
            "schema_version": EPISODE_SCHEMA_VERSION,
        "generation_status": "engineering_preliminary; Python configuration recorded; repeated-seed and sensitivity checks required",
            "scales": {key: scales[key] for key in sorted(scales, key=int)},
            "notes": [
                "This root manifest indexes independent scale runs.",
                "Use summarize-preliminary to aggregate metrics and plots.",
            ],
        }
    )
    _write_json(path, payload)


def _run_episode(
    *,
    config: FormalExperimentConfig,
    trainer: Any,
    scale_root: Path,
    phase: str,
    split: str,
    serial: int,
    seed: int,
    update_policy: bool,
    update_index: int | None,
    episode_index: int,
    episode_total: int,
    metric_scales: Mapping[str, float] | None = None,
    flush_update: bool = True,
) -> tuple[dict[str, Any], dict[str, float] | None, list[dict[str, Any]]]:
    require_new_episode(scale_root / split / f"episode_{serial:06d}")
    _seed_everything(seed)
    started_at = time.perf_counter()
    _progress(
        config,
        f"[{config.uav_count} UAV][{phase}] episode {episode_index}/{episode_total} start seed={seed}",
    )
    environment = _make_environment(config, seed, metric_scales=metric_scales)
    observations, info = environment.reset(
        seed=seed,
        scenario_config=_scenario_for(
            config,
            seed,
            split=split,
            episode_index=max(0, int(episode_index) - 1),
        ),
    )
    episode_id = f"episode_{serial:06d}"
    event_file = scale_root / "diagnostics" / f"{split}_cooperative_updates.jsonl"
    event_file.parent.mkdir(parents=True, exist_ok=True)
    event_file.touch(exist_ok=True)
    diagnostics_path = scale_root / "diagnostics" / "attention_maps.jsonl"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / split / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split=split,
        seed=seed,
        phase=phase,
        environment=environment,
        normalization_fitted=trainer.normalizer.fitted and trainer.central_normalizer.fitted,
        diagnostics_path=diagnostics_path,
    )
    rewards_history: list[float] = []
    learning_rewards_history: list[float] = []
    learning_component_history: list[dict[str, Any]] = []
    inference_times: list[float] = []
    attention_rows: list[dict[str, Any]] = []
    effective_edges_seen = 0
    max_neighbor_updates = 0
    progress_stride = max(1, config.max_steps // 5)
    for step in range(config.max_steps):
        if step == 0 or (step + 1) % progress_stride == 0 or step + 1 == config.max_steps:
            _progress(
                config,
                f"[{config.uav_count} UAV][{phase}] episode {episode_index}/{episode_total} "
                f"step {step + 1}/{config.max_steps}",
            )
        _assert_observation_finite(observations)
        inference_start = time.perf_counter()
        decision = trainer.act(
            observations,
            info["central_state"],
            deterministic=not update_policy,
        )
        inference_times.append(time.perf_counter() - inference_start)
        quality_log_path = scale_root / "diagnostics" / "measurement_quality.jsonl"
        for uav_id, sample in decision.samples.items():
            local_observation = decision.observations.get(uav_id, {})
            policy_diagnostics = decision.measurement_diagnostics.get(uav_id, {})
            probabilities = np.asarray(sample.measurement_probabilities, dtype=float).reshape(-1)
            quality_risk = np.asarray(
                local_observation.get("measurement_quality_risk", []), dtype=float
            ).reshape(-1)
            quality_valid = np.asarray(
                local_observation.get("measurement_quality_valid", []), dtype=float
            ).reshape(-1)
            quality_record = {
                    "episode_id": episode_id,
                    "phase": phase,
                    "split": split,
                    "step": step,
                    "timestamp": float((step + 1) * environment.config.motion.dt_s),
                    "uav_id": uav_id,
                    "mappo_variant": config.mappo_variant,
                    "quality_feature_enabled": int(
                        config.mappo_variant
                        in {
                            "quality_aware_mappo",
                            "budgeted_quality_aware_mappo",
                            "rate_controlled_quality_aware_mappo",
                            "adaptive_budgeted_quality_aware_mappo",
                            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
                            "communication_constrained_proxy_guided_mappo",
                            "persistent_proxy_guided_quality_mappo",
                        }
                    ),
                    "measurement_budget_feature_enabled": int(
                        bool(config.measurement_budget_enabled)
                    ),
                    "measurement_uav_id": decision.actions[uav_id].get("measurement_uav_id", "none"),
                    "measurement_probability": float(
                        probabilities[sample.measurement_index]
                        if sample.measurement_index < probabilities.size else 0.0
                    ),
                    "measurement_probabilities": probabilities.tolist(),
                    "none_probability": float(
                        probabilities[config.uav_count]
                        if probabilities.size > config.uav_count else 0.0
                    ),
                    "selected_quality_risk": float(sample.selected_quality_risk),
                     "selected_quality_valid": float(sample.selected_quality_valid),
                     "measurement_logits_raw": sample.measurement_logits_raw.tolist(),
                     "measurement_quality_penalty": sample.measurement_quality_penalty.tolist(),
                     "measurement_logits_adjusted": sample.measurement_logits_adjusted.tolist(),
                     "measurement_head_mode": config.measurement_head_mode,
                     "candidate_source_ids": list(
                         policy_diagnostics.get("candidate_source_ids", [])
                     ),
                     "candidate_visible": np.asarray(
                         policy_diagnostics.get("candidate_visible", []), dtype=float
                     ).tolist(),
                     "candidate_action_mask": np.asarray(
                         policy_diagnostics.get("candidate_action_mask", []), dtype=float
                     ).tolist(),
                     "candidate_action_features": np.asarray(
                         policy_diagnostics.get("candidate_action_features", []), dtype=float
                     ).tolist(),
                     "candidate_quality_scores": np.asarray(
                         policy_diagnostics.get("candidate_quality_scores", []), dtype=float
                     ).tolist(),
                     "none_context_summary": np.asarray(
                         policy_diagnostics.get("none_context_summary", []), dtype=float
                     ).tolist(),
                     "none_logit_raw": float(
                         policy_diagnostics.get("none_logit_raw", 0.0)
                     ),
                     "deterministic_action": int(
                         bool(policy_diagnostics.get("selected_action_deterministic", not update_policy))
                     ),
                     "measurement_quality_logit_penalty": float(
                         sample.measurement_quality_logit_penalty
                     ),
                    "quality_actor_branch": (
                        config.mappo_variant
                        if config.mappo_variant
                        in {
                            "quality_aware_mappo",
                            "budgeted_quality_aware_mappo",
                            "rate_controlled_quality_aware_mappo",
                            "adaptive_budgeted_quality_aware_mappo",
                            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
                            "communication_constrained_proxy_guided_mappo",
                            "persistent_proxy_guided_quality_mappo",
                        }
                        else "baseline_mappo"
                     ),
                     "measurement_rate_target": float(
                         policy_diagnostics.get("measurement_rate_target", 0.0)
                     ),
                     "measurement_rate_constraint_enabled": int(
                         bool(config.measurement_rate_constraint_enabled)
                     ),
                     "measurement_rate_constraint_coef": float(
                         config.measurement_rate_constraint_coef
                         if config.measurement_rate_constraint_enabled
                         else 0.0
                     ),
                     "measurement_rate_expected_non_none_probability": float(
                         1.0
                         - (
                             probabilities[config.uav_count]
                             if probabilities.size > config.uav_count
                             else 0.0
                         )
                     ),
                     "measurement_budget_remaining_ratio": float(
                         local_observation.get("measurement_budget_remaining_ratio", 1.0)
                     ),
                     "quality_valid_count": float(quality_valid.sum()) if quality_valid.size else 0.0,
                    "quality_min_risk": float(np.min(quality_risk)) if quality_risk.size else -1.0,
                     "quality_mean_risk": float(np.mean(quality_risk)) if quality_risk.size else -1.0,
                    "measurement_proxy_features_enabled": int(
                        bool(config.measurement_proxy_features_enabled)
                    ),
                    "measurement_proxy_guidance_enabled": int(
                        bool(config.measurement_proxy_guidance_enabled)
                    ),
                    "measurement_proxy_teacher_coef": float(
                        config.measurement_proxy_teacher_coef
                    ),
                    "measurement_proxy_temperature": float(
                        config.measurement_proxy_temperature
                    ),
                    "measurement_proxy_communication_cost_weight": float(
                        config.measurement_proxy_communication_cost_weight
                    ),
                    "measurement_proxy_gain": sample.measurement_proxy_gain.tolist(),
                    "measurement_proxy_cost": sample.measurement_proxy_cost.tolist(),
                    "candidate_full_packet_cost": candidate_communication_cost(sample.measurement_proxy_cost, config.measurement_communication_fixed_packet_fraction).tolist(),
                    "teacher_uses_full_packet_cost": bool(config.measurement_proxy_full_packet_cost),
                    "measurement_proxy_valid": sample.measurement_proxy_valid.tolist(),
                     "measurement_proxy_utility": sample.measurement_proxy_utility.tolist(),
                     "measurement_persistent_proxy_features_enabled": int(
                         bool(config.measurement_persistent_proxy_features_enabled)
                     ),
                     "measurement_persistent_proxy_guidance_enabled": int(
                         bool(config.measurement_persistent_proxy_guidance_enabled)
                     ),
                     "measurement_persistent_proxy_horizon": int(
                         config.measurement_persistent_proxy_horizon
                     ),
                     "persistent_proxy_gain": sample.persistent_proxy_gain.tolist(),
                     "persistent_proxy_cost": sample.persistent_proxy_cost.tolist(),
                     "persistent_proxy_valid": sample.persistent_proxy_valid.tolist(),
                     "persistent_proxy_utility": sample.persistent_proxy_utility.tolist(),
                 }
            _append_jsonl(quality_log_path, quality_record)
            _append_jsonl(
                scale_root / "diagnostics" / f"{split}_measurement_quality.jsonl",
                quality_record,
            )
        _assert_action_feasible(environment, decision.actions)
        for uav_id, trace in decision.graph_traces.items():
            _assert_graph_trace(trace)
            attention_rows.extend(
                _attention_rows(
                    episode_id=episode_id,
                    phase=phase,
                    step=step,
                    timestamp=(step + 1) * environment.config.motion.dt_s,
                    uav_id=uav_id,
                    trace=trace,
                )
            )
        update_counts_before = accepted_counts(environment)
        next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
        update_events_path = scale_root / "diagnostics" / "cooperative_updates.jsonl"
        update_events = collect_window_events(environment, update_counts_before, run_root=scale_root,
                                             episode_id=episode_id, split=split, step=step)
        for event in update_events:
            update_record = {
                    "episode_id": episode_id,
                    "phase": phase,
                    "split": split,
                    "step": step,
                    **dict(event),
                }
            _append_jsonl(update_events_path, update_record)
            _append_jsonl(
                scale_root / "diagnostics" / f"{split}_cooperative_updates.jsonl",
                update_record,
            )
        if update_policy:
            trainer.reconcile_action_execution(decision, next_info)
            if config.ppo_action_logprob_mode == "sampled":
                execution_log = scale_root / "diagnostics" / "ppo_action_execution.jsonl"
                for uav_id, sample in decision.samples.items():
                    diagnostic = decision.measurement_diagnostics.get(uav_id, {})
                    _append_jsonl(
                        execution_log,
                        {
                            "episode_id": episode_id,
                            "phase": phase,
                            "split": split,
                            "step": step,
                            "uav_id": uav_id,
                            "ppo_action_logprob_mode": config.ppo_action_logprob_mode,
                            "sampled_measurement_index": int(
                                diagnostic.get("sampled_measurement_index", sample.measurement_index)
                            ),
                            "executed_measurement_index": int(
                                diagnostic.get("executed_measurement_index", sample.executed_measurement_index)
                            ),
                            "action_execution_changed": int(
                                diagnostic.get("action_execution_changed", sample.action_execution_changed)
                            ),
                            "sampled_bandwidth": list(
                                diagnostic.get("sampled_bandwidth", np.asarray(sample.bandwidth, dtype=float).tolist())
                            ),
                            "executed_bandwidth": list(
                                diagnostic.get("executed_bandwidth", np.asarray(sample.executed_bandwidth, dtype=float).tolist())
                            ),
                            "sampled_delta_v": list(
                                diagnostic.get("sampled_delta_v", np.asarray(sample.delta_v, dtype=float).tolist())
                            ),
                            "executed_delta_v": list(
                                diagnostic.get("executed_delta_v", np.asarray(sample.executed_delta_v, dtype=float).tolist())
                            ),
                        },
                    )
            trainer.add_transition(
                decision,
                next_info.get("learning_rewards", rewards),
                next_info["central_state"],
                done=bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        rewards_history.extend(float(value) for value in rewards.values())
        learning_rewards_history.extend(
            float(value)
            for value in next_info.get("learning_rewards", rewards).values()
        )
        learning_components = next_info.get("learning_reward_components", {})
        if isinstance(learning_components, Mapping):
            learning_component_history.append(dict(learning_components))
        effective_edges_seen += len(next_info["central_state"]["effective_edges"])
        max_neighbor_updates = max(
            max_neighbor_updates,
            max(ekf.valid_neighbor_count for ekf in environment.ekfs.values()),
        )
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=decision.actions,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source="mappo",
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    episode_summary = writer.finalize()
    episode_summary.update(
        {
            "seed": seed,
            "mean_reward": float(np.mean(rewards_history)) if rewards_history else 0.0,
            "mean_learning_reward": (
                float(np.mean(learning_rewards_history))
                if learning_rewards_history
                else 0.0
            ),
            "cumulative_return": (
                float(np.sum(rewards_history) / max(1, config.uav_count))
                if rewards_history
                else 0.0
            ),
            "return_per_window": (
                float(
                    np.sum(rewards_history)
                    / max(1, len(rewards_history) // max(1, config.uav_count))
                    / max(1, config.uav_count)
                )
                if rewards_history
                else 0.0
            ),
            "learning_cumulative_return": (
                float(np.sum(learning_rewards_history) / max(1, config.uav_count))
                if learning_rewards_history
                else 0.0
            ),
            # Explicit aliases make the two reward channels easy to consume
            # without changing the historical cumulative-return fields.
            "reporting_reward_total": (
                float(np.sum(rewards_history) / max(1, config.uav_count))
                if rewards_history
                else 0.0
            ),
            "learning_reward_total": (
                float(np.sum(learning_rewards_history) / max(1, config.uav_count))
                if learning_rewards_history
                else 0.0
            ),
            "communication_applicable": bool(config.uav_count > 1),
            "single_uav_learning_mode": str(config.single_uav_learning_mode),
            "localization_progress_score": _finite_mean(
                float(row.get("localization_progress_score", 0.0))
                for row in learning_component_history
            ),
            "goal_progress_score": _finite_mean(
                float(row.get("goal_progress_score", 0.0))
                for row in learning_component_history
            ),
            "goal_distance_initial": (
                float(learning_component_history[0].get("previous_goal_distance", 0.0))
                if learning_component_history
                else 0.0
            ),
            "goal_distance_final": (
                float(learning_component_history[-1].get("current_goal_distance", 0.0))
                if learning_component_history
                else 0.0
            ),
            "effective_edges_seen": effective_edges_seen,
            "max_neighbor_updates": max_neighbor_updates,
            "collision_occurred": episode_summary.get("collision_occurred", False),
            "collision_step": episode_summary.get("collision_step", -1),
            "termination_reason": episode_summary.get("termination_reason", "running"),
            "mean_action_saturation": episode_summary.get("mean_action_saturation", 0.0),
            "max_action_saturation": episode_summary.get("max_action_saturation", 0.0),
            "update": update_index,
            "mean_policy_decision_ms": float(np.mean(inference_times) * 1000.0),
        }
    )
    update_stats: dict[str, float] | None = None
    if update_policy:
        batch = getattr(trainer, "_episode_batch_metadata", [])
        batch.append({"episode_id": episode_id, "windows": len(rewards_history) // config.uav_count,
                      "mean_reward": episode_summary["mean_reward"],
                      "mean_learning_reward": episode_summary["mean_learning_reward"],
                      "collision": bool(episode_summary.get("collision_occurred"))})
        trainer._episode_batch_metadata = batch
        _append_jsonl(scale_root / "diagnostics/training_episodes.jsonl", {
            **episode_summary, "training_episode": episode_index,
            "window_count": len(rewards_history) // config.uav_count,
            "cumulative_return": sum(rewards_history) / config.uav_count,
            "return_per_window": (
                sum(rewards_history) / max(1, len(rewards_history) // config.uav_count) / config.uav_count
            ),
            "learning_cumulative_return": sum(learning_rewards_history) / config.uav_count,
            "assigned_update": update_index,
        })
    if update_policy and flush_update:
        update_stats = trainer.update()
        if not all(np.isfinite(float(value)) for value in update_stats.values()):
            raise RuntimeError(f"MAPPO update returned non-finite statistics: {update_stats}")
        _append_jsonl(
            scale_root / "diagnostics" / "training_metrics.jsonl",
            {
                "uav_count": config.uav_count,
                "phase": phase,
                "episode": episode_id,
                "episode_id": episode_id,
                "update": update_index,
                "seed": seed,
                "reward": sum(b["windows"] * b["mean_reward"] for b in batch) / sum(b["windows"] for b in batch),
                "mean_reward": sum(b["windows"] * b["mean_reward"] for b in batch) / sum(b["windows"] for b in batch),
                "learning_reward": sum(b["windows"] * b["mean_learning_reward"] for b in batch) / sum(b["windows"] for b in batch),
                "mean_learning_reward": sum(b["windows"] * b["mean_learning_reward"] for b in batch) / sum(b["windows"] for b in batch),
                "learning_reward_total": episode_summary.get(
                    "learning_reward_total", episode_summary.get("learning_cumulative_return", 0.0)
                ),
                "reporting_reward_total": episode_summary.get(
                    "reporting_reward_total", episode_summary.get("cumulative_return", 0.0)
                ),
                "communication_applicable": bool(config.uav_count > 1),
                "single_uav_learning_mode": str(config.single_uav_learning_mode),
                "localization_progress_score": episode_summary.get(
                    "localization_progress_score", 0.0
                ),
                "goal_progress_score": episode_summary.get(
                    "goal_progress_score", 0.0
                ),
                "goal_distance_initial": episode_summary.get(
                    "goal_distance_initial", 0.0
                ),
                "goal_distance_final": episode_summary.get(
                    "goal_distance_final", 0.0
                ),
                "episode_ids": [b["episode_id"] for b in batch],
                "rollout_episode_count": len(batch),
                "rollout_window_count": sum(b["windows"] for b in batch),
                "completed_training_episodes": episode_index,
                "communication_multiplier_updates": int(config.measurement_communication_constraint_enabled),
                "collision_occurred": episode_summary.get("collision_occurred", False),
                "collision_step": episode_summary.get("collision_step", -1),
                "termination_reason": episode_summary.get("termination_reason", "running"),
                "outcome_class": episode_summary.get("outcome_class", "unknown_short"),
                "terminated": episode_summary.get("terminated", False),
                "truncated": episode_summary.get("truncated", False),
                "window_count": episode_summary.get(
                    "window_count", len(rewards_history) // max(1, config.uav_count)
                ),
                "max_steps": episode_summary.get("max_steps", config.max_steps),
                "minimum_clearance": episode_summary.get(
                    "minimum_clearance", float("nan")
                ),
                "pre_projection_minimum_clearance": episode_summary.get(
                    "pre_projection_minimum_clearance", float("nan")
                ),
                "collision_types": episode_summary.get("collision_types", []),
                "collision_objects": episode_summary.get("collision_objects", []),
                "safety_execution_mode": episode_summary.get(
                    "safety_execution_mode", "legacy"
                ),
                "mean_action_saturation": episode_summary.get("mean_action_saturation", 0.0),
                "max_action_saturation": episode_summary.get("max_action_saturation", 0.0),
                "safety_projection_count": episode_summary.get("safety_projection_count", 0),
                "boundary_projection_count": episode_summary.get("boundary_projection_count", 0),
                **update_stats,
            },
        )
        trainer._episode_batch_metadata = []
    _progress(
        config,
        f"[{config.uav_count} UAV][{phase}] episode {episode_index}/{episode_total} complete "
        + (("sealed generation; " if config.sealed_test_generation and split == "test" else f"mean_reward={episode_summary['mean_reward']:.6f} ")
         + f"elapsed={time.perf_counter() - started_at:.1f}s"),
    )
    metrics = ({"episode_id": episode_id, "split": "test", "phase": "sealed_generation", "seed": seed}
               if config.sealed_test_generation and split == "test"
               else _episode_metrics(scale_root / split / episode_id, episode_summary))
    if config.random_stream_mode == "indexed-v1" and metrics.get("accepted_update_count") == 0:
        for field in ("beneficial_fraction", "neutral_fraction", "harmful_fraction"):
            metrics[field] = "NA"
    return episode_summary, update_stats, attention_rows + [metrics]


def run_preliminary(
    config: FormalExperimentConfig,
    *,
    measurement_quality_model: Mapping[str, Any] | None = None,
    measurement_proxy_model: Mapping[str, Any] | None = None,
    measurement_persistent_proxy_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one independent UAV scale and write its preliminary artifacts."""

    _validate_scenario_pool(config)

    _progress(
        config,
        f"[{config.uav_count} UAV] start output={Path(config.output_dir) / f'scale_{config.uav_count}'} "
        f"train={config.train_episodes} validation={config.validation_episodes} test={config.test_episodes} "
        f"steps={config.max_steps}",
    )
    _seed_everything(config.seed + _SCALE_OFFSETS[config.uav_count])
    scale_root = _prepare_scale_output(config)
    scale_root.parent.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    normalizer, central_normalizer, metric_scales, warmup_summary = _fit_normalizer(
        config,
        measurement_quality_model=measurement_quality_model,
    )
    quality_model = warmup_summary.get("measurement_quality_model")
    visual_odometry_quality_model = warmup_summary.get("visual_odometry_quality_model")
    potential_statistics = warmup_summary.get("potential_statistics")
    accepted_feedback_statistics = warmup_summary.get(
        "accepted_update_feedback_statistics"
    )
    needs_potential_statistics = (
        config.learning_reward_mode == "progress_potential_v2"
        or config.learning_feedback_mode == "loc_progress_v2"
        or config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
    )
    if needs_potential_statistics:
        if not isinstance(potential_statistics, Mapping):
            raise RuntimeError(
                "learning feedback requires warm-up potential statistics"
            )
        config = replace(
            config,
            potential_statistics=copy.deepcopy(dict(potential_statistics)),
        )
    if config.learning_feedback_mode == "accepted_update_progress_v2":
        if not isinstance(accepted_feedback_statistics, Mapping):
            raise RuntimeError(
                "accepted_update_progress_v2 requires warm-up feedback statistics"
            )
        config = replace(
            config,
            accepted_update_feedback_location=float(
                accepted_feedback_statistics["accepted_update_feedback_location"]
            ),
            accepted_update_feedback_scale=float(
                accepted_feedback_statistics["accepted_update_feedback_scale"]
            ),
            accepted_update_feedback_none_baseline=float(
                accepted_feedback_statistics["accepted_update_feedback_none_baseline"]
            ),
        )
    if config.visual_odometry_quality_calibration_enabled:
        if not isinstance(visual_odometry_quality_model, Mapping):
            raise RuntimeError(
                "visual odometry quality calibration did not produce a warm-up model"
            )
        # The warm-up environment uses the historical fixed covariance while
        # collecting the fitting records.  All subsequent phases receive the
        # same frozen model; no validation/test refit is possible.
        config = replace(
            config,
            visual_odometry_quality_model=copy.deepcopy(dict(visual_odometry_quality_model)),
        )
    if config.frozen_normalization_path is not None:
        frozen = json.loads(Path(config.frozen_normalization_path).read_text(encoding="utf-8"))
        normalizer = ObservationNormalizer.from_state_dict(frozen)
        central_normalizer = FeatureNormalizer.from_state_dict(frozen["central_normalizer"])
        metric_scales = dict(frozen["metric_scales"])
        frozen_potential_statistics = frozen.get("potential_statistics")
        if needs_potential_statistics:
            if not isinstance(frozen_potential_statistics, Mapping):
                raise RuntimeError(
                    "frozen normalization is missing required potential statistics"
                )
            required_frozen_delta_scales = []
            if config.learning_feedback_mode == "loc_progress_v2":
                required_frozen_delta_scales.append("delta_u_scale")
            if config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}:
                required_frozen_delta_scales.extend(
                    ["delta_u_scale", "delta_goal_scale"]
                )
            for required_scale in dict.fromkeys(required_frozen_delta_scales):
                if (
                    required_scale not in frozen_potential_statistics
                    or not np.isfinite(
                        float(frozen_potential_statistics[required_scale])
                    )
                    or float(frozen_potential_statistics[required_scale]) <= 0.0
                ):
                    raise RuntimeError(
                        "frozen normalization is missing a finite positive "
                        f"{required_scale}"
                    )
            config = replace(
                config,
                potential_statistics=copy.deepcopy(dict(frozen_potential_statistics)),
                potential_scale_source="frozen",
            )
            # The payload is the provenance artifact consumed by later
            # phases.  Replace the warm-up probe statistics with the exact
            # frozen values before serialising it; otherwise a frozen
            # follower would run with one scale but report another.
            warmup_summary["potential_statistics"] = copy.deepcopy(
                dict(frozen_potential_statistics)
            )
            warmup_summary["potential_scale_source"] = "frozen"
        warmup_summary["normalization_source"] = config.frozen_normalization_path
    quality_enabled = config.mappo_variant in {
        "quality_aware_mappo",
        "budgeted_quality_aware_mappo",
        "rate_controlled_quality_aware_mappo",
        "adaptive_budgeted_quality_aware_mappo",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        "communication_constrained_proxy_guided_mappo",
        "persistent_proxy_guided_quality_mappo",
    }
    budget_enabled = config.mappo_variant in {
        "budgeted_quality_aware_mappo",
        "rate_controlled_quality_aware_mappo",
        "adaptive_budgeted_quality_aware_mappo",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        "communication_constrained_proxy_guided_mappo",
        "persistent_proxy_guided_quality_mappo",
    }
    persistent_enabled = bool(config.measurement_persistent_proxy_features_enabled)
    persistent_model = (
        copy.deepcopy(measurement_persistent_proxy_model)
        if measurement_persistent_proxy_model is not None
        else copy.deepcopy(config.measurement_persistent_proxy_model)
    )
    if persistent_enabled and not isinstance(persistent_model, Mapping):
        raise RuntimeError(
            "persistent-proxy MAPPO did not receive a train-fitted persistent proxy model"
        )
    proxy_enabled = bool(config.measurement_proxy_features_enabled)
    proxy_model = (
        copy.deepcopy(measurement_proxy_model)
        if measurement_proxy_model is not None
        else copy.deepcopy(config.measurement_proxy_model)
    )
    if proxy_enabled and not isinstance(proxy_model, Mapping):
        raise RuntimeError("proxy-guided MAPPO did not receive a train-fitted measurement proxy model")
    if quality_enabled and not isinstance(quality_model, Mapping):
        raise RuntimeError("quality-aware MAPPO did not receive a warm-up-fitted quality model")
    normalization_payload = normalizer.state_dict()
    normalization_payload["central_normalizer"] = central_normalizer.state_dict()
    normalization_payload["metric_scales"] = metric_scales
    normalization_payload["metric_scale_source"] = warmup_summary.get(
        "metric_scale_source", config.metric_scale_source
    )
    normalization_payload["metric_clip_fractions"] = warmup_summary.get("metric_clip_fractions", {})
    normalization_payload["potential_statistics"] = copy.deepcopy(
        warmup_summary.get("potential_statistics")
    )
    normalization_payload["potential_scale_source"] = str(
        warmup_summary.get("potential_scale_source", config.potential_scale_source)
    )
    normalization_payload["single_uav_braking_normalization_floors"] = copy.deepcopy(
        warmup_summary.get("single_uav_braking_normalization_floors", {})
    )
    normalization_payload["single_uav_braking_feature_scale_adjustments"] = copy.deepcopy(
        warmup_summary.get("single_uav_braking_feature_scale_adjustments", {})
    )
    normalization_payload["accepted_update_feedback_statistics"] = copy.deepcopy(
        accepted_feedback_statistics
    )
    normalization_payload["measurement_quality_model"] = quality_model
    normalization_payload["visual_odometry_quality_model"] = visual_odometry_quality_model
    normalization_payload["measurement_proxy_model"] = proxy_model
    normalization_payload["measurement_persistent_proxy_model"] = persistent_model
    _write_json(scale_root / "normalization_stats.json", normalization_payload)
    _write_json(scale_root / "checks" / "warmup_summary.json", warmup_summary)
    if quality_model is not None:
        _write_json(scale_root / "calibration" / "measurement_quality_model.json", quality_model)
    if proxy_model is not None:
        _write_json(scale_root / "calibration" / "measurement_proxy_model.json", proxy_model)
    if persistent_model is not None:
        _write_json(
            scale_root / "calibration" / "measurement_persistent_proxy_model.json",
            persistent_model,
        )
    if visual_odometry_quality_model is not None:
        _write_json(
            scale_root / "calibration" / "visual_odometry_quality_model.json",
            visual_odometry_quality_model,
        )

    from ..learning.mappo import MAPPOTrainer

    trainer_environment = _make_environment(
        config,
        _scale_seed(config.seed, config.uav_count, "train"),
        metric_scales=metric_scales,
    )
    trainer = MAPPOTrainer.from_environment(
        trainer_environment,
        seed=_scale_seed(config.seed, config.uav_count, "train"),
        normalizer=normalizer,
        central_normalizer=central_normalizer,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_output_dim=config.gat_output_dim,
        gat_heads=config.gat_heads,
        gat_layers=config.gat_layers,
        actor_hidden_dim=config.actor_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        max_obstacles=config.max_obstacles,
        policy_epochs=config.policy_epochs,
        minibatch_size=config.minibatch_size,
        learning_rate=config.learning_rate,
        max_grad_norm=float(config.max_grad_norm),
        actor_learning_rate_scale=float(config.actor_learning_rate_scale),
        entropy_coef=config.entropy_coef,
        initial_delta_std=config.initial_delta_std,
        clip_epsilon=config.clip_epsilon,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        critic_loss_type=config.critic_loss_type,
        value_clip_enabled=bool(config.value_clip_enabled),
        value_clip_epsilon=float(config.value_clip_epsilon),
        target_kl=(float(config.target_kl) if config.target_kl is not None else None),
        bootstrap_time_limits=bool(config.bootstrap_time_limits),
        episode_objective=config.episode_objective,
        include_remaining_time=config.include_remaining_time,
        ppo_action_logprob_mode=config.ppo_action_logprob_mode,
        node_feature_dim=config.node_feature_dim,
        encoder_type=config.encoder_type,
        measurement_head_mode=config.measurement_head_mode,
        gradient_clip_scope=config.gradient_clip_scope,
        measurement_proxy_full_packet_cost=config.measurement_proxy_full_packet_cost,
        auxiliary_gradient_diagnostics=config.auxiliary_gradient_diagnostics,
        parameter_initialization_seed=(
            int(config.seed) if config.branch_invariant_initialization else None
        ),
         device=config.device,
         use_measurement_quality_features=quality_enabled,
         use_measurement_budget_features=budget_enabled,
         measurement_rate_constraint_enabled=bool(config.measurement_rate_constraint_enabled),
         measurement_rate_constraint_coef=float(config.measurement_rate_constraint_coef),
         measurement_rate_tolerance=float(config.measurement_rate_tolerance),
         measurement_rate_target_by_uav=(
             dict(config.measurement_rate_target_by_uav)
             if config.measurement_rate_target_by_uav is not None
             else None
         ),
         measurement_rate_dual_enabled=bool(config.measurement_rate_dual_enabled),
         measurement_rate_dual_initial=float(config.measurement_rate_dual_initial),
         measurement_rate_dual_step_size=float(config.measurement_rate_dual_step_size),
         measurement_rate_dual_max=float(config.measurement_rate_dual_max),
         use_measurement_proxy_features=proxy_enabled,
         measurement_proxy_guidance_enabled=bool(config.measurement_proxy_guidance_enabled),
         measurement_proxy_teacher_coef=float(config.measurement_proxy_teacher_coef),
         measurement_proxy_temperature=float(config.measurement_proxy_temperature),
         measurement_proxy_communication_cost_weight=float(
             config.measurement_proxy_communication_cost_weight
         ),
         measurement_communication_constraint_enabled=bool(
             config.measurement_communication_constraint_enabled
         ),
         measurement_communication_cost_target_by_uav=(
             dict(config.measurement_communication_cost_target_by_uav)
             if config.measurement_communication_cost_target_by_uav is not None
             else None
         ),
         measurement_communication_cost_tolerance=float(
             config.measurement_communication_cost_tolerance
         ),
         measurement_communication_dual_initial=float(config.measurement_communication_dual_initial),
         measurement_communication_dual_step_size=float(config.measurement_communication_dual_step_size),
         measurement_communication_dual_max=float(config.measurement_communication_dual_max),
         measurement_communication_fixed_packet_fraction=float(
             config.measurement_communication_fixed_packet_fraction
         ),
         measurement_quality_logit_penalty=(
             float(config.measurement_quality_logit_penalty) if quality_enabled else 0.0
         ),
         measurement_quality_model=quality_model if quality_enabled else None,
         measurement_proxy_model=proxy_model if proxy_enabled else None,
         use_measurement_persistent_proxy_features=persistent_enabled,
         measurement_persistent_proxy_guidance_enabled=bool(
             config.measurement_persistent_proxy_guidance_enabled
         ),
         measurement_persistent_proxy_horizon=int(config.measurement_persistent_proxy_horizon),
         measurement_persistent_proxy_model=(
             persistent_model if persistent_enabled else None
         ),
         measurement_persistent_proxy_target_relative_cost=bool(
             config.measurement_persistent_proxy_target_relative_cost
         ),
         measurement_proxy_target_relative_cost=bool(
             config.measurement_proxy_target_relative_cost
         ),
      )
    initial_parameter_hashes = trainer.policy.initial_parameter_initialization_hashes()
    _write_json(
        scale_root / "diagnostics" / "initial_parameter_hashes.json",
        {
            "schema_version": "parameter-initialization-v1",
            "branch_invariant_initialization": bool(
                config.branch_invariant_initialization
            ),
            "parameter_initialization_seed": (
                int(config.seed) if config.branch_invariant_initialization else None
            ),
            "modules": initial_parameter_hashes,
            "shared_modules": [
                "gat",
                "actor_body",
                "bandwidth_head",
                "delta_mean_head",
                "delta_logstd",
                "critic",
            ],
        },
    )
    initial_parameters = _parameter_vector(trainer)
    episodes: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    if config.development_monitor_enabled:
        from .training_monitor import run_training_monitor
        run_training_monitor(config, trainer, scale_root, metric_scales, completed_episodes=0)
    for index in range(config.train_episodes):
        summary, _stats, rows = _run_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            phase="train",
            split="train",
            serial=index + 1,
            seed=_scale_seed(config.seed, config.uav_count, "train", index),
            update_policy=True,
            update_index=index // config.rollout_episodes_per_update + 1,
            flush_update=((index + 1) % config.rollout_episodes_per_update == 0 or index + 1 == config.train_episodes),
            episode_index=index + 1,
            episode_total=config.train_episodes,
            metric_scales=metric_scales,
        )
        episodes.append(summary)
        episode_metrics.append(rows[-1])
        attention_rows.extend(rows[:-1])
        if config.development_monitor_enabled and (index + 1) % 20 == 0:
            run_training_monitor(config, trainer, scale_root, metric_scales, completed_episodes=index + 1)

    update_count = math.ceil(config.train_episodes / config.rollout_episodes_per_update)
    _save_checkpoint(scale_root / "checkpoints" / "final.pt", trainer, config, update_count)
    final_parameters = _parameter_vector(trainer)
    parameter_delta = {
        "initial_l2": float(initial_parameters.norm().cpu()),
        "final_l2": float(final_parameters.norm().cpu()),
        "delta_l2": float((final_parameters - initial_parameters).norm().cpu()),
        "updates": update_count,
    }
    if parameter_delta["delta_l2"] <= 0.0:
        raise RuntimeError("MAPPO parameters did not change during preliminary training")
    _write_json(scale_root / "diagnostics" / "parameter_delta.json", parameter_delta)

    validation_episodes: list[dict[str, Any]] = []
    for index in range(config.validation_episodes):
        summary, _stats, rows = _run_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            phase="validation",
            split="validation",
            serial=400000 + index + 1,
            seed=_scale_seed(config.seed, config.uav_count, "validation", index),
            update_policy=False,
            update_index=None,
            episode_index=index + 1,
            episode_total=config.validation_episodes,
            metric_scales=metric_scales,
        )
        validation_episodes.append(summary)
        episode_metrics.append(rows[-1])
        attention_rows.extend(rows[:-1])

    test_episodes: list[dict[str, Any]] = []
    for index in range(config.test_episodes):
        summary, _stats, rows = _run_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            phase="test",
            split="test",
            serial=500000 + index + 1,
            seed=_scale_seed(config.seed, config.uav_count, "test", index),
            update_policy=False,
            update_index=None,
            episode_index=index + 1,
            episode_total=config.test_episodes,
            metric_scales=metric_scales,
        )
        test_episodes.append(summary)
        episode_metrics.append(rows[-1])
        attention_rows.extend(rows[:-1])

    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", episode_metrics)
    _write_episode_metrics(
        scale_root / "diagnostics" / "validation_episode_metrics.csv",
        [row for row in episode_metrics if str(row.get("split", "")) == "validation"],
    )
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)
    all_episodes = episodes + validation_episodes + test_episodes
    scale_manifest = _write_scale_manifest(scale_root=scale_root, config=config, episodes=all_episodes)
    _update_root_manifest(scale_root.parent, scale_manifest)
    profile = _scenario_profile(config)
    integration_summary = {
        "uav_count": config.uav_count,
        "algorithm_role": (
            "single_agent_ppo_sanity"
            if config.uav_count == 1
            else "shared_policy_mappo"
        ),
        "cooperative_links_expected": bool(config.uav_count > 1),
        "communication_applicable": bool(config.uav_count > 1),
        "single_uav_learning_mode": str(config.single_uav_learning_mode),
        "single_uav_learning_active": bool(
            config.uav_count == 1
            and config.single_uav_learning_mode in {"loc_goal_v1", "loc_goal_v2"}
        ),
        "single_uav_learning_weights": {
            str(key): float(value)
            for key, value in config.single_uav_learning_weights.items()
        },
        "single_uav_navigation_reward": {
            "goal_progress_coef": float(config.single_uav_goal_progress_coef),
            "localization_cost_weight": float(
                config.single_uav_localization_cost_weight
            ),
            "goal_bonus": float(config.single_uav_goal_bonus),
            "goal_threshold_m": float(config.single_uav_goal_threshold_m),
            "goal_speed_threshold_mps": float(
                config.single_uav_goal_speed_threshold_mps
            ),
            "near_goal_radius_m": float(config.single_uav_near_goal_radius_m),
            "near_goal_speed_penalty_weight": float(
                config.single_uav_near_goal_speed_penalty_weight
            ),
            "speed_control_mode": str(config.single_uav_speed_control_mode),
            "target_speed_far_mps": float(config.single_uav_target_speed_far_mps),
            "braking_acceleration_mps2": float(
                config.single_uav_braking_acceleration_mps2
            ),
            "braking_margin_m": float(config.single_uav_braking_margin_m),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "goal_hold_reward": float(config.single_uav_goal_hold_reward),
            "speed_deficit_penalty_weight": float(
                config.single_uav_speed_deficit_penalty_weight
            ),
            "goal_direction_coef": float(config.single_uav_goal_direction_coef),
            "braking_action_reward_coef": float(
                config.single_uav_braking_action_reward_coef
            ),
            "target_velocity_action_reward_coef": float(
                config.single_uav_target_velocity_action_reward_coef
            ),
            "braking_safety_projection_enabled": bool(
                config.single_uav_braking_safety_projection_enabled
            ),
            "braking_curriculum_fraction": float(
                config.single_uav_braking_curriculum_fraction
            ),
            "stable_goal_definition": (
                "estimated_goal_distance_at_or_below_goal_threshold_and_actual_speed_at_or_below_speed_threshold_for_configured_consecutive_hold_steps"
            ),
            "goal_termination": "loc_goal_v2_terminates_when_stable_goal_is_reached",
        },
        "motion_action_contract": MotionConfig().action_contract(),
        "scenario_mode": (
            "fixed_generated_geometry_seed"
            if config.scenario_seed is not None
            else "phase_seeded_generated_geometry"
        ),
        "scenario_seed": (
            int(config.scenario_seed) if config.scenario_seed is not None else None
        ),
        "train_episodes": len(episodes),
        "validation_episodes": len(validation_episodes),
        "test_episodes": len(test_episodes),
        "max_steps": config.max_steps,
        "parameter_delta": parameter_delta,
        "parameter_initialization": {
            "branch_invariant": bool(config.branch_invariant_initialization),
            "seed": (
                int(config.seed) if config.branch_invariant_initialization else None
            ),
            "module_hashes": initial_parameter_hashes,
            "shared_modules": [
                "gat",
                "actor_body",
                "bandwidth_head",
                "delta_mean_head",
                "delta_logstd",
                "critic",
            ],
        },
        "schema_version": EPISODE_SCHEMA_VERSION,
        "calibration_status": "synthetic_configured; no_external_dataset; preliminary_only",
        "result_status": (
            "single_uav_sanity_only"
            if config.uav_count == 1
            else (
                "moderate_profile_closed_loop_pilot"
                if profile is not None
                else "preliminary_only"
            )
        ),
        "mappo_variant": config.mappo_variant,
        "node_feature_dim": config.node_feature_dim,
        "dynamic_obstacle_motion_features_enabled": bool(
            config.dynamic_obstacle_motion_features_enabled
        ),
        "safety_execution": {
            "mode": str(config.safety_execution_mode),
            "dynamic_obstacle_swept_projection_enabled": bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
            "communication_occluder": {
                "screen_id": "screening_moving_nlos_screen",
                "physical_collision_enabled": bool(
                    config.communication_occluder_physical_collision_enabled
                ),
                "affects_los": True,
                "excluded_from": [
                    "physical_collision",
                    "clearance_violation",
                    "safety_projection",
                    "dynamic_physical_obstacle_features",
                ]
                if not config.communication_occluder_physical_collision_enabled
                else [],
            },
        },
        "visual_odometry_quality_calibration": {
            "enabled": bool(config.visual_odometry_quality_calibration_enabled),
            "model_source": (
                "warmup_only_tracking_quality_and_execution_time_vo_innovation"
                if config.visual_odometry_quality_calibration_enabled
                else "disabled; fixed SensorConfig covariance"
            ),
            "calibration_file": (
                "calibration/visual_odometry_quality_model.json"
                if config.visual_odometry_quality_calibration_enabled
                else None
            ),
            "forbidden_inputs": [
                "true_position",
                "true_error",
                "physical_los_nlos",
                "future_state",
                "future_packet_outcome",
            ],
        },
        "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
        "measurement_budget_enabled": bool(config.measurement_budget_enabled),
        "measurement_budget_by_uav": (
            {str(key): int(value) for key, value in config.measurement_budget_by_uav.items()}
            if config.measurement_budget_by_uav is not None
            else None
        ),
        "measurement_rate_constraint": {
            "enabled": bool(config.measurement_rate_constraint_enabled),
            "coefficient": float(config.measurement_rate_constraint_coef),
            "tolerance": float(config.measurement_rate_tolerance),
            "target_by_uav": (
                {str(key): float(value) for key, value in config.measurement_rate_target_by_uav.items()}
                if config.measurement_rate_target_by_uav is not None
                else None
            ),
            "fit_source": (
                "baseline_mappo train split only"
                if config.measurement_rate_constraint_enabled
                else "disabled"
            ),
            "applies_during": "training_updates_only",
            "dual_enabled": bool(config.measurement_rate_dual_enabled),
            "dual_initial": float(config.measurement_rate_dual_initial),
            "dual_step_size": float(config.measurement_rate_dual_step_size),
            "dual_max": float(config.measurement_rate_dual_max),
        },
        "measurement_overhead": {
            "enabled": bool(config.measurement_overhead_enabled),
            "equivalent_packet_size_bytes": 512,
            "semantics": (
                "one equivalent packet per selected measurement action; queue/channel KPI only"
                if config.measurement_overhead_enabled
                else "disabled; no measurement-induced queue load"
            ),
            "mac_protocol_modeled": False,
            "automatic_retransmission_modeled": False,
        },
        "measurement_quality": {
             "enabled": quality_enabled,
             "logit_penalty": (
                 float(config.measurement_quality_logit_penalty) if quality_enabled else 0.0
             ),
             "penalty_semantics": (
                 "candidate_logit -= coefficient * risk * validity; none logit unchanged"
                 if quality_enabled
                 else "disabled for baseline_mappo"
             ),
             "fit_source": "warmup_actor_visible_observations_only",
            "calibration_file": "calibration/measurement_quality_model.json",
            "forbidden_inputs": ["true_position", "true_error", "physical_los_nlos", "future_state"],
        },
        "measurement_proxy": {
            "features_enabled": bool(config.measurement_proxy_features_enabled),
            "guidance_enabled": bool(config.measurement_proxy_guidance_enabled),
            "target_relative_cost": bool(config.measurement_proxy_target_relative_cost),
            "teacher_coefficient": float(config.measurement_proxy_teacher_coef),
            "temperature": float(config.measurement_proxy_temperature),
            "communication_cost_weight": float(
                config.measurement_proxy_communication_cost_weight
            ),
            "utility": (
                "information_gain_norm - communication_cost_weight * "
                "max(0, proxy_cost_norm - baseline_train_target_cost)"
                if config.measurement_proxy_target_relative_cost
                else "information_gain_norm - communication_cost_weight * proxy_cost_norm"
            ),
            "fit_source": (
                "baseline_mappo/train actor-visible observations only"
                if config.measurement_proxy_features_enabled
                else "disabled"
            ),
            "calibration_file": "calibration/measurement_proxy_model.json",
            "forbidden_inputs": [
                "true_position",
                "true_error",
                "physical_los_nlos",
                "future_state",
                "future_packet_outcome",
                "post_update_nis",
            ],
        },
        "measurement_persistent_proxy": {
            "features_enabled": bool(config.measurement_persistent_proxy_features_enabled),
            "guidance_enabled": bool(config.measurement_persistent_proxy_guidance_enabled),
            "horizon": int(config.measurement_persistent_proxy_horizon),
            "target_relative_cost": bool(
                config.measurement_persistent_proxy_target_relative_cost
            ),
            "utility": (
                "persistent_information_gain_norm - communication_cost_weight * "
                "max(0, proxy_cost_norm - baseline_train_target_cost)"
                if config.measurement_persistent_proxy_target_relative_cost
                else "persistent_information_gain_norm - communication_cost_weight * proxy_cost_norm"
            ),
            "fit_source": (
                "baseline_mappo/train actor-visible graph observations only"
                if config.measurement_persistent_proxy_features_enabled
                else "disabled"
            ),
            "calibration_file": "calibration/measurement_persistent_proxy_model.json",
            "gradient_scope": "measurement_head_only; detached teacher target",
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
        "measurement_budget": {
            "enabled": bool(config.measurement_budget_enabled),
            "limits": (
                {str(key): int(value) for key, value in config.measurement_budget_by_uav.items()}
                if config.measurement_budget_by_uav is not None
                else None
            ),
            "fit_source": (
                "baseline_mappo train split only"
                if config.measurement_budget_enabled
                else "disabled"
            ),
            "actor_feature": (
                "remaining budget ratio"
                if budget_enabled
                else "disabled"
            ),
        },
        "measurement_rate_dual_state": trainer.measurement_rate_dual_state_dict(),
        "measurement_communication_dual_state": trainer.measurement_communication_dual_state_dict(),
        "measurement_communication_constraint": {
            "enabled": bool(config.measurement_communication_constraint_enabled),
            "target_by_uav": (
                {str(key): float(value) for key, value in config.measurement_communication_cost_target_by_uav.items()}
                if config.measurement_communication_cost_target_by_uav is not None else None
            ),
            "tolerance": float(config.measurement_communication_cost_tolerance),
            "dual_initial": float(config.measurement_communication_dual_initial),
            "dual_step_size": float(config.measurement_communication_dual_step_size),
            "dual_max": float(config.measurement_communication_dual_max),
            "fixed_packet_cost_fraction": float(config.measurement_communication_fixed_packet_fraction),
            "fit_source": "baseline_mappo train split only" if config.measurement_communication_constraint_enabled else "disabled",
            "applies_during": "training_updates_only",
        },
        "scenario_profile": profile,
        "screening_geometry": profile["screening_geometry"] if profile else None,
        "profile_allowed_for_training": bool(profile and profile["allowed_for_training"]),
        "communication_parameters": asdict(_effective_channel_config(config)),
        "topology_parameters": asdict(_effective_topology_config(config)),
        "ekf_config": {
            "source": config.ekf_config_source,
            "parameters": {
                "vo_position_std_m": config.vo_position_std_m,
                "process_position_std_m": config.process_position_std_m,
                "process_velocity_std_mps": config.process_velocity_std_mps,
                "nlos_variance_multiplier": config.nlos_variance_multiplier,
            },
        },
        "command_line": config.command_line,
        "scale_root": str(scale_root),
    }
    _write_json(scale_root / "checks" / "integration_summary.json", integration_summary)
    _progress(config, f"[{config.uav_count} UAV] finished; artifacts={scale_root}")
    return {
        **integration_summary,
        "manifest": str(scale_root / "manifest.json"),
        "normalization_stats": str(scale_root / "normalization_stats.json"),
        "checkpoint": str(scale_root / "checkpoints" / "final.pt"),
    }


def _scale_directories(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.glob("scale_*")
            if path.is_dir()
            and path.name[6:].isdigit()
            and int(path.name[6:]) in _ALLOWED_UAV_COUNTS
        ],
        key=lambda path: int(path.name[6:]),
    )


def _plot_training_cumulative_reward(
    training_episode_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Plot the recorded per-training-episode cumulative rewards.

    ``training_episodes.jsonl`` is the source because it preserves the raw
    episode return before the update-level aggregation used by
    ``training_metrics.jsonl``.  No smoothing or cross-episode averaging is
    applied here.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#7c3aed", 3: "#2563eb", 5: "#16a34a", 7: "#dc2626"}
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    scales = sorted({int(_number(row, "uav_count", 0.0)) for row in training_episode_rows})
    for count in scales:
        selected = [
            row
            for row in training_episode_rows
            if int(_number(row, "uav_count", 0.0)) == count
        ]
        selected.sort(key=lambda row: int(_number(row, "training_episode", 0.0)))
        x = [
            int(_number(row, "training_episode", index + 1))
            for index, row in enumerate(selected)
        ]
        color = colors.get(count, "#7c3aed")
        axis.plot(
            x,
            [_number(row, "cumulative_return", 0.0) for row in selected],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=f"{count} UAV report return",
        )
        if any("learning_cumulative_return" in row for row in selected):
            axis.plot(
                x,
                [_number(row, "learning_cumulative_return", 0.0) for row in selected],
                color=color,
                linewidth=1.3,
                linestyle="--",
                marker=".",
                markersize=4,
                label=f"{count} UAV learning return",
            )
    axis.set_title("Training cumulative reward")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Cumulative reward per episode")
    axis.grid(alpha=0.25)
    if scales:
        axis.legend()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_mappo_training_loss(
    training_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Plot total, actor, and critic loss from each MAPPO update."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#7c3aed", 3: "#2563eb", 5: "#16a34a", 7: "#dc2626"}
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    actor_axis = axis.twinx()
    scales = sorted({int(_number(row, "uav_count", 0.0)) for row in training_rows})
    styles = (("loss", "total loss", "-"), ("critic_loss", "critic loss", ":"))
    for count in scales:
        selected = [
            row
            for row in training_rows
            if int(_number(row, "uav_count", 0.0)) == count
        ]
        selected.sort(key=lambda row: int(_number(row, "update", 0.0)))
        x = [
            int(_number(row, "update", index + 1))
            for index, row in enumerate(selected)
        ]
        color = colors.get(count, "#7c3aed")
        for metric, label, linestyle in styles:
            axis.plot(
                x,
                [_number(row, metric, 0.0) for row in selected],
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                marker="o" if metric == "loss" else None,
                markersize=3,
                label=f"{count} UAV {label}",
            )
        actor_axis.plot(
            x,
            [_number(row, "actor_loss", 0.0) for row in selected],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{count} UAV actor loss",
        )
    axis.set_title("MAPPO training loss")
    axis.set_xlabel("PPO update")
    axis.set_ylabel("Total / critic loss")
    actor_axis.set_ylabel("Actor loss")
    axis.grid(alpha=0.25)
    if scales:
        left_handles, left_labels = axis.get_legend_handles_labels()
        right_handles, right_labels = actor_axis.get_legend_handles_labels()
        axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            ncol=2,
            loc="upper right",
        )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize_preliminary(
    output_dir: str | Path,
    *,
    logs_output: str | Path | None = None,
    plots_output: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate metrics and create reproducible scale or training-trend PNG plots."""

    root = Path(output_dir)
    scale_dirs = _scale_directories(root)
    if not scale_dirs:
        raise FileNotFoundError(
            f"no supported scale directories ({', '.join(f'scale_{count}' for count in _ALLOWED_UAV_COUNTS)}) found under {root}"
        )
    result_name = root.name
    log_root = Path(logs_output) if logs_output is not None else Path("logs/preliminary") / result_name
    plot_root = Path(plots_output) if plots_output is not None else Path("plots/preliminary") / result_name
    log_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    training_episode_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for scale_dir in scale_dirs:
        metrics_path = scale_dir / "diagnostics" / "episode_metrics.csv"
        if metrics_path.exists():
            episode_rows.extend(_read_csv(metrics_path))
        training_episode_path = scale_dir / "diagnostics" / "training_episodes.jsonl"
        if training_episode_path.exists():
            with training_episode_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        training_episode_rows.append(json.loads(line))
        training_path = scale_dir / "diagnostics" / "training_metrics.jsonl"
        if training_path.exists():
            with training_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        training_rows.append(json.loads(line))
    _write_episode_metrics(log_root / "episode_metrics.csv", episode_rows)
    training_episode_columns = sorted(
        {key for row in training_episode_rows for key in row}
    )
    _write_csv(
        log_root / "training_episodes.csv",
        training_episode_columns,
        training_episode_rows,
    )
    training_columns = sorted({key for row in training_rows for key in row})
    _write_csv(log_root / "training_metrics.csv", training_columns, training_rows)

    test_rows = [row for row in episode_rows if row.get("split") == "test"]
    summary_columns = [
        "uav_count",
        "test_episodes",
        "mean_reward",
        "position_rmse_m",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "collision_rate",
    ]
    scale_summary: list[dict[str, Any]] = []
    for count in sorted({int(row["uav_count"]) for row in test_rows}):
        selected = [row for row in test_rows if int(row["uav_count"]) == count]
        scale_summary.append(
            {
                "uav_count": count,
                "test_episodes": len(selected),
                **{
                    metric: _finite_mean(float(row.get(metric, 0.0)) for row in selected)
                    for metric in summary_columns[2:]
                },
            }
        )
    _write_csv(log_root / "scale_summary.csv", summary_columns, scale_summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#7c3aed", 3: "#2563eb", 5: "#16a34a", 7: "#dc2626"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for count in sorted({int(row["uav_count"]) for row in training_rows}):
        selected = [row for row in training_rows if int(row["uav_count"]) == count]
        selected.sort(key=lambda row: int(row.get("update", 0)))
        x = [int(row.get("update", index + 1)) for index, row in enumerate(selected)]
        axes[0].plot(x, [float(row.get("mean_reward", 0.0)) for row in selected], label=f"{count} UAV", color=colors[count])
        axes[1].plot(x, [float(row.get("loss", 0.0)) for row in selected], label=f"{count} UAV", color=colors[count])
    axes[0].set_title("Training reward")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Mean team reward")
    axes[1].set_title("MAPPO loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(plot_root / "learning_curves.png", dpi=180)
    plt.close(figure)

    _plot_training_cumulative_reward(
        training_episode_rows,
        plot_root / "training_cumulative_reward.png",
    )
    _plot_mappo_training_loss(
        training_rows,
        plot_root / "mappo_training_loss.png",
    )

    metric_names = ["mean_reward", "position_rmse_m", "packet_loss", "connectivity"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, metric_names):
        values = [float(row.get(metric, 0.0)) for row in scale_summary]
        labels = [str(row["uav_count"]) for row in scale_summary]
        axis.bar(labels, values, color=[colors[int(label)] for label in labels])
        axis.set_title(metric)
        axis.set_xlabel("UAV count")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(plot_root / "scale_comparison.png", dpi=180)
    plt.close(figure)

    communication_metric_names = ["packet_loss", "delay_s", "aoi_s", "position_rmse_m"]
    communication_plot_mode = "test_scale_comparison"
    communication_plot_source = "scale_summary.csv"
    communication_plot_split = "test"
    communication_plot_x_axis = "uav_count"
    communication_plot_x_label = "UAV count"
    communication_plot_aggregation = "mean over test episodes for each UAV scale"
    communication_plot_rows: list[Mapping[str, Any]] = scale_summary
    communication_plot_x_values: list[int] = [int(row["uav_count"]) for row in scale_summary]

    # A single scale cannot form a meaningful cross-scale line.  For the
    # single-UAV workflow, expose the per-training-episode diagnostics instead.
    # The train split is intentionally used here: this is a learning-trend
    # diagnostic, not a replacement for the held-out test summary above.
    if len(scale_dirs) == 1:
        single_scale = int(scale_dirs[0].name[6:])
        training_trend_rows = sorted(
            [
                row
                for row in episode_rows
                if str(row.get("split", "")) == "train"
                and str(row.get("uav_count", "")) == str(single_scale)
            ],
            key=lambda row: str(row.get("episode_id", "")),
        )
        if training_trend_rows:
            communication_plot_mode = "single_scale_training_episode_trend"
            communication_plot_source = "diagnostics/episode_metrics.csv"
            communication_plot_split = "train"
            communication_plot_x_axis = "training_episode"
            communication_plot_x_label = "Training episode"
            communication_plot_aggregation = (
                "one row per training episode; no cross-episode averaging"
            )
            communication_plot_rows = training_trend_rows
            communication_plot_x_values = list(range(1, len(training_trend_rows) + 1))

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, communication_metric_names):
        values = [_number(row, metric) for row in communication_plot_rows]
        axis.plot(
            communication_plot_x_values,
            values,
            marker="o",
            linewidth=1.8,
            color="#7c3aed",
        )
        axis.set_title(metric)
        axis.set_xlabel(communication_plot_x_label)
        axis.grid(alpha=0.25)
    figure.savefig(plot_root / "communication_localization_metrics.png", dpi=180)
    plt.close(figure)

    single_uav_diagnostics: dict[str, Any] | None = None
    single_uav_navigation_analysis: dict[str, Any] | None = None
    if len(scale_dirs) == 1 and int(scale_dirs[0].name[6:]) == 1:
        test_episode_paths = sorted(
            path
            for path in (scale_dirs[0] / "test").glob("episode_*")
            if path.is_dir()
        )
        if test_episode_paths:
            required_diagnostic_files = (
                "scenario.json",
                "trajectory.csv",
                "ground_truth.csv",
                "estimates.csv",
                "rewards.csv",
                "obstacles.csv",
            )
            selected_episode = test_episode_paths[0]
            if all((selected_episode / name).is_file() for name in required_diagnostic_files):
                from .single_uav_diagnostics import (
                    SingleUAVDiagnosticsConfig,
                    plot_single_uav_diagnostics,
                )

                single_uav_diagnostics = plot_single_uav_diagnostics(
                    SingleUAVDiagnosticsConfig(
                        episode_path=selected_episode,
                        output_dir=plot_root / f"single_uav_diagnostics_{selected_episode.name}",
                        force=True,
                    )
                )
        single_scale_mode = next(
            (
                str(row.get("single_uav_learning_mode", "off"))
                for row in episode_rows
                if str(row.get("uav_count", "")) == "1"
                and str(row.get("split", "")) in {"train", "validation", "test"}
            ),
            "off",
        )
        if single_scale_mode == "loc_goal_v2":
            from .single_uav_navigation_analysis import analyze_single_uav_navigation

            single_uav_navigation_analysis = analyze_single_uav_navigation(
                scale_dirs[0],
                output_dir=plot_root / "single_uav_navigation_analysis",
                metrics_output_path=log_root / "single_uav_navigation_analysis.csv",
                force=True,
            )

    plot_paths: dict[str, Any] = {
        "learning_curves": str(plot_root / "learning_curves.png"),
        "training_cumulative_reward": str(
            plot_root / "training_cumulative_reward.png"
        ),
        "mappo_training_loss": str(plot_root / "mappo_training_loss.png"),
        "scale_comparison": str(plot_root / "scale_comparison.png"),
        "communication_localization_metrics": str(
            plot_root / "communication_localization_metrics.png"
        ),
    }
    if single_uav_diagnostics is not None:
        plot_paths["single_uav_diagnostics"] = single_uav_diagnostics
    if single_uav_navigation_analysis is not None:
        plot_paths["single_uav_navigation_analysis"] = single_uav_navigation_analysis

    summary = {
        "output_dir": str(root),
        "scales": [int(path.name[6:]) for path in scale_dirs],
        "episode_rows": len(episode_rows),
        "training_episode_rows": len(training_episode_rows),
        "training_rows": len(training_rows),
        "logs": {
            "episode_metrics": str(log_root / "episode_metrics.csv"),
            "training_episodes": str(log_root / "training_episodes.csv"),
            "training_metrics": str(log_root / "training_metrics.csv"),
            "scale_summary": str(log_root / "scale_summary.csv"),
            "single_uav_navigation_analysis": (
                str(log_root / "single_uav_navigation_analysis.csv")
                if single_uav_navigation_analysis is not None
                else None
            ),
        },
        "plots": plot_paths,
        "figure_contract": {
            "training_cumulative_reward": {
                "source": "diagnostics/training_episodes.jsonl",
                "split": "train",
                "x_axis": "training_episode",
                "points": len(training_episode_rows),
                "aggregation": "one row per training episode; no smoothing or cross-episode averaging",
                "denominator": "uav_count; for the single-UAV run this is one",
                "claim_boundary": "training-process diagnostic; not held-out test evidence",
            },
            "mappo_training_loss": {
                "source": "diagnostics/training_metrics.jsonl",
                "split": "train",
                "x_axis": "ppo_update",
                "points": len(training_rows),
                "series": ["loss", "critic_loss", "actor_loss"],
                "y_axes": {
                    "left": ["loss", "critic_loss"],
                    "right": ["actor_loss"],
                },
                "raw": True,
                "smoothing": "none",
                "aggregation": "one row per PPO update; values are the trainer-reported update means",
                "claim_boundary": "optimization diagnostic; loss decrease alone does not establish policy improvement",
            },
            "communication_localization_metrics": {
                "mode": communication_plot_mode,
                "source": communication_plot_source,
                "split": communication_plot_split,
                "x_axis": communication_plot_x_axis,
                "x_label": communication_plot_x_label,
                "points": len(communication_plot_rows),
                "aggregation": communication_plot_aggregation,
                "claim_boundary": (
                    "single-scale training diagnostic; not held-out test evidence"
                    if communication_plot_mode == "single_scale_training_episode_trend"
                    else "test aggregate comparison across UAV scales"
                ),
            },
            "single_uav_navigation_analysis": {
                "source": "train/validation/test episode rewards.csv",
                "split": "train + validation + test",
                "x_axes": {
                    "minimum_goal_distance": "episode index within split",
                    "entry_rates": "split",
                    "near_goal_speed": "split, conditional on entering the near-goal region",
                    "speed_distance_relation": "estimated_goal_distance_m",
                    "braking_action_summary": "split, conditional on action samples inside the near-goal region",
                },
                "action_space_audit": {
                    "source": "train/validation/test episode actions.csv",
                    "braking_definition": "negative projection of applied delta_v onto pre-action velocity",
                    "effective_braking_definition": "braking action with post-action speed below pre-action speed",
                },
                "raw": True,
                "smoothing": "none",
                "claim_boundary": "single-seed navigation failure diagnostic; not manuscript-level convergence or generalization evidence",
            }
        },
        "result_status": (
            "engineering_single_uav_sanity_only"
            if scale_dirs and all(int(path.name[6:]) == 1 for path in scale_dirs)
            else "engineering_preliminary_only"
        ),
    }
    _write_json(root / "summary.json", summary)
    return summary


def validate_preliminary(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    scale_reports: dict[str, Any] = {}
    for scale_dir in _scale_directories(root):
        scale_reports[scale_dir.name] = validate_episode_benchmark(scale_dir)
    plots = {
        name: (root / "summary.json").exists()
        for name in ("summary.json",)
    }
    ok = bool(scale_reports) and all(report["ok"] for report in scale_reports.values())
    return {
        "path": str(root),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "scales": scale_reports,
        "summary_present": plots["summary.json"],
        "ok": ok,
    }


__all__ = [
    "FormalExperimentConfig",
    "fit_shared_training_metric_scales",
    "default_ekf_parameters",
    "load_selected_ekf_config",
    "run_preliminary",
    "summarize_preliminary",
    "validate_preliminary",
]
