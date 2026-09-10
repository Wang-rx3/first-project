"""Pure Python multi-UAV communication-localization environment.

This environment is deliberately simulator-independent.  It provides the
state transition and observation contract required by a future MAPPO trainer;
it does not contain a policy optimizer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..synthetic.channel_model import ChannelConfig
from ..schema import EPISODE_SCHEMA_VERSION
from .ekf import EKFConfig, LocalEKF, STATE_DIM, covariance_intersection
from .graph_interface import GraphBatch, build_graph_batch
from .motion_model import MotionConfig, UAVKinematicState, apply_velocity_increment
from .obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    collision_event,
    is_physical_obstacle,
    joint_motion_collision_diagnostics,
    moving_obstacle_swept_segment_intersects,
    point_obstacle_distance,
    segment_intersects_obstacle,
)
from .sensor_model import SensorConfig, sample_gnss, sample_imu, sample_visual_odometry
from ..learning.visual_odometry_quality import covariance_for_quality
from .topology import TopologyConfig, TopologySnapshot, build_topology
from .random_streams import indexed_rng


MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES = 512


def _as_bool(value: Any, default: bool = True) -> bool:
    """Parse scenario booleans without treating the string ``"false"`` as true."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "off", "disabled"}:
            return False
        if normalized in {"true", "1", "yes", "on", "enabled"}:
            return True
    return bool(default)


@dataclass(frozen=True)
class RewardConfig:
    uncertainty_scale: float = 100.0
    aoi_scale: float = 1.0
    delay_scale: float = 1.0
    collision_penalty: float = 10.0
    energy_weight: float = 0.01
    # When supplied, these are fitted only from the training split (normally
    # the training warm-up trajectories). Validation and test environments
    # receive the same values and never refit them.
    metric_scales: Mapping[str, float] | None = None
    metric_scale_source: str = "default"
    # ``p95`` preserves the historical one-sided normalization.  The
    # warm-up-quantile mode is an opt-in, training-only re-scaling for the
    # proposed reward-stability experiment; it maps the observed uncertainty
    # interval to [0, 1] while retaining the physical reporting quantity.
    uncertainty_normalization: str = "p95"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "uncertainty": 0.35,
            "loss": 0.20,
            "aoi": 0.15,
            "delay": 0.15,
            "connectivity": 0.15,
        }
    )

    # Optional single-UAV learning channel.  ``off`` preserves the historical
    # learning reward exactly.  ``loc_goal_v1`` is enabled explicitly for the
    # localization-first single-UAV baseline; it never changes the reporting
    # reward or the physical communication metrics.
    single_uav_learning_mode: str = "off"
    single_uav_learning_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "localization_progress": 0.8,
            "goal_progress": 0.2,
        }
    )

    # The historical reporting reward is kept intact.  The optional learning
    # channel is bounded and is passed to PPO separately by the formal runner.
    learning_reward_mode: str = "legacy"
    learning_reward_clip: float = 1.0
    learning_collision_penalty_cap: float = 1.0
    learning_potential_coef: float = 0.25
    learning_gamma: float = 0.99
    # Optional v4 training-only feedback channels.  The reporting reward is
    # deliberately independent of these fields so validation metrics retain
    # the physical objective definition used by historical runs.
    learning_feedback_mode: str = "v3"
    loc_progress_coef: float = 0.0
    actual_measurement_cost_enabled: bool = False
    actual_measurement_cost_target: float | None = None
    actual_measurement_cost_coef: float = 0.0
    # Training-only delayed credit for an accepted cooperative update.  The
    # credit is computed from the EKF's pre-restoration linear covariance
    # reduction and innovation consistency; it never uses truth or RMSE.
    accepted_update_feedback_coef: float = 0.0
    accepted_update_feedback_delay_steps: int = 1
    accepted_update_feedback_location: float = 0.0
    accepted_update_feedback_scale: float = 1.0
    accepted_update_feedback_none_baseline: float = 0.0
    # Frozen warm-up statistics for the optional training-only progress
    # potential.  Reporting objectives and their physical units are not
    # changed by this mapping.
    potential_statistics: Mapping[str, Any] | None = None
    potential_scale_source: str = "default"
    potential_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.single_uav_learning_mode not in {"off", "loc_goal_v1"}:
            raise ValueError(
                "single_uav_learning_mode must be off or loc_goal_v1"
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
        if self.learning_reward_mode == "report_direct" and (
            self.learning_feedback_mode != "v3" or self.learning_potential_coef != 0.0
            or self.loc_progress_coef != 0.0 or self.actual_measurement_cost_enabled
            or self.actual_measurement_cost_coef != 0.0
            or self.accepted_update_feedback_coef != 0.0
        ):
            raise ValueError("report_direct forbids additional learning feedback")
        if self.uncertainty_normalization not in {"p95", "warmup_q05_q95"}:
            raise ValueError(
                "uncertainty_normalization must be p95 or warmup_q05_q95"
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
        for name, value in (
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
        if self.learning_feedback_mode == "loc_progress_v2":
            if self.potential_statistics is None:
                raise ValueError(
                    "loc_progress_v2 requires warm-up-fitted potential statistics"
                )
            if not isinstance(self.potential_statistics, Mapping):
                raise ValueError(
                    "loc_progress_v2 requires potential_statistics to be a mapping"
                )
            if "delta_u_scale" not in self.potential_statistics:
                raise ValueError(
                    "loc_progress_v2 requires potential_statistics[delta_u_scale]"
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
        for name, value in (
            ("learning_reward_clip", self.learning_reward_clip),
            ("learning_collision_penalty_cap", self.learning_collision_penalty_cap),
            ("learning_potential_coef", self.learning_potential_coef),
            ("learning_gamma", self.learning_gamma),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < float(self.learning_gamma) <= 1.0:
            raise ValueError("learning_gamma must be in (0, 1]")
        if not np.isfinite(float(self.potential_epsilon)) or float(self.potential_epsilon) <= 0.0:
            raise ValueError("potential_epsilon must be finite and positive")
        if self.potential_statistics is not None:
            if not isinstance(self.potential_statistics, Mapping):
                raise ValueError("potential_statistics must be a mapping when supplied")
            for key in ("u_center", "u_scale", "c_center", "c_scale"):
                if key not in self.potential_statistics:
                    raise ValueError(f"potential_statistics is missing {key}")
                if not np.isfinite(float(self.potential_statistics[key])):
                    raise ValueError(f"potential_statistics[{key}] must be finite")
            if float(self.potential_statistics["u_scale"]) <= 0.0 or float(
                self.potential_statistics["c_scale"]
            ) <= 0.0:
                raise ValueError("potential statistics scales must be positive")
            if "delta_u_scale" in self.potential_statistics and (
                not np.isfinite(float(self.potential_statistics["delta_u_scale"]))
                or float(self.potential_statistics["delta_u_scale"]) <= 0.0
            ):
                raise ValueError("potential_statistics[delta_u_scale] must be positive")
            if "delta_goal_scale" in self.potential_statistics and (
                not np.isfinite(float(self.potential_statistics["delta_goal_scale"]))
                or float(self.potential_statistics["delta_goal_scale"]) <= 0.0
            ):
                raise ValueError(
                    "potential_statistics[delta_goal_scale] must be positive"
                )
            if self.single_uav_learning_mode == "loc_goal_v1":
                for key in ("delta_u_scale", "delta_goal_scale"):
                    if key not in self.potential_statistics:
                        raise ValueError(
                            "loc_goal_v1 requires "
                            f"potential_statistics[{key}]"
                        )

    @property
    def lambda_coll(self) -> float:
        return float(self.collision_penalty)

    @property
    def lambda_energy(self) -> float:
        return float(self.energy_weight)


@dataclass
class MessageRecord:
    """One delayed neighbor-state packet kept by the environment."""

    source: str
    target: str
    state: np.ndarray
    covariance: np.ndarray
    send_time_s: float
    arrival_step: int
    delay_s: float
    arrival_time_s: float | None = None
    # ``state`` and ``covariance`` are kept at ``last_propagated_time_s``.
    # ``send_time_s`` remains immutable and identifies the packet, so a
    # packet is not fused repeatedly as if it were a new observation.
    last_propagated_time_s: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    random_stream_mode: str = "legacy"
    uav_count: int = 3
    max_steps: int = 100
    seed: int = 2026
    motion: MotionConfig = field(default_factory=MotionConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    gnss_mode: str = "nominal"
    # Mechanism-audit switch.  Communication, message buffering, UWB sensing
    # and graph visibility remain active when this is false; only the local
    # EKF consumption of an otherwise eligible neighbor range/state pair is
    # blocked.  The default preserves the research environment behaviour.
    cooperative_ekf_enabled: bool = True
    # Validation-only resource-coupling switch.  When enabled, each selected
    # measurement adds one equivalent 512-byte packet to the selected
    # source->target queue for the current decision window.  It does not alter
    # EKF message-arrival semantics and is disabled for all existing runs.
    measurement_overhead_enabled: bool = False
    # Fixed per-UAV measurement budget fitted from the baseline training split.
    # The same limits are reused for training, validation, and test.
    measurement_budget_enabled: bool = False
    measurement_budget_by_uav: Mapping[str, int] | None = None
    # When enabled, expose the known per-link UWB update schedule to the
    # actor and reject repeated selections before the physical update period
    # has elapsed.  The default is deliberately disabled so historical
    # experiments retain their exact action semantics.
    measurement_update_eligibility_enabled: bool = False
    # Safety projection is part of the physical execution model.  The
    # diagnostic screening experiment can disable it in a separate replay to
    # reveal pre-projection risk; the default preserves all existing runs.
    safety_projection_enabled: bool = True
    # Optional actor-visible dynamic-obstacle motion features.  Keeping this
    # disabled preserves the historical four-value obstacle aggregate and its
    # 37-dimensional node-feature contract; the safety smoke explicitly
    # enables the extended eight-value aggregate.
    dynamic_obstacle_motion_features_enabled: bool = False
    # Optional one-step swept safety envelope for moving obstacles.  This is
    # disabled by default so historical experiments retain their exact
    # execution semantics; the corrected safety study enables it explicitly.
    dynamic_obstacle_swept_projection_enabled: bool = False
    # Versioned execution contract. ``legacy`` preserves historical
    # per-UAV/current-obstacle projection; ``joint_swept_v1`` evaluates all
    # candidate UAV motions together against predicted obstacle geometry.
    safety_execution_mode: str = "legacy"
    # Optional frozen warm-up-fitted mapping from execution-time VO tracking
    # quality to a conservative measurement covariance. ``None`` preserves
    # the historical fixed VO covariance exactly.
    visual_odometry_quality_calibration_enabled: bool = False
    visual_odometry_quality_model: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.random_stream_mode not in {"legacy", "indexed-v1"}:
            raise ValueError("unknown random stream mode")
        if self.safety_execution_mode not in {"legacy", "joint_swept_v1"}:
            raise ValueError("safety_execution_mode must be legacy or joint_swept_v1")
        if self.uav_count < 1:
            raise ValueError("uav_count must be at least 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.measurement_budget_enabled and self.measurement_budget_by_uav is None:
            raise ValueError(
                "measurement_budget_by_uav is required when measurement_budget_enabled is true"
            )
        if self.measurement_budget_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            actual = {str(key) for key in self.measurement_budget_by_uav}
            if actual != expected:
                raise ValueError(
                    "measurement_budget_by_uav must contain exactly all UAV identifiers"
                )
            for uav_id, value in self.measurement_budget_by_uav.items():
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                    raise ValueError(f"measurement budget for {uav_id} must be a non-negative integer")
        if self.visual_odometry_quality_model is not None and not isinstance(
            self.visual_odometry_quality_model, Mapping
        ):
            raise ValueError("visual_odometry_quality_model must be a mapping when supplied")
        if (
            self.visual_odometry_quality_calibration_enabled
            and self.visual_odometry_quality_model is None
        ):
            raise ValueError(
                "visual_odometry_quality_model is required when visual odometry quality calibration is enabled"
            )


def _state_vector(state: UAVKinematicState, accel_bias: np.ndarray, gyro_bias: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            state.position_m,
            state.velocity_mps,
            state.attitude_rad,
            accel_bias,
            gyro_bias,
        ]
    ).astype(float)


class MultiUAVJointEnv:
    """A deterministic multi-agent environment with local observations."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.uav_ids = [f"uav_{index:02d}" for index in range(self.config.uav_count)]
        self.states: dict[str, UAVKinematicState] = {}
        self.goals: dict[str, np.ndarray] = {}
        self.ekfs: dict[str, LocalEKF] = {}
        self.accel_bias: dict[str, np.ndarray] = {}
        self.gyro_bias: dict[str, np.ndarray] = {}
        self.obstacles: list[Obstacle] = []
        self.queue_state: dict[tuple[str, str], float] = {}
        self.aoi_state: dict[tuple[str, str], float] = {}
        self.uwb_update_times: dict[tuple[str, str], float] = {}
        self.neighbor_beliefs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.neighbor_belief_timestamps: dict[tuple[str, str], float] = {}
        self.neighbor_belief_message_times: dict[tuple[str, str], float] = {}
        self.ci_weights: dict[tuple[str, str], float] = {}
        self.stale_neighbor_messages = 0
        self.reused_uwb_measurement_skips = 0
        self.uwb_update_period_skips = 0
        self.cooperative_ekf_update_opportunities = 0
        self.cooperative_ekf_updates_blocked = 0
        self.relative_range_update_events: list[dict[str, Any]] = []
        # Per-window diagnostic records for the cooperative-measurement gate
        # funnel.  These records are exposed through ``info`` only and never
        # enter actor observations, critic state, rewards, or state updates.
        self.measurement_gate_events: list[dict[str, Any]] = []
        self.measurement_overhead_events: list[dict[str, Any]] = []
        self.measurement_budget_events: list[dict[str, Any]] = []
        self.measurement_budget_step_events: list[dict[str, Any]] = []
        self.measurement_action_diagnostics: dict[str, dict[str, Any]] = {}
        self.invalid_measurement_action_events: list[dict[str, Any]] = []
        self.measurement_transition_invalidated_events: list[dict[str, Any]] = []
        self.measurement_budget_limits: dict[str, int] = {}
        self.measurement_budget_remaining: dict[str, int] = {}
        self.measurement_budget_used: dict[str, int] = {}
        self.pending_messages: list[MessageRecord] = []
        self.arrived_messages: dict[tuple[str, str], MessageRecord] = {}
        self.message_events: list[dict[str, Any]] = []
        self.snapshot = TopologySnapshot()
        self.graph_batch: GraphBatch | None = None
        self.timestep = 0
        self.last_transition: dict[str, Any] = {}
        self.scenario_metadata: dict[str, Any] = {}
        self._learning_potential = 0.0
        self._previous_position_logdet = 0.0
        self._previous_estimated_goal_distance = 0.0
        self._pending_update_feedback: list[dict[str, Any]] = []
        self._last_update_feedback_records: list[dict[str, Any]] = []

    def _position_logdet_uncertainty(self) -> float:
        """Return the mean local position-covariance log determinant."""

        epsilon = float(self.config.reward.potential_epsilon)
        values: list[float] = []
        for ekf in self.ekfs.values():
            covariance = np.asarray(ekf.covariance[:3, :3], dtype=float)
            covariance = 0.5 * (covariance + covariance.T) + epsilon * np.eye(3)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0.0 or not np.isfinite(logdet):
                diagonal = np.maximum(np.diag(covariance), epsilon)
                logdet = float(np.log(diagonal).sum())
            if np.isfinite(logdet):
                values.append(float(logdet))
        return float(np.mean(values)) if values else 0.0

    def estimated_goal_distance(self) -> float:
        """Return mean EKF-estimated distance to each configured goal.

        This helper deliberately uses the EKF state and the configured goal,
        never the simulator truth state.  The formal warm-up fitter and the
        single-UAV learning reward therefore share the same observable
        quantity.
        """

        values: list[float] = []
        for uav_id, ekf in self.ekfs.items():
            goal = np.asarray(self.goals.get(uav_id, np.zeros(3)), dtype=float)
            estimate = np.asarray(ekf.state[:3], dtype=float)
            distance = float(np.linalg.norm(estimate - goal))
            if np.isfinite(distance):
                values.append(distance)
        return float(np.mean(values)) if values else 0.0

    def _learning_progress_potential(self, j_com: float) -> tuple[float, float, float]:
        """Return ``(phi, logdet_uncertainty, normalized_j_com)`` for learning.

        This path is deliberately separated from :meth:`_rewards`' reporting
        objective.  It uses only the current local EKF covariance and current
        communication objective, with robust statistics frozen from warm-up.
        """

        epsilon = float(self.config.reward.potential_epsilon)
        uncertainty_logdet = self._position_logdet_uncertainty()
        statistics = self.config.reward.potential_statistics or {}
        u_center = float(statistics.get("u_center", 0.0))
        u_scale = max(float(statistics.get("u_scale", 1.0)), epsilon)
        c_center = float(statistics.get("c_center", 0.0))
        c_scale = max(float(statistics.get("c_scale", 1.0)), epsilon)
        normalized_u = float(np.clip((uncertainty_logdet - u_center) / u_scale, -3.0, 3.0))
        normalized_c = float(np.clip((float(j_com) - c_center) / c_scale, -3.0, 3.0))
        phi = float(-np.tanh(0.35 * normalized_u + 0.65 * normalized_c))
        return phi, uncertainty_logdet, normalized_c

    def reset(
        self,
        *,
        seed: int | None = None,
        scenario_config: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        self._episode_random_seed = int(self.config.seed if seed is None else seed)
        self._topology_draw_index = -1
        if seed is not None:
            self.config = copy.copy(self.config)
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng(self.config.seed)
        scenario = dict(scenario_config or {})
        # Keep only JSON-compatible scenario metadata.  The actor receives
        # local estimated features, never this full truth/configuration map.
        self.scenario_metadata = copy.deepcopy(scenario)
        self.timestep = 0
        self._learning_potential = 0.0
        lower = np.asarray(self.config.motion.bounds_min_m, dtype=float)
        upper = np.asarray(self.config.motion.bounds_max_m, dtype=float)
        center = (lower + upper) / 2.0
        radius = float(scenario.get("initial_radius_m", min(18.0, (upper[0] - lower[0]) / 5.0)))
        self.states = {}
        self.goals = {}
        self.ekfs = {}
        self.accel_bias = {}
        self.gyro_bias = {}
        configured_positions = scenario.get("initial_positions", {})
        configured_velocities = scenario.get("initial_velocities", {})
        configured_goals = scenario.get("goals", {})
        for index, uav_id in enumerate(self.uav_ids):
            angle = 2.0 * np.pi * index / self.config.uav_count
            position = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 8.0 + index], dtype=float)
            velocity = np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=float)
            if isinstance(configured_positions, Mapping) and uav_id in configured_positions:
                position = np.asarray(configured_positions[uav_id], dtype=float).reshape(3)
            if isinstance(configured_velocities, Mapping) and uav_id in configured_velocities:
                velocity = np.asarray(configured_velocities[uav_id], dtype=float).reshape(3)
            attitude = np.array([0.0, 0.0, angle + np.pi / 2.0], dtype=float)
            state = UAVKinematicState(
                position_m=np.clip(position, lower, upper),
                velocity_mps=velocity,
                attitude_rad=attitude,
                acceleration_mps2=np.zeros(3),
                angular_velocity_rps=np.zeros(3),
            )
            self.states[uav_id] = state
            goal = center + np.array([-radius * np.cos(angle), -radius * np.sin(angle), 10.0], dtype=float)
            if isinstance(configured_goals, Mapping) and uav_id in configured_goals:
                goal = np.asarray(configured_goals[uav_id], dtype=float).reshape(3)
            self.goals[uav_id] = np.clip(goal, lower, upper)
            self.accel_bias[uav_id] = np.zeros(3)
            self.gyro_bias[uav_id] = np.zeros(3)
            self.ekfs[uav_id] = LocalEKF(_state_vector(state, self.accel_bias[uav_id], self.gyro_bias[uav_id]), self.config.ekf)

        self.obstacles = self._make_obstacles(scenario)
        self.queue_state = {}
        self.aoi_state = {}
        self.uwb_update_times = {}
        self.neighbor_beliefs = {}
        self.neighbor_belief_timestamps = {}
        self.neighbor_belief_message_times = {}
        self.ci_weights = {}
        self.stale_neighbor_messages = 0
        self.reused_uwb_measurement_skips = 0
        self.uwb_update_period_skips = 0
        self.cooperative_ekf_update_opportunities = 0
        self.cooperative_ekf_updates_blocked = 0
        self.relative_range_update_events = []
        self.measurement_gate_events = []
        self.measurement_overhead_events = []
        self.measurement_budget_events = []
        self.measurement_budget_step_events = []
        self.measurement_action_diagnostics = {}
        self.invalid_measurement_action_events = []
        self.measurement_transition_invalidated_events = []
        if self.config.measurement_budget_enabled:
            configured_budgets = self.config.measurement_budget_by_uav or {}
            self.measurement_budget_limits = {
                uav_id: int(configured_budgets[uav_id]) for uav_id in self.uav_ids
            }
            self.measurement_budget_remaining = dict(self.measurement_budget_limits)
            self.measurement_budget_used = {uav_id: 0 for uav_id in self.uav_ids}
        else:
            self.measurement_budget_limits = {}
            self.measurement_budget_remaining = {}
            self.measurement_budget_used = {}
        self.pending_messages = []
        self.arrived_messages = {}
        self.message_events = []
        self._pending_update_feedback = []
        self._last_update_feedback_records = []
        # A neutral equal-share probe exposes feasible candidate edges in the
        # initial observation.  It is only the previous-window state; the
        # first user action still controls the next decision window.
        initial_bandwidth = {
            uav_id: {
                target: 1.0 / max(1, self.config.uav_count - 1)
                for target in self.uav_ids
                if target != uav_id
            }
            for uav_id in self.uav_ids
        }
        self.snapshot = self._refresh_topology(initial_bandwidth)
        # A bandwidth allocation creates a potential/candidate channel at
        # reset, but no neighbor message has arrived yet. The effective graph
        # used by EKF/GAT is therefore initially empty.
        self.snapshot.effective_edges = set()
        self.queue_state = self.snapshot.queue_next.copy()
        self.graph_batch = self._build_graph_batch()
        self._previous_position_logdet = self._position_logdet_uncertainty()
        self._previous_estimated_goal_distance = self.estimated_goal_distance()
        if self.config.reward.learning_reward_mode == "progress_potential_v2":
            self._learning_potential = self._learning_progress_potential(0.0)[0]
        obs = self._observations()
        info = self._info(collision=0, minimum_clearance=float("inf"), sensor_packets={})
        return obs, info

    def step(
        self,
        action: Mapping[str, Any] | Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float], bool, bool, dict[str, Any]]:
        # These diagnostics describe only the current decision window.  They
        # are not actor inputs and do not affect the random process.
        self.invalid_measurement_action_events = []
        self.measurement_transition_invalidated_events = []
        actions = self._normalize_actions(action)
        self.relative_range_update_events = []
        self.measurement_gate_events = []
        for ekf in self.ekfs.values():
            ekf.begin_step()
        previous_truth_positions = {
            uav_id: state.position_m.copy() for uav_id, state in self.states.items()
        }
        previous_estimate_states = {
            uav_id: ekf.state.copy() for uav_id, ekf in self.ekfs.items()
        }
        previous_estimate_covariances = {
            uav_id: ekf.covariance.copy() for uav_id, ekf in self.ekfs.items()
        }
        sender_payloads = {
            uav_id: (ekf.state.copy(), ekf.covariance.copy())
            for uav_id, ekf in self.ekfs.items()
        }
        action_info: dict[str, Any] = {}
        # Compute the position that the same command would produce before the
        # scene safety projector is applied.  This is a deterministic
        # diagnostic calculation: it does not mutate state or consume RNG.
        # It is compared with the post-projector state after dynamic
        # obstacles have advanced, which is the actual execution order.
        unprojected_positions: dict[str, np.ndarray] = {}
        for uav_id in self.uav_ids:
            command = actions[uav_id]
            nominal_state, _ = apply_velocity_increment(
                self.states[uav_id],
                command["delta_v"],
                self.config.motion,
                safety_projector=None,
            )
            unprojected_positions[uav_id] = nominal_state.position_m.copy()
        joint_mode = self.config.safety_execution_mode == "joint_swept_v1"
        joint_pre_projection: dict[str, object] = {
            "collision": False,
            "physical_contact": False,
            "clearance_violation": False,
            "minimum_clearance_m": float("inf"),
            "uav_pair_conflicts": [],
            "obstacle_conflicts": [],
            "ignored_nonphysical_obstacle_ids": [],
        }
        joint_projection_reasons: dict[str, set[str]] = {
            uav_id: set() for uav_id in self.uav_ids
        }
        obstacles_before = copy.deepcopy(self.obstacles) if joint_mode else []
        if joint_mode:
            obstacles_predicted = copy.deepcopy(self.obstacles)
            lower = np.asarray(self.config.motion.bounds_min_m, dtype=float)
            upper = np.asarray(self.config.motion.bounds_max_m, dtype=float)
            for obstacle in obstacles_predicted:
                obstacle.update(self.config.motion.dt_s, lower, upper)
            joint_pre_projection = joint_motion_collision_diagnostics(
                previous_truth_positions,
                unprojected_positions,
                obstacles_before,
                obstacles_predicted,
                dt_s=self.config.motion.dt_s,
                bounds_min=lower,
                bounds_max=upper,
                safety_distance_m=self.config.motion.safety_distance_m,
            )
            for conflict in joint_pre_projection["uav_pair_conflicts"]:
                reason = str(conflict.get("conflict_type", "uav_uav_joint"))
                for uav_id in (
                    str(conflict["source_uav_id"]),
                    str(conflict["target_uav_id"]),
                ):
                    joint_projection_reasons[uav_id].add("uav_uav_" + reason)
            for conflict in joint_pre_projection["obstacle_conflicts"]:
                uav_id = str(conflict["uav_id"])
                joint_projection_reasons[uav_id].add(
                    "obstacle_" + str(conflict.get("conflict_type", "swept_path"))
                )

        next_states: dict[str, UAVKinematicState] = {}
        for uav_id in self.uav_ids:
            command = actions[uav_id]
            old_state = self.states[uav_id]
            if joint_mode:
                # A conflicting candidate is symmetrically held at its own
                # previous position.  No UAV is tested against a partially
                # updated state, and all non-conflicting candidates retain
                # their requested motion.
                projected = (
                    (lambda _proposed, uid=uav_id: previous_truth_positions[uid].copy())
                    if self.config.safety_projection_enabled
                    and joint_projection_reasons[uav_id]
                    else None
                )
            else:
                projected = (
                    (lambda proposed, uid=uav_id: self._safety_project(uid, proposed))
                    if self.config.safety_projection_enabled
                    else None
                )
            next_state, execution = apply_velocity_increment(
                old_state,
                command["delta_v"],
                self.config.motion,
                safety_projector=projected,
            )
            if joint_mode and joint_projection_reasons[uav_id]:
                execution["safety_projection_reason"] = ";".join(
                    sorted(joint_projection_reasons[uav_id])
                )
            next_states[uav_id] = next_state
            action_info[uav_id] = execution
        self.states = next_states
        self._update_obstacles()
        positions = {uav_id: state.position_m for uav_id, state in self.states.items()}
        collision, minimum_clearance = collision_event(
            positions,
            self.obstacles,
            self.config.motion.safety_distance_m,
        )
        if joint_mode:
            joint_post_projection = joint_motion_collision_diagnostics(
                previous_truth_positions,
                positions,
                obstacles_before,
                self.obstacles,
                dt_s=self.config.motion.dt_s,
                bounds_min=np.asarray(self.config.motion.bounds_min_m, dtype=float),
                bounds_max=np.asarray(self.config.motion.bounds_max_m, dtype=float),
                safety_distance_m=self.config.motion.safety_distance_m,
            )
            collision = int(bool(collision or joint_post_projection["collision"]))
            minimum_clearance = float(
                min(
                    float(minimum_clearance),
                    float(joint_post_projection["minimum_clearance_m"]),
                )
            )
            pre_projection_collision = int(bool(joint_pre_projection["collision"]))
            pre_projection_minimum_clearance = float(
                joint_pre_projection["minimum_clearance_m"]
            )
        else:
            joint_post_projection = {
                "collision": False,
                "physical_contact": False,
                "clearance_violation": False,
                "minimum_clearance_m": float("inf"),
                "uav_pair_conflicts": [],
                "obstacle_conflicts": [],
                "ignored_nonphysical_obstacle_ids": [],
            }
            pre_projection_collision, pre_projection_minimum_clearance = collision_event(
                unprojected_positions,
                self.obstacles,
                self.config.motion.safety_distance_m,
            )
            if collision:
                legacy_pair_conflicts = []
                legacy_obstacle_conflicts = []
                ids = sorted(positions)
                for index, source in enumerate(ids):
                    for target in ids[index + 1 :]:
                        distance = float(
                            np.linalg.norm(positions[source] - positions[target])
                        )
                        if distance < self.config.motion.safety_distance_m:
                            legacy_pair_conflicts.append(
                                {
                                    "source_uav_id": source,
                                    "target_uav_id": target,
                                    "endpoint_distance_m": distance,
                                    "swept_distance_m": distance,
                                    "physical_contact": bool(distance <= 1e-9),
                                    "clearance_violation": bool(distance > 1e-9),
                                    "conflict_type": "endpoint_clearance",
                                }
                            )
                for uav_id, position in positions.items():
                    for obstacle in self.obstacles:
                        if not is_physical_obstacle(obstacle):
                            continue
                        clearance = point_obstacle_distance(position, obstacle)
                        if clearance < self.config.motion.safety_distance_m:
                            legacy_obstacle_conflicts.append(
                                {
                                    "uav_id": uav_id,
                                    "obstacle_id": str(obstacle.obstacle_id),
                                    "endpoint_clearance_m": float(clearance),
                                    "physical_contact": bool(clearance <= 1e-9),
                                    "clearance_violation": bool(clearance > 1e-9),
                                    "conflict_type": "endpoint_clearance",
                                }
                            )
                joint_post_projection = {
                    "collision": True,
                    "physical_contact": any(
                        bool(conflict.get("physical_contact", False))
                        for conflict in legacy_pair_conflicts + legacy_obstacle_conflicts
                    ),
                    "clearance_violation": any(
                        bool(conflict.get("clearance_violation", False))
                        for conflict in legacy_pair_conflicts + legacy_obstacle_conflicts
                    ),
                    "minimum_clearance_m": float(minimum_clearance),
                    "uav_pair_conflicts": legacy_pair_conflicts,
                    "obstacle_conflicts": legacy_obstacle_conflicts,
                    "ignored_nonphysical_obstacle_ids": sorted(
                        str(obstacle.obstacle_id)
                        for obstacle in self.obstacles
                        if not is_physical_obstacle(obstacle)
                    ),
                }

        sensor_packets: dict[str, dict[str, Any]] = {}
        for uav_id, state in self.states.items():
            sensor_packets[uav_id] = {
                "imu": sample_imu(
                    state,
                    self.accel_bias[uav_id],
                    self.gyro_bias[uav_id],
                    self._sensor_rng("imu", uav_id),
                    self.config.sensor,
                ),
                "vo": sample_visual_odometry(
                    state,
                    self.config.motion.dt_s,
                    self._sensor_rng("vo", uav_id),
                    self.config.sensor,
                    reference_position=previous_truth_positions[uav_id],
                ),
                "gnss": sample_gnss(state, self.config.gnss_mode, self._sensor_rng("gnss", uav_id), self.config.sensor),
            }
        # The action is selected from the pre-transition observation, while
        # the communication queue is evaluated after the commanded motion.
        # Recheck the physical potential edge here so a link that left the
        # communication range cannot create a measurement packet or an EKF
        # update.  This is a deterministic geometry check and consumes no RNG.
        post_motion_potential_edges = self._potential_edges_for_current_states()
        actions = self._validate_post_motion_measurements(
            actions,
            post_motion_potential_edges,
        )
        self.measurement_budget_step_events = self._consume_measurement_budget(actions)
        self.measurement_budget_events.extend(copy.deepcopy(self.measurement_budget_step_events))
        bandwidth = {
            uav_id: dict(actions[uav_id].get("bandwidth", {}))
            for uav_id in self.uav_ids
        }
        measurement_arrivals = self._measurement_overhead_arrivals(
            actions,
            allowed_potential_edges=post_motion_potential_edges,
        )
        queue_before_overhead = dict(self.queue_state)
        self.snapshot = self._refresh_topology(
            bandwidth,
            measurement_arrivals=measurement_arrivals,
        )
        self.measurement_overhead_events = self._measurement_overhead_rows(
            actions,
            measurement_arrivals,
            queue_before_overhead,
        )
        transmitted_edges = set(self.snapshot.effective_edges)
        self.queue_state = self.snapshot.queue_next.copy()
        applied_bandwidth = {
            uav_id: {
                target: float(
                    self.snapshot.metrics[(uav_id, target)].channel_bandwidth_hz
                    / self.config.channel.total_bandwidth_hz
                )
                for source, target in sorted(self.snapshot.potential_edges)
                if source == uav_id
            }
            for uav_id in self.uav_ids
        }

        estimates: dict[str, np.ndarray] = {}
        covariances: dict[str, np.ndarray] = {}
        transition_matrices: dict[str, np.ndarray] = {}
        for uav_id, ekf in self.ekfs.items():
            transition_matrices[uav_id] = ekf.predict(
                sensor_packets[uav_id]["imu"], self.config.motion.dt_s
            )
            vo = sensor_packets[uav_id]["vo"]
            vo_covariance = np.eye(3, dtype=float) * float(self.config.sensor.vo_position_std_m) ** 2
            vo_quality_scale = 1.0
            vo_quality_valid = False
            vo_quality_bin = -1
            if self.config.visual_odometry_quality_calibration_enabled:
                vo_covariance, vo_quality_scale, vo_quality_valid, vo_quality_bin = covariance_for_quality(
                    float(self.config.sensor.vo_position_std_m),
                    vo.get("quality"),
                    self.config.visual_odometry_quality_model or {},
                )
            # These are diagnostic annotations on the already generated
            # sensor packet. They do not expose truth and do not affect action
            # selection; the EKF receives only the covariance below.
            vo["quality_covariance_scale"] = float(vo_quality_scale)
            vo["quality_covariance_valid"] = int(vo_quality_valid)
            vo["quality_covariance_bin"] = int(vo_quality_bin)
            vo["quality_covariance_trace"] = float(np.trace(vo_covariance))
            if vo["valid"]:
                ekf.update_visual_delta(
                    vo["delta_position"],
                    covariance=vo_covariance,
                    previous_state=previous_estimate_states[uav_id],
                    previous_covariance=previous_estimate_covariances[uav_id],
                    transition_matrix=transition_matrices[uav_id],
                    dt_s=self.config.motion.dt_s,
                    previous_time_s=self.timestep * self.config.motion.dt_s,
                    current_time_s=(self.timestep + 1) * self.config.motion.dt_s,
                )
            gnss = sensor_packets[uav_id]["gnss"]
            if gnss["valid"]:
                ekf.update_position(gnss["position"], np.eye(3) * float(gnss["std"]) ** 2)

        next_timestep = self.timestep + 1
        self._enqueue_messages(transmitted_edges, sender_payloads)
        self._release_messages(next_timestep)
        self.snapshot.effective_edges = self._visible_effective_edges()

        step_cooperative_ekf_update_opportunities = 0
        step_cooperative_ekf_updates_blocked = 0
        gate_events: dict[tuple[str, str], dict[str, Any]] = {}
        for target in self.uav_ids:
            source = str(actions[target]["measurement_uav_id"])
            if source == "none":
                continue
            gate_events[(source, target)] = {
                "source_uav_id": source,
                "target_uav_id": target,
                "selected_non_none": True,
                "uwb_valid": False,
                "effective_edge": False,
                "message_arrived": False,
                "update_period_passed": False,
                "new_packet": False,
                "ekf_update_attempted": False,
                "ekf_update_accepted": False,
                "failure_stage": "selected_pair_not_observed",
            }
        for (source, target), measurement in self.snapshot.uwb_measurements.items():
            selected_measurement = actions[target]["measurement_uav_id"]
            if selected_measurement != source:
                continue
            gate = gate_events.setdefault(
                (source, target),
                {
                    "source_uav_id": source,
                    "target_uav_id": target,
                    "selected_non_none": True,
                    "uwb_valid": False,
                    "effective_edge": False,
                    "message_arrived": False,
                    "update_period_passed": False,
                    "new_packet": False,
                    "ekf_update_attempted": False,
                    "ekf_update_accepted": False,
                    "failure_stage": "selected_pair_not_observed",
                },
            )
            gate["uwb_valid"] = bool(measurement["valid"])
            if not measurement["valid"]:
                gate["failure_stage"] = "uwb_invalid"
                continue
            gate["effective_edge"] = bool((source, target) in self.snapshot.effective_edges)
            if (source, target) not in self.snapshot.effective_edges:
                gate["failure_stage"] = "communication_edge_ineffective"
                continue
            nlos_bias = self.config.sensor.uwb_nlos_bias_m if not measurement["los"] else 0.0
            message = self.arrived_messages.get((source, target))
            if message is None:
                gate["failure_stage"] = "message_not_arrived"
                continue
            gate["message_arrived"] = True
            current_measurement_time = next_timestep * self.config.motion.dt_s
            last_update_time = self.uwb_update_times.get((source, target))
            if (
                last_update_time is not None
                and current_measurement_time - float(last_update_time)
                < self.config.sensor.uwb_update_period_s - 1e-9
            ):
                self.uwb_update_period_skips += 1
                gate["failure_stage"] = "update_period_not_elapsed"
                continue
            gate["update_period_passed"] = True
            source_position = message.state[:3].copy()
            source_covariance = message.covariance[:3, :3].copy()
            belief_key = (target, source)
            previous_belief = self.neighbor_beliefs.get(belief_key)
            message_time = float(
                message.last_propagated_time_s
                if message.last_propagated_time_s is not None
                else message.arrival_time_s
                if message.arrival_time_s is not None
                else next_timestep * self.config.motion.dt_s
            )
            previous_time = self.neighbor_belief_timestamps.get(belief_key)
            previous_message_time = self.neighbor_belief_message_times.get(belief_key)
            is_new_packet = (
                previous_message_time is None
                or not np.isclose(
                    float(previous_message_time),
                    float(message.send_time_s),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            # A physical UWB range sample is only paired with a neighbor
            # state once.  Reusing the same delayed neighbor packet with a
            # sequence of new ranges would count the packet uncertainty as if
            # it were independent at every step, which is not valid for a
            # local EKF without an explicit cross-covariance state.  The
            # range remains available to topology/GAT logging; only the EKF
            # update waits for a newly arrived neighbor packet.
            if not is_new_packet:
                self.reused_uwb_measurement_skips += 1
                gate["failure_stage"] = "reused_neighbor_packet"
                continue
            gate["new_packet"] = True
            # Advance the physical EKF update schedule even when this sample
            # is rejected by the innovation gate.  Otherwise a persistent
            # outlier would be retried at every decision step rather than at
            # the declared sensor update frequency.
            self.uwb_update_times[(source, target)] = current_measurement_time
            previous_is_fresh = (
                previous_belief is not None
                and previous_time is not None
                and message_time - previous_time <= self.config.ekf.stale_message_limit_s
            )
            # CI is used once when a genuinely new packet is compared with a
            # previous packet from the same neighbor.  Repeating CI on the
            # same packet would manufacture confidence without new data.
            if previous_is_fresh and is_new_packet:
                fused_position, fused_covariance, ci_weight = covariance_intersection(
                    previous_belief[0],
                    previous_belief[1],
                    source_position,
                    source_covariance,
                )
                self.ci_weights[belief_key] = ci_weight
            else:
                fused_position, fused_covariance = source_position, source_covariance
                self.ci_weights[belief_key] = 1.0
            self.neighbor_beliefs[belief_key] = (
                np.asarray(fused_position, dtype=float).copy(),
                np.asarray(fused_covariance, dtype=float).copy(),
            )
            self.neighbor_belief_timestamps[belief_key] = float(message_time)
            self.neighbor_belief_message_times[belief_key] = float(message.send_time_s)
            self.cooperative_ekf_update_opportunities += 1
            step_cooperative_ekf_update_opportunities += 1
            if not self.config.cooperative_ekf_enabled:
                self.cooperative_ekf_updates_blocked += 1
                step_cooperative_ekf_updates_blocked += 1
                gate["failure_stage"] = "cooperative_ekf_disabled"
                continue
            target_ekf = self.ekfs[target]
            prior_position = target_ekf.position.copy()
            prior_error = float(
                np.linalg.norm(self.states[target].position_m - prior_position)
            )
            j_loc_before = self._diagnostic_j_loc()
            accepted = target_ekf.update_relative_range(
                measurement["measured_range"],
                fused_position,
                fused_covariance,
                measurement_variance=(
                    float(measurement["measurement_std"]) ** 2
                    * (self.config.sensor.uwb_nlos_variance_multiplier if not measurement["los"] else 1.0)
                ),
                nlos_bias_m=nlos_bias,
                timestamp=float(message_time),
            )
            gate["ekf_update_attempted"] = True
            gate["ekf_update_accepted"] = bool(accepted)
            gate["failure_stage"] = "accepted" if accepted else "innovation_rejected"
            posterior_position = target_ekf.position.copy()
            event = dict(target_ekf.last_relative_range_update)
            event.update(
                {
                    "source_uav_id": source,
                    "target_uav_id": target,
                    "los": bool(measurement["los"]),
                    "message_age_s": float(max(0.0, current_measurement_time - message.send_time_s)),
                    "message_delay_s": float(message.delay_s),
                    "measurement_time_s": float(current_measurement_time),
                    "message_send_time_s": float(message.send_time_s),
                    "message_state_time_s": float(message_time),
                    "ci_weight": float(self.ci_weights.get(belief_key, 1.0)),
                    "accepted": bool(accepted),
                    "true_position_error_before_m": prior_error,
                    "true_position_error_after_m": float(
                        np.linalg.norm(self.states[target].position_m - posterior_position)
                    ),
                    "j_loc_before": float(j_loc_before),
                    "j_loc_after": float(self._diagnostic_j_loc()),
                }
            )
            self.relative_range_update_events.append(event)
        self.measurement_gate_events = [
            dict(gate_events[key]) for key in sorted(gate_events)
        ]
        # Boundary and obstacle projections are known simulator events, but
        # the EKF must not receive the true projected position.  Represent the
        # unmodelled non-smooth transition by covariance inflation only.
        for uav_id, ekf in self.ekfs.items():
            execution = action_info[uav_id]
            if execution.get("boundary_projected") or execution.get("safety_projected"):
                ekf.inflate_motion_uncertainty()
        lower_bounds = np.asarray(self.config.motion.bounds_min_m, dtype=float)
        upper_bounds = np.asarray(self.config.motion.bounds_max_m, dtype=float)
        estimate_projection_count = 0
        for ekf in self.ekfs.values():
            estimate_projection_count += int(
                ekf.project_position_bounds(lower_bounds, upper_bounds)
            )
        for uav_id, ekf in self.ekfs.items():
            estimates[uav_id] = ekf.state.copy()
            covariances[uav_id] = ekf.covariance.copy()

        self.graph_batch = self._build_graph_batch(estimates, covariances)
        rewards, reward_info = self._rewards(collision, actions, action_info)
        self.timestep += 1
        collision_terminated = bool(collision)
        time_limit = self.timestep >= self.config.max_steps
        terminated = collision_terminated
        truncated = bool(time_limit and not collision_terminated)
        termination_reason = (
            "collision"
            if collision_terminated
            else "time_limit"
            if truncated
            else "running"
        )
        obs = self._observations()
        info = self._info(
            collision=collision,
            minimum_clearance=minimum_clearance,
            sensor_packets=sensor_packets,
        )
        info.update(reward_info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["termination_reason"] = termination_reason
        info["task_complete"] = False
        info["safety_projection_enabled"] = bool(self.config.safety_projection_enabled)
        info["safety_execution_mode"] = self.config.safety_execution_mode
        info["pre_projection_collision"] = int(pre_projection_collision)
        info["pre_projection_minimum_clearance"] = float(pre_projection_minimum_clearance)
        pre_pair_conflicts = list(joint_pre_projection.get("uav_pair_conflicts", []))
        pre_obstacle_conflicts = list(joint_pre_projection.get("obstacle_conflicts", []))
        post_pair_conflicts = list(joint_post_projection.get("uav_pair_conflicts", []))
        post_obstacle_conflicts = list(joint_post_projection.get("obstacle_conflicts", []))
        obstacle_by_id = {
            str(obstacle.obstacle_id): obstacle for obstacle in self.obstacles
        }
        collision_types = sorted(
            {
                str(conflict.get("conflict_type", "uav_uav"))
                for conflict in post_pair_conflicts
            }
            | {
                str(conflict.get("conflict_type", "obstacle"))
                for conflict in post_obstacle_conflicts
            }
        )
        collision_objects = [
            {
                "type": "uav_uav",
                "source_uav_id": str(conflict.get("source_uav_id", "")),
                "target_uav_id": str(conflict.get("target_uav_id", "")),
                "conflict_type": str(conflict.get("conflict_type", "unknown")),
                "physical_contact": bool(conflict.get("physical_contact", False)),
                "clearance_violation": bool(conflict.get("clearance_violation", False)),
            }
            for conflict in post_pair_conflicts
        ] + [
            {
                "type": "uav_obstacle",
                "uav_id": str(conflict.get("uav_id", "")),
                "obstacle_id": str(conflict.get("obstacle_id", "")),
                "conflict_type": str(conflict.get("conflict_type", "unknown")),
                "physical_contact": bool(conflict.get("physical_contact", False)),
                "clearance_violation": bool(conflict.get("clearance_violation", False)),
                "physical_collision_enabled": bool(
                    is_physical_obstacle(
                        obstacle_by_id.get(str(conflict.get("obstacle_id", "")))
                    )
                    if obstacle_by_id.get(str(conflict.get("obstacle_id", ""))) is not None
                    else True
                ),
            }
            for conflict in post_obstacle_conflicts
        ]
        info["collision_types"] = collision_types
        info["collision_objects"] = collision_objects
        info["pre_projection_uav_pair_conflicts"] = copy.deepcopy(pre_pair_conflicts)
        info["pre_projection_obstacle_conflicts"] = copy.deepcopy(pre_obstacle_conflicts)
        info["post_projection_uav_pair_conflicts"] = copy.deepcopy(post_pair_conflicts)
        info["post_projection_obstacle_conflicts"] = copy.deepcopy(post_obstacle_conflicts)
        info["pre_projection_ignored_nonphysical_obstacle_ids"] = sorted(
            str(value)
            for value in joint_pre_projection.get(
                "ignored_nonphysical_obstacle_ids", []
            )
        )
        info["post_projection_ignored_nonphysical_obstacle_ids"] = sorted(
            str(value)
            for value in joint_post_projection.get(
                "ignored_nonphysical_obstacle_ids", []
            )
        )
        physical_contact = bool(
            joint_post_projection.get("physical_contact", False)
            or (collision and minimum_clearance <= 1e-9 and not post_pair_conflicts and not post_obstacle_conflicts)
        )
        info["physical_collision"] = bool(collision and physical_contact)
        info["protection_distance_violation"] = bool(
            collision and not physical_contact
        )
        info["physical_contact"] = bool(physical_contact)
        info["clearance_violation"] = bool(
            collision and not physical_contact
        )
        info["collision_classes"] = sorted(
            {
                "physical_contact"
                if bool(conflict.get("physical_contact", False))
                else "clearance_violation"
                for conflict in post_pair_conflicts + post_obstacle_conflicts
            }
        )
        info["ignored_nonphysical_obstacle_ids"] = sorted(
            set(
                str(value)
                for value in list(joint_pre_projection.get("ignored_nonphysical_obstacle_ids", []))
                + list(joint_post_projection.get("ignored_nonphysical_obstacle_ids", []))
            )
        )
        info["physical_obstacle_ids"] = sorted(
            str(obstacle.obstacle_id)
            for obstacle in self.obstacles
            if is_physical_obstacle(obstacle)
        )
        info["communication_occluder_ids"] = sorted(
            str(obstacle.obstacle_id)
            for obstacle in self.obstacles
            if obstacle.active and not is_physical_obstacle(obstacle)
        )
        info["unavoidable_dynamic_collision"] = bool(
            collision
            and any(
                str(conflict.get("conflict_type", ""))
                in {"moving_swept_path", "endpoint_clearance"}
                for conflict in post_obstacle_conflicts
            )
            and not any(
                bool(execution.get("safety_projected", False))
                for execution in action_info.values()
            )
        )
        info["team_reward"] = float(np.mean(list(rewards.values()))) if rewards else 0.0
        info["stale_neighbor_messages"] = int(self.stale_neighbor_messages)
        info["reused_uwb_measurement_skips"] = int(self.reused_uwb_measurement_skips)
        info["uwb_update_period_skips"] = int(self.uwb_update_period_skips)
        info["cooperative_ekf_enabled"] = bool(self.config.cooperative_ekf_enabled)
        info["cooperative_ekf_update_opportunities"] = int(
            self.cooperative_ekf_update_opportunities
        )
        info["cooperative_ekf_updates_blocked"] = int(
            self.cooperative_ekf_updates_blocked
        )
        info["step_cooperative_ekf_update_opportunities"] = int(
            step_cooperative_ekf_update_opportunities
        )
        info["step_cooperative_ekf_updates_blocked"] = int(
            step_cooperative_ekf_updates_blocked
        )
        info["measurement_gate_events"] = copy.deepcopy(self.measurement_gate_events)
        info["measurement_overhead_enabled"] = bool(self.config.measurement_overhead_enabled)
        info["measurement_overhead_packet_size_bytes"] = int(MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES)
        info["measurement_overhead_events"] = copy.deepcopy(self.measurement_overhead_events)
        info["measurement_budget_enabled"] = bool(self.config.measurement_budget_enabled)
        info["measurement_budget_limits"] = copy.deepcopy(self.measurement_budget_limits)
        info["measurement_budget_remaining"] = copy.deepcopy(self.measurement_budget_remaining)
        info["measurement_budget_used"] = copy.deepcopy(self.measurement_budget_used)
        info["measurement_budget_events"] = copy.deepcopy(self.measurement_budget_step_events)
        info["measurement_update_eligibility_enabled"] = bool(
            self.config.measurement_update_eligibility_enabled
        )
        info["measurement_update_period_s"] = float(
            self.config.sensor.uwb_update_period_s
        )
        info["measurement_update_eligibility"] = {
            uav_id: self._measurement_update_eligibility(uav_id).astype(bool).tolist()
            for uav_id in self.uav_ids
        }
        info["visual_odometry_quality_calibration_enabled"] = bool(
            self.config.visual_odometry_quality_calibration_enabled
        )
        info["visual_odometry_quality_diagnostics"] = {
            uav_id: {
                "tracking_quality": float(packet.get("vo", {}).get("quality", 0.0)),
                "valid": int(packet.get("vo", {}).get("valid", 0)),
                "covariance_scale": float(
                    packet.get("vo", {}).get("quality_covariance_scale", 1.0)
                ),
                "covariance_bin": int(packet.get("vo", {}).get("quality_covariance_bin", -1)),
                "covariance_trace": float(
                    packet.get("vo", {}).get("quality_covariance_trace", 0.0)
                ),
            }
            for uav_id, packet in sensor_packets.items()
        }
        info["invalid_measurement_action_events"] = copy.deepcopy(
            self.invalid_measurement_action_events
        )
        info["measurement_transition_invalidated_events"] = copy.deepcopy(
            self.measurement_transition_invalidated_events
        )
        info["invalid_measurement_action_count"] = int(
            len(self.invalid_measurement_action_events)
        )
        info["measurement_transition_invalidated_count"] = int(
            len(self.measurement_transition_invalidated_events)
        )
        info["estimate_position_projection_count"] = int(estimate_projection_count)
        info["actions"] = actions
        info["applied_bandwidth"] = applied_bandwidth
        info["action_execution"] = action_info
        info["safety_projection_count"] = int(
            sum(bool(execution.get("safety_projected", False)) for execution in action_info.values())
        )
        info["boundary_projection_count"] = int(
            sum(bool(execution.get("boundary_projected", False)) for execution in action_info.values())
        )
        info["action_saturation_rate"] = float(
            np.mean([bool(execution.get("action_clipped", False)) for execution in action_info.values()])
            if action_info
            else 0.0
        )
        info["ekf_diagnostics"] = {
            uav_id: ekf.diagnostics_snapshot() for uav_id, ekf in self.ekfs.items()
        }
        info["message_buffer"] = {
            "pending": len(self.pending_messages),
            "arrived": len(self.arrived_messages),
            "events": copy.deepcopy(self.message_events),
        }
        for uav_id in self.uav_ids:
            action_info[uav_id]["bandwidth_allocation"] = applied_bandwidth[uav_id]
        self.last_transition = {
            "observation": obs,
            "reward": rewards,
            "info": info,
            "snapshot": self.snapshot,
            "collision": collision,
            "termination_reason": termination_reason,
        }
        return obs, rewards, terminated, truncated, info

    def _make_obstacles(self, scenario: Mapping[str, Any]) -> list[Obstacle]:
        if "obstacles" in scenario:
            return list(scenario["obstacles"])
        specs = scenario.get("obstacle_specs", [])
        if isinstance(specs, list) and specs:
            obstacles: list[Obstacle] = []
            for spec in specs:
                if not isinstance(spec, Mapping):
                    continue
                geometry = str(spec.get("geometry_type", "")).lower()
                obstacle_id = str(spec.get("obstacle_id", f"obstacle_{len(obstacles)}"))
                center = np.asarray(spec.get("center_m", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
                velocity = np.asarray(spec.get("velocity_mps", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
                active = bool(int(spec.get("active", 1)))
                if geometry in {"box", "aabb", "building"}:
                    obstacles.append(
                        BoxObstacle(
                            obstacle_id,
                            center,
                            np.asarray(spec.get("size_m", (1.0, 1.0, 1.0)), dtype=float).reshape(3),
                            velocity,
                            str(spec.get("obstacle_type", "building")),
                            active,
                            _as_bool(spec.get("physical_collision_enabled"), True),
                        )
                    )
                elif geometry in {"sphere", "vehicle", "person", "pedestrian"}:
                    obstacles.append(
                        SphereObstacle(
                            obstacle_id,
                            center,
                            float(spec.get("radius_m", 1.0)),
                            velocity,
                            str(spec.get("obstacle_type", "dynamic_vehicle")),
                            active,
                            _as_bool(spec.get("physical_collision_enabled"), True),
                        )
                    )
            if obstacles:
                return obstacles
        return [
            BoxObstacle(
                "building_0",
                np.array([85.0, 100.0, 12.5]),
                np.array([30.0, 42.0, 25.0]),
            ),
            BoxObstacle(
                "building_1",
                np.array([135.0, 78.0, 20.0]),
                np.array([34.0, 28.0, 40.0]),
            ),
            SphereObstacle(
                "vehicle_0",
                np.array([100.0, 130.0, 1.5]),
                3.0,
                np.array([2.0, -1.5, 0.0]),
            ),
        ]

    def _update_obstacles(self) -> None:
        lower = np.asarray(self.config.motion.bounds_min_m, dtype=float)
        upper = np.asarray(self.config.motion.bounds_max_m, dtype=float)
        for obstacle in self.obstacles:
            obstacle.update(self.config.motion.dt_s, lower, upper)

    def _safety_project(self, uav_id: str, proposed: np.ndarray) -> np.ndarray:
        current = self.states[uav_id].position_m
        for obstacle in self.obstacles:
            if not is_physical_obstacle(obstacle):
                continue
            if (
                (
                    self.config.dynamic_obstacle_swept_projection_enabled
                    and moving_obstacle_swept_segment_intersects(
                        current,
                        proposed,
                        obstacle,
                        self.config.motion.dt_s,
                        self.config.motion.bounds_min_m,
                        self.config.motion.bounds_max_m,
                        self.config.motion.safety_distance_m,
                    )
                )
                or
                segment_intersects_obstacle(current, proposed, obstacle)
                or point_obstacle_distance(proposed, obstacle) < self.config.motion.safety_distance_m
            ):
                return current.copy()
        for other_id, state in self.states.items():
            if other_id != uav_id and np.linalg.norm(proposed - state.position_m) < self.config.motion.safety_distance_m:
                return current.copy()
        return proposed

    def _sensor_rng(self, kind: str, uav_id: str) -> np.random.Generator:
        if self.config.random_stream_mode == "legacy":
            return self.rng
        return indexed_rng(self._episode_random_seed, self.timestep, kind, uav_id)

    def _refresh_topology(
        self,
        bandwidth: dict[str, dict[str, float]],
        *,
        measurement_arrivals: Mapping[tuple[str, str], float] | None = None,
    ) -> TopologySnapshot:
        draw_index = self._topology_draw_index
        self._topology_draw_index += 1
        return build_topology(
            self.states,
            self.obstacles,
            bandwidth,
            self.queue_state,
            self.aoi_state,
            rng=self.rng,
            dt_s=self.config.motion.dt_s,
            channel_config=self.config.channel,
            topology_config=self.config.topology,
            sensor_config=self.config.sensor,
            # A successful packet is not fresh information until its delayed
            # arrival window is released by _release_messages.
            aoi_reset_on_success=False,
            measurement_arrivals=measurement_arrivals,
            rng_factory=(lambda kind, source, target: indexed_rng(self._episode_random_seed, draw_index, kind, source, target))
            if self.config.random_stream_mode == "indexed-v1" else None,
        )

    def _potential_edges_for_current_states(self) -> set[tuple[str, str]]:
        """Return deterministic directed potential edges for current truth states.

        This helper intentionally contains only the range test used by the
        topology definition.  It does not sample UWB or channel randomness,
        so using it for post-motion action validation cannot change the
        episode's random sequence.
        """

        maximum_range = float(self.config.topology.max_comm_range_m)
        return {
            (source, target)
            for source in self.uav_ids
            for target in self.uav_ids
            if source != target
            and float(
                np.linalg.norm(
                    self.states[source].position_m - self.states[target].position_m
                )
            ) <= maximum_range
        }

    def _validate_post_motion_measurements(
        self,
        actions: Mapping[str, Mapping[str, Any]],
        potential_edges: set[tuple[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """Drop measurement requests whose directed edge vanished after motion."""

        validated: dict[str, dict[str, Any]] = {
            uav_id: {
                **dict(command),
                "bandwidth": dict(command.get("bandwidth", {})),
                "delta_v": np.asarray(command.get("delta_v", (0.0, 0.0, 0.0)), dtype=float).copy(),
            }
            for uav_id, command in actions.items()
        }
        for target in self.uav_ids:
            source = str(validated[target].get("measurement_uav_id", "none"))
            if source == "none" or source == target:
                continue
            key = (source, target)
            if key in potential_edges:
                continue
            self.measurement_transition_invalidated_events.append(
                {
                    "step": int(self.timestep),
                    "timestamp": float(self.timestep * self.config.motion.dt_s),
                    "source_uav_id": source,
                    "target_uav_id": target,
                    "invalid_measurement_action": 0,
                    "measurement_transition_invalidated": 1,
                    "reason": "post_motion_not_potential",
                    "pre_action_potential_edge": int(key in self.snapshot.potential_edges),
                    "post_motion_potential_edge": 0,
                }
            )
            validated[target]["measurement_uav_id"] = "none"
        return validated

    def _measurement_overhead_arrivals(
        self,
        actions: Mapping[str, Mapping[str, Any]],
        *,
        allowed_potential_edges: set[tuple[str, str]] | None = None,
    ) -> dict[tuple[str, str], float]:
        """Return one equivalent packet load for each selected measurement."""

        if not self.config.measurement_overhead_enabled:
            return {}
        if int(self.config.channel.packet_size_bytes) != MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES:
            raise ValueError(
                "measurement overhead requires the channel packet size to be "
                f"{MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES} bytes"
            )
        allowed = (
            set(allowed_potential_edges)
            if allowed_potential_edges is not None
            else self._potential_edges_for_current_states()
        )
        arrivals: dict[tuple[str, str], float] = {}
        for target in self.uav_ids:
            source = str(actions[target].get("measurement_uav_id", "none"))
            key = (source, target)
            if source != "none" and source != target and key in allowed:
                arrivals[key] = arrivals.get(key, 0.0) + 1.0
        return arrivals

    def _measurement_overhead_rows(
        self,
        actions: Mapping[str, Mapping[str, Any]],
        measurement_arrivals: Mapping[tuple[str, str], float],
        queue_before: Mapping[tuple[str, str], float],
    ) -> list[dict[str, Any]]:
        """Describe the queue-load proxy without changing EKF semantics."""

        if not self.config.measurement_overhead_enabled:
            return []
        timestamp = (self.timestep + 1) * self.config.motion.dt_s
        rows: list[dict[str, Any]] = []
        for target in self.uav_ids:
            source = str(actions[target].get("measurement_uav_id", "none"))
            if source == "none" or source == target:
                continue
            key = (source, target)
            metrics = self.snapshot.metrics.get(key)
            requested = 1.0
            if key not in self.snapshot.potential_edges:
                # _validate_post_motion_measurements should make this
                # unreachable.  Keep the guard explicit so a future caller
                # cannot silently write a partial load.
                raise RuntimeError(
                    f"selected measurement edge {source}->{target} is not potential"
                )
            applied = float(measurement_arrivals.get(key, 0.0))
            if abs(applied - requested) > 1e-12:
                raise RuntimeError(
                    f"selected measurement edge {source}->{target} did not receive exactly one packet"
                )
            before = max(0.0, float(queue_before.get(key, 0.0)))
            with_load = float(metrics.queue_length_packets) if metrics is not None else before + applied
            next_queue = float(self.snapshot.queue_next.get(key, before + applied))
            rows.append(
                {
                    "timestamp": float(timestamp),
                    "source_uav_id": source,
                    "target_uav_id": target,
                    "selected_non_none": 1,
                    "invalid_measurement_action": 0,
                    "overhead_mode": "equivalent_queue_load",
                    "payload_size_bytes": int(MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES),
                    "load_packets_requested": float(requested),
                    "load_packets_applied": float(applied),
                    "potential_edge": int(key in self.snapshot.potential_edges),
                    "candidate_edge": int(metrics.candidate_edge) if metrics is not None else 0,
                    "effective_edge": int(key in self.snapshot.effective_edges),
                    "queue_before_packets": float(before),
                    "queue_with_measurement_packets": float(with_load),
                    "queue_next_packets": float(next_queue),
                    "channel_bandwidth_hz": float(metrics.channel_bandwidth_hz) if metrics is not None else 0.0,
                    "rate_bps": float(metrics.rate_bps) if metrics is not None else 0.0,
                    "packet_loss": float(metrics.packet_loss) if metrics is not None else 1.0,
                    "delay_s": float(metrics.delay_s) if metrics is not None else self.config.channel.max_delay_s,
                    "aoi_s": float(metrics.aoi_s) if metrics is not None else self.config.channel.aoi_cap_s,
                    "link_state": str(metrics.link_state) if metrics is not None else "not_potential",
                    "ekf_behavior_unchanged": 1,
                }
            )
        return rows

    def _enqueue_messages(
        self,
        transmitted_edges: set[tuple[str, str]],
        sender_payloads: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Create packets from successful transmissions for the next window."""

        current_time = self.timestep * self.config.motion.dt_s
        dt_s = self.config.motion.dt_s
        for source, target in sorted(transmitted_edges):
            metrics = self.snapshot.metrics.get((source, target))
            payload = sender_payloads.get(source)
            if metrics is None or payload is None:
                continue
            delay_s = max(0.0, float(metrics.delay_s))
            arrival_step = self.timestep + max(1, int(np.ceil(delay_s / max(dt_s, 1e-9))))
            self.pending_messages.append(
                MessageRecord(
                    source=source,
                    target=target,
                    state=np.asarray(payload[0], dtype=float).copy(),
                    covariance=np.asarray(payload[1], dtype=float).copy(),
                    send_time_s=current_time,
                    arrival_step=arrival_step,
                    delay_s=delay_s,
                )
            )

    def _propagate_message(self, message: MessageRecord, current_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a delayed constant-velocity state to its arrival time."""

        age = max(0.0, float(current_time_s) - float(message.send_time_s))
        return self._propagate_state_covariance(message.state, message.covariance, age)

    def _propagate_state_covariance(
        self,
        state_at_start: np.ndarray,
        covariance_at_start: np.ndarray,
        age_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate one packet state/covariance by a nonnegative interval."""

        age = max(0.0, float(age_s))
        state = np.asarray(state_at_start, dtype=float).copy()
        covariance = np.asarray(covariance_at_start, dtype=float).copy()
        transition = np.eye(STATE_DIM, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * age
        state[0:3] += state[3:6] * age
        process = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        process[0:3, 0:3] = np.eye(3) * self.config.ekf.process_position_std_m**2 * max(1.0, age)
        process[3:6, 3:6] = np.eye(3) * self.config.ekf.process_velocity_std_mps**2 * max(1.0, age)
        covariance = transition @ covariance @ transition.T + process
        covariance = 0.5 * (covariance + covariance.T)
        return state, covariance

    def _release_messages(self, next_timestep: int) -> None:
        """Move arrived packets into the usable-message buffer and expire old ones."""

        current_time = next_timestep * self.config.motion.dt_s
        remaining: list[MessageRecord] = []
        self.message_events = []
        for message in self.pending_messages:
            if message.arrival_step > next_timestep:
                remaining.append(message)
                continue
            age = current_time - message.send_time_s
            if age > self.config.ekf.stale_message_limit_s:
                self.stale_neighbor_messages += 1
                self.message_events.append(
                    {"source": message.source, "target": message.target, "status": "expired_before_arrival"}
                )
                continue
            state, covariance = self._propagate_message(message, current_time)
            message.state = state
            message.covariance = covariance
            message.arrival_time_s = current_time
            message.last_propagated_time_s = current_time
            self.arrived_messages[(message.source, message.target)] = message
            key = (message.source, message.target)
            self.aoi_state[key] = min(self.config.channel.aoi_cap_s, float(message.delay_s))
            if key in self.snapshot.metrics:
                self.snapshot.metrics[key] = replace(
                    self.snapshot.metrics[key],
                    aoi_s=self.aoi_state[key],
                )
            self.message_events.append(
                {
                    "source": message.source,
                    "target": message.target,
                    "status": "arrived",
                    "send_time_s": float(message.send_time_s),
                    "arrival_time_s": float(current_time),
                    "delay_s": float(message.delay_s),
                }
            )
        self.pending_messages = remaining
        expired_keys = []
        for key, message in self.arrived_messages.items():
            if current_time - message.send_time_s > self.config.ekf.stale_message_limit_s:
                expired_keys.append(key)
        for key in expired_keys:
            self.arrived_messages.pop(key, None)
            self.stale_neighbor_messages += 1
        self._advance_arrived_messages(current_time)

    def _advance_arrived_messages(self, current_time_s: float) -> None:
        """Propagate still-valid arrived packets to the current time.

        A delayed packet can remain usable for a bounded freshness window,
        but its state uncertainty must grow as it is propagated.  This avoids
        treating the same packet as a sequence of independent, zero-age
        neighbor observations.
        """

        for message in self.arrived_messages.values():
            last_time = message.last_propagated_time_s
            if last_time is None:
                last_time = message.arrival_time_s
            if last_time is None or current_time_s <= float(last_time):
                continue
            state, covariance = self._propagate_state_covariance(
                message.state,
                message.covariance,
                float(current_time_s) - float(last_time),
            )
            message.state = state
            message.covariance = covariance
            message.last_propagated_time_s = float(current_time_s)

    def _visible_effective_edges(self) -> set[tuple[str, str]]:
        current_time = (self.timestep + 1) * self.config.motion.dt_s
        return {
            key
            for key, message in self.arrived_messages.items()
            if key in self.snapshot.potential_edges
            and current_time - message.send_time_s <= self.config.ekf.stale_message_limit_s
        }

    @staticmethod
    def _predicted_obstacle_clearance(
        point: np.ndarray,
        obstacle: Obstacle,
        dt_s: float,
    ) -> float:
        """Return the surface clearance at a one-window predicted point."""

        point = np.asarray(point, dtype=float).reshape(3)
        velocity = np.asarray(
            obstacle.velocity_mps if obstacle.velocity_mps is not None else np.zeros(3),
            dtype=float,
        ).reshape(3)
        predicted_obstacle = copy.copy(obstacle)
        predicted_obstacle.center_m = np.asarray(obstacle.center_m, dtype=float) + velocity * float(dt_s)
        if isinstance(predicted_obstacle, BoxObstacle):
            half_size = np.asarray(predicted_obstacle.size_m, dtype=float) / 2.0
            delta = np.maximum(
                np.abs(point - np.asarray(predicted_obstacle.center_m, dtype=float)) - half_size,
                0.0,
            )
            return float(np.linalg.norm(delta))
        if isinstance(predicted_obstacle, SphereObstacle):
            return max(
                0.0,
                float(
                    np.linalg.norm(
                        point - np.asarray(predicted_obstacle.center_m, dtype=float)
                    )
                    - predicted_obstacle.radius_m
                ),
            )
        return float(point_obstacle_distance(point, predicted_obstacle))

    def _local_obstacle_features(self, estimate: np.ndarray) -> np.ndarray:
        """Build the actor-visible aggregate obstacle observation.

        The legacy four values are retained byte-for-byte when the extension
        switch is disabled.  The optional four-value extension contains only
        aggregate, execution-time estimates; obstacle IDs, absolute obstacle
        coordinates, collision labels, and future simulator states are not
        exposed to the actor.
        """

        estimate = np.asarray(estimate, dtype=float).reshape(-1)
        position = np.zeros(3, dtype=float)
        position[: min(3, estimate.size)] = estimate[: min(3, estimate.size)]
        velocity = np.zeros(3, dtype=float)
        if estimate.size >= 6:
            velocity[:] = estimate[3:6]
        physical_obstacles = [
            obstacle for obstacle in self.obstacles if is_physical_obstacle(obstacle)
        ]
        current_min_clearance = min(
            (point_obstacle_distance(position, obstacle) for obstacle in physical_obstacles),
            default=0.0,
        )
        base = np.asarray(
            [
                float(current_min_clearance),
                float(sum(isinstance(obstacle, BoxObstacle) for obstacle in physical_obstacles)),
                float(sum(isinstance(obstacle, SphereObstacle) for obstacle in physical_obstacles)),
                float(self.timestep),
            ],
            dtype=float,
        )
        if not self.config.dynamic_obstacle_motion_features_enabled:
            return base

        dynamic_obstacles: list[Obstacle] = []
        for obstacle in physical_obstacles:
            if not obstacle.active or obstacle.velocity_mps is None:
                continue
            obstacle_velocity = np.asarray(obstacle.velocity_mps, dtype=float).reshape(-1)
            if obstacle_velocity.size != 3 or not np.isfinite(obstacle_velocity).all():
                continue
            obstacle_type = str(getattr(obstacle, "obstacle_type", "")).lower()
            # Buildings with a zero velocity are static scene geometry.  A
            # non-building obstacle remains a dynamic object even when it is
            # momentarily stationary, so its observation remains valid.
            if obstacle_type in {"building", "static", "static_obstacle"} and np.linalg.norm(obstacle_velocity) < 1e-12:
                continue
            dynamic_obstacles.append(obstacle)

        if not dynamic_obstacles:
            # Zero is a conservative finite sentinel.  The validity flag is
            # explicit, so missing motion is never interpreted as safety.
            return np.concatenate([base, np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)])

        current_clearances: list[float] = []
        predicted_clearances: list[float] = []
        closing_rates: list[float] = []
        dt_s = float(self.config.motion.dt_s)
        predicted_position = position + velocity * dt_s
        for obstacle in dynamic_obstacles:
            obstacle_velocity = np.asarray(obstacle.velocity_mps, dtype=float).reshape(3)
            relative = np.asarray(obstacle.center_m, dtype=float).reshape(3) - position
            distance = float(np.linalg.norm(relative))
            direction = relative / (distance + 1e-12)
            range_rate = float(np.dot(direction, obstacle_velocity - velocity))
            current_clearances.append(float(point_obstacle_distance(position, obstacle)))
            closing_rates.append(max(0.0, -range_rate))
            predicted_clearances.append(
                self._predicted_obstacle_clearance(predicted_position, obstacle, dt_s)
            )
        dynamic_features = np.asarray(
            [
                min(current_clearances),
                max(closing_rates),
                min(predicted_clearances),
                1.0,
            ],
            dtype=float,
        )
        return np.concatenate([base, dynamic_features])

    def _build_graph_batch(
        self,
        estimates: dict[str, np.ndarray] | None = None,
        covariances: dict[str, np.ndarray] | None = None,
    ) -> GraphBatch:
        estimates = estimates or {uav_id: ekf.state.copy() for uav_id, ekf in self.ekfs.items()}
        covariances = covariances or {uav_id: ekf.covariance.copy() for uav_id, ekf in self.ekfs.items()}
        local_obstacles = {
            uav_id: self._local_obstacle_features(
                np.asarray(estimate, dtype=float)
            )
            for uav_id, estimate in estimates.items()
        }
        return build_graph_batch(
            self.states,
            estimates,
            covariances,
            self.goals,
            local_obstacles,
            self.snapshot,
            visible_messages={
                # graph_interface uses (target, source) to answer the local
                # observation of receiver target.  The environment keeps the
                # physical packet key as (source, target) everywhere else.
                (message.target, message.source): {
                    "state": message.state,
                    "covariance": message.covariance,
                }
                for key, message in self.arrived_messages.items()
            },
        )

    def _observations(self) -> dict[str, dict[str, Any]]:
        if self.graph_batch is None:
            return {}
        observations: dict[str, dict[str, Any]] = {}
        for uav_id, local in self.graph_batch.local_observations.items():
            measurement_update_eligibility = self._measurement_update_eligibility(uav_id)
            observation = {
                **local,
                "remaining_time_fraction": max(0.0, 1.0 - self.timestep / self.config.max_steps),
                # ``local`` is deliberately actor-visible.  The centralized
                # graph remains available through info/critic_state only.
                "node_features": local["node_features"].copy(),
                "effective_edge_index": local["effective_edge_index"].copy(),
                "effective_edge_features": local["effective_edge_features"].copy(),
                "candidate_edge_index": local["candidate_edge_index"].copy(),
                "potential_edge_index": local["potential_edge_index"].copy(),
                "potential_edge_features": {
                    target: features.copy()
                    for target, features in local["potential_edge_features"].items()
                },
                "bandwidth_mask": local["bandwidth_mask"].copy(),
                "action_mask": local["action_mask"].copy(),
                "measurement_update_eligibility": measurement_update_eligibility.copy(),
            }
            if self.config.measurement_update_eligibility_enabled:
                action_mask = np.asarray(observation["action_mask"], dtype=bool).copy()
                action_mask[: self.config.uav_count] &= measurement_update_eligibility
                # ``none`` is always a legal fallback while every candidate
                # link is waiting for its next physical sensor period.
                action_mask[self.config.uav_count] = True
                observation["action_mask"] = action_mask
            if self.config.measurement_budget_enabled:
                action_mask = np.asarray(observation["action_mask"], dtype=bool).copy()
                if self.measurement_budget_remaining.get(uav_id, 0) <= 0:
                    action_mask[: self.config.uav_count] = False
                    action_mask[self.config.uav_count] = True
                observation["action_mask"] = action_mask
                limit = float(self.measurement_budget_limits.get(uav_id, 0))
                remaining = float(self.measurement_budget_remaining.get(uav_id, 0))
                observation["measurement_budget_limit"] = limit
                observation["measurement_budget_used"] = float(
                    self.measurement_budget_used.get(uav_id, 0)
                )
                observation["measurement_budget_remaining"] = remaining
                observation["measurement_budget_remaining_ratio"] = (
                    float(np.clip(remaining / limit, 0.0, 1.0)) if limit > 0.0 else 0.0
                )
            observations[uav_id] = observation
        return observations

    def _measurement_update_eligibility(self, target_uav_id: str) -> np.ndarray:
        """Return source-UAV eligibility for the next decision window.

        A policy acts before the next transition, so eligibility is evaluated
        at that transition's timestamp.  It uses only the last physical UWB
        update time recorded by this environment and never uses future
        measurements, true state, or the eventual EKF result.
        """

        next_measurement_time = (self.timestep + 1) * self.config.motion.dt_s
        period = float(self.config.sensor.uwb_update_period_s)
        values = np.zeros(self.config.uav_count, dtype=bool)
        for index, source_uav_id in enumerate(self.uav_ids):
            if source_uav_id == target_uav_id:
                continue
            last_update_time = self.uwb_update_times.get((source_uav_id, target_uav_id))
            values[index] = (
                last_update_time is None
                or next_measurement_time - float(last_update_time) >= period - 1e-9
            )
        return values

    def _normalize_actions(self, action: Mapping[str, Any] | Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        if "delta_v" in action:
            actions: dict[str, Mapping[str, Any]] = {self.uav_ids[0]: action}  # type: ignore[dict-item]
        else:
            actions = {str(key): value for key, value in action.items()}  # type: ignore[union-attr]
        normalized: dict[str, dict[str, Any]] = {}
        self.measurement_action_diagnostics = {}
        for uav_id in self.uav_ids:
            raw = dict(actions.get(uav_id, {}))
            delta_v = np.asarray(raw.get("delta_v", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
            potential_sources = {
                source
                for source, target in self.snapshot.potential_edges
                if target == uav_id
            }
            measurement_candidates = {
                source
                for source, target in self.snapshot.candidate_edges
                if target == uav_id
                and source in potential_sources
                and self.snapshot.uwb_measurements.get((source, uav_id), {}).get("valid", 0)
            }
            candidate_count_before_timing = int(len(measurement_candidates))
            eligible_candidate_count = candidate_count_before_timing
            timing_filtered_count = 0
            if self.config.measurement_update_eligibility_enabled:
                eligibility = self._measurement_update_eligibility(uav_id)
                eligible_sources = {
                    source
                    for index, source in enumerate(self.uav_ids)
                    if bool(eligibility[index])
                }
                measurement_candidates &= eligible_sources
                eligible_candidate_count = int(len(measurement_candidates))
                timing_filtered_count = int(
                    candidate_count_before_timing - eligible_candidate_count
                )
            budget_before = int(self.measurement_budget_remaining.get(uav_id, 0))
            if self.config.measurement_budget_enabled and self.measurement_budget_remaining.get(uav_id, 0) <= 0:
                measurement_candidates = set()
            budget_filtered = bool(
                self.config.measurement_budget_enabled
                and budget_before <= 0
                and candidate_count_before_timing > 0
            )
            measurement_uav_id = str(raw.get("measurement_uav_id", "none"))
            requested_measurement_uav_id = measurement_uav_id
            invalid_reason = ""
            if measurement_uav_id != "none" and measurement_uav_id not in measurement_candidates:
                key = (measurement_uav_id, uav_id)
                reason = "pre_action_not_actor_eligible"
                if self.config.measurement_update_eligibility_enabled:
                    source_index = (
                        self.uav_ids.index(measurement_uav_id)
                        if measurement_uav_id in self.uav_ids
                        else -1
                    )
                    if source_index >= 0:
                        eligibility = self._measurement_update_eligibility(uav_id)
                        if not bool(eligibility[source_index]):
                            reason = "pre_action_update_period_not_elapsed"
                invalid_reason = reason
                self.invalid_measurement_action_events.append(
                    {
                        "step": int(self.timestep),
                        "timestamp": float(self.timestep * self.config.motion.dt_s),
                        "source_uav_id": measurement_uav_id,
                        "target_uav_id": uav_id,
                        "invalid_measurement_action": 1,
                        "measurement_transition_invalidated": 0,
                        "reason": reason,
                        "pre_action_potential_edge": int(key in self.snapshot.potential_edges),
                        "pre_action_candidate_edge": int(key in self.snapshot.candidate_edges),
                    }
                )
            normalized_measurement_uav_id = (
                measurement_uav_id
                if measurement_uav_id == "none" or measurement_uav_id in measurement_candidates
                else "none"
            )
            none_forced_by_timing_mask = bool(
                normalized_measurement_uav_id == "none"
                and candidate_count_before_timing > 0
                and eligible_candidate_count == 0
                and timing_filtered_count > 0
            )
            none_when_candidate_eligible = bool(
                normalized_measurement_uav_id == "none"
                and eligible_candidate_count > 0
                and not budget_filtered
            )
            self.measurement_action_diagnostics[uav_id] = {
                "step": int(self.timestep),
                "uav_id": uav_id,
                "requested_measurement_uav_id": requested_measurement_uav_id,
                "normalized_measurement_uav_id": normalized_measurement_uav_id,
                "candidate_count_before_timing": candidate_count_before_timing,
                "eligible_candidate_count": int(eligible_candidate_count),
                "timing_filtered_candidate_count": timing_filtered_count,
                "budget_remaining_before": budget_before,
                "budget_filtered": int(budget_filtered),
                "none_forced_by_timing_mask": int(none_forced_by_timing_mask),
                "none_when_candidate_eligible": int(none_when_candidate_eligible),
                "invalid_reason": invalid_reason,
            }
            normalized[uav_id] = {
                "measurement_uav_id": normalized_measurement_uav_id,
                "bandwidth": {str(key): float(value) for key, value in dict(raw.get("bandwidth", {})).items()},
                "delta_v": delta_v,
            }
        return normalized

    def _consume_measurement_budget(
        self,
        actions: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Consume at most one fixed budget unit per UAV and decision step."""

        if not self.config.measurement_budget_enabled:
            return []
        rows: list[dict[str, Any]] = []
        for uav_id in self.uav_ids:
            before = int(self.measurement_budget_remaining.get(uav_id, 0))
            selected = str(actions[uav_id].get("measurement_uav_id", "none"))
            consumed = int(selected != "none" and before > 0)
            after = max(0, before - consumed)
            self.measurement_budget_remaining[uav_id] = after
            self.measurement_budget_used[uav_id] = int(
                self.measurement_budget_used.get(uav_id, 0) + consumed
            )
            rows.append(
                {
                    "step": int(self.timestep),
                    "uav_id": uav_id,
                    "measurement_uav_id": selected,
                    "budget_limit": int(self.measurement_budget_limits[uav_id]),
                    "budget_remaining_before": before,
                    "budget_consumed": consumed,
                    "budget_remaining_after": after,
                    "budget_used": int(self.measurement_budget_used[uav_id]),
                }
            )
        return rows

    @staticmethod
    def _accepted_update_feedback_quality(
        event: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        """Return ``(quality, trace_gain, consistency, directional_info)``.

        The gain is deliberately measured before the conservative local
        marginal-covariance restoration.  A local EKF does not maintain the
        full cross-covariance with the neighbour, so the restored covariance
        is the physical reporting state and is not silently treated as a
        guaranteed reduction.  This bounded quantity is only a training
        credit proxy and is logged with that meaning.
        """

        if not bool(event.get("accepted", False)):
            return 0.0, 0.0, 0.0, 0.0
        before = float(event.get("position_covariance_trace_before_m2", float("nan")))
        after_linear = float(
            event.get("position_covariance_trace_after_linear_m2", float("nan"))
        )
        nis = float(event.get("nis", float("nan")))
        if not np.isfinite(before) or before <= 1e-12 or not np.isfinite(after_linear):
            return 0.0, 0.0, 0.0, 0.0
        relative_gain = float(np.clip((before - after_linear) / before, 0.0, 1.0))
        local_direction_variance = float(
            event.get("local_direction_variance_m2", float("nan"))
        )
        effective_measurement_variance = float(
            event.get("effective_measurement_variance_m2", float("nan"))
        )
        if (
            np.isfinite(local_direction_variance)
            and local_direction_variance >= 0.0
            and np.isfinite(effective_measurement_variance)
            and effective_measurement_variance > 0.0
        ):
            # This is the observable information fraction of the scalar range
            # in its current EKF direction.  It distinguishes candidates even
            # when the 3-D marginal trace is nearly isotropic, without using
            # truth, RMSE, or a future state.
            directional_information = float(
                np.clip(
                    local_direction_variance
                    / (local_direction_variance + effective_measurement_variance),
                    0.0,
                    1.0,
                )
            )
        else:
            directional_information = relative_gain
        if not np.isfinite(nis):
            consistency = 0.0
        else:
            consistency = float(np.exp(-0.5 * np.clip(max(0.0, nis), 0.0, 20.0)))
        quality = float(
            np.clip(0.5 * directional_information + 0.5 * consistency, 0.0, 1.0)
        )
        return quality, relative_gain, consistency, directional_information

    def _accepted_update_feedback(
        self,
        actions: Mapping[str, Mapping[str, Any]],
        *,
        terminal: bool,
    ) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
        """Schedule and release delayed credit for selected UWB updates.

        A selected source is linked to an event only by ``(target, source)``
        and the current transition's event list.  The feedback is released on
        a later decision window (or flushed at a terminal transition), so the
        reward channel exposes the intended selection -> communication -> EKF
        update -> learning-feedback order.  No true position, RMSE, LOS/NLOS
        truth, or future state is read here.
        """

        current_step = int(self.timestep)
        due = [
            item
            for item in self._pending_update_feedback
            if int(item.get("release_step", current_step + 1)) <= current_step
        ]
        self._pending_update_feedback = [
            item
            for item in self._pending_update_feedback
            if int(item.get("release_step", current_step + 1)) > current_step
        ]
        records: list[dict[str, Any]] = []
        current_events = list(self.relative_range_update_events)
        delay = int(self.config.reward.accepted_update_feedback_delay_steps)
        for target_uav_id in self.uav_ids:
            selected_source = str(actions[target_uav_id].get("measurement_uav_id", "none"))
            if selected_source in {"none", target_uav_id}:
                continue
            matching = [
                event
                for event in current_events
                if str(event.get("target_uav_id", "")) == target_uav_id
                and str(event.get("source_uav_id", "")) == selected_source
            ]
            event = matching[-1] if matching else None
            quality, relative_gain, consistency, directional_information = (
                self._accepted_update_feedback_quality(event)
                if event is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            # A terminal transition has no subsequent decision window.  Flush
            # its current action's credit at the terminal reward so the last
            # accepted update is not silently discarded.
            release_step = (
                current_step
                if terminal or delay <= 0
                else current_step + delay
            )
            pending = {
                "target_uav_id": target_uav_id,
                "source_uav_id": selected_source,
                "selected_step": current_step,
                "release_step": release_step,
                "accepted": int(bool(event and event.get("accepted", False))),
                "event_observed": int(event is not None),
                "quality": float(quality),
                "none_baseline_quality": float(
                    self.config.reward.accepted_update_feedback_none_baseline
                ),
                "relative_covariance_gain": float(relative_gain),
                "innovation_consistency": float(consistency),
                "directional_information_fraction": float(directional_information),
                "feedback_covariance_source": "pre_restoration_linear_update",
            }
            if self.config.reward.learning_feedback_mode == "accepted_update_progress_v2":
                raw_difference = float(
                    quality
                    - self.config.reward.accepted_update_feedback_none_baseline
                )
                normalized_difference = (
                    raw_difference - self.config.reward.accepted_update_feedback_location
                ) / max(self.config.reward.accepted_update_feedback_scale, 1e-6)
                pending["feedback_difference_raw"] = float(raw_difference)
                pending["feedback_location"] = float(
                    self.config.reward.accepted_update_feedback_location
                )
                pending["feedback_scale"] = float(
                    self.config.reward.accepted_update_feedback_scale
                )
                pending["feedback_normalized"] = float(
                    np.tanh(np.clip(normalized_difference, -8.0, 8.0))
                )
                pending["feedback_value"] = float(pending["feedback_normalized"])
                pending["feedback_source"] = "accepted_update_minus_none_robust_warmup_scale"
            else:
                pending["feedback_value"] = float(quality)
                pending["feedback_source"] = "pre_restoration_quality_v1"
            records.append(dict(pending))
            if release_step <= current_step:
                due.append(pending)
            else:
                self._pending_update_feedback.append(pending)
        denominator = max(1, len(self.uav_ids))
        released_quality = float(
            sum(
                float(item.get("feedback_value", item.get("quality", 0.0)))
                for item in due
            )
            / denominator
        )
        return released_quality, records, due

    def _rewards(
        self,
        collision: int,
        actions: dict[str, dict[str, Any]],
        action_execution: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        weights = self.config.reward.weights
        all_candidate = sorted(self.snapshot.candidate_edges)
        losses = [self.snapshot.metrics[key].packet_loss for key in all_candidate]
        delays = [self.snapshot.metrics[key].delay_s for key in all_candidate]
        aois = [self.snapshot.metrics[key].aoi_s for key in all_candidate]
        global_loss = float(np.mean(losses)) if losses else 1.0
        global_delay = float(np.mean(delays)) if delays else self.config.channel.max_delay_s
        global_aoi = float(np.mean(aois)) if aois else self.config.channel.aoi_cap_s
        uncertainty_raw = float(
            np.mean([np.trace(ekf.covariance[:3, :3]) for ekf in self.ekfs.values()])
        ) if self.ekfs else 0.0

        def scale(name: str, fallback: float) -> float:
            configured = self.config.reward.metric_scales or {}
            return max(1e-9, float(configured.get(name, fallback)))

        configured_scales = self.config.reward.metric_scales or {}
        if self.config.reward.uncertainty_normalization == "warmup_q05_q95":
            if "uncertainty_q05" not in configured_scales or "uncertainty_q95" not in configured_scales:
                raise RuntimeError(
                    "warmup_q05_q95 uncertainty normalization requires frozen "
                    "uncertainty_q05 and uncertainty_q95 scales"
                )
            lower = float(configured_scales["uncertainty_q05"])
            upper = float(configured_scales["uncertainty_q95"])
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                raise RuntimeError("invalid warm-up uncertainty quantile bounds")
            uncertainty_norm = float(np.clip((uncertainty_raw - lower) / (upper - lower), 0.0, 1.0))
        else:
            uncertainty_norm = float(np.clip(uncertainty_raw / scale("uncertainty", self.config.reward.uncertainty_scale), 0.0, 1.0))
        loss_norm = float(np.clip(global_loss / scale("loss", 1.0), 0.0, 1.0))
        aoi_norm = float(np.clip(global_aoi / scale("aoi", self.config.reward.aoi_scale), 0.0, 1.0))
        delay_norm = float(np.clip(global_delay / scale("delay", self.config.reward.delay_scale), 0.0, 1.0))

        # Effective directed links are converted to an undirected graph for
        # the network-connectivity term. The metric is the largest connected
        # component fraction, not merely the fraction of nodes with a neighbor.
        adjacency = {uav_id: set() for uav_id in self.uav_ids}
        for source, target in self.snapshot.effective_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        components: list[int] = []
        unseen = set(self.uav_ids)
        while unseen:
            root = unseen.pop()
            component = {root}
            frontier = [root]
            while frontier:
                node = frontier.pop()
                for neighbor in adjacency[node]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
            components.append(len(component))
        connectivity = float(max(components, default=0) / max(1, len(self.uav_ids)))
        connectivity_norm = float(np.clip(connectivity / scale("connectivity", 1.0), 0.0, 1.0))

        j_loc = float(weights.get("uncertainty", weights.get("unc", 0.0)) * uncertainty_norm)
        network_weight = weights.get("network", weights.get("connectivity", weights.get("net", 0.0)))
        j_com = float(
            weights.get("loss", 0.0) * loss_norm
            + weights.get("aoi", 0.0) * aoi_norm
            + weights.get("delay", 0.0) * delay_norm
            - network_weight * connectivity_norm
        )
        joint_objective = float(j_loc + j_com)

        max_delta = max(
            1e-9,
            self.config.motion.max_acceleration_mps2 * self.config.motion.dt_s,
        )
        executed_deltas = [
            float(
                np.linalg.norm(
                    np.asarray(
                        action_execution[uav_id].get("delta_v_applied", actions[uav_id]["delta_v"]),
                        dtype=float,
                    )
                )
            )
            for uav_id in self.uav_ids
        ]
        energy_normalized = float(np.clip(np.mean(executed_deltas) / max_delta, 0.0, 1.0))
        collision_cost = float(self.config.reward.lambda_coll * float(collision))
        energy_cost = float(self.config.reward.lambda_energy * energy_normalized)
        team_reward = float(-(joint_objective + collision_cost + energy_cost))
        rewards = {uav_id: team_reward for uav_id in self.uav_ids}
        reporting_potential = float(np.clip(-joint_objective, -1.0, 1.0))
        previous_potential = float(self._learning_potential)
        progress_potential = reporting_potential
        progress_uncertainty_logdet = 0.0
        progress_normalized_j_com = 0.0
        if self.config.reward.learning_reward_mode == "progress_potential_v2":
            (
                progress_potential,
                progress_uncertainty_logdet,
                progress_normalized_j_com,
            ) = self._learning_progress_potential(j_com)
            potential_for_shaping = progress_potential
        else:
            potential_for_shaping = reporting_potential
        potential_shaping = float(
            self.config.reward.learning_potential_coef
            * (
                self.config.reward.learning_gamma * potential_for_shaping
                - previous_potential
            )
        )
        learning_collision_cost = (
            float(collision_cost)
            if self.config.reward.learning_reward_mode
            in {"report_plus_feedback", "report_direct"}
            else min(
                collision_cost,
                float(self.config.reward.learning_collision_penalty_cap),
            )
        )
        bounded_learning_reward = float(
            np.clip(
                -(joint_objective + energy_cost + learning_collision_cost)
                + potential_shaping,
                -float(self.config.reward.learning_reward_clip),
                float(self.config.reward.learning_reward_clip),
            )
        )
        v3_learning_reward = (
            team_reward
            if self.config.reward.learning_reward_mode
            in {"legacy", "report_plus_feedback", "report_direct"}
            else bounded_learning_reward
        )
        # v4 feedback is a training-only residual on top of the existing
        # bounded learning channel.  It is calculated after the EKF update so
        # the sign is physically interpretable: a reduction in covariance
        # log-determinant is positive progress.  The reporting reward above is
        # never changed.
        current_position_logdet = self._position_logdet_uncertainty()
        previous_position_logdet = float(self._previous_position_logdet)
        current_goal_distance = self.estimated_goal_distance()
        previous_goal_distance = float(self._previous_estimated_goal_distance)
        single_uav_learning_active = bool(
            len(self.uav_ids) == 1
            and self.config.reward.single_uav_learning_mode == "loc_goal_v1"
        )
        single_uav_localization_progress_delta = float(
            previous_position_logdet - current_position_logdet
        )
        single_uav_goal_progress_delta = float(
            previous_goal_distance - current_goal_distance
        )
        potential_statistics = self.config.reward.potential_statistics or {}
        if self.config.reward.learning_feedback_mode == "loc_progress_v2":
            progress_scale = max(
                float(potential_statistics["delta_u_scale"]),
                float(self.config.reward.potential_epsilon),
            )
        else:
            progress_scale = max(
                float(potential_statistics.get("u_scale", 1.0)),
                float(self.config.reward.potential_epsilon),
            )
        loc_progress_delta_raw = float(
            (previous_position_logdet - current_position_logdet) / progress_scale
        )
        loc_progress_delta = float(
            np.clip(loc_progress_delta_raw, -3.0, 3.0)
            if self.config.reward.learning_feedback_mode == "loc_progress_v2"
            else loc_progress_delta_raw
        )
        self._previous_position_logdet = current_position_logdet
        self._previous_estimated_goal_distance = current_goal_distance
        single_uav_localization_progress_scale = max(
            float(potential_statistics.get("delta_u_scale", 1.0)),
            float(self.config.reward.potential_epsilon),
        )
        single_uav_goal_progress_scale = max(
            float(potential_statistics.get("delta_goal_scale", 1.0)),
            float(self.config.reward.potential_epsilon),
        )
        if single_uav_learning_active:
            # RewardConfig validates these for an explicitly configured
            # loc_goal_v1 environment.  Keep this guard close to the actual
            # calculation so an accidentally unfitted direct environment
            # fails with a useful message rather than silently using scale 1.
            for required_scale in ("delta_u_scale", "delta_goal_scale"):
                if required_scale not in potential_statistics:
                    raise RuntimeError(
                        "loc_goal_v1 requires warm-up-fitted "
                        f"potential_statistics[{required_scale}]"
                    )
        single_uav_localization_progress_score = float(
            np.tanh(
                single_uav_localization_progress_delta
                / single_uav_localization_progress_scale
            )
            if single_uav_learning_active
            else 0.0
        )
        single_uav_goal_progress_score = float(
            np.tanh(
                single_uav_goal_progress_delta / single_uav_goal_progress_scale
            )
            if single_uav_learning_active
            else 0.0
        )
        single_uav_learning_weights = self.config.reward.single_uav_learning_weights
        single_uav_localization_component = float(
            single_uav_learning_weights["localization_progress"]
            * single_uav_localization_progress_score
        )
        single_uav_goal_component = float(
            single_uav_learning_weights["goal_progress"]
            * single_uav_goal_progress_score
        )
        single_uav_learning_collision_cost = float(
            min(collision_cost, float(self.config.reward.learning_collision_penalty_cap))
        )
        single_uav_learning_core = float(
            single_uav_localization_component
            + single_uav_goal_component
            - energy_cost
            - single_uav_learning_collision_cost
        )
        single_uav_learning_reward = float(
            np.clip(
                single_uav_learning_core,
                -float(self.config.reward.learning_reward_clip),
                float(self.config.reward.learning_reward_clip),
            )
        )
        actual_measurement_count = int(
            sum(
                str(actions[uav_id].get("measurement_uav_id", "none")) not in {"none", uav_id}
                for uav_id in self.uav_ids
            )
        )
        actual_measurement_bytes = int(
            actual_measurement_count * MEASUREMENT_OVERHEAD_PACKET_SIZE_BYTES
        )
        actual_measurement_rate = float(actual_measurement_count / max(1, len(self.uav_ids)))
        actual_load_target = (
            float(self.config.reward.actual_measurement_cost_target)
            if self.config.reward.actual_measurement_cost_target is not None
            else 0.0
        )
        actual_load_excess = float(max(0.0, actual_measurement_rate - actual_load_target))
        feedback_mode = self.config.reward.learning_feedback_mode
        loc_progress_reward = float(
            self.config.reward.loc_progress_coef * loc_progress_delta
            if feedback_mode in {
                "loc_progress_v1",
                "loc_progress_v2",
                "loc_progress_actual_load_v1",
            }
            else 0.0
        )
        accepted_update_feedback_reward = 0.0
        update_feedback_records: list[dict[str, Any]] = []
        released_update_feedback: list[dict[str, Any]] = []
        if feedback_mode in {
            "accepted_update_progress_v1",
            "accepted_update_progress_v2",
        }:
            quality_mean, update_feedback_records, released_update_feedback = (
                self._accepted_update_feedback(
                    actions,
                    terminal=bool(
                        collision or (self.timestep + 1 >= self.config.max_steps)
                    ),
                )
            )
            accepted_update_feedback_reward = float(
                self.config.reward.accepted_update_feedback_coef * quality_mean
            )
        self._last_update_feedback_records = copy.deepcopy(update_feedback_records)
        actual_load_reward = float(
            -self.config.reward.actual_measurement_cost_coef * actual_load_excess
            if (
                self.config.reward.actual_measurement_cost_enabled
                and feedback_mode in {"actual_load_v1", "loc_progress_actual_load_v1"}
            )
            else 0.0
        )
        feedback_reward = float(
            loc_progress_reward
            + actual_load_reward
            + accepted_update_feedback_reward
        )
        if self.config.reward.learning_reward_mode == "report_plus_feedback":
            # This is the staged clean-objective intervention: start from the
            # exact physical report reward and add only the explicitly
            # requested residual feedback.  In particular, do not apply the
            # historical component-wise collision cap or the bounded-potential
            # clip to this path; otherwise the learning target would no longer
            # be the report objective being tested.
            learning_reward = float(v3_learning_reward + feedback_reward)
        else:
            learning_reward = float(
                np.clip(
                    v3_learning_reward + feedback_reward,
                    -float(self.config.reward.learning_reward_clip),
                    float(self.config.reward.learning_reward_clip),
                )
            )
        if self.config.reward.learning_reward_mode == "report_direct":
            learning_reward = float(team_reward)
            learning_collision_cost = float(collision_cost)
            potential_shaping = 0.0
        if single_uav_learning_active:
            # Explicit single-UAV mode uses a separate learning channel.
            # The reporting reward and physical communication metrics above
            # remain unchanged; no-link communication fallbacks are therefore
            # not charged to this PPO reward.
            learning_reward = single_uav_learning_reward
        self._learning_potential = potential_for_shaping
        reward_components = {
            uav_id: {
                "collision_penalty": collision_cost,
                "energy_penalty": energy_cost,
                "energy_normalized": energy_normalized,
            }
            for uav_id in self.uav_ids
        }
        return rewards, {
            "reporting_rewards": dict(rewards),
            "learning_rewards": {
                uav_id: float(learning_reward) for uav_id in self.uav_ids
            },
            "learning_potential": float(reporting_potential),
            "learning_progress_potential": float(progress_potential),
            "learning_progress_logdet_uncertainty": float(progress_uncertainty_logdet),
            "learning_progress_normalized_j_com": float(progress_normalized_j_com),
            "communication_applicable": bool(len(self.uav_ids) > 1),
            "single_uav_learning_mode": str(
                self.config.reward.single_uav_learning_mode
            ),
            "single_uav_learning_active": bool(single_uav_learning_active),
            "learning_reward_components": {
                "mode": self.config.reward.learning_reward_mode,
                "reporting_reward": float(team_reward),
                "reporting_reward_total": float(team_reward),
                "learning_reward_total": float(learning_reward),
                "bounded_reward": float(bounded_learning_reward),
                "potential": float(reporting_potential),
                "progress_potential": float(progress_potential),
                "progress_logdet_uncertainty": float(progress_uncertainty_logdet),
                "progress_normalized_j_com": float(progress_normalized_j_com),
                "previous_potential": float(previous_potential),
                "potential_shaping": float(potential_shaping),
                "collision_cost_raw": float(collision_cost),
                "collision_cost_learning": float(learning_collision_cost),
                "energy_cost": float(energy_cost),
                "communication_applicable": bool(len(self.uav_ids) > 1),
                "communication_reward_in_learning": bool(
                    not single_uav_learning_active
                ),
                "single_uav_learning_mode": str(
                    self.config.reward.single_uav_learning_mode
                ),
                "single_uav_learning_active": bool(single_uav_learning_active),
                "single_uav_localization_progress_delta": float(
                    single_uav_localization_progress_delta
                ),
                "single_uav_localization_progress_scale": float(
                    single_uav_localization_progress_scale
                ),
                "localization_progress_score": float(
                    single_uav_localization_progress_score
                ),
                "single_uav_localization_component": float(
                    single_uav_localization_component
                ),
                "previous_goal_distance": float(previous_goal_distance),
                "current_goal_distance": float(current_goal_distance),
                "goal_progress_delta": float(single_uav_goal_progress_delta),
                "goal_progress_scale": float(single_uav_goal_progress_scale),
                "goal_progress_score": float(single_uav_goal_progress_score),
                "single_uav_goal_component": float(single_uav_goal_component),
                "single_uav_learning_energy_cost": float(energy_cost),
                "single_uav_learning_collision_cost": float(
                    single_uav_learning_collision_cost
                ),
                "single_uav_learning_core": float(single_uav_learning_core),
                "single_uav_learning_reward": float(single_uav_learning_reward),
                "learning_feedback_mode": feedback_mode,
                "v3_learning_reward": float(v3_learning_reward),
                "position_logdet_previous": float(previous_position_logdet),
                "position_logdet_current": float(current_position_logdet),
                "loc_progress_delta_raw": float(loc_progress_delta_raw),
                "loc_progress_delta": float(loc_progress_delta),
                "loc_progress_scale": float(progress_scale),
                "loc_progress_scale_source": (
                    "warmup_adjacent_window_mad"
                    if feedback_mode == "loc_progress_v2"
                    else "potential_u_scale"
                ),
                "loc_progress_reward": float(loc_progress_reward),
                "actual_measurement_count": int(actual_measurement_count),
                "actual_measurement_bytes": int(actual_measurement_bytes),
                "actual_measurement_rate": float(actual_measurement_rate),
                "actual_measurement_cost_target": float(actual_load_target),
                "actual_measurement_load_excess": float(actual_load_excess),
                "actual_measurement_load_reward": float(actual_load_reward),
                "accepted_update_feedback_reward": float(
                    accepted_update_feedback_reward
                ),
                "accepted_update_feedback_coef": float(
                    self.config.reward.accepted_update_feedback_coef
                ),
                "accepted_update_feedback_delay_steps": int(
                    self.config.reward.accepted_update_feedback_delay_steps
                ),
                "accepted_update_feedback_records": copy.deepcopy(
                    update_feedback_records
                ),
                "released_update_feedback_records": copy.deepcopy(
                    released_update_feedback
                ),
                "pending_update_feedback_count": int(
                    len(self._pending_update_feedback)
                ),
                "accepted_update_feedback_covariance_source": (
                    "pre_restoration_linear_update"
                    if feedback_mode in {
                        "accepted_update_progress_v1",
                        "accepted_update_progress_v2",
                    }
                    else "not_applicable"
                ),
                "accepted_update_feedback_normalization": {
                    "mode": feedback_mode,
                    "none_baseline": float(
                        self.config.reward.accepted_update_feedback_none_baseline
                    ),
                    "location": float(
                        self.config.reward.accepted_update_feedback_location
                    ),
                    "scale": float(
                        self.config.reward.accepted_update_feedback_scale
                    ),
                    "source": (
                        "warmup_only"
                        if feedback_mode == "accepted_update_progress_v2"
                        else "not_applicable"
                    ),
                },
                "feedback_reward": float(feedback_reward),
            },
            "joint_objective": joint_objective,
            "objective_components": {
                "j_loc": j_loc,
                "j_com": j_com,
                "joint_objective": joint_objective,
            },
            "global_metrics": {
                "uncertainty": uncertainty_raw,
                "uncertainty_normalized": uncertainty_norm,
                "packet_loss": global_loss,
                "packet_loss_normalized": loss_norm,
                "delay": global_delay,
                "delay_normalized": delay_norm,
                "aoi": global_aoi,
                "aoi_normalized": aoi_norm,
                "connectivity": connectivity,
                "connectivity_normalized": connectivity_norm,
                "communication_applicable": bool(len(self.uav_ids) > 1),
                "connectivity_definition": "largest_connected_component_fraction",
                "metric_scale_source": self.config.reward.metric_scale_source,
                "uncertainty_normalization": self.config.reward.uncertainty_normalization,
                "metric_scales": {
                    "uncertainty": scale("uncertainty", self.config.reward.uncertainty_scale),
                    "uncertainty_q05": float(configured_scales.get("uncertainty_q05", float("nan"))),
                    "uncertainty_q95": float(configured_scales.get("uncertainty_q95", float("nan"))),
                    "loss": scale("loss", 1.0),
                    "aoi": scale("aoi", self.config.reward.aoi_scale),
                    "delay": scale("delay", self.config.reward.delay_scale),
                    "connectivity": scale("connectivity", 1.0),
                },
            },
            "reward_components": reward_components,
        }

    def _diagnostic_j_loc(self) -> float:
        """Recompute the documented localization term without changing state."""

        uncertainty_raw = float(
            np.mean([np.trace(ekf.covariance[:3, :3]) for ekf in self.ekfs.values()])
        ) if self.ekfs else 0.0
        configured = self.config.reward.metric_scales or {}
        if self.config.reward.uncertainty_normalization == "warmup_q05_q95":
            if "uncertainty_q05" not in configured or "uncertainty_q95" not in configured:
                raise RuntimeError("missing frozen uncertainty quantile bounds")
            lower = float(configured["uncertainty_q05"])
            upper = float(configured["uncertainty_q95"])
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                raise RuntimeError("invalid frozen uncertainty quantile bounds")
            normalized = float(np.clip((uncertainty_raw - lower) / (upper - lower), 0.0, 1.0))
        else:
            scale = max(
                1e-9,
                float(configured.get("uncertainty", self.config.reward.uncertainty_scale)),
            )
            normalized = float(np.clip(uncertainty_raw / scale, 0.0, 1.0))
        weight = float(
            self.config.reward.weights.get(
                "uncertainty",
                self.config.reward.weights.get("unc", 0.0),
            )
        )
        return float(weight * normalized)

    def _info(self, *, collision: int, minimum_clearance: float, sensor_packets: dict[str, Any]) -> dict[str, Any]:
        central = {
            "remaining_time_fraction": max(0.0, 1.0 - self.timestep / self.config.max_steps),
            "estimates": {uav_id: ekf.state.copy() for uav_id, ekf in self.ekfs.items()},
            "covariances": {uav_id: ekf.covariance.copy() for uav_id, ekf in self.ekfs.items()},
            "potential_edges": sorted(self.snapshot.potential_edges),
            "candidate_edges": sorted(self.snapshot.candidate_edges),
            "effective_edges": sorted(self.snapshot.effective_edges),
            "uwb_edges": sorted(self.snapshot.uwb_edges),
            "ci_weights": {
                f"{target}<-{source}": float(weight)
                for (target, source), weight in self.ci_weights.items()
            },
            "obstacles": [self._obstacle_payload(obstacle) for obstacle in self.obstacles],
            "graph": {
                "node_features": self.graph_batch.node_features.copy() if self.graph_batch is not None else np.empty((0, 0)),
                "effective_edge_index": self.graph_batch.effective_edge_index.copy() if self.graph_batch is not None else np.empty((2, 0), dtype=np.int64),
                "effective_edge_features": self.graph_batch.effective_edge_features.copy() if self.graph_batch is not None else np.empty((0, 11)),
                "candidate_edge_index": self.graph_batch.candidate_edge_index.copy() if self.graph_batch is not None else np.empty((2, 0), dtype=np.int64),
                "candidate_edge_features": self.graph_batch.candidate_edge_features.copy() if self.graph_batch is not None else np.empty((0, 11)),
                "potential_edge_index": self.graph_batch.potential_edge_index.copy() if self.graph_batch is not None else np.empty((2, 0), dtype=np.int64),
                "potential_edge_features": self.graph_batch.potential_edge_features.copy() if self.graph_batch is not None else np.empty((0, 11)),
            },
        }

        return {
            "timestamp": self.timestep * self.config.motion.dt_s,
            "central_state": central,
            "measurement_update_eligibility_enabled": bool(
                self.config.measurement_update_eligibility_enabled
            ),
            "measurement_update_period_s": float(
                self.config.sensor.uwb_update_period_s
            ),
            "measurement_update_eligibility": {
                uav_id: self._measurement_update_eligibility(uav_id).astype(bool).tolist()
                for uav_id in self.uav_ids
            },
            "sensor_packets": sensor_packets,
            "collision": collision,
            "minimum_clearance": minimum_clearance,
            "topology": self.snapshot,
            "schema_version": EPISODE_SCHEMA_VERSION,
            "ekf_diagnostics": {
                uav_id: ekf.diagnostics_snapshot() for uav_id, ekf in self.ekfs.items()
            },
            "relative_range_update_events": copy.deepcopy(self.relative_range_update_events),
            "measurement_gate_events": copy.deepcopy(self.measurement_gate_events),
            "measurement_budget_enabled": bool(self.config.measurement_budget_enabled),
            "measurement_budget_limits": copy.deepcopy(self.measurement_budget_limits),
            "measurement_budget_remaining": copy.deepcopy(self.measurement_budget_remaining),
            "measurement_budget_used": copy.deepcopy(self.measurement_budget_used),
            "measurement_budget_events": copy.deepcopy(self.measurement_budget_step_events),
            # Diagnostics make it possible to distinguish a policy choosing
            # ``none`` while a candidate is eligible from a physically forced
            # ``none`` while every candidate waits for its UWB period.
            "measurement_action_diagnostics": copy.deepcopy(
                self.measurement_action_diagnostics
            ),
            "communication_applicable": bool(len(self.uav_ids) > 1),
            "single_uav_learning_mode": str(
                self.config.reward.single_uav_learning_mode
            ),
            "learning_reward_mode": self.config.reward.learning_reward_mode,
            "learning_potential": float(self._learning_potential),
            "task_complete": False,
        }

    @staticmethod
    def _obstacle_payload(obstacle: Obstacle) -> dict[str, Any]:
        """Return critic-visible obstacle geometry without exposing it to actors."""

        payload: dict[str, Any] = {
            "obstacle_id": obstacle.obstacle_id,
            "obstacle_type": obstacle.obstacle_type,
            "active": int(obstacle.active),
            "physical_collision_enabled": bool(is_physical_obstacle(obstacle)),
            "affects_los": True,
            "semantic_role": (
                "communication_occluder"
                if not is_physical_obstacle(obstacle)
                else "physical_obstacle"
            ),
            "center_m": obstacle.center_m.copy(),
            "velocity_mps": (
                obstacle.velocity_mps.copy()
                if obstacle.velocity_mps is not None
                else np.zeros(3, dtype=float)
            ),
        }
        if isinstance(obstacle, BoxObstacle):
            payload["geometry_type"] = "box"
            payload["size_m"] = obstacle.size_m.copy()
        else:
            payload["geometry_type"] = "sphere"
            payload["radius_m"] = float(obstacle.radius_m)
        return payload


__all__ = ["MessageRecord", "MultiUAVJointEnv", "RewardConfig", "SimulationConfig"]
