"""Minimal reproducible three-UAV closed-loop pilot experiment.

The existing synthetic benchmark generator intentionally remains a fixed
baseline.  This module is the learning-side pilot: it drives the same
``MultiUAVJointEnv`` with the real GAT/MAPPO policy, records episode-v1.3
artifacts, and writes separate diagnostics for attention and PPO updates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION
from ..simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from ..simulation.ekf import EKFConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.sensor_model import SensorConfig
from ..simulation.scenario import generate_scenario
from ..simulation.topology import TopologyConfig
from ..synthetic.channel_model import ChannelConfig
from ..synthetic.episode_generator import (
    _jsonable,
    _ekf_diagnostic_row,
    _obstacle_row,
    _round,
    _sha256,
    _trajectory_rows,
    _write_csv,
    _write_json,
)
from ..learning.normalization import FeatureNormalizer, MetricNormalizer, ObservationNormalizer
from .termination import classify_episode_outcome


_PHASE_OFFSETS = {
    "warmup": 0,
    "probe": 100_000,
    "smoke": 200_000,
    "train": 300_000,
    "validation": 400_000,
}

_EPISODE_FILES = (
    "ground_truth.csv",
    "imu.csv",
    "uwb_ranges.csv",
    "visual_odometry.csv",
    "gnss.csv",
    "links.csv",
    "packet_events.csv",
    "actions.csv",
    "rewards.csv",
    "obstacles.csv",
    "trajectory.csv",
    "estimates.csv",
    "ekf_diagnostics.csv",
    "graph_features.csv",
)

_MEASUREMENT_OVERHEAD_COLUMNS = (
    "episode_id",
    "scenario_id",
    "seed",
    "split",
    "step",
    "timestamp",
    "source_uav_id",
    "target_uav_id",
    "selected_non_none",
    "invalid_measurement_action",
    "measurement_transition_invalidated",
    "overhead_mode",
    "payload_size_bytes",
    "load_packets_requested",
    "load_packets_applied",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "queue_before_packets",
    "queue_with_measurement_packets",
    "queue_next_packets",
    "channel_bandwidth_hz",
    "rate_bps",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "link_state",
    "ekf_behavior_unchanged",
)

_MEASUREMENT_BUDGET_COLUMNS = (
    "episode_id",
    "scenario_id",
    "seed",
    "split",
    "step",
    "uav_id",
    "measurement_uav_id",
    "budget_limit",
    "budget_remaining_before",
    "budget_consumed",
    "budget_remaining_after",
    "budget_used",
)


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the reproducible pilot, not a formal experiment."""

    output_dir: Path
    seed: int = 2026
    uav_count: int = 3
    gnss_mode: str = "denied"
    max_steps: int = 20
    warmup_episodes: int = 2
    integration_steps: int = 5
    train_episodes: int = 10
    validation_episodes: int = 2
    device: str = "cpu"
    force: bool = False

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("the minimum closed-loop pilot is fixed to 3 UAVs")
        if self.max_steps < 1 or self.integration_steps < 1:
            raise ValueError("max_steps and integration_steps must be positive")
        if self.warmup_episodes < 1 or self.train_episodes < 1 or self.validation_episodes < 1:
            raise ValueError("warmup, train, and validation episode counts must be positive")
        if self.device != "cpu":
            raise ValueError("the reproducible pilot currently supports device='cpu' only")


def _seed_everything(seed: int) -> None:
    """Seed every RNG used by the pilot and request deterministic CPU kernels."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "PyTorch is required for run-pilot; use the project environment or "
            "C:\\Users\\lenovo\\anaconda3\\python.exe"
        ) from exc
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _episode_seed(base_seed: int, phase: str, index: int = 0) -> int:
    return int(base_seed + _PHASE_OFFSETS[phase] + 1009 * index)


def _scenario_for(config: PilotConfig, seed: int) -> dict[str, Any]:
    """Return the same seeded scene contract used by formal experiments."""

    return generate_scenario(
        seed=int(seed),
        uav_count=config.uav_count,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )


def _make_environment(
    config: PilotConfig,
    seed: int,
    *,
    reliable_probe: bool = False,
    metric_scales: Mapping[str, float] | None = None,
) -> MultiUAVJointEnv:
    channel = ChannelConfig()
    if reliable_probe:
        # The probe is a connectivity assertion, not a communication result.
        # Remove stochastic shadowing/measurement noise and make packet loss
        # negligible so a failed probe identifies an interface defect.
        channel = ChannelConfig(
            shadowing_std_los_db=0.0,
            shadowing_std_nlos_db=0.0,
            rssi_measurement_std_db=0.0,
            loss_beta0=-100.0,
        )
    motion = MotionConfig(
        dt_s=0.2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    sensor = SensorConfig(uwb_max_range_m=100.0)
    topology = TopologyConfig(
        max_comm_range_m=75.0,
        uwb_range_m=100.0,
        message_arrivals_per_window=1.0,
    )
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=config.uav_count,
            max_steps=config.max_steps,
            seed=seed,
            motion=motion,
            sensor=sensor,
            channel=channel,
            topology=topology,
            reward=RewardConfig(
                metric_scales=dict(metric_scales) if metric_scales is not None else None,
                metric_scale_source="training_warmup_p95" if metric_scales is not None else "default",
            ),
            gnss_mode=config.gnss_mode,
        )
    )


def _probe_actions(environment: MultiUAVJointEnv) -> dict[str, dict[str, Any]]:
    """Allocate equal bandwidth and select a valid incoming UWB source."""

    actions: dict[str, dict[str, Any]] = {}
    for uav_id in environment.uav_ids:
        outgoing = sorted(
            target
            for source, target in environment.snapshot.potential_edges
            if source == uav_id
        )
        incoming = sorted(
            source
            for source, target in environment.snapshot.candidate_edges
            if target == uav_id
            and environment.snapshot.uwb_measurements.get((source, uav_id), {}).get("valid", 0)
        )
        share = 1.0 / len(outgoing) if outgoing else 0.0
        actions[uav_id] = {
            "measurement_uav_id": incoming[0] if incoming else "none",
            "bandwidth": {target: share for target in outgoing},
            "delta_v": np.zeros(3, dtype=float),
        }
    return actions


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return True
    if array.dtype.kind in "OUSb":
        return True
    return bool(np.isfinite(array).all())


def _assert_observation_finite(observations: Mapping[str, Mapping[str, Any]]) -> None:
    for uav_id, observation in observations.items():
        if not _finite(observation):
            raise RuntimeError(f"non-finite actor observation for {uav_id}")


def _assert_action_feasible(environment: MultiUAVJointEnv, actions: Mapping[str, Mapping[str, Any]]) -> None:
    max_delta = environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s
    for uav_id, action in actions.items():
        bandwidth = {str(key): float(value) for key, value in dict(action.get("bandwidth", {})).items()}
        if sum(max(0.0, value) for value in bandwidth.values()) > 1.0 + 1e-6:
            raise RuntimeError(f"bandwidth action is infeasible for {uav_id}")
        delta = np.asarray(action.get("delta_v", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) > max_delta + 1e-5:
            raise RuntimeError(f"velocity action is infeasible for {uav_id}")


def _assert_attention_trace(trace: Mapping[str, Any]) -> None:
    embeddings = np.asarray(trace.get("node_embeddings", np.empty((0, 0))))
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or not np.isfinite(embeddings).all():
        raise RuntimeError("GAT trace has invalid node embeddings")
    if int(trace.get("layer_count", 0)) < 1 or not bool(trace.get("virtual_self_loops_added", False)):
        raise RuntimeError("GAT trace is missing layer or self-loop metadata")
    for layer in trace.get("attention_maps", []):
        edge_index = np.asarray(layer["edge_index"], dtype=np.int64)
        attention = np.asarray(layer["attention"], dtype=float)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise RuntimeError("GAT attention edge index has invalid shape")
        if attention.ndim != 2 or attention.shape[0] != edge_index.shape[1]:
            raise RuntimeError("GAT attention coefficients have invalid shape")
        if not np.isfinite(attention).all():
            raise RuntimeError("GAT attention coefficients are non-finite")
        for target in range(embeddings.shape[0]):
            positions = np.flatnonzero(edge_index[1] == target)
            if positions.size and not np.allclose(attention[positions].sum(axis=0), 1.0, atol=1e-5):
                raise RuntimeError("GAT attention is not normalized over incoming edges")


def _assert_graph_trace(trace: Mapping[str, Any]) -> None:
    """Validate either the historical GAT trace or the role-aware MLP trace.

    The MLP ablation deliberately has no attention map. It still exposes a
    finite embedding and an explicit fixed-slot presence mask so the episode
    schema can distinguish no graph message passing from a missing trace.
    """

    if str(trace.get("encoder_type", "gat")) in {"mlp_flat", "mlp_slots_v2"}:
        embeddings = np.asarray(trace.get("node_embeddings", np.empty((0, 0))))
        presence = np.asarray(trace.get("flat_presence_mask", []), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] < 1 or not np.isfinite(embeddings).all():
            raise RuntimeError("MLP graph trace has invalid embeddings")
        if int(trace.get("layer_count", -1)) != 0:
            raise RuntimeError("MLP graph trace must report zero GAT layers")
        if list(trace.get("attention_maps", [])):
            raise RuntimeError("MLP graph trace must not fabricate attention maps")
        if presence.ndim != 1 or presence.size < 1 or not np.isfinite(presence).all():
            raise RuntimeError("MLP graph trace has an invalid fixed-slot mask")
        if not np.isin(presence, (0.0, 1.0)).all():
            raise RuntimeError("MLP graph trace presence mask must be binary")
        return
    _assert_attention_trace(trace)


def _parameter_norm(trainer: Any) -> float:
    import torch

    with torch.no_grad():
        return float(_parameter_vector(trainer).norm().cpu())


def _parameter_vector(trainer: Any) -> Any:
    import torch

    with torch.no_grad():
        values = [parameter.detach().reshape(-1).cpu() for parameter in trainer.policy.parameters()]
        return torch.cat(values)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(payload)), ensure_ascii=False, sort_keys=True) + "\n")


class _EpisodeTraceWriter:
    """Write one policy-driven episode using the episode-v1.3 contract."""

    def __init__(
        self,
        *,
        episode_root: Path,
        episode_id: str,
        scenario_id: str,
        split: str,
        seed: int,
        phase: str,
        environment: MultiUAVJointEnv,
        normalization_fitted: bool,
        diagnostics_path: Path,
        policy_source: str | None = None,
    ) -> None:
        self.episode_root = episode_root
        self.episode_id = episode_id
        self.scenario_id = scenario_id
        self.split = split
        self.seed = int(seed)
        self.phase = phase
        self.environment = environment
        self.normalization_fitted = bool(normalization_fitted)
        self.diagnostics_path = diagnostics_path
        self.policy_source = str(policy_source or phase)
        self.rows: dict[str, list[dict[str, Any]]] = {
            filename[:-4]: []
            for filename in _EPISODE_FILES
            if filename.endswith(".csv") and filename != "trajectory.csv"
        }
        self.step_count = 0
        self.collision_steps: list[int] = []
        self.termination_reason = "running"
        self.terminated = False
        self.truncated = False
        self.action_saturation_values: list[float] = []
        self.safety_projection_count = 0
        self.boundary_projection_count = 0
        self.boundary_penalty_sum = 0.0
        self.braking_safety_projection_count = 0
        self.minimum_clearances: list[float] = []
        self.pre_projection_minimum_clearances: list[float] = []
        self.collision_types: set[str] = set()
        self.collision_objects: list[dict[str, Any]] = []
        self.pre_projection_collision_steps: list[int] = []
        self.unavoidable_dynamic_collision_steps: list[int] = []
        self.physical_collision_steps: list[int] = []
        self.clearance_violation_steps: list[int] = []
        self.ignored_nonphysical_obstacle_ids: set[str] = set()
        self.physical_obstacle_ids: set[str] = set()
        self.communication_occluder_ids: set[str] = set()
        self.measurement_overhead_rows: list[dict[str, Any]] = []
        self.measurement_budget_rows: list[dict[str, Any]] = []

    @property
    def common(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "split": self.split,
        }

    def record_step(
        self,
        *,
        step: int,
        observations_before: Mapping[str, Mapping[str, Any]],
        observations_after: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, Mapping[str, Any]],
        rewards: Mapping[str, float],
        learning_rewards: Mapping[str, float] | None = None,
        info: Mapping[str, Any],
        graph_traces: Mapping[str, Mapping[str, Any]] | None,
        policy_source: str,
        action_start: float,
        timestamp: float,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        if learning_rewards is None:
            learning_rewards = rewards
        self.step_count = max(self.step_count, int(step) + 1)
        self.boundary_penalty_sum += float(info.get("boundary_penalty_sum", 0.0))
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.termination_reason = str(info.get("termination_reason", "running"))
        if int(info.get("collision", 0)):
            self.collision_steps.append(int(step))
        if bool(info.get("physical_collision", False)):
            self.physical_collision_steps.append(int(step))
        if bool(
            info.get("clearance_violation", False)
            or info.get("protection_distance_violation", False)
        ):
            self.clearance_violation_steps.append(int(step))
        self.ignored_nonphysical_obstacle_ids.update(
            str(value) for value in info.get("ignored_nonphysical_obstacle_ids", [])
        )
        self.physical_obstacle_ids.update(
            str(value) for value in info.get("physical_obstacle_ids", [])
        )
        self.communication_occluder_ids.update(
            str(value) for value in info.get("communication_occluder_ids", [])
        )
        minimum_clearance = float(info.get("minimum_clearance", float("inf")))
        if np.isfinite(minimum_clearance):
            self.minimum_clearances.append(minimum_clearance)
        pre_clearance = float(
            info.get("pre_projection_minimum_clearance", float("inf"))
        )
        if np.isfinite(pre_clearance):
            self.pre_projection_minimum_clearances.append(pre_clearance)
        if int(info.get("pre_projection_collision", 0)):
            self.pre_projection_collision_steps.append(int(step))
        if bool(info.get("unavoidable_dynamic_collision", False)):
            self.unavoidable_dynamic_collision_steps.append(int(step))
        self.collision_types.update(str(value) for value in info.get("collision_types", []))
        for value in info.get("collision_objects", []):
            if isinstance(value, Mapping):
                item = dict(value)
                if item not in self.collision_objects:
                    self.collision_objects.append(item)
        common = self.common
        snapshot = info["topology"]
        central = info["central_state"]
        sensor_packets = info["sensor_packets"]
        for event in info.get("measurement_overhead_events", []):
            self.measurement_overhead_rows.append(
                {
                    **common,
                    "step": int(step),
                    **dict(event),
                }
            )
        for event in info.get("measurement_budget_events", []):
            self.measurement_budget_rows.append(
                {
                    **common,
                    "step": int(step),
                    **dict(event),
                }
            )
        for obstacle in self.environment.obstacles:
            self.rows["obstacles"].append(_obstacle_row(common, timestamp, obstacle))

        effective_actions = info.get("actions", actions)
        invalid_targets = {
            str(event.get("target_uav_id", ""))
            for event in info.get("invalid_measurement_action_events", [])
        }
        transition_invalidated_targets = {
            str(event.get("target_uav_id", ""))
            for event in info.get("measurement_transition_invalidated_events", [])
        }
        for uav_id, state in self.environment.states.items():
            true_position = state.position_m
            self.rows["ground_truth"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "x": _round(true_position[0]),
                    "y": _round(true_position[1]),
                    "z": _round(true_position[2]),
                    "vx": _round(state.velocity_mps[0]),
                    "vy": _round(state.velocity_mps[1]),
                    "vz": _round(state.velocity_mps[2]),
                    "roll": _round(state.attitude_rad[0]),
                    "pitch": _round(state.attitude_rad[1]),
                    "yaw": _round(state.attitude_rad[2]),
                    "ax": _round(state.acceleration_mps2[0]),
                    "ay": _round(state.acceleration_mps2[1]),
                    "az": _round(state.acceleration_mps2[2]),
                    "wx": _round(state.angular_velocity_rps[0]),
                    "wy": _round(state.angular_velocity_rps[1]),
                    "wz": _round(state.angular_velocity_rps[2]),
                }
            )
            estimate = np.asarray(central["estimates"][uav_id])
            covariance = np.asarray(central["covariances"][uav_id])
            if not np.isfinite(estimate).all() or not np.isfinite(covariance).all():
                raise RuntimeError(f"non-finite EKF output for {uav_id}")
            ekf = self.environment.ekfs[uav_id]
            self.rows["estimates"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "x": _round(estimate[0]),
                    "y": _round(estimate[1]),
                    "z": _round(estimate[2]),
                    "vx": _round(estimate[3]),
                    "vy": _round(estimate[4]),
                    "vz": _round(estimate[5]),
                    "roll": _round(estimate[6]),
                    "pitch": _round(estimate[7]),
                    "yaw": _round(estimate[8]),
                    "covariance_flat_json": json.dumps(_jsonable(covariance.reshape(-1)), ensure_ascii=False),
                    "valid_observation_count": ekf.valid_observation_count,
                    "valid_neighbor_count": ekf.valid_neighbor_count,
                    "last_neighbor_timestamp": _round(
                        ekf.last_neighbor_timestamp if ekf.last_neighbor_timestamp is not None else -1.0
                    ),
                }
            )
            self.rows["ekf_diagnostics"].append(
                _ekf_diagnostic_row(
                    common,
                    timestamp,
                    uav_id,
                    true_position,
                    estimate,
                    covariance,
                    info.get("ekf_diagnostics", {}).get(uav_id),
                    info.get("visual_odometry_quality_diagnostics", {}).get(uav_id),
                )
            )
            packets = sensor_packets[uav_id]
            imu = packets["imu"]
            self.rows["imu"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "accel_x": _round(imu["accel"][0]),
                    "accel_y": _round(imu["accel"][1]),
                    "accel_z": _round(imu["accel"][2]),
                    "gyro_x": _round(imu["gyro"][0]),
                    "gyro_y": _round(imu["gyro"][1]),
                    "gyro_z": _round(imu["gyro"][2]),
                    "bias_accel_x": _round(imu["bias_accel"][0]),
                    "bias_accel_y": _round(imu["bias_accel"][1]),
                    "bias_accel_z": _round(imu["bias_accel"][2]),
                    "bias_gyro_x": _round(imu["bias_gyro"][0]),
                    "bias_gyro_y": _round(imu["bias_gyro"][1]),
                    "bias_gyro_z": _round(imu["bias_gyro"][2]),
                    "saturation_flag": 0,
                }
            )
            vo = packets["vo"]
            vo_delta = vo["delta_position"]
            self.rows["visual_odometry"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "vo_dx": _round(vo_delta[0] if vo_delta is not None else None),
                    "vo_dy": _round(vo_delta[1] if vo_delta is not None else None),
                    "vo_dz": _round(vo_delta[2] if vo_delta is not None else None),
                    "vo_dyaw": _round(vo["delta_yaw"]),
                    "valid": vo["valid"],
                    "feature_count": int(120 if vo["valid"] else 12),
                    "tracking_quality": _round(vo["quality"]),
                }
            )
            gnss = packets["gnss"]
            gnss_position = gnss["position"]
            self.rows["gnss"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "x": _round(gnss_position[0] if gnss_position is not None else None),
                    "y": _round(gnss_position[1] if gnss_position is not None else None),
                    "z": _round(gnss_position[2] if gnss_position is not None else None),
                    "hdop": _round(gnss["std"]),
                    "fix_type": gnss["fix_type"],
                    "valid": gnss["valid"],
                    "error_scale": _round(gnss["std"]),
                    "mode": gnss["mode"],
                }
            )
            local = observations_after[uav_id]
            self.rows["graph_features"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "node_features_json": json.dumps(_jsonable(local["node_features"]), ensure_ascii=False),
                    "effective_edge_index_json": json.dumps(_jsonable(local["effective_edge_index"]), ensure_ascii=False),
                    "effective_edge_features_json": json.dumps(_jsonable(local["effective_edge_features"]), ensure_ascii=False),
                    "candidate_edge_index_json": json.dumps(_jsonable(local["candidate_edge_index"]), ensure_ascii=False),
                    "candidate_edge_features_json": json.dumps(_jsonable(local["candidate_edge_features"]), ensure_ascii=False),
                    "action_mask_json": json.dumps(
                        {
                            "measurement": _jsonable(observations_before[uav_id]["action_mask"]),
                            "bandwidth": _jsonable(observations_before[uav_id]["bandwidth_mask"]),
                        },
                        ensure_ascii=False,
                    ),
                    "potential_edge_index_json": json.dumps(_jsonable(local["potential_edge_index"]), ensure_ascii=False),
                    "potential_edge_features_json": json.dumps(_jsonable(local["potential_edge_features"]), ensure_ascii=False),
                    "bandwidth_mask_json": json.dumps(_jsonable(local["bandwidth_mask"]), ensure_ascii=False),
                    "candidate_incoming_edge_features_json": json.dumps(
                        _jsonable(local["candidate_incoming_edge_features"]), ensure_ascii=False
                    ),
                    "candidate_incoming_links_json": json.dumps(
                        _jsonable(local["candidate_incoming_links"]), ensure_ascii=False
                    ),
                    "uwb_available_json": json.dumps(_jsonable(local["uwb_available"]), ensure_ascii=False),
                    "uwb_ranges_json": json.dumps(_jsonable(local["uwb_ranges"]), ensure_ascii=False),
                    "central_state_json": json.dumps(_jsonable(central), ensure_ascii=False),
                }
            )
            raw_action = actions[uav_id]
            action = effective_actions[uav_id]
            applied_bandwidth = info["applied_bandwidth"][uav_id]
            applied_action = {**action, "bandwidth": applied_bandwidth}
            action_diagnostics = info.get("measurement_action_diagnostics", {}).get(
                uav_id, {}
            )
            if not isinstance(action_diagnostics, Mapping):
                action_diagnostics = {}
            execution = info["action_execution"][uav_id]
            self.rows["actions"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "measurement_uav_id": action["measurement_uav_id"],
                    "invalid_measurement_action": int(
                        uav_id in invalid_targets
                    ),
                    "measurement_transition_invalidated": int(
                        uav_id in transition_invalidated_targets
                    ),
                    "candidate_count_before_timing": int(
                        action_diagnostics.get("candidate_count_before_timing", 0)
                    ),
                    "eligible_candidate_count": int(
                        action_diagnostics.get("eligible_candidate_count", 0)
                    ),
                    "timing_filtered_candidate_count": int(
                        action_diagnostics.get("timing_filtered_candidate_count", 0)
                    ),
                    "none_forced_by_timing_mask": int(
                        action_diagnostics.get("none_forced_by_timing_mask", 0)
                    ),
                    "none_when_candidate_eligible": int(
                        action_diagnostics.get("none_when_candidate_eligible", 0)
                    ),
                    "budget_filtered": int(
                        action_diagnostics.get("budget_filtered", 0)
                    ),
                    "measurement_action_invalid_reason": str(
                        action_diagnostics.get("invalid_reason", "")
                    ),
                    "topology_mode": "maintain",
                    "bandwidth_ratio": _round(sum(applied_bandwidth.values())),
                    "bandwidth_allocations_json": json.dumps(_jsonable(applied_bandwidth), ensure_ascii=False, sort_keys=True),
                    "action_mask_json": json.dumps(
                        {
                            "measurement": _jsonable(observations_before[uav_id]["action_mask"]),
                            "bandwidth": _jsonable(observations_before[uav_id]["bandwidth_mask"]),
                        },
                        ensure_ascii=False,
                    ),
                    "delta_vx": _round(action["delta_v"][0]),
                    "delta_vy": _round(action["delta_v"][1]),
                    "delta_vz": _round(action["delta_v"][2]),
                    "applied_window_start": _round(action_start),
                    "applied_window_end": _round(timestamp),
                    "action_raw": json.dumps(_jsonable(raw_action), ensure_ascii=False, sort_keys=True),
                    "action_applied": json.dumps(_jsonable(applied_action), ensure_ascii=False, sort_keys=True),
                    "action_clipped": int(execution["action_clipped"]),
                    "pre_speed_mps": _round(execution.get("pre_speed_mps", 0.0)),
                    "post_speed_mps": _round(execution.get("post_speed_mps", 0.0)),
                    "speed_change_mps": _round(execution.get("speed_change_mps", 0.0)),
                    "speed_reduction_mps": _round(execution.get("speed_reduction_mps", 0.0)),
                    "delta_v_commanded_norm_mps": _round(
                        execution.get("delta_v_commanded_norm_mps", 0.0)
                    ),
                    "delta_v_applied_norm_mps": _round(
                        execution.get("delta_v_applied_norm_mps", 0.0)
                    ),
                    "commanded_longitudinal_delta_v_mps": _round(
                        execution.get("commanded_longitudinal_delta_v_mps", 0.0)
                    ),
                    "applied_longitudinal_delta_v_mps": _round(
                        execution.get("applied_longitudinal_delta_v_mps", 0.0)
                    ),
                    "commanded_braking_delta_v_mps": _round(
                        execution.get("commanded_braking_delta_v_mps", 0.0)
                    ),
                    "braking_delta_v_mps": _round(
                        execution.get("braking_delta_v_mps", 0.0)
                    ),
                    "braking_action": int(execution.get("braking_action", 0)),
                    "brake_effective": int(execution.get("brake_effective", 0)),
                    "brake_execution_loss_mps": _round(
                        execution.get("brake_execution_loss_mps", 0.0)
                    ),
                    "brake_blocked": int(execution.get("brake_blocked", 0)),
                    "speed_limit_limited": int(
                        execution.get("speed_limit_limited", 0)
                    ),
                    "safety_projected": int(execution.get("safety_projected", False)),
                    "boundary_projected": int(execution.get("boundary_projected", False)),
                    "braking_safety_projection_enabled": int(
                        execution.get("braking_safety_projection_enabled", False)
                    ),
                    "braking_safety_projected": int(
                        execution.get("braking_safety_projected", False)
                    ),
                    "braking_safety_projection_reason": str(
                        execution.get("braking_safety_projection_reason", "")
                    ),
                    "braking_safety_target_speed_mps": _round(
                        execution.get("braking_safety_target_speed_mps", 0.0)
                    ),
                    "braking_safety_predicted_speed_mps": _round(
                        execution.get("braking_safety_predicted_speed_mps", 0.0)
                    ),
                    "braking_safety_projection_delta_v_json": json.dumps(
                        _jsonable(
                            execution.get(
                                "braking_safety_projection_delta_v", (0.0, 0.0, 0.0)
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "policy_source": policy_source,
                }
            )
            self.safety_projection_count += int(
                bool(execution.get("safety_projected", False))
            )
            self.boundary_projection_count += int(
                bool(execution.get("boundary_projected", False))
            )
            self.braking_safety_projection_count += int(
                bool(execution.get("braking_safety_projected", False))
            )
            self.action_saturation_values.append(
                float(bool(execution.get("action_clipped", False)))
            )
            global_metrics = info["global_metrics"]
            covariance_position = np.asarray(covariance[:3, :3])
            learning_components = info.get("learning_reward_components", {})
            if not isinstance(learning_components, Mapping):
                learning_components = {}
            boundary_rewards = learning_components.get(
                "boundary_reward_by_uav", info.get("boundary_reward_by_uav", {})
            )
            if not isinstance(boundary_rewards, Mapping):
                boundary_rewards = {}
            learning_reward_preclip = learning_components.get(
                "learning_reward_preclip_by_uav", {}
            )
            if not isinstance(learning_reward_preclip, Mapping):
                learning_reward_preclip = {}
            self.rows["rewards"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "reward_total": _round(rewards[uav_id]),
                    "learning_reward_total": _round(learning_rewards[uav_id]),
                    "base_learning_reward": _round(
                        learning_components.get(
                            "base_learning_reward", learning_rewards[uav_id]
                        )
                    ),
                    "boundary_reward": _round(
                        boundary_rewards.get(uav_id, info.get("boundary_reward", 0.0))
                    ),
                    "boundary_projected": int(
                        bool(execution.get("boundary_projected", False))
                    ),
                    "boundary_projection_count": int(
                        info.get("boundary_projection_count", 0)
                    ),
                    "learning_reward_preclip": _round(
                        learning_reward_preclip.get(
                            uav_id, learning_rewards[uav_id]
                        )
                    ),
                    "learning_reward_mode": info.get("learning_reward_mode", "legacy"),
                    "single_uav_learning_mode": learning_components.get(
                        "single_uav_learning_mode",
                        info.get("single_uav_learning_mode", "off"),
                    ),
                    "single_uav_learning_active": int(
                        bool(learning_components.get("single_uav_learning_active", False))
                    ),
                    "communication_applicable": int(
                        bool(
                            learning_components.get(
                                "communication_applicable",
                                info.get("communication_applicable", False),
                            )
                        )
                    ),
                    "communication_reward_in_learning": int(
                        bool(
                            learning_components.get(
                                "communication_reward_in_learning", True
                            )
                        )
                    ),
                    "learning_potential": _round(info.get("learning_potential", 0.0)),
                    "learning_progress_potential": _round(
                        info.get("learning_progress_potential", info.get("learning_potential", 0.0))
                    ),
                    "learning_progress_logdet_uncertainty": _round(
                        info.get("learning_progress_logdet_uncertainty", 0.0)
                    ),
                    "learning_progress_normalized_j_com": _round(
                        info.get("learning_progress_normalized_j_com", 0.0)
                    ),
                    "learning_feedback_mode": learning_components.get(
                        "learning_feedback_mode", "v3"
                    ),
                    "v3_learning_reward": _round(
                        learning_components.get("v3_learning_reward", learning_rewards[uav_id])
                    ),
                    "position_logdet_previous": _round(
                        learning_components.get("position_logdet_previous", 0.0)
                    ),
                    "position_logdet_current": _round(
                        learning_components.get("position_logdet_current", 0.0)
                    ),
                    "loc_progress_delta": _round(
                        learning_components.get("loc_progress_delta", 0.0)
                    ),
                    "loc_progress_delta_raw": _round(
                        learning_components.get("loc_progress_delta_raw", 0.0)
                    ),
                    "loc_progress_scale": _round(
                        learning_components.get("loc_progress_scale", 0.0)
                    ),
                    "loc_progress_scale_source": learning_components.get(
                        "loc_progress_scale_source", "unknown"
                    ),
                    "loc_progress_reward": _round(
                        learning_components.get("loc_progress_reward", 0.0)
                    ),
                    "single_uav_localization_progress_delta": _round(
                        learning_components.get(
                            "single_uav_localization_progress_delta", 0.0
                        )
                    ),
                    "single_uav_localization_progress_scale": _round(
                        learning_components.get(
                            "single_uav_localization_progress_scale", 0.0
                        )
                    ),
                    "localization_progress_score": _round(
                        learning_components.get("localization_progress_score", 0.0)
                    ),
                    "single_uav_localization_component": _round(
                        learning_components.get(
                            "single_uav_localization_component", 0.0
                        )
                    ),
                    "previous_goal_distance": _round(
                        learning_components.get("previous_goal_distance", 0.0)
                    ),
                    "current_goal_distance": _round(
                        learning_components.get("current_goal_distance", 0.0)
                    ),
                    "goal_progress_delta": _round(
                        learning_components.get("goal_progress_delta", 0.0)
                    ),
                    "goal_progress_scale": _round(
                        learning_components.get("goal_progress_scale", 0.0)
                    ),
                    "goal_progress_score": _round(
                        learning_components.get("goal_progress_score", 0.0)
                    ),
                    "single_uav_goal_component": _round(
                        learning_components.get("single_uav_goal_component", 0.0)
                    ),
                    "single_uav_goal_progress_reward": _round(
                        learning_components.get(
                            "single_uav_goal_progress_reward", 0.0
                        )
                    ),
                    "single_uav_goal_bonus": _round(
                        learning_components.get("single_uav_goal_bonus", 0.0)
                    ),
                    "single_uav_goal_bonus_triggered": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_bonus_triggered", False
                            )
                        )
                    ),
                    "single_uav_goal_reached": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_reached", False
                            )
                        )
                    ),
                    "single_uav_goal_in_radius": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_in_radius", False
                            )
                        )
                    ),
                    "single_uav_goal_speed_ok": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_speed_ok", False
                            )
                        )
                    ),
                    "single_uav_goal_stable": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_stable", False
                            )
                        )
                    ),
                    "single_uav_goal_condition": int(
                        bool(
                            learning_components.get(
                                "single_uav_goal_condition", False
                            )
                        )
                    ),
                    "single_uav_stable_steps": int(
                        learning_components.get("single_uav_stable_steps", 0)
                    ),
                    "single_uav_stable_hold_steps": int(
                        learning_components.get("single_uav_stable_hold_steps", 1)
                    ),
                    "single_uav_goal_threshold_m": _round(
                        learning_components.get(
                            "single_uav_goal_threshold_m", 0.0
                        )
                    ),
                    "single_uav_speed_mps": _round(
                        learning_components.get("single_uav_speed_mps", 0.0)
                    ),
                    "single_uav_goal_speed_threshold_mps": _round(
                        learning_components.get(
                            "single_uav_goal_speed_threshold_mps", 0.0
                        )
                    ),
                    "single_uav_near_goal_radius_m": _round(
                        learning_components.get(
                            "single_uav_near_goal_radius_m", 0.0
                        )
                    ),
                    "single_uav_speed_control_mode": learning_components.get(
                        "single_uav_speed_control_mode", "fixed_near_goal"
                    ),
                    "single_uav_braking_distance_speed_profile": learning_components.get(
                        "single_uav_braking_distance_speed_profile",
                        info.get("single_uav_braking_distance_speed_profile", "legacy"),
                    ),
                    "single_uav_target_speed_far_mps": _round(
                        learning_components.get(
                            "single_uav_target_speed_far_mps", 0.0
                        )
                    ),
                    "single_uav_target_speed_mps": _round(
                        learning_components.get(
                            "single_uav_target_speed_mps", 0.0
                        )
                    ),
                    "single_uav_stopping_distance_m": _round(
                        learning_components.get(
                            "single_uav_stopping_distance_m", 0.0
                        )
                    ),
                    "single_uav_brake_margin_m": _round(
                        learning_components.get("single_uav_brake_margin_m", 0.0)
                    ),
                    "single_uav_braking_acceleration_mps2": _round(
                        learning_components.get(
                            "single_uav_braking_acceleration_mps2", 0.0
                        )
                    ),
                    "single_uav_braking_margin_m": _round(
                        learning_components.get("single_uav_braking_margin_m", 0.0)
                    ),
                    "single_uav_speed_excess_mps": _round(
                        learning_components.get(
                            "single_uav_speed_excess_mps", 0.0
                        )
                    ),
                    "single_uav_speed_deficit_mps": _round(
                        learning_components.get(
                            "single_uav_speed_deficit_mps", 0.0
                        )
                    ),
                    "single_uav_speed_deficit_cost": _round(
                        learning_components.get(
                            "single_uav_speed_deficit_cost", 0.0
                        )
                    ),
                    "single_uav_speed_deficit_penalty_weight": _round(
                        learning_components.get(
                            "single_uav_speed_deficit_penalty_weight", 0.0
                        )
                    ),
                    "single_uav_near_goal_speed_cost": _round(
                        learning_components.get(
                            "single_uav_near_goal_speed_cost", 0.0
                        )
                    ),
                    "single_uav_near_goal_speed_penalty_weight": _round(
                        learning_components.get(
                            "single_uav_near_goal_speed_penalty_weight", 0.0
                        )
                    ),
                    "single_uav_localization_cost": _round(
                        learning_components.get(
                            "single_uav_localization_cost", 0.0
                        )
                    ),
                    "single_uav_localization_cost_weight": _round(
                        learning_components.get(
                            "single_uav_localization_cost_weight", 0.0
                        )
                    ),
                    "single_uav_learning_energy_cost": _round(
                        learning_components.get("single_uav_learning_energy_cost", 0.0)
                    ),
                    "single_uav_learning_collision_cost": _round(
                        learning_components.get(
                            "single_uav_learning_collision_cost", 0.0
                        )
                    ),
                    "single_uav_goal_progress_signal": _round(
                        learning_components.get(
                            "single_uav_goal_progress_signal", 0.0
                        )
                    ),
                    "single_uav_goal_progress_reward_scale_m": _round(
                        learning_components.get(
                            "single_uav_goal_progress_reward_scale_m", 0.0
                        )
                    ),
                    "single_uav_goal_hold_reward": _round(
                        learning_components.get("single_uav_goal_hold_reward", 0.0)
                    ),
                    "single_uav_goal_direction_signal": _round(
                        learning_components.get(
                            "single_uav_goal_direction_signal", 0.0
                        )
                    ),
                    "single_uav_goal_direction_coef": _round(
                        learning_components.get(
                            "single_uav_goal_direction_coef", 0.0
                        )
                    ),
                    "single_uav_goal_direction_reward": _round(
                        learning_components.get(
                            "single_uav_goal_direction_reward", 0.0
                        )
                    ),
                    "single_uav_braking_need_signal": _round(
                        learning_components.get(
                            "single_uav_braking_need_signal", 0.0
                        )
                    ),
                    "single_uav_braking_action_signal": _round(
                        learning_components.get(
                            "single_uav_braking_action_signal", 0.0
                        )
                    ),
                    "single_uav_effective_braking_signal": _round(
                        learning_components.get(
                            "single_uav_effective_braking_signal", 0.0
                        )
                    ),
                    "single_uav_braking_action_reward": _round(
                        learning_components.get(
                            "single_uav_braking_action_reward", 0.0
                        )
                    ),
                    "single_uav_braking_action_reward_coef": _round(
                        learning_components.get(
                            "single_uav_braking_action_reward_coef", 0.0
                        )
                    ),
                    "single_uav_target_velocity_action_error": _round(
                        learning_components.get(
                            "single_uav_target_velocity_action_error", 0.0
                        )
                    ),
                    "single_uav_target_velocity_action_reward": _round(
                        learning_components.get(
                            "single_uav_target_velocity_action_reward", 0.0
                        )
                    ),
                    "single_uav_target_velocity_action_reward_coef": _round(
                        learning_components.get(
                            "single_uav_target_velocity_action_reward_coef", 0.0
                        )
                    ),
                    "single_uav_terminal_braking_radius_m": _round(
                        learning_components.get(
                            "single_uav_terminal_braking_radius_m", 0.0
                        )
                    ),
                    "single_uav_terminal_braking_active": int(
                        bool(
                            learning_components.get(
                                "single_uav_terminal_braking_active", False
                            )
                        )
                    ),
                    "single_uav_terminal_braking_need_signal": _round(
                        learning_components.get(
                            "single_uav_terminal_braking_need_signal", 0.0
                        )
                    ),
                    "single_uav_terminal_speed_reduction_signal": _round(
                        learning_components.get(
                            "single_uav_terminal_speed_reduction_signal", 0.0
                        )
                    ),
                    "single_uav_terminal_forward_accel_signal": _round(
                        learning_components.get(
                            "single_uav_terminal_forward_accel_signal", 0.0
                        )
                    ),
                    "single_uav_terminal_braking_reward": _round(
                        learning_components.get(
                            "single_uav_terminal_braking_reward", 0.0
                        )
                    ),
                    "single_uav_terminal_braking_reward_coef": _round(
                        learning_components.get(
                            "single_uav_terminal_braking_reward_coef", 0.0
                        )
                    ),
                    "single_uav_terminal_acceleration_penalty": _round(
                        learning_components.get(
                            "single_uav_terminal_acceleration_penalty", 0.0
                        )
                    ),
                    "single_uav_terminal_acceleration_penalty_coef": _round(
                        learning_components.get(
                            "single_uav_terminal_acceleration_penalty_coef", 0.0
                        )
                    ),
                    "single_uav_pre_speed_excess_mps": _round(
                        learning_components.get(
                            "single_uav_pre_speed_excess_mps", 0.0
                        )
                    ),
                    "single_uav_post_speed_excess_mps": _round(
                        learning_components.get(
                            "single_uav_post_speed_excess_mps", 0.0
                        )
                    ),
                    "single_uav_braking_progress_signal": _round(
                        learning_components.get(
                            "single_uav_braking_progress_signal", 0.0
                        )
                    ),
                    "single_uav_braking_progress_reward": _round(
                        learning_components.get(
                            "single_uav_braking_progress_reward", 0.0
                        )
                    ),
                    "single_uav_braking_progress_reward_coef": _round(
                        learning_components.get(
                            "single_uav_braking_progress_reward_coef", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_active": int(
                        bool(
                            learning_components.get(
                                "single_uav_terminal_radial_progress_active", False
                            )
                        )
                    ),
                    "single_uav_terminal_radial_progress_m": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_m", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_speed_excess_mps": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_speed_excess_mps", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_speed_gate": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_speed_gate", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_signal": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_signal", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_reward": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_reward", 0.0
                        )
                    ),
                    "single_uav_terminal_radial_progress_reward_coef": _round(
                        learning_components.get(
                            "single_uav_terminal_radial_progress_reward_coef", 0.0
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_active": int(
                        bool(
                            learning_components.get(
                                "single_uav_terminal_target_velocity_vector_active",
                                False,
                            )
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_error_pre_mps": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_error_pre_mps",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_error_post_mps": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_error_post_mps",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_error_reduction_mps": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_error_reduction_mps",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_signal": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_signal", 0.0
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_reward": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_reward", 0.0
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_reward_coef": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_reward_coef",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_radial_gate_enabled": int(
                        bool(
                            learning_components.get(
                                "single_uav_terminal_target_velocity_vector_radial_gate_enabled",
                                False,
                            )
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_activation_distance_m": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_activation_distance_m",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_post_radial_velocity_mps": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_post_radial_velocity_mps",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_radial_gate": _round(
                        learning_components.get(
                            "single_uav_terminal_target_velocity_vector_radial_gate",
                            0.0,
                        )
                    ),
                    "single_uav_terminal_target_velocity_vector_mps": json.dumps(
                        _jsonable(
                            learning_components.get(
                                "single_uav_terminal_target_velocity_vector_mps",
                                [0.0, 0.0, 0.0],
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "single_uav_learning_core": _round(
                        learning_components.get("single_uav_learning_core", 0.0)
                    ),
                    "single_uav_learning_reward_preclip": _round(
                        learning_components.get(
                            "single_uav_learning_reward_preclip", 0.0
                        )
                    ),
                    "single_uav_learning_reward": _round(
                        learning_components.get("single_uav_learning_reward", 0.0)
                    ),
                    "actual_measurement_count": int(
                        learning_components.get("actual_measurement_count", 0)
                    ),
                    "actual_measurement_bytes": int(
                        learning_components.get("actual_measurement_bytes", 0)
                    ),
                    "actual_measurement_rate": _round(
                        learning_components.get("actual_measurement_rate", 0.0)
                    ),
                    "actual_measurement_cost_target": _round(
                        learning_components.get("actual_measurement_cost_target", 0.0)
                    ),
                    "actual_measurement_load_excess": _round(
                        learning_components.get("actual_measurement_load_excess", 0.0)
                    ),
                    "actual_measurement_load_reward": _round(
                        learning_components.get("actual_measurement_load_reward", 0.0)
                    ),
                    "accepted_update_feedback_reward": _round(
                        learning_components.get(
                            "accepted_update_feedback_reward", 0.0
                        )
                    ),
                    "accepted_update_feedback_coef": _round(
                        learning_components.get(
                            "accepted_update_feedback_coef", 0.0
                        )
                    ),
                    "accepted_update_feedback_delay_steps": int(
                        learning_components.get(
                            "accepted_update_feedback_delay_steps", 0
                        )
                    ),
                    "accepted_update_feedback_covariance_source": str(
                        learning_components.get(
                            "accepted_update_feedback_covariance_source",
                            "not_applicable",
                        )
                    ),
                    "accepted_update_feedback_records_json": json.dumps(
                        _jsonable(
                            learning_components.get(
                                "accepted_update_feedback_records", []
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "released_update_feedback_records_json": json.dumps(
                        _jsonable(
                            learning_components.get(
                                "released_update_feedback_records", []
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "pending_update_feedback_count": int(
                        learning_components.get(
                            "pending_update_feedback_count", 0
                        )
                    ),
                    "feedback_reward": _round(
                        learning_components.get("feedback_reward", 0.0)
                    ),
                    "reward_localization": _round(-float(np.trace(covariance_position))),
                    "reward_packet_loss": _round(-global_metrics["packet_loss"]),
                    "reward_aoi": _round(-global_metrics["aoi"]),
                    "reward_delay": _round(-global_metrics["delay"]),
                    "reward_connectivity": _round(global_metrics["connectivity"]),
                    "reward_collision": _round(-self.environment.config.reward.collision_penalty * info["collision"]),
                    "reward_energy": _round(-info["reward_components"][uav_id]["energy_penalty"]),
                    "joint_objective": _round(info["joint_objective"]),
                    "normalized_uncertainty": _round(global_metrics["uncertainty_normalized"]),
                    "normalized_packet_loss": _round(global_metrics["packet_loss_normalized"]),
                    "normalized_aoi": _round(global_metrics["aoi_normalized"]),
                    "normalized_delay": _round(global_metrics["delay_normalized"]),
                    "normalized_connectivity": _round(global_metrics["connectivity_normalized"]),
                    "collision": info["collision"],
                    "task_complete": int(bool(info.get("task_complete", False))),
                }
            )

        for (source, target), metrics in snapshot.metrics.items():
            key = (source, target)
            self.rows["links"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "src_uav_id": source,
                    "dst_uav_id": target,
                    "distance": _round(metrics.distance_m),
                    "los": metrics.los,
                    "tx_power_dbm": _round(metrics.tx_power_dbm),
                    "rx_power_dbm": _round(metrics.rx_power_dbm),
                    "noise_power_dbm": _round(metrics.noise_power_dbm),
                    "rssi": _round(metrics.rssi_dbm),
                    "snr": _round(metrics.snr_db),
                    "channel_bandwidth_hz": _round(metrics.channel_bandwidth_hz),
                    "rate_bps": _round(metrics.rate_bps),
                    "bandwidth_kbps": _round(metrics.bandwidth_kbps),
                    "packet_loss": _round(metrics.packet_loss),
                    "raw_delay": _round(metrics.raw_delay_s),
                    "delay": _round(metrics.delay_s),
                    "aoi": _round(metrics.aoi_s),
                    "queue_length": _round(metrics.queue_length_packets),
                    "potential_edge": int(key in snapshot.potential_edges),
                    "candidate_edge": metrics.candidate_edge,
                    # ``effective_edge`` is the graph edge usable by EKF/GAT:
                    # packet success alone is not enough before discrete
                    # arrival.
                    "effective_edge": int(key in snapshot.effective_edges),
                    "uwb_available": int(key in snapshot.uwb_edges),
                    "link_state": metrics.link_state,
                }
            )
            if key in snapshot.uwb_measurements:
                uwb = snapshot.uwb_measurements[key]
                self.rows["uwb_ranges"].append(
                    {
                        **common,
                        "timestamp": timestamp,
                        "src_uav_id": source,
                        "dst_uav_id": target,
                        "true_range": _round(uwb["true_range"]),
                        "measured_range": _round(uwb["measured_range"]),
                        "los": uwb["los"],
                        "nlos_reason": uwb["nlos_reason"],
                        "measurement_std": _round(uwb["measurement_std"]),
                        "valid": uwb["valid"],
                    }
                )
            self.rows["packet_events"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "packet_id": f"{self.episode_id}_{step:06d}_{source}_{target}",
                    "src_uav_id": source,
                    "dst_uav_id": target,
                    "send_timestamp": timestamp,
                    "receive_timestamp": _round(
                        timestamp
                        + np.ceil(metrics.delay_s / self.environment.config.motion.dt_s)
                        * self.environment.config.motion.dt_s
                    ) if metrics.packet_success else "",
                    "size_bytes": self.environment.config.channel.packet_size_bytes,
                    "success": metrics.packet_success,
                    "drop_reason": metrics.drop_reason,
                    "delay": _round(metrics.delay_s),
                    "aoi_after": _round(metrics.aoi_s),
                }
            )

        for (source, target), uwb in snapshot.uwb_measurements.items():
            if (source, target) in snapshot.metrics:
                continue
            self.rows["uwb_ranges"].append(
                {
                    **common,
                    "timestamp": timestamp,
                    "src_uav_id": source,
                    "dst_uav_id": target,
                    "true_range": _round(uwb["true_range"]),
                    "measured_range": _round(uwb["measured_range"]),
                    "los": uwb["los"],
                    "nlos_reason": uwb["nlos_reason"],
                    "measurement_std": _round(uwb["measurement_std"]),
                    "valid": uwb["valid"],
                }
            )

        if graph_traces:
            for uav_id, trace in graph_traces.items():
                _append_jsonl(
                    self.diagnostics_path,
                    {
                        "episode_id": self.episode_id,
                        "scenario_id": self.scenario_id,
                        "seed": self.seed,
                        "split": self.split,
                        "phase": self.phase,
                        "timestamp": timestamp,
                        "uav_id": uav_id,
                        "trace": trace,
                    },
                )

    def finalize(self) -> dict[str, Any]:
        self.episode_root.mkdir(parents=True, exist_ok=True)
        file_rows = {
            "ground_truth.csv": self.rows["ground_truth"],
            "imu.csv": self.rows["imu"],
            "uwb_ranges.csv": self.rows["uwb_ranges"],
            "visual_odometry.csv": self.rows["visual_odometry"],
            "gnss.csv": self.rows["gnss"],
            "links.csv": self.rows["links"],
            "packet_events.csv": self.rows["packet_events"],
            "actions.csv": self.rows["actions"],
            "rewards.csv": self.rows["rewards"],
            "obstacles.csv": self.rows["obstacles"],
            "trajectory.csv": _trajectory_rows(
                self.environment,
                self.rows["ground_truth"],
                self.rows["actions"],
                self.rows["links"],
            ),
            "estimates.csv": self.rows["estimates"],
            "ekf_diagnostics.csv": self.rows["ekf_diagnostics"],
            "graph_features.csv": self.rows["graph_features"],
        }
        if self.environment.config.measurement_overhead_enabled:
            file_rows["measurement_overhead.csv"] = self.measurement_overhead_rows
        if self.environment.config.measurement_budget_enabled:
            file_rows["measurement_budget.csv"] = self.measurement_budget_rows
        for filename, values in file_rows.items():
            columns = (
                _MEASUREMENT_OVERHEAD_COLUMNS
                if filename == "measurement_overhead.csv"
                else _MEASUREMENT_BUDGET_COLUMNS
                if filename == "measurement_budget.csv"
                else EPISODE_FILE_SPECS[filename][0]
            )
            _write_csv(self.episode_root / filename, columns, values)
        reward_config = asdict(self.environment.config.reward)
        reward_config["weights"] = dict(self.environment.config.reward.weights)
        reward_config["lambda_coll"] = self.environment.config.reward.lambda_coll
        reward_config["lambda_energy"] = self.environment.config.reward.lambda_energy
        _write_json(
            self.episode_root / "scenario.json",
            _jsonable({
                **self.common,
                "engine": "python_sim",
                "engine_status": "pure_python_action_driven_environment",
                "phase": self.phase,
                "policy_source": self.policy_source,
                "uav_count": len(self.environment.uav_ids),
                "duration_s": self.step_count * self.environment.config.motion.dt_s,
                "rate_execution": "decision_rate_baseline",
                "substep_simulation": False,
                "action_semantics": "action_at_t_applies_over_[t,t+decision_dt)_and_returns_observation_at_t+1",
                "motion_action_contract": self.environment.config.motion.action_contract(),
                "coordinate_frame": "ENU",
                "units": {"position": "m", "time": "s", "angles": "rad"},
                "normalization_fitted_on_training_observations": self.normalization_fitted,
                "measurement_overhead_enabled": bool(self.environment.config.measurement_overhead_enabled),
                "measurement_overhead_packet_size_bytes": 512,
                "measurement_budget_enabled": bool(self.environment.config.measurement_budget_enabled),
                "measurement_budget_limits": {
                    str(key): int(value)
                    for key, value in self.environment.measurement_budget_limits.items()
                },
                "calibration_status": "synthetic_configured; no_external_dataset; pilot_only",
                "termination_reason": self.termination_reason,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "collision_steps": list(self.collision_steps),
                "physical_collision_steps": list(self.physical_collision_steps),
                "clearance_violation_steps": list(self.clearance_violation_steps),
                "outcome_class": classify_episode_outcome(
                    {
                        "termination_reason": self.termination_reason,
                        "terminated": self.terminated,
                        "truncated": self.truncated,
                        "collision_occurred": bool(self.collision_steps),
                        "window_count": self.step_count,
                    },
                    max_steps=int(self.environment.config.max_steps),
                ),
                "max_steps": int(self.environment.config.max_steps),
                "minimum_clearance": (
                    float(min(self.minimum_clearances))
                    if self.minimum_clearances
                    else float("nan")
                ),
                "pre_projection_minimum_clearance": (
                    float(min(self.pre_projection_minimum_clearances))
                    if self.pre_projection_minimum_clearances
                    else float("nan")
                ),
                "collision_types": sorted(self.collision_types),
                "collision_objects": list(self.collision_objects),
                "physical_obstacle_ids": sorted(self.physical_obstacle_ids),
                "communication_occluder_ids": sorted(self.communication_occluder_ids),
                "ignored_nonphysical_obstacle_ids": sorted(
                    self.ignored_nonphysical_obstacle_ids
                ),
                "pre_projection_collision_steps": list(self.pre_projection_collision_steps),
                "unavoidable_dynamic_collision_steps": list(
                    self.unavoidable_dynamic_collision_steps
                ),
                "safety_execution_mode": self.environment.config.safety_execution_mode,
                "braking_safety_projection_count": int(
                    self.braking_safety_projection_count
                ),
                "boundary_events": int(self.boundary_projection_count),
                "boundary_penalty_sum": float(self.boundary_penalty_sum),
                "reward_config": reward_config,
                "scenario_generation": _jsonable(self.environment.scenario_metadata),
                "goals": {
                    uav_id: np.asarray(goal, dtype=float).tolist()
                    for uav_id, goal in self.environment.goals.items()
                },
            }),
        )
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "phase": self.phase,
            "uav_count": len(self.environment.uav_ids),
            "path": str(self.episode_root),
            "files": {filename: _sha256(self.episode_root / filename) for filename in file_rows},
            "steps": self.step_count,
            "window_count": self.step_count,
            "max_steps": int(self.environment.config.max_steps),
            "termination_reason": self.termination_reason,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "collision_occurred": bool(self.collision_steps),
            "collision_step": self.collision_steps[0] if self.collision_steps else -1,
            "physical_collision_steps": list(self.physical_collision_steps),
            "clearance_violation_steps": list(self.clearance_violation_steps),
            "outcome_class": classify_episode_outcome(
                {
                    "termination_reason": self.termination_reason,
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                    "collision_occurred": bool(self.collision_steps),
                    "window_count": self.step_count,
                },
                max_steps=int(self.environment.config.max_steps),
            ),
            "minimum_clearance": float(min(self.minimum_clearances)) if self.minimum_clearances else float("nan"),
            "pre_projection_minimum_clearance": (
                float(min(self.pre_projection_minimum_clearances))
                if self.pre_projection_minimum_clearances
                else float("nan")
            ),
            "collision_types": sorted(self.collision_types),
            "collision_objects": list(self.collision_objects),
            "physical_obstacle_ids": sorted(self.physical_obstacle_ids),
            "communication_occluder_ids": sorted(self.communication_occluder_ids),
            "ignored_nonphysical_obstacle_ids": sorted(
                self.ignored_nonphysical_obstacle_ids
            ),
            "pre_projection_collision_steps": list(self.pre_projection_collision_steps),
            "unavoidable_dynamic_collision_steps": list(self.unavoidable_dynamic_collision_steps),
            "safety_execution_mode": self.environment.config.safety_execution_mode,
            "mean_action_saturation": float(np.mean(self.action_saturation_values)) if self.action_saturation_values else 0.0,
            "max_action_saturation": float(np.max(self.action_saturation_values)) if self.action_saturation_values else 0.0,
            "safety_projection_count": int(self.safety_projection_count),
            "boundary_projection_count": int(self.boundary_projection_count),
            "boundary_events": int(self.boundary_projection_count),
            "boundary_penalty_sum": float(self.boundary_penalty_sum),
            "braking_safety_projection_count": int(
                self.braking_safety_projection_count
            ),
        }


def _run_probe(config: PilotConfig, output_root: Path) -> dict[str, Any]:
    """Run two identical physical probes and compare their canonical traces."""

    def one_run(seed: int) -> tuple[dict[str, Any], str]:
        _seed_everything(seed)
        environment = _make_environment(config, seed, reliable_probe=True)
        observations, _ = environment.reset(seed=seed, scenario_config=_scenario_for(config, seed))
        records: list[dict[str, Any]] = []
        effective_seen = 0
        neighbor_updates = 0
        for step in range(config.integration_steps):
            actions = _probe_actions(environment)
            _assert_observation_finite(observations)
            _assert_action_feasible(environment, actions)
            next_observations, rewards, terminated, truncated, info = environment.step(actions)
            effective_seen += len(info["central_state"]["effective_edges"])
            neighbor_updates += sum(ekf.valid_neighbor_count for ekf in environment.ekfs.values())
            records.append(
                {
                    "step": step,
                    "positions": {uid: environment.states[uid].position_m.tolist() for uid in environment.uav_ids},
                    "estimates": {uid: environment.ekfs[uid].state.tolist() for uid in environment.uav_ids},
                    "covariance_diag": {
                        uid: np.diag(environment.ekfs[uid].covariance).tolist() for uid in environment.uav_ids
                    },
                    "rewards": rewards,
                    "effective_edges": sorted(info["central_state"]["effective_edges"]),
                    "valid_neighbor_counts": {
                        uid: environment.ekfs[uid].valid_neighbor_count for uid in environment.uav_ids
                    },
                }
            )
            observations = next_observations
            if terminated or truncated:
                break
        canonical = json.dumps(_jsonable(records), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        summary = {
            "seed": seed,
            "steps": len(records),
            "effective_edges_seen": effective_seen,
            "neighbor_updates": neighbor_updates,
            "finite": True,
            "trace_sha256": digest,
        }
        return summary, digest

    first, first_hash = one_run(_episode_seed(config.seed, "probe"))
    second, second_hash = one_run(_episode_seed(config.seed, "probe"))
    result = {
        "phase": "probe",
        "deterministic_match": first_hash == second_hash,
        "first": first,
        "second": second,
        "effective_edges_seen": first["effective_edges_seen"],
        "neighbor_updates": first["neighbor_updates"],
    }
    if not result["deterministic_match"]:
        raise RuntimeError("same-seed physical probe is not reproducible")
    if first["effective_edges_seen"] < 1:
        raise RuntimeError("probe did not produce an effective communication edge")
    if first["neighbor_updates"] < 1:
        raise RuntimeError("probe did not produce a valid neighbor EKF update")
    _write_json(output_root / "checks" / "probe_summary.json", result)
    _write_json(output_root / "checks" / "determinism.json", {"probe_match": True, "sha256": first_hash})
    return result


def _run_policy_episode(
    *,
    config: PilotConfig,
    trainer: Any,
    phase: str,
    split: str,
    serial: int,
    seed: int,
    update_policy: bool,
    output_root: Path,
    diagnostics_path: Path,
    update_index: int | None = None,
    metric_scales: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=_scenario_for(config, seed))
    writer = _EpisodeTraceWriter(
        episode_root=output_root / split / f"episode_{serial:06d}",
        episode_id=f"episode_{serial:06d}",
        scenario_id=f"scenario_{serial:06d}",
        split=split,
        seed=seed,
        phase=phase,
        environment=environment,
        normalization_fitted=trainer.normalizer.fitted and trainer.central_normalizer.fitted,
        diagnostics_path=diagnostics_path,
    )
    total_rewards: list[float] = []
    effective_edges_seen = 0
    max_neighbor_updates = 0
    for step in range(config.max_steps):
        _assert_observation_finite(observations)
        decision = trainer.act(
            observations,
            info["central_state"],
            deterministic=not update_policy,
        )
        _assert_action_feasible(environment, decision.actions)
        for trace in decision.graph_traces.values():
            _assert_graph_trace(trace)
        next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
        if update_policy:
            trainer.add_transition(
                decision,
                rewards,
                next_info["central_state"],
                done=bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        total_rewards.extend(float(value) for value in rewards.values())
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
            info=next_info,
            graph_traces=decision.graph_traces,
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
    update_stats: dict[str, float] | None = None
    if update_policy:
        update_stats = trainer.update()
        if not all(np.isfinite(float(value)) for value in update_stats.values()):
            raise RuntimeError(f"MAPPO update returned non-finite statistics: {update_stats}")
        mean_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
        _append_jsonl(
            output_root / "diagnostics" / "training_metrics.jsonl",
            {
                "phase": phase,
                "episode": writer.episode_id,
                "update": update_index,
                "episode_id": writer.episode_id,
                "seed": seed,
                "reward": mean_reward,
                "mean_reward": mean_reward,
                "effective_edges_seen": effective_edges_seen,
                "max_neighbor_updates": max_neighbor_updates,
                "collision_occurred": episode_summary.get("collision_occurred", False),
                "collision_step": episode_summary.get("collision_step", -1),
                "termination_reason": episode_summary.get("termination_reason", "running"),
                "mean_action_saturation": episode_summary.get("mean_action_saturation", 0.0),
                "max_action_saturation": episode_summary.get("max_action_saturation", 0.0),
                **update_stats,
            },
        )
    episode_summary.update(
        {
            "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "effective_edges_seen": effective_edges_seen,
            "max_neighbor_updates": max_neighbor_updates,
            "update": update_stats,
        }
    )
    return episode_summary, update_stats


def _fit_warmup_normalizer(
    config: PilotConfig,
) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, float], dict[str, Any]]:
    from ..learning.mappo import central_state_to_vector

    samples: list[dict[str, dict[str, Any]]] = []
    metric_rows: list[Mapping[str, Any]] = []
    central_vectors: list[np.ndarray] = []
    steps = 0
    for index in range(config.warmup_episodes):
        seed = _episode_seed(config.seed, "warmup", index)
        environment = _make_environment(config, seed)
        observations, info = environment.reset(seed=seed, scenario_config=_scenario_for(config, seed))
        samples.append(copy.deepcopy(observations))
        central_vectors.append(
            central_state_to_vector(
                info.get("central_state"),
                uav_count=config.uav_count,
                node_feature_dim=37,
                edge_feature_dim=11,
                max_obstacles=8,
            )
        )
        for _ in range(config.max_steps):
            actions = _probe_actions(environment)
            next_observations, _rewards, terminated, truncated, next_info = environment.step(actions)
            metric_rows.append(dict(next_info.get("global_metrics", {})))
            central_vectors.append(
                central_state_to_vector(
                    next_info.get("central_state"),
                    uav_count=config.uav_count,
                    node_feature_dim=37,
                    edge_feature_dim=11,
                    max_obstacles=8,
                )
            )
            samples.append(copy.deepcopy(next_observations))
            steps += 1
            observations = next_observations
            if terminated or truncated:
                break
    normalizer = ObservationNormalizer().fit(samples)
    central_normalizer = FeatureNormalizer().fit(np.asarray(central_vectors, dtype=np.float64))
    metric_normalizer = MetricNormalizer().fit(metric_rows)
    metric_scales = dict(metric_normalizer.scales or {})
    return normalizer, central_normalizer, metric_scales, {
        "episodes": config.warmup_episodes,
        "steps": steps,
        "observation_snapshots": len(samples),
        "fitted_keys": sorted(normalizer.normalizers),
        "central_feature_dim": central_normalizer.feature_dim,
        "central_normalization_fitted": central_normalizer.fitted,
        "metric_scales": metric_scales,
        "metric_scale_source": "training_warmup_p95",
    }


def _copy_calibration_files(output_root: Path) -> None:
    target = output_root / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    _write_json(
        target / "localization_noise.json",
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "calibration_status": "synthetic_configured; no_external_dataset",
            "external_dataset_required": False,
            "parameter_sources": ["Python SensorConfig", "Python EKFConfig", "Python scenario generator"],
            "parameters": asdict(SensorConfig()),
            "ekf_parameters": asdict(EKFConfig()),
            "calibration_policy": "controlled parameter sweeps, repeated seeds, and ablation checks",
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
            "parameters": asdict(ChannelConfig()),
            "calibration_policy": "controlled parameter sweeps, repeated seeds, and ablation checks",
        },
    )


def _prepare_output(config: PilotConfig) -> Path:
    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()) and not config.force:
        raise FileExistsError(
            f"pilot output is not empty: {root}; choose another path or pass --force"
        )
    if config.force and root.exists():
        # Only remove files and directories owned by this pilot contract.
        managed_directories = ("train", "validation", "calibration", "diagnostics", "checks")
        managed_files = ("manifest.json", "normalization_stats.json")
        for name in managed_directories:
            target = root / name
            if target.exists():
                shutil.rmtree(target)
        for name in managed_files:
            target = root / name
            if target.exists():
                target.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_pilot(config: PilotConfig) -> dict[str, Any]:
    """Run the complete two-stage pilot and return a JSON-safe summary."""

    _seed_everything(config.seed)
    output_root = _prepare_output(config)
    _copy_calibration_files(output_root)
    normalizer, central_normalizer, metric_scales, warmup_summary = _fit_warmup_normalizer(config)
    normalization_payload = normalizer.state_dict()
    normalization_payload["central_normalizer"] = central_normalizer.state_dict()
    normalization_payload["metric_scales"] = metric_scales
    normalization_payload["metric_scale_source"] = "training_warmup_p95"
    _write_json(output_root / "normalization_stats.json", normalization_payload)

    probe_summary = _run_probe(config, output_root)

    # Import learning modules lazily so non-learning CLI commands remain
    # usable with the bundled NumPy-only runtime.
    from ..learning.mappo import MAPPOTrainer

    diagnostics_path = output_root / "diagnostics" / "attention_maps.jsonl"
    smoke_environment = _make_environment(
        config,
        _episode_seed(config.seed, "smoke"),
        metric_scales=metric_scales,
    )
    smoke_trainer = MAPPOTrainer.from_environment(
        smoke_environment,
        seed=_episode_seed(config.seed, "smoke"),
        normalizer=normalizer,
        central_normalizer=central_normalizer,
        gat_hidden_dim=16,
        gat_output_dim=16,
        gat_heads=1,
        gat_layers=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        max_obstacles=8,
        policy_epochs=2,
        minibatch_size=32,
        device=config.device,
    )
    smoke_summary, _ = _run_policy_episode(
        config=config,
        trainer=smoke_trainer,
        phase="smoke",
        split="validation",
        serial=900001,
        seed=_episode_seed(config.seed, "smoke"),
        update_policy=False,
        output_root=output_root,
        diagnostics_path=diagnostics_path,
        update_index=None,
        metric_scales=metric_scales,
    )

    train_environment = _make_environment(
        config,
        _episode_seed(config.seed, "train"),
        metric_scales=metric_scales,
    )
    trainer = MAPPOTrainer.from_environment(
        train_environment,
        seed=_episode_seed(config.seed, "train"),
        normalizer=normalizer,
        central_normalizer=central_normalizer,
        gat_hidden_dim=16,
        gat_output_dim=16,
        gat_heads=1,
        gat_layers=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        max_obstacles=8,
        policy_epochs=2,
        minibatch_size=32,
        device=config.device,
    )
    initial_parameter_vector = _parameter_vector(trainer)
    initial_parameter_norm = float(initial_parameter_vector.norm().cpu())
    episodes: list[dict[str, Any]] = []
    for index in range(config.train_episodes):
        episode_summary, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            phase="train",
            split="train",
            serial=index + 1,
            seed=_episode_seed(config.seed, "train", index),
            update_policy=True,
            output_root=output_root,
            diagnostics_path=diagnostics_path,
            update_index=index + 1,
            metric_scales=metric_scales,
        )
        episodes.append(episode_summary)
    final_parameter_vector = _parameter_vector(trainer)
    final_parameter_norm = float(final_parameter_vector.norm().cpu())
    parameter_delta_l2 = float((final_parameter_vector - initial_parameter_vector).norm().cpu())
    parameter_delta = {
        "initial_l2": initial_parameter_norm,
        "final_l2": final_parameter_norm,
        "delta_l2": parameter_delta_l2,
        "updates": config.train_episodes,
    }
    _write_json(output_root / "diagnostics" / "parameter_delta.json", parameter_delta)
    if parameter_delta["delta_l2"] <= 0.0:
        raise RuntimeError("MAPPO policy parameters did not change after training")

    validation_episodes: list[dict[str, Any]] = []
    for index in range(config.validation_episodes):
        episode_summary, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            phase="validation",
            split="validation",
            serial=910000 + index + 1,
            seed=_episode_seed(config.seed, "validation", index),
            update_policy=False,
            output_root=output_root,
            diagnostics_path=diagnostics_path,
            update_index=None,
            metric_scales=metric_scales,
        )
        validation_episodes.append(episode_summary)

    manifest = {
        "dataset_id": "python_multi_uav_mappo_pilot",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "engine_status": "pure_python_action_driven_environment_with_mappo",
        "generation_status": "pilot_complete; Python configuration recorded; formal repeated-seed experiments remain",
        "policy_source": "mappo",
        "config": _jsonable({**asdict(config), "output_dir": str(config.output_dir)}),
        "coordinate_frame": "ENU",
        "units": {"position": "m", "time": "s", "angles": "rad"},
        "calibration": {
            "localization": "calibration/localization_noise.json",
            "communication": "calibration/communication_model.json",
        },
        "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
        "episodes": episodes + [smoke_summary] + validation_episodes,
        "notes": [
            "This directory is a reproducibility pilot, not a formal result dataset.",
            "The existing python_sim_benchmark baseline is not overwritten or reused as MAPPO output.",
            "GAT attention and MAPPO update diagnostics are stored under diagnostics/.",
            "All calibration files are generated from the declared Python simulation configuration; no external dataset is used.",
        ],
    }
    _write_json(output_root / "manifest.json", manifest)
    summary = {
        "output_dir": str(output_root),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "warmup": warmup_summary,
        "probe": probe_summary,
        "smoke": smoke_summary,
        "train_episodes": len(episodes),
        "validation_episodes": len(validation_episodes),
        "parameter_delta": parameter_delta,
        "manifest": str(output_root / "manifest.json"),
    }
    _write_json(output_root / "checks" / "integration_summary.json", summary)
    return summary


__all__ = ["PilotConfig", "run_pilot"]
