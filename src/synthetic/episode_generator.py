"""Episode export for the pure-Python multi-UAV simulation environment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..schema import (
    ACTIONS_COLUMNS,
    EPISODE_FILE_SPECS,
    EPISODE_LINK_COLUMNS,
    EPISODE_OBSTACLE_COLUMNS,
    EPISODE_SCHEMA_VERSION,
    EPISODE_TRAJECTORY_COLUMNS,
    ESTIMATE_COLUMNS,
    GRAPH_FEATURE_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    GNSS_COLUMNS,
    IMU_COLUMNS,
    PACKET_EVENT_COLUMNS,
    REWARDS_COLUMNS,
    UWB_COLUMNS,
    VISUAL_ODOMETRY_COLUMNS,
)
from ..simulation.environment import MultiUAVJointEnv, SimulationConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.obstacle_model import BoxObstacle, SphereObstacle
from ..simulation.sensor_model import SensorConfig
from ..simulation.topology import TopologyConfig
from .channel_model import ChannelConfig, compute_link_metrics


@dataclass(frozen=True)
class EpisodeBenchmarkConfig:
    """Configuration for deterministic pure-Python episodes."""

    output_dir: Path
    episodes: int = 3
    duration_s: float = 10.0
    seed: int = 2026
    physics_hz: int = 100
    uwb_hz: int = 10
    vision_hz: int = 10
    gnss_hz: int = 5
    link_hz: int = 10
    decision_hz: int = 5
    area_size: float = 200.0
    max_comm_range: float = 75.0
    uav_counts: tuple[int, ...] = (3, 5, 7)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.duration_s <= 0:
            raise ValueError("episodes and duration_s must be positive")
        for name, rate in (
            ("uwb_hz", self.uwb_hz),
            ("vision_hz", self.vision_hz),
            ("gnss_hz", self.gnss_hz),
            ("link_hz", self.link_hz),
            ("decision_hz", self.decision_hz),
        ):
            if rate <= 0 or self.physics_hz % rate != 0:
                raise ValueError(f"{name} must be a positive divisor of physics_hz")

    def split_for(self, episode_index: int) -> str:
        train_end = max(1, math.floor(self.episodes * 0.6))
        validation_end = max(train_end + 1, math.floor(self.episodes * 0.8))
        if episode_index < train_end:
            return "train"
        if episode_index < validation_end:
            return "validation"
        return "test"

    def uav_count_for(self, episode_index: int, split: str) -> int:
        if split == "test":
            return max(self.uav_counts)
        return self.uav_counts[episode_index % max(1, len(self.uav_counts) - 1)]


def _write_csv(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _round(value: Any, digits: int = 9) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return int(value) if isinstance(value, (bool, np.bool_)) else int(value)
    return value


def _ekf_diagnostic_row(
    common: Mapping[str, Any],
    timestamp: float,
    uav_id: str,
    true_position: np.ndarray,
    estimate: np.ndarray,
    covariance: np.ndarray,
    diagnostics: Mapping[str, Any] | None,
    vo_quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the v1.3 EKF consistency row without exposing truth to policy code."""

    truth = np.asarray(true_position, dtype=float).reshape(3)
    estimate_vector = np.asarray(estimate, dtype=float).reshape(-1)
    covariance_matrix = np.asarray(covariance, dtype=float).reshape(15, 15)
    position_error = truth - estimate_vector[:3]
    position_covariance = 0.5 * (
        covariance_matrix[:3, :3] + covariance_matrix[:3, :3].T
    )
    position_nees = float(
        position_error @ np.linalg.pinv(position_covariance) @ position_error
    )
    payload = dict(diagnostics or {})
    quality_payload = dict(vo_quality or {})
    return {
        **common,
        "timestamp": timestamp,
        "uav_id": uav_id,
        "nis_vo": _round(payload.get("nis_vo", -1.0)),
        "nis_uwb_mean": _round(payload.get("nis_uwb_mean", -1.0)),
        "nis_uwb_max": _round(payload.get("nis_uwb_max", -1.0)),
        "nis_gnss": _round(payload.get("nis_gnss", -1.0)),
        "vo_update_attempted": int(payload.get("vo_update_attempted", 0)),
        "vo_update_accepted": int(payload.get("vo_update_accepted", 0)),
        "vo_update_rejected": int(payload.get("vo_update_rejected", 0)),
        "uwb_update_attempted": int(payload.get("uwb_update_attempted", 0)),
        "uwb_update_accepted": int(payload.get("uwb_update_accepted", 0)),
        "uwb_update_rejected": int(payload.get("uwb_update_rejected", 0)),
        "gnss_update_attempted": int(payload.get("gnss_update_attempted", 0)),
        "gnss_update_accepted": int(payload.get("gnss_update_accepted", 0)),
        "gnss_update_rejected": int(payload.get("gnss_update_rejected", 0)),
        "rejected_observation_count": int(payload.get("rejected_observation_count", 0)),
        "position_error_m": _round(np.linalg.norm(position_error)),
        "position_nees": _round(position_nees),
        "position_covariance_trace_m2": _round(np.trace(position_covariance)),
        "vo_measurement_covariance_trace_m2": _round(
            payload.get("vo_measurement_covariance_trace", -1.0)
        ),
        "vo_measurement_covariance_scale": _round(
            payload.get("vo_measurement_covariance_scale", 1.0)
        ),
        "vo_quality_tracking": _round(quality_payload.get("tracking_quality", -1.0)),
        "vo_quality_covariance_scale": _round(
            quality_payload.get("covariance_scale", 1.0)
        ),
        "vo_quality_covariance_valid": int(
            quality_payload.get("valid", 0)
        ),
        "vo_quality_covariance_bin": int(
            quality_payload.get("covariance_bin", -1)
        ),
        "nis_vo_accepted": _round(payload.get("nis_vo_accepted", -1.0)),
        "nis_vo_rejected": _round(payload.get("nis_vo_rejected", -1.0)),
        "nis_uwb_accepted_mean": _round(payload.get("nis_uwb_accepted_mean", -1.0)),
        "nis_uwb_rejected_mean": _round(payload.get("nis_uwb_rejected_mean", -1.0)),
        "nis_gnss_accepted": _round(payload.get("nis_gnss_accepted", -1.0)),
        "nis_gnss_rejected": _round(payload.get("nis_gnss_rejected", -1.0)),
    }


def _profile(index: int, split: str) -> dict[str, Any]:
    return {
        "obstacle_density": ("low", "medium", "high")[index % 3],
        "gnss_mode": ("nominal", "degraded", "denied", "intermittent")[index % 4],
        "communication_mode": ("good", "moderate", "poor")[(index + 1) % 3],
        "split": split,
    }


def _make_environment(config: EpisodeBenchmarkConfig, uav_count: int, seed: int, profile: dict[str, Any]) -> MultiUAVJointEnv:
    mode = profile["communication_mode"]
    bandwidth_scale = {"good": 1.0, "moderate": 0.5, "poor": 0.25}[mode]
    motion = MotionConfig(
        dt_s=1.0 / config.decision_hz,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(config.area_size, config.area_size, 60.0),
    )
    sensor = SensorConfig(uwb_max_range_m=min(100.0, config.max_comm_range * 1.35))
    topology = TopologyConfig(
        max_comm_range_m=config.max_comm_range,
        uwb_range_m=sensor.uwb_max_range_m,
        message_arrivals_per_window=bandwidth_scale,
    )
    env_config = SimulationConfig(
        uav_count=uav_count,
        max_steps=max(1, round(config.duration_s * config.decision_hz)),
        seed=seed,
        motion=motion,
        sensor=sensor,
        channel=config.channel,
        topology=topology,
        gnss_mode=profile["gnss_mode"],
    )
    return MultiUAVJointEnv(env_config)


def _baseline_actions(env: MultiUAVJointEnv) -> dict[str, dict[str, Any]]:
    actions: dict[str, dict[str, Any]] = {}
    for uav_id, state in env.states.items():
        outgoing_candidates = sorted(
            target for source, target in env.snapshot.candidate_edges if source == uav_id
        )
        incoming_candidates = sorted(
            source
            for source, target in env.snapshot.candidate_edges
            if target == uav_id
            and env.snapshot.uwb_measurements.get((source, uav_id), {}).get("valid", 0)
        )
        bandwidth = {
            target: 1.0 / len(outgoing_candidates)
            for target in outgoing_candidates
        } if outgoing_candidates else {}
        delta = env.goals[uav_id] - state.position_m
        norm = float(np.linalg.norm(delta))
        desired = delta / norm * min(3.0, norm) if norm > 1e-9 else np.zeros(3)
        delta_v = desired - state.velocity_mps
        selected_measurement = incoming_candidates[0] if incoming_candidates else "none"
        actions[uav_id] = {
            "measurement_uav_id": selected_measurement,
            "bandwidth": bandwidth,
            "delta_v": delta_v,
        }
    return actions


def _obstacle_row(common: dict[str, Any], timestamp: float, obstacle: Any) -> dict[str, Any]:
    velocity = obstacle.velocity_mps if obstacle.velocity_mps is not None else np.zeros(3)
    if isinstance(obstacle, BoxObstacle):
        minimum, maximum = obstacle.minimum_m, obstacle.maximum_m
        return {
            **common,
            "timestamp": timestamp,
            "obstacle_id": obstacle.obstacle_id,
            "obstacle_type": obstacle.obstacle_type,
            "geometry_type": "box",
            "x": _round(obstacle.center_m[0]),
            "y": _round(obstacle.center_m[1]),
            "z": _round(obstacle.center_m[2]),
            "vx": _round(velocity[0]),
            "vy": _round(velocity[1]),
            "vz": _round(velocity[2]),
            "x_min": _round(minimum[0]),
            "x_max": _round(maximum[0]),
            "y_min": _round(minimum[1]),
            "y_max": _round(maximum[1]),
            "z_min": _round(minimum[2]),
            "z_max": _round(maximum[2]),
            # A scalar radius is retained for the flat schema even though the
            # exact box geometry is represented by x_min/x_max/... fields.
            "radius": _round(0.5 * float(np.linalg.norm(obstacle.size_m))),
            "active": int(obstacle.active),
            "physical_collision_enabled": int(getattr(obstacle, "physical_collision_enabled", True)),
        }
    return {
        **common,
        "timestamp": timestamp,
        "obstacle_id": obstacle.obstacle_id,
        "obstacle_type": obstacle.obstacle_type,
        "geometry_type": "sphere",
        "x": _round(obstacle.center_m[0]),
        "y": _round(obstacle.center_m[1]),
        "z": _round(obstacle.center_m[2]),
        "vx": _round(velocity[0]),
        "vy": _round(velocity[1]),
        "vz": _round(velocity[2]),
        "x_min": "",
        "x_max": "",
        "y_min": "",
        "y_max": "",
        "z_min": "",
        "z_max": "",
        "radius": _round(obstacle.radius_m),
        "active": int(obstacle.active),
        "physical_collision_enabled": int(getattr(obstacle, "physical_collision_enabled", True)),
    }


def _build_episode(config: EpisodeBenchmarkConfig, episode_index: int, root: Path, allow_overwrite: bool) -> dict[str, Any]:
    split = config.split_for(episode_index)
    episode_id = f"episode_{episode_index + 1:06d}"
    scenario_id = f"scenario_{episode_index + 1:06d}"
    seed = config.seed + episode_index * 1009
    profile = _profile(episode_index, split)
    uav_count = config.uav_count_for(episode_index, split)
    episode_root = root / split / episode_id
    if episode_root.exists() and any(episode_root.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"episode directory is not empty: {episode_root}")
    episode_root.mkdir(parents=True, exist_ok=True)
    env = _make_environment(config, uav_count, seed, profile)
    observations, reset_info = env.reset(seed=seed)
    common = {"episode_id": episode_id, "scenario_id": scenario_id, "seed": seed, "split": split}
    rows: dict[str, list[dict[str, Any]]] = {
        name[:-4]: [] for name in EPISODE_FILE_SPECS if name.endswith(".csv")
    }
    max_steps = env.config.max_steps
    for step in range(max_steps):
        actions = _baseline_actions(env)
        action_start = step * env.config.motion.dt_s
        observations, rewards, terminated, truncated, info = env.step(actions)
        timestamp = round((step + 1) * env.config.motion.dt_s, 6)
        snapshot = info["topology"]
        central = info["central_state"]
        sensor_packets = info["sensor_packets"]
        for obstacle in env.obstacles:
            rows["obstacles"].append(_obstacle_row(common, timestamp, obstacle))

        for uav_id, state in env.states.items():
            state_id = uav_id
            true_position = state.position_m
            rows["ground_truth"].append({
                **common,
                "timestamp": timestamp,
                "uav_id": state_id,
                "x": _round(true_position[0]), "y": _round(true_position[1]), "z": _round(true_position[2]),
                "vx": _round(state.velocity_mps[0]), "vy": _round(state.velocity_mps[1]), "vz": _round(state.velocity_mps[2]),
                "roll": _round(state.attitude_rad[0]), "pitch": _round(state.attitude_rad[1]), "yaw": _round(state.attitude_rad[2]),
                "ax": _round(state.acceleration_mps2[0]), "ay": _round(state.acceleration_mps2[1]), "az": _round(state.acceleration_mps2[2]),
                "wx": _round(state.angular_velocity_rps[0]), "wy": _round(state.angular_velocity_rps[1]), "wz": _round(state.angular_velocity_rps[2]),
            })
            estimate = np.asarray(central["estimates"][uav_id])
            covariance = np.asarray(central["covariances"][uav_id])
            ekf = env.ekfs[uav_id]
            rows["estimates"].append({
                **common,
                "timestamp": timestamp,
                "uav_id": uav_id,
                "x": _round(estimate[0]), "y": _round(estimate[1]), "z": _round(estimate[2]),
                "vx": _round(estimate[3]), "vy": _round(estimate[4]), "vz": _round(estimate[5]),
                "roll": _round(estimate[6]), "pitch": _round(estimate[7]), "yaw": _round(estimate[8]),
                "covariance_flat_json": json.dumps(_jsonable(covariance.reshape(-1)), ensure_ascii=False),
                "valid_observation_count": ekf.valid_observation_count,
                "valid_neighbor_count": ekf.valid_neighbor_count,
                "last_neighbor_timestamp": _round(ekf.last_neighbor_timestamp if ekf.last_neighbor_timestamp is not None else -1.0),
            })
            rows["ekf_diagnostics"].append(
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
            imu = sensor_packets[uav_id]["imu"]
            rows["imu"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "accel_x": _round(imu["accel"][0]), "accel_y": _round(imu["accel"][1]), "accel_z": _round(imu["accel"][2]),
                "gyro_x": _round(imu["gyro"][0]), "gyro_y": _round(imu["gyro"][1]), "gyro_z": _round(imu["gyro"][2]),
                "bias_accel_x": _round(imu["bias_accel"][0]), "bias_accel_y": _round(imu["bias_accel"][1]), "bias_accel_z": _round(imu["bias_accel"][2]),
                "bias_gyro_x": _round(imu["bias_gyro"][0]), "bias_gyro_y": _round(imu["bias_gyro"][1]), "bias_gyro_z": _round(imu["bias_gyro"][2]),
                "saturation_flag": 0,
            })
            vo = sensor_packets[uav_id]["vo"]
            vo_delta = vo["delta_position"]
            rows["visual_odometry"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "vo_dx": _round(vo_delta[0] if vo_delta is not None else None),
                "vo_dy": _round(vo_delta[1] if vo_delta is not None else None),
                "vo_dz": _round(vo_delta[2] if vo_delta is not None else None),
                "vo_dyaw": _round(vo["delta_yaw"]), "valid": vo["valid"],
                "feature_count": int(120 if vo["valid"] else 12), "tracking_quality": _round(vo["quality"]),
            })
            gnss = sensor_packets[uav_id]["gnss"]
            gnss_position = gnss["position"]
            rows["gnss"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "x": _round(gnss_position[0] if gnss_position is not None else None),
                "y": _round(gnss_position[1] if gnss_position is not None else None),
                "z": _round(gnss_position[2] if gnss_position is not None else None),
                "hdop": _round(gnss["std"]), "fix_type": gnss["fix_type"], "valid": gnss["valid"],
                "error_scale": _round(gnss["std"]), "mode": gnss["mode"],
            })
            local = observations[uav_id]
            rows["graph_features"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "node_features_json": json.dumps(_jsonable(local["node_features"]), ensure_ascii=False),
                "effective_edge_index_json": json.dumps(_jsonable(local["effective_edge_index"]), ensure_ascii=False),
                "effective_edge_features_json": json.dumps(_jsonable(local["effective_edge_features"]), ensure_ascii=False),
                "candidate_edge_index_json": json.dumps(_jsonable(local["candidate_edge_index"]), ensure_ascii=False),
                "candidate_edge_features_json": json.dumps(_jsonable(local["candidate_edge_features"]), ensure_ascii=False),
                "action_mask_json": json.dumps({
                    "measurement": _jsonable(local["action_mask"]),
                    "bandwidth": _jsonable(local["bandwidth_mask"]),
                }, ensure_ascii=False),
                "potential_edge_index_json": json.dumps(
                    _jsonable(local["potential_edge_index"]),
                    ensure_ascii=False,
                ),
                "potential_edge_features_json": json.dumps(
                    _jsonable(local["potential_edge_features"]),
                    ensure_ascii=False,
                ),
                "bandwidth_mask_json": json.dumps(
                    _jsonable(local["bandwidth_mask"]),
                    ensure_ascii=False,
                ),
                "candidate_incoming_edge_features_json": json.dumps(
                    _jsonable(local["candidate_incoming_edge_features"]),
                    ensure_ascii=False,
                ),
                "candidate_incoming_links_json": json.dumps(
                    _jsonable(local["candidate_incoming_links"]),
                    ensure_ascii=False,
                ),
                "uwb_available_json": json.dumps(
                    _jsonable(local["uwb_available"]),
                    ensure_ascii=False,
                ),
                "uwb_ranges_json": json.dumps(
                    _jsonable(local["uwb_ranges"]),
                    ensure_ascii=False,
                ),
                "central_state_json": json.dumps(
                    _jsonable(central),
                    ensure_ascii=False,
                ),
            })

            action = actions[uav_id]
            action_json = json.dumps(_jsonable(action), ensure_ascii=False, sort_keys=True)
            applied_action = {**action, "bandwidth": info["applied_bandwidth"][uav_id]}
            applied_action_json = json.dumps(_jsonable(applied_action), ensure_ascii=False, sort_keys=True)
            rows["actions"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "measurement_uav_id": action["measurement_uav_id"],
                "topology_mode": "maintain", "bandwidth_ratio": _round(sum(info["applied_bandwidth"][uav_id].values())),
                "bandwidth_allocations_json": json.dumps(_jsonable(info["applied_bandwidth"][uav_id]), ensure_ascii=False, sort_keys=True),
                "action_mask_json": json.dumps({
                    "measurement": _jsonable(observations[uav_id]["action_mask"]),
                    "bandwidth": _jsonable(observations[uav_id]["bandwidth_mask"]),
                }, ensure_ascii=False),
                "delta_vx": _round(action["delta_v"][0]), "delta_vy": _round(action["delta_v"][1]), "delta_vz": _round(action["delta_v"][2]),
                "applied_window_start": _round(action_start), "applied_window_end": _round(timestamp),
                "action_raw": action_json, "action_applied": applied_action_json,
                "action_clipped": int(info["action_execution"][uav_id]["action_clipped"]), "policy_source": "python_baseline",
            })
            global_metrics = info["global_metrics"]
            reward = rewards[uav_id]
            objective = info["joint_objective"]
            rows["rewards"].append({
                **common, "timestamp": timestamp, "uav_id": uav_id,
                "reward_total": _round(reward),
                "reward_localization": _round(-float(np.trace(covariance[:3, :3]))),
                "reward_packet_loss": _round(-global_metrics["packet_loss"]),
                "reward_aoi": _round(-global_metrics["aoi"]), "reward_delay": _round(-global_metrics["delay"]),
                "reward_connectivity": _round(global_metrics["connectivity"]),
                "reward_collision": _round(-env.config.reward.collision_penalty * info["collision"]),
                "reward_energy": _round(-info["reward_components"][uav_id]["energy_penalty"]),
                "joint_objective": _round(objective),
                "normalized_uncertainty": _round(global_metrics["uncertainty_normalized"]),
                "normalized_packet_loss": _round(global_metrics["packet_loss_normalized"]),
                "normalized_aoi": _round(global_metrics["aoi_normalized"]),
                "normalized_delay": _round(global_metrics["delay_normalized"]),
                "normalized_connectivity": _round(global_metrics["connectivity_normalized"]),
                "collision": info["collision"], "task_complete": 0,
            })

        for (source, target), metrics in snapshot.metrics.items():
            key = (source, target)
            rows["links"].append({
                **common, "timestamp": timestamp, "src_uav_id": source, "dst_uav_id": target,
                "distance": _round(float(np.linalg.norm(env.states[source].position_m - env.states[target].position_m))),
                "los": metrics.los, "tx_power_dbm": _round(metrics.tx_power_dbm), "rx_power_dbm": _round(metrics.rx_power_dbm),
                "noise_power_dbm": _round(metrics.noise_power_dbm), "rssi": _round(metrics.rssi_dbm), "snr": _round(metrics.snr_db),
                "channel_bandwidth_hz": _round(metrics.channel_bandwidth_hz), "rate_bps": _round(metrics.rate_bps),
                "bandwidth_kbps": _round(metrics.bandwidth_kbps), "packet_loss": _round(metrics.packet_loss),
                "raw_delay": _round(metrics.raw_delay_s),
                "delay": _round(metrics.delay_s), "aoi": _round(metrics.aoi_s), "queue_length": _round(metrics.queue_length_packets),
                "potential_edge": int(key in snapshot.potential_edges), "candidate_edge": metrics.candidate_edge,
                "effective_edge": int(key in snapshot.effective_edges), "uwb_available": int(key in snapshot.uwb_edges), "link_state": metrics.link_state,
            })
            uwb = snapshot.uwb_measurements[key]
            rows["uwb_ranges"].append({
                **common, "timestamp": timestamp, "src_uav_id": source, "dst_uav_id": target,
                "true_range": _round(uwb["true_range"]), "measured_range": _round(uwb["measured_range"]),
                "los": uwb["los"], "nlos_reason": uwb["nlos_reason"], "measurement_std": _round(uwb["measurement_std"]), "valid": uwb["valid"],
            })
            rows["packet_events"].append({
                **common, "timestamp": timestamp, "packet_id": f"{episode_id}_{step:06d}_{source}_{target}",
                "src_uav_id": source, "dst_uav_id": target, "send_timestamp": timestamp,
                "receive_timestamp": _round(
                    timestamp + np.ceil(metrics.delay_s / env.config.motion.dt_s) * env.config.motion.dt_s
                ) if metrics.packet_success else "",
                "size_bytes": config.channel.packet_size_bytes, "success": metrics.packet_success,
                "drop_reason": metrics.drop_reason, "delay": _round(metrics.delay_s), "aoi_after": _round(metrics.aoi_s),
            })
        # UWB can remain physically available outside the radio communication
        # range, so retain those measurements even when no links.csv row
        # exists for the pair.
        for (source, target), uwb in snapshot.uwb_measurements.items():
            if (source, target) in snapshot.metrics:
                continue
            rows["uwb_ranges"].append({
                **common, "timestamp": timestamp, "src_uav_id": source, "dst_uav_id": target,
                "true_range": _round(uwb["true_range"]), "measured_range": _round(uwb["measured_range"]),
                "los": uwb["los"], "nlos_reason": uwb["nlos_reason"],
                "measurement_std": _round(uwb["measurement_std"]), "valid": uwb["valid"],
            })
        if terminated or truncated:
            break

    scenario = {
        **common,
        "engine": "python_sim",
        "engine_status": "pure_python_action_driven_environment",
        "rate_execution": "decision_rate_baseline",
        "substep_simulation": False,
        "action_semantics": "action_at_t_applies_over_[t,t+decision_dt)_and_returns_observation_at_t+1",
        "uav_count": uav_count,
        "duration_s": config.duration_s,
        "rates_hz": {"decision": config.decision_hz, "links": config.link_hz, "uwb": config.uwb_hz, "vision": config.vision_hz, "gnss": config.gnss_hz},
        "coordinate_frame": "ENU",
        "units": {"position": "m", "time": "s", "angles": "rad"},
        "profile": profile,
        "channel_model": {"name": "log_distance_los_nlos_effective_link", "calibration_status": "synthetic_configured; no_external_dataset"},
        "calibration_status": "synthetic Python configuration; no external dataset required",
        "termination_reason": info.get("termination_reason", "running"),
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
        "task_complete": False,
    }
    file_rows = {
        "ground_truth.csv": rows["ground_truth"], "imu.csv": rows["imu"], "uwb_ranges.csv": rows["uwb_ranges"],
        "visual_odometry.csv": rows["visual_odometry"], "gnss.csv": rows["gnss"], "links.csv": rows["links"],
        "packet_events.csv": rows["packet_events"], "actions.csv": rows["actions"], "rewards.csv": rows["rewards"],
        "obstacles.csv": rows["obstacles"], "trajectory.csv": _trajectory_rows(env, rows["ground_truth"], rows["actions"], rows["links"]),
        "estimates.csv": rows["estimates"], "graph_features.csv": rows["graph_features"],
        "ekf_diagnostics.csv": rows["ekf_diagnostics"],
    }
    for filename, values in file_rows.items():
        columns = EPISODE_FILE_SPECS[filename][0]
        _write_csv(episode_root / filename, columns, values)
    _write_json(episode_root / "scenario.json", scenario)
    return {
        "episode_id": episode_id, "scenario_id": scenario_id, "split": split, "uav_count": uav_count, "path": str(episode_root),
        "files": {filename: _sha256(episode_root / filename) for filename in file_rows},
    }


def _trajectory_rows(
    env: MultiUAVJointEnv,
    ground_truth_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create compatibility trajectory rows from true states and link summaries."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ground_truth_rows:
        grouped[(row["timestamp"], row["uav_id"])] = row
    action_map = {(row["timestamp"], row["uav_id"]): row for row in action_rows}
    link_values: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in link_rows:
        link_values.setdefault((row["timestamp"], row["src_uav_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for (timestamp, uav_id), truth in grouped.items():
        action = action_map.get((timestamp, uav_id), {})
        links = link_values.get((timestamp, uav_id), [])
        effective = [row for row in links if int(row.get("effective_edge", 0))]
        # trajectory.csv is a flat compatibility view. Its RSSI/SNR columns
        # are numeric schema fields, so an empty outgoing edge set must use
        # explicit finite sentinels rather than an empty string.
        mean = lambda field, default=0.0: (
            sum(float(row[field]) for row in links) / len(links)
            if links
            else default
        )
        result.append({
            "episode_id": truth["episode_id"], "scenario_id": truth["scenario_id"], "seed": truth["seed"], "timestamp": timestamp, "split": truth["split"], "uav_id": uav_id,
            "x": truth["x"], "y": truth["y"], "z": truth["z"], "vx": truth["vx"], "vy": truth["vy"], "vz": truth["vz"],
            "roll": truth["roll"], "pitch": truth["pitch"], "yaw": truth["yaw"], "accel_x": truth["ax"], "accel_y": truth["ay"], "accel_z": truth["az"],
            "gyro_x": truth["wx"], "gyro_y": truth["wy"], "gyro_z": truth["wz"], "uwb_range": min((float(row["distance"]) for row in effective), default=""),
            "neighbor_ids": ";".join(row["dst_uav_id"] for row in effective) or "none",
            "rssi": mean("rssi", env.config.channel.min_rssi_dbm),
            "snr": mean("snr", env.config.channel.min_snr_db),
            "packet_loss": mean("packet_loss", 1.0), "delay": mean("delay", env.config.channel.max_delay_s), "aoi": mean("aoi", env.config.channel.aoi_cap_s),
            "gnss_mode": env.config.gnss_mode, "obstacle_state": "scene_geometry", "action_vx": action.get("delta_vx", 0.0), "action_vy": action.get("delta_vy", 0.0), "action_vz": action.get("delta_vz", 0.0),
            "ground_truth_x": truth["x"], "ground_truth_y": truth["y"], "ground_truth_z": truth["z"],
        })
    return result


def preset_config(output_dir: Path, preset: str, seed: int = 2026) -> EpisodeBenchmarkConfig:
    presets = {
        "smoke": {"episodes": 3, "duration_s": 2.0},
        "pilot": {"episodes": 30, "duration_s": 30.0},
        "thesis": {"episodes": 420, "duration_s": 60.0},
    }
    if preset not in presets:
        raise ValueError(f"unknown preset {preset!r}; choose smoke, pilot, or thesis")
    return EpisodeBenchmarkConfig(output_dir=output_dir, seed=seed, **presets[preset])


def generate_episode_benchmark(config: EpisodeBenchmarkConfig, allow_overwrite: bool = False) -> dict[str, Any]:
    root = Path(config.output_dir)
    protected = [root / "manifest.json", root / "train", root / "validation", root / "test"]
    if not allow_overwrite and any(path.exists() for path in protected):
        raise FileExistsError(f"hierarchical benchmark already exists at {root}; use --force to replace episodes")
    root.mkdir(parents=True, exist_ok=True)
    calibration_dir = root / "calibration"
    _write_json(calibration_dir / "localization_noise.json", {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "verified": False,
        "parameter_sources": ["Python SensorConfig", "Python scenario generator"],
        "calibration_status": "synthetic_configured; no_external_dataset",
        "external_dataset_required": False,
        "parameters": {
            "imu": "SensorConfig IMU bias and white-noise parameters",
            "uwb": "SensorConfig LOS/NLOS range-error parameters",
            "gnss": "Python nominal/degraded/denied mode configuration",
        },
        "calibration_policy": "controlled parameter sweeps, repeated seeds, and ablation checks",
    })
    _write_json(calibration_dir / "communication_model.json", {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "verified": False,
        "parameter_sources": ["Python ChannelConfig", "Python queue and packet model"],
        "calibration_status": "synthetic_configured; no_external_dataset",
        "external_dataset_required": False,
        "model": {
            "name": "log_distance_los_nlos_effective_link_with_queue_and_aoi",
            "scope": "statistical link model; no OFDM, MIMO, beamforming, or implicit retransmission",
            "equations": {
                "path_loss": "PL0 + 10*n_l*log10(max(d,d0)/d0)",
                "received_power": "Ptx - path_loss - shadowing - obstruction_loss",
                "snr": "Prx - (N0 + 10*log10(b) + F_nf)",
                "rate": "eta_phy*b*log2(1 + 10**(SNR/10))",
                "packet_loss": "sigmoid(beta0 - beta1*SNR + beta2*NLOS + beta3*queue)",
                "raw_delay": "distance/c_prop + packet_bits/rate + queue*packet_bits/rate when rate>0; max_delay when unavailable",
                "delay": "min(raw_delay, max_delay)",
                "aoi": "delay on packet arrival; increment until arrival or loss, capped by aoi_cap",
            },
            "units": {
                "power": "dBm",
                "rssi_snr": "dB",
                "bandwidth": "Hz",
                "rate": "bit/s",
                "delay_aoi": "s",
                "queue": "packets",
                "packet_loss": "[0,1]",
            },
            "parameters": asdict(config.channel),
        },
        "fitting_notes": "Set beta, path-loss, shadowing, and queue/delay parameters in the Python configuration and evaluate them with sensitivity and repeated-seed experiments.",
    })
    episodes = [
        _build_episode(config, episode_index, root, allow_overwrite)
        for episode_index in range(config.episodes)
    ]
    manifest = {
        "dataset_id": "python_multi_uav_joint_benchmark",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "engine_status": "pure_python_action_driven_environment",
        "generation_status": "schema_complete; Python configuration recorded",
        "config": {**asdict(config), "output_dir": str(root), "uav_counts": list(config.uav_counts)},
        "coordinate_frame": "ENU",
        "units": {"position": "m", "time": "s", "angles": "rad"},
        "calibration": {"localization": "calibration/localization_noise.json", "communication": "calibration/communication_model.json"},
        "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
        "episodes": episodes,
        "notes": [
            "The benchmark is generated by a pure Python action-driven environment.",
            "The current benchmark is a decision-rate baseline; configured physics and sensor rates are metadata until asynchronous substeps are implemented.",
            "All calibration files are generated from the Python configuration; report parameter sweeps and repeated seeds with formal results.",
            "MAPPO training is intentionally outside this environment package; the reset/step contract is stable for later trainers.",
        ],
    }
    _write_json(root / "manifest.json", manifest)
    return {
        "output_dir": str(root), "episodes": len(episodes),
        "splits": {split: sum(1 for episode in episodes if episode["split"] == split) for split in ("train", "validation", "test")},
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine_status": manifest["engine_status"],
    }


__all__ = ["EpisodeBenchmarkConfig", "generate_episode_benchmark", "preset_config"]
