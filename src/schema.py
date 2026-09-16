"""Canonical schemas and lightweight validation for Python-generated data.

The project deliberately keeps the interchange format flat and CSV-friendly so
that every pure-Python simulation episode can be inspected without requiring a
specific runtime. Edge-level communication data lives in ``links.csv``;
per-UAV state and sensor data lives in ``trajectory.csv``. Legacy import
adapters may still read old exports, but they are not active experiment data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "uav_id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "uwb_range",
    "neighbor_ids",
    "rssi",
    "snr",
    "packet_loss",
    "delay",
    "aoi",
    "gnss_mode",
    "obstacle_state",
    "action_vx",
    "action_vy",
    "action_vz",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
)

TRAJECTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "uav_id",
    "x",
    "y",
    "z",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
    "action_vx",
    "action_vy",
    "action_vz",
)

LINK_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "src_uav_id",
    "dst_uav_id",
    "distance",
    "los",
    "tx_power_dbm",
    "rx_power_dbm",
    "noise_power_dbm",
    "rssi",
    "snr",
    "channel_bandwidth_hz",
    "rate_bps",
    "packet_loss",
    "delay",
    "aoi",
    "queue_length",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "uwb_available",
    "link_state",
    # Deprecated compatibility alias; use rate_bps in new experiments.
    "bandwidth_kbps",
)

LINK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "src_uav_id",
    "dst_uav_id",
    "distance",
    "los",
    "tx_power_dbm",
    "rx_power_dbm",
    "noise_power_dbm",
    "rssi",
    "snr",
    "channel_bandwidth_hz",
    "rate_bps",
    "packet_loss",
    "delay",
    "aoi",
    "queue_length",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "uwb_available",
    "link_state",
)

OBSTACLE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "obstacle_id",
    "obstacle_type",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "radius",
    "active",
    # Additive semantic flag: false marks a communication-only occluder.  It
    # does not invalidate older episode-v1.3 files, which implicitly used the
    # historical physical-obstacle meaning.
    "physical_collision_enabled",
)

OBSTACLE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "scenario_id",
    "split",
    "obstacle_id",
    "obstacle_type",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "radius",
    "active",
)

EPISODE_SCHEMA_VERSION = "episode-v1.3"
EPISODE_COMMON_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "scenario_id",
    "seed",
    "timestamp",
    "split",
)

GROUND_TRUTH_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "ax",
    "ay",
    "az",
    "wx",
    "wy",
    "wz",
)

IMU_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "bias_accel_x",
    "bias_accel_y",
    "bias_accel_z",
    "bias_gyro_x",
    "bias_gyro_y",
    "bias_gyro_z",
    "saturation_flag",
)

UWB_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "src_uav_id",
    "dst_uav_id",
    "true_range",
    "measured_range",
    "los",
    "nlos_reason",
    "measurement_std",
    "valid",
)

VISUAL_ODOMETRY_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "vo_dx",
    "vo_dy",
    "vo_dz",
    "vo_dyaw",
    "valid",
    "feature_count",
    "tracking_quality",
)

GNSS_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "x",
    "y",
    "z",
    "hdop",
    "fix_type",
    "valid",
    "error_scale",
    "mode",
)

EPISODE_LINK_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "src_uav_id",
    "dst_uav_id",
    "distance",
    "los",
    "tx_power_dbm",
    "rx_power_dbm",
    "noise_power_dbm",
    "rssi",
    "snr",
    "channel_bandwidth_hz",
    "rate_bps",
    "bandwidth_kbps",
    "packet_loss",
    "raw_delay",
    "delay",
    "aoi",
    "queue_length",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "uwb_available",
    "link_state",
)

PACKET_EVENT_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "packet_id",
    "src_uav_id",
    "dst_uav_id",
    "send_timestamp",
    "receive_timestamp",
    "size_bytes",
    "success",
    "drop_reason",
    "delay",
    "aoi_after",
)

ACTIONS_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "measurement_uav_id",
    # Optional v1.3 diagnostic: the environment converted a requested
    # non-empty measurement to ``none`` before execution.
    "invalid_measurement_action",
    # Optional diagnostic: the action was valid at decision time but the
    # selected edge disappeared after the motion window was executed.
    "measurement_transition_invalidated",
    "topology_mode",
    "bandwidth_ratio",
    "bandwidth_allocations_json",
    "action_mask_json",
    "delta_vx",
    "delta_vy",
    "delta_vz",
    "applied_window_start",
    "applied_window_end",
    "action_raw",
    "action_applied",
    "action_clipped",
    # Action-to-motion braking diagnostics.  These are additive so historical
    # episode-v1.3 action logs remain readable without them.
    "pre_speed_mps",
    "post_speed_mps",
    "speed_change_mps",
    "speed_reduction_mps",
    "delta_v_commanded_norm_mps",
    "delta_v_applied_norm_mps",
    "commanded_longitudinal_delta_v_mps",
    "applied_longitudinal_delta_v_mps",
    "commanded_braking_delta_v_mps",
    "braking_delta_v_mps",
    "braking_action",
    "brake_effective",
    "brake_execution_loss_mps",
    "brake_blocked",
    "speed_limit_limited",
    "safety_projected",
    "boundary_projected",
    # Optional single-UAV braking safety-supervisor diagnostics.  These keep
    # the learned command separate from any transparent execution projection.
    "braking_safety_projection_enabled",
    "braking_safety_projected",
    "braking_safety_projection_reason",
    "braking_safety_target_speed_mps",
    "braking_safety_predicted_speed_mps",
    "braking_safety_projection_delta_v_json",
    "policy_source",
    # Timing-mask and budget diagnostics.  These are additive so old v1.3
    # episodes remain readable while new runs can distinguish a policy choice
    # of ``none`` from a physically forced ``none``.
    "candidate_count_before_timing",
    "eligible_candidate_count",
    "timing_filtered_candidate_count",
    "none_forced_by_timing_mask",
    "none_when_candidate_eligible",
    "budget_filtered",
    "measurement_action_invalid_reason",
)

# These two projection diagnostics were added after the first episode-v1.3
# exports.  Keep them optional for validation so the historical v1.3 results
# remain readable; all newly written policy episodes include both columns.
ACTIONS_REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    column for column in ACTIONS_COLUMNS
    if column not in {
        "safety_projected",
        "boundary_projected",
        "pre_speed_mps",
        "post_speed_mps",
        "speed_change_mps",
        "speed_reduction_mps",
        "delta_v_commanded_norm_mps",
        "delta_v_applied_norm_mps",
        "commanded_longitudinal_delta_v_mps",
        "applied_longitudinal_delta_v_mps",
        "commanded_braking_delta_v_mps",
        "braking_delta_v_mps",
        "braking_action",
        "brake_effective",
        "brake_execution_loss_mps",
        "brake_blocked",
        "speed_limit_limited",
        "braking_safety_projection_enabled",
        "braking_safety_projected",
        "braking_safety_projection_reason",
        "braking_safety_target_speed_mps",
        "braking_safety_predicted_speed_mps",
        "braking_safety_projection_delta_v_json",
        "invalid_measurement_action",
        "measurement_transition_invalidated",
        "candidate_count_before_timing",
        "eligible_candidate_count",
        "timing_filtered_candidate_count",
        "none_forced_by_timing_mask",
        "none_when_candidate_eligible",
        "budget_filtered",
        "measurement_action_invalid_reason",
    }
)

REWARDS_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "reward_total",
    "reward_localization",
    "reward_packet_loss",
    "reward_aoi",
    "reward_delay",
    "reward_connectivity",
    "reward_collision",
    "reward_energy",
    "joint_objective",
    "normalized_uncertainty",
    "normalized_packet_loss",
    "normalized_aoi",
    "normalized_delay",
    "normalized_connectivity",
    "collision",
    "task_complete",
)

# Additive learning-channel diagnostics.  They are intentionally not part of
# ``REWARDS_COLUMNS``/the required v1.3 contract so historical episodes remain
# valid.  New writers include them through ``EPISODE_FILE_SPECS``.
REWARDS_EXTENDED_COLUMNS: tuple[str, ...] = REWARDS_COLUMNS + (
    "learning_reward_total",
    "base_learning_reward",
    "boundary_reward",
    "boundary_projected",
    "boundary_projection_count",
    "learning_reward_preclip",
    "learning_reward_mode",
    "learning_potential",
    "learning_progress_potential",
    "learning_progress_logdet_uncertainty",
    "learning_progress_normalized_j_com",
    "learning_feedback_mode",
    "v3_learning_reward",
    "position_logdet_previous",
    "position_logdet_current",
    "loc_progress_delta",
    "loc_progress_reward",
    "actual_measurement_count",
    "actual_measurement_bytes",
    "actual_measurement_rate",
    "actual_measurement_cost_target",
    "actual_measurement_load_excess",
    "actual_measurement_load_reward",
    "feedback_reward",
    # Delayed action-linked EKF credit is a training-only diagnostic.  It is
    # not part of the physical report-reward contract.
    "accepted_update_feedback_reward",
    "accepted_update_feedback_coef",
    "accepted_update_feedback_delay_steps",
    "accepted_update_feedback_covariance_source",
    "accepted_update_feedback_records_json",
    "released_update_feedback_records_json",
    "pending_update_feedback_count",
    # Explicit localization-first single-UAV learning diagnostics.  These
    # fields are additive and therefore remain outside the required legacy
    # reward contract.
    "single_uav_learning_mode",
    "single_uav_learning_active",
    "communication_applicable",
    "communication_reward_in_learning",
    "single_uav_localization_progress_delta",
    "single_uav_localization_progress_scale",
    "localization_progress_score",
    "single_uav_localization_component",
    "previous_goal_distance",
    "current_goal_distance",
    "goal_progress_delta",
    "goal_progress_scale",
    "goal_progress_score",
    "single_uav_goal_component",
    "single_uav_goal_progress_reward",
    "single_uav_goal_bonus",
    "single_uav_goal_bonus_triggered",
    "single_uav_goal_reached",
    "single_uav_goal_in_radius",
    "single_uav_goal_speed_ok",
    "single_uav_goal_stable",
    "single_uav_goal_condition",
    "single_uav_stable_steps",
    "single_uav_stable_hold_steps",
    "single_uav_goal_threshold_m",
    "single_uav_speed_mps",
    "single_uav_goal_speed_threshold_mps",
    "single_uav_near_goal_radius_m",
    "single_uav_speed_control_mode",
    "single_uav_braking_distance_speed_profile",
    "single_uav_target_speed_far_mps",
    "single_uav_target_speed_mps",
    "single_uav_stopping_distance_m",
    "single_uav_brake_margin_m",
    "single_uav_braking_acceleration_mps2",
    "single_uav_braking_margin_m",
    "single_uav_speed_excess_mps",
    "single_uav_speed_deficit_mps",
    "single_uav_speed_deficit_cost",
    "single_uav_speed_deficit_penalty_weight",
    "single_uav_near_goal_speed_cost",
    "single_uav_near_goal_speed_penalty_weight",
    "single_uav_localization_cost",
    "single_uav_localization_cost_weight",
    "single_uav_goal_progress_signal",
    "single_uav_goal_progress_reward_scale_m",
    "single_uav_goal_hold_reward",
    "single_uav_goal_direction_signal",
    "single_uav_goal_direction_coef",
    "single_uav_goal_direction_reward",
    "single_uav_braking_need_signal",
    "single_uav_braking_action_signal",
    "single_uav_effective_braking_signal",
    "single_uav_braking_action_reward",
    "single_uav_braking_action_reward_coef",
    "single_uav_target_velocity_action_error",
    "single_uav_target_velocity_action_reward",
    "single_uav_target_velocity_action_reward_coef",
    "single_uav_terminal_braking_radius_m",
    "single_uav_terminal_braking_active",
    "single_uav_terminal_braking_need_signal",
    "single_uav_terminal_speed_reduction_signal",
    "single_uav_terminal_forward_accel_signal",
    "single_uav_terminal_braking_reward",
    "single_uav_terminal_braking_reward_coef",
    "single_uav_terminal_acceleration_penalty",
    "single_uav_terminal_acceleration_penalty_coef",
    "single_uav_pre_speed_excess_mps",
    "single_uav_post_speed_excess_mps",
    "single_uav_braking_progress_signal",
    "single_uav_braking_progress_reward",
    "single_uav_braking_progress_reward_coef",
    "single_uav_terminal_radial_progress_active",
    "single_uav_terminal_radial_progress_m",
    "single_uav_terminal_radial_progress_speed_excess_mps",
    "single_uav_terminal_radial_progress_speed_gate",
    "single_uav_terminal_radial_progress_signal",
    "single_uav_terminal_radial_progress_reward",
    "single_uav_terminal_radial_progress_reward_coef",
    "single_uav_terminal_target_velocity_vector_active",
    "single_uav_terminal_target_velocity_vector_error_pre_mps",
    "single_uav_terminal_target_velocity_vector_error_post_mps",
    "single_uav_terminal_target_velocity_vector_error_reduction_mps",
    "single_uav_terminal_target_velocity_vector_signal",
    "single_uav_terminal_target_velocity_vector_reward",
    "single_uav_terminal_target_velocity_vector_reward_coef",
    "single_uav_terminal_target_velocity_vector_radial_gate_enabled",
    "single_uav_terminal_target_velocity_vector_activation_distance_m",
    "single_uav_terminal_target_velocity_vector_post_radial_velocity_mps",
    "single_uav_terminal_target_velocity_vector_radial_gate",
    "single_uav_terminal_target_velocity_vector_mps",
    "single_uav_learning_energy_cost",
    "single_uav_learning_collision_cost",
    "single_uav_learning_core",
    "single_uav_learning_reward_preclip",
    "single_uav_learning_reward",
)

EPISODE_OBSTACLE_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "obstacle_id",
    "obstacle_type",
    "geometry_type",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "radius",
    "active",
    # Additive semantic flag: false marks a communication-only occluder.
    # It is intentionally not required so historical episode-v1.3 files
    # remain readable and retain their implicit physical-obstacle semantics.
    "physical_collision_enabled",
)

EPISODE_TRAJECTORY_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + tuple(
    column for column in TRAJECTORY_COLUMNS if column not in EPISODE_COMMON_COLUMNS
)

ESTIMATE_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "covariance_flat_json",
    "valid_observation_count",
    "valid_neighbor_count",
    "last_neighbor_timestamp",
)

EKF_DIAGNOSTICS_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "nis_vo",
    "nis_uwb_mean",
    "nis_uwb_max",
    "nis_gnss",
    "vo_update_attempted",
    "vo_update_accepted",
    "vo_update_rejected",
    "uwb_update_attempted",
    "uwb_update_accepted",
    "uwb_update_rejected",
    "gnss_update_attempted",
    "gnss_update_accepted",
    "gnss_update_rejected",
    "rejected_observation_count",
    "position_error_m",
    "position_nees",
    "position_covariance_trace_m2",
    # Additive VO-quality diagnostics.  These columns expose the frozen
    # execution-time covariance mapping without changing the required legacy
    # v1.3 fields below.
    "vo_measurement_covariance_trace_m2",
    "vo_measurement_covariance_scale",
    "vo_quality_tracking",
    "vo_quality_covariance_scale",
    "vo_quality_covariance_valid",
    "vo_quality_covariance_bin",
    # Accepted and rejected innovation statistics are separate so a large
    # rejected outlier is not misread as an innovation used by the filter.
    "nis_vo_accepted",
    "nis_vo_rejected",
    "nis_uwb_accepted_mean",
    "nis_uwb_rejected_mean",
    "nis_gnss_accepted",
    "nis_gnss_rejected",
)

# v1.3 originally contained only the aggregate NIS fields.  The accepted /
# rejected split is additive and must not invalidate old v1.3 directories.
EKF_DIAGNOSTICS_REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in EKF_DIAGNOSTICS_COLUMNS
    if column
    not in {
        "nis_vo_accepted",
        "nis_vo_rejected",
        "nis_uwb_accepted_mean",
        "nis_uwb_rejected_mean",
        "nis_gnss_accepted",
        "nis_gnss_rejected",
        "vo_measurement_covariance_trace_m2",
        "vo_measurement_covariance_scale",
        "vo_quality_tracking",
        "vo_quality_covariance_scale",
        "vo_quality_covariance_valid",
        "vo_quality_covariance_bin",
    }
)

GRAPH_FEATURE_BASE_COLUMNS: tuple[str, ...] = EPISODE_COMMON_COLUMNS + (
    "uav_id",
    "node_features_json",
    "effective_edge_index_json",
    "effective_edge_features_json",
    "candidate_edge_index_json",
    "candidate_edge_features_json",
    "action_mask_json",
)

# Additive v1.2 fields.  They expose the potential outgoing edges used by the
# bandwidth head without making the older graph feature files unreadable.
GRAPH_FEATURE_COLUMNS: tuple[str, ...] = GRAPH_FEATURE_BASE_COLUMNS + (
    "potential_edge_index_json",
    "potential_edge_features_json",
    "bandwidth_mask_json",
    "candidate_incoming_edge_features_json",
    "candidate_incoming_links_json",
    "uwb_available_json",
    "uwb_ranges_json",
    "central_state_json",
)

EPISODE_FILE_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ground_truth.csv": (GROUND_TRUTH_COLUMNS, ("seed", "timestamp", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz")),
    "imu.csv": (IMU_COLUMNS, ("seed", "timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "bias_accel_x", "bias_accel_y", "bias_accel_z", "bias_gyro_x", "bias_gyro_y", "bias_gyro_z", "saturation_flag")),
    "uwb_ranges.csv": (UWB_COLUMNS, ("seed", "timestamp", "true_range", "measured_range", "los", "measurement_std", "valid")),
    "visual_odometry.csv": (VISUAL_ODOMETRY_COLUMNS, ("seed", "timestamp", "vo_dx", "vo_dy", "vo_dz", "vo_dyaw", "valid", "feature_count", "tracking_quality")),
    "gnss.csv": (GNSS_COLUMNS, ("seed", "timestamp", "x", "y", "z", "hdop", "valid", "error_scale")),
    "links.csv": (EPISODE_LINK_COLUMNS, ("seed", "timestamp", "distance", "los", "tx_power_dbm", "rx_power_dbm", "noise_power_dbm", "rssi", "snr", "channel_bandwidth_hz", "rate_bps", "bandwidth_kbps", "packet_loss", "raw_delay", "delay", "aoi", "queue_length", "potential_edge", "candidate_edge", "effective_edge", "uwb_available")),
    "packet_events.csv": (PACKET_EVENT_COLUMNS, ("seed", "timestamp", "send_timestamp", "receive_timestamp", "size_bytes", "success", "delay", "aoi_after")),
    "actions.csv": (
        ACTIONS_COLUMNS,
        (
            "seed",
            "timestamp",
            "bandwidth_ratio",
            "delta_vx",
            "delta_vy",
            "delta_vz",
            "applied_window_start",
            "applied_window_end",
            "action_clipped",
            "pre_speed_mps",
            "post_speed_mps",
            "speed_change_mps",
            "speed_reduction_mps",
            "delta_v_commanded_norm_mps",
            "delta_v_applied_norm_mps",
            "commanded_longitudinal_delta_v_mps",
            "applied_longitudinal_delta_v_mps",
            "commanded_braking_delta_v_mps",
            "braking_delta_v_mps",
            "braking_action",
            "brake_effective",
            "brake_execution_loss_mps",
            "brake_blocked",
            "speed_limit_limited",
            "braking_safety_projection_enabled",
            "braking_safety_projected",
            "braking_safety_target_speed_mps",
            "braking_safety_predicted_speed_mps",
            "safety_projected",
            "boundary_projected",
        ),
    ),
    "rewards.csv": (REWARDS_EXTENDED_COLUMNS, ("seed", "timestamp", "reward_total", "learning_reward_total", "base_learning_reward", "boundary_reward", "boundary_projected", "boundary_projection_count", "learning_reward_preclip", "learning_potential", "reward_localization", "reward_packet_loss", "reward_aoi", "reward_delay", "reward_connectivity", "reward_collision", "reward_energy", "joint_objective", "normalized_uncertainty", "normalized_packet_loss", "normalized_aoi", "normalized_delay", "normalized_connectivity", "collision", "task_complete")),
    "obstacles.csv": (EPISODE_OBSTACLE_COLUMNS, ("seed", "timestamp", "x", "y", "z", "vx", "vy", "vz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "radius", "active")),
    "trajectory.csv": (EPISODE_TRAJECTORY_COLUMNS, ("seed", "timestamp", "x", "y", "z", "vx", "vy", "vz", "yaw", "ground_truth_x", "ground_truth_y", "ground_truth_z", "action_vx", "action_vy", "action_vz")),
    "estimates.csv": (ESTIMATE_COLUMNS, ("seed", "timestamp", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "valid_observation_count", "valid_neighbor_count", "last_neighbor_timestamp")),
    "ekf_diagnostics.csv": (EKF_DIAGNOSTICS_COLUMNS, ("seed", "timestamp", "nis_vo", "nis_uwb_mean", "nis_uwb_max", "nis_gnss", "vo_update_attempted", "vo_update_accepted", "vo_update_rejected", "uwb_update_attempted", "uwb_update_accepted", "uwb_update_rejected", "gnss_update_attempted", "gnss_update_accepted", "gnss_update_rejected", "rejected_observation_count", "position_error_m", "position_nees", "position_covariance_trace_m2", "nis_vo_accepted", "nis_vo_rejected", "nis_uwb_accepted_mean", "nis_uwb_rejected_mean", "nis_gnss_accepted", "nis_gnss_rejected")),
    "graph_features.csv": (GRAPH_FEATURE_COLUMNS, ("seed", "timestamp")),
}

EPISODE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "ground_truth.csv": GROUND_TRUTH_COLUMNS,
    "imu.csv": IMU_COLUMNS,
    "uwb_ranges.csv": EPISODE_COMMON_COLUMNS + ("src_uav_id", "dst_uav_id", "true_range", "los", "measurement_std", "valid"),
    "visual_odometry.csv": EPISODE_COMMON_COLUMNS + ("uav_id", "valid", "feature_count", "tracking_quality"),
    "gnss.csv": EPISODE_COMMON_COLUMNS + ("uav_id", "hdop", "fix_type", "valid", "error_scale", "mode"),
    "links.csv": EPISODE_LINK_COLUMNS,
    "packet_events.csv": EPISODE_COMMON_COLUMNS + ("packet_id", "src_uav_id", "dst_uav_id", "send_timestamp", "size_bytes", "success", "delay", "aoi_after"),
    "actions.csv": ACTIONS_REQUIRED_COLUMNS,
    "rewards.csv": REWARDS_COLUMNS,
    "obstacles.csv": EPISODE_COMMON_COLUMNS + ("obstacle_id", "obstacle_type", "geometry_type", "x", "y", "z", "vx", "vy", "vz", "radius", "active"),
    "trajectory.csv": EPISODE_COMMON_COLUMNS + ("uav_id", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "neighbor_ids", "rssi", "snr", "packet_loss", "delay", "aoi", "gnss_mode", "obstacle_state", "action_vx", "action_vy", "action_vz", "ground_truth_x", "ground_truth_y", "ground_truth_z"),
    "estimates.csv": ESTIMATE_COLUMNS,
    "ekf_diagnostics.csv": EKF_DIAGNOSTICS_REQUIRED_COLUMNS,
    "graph_features.csv": GRAPH_FEATURE_BASE_COLUMNS,
}

# ``episode-v1.1`` and ``episode-v1.2`` are read-only compatibility contracts.
# v1.2 adds graph exports, potential/UWB edge flags, per-edge bandwidth JSON,
# and normalized reward components; v1.3 adds EKF consistency diagnostics.
# These additions must not make an older benchmark unreadable.
EPISODE_SCHEMA_V1_1 = "episode-v1.1"
EPISODE_SCHEMA_V1_2 = "episode-v1.2"
EPISODE_SCHEMA_V1_3 = "episode-v1.3"
_V12_ONLY_FILES = {"estimates.csv", "graph_features.csv"}
_V13_ONLY_FILES = {"ekf_diagnostics.csv"}
_V12_ONLY_COLUMNS = {
    "links.csv": {"potential_edge", "uwb_available", "raw_delay"},
    "actions.csv": {
        "measurement_uav_id",
        "bandwidth_allocations_json",
        "action_mask_json",
        "applied_window_start",
        "applied_window_end",
    },
    "rewards.csv": {
        "joint_objective",
        "normalized_uncertainty",
        "normalized_packet_loss",
        "normalized_aoi",
        "normalized_delay",
        "normalized_connectivity",
    },
}


def _episode_contract(
    schema_version: str | None,
) -> tuple[
    dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    dict[str, tuple[str, ...]],
]:
    """Return the file/column contract for the current or legacy schema."""

    if schema_version not in {EPISODE_SCHEMA_V1_1, EPISODE_SCHEMA_V1_2}:
        return EPISODE_FILE_SPECS, EPISODE_REQUIRED_COLUMNS

    # v1.2 includes the estimate and graph exports introduced by v1.2;
    # only v1.1 omits them.  Both legacy contracts omit the v1.3 EKF
    # diagnostics file.  Keeping this distinction makes validation strict
    # without making the historical v1.2 benchmark look like v1.1.
    excluded_files = set(_V13_ONLY_FILES)
    if schema_version == EPISODE_SCHEMA_V1_1:
        excluded_files.update(_V12_ONLY_FILES)
    file_specs = {
        filename: spec
        for filename, spec in EPISODE_FILE_SPECS.items()
        if filename not in excluded_files
    }
    required: dict[str, tuple[str, ...]] = {}
    for filename in file_specs:
        excluded = _V12_ONLY_COLUMNS.get(filename, set()) if schema_version == EPISODE_SCHEMA_V1_1 else set()
        required[filename] = tuple(
            column
            for column in EPISODE_REQUIRED_COLUMNS[filename]
            if column not in excluded
        )
    return file_specs, required


@dataclass
class CsvValidation:
    """Validation result for one canonical CSV file."""

    path: str
    required_columns: tuple[str, ...]
    row_count: int = 0
    missing_columns: list[str] = field(default_factory=list)
    invalid_rows: list[str] = field(default_factory=list)
    empty_required_values: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_columns
            or self.invalid_rows
            or self.empty_required_values
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "ok": self.ok,
            "required_columns": list(self.required_columns),
            "row_count": self.row_count,
            "missing_columns": self.missing_columns,
            "invalid_rows": self.invalid_rows,
            "empty_required_values": self.empty_required_values,
        }


def _is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def validate_csv(
    path: str | Path,
    required_columns: Sequence[str],
    numeric_columns: Iterable[str] = (),
    max_errors: int = 20,
) -> CsvValidation:
    """Validate a canonical CSV without loading the full file into memory."""

    csv_path = Path(path)
    result = CsvValidation(str(csv_path), tuple(required_columns))
    if not csv_path.exists():
        result.invalid_rows.append("file does not exist")
        return result

    numeric = set(numeric_columns)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(reader.fieldnames or ())
            result.missing_columns = [
                column for column in required_columns if column not in headers
            ]
            if result.missing_columns:
                return result

            for line_number, row in enumerate(reader, start=2):
                result.row_count += 1
                for column in required_columns:
                    if not str(row.get(column, "")).strip():
                        if len(result.empty_required_values) < max_errors:
                            result.empty_required_values.append(
                                f"line {line_number}: {column} is empty"
                            )
                for column in numeric:
                    value = str(row.get(column, "")).strip()
                    if value and not _is_number(value):
                        if len(result.invalid_rows) < max_errors:
                            result.invalid_rows.append(
                                f"line {line_number}: {column}={value!r} is not numeric"
                            )
    except (OSError, UnicodeError, csv.Error) as exc:
        result.invalid_rows.append(f"could not read CSV: {exc}")
    return result


def validate_dataset_dir(path: str | Path) -> dict[str, object]:
    """Validate all canonical files in a generated benchmark directory."""

    root = Path(path)
    files = {
        "trajectory": validate_csv(
            root / "trajectory.csv",
            TRAJECTORY_REQUIRED_COLUMNS,
            numeric_columns=(
                "timestamp",
                "x",
                "y",
                "z",
                "ground_truth_x",
                "ground_truth_y",
                "ground_truth_z",
                "action_vx",
                "action_vy",
                "action_vz",
            ),
        ),
        "links": validate_csv(
            root / "links.csv",
            LINK_REQUIRED_COLUMNS,
            numeric_columns=(
                "timestamp",
                "distance",
                "los",
                "tx_power_dbm",
                "rx_power_dbm",
                "noise_power_dbm",
                "rssi",
                "snr",
                "channel_bandwidth_hz",
                "rate_bps",
                "packet_loss",
                "delay",
                "aoi",
                "queue_length",
                "potential_edge",
                "candidate_edge",
                "effective_edge",
                "uwb_available",
            ),
        ),
        "obstacles": validate_csv(
            root / "obstacles.csv",
            OBSTACLE_REQUIRED_COLUMNS,
            numeric_columns=(
                "timestamp",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "radius",
                "active",
            ),
        ),
    }
    reports = {name: report.to_dict() for name, report in files.items()}
    return {
        "path": str(root),
        "ok": all(report.ok for report in files.values()),
        "files": reports,
        "metadata_present": (root / "metadata.json").exists(),
    }


def validate_episode_dir(
    path: str | Path,
    *,
    schema_version: str | None = None,
) -> dict[str, object]:
    """Validate one episode directory against a current or legacy contract."""

    root = Path(path)
    if schema_version is None:
        scenario_path = root / "scenario.json"
        if scenario_path.exists():
            try:
                scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
                detected_version = scenario_payload.get("schema_version")
                if detected_version:
                    schema_version = str(detected_version)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    file_reports: dict[str, dict[str, object]] = {}
    file_specs, required_columns = _episode_contract(schema_version)
    for filename, (required, numeric) in file_specs.items():
        report = validate_csv(
            root / filename,
            required_columns[filename],
            numeric_columns=numeric,
        )
        file_reports[filename] = report.to_dict()

    scenario_present = (root / "scenario.json").exists()
    integrity_issues: list[str] = []
    ground_truth_path = root / "ground_truth.csv"
    if ground_truth_path.exists():
        seen_ground_truth: set[tuple[str, str]] = set()
        last_timestamp: dict[str, float] = {}
        try:
            with ground_truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("uav_id", ""), row.get("timestamp", ""))
                    if key in seen_ground_truth:
                        integrity_issues.append(f"duplicate ground_truth key: {key}")
                    seen_ground_truth.add(key)
                    uav_id = row.get("uav_id", "")
                    timestamp = float(row["timestamp"])
                    if uav_id in last_timestamp and timestamp < last_timestamp[uav_id]:
                        integrity_issues.append(f"ground_truth timestamp moved backwards for {uav_id}")
                    last_timestamp[uav_id] = timestamp
        except (OSError, ValueError, csv.Error) as exc:
            integrity_issues.append(f"could not inspect ground_truth integrity: {exc}")

    known_uavs: set[str] = set()
    if ground_truth_path.exists():
        try:
            with ground_truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
                known_uavs = {row.get("uav_id", "") for row in csv.DictReader(handle)}
        except (OSError, csv.Error) as exc:
            integrity_issues.append(f"could not collect UAV ids: {exc}")

    links_path = root / "links.csv"
    if links_path.exists():
        seen_links: set[tuple[str, str, str]] = set()
        try:
            with links_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("timestamp", ""), row.get("src_uav_id", ""), row.get("dst_uav_id", ""))
                    if key in seen_links:
                        integrity_issues.append(f"duplicate link key: {key}")
                    seen_links.add(key)
                    if row.get("src_uav_id") not in known_uavs or row.get("dst_uav_id") not in known_uavs:
                        integrity_issues.append(f"link references unknown UAV: {key}")
        except (OSError, csv.Error) as exc:
            integrity_issues.append(f"could not inspect link integrity: {exc}")

    packet_path = root / "packet_events.csv"
    if packet_path.exists():
        seen_packets: set[str] = set()
        try:
            with packet_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    packet_id = row.get("packet_id", "")
                    if packet_id in seen_packets:
                        integrity_issues.append(f"duplicate packet_id: {packet_id}")
                    seen_packets.add(packet_id)
        except (OSError, csv.Error) as exc:
            integrity_issues.append(f"could not inspect packet integrity: {exc}")

    if len(integrity_issues) > 20:
        integrity_issues = integrity_issues[:20] + ["additional integrity issues omitted"]
    return {
        "path": str(root),
        "ok": all(bool(report["ok"]) for report in file_reports.values()) and scenario_present and not integrity_issues,
        "scenario_present": scenario_present,
        "files": file_reports,
        "integrity_issues": integrity_issues,
    }


def validate_episode_benchmark(path: str | Path) -> dict[str, object]:
    """Validate the manifest, calibration files, and every episode directory."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    calibration_paths = (
        root / "calibration" / "localization_noise.json",
        root / "calibration" / "communication_model.json",
    )
    manifest_payload: dict[str, object] = {}
    manifest_version = EPISODE_SCHEMA_VERSION
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_version = str(manifest_payload.get("schema_version", manifest_version))
        except (OSError, ValueError, TypeError):
            # The detailed parse error is reported below; use the current
            # contract for the best available diagnostic in the meantime.
            pass
    episodes: list[Path] = []
    for split in ("train", "validation", "test"):
        split_root = root / split
        if split_root.exists():
            episodes.extend(sorted(split_root.glob("episode_*")))

    episode_reports = [
        validate_episode_dir(episode, schema_version=manifest_version)
        for episode in episodes
    ]
    manifest_issues: list[str] = []
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            seen_scenarios: dict[str, str] = {}
            for entry in manifest_payload.get("episodes", []):
                scenario_id = str(entry.get("scenario_id", ""))
                split = str(entry.get("split", ""))
                if scenario_id in seen_scenarios and seen_scenarios[scenario_id] != split:
                    manifest_issues.append(f"scenario appears in multiple splits: {scenario_id}")
                seen_scenarios[scenario_id] = split
                episode_path = root / split / str(entry.get("episode_id", ""))
                for filename, expected_hash in dict(entry.get("files", {})).items():
                    file_path = episode_path / filename
                    if not file_path.exists():
                        manifest_issues.append(f"manifest file is missing: {file_path}")
                    elif _file_sha256(file_path) != expected_hash:
                        manifest_issues.append(f"manifest hash mismatch: {file_path}")
        except (OSError, ValueError, TypeError) as exc:
            manifest_issues.append(f"could not inspect manifest: {exc}")
    return {
        "path": str(root),
        "schema_version": manifest_version,
        "manifest_present": manifest_path.exists(),
        "calibration_present": all(path.exists() for path in calibration_paths),
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
        "manifest_issues": manifest_issues,
        "ok": (
            manifest_path.exists()
            and all(path.exists() for path in calibration_paths)
            and bool(episode_reports)
            and all(bool(report["ok"]) for report in episode_reports)
            and not manifest_issues
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
