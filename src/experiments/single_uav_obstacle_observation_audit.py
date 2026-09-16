"""Read-only audit of the single-UAV obstacle observation contract.

The audit deliberately stops before ``env.step``.  It constructs paired
scenes with identical UAV/goal state and equal-distance spherical obstacles,
then traces the observation that reaches the frozen actor.  It is diagnostic
only: no PPO update, checkpoint save, reward change, or safety-policy change
is performed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import MotionConfig
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import _make_environment
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_navigation_preserving_finetune import _read_checkpoint


DEFAULT_BASELINE_CHECKPOINT = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_STAGE1_CHECKPOINT = Path(
    "data/synthetic/single_uav_static_obstacle_avoidance_stage1_formal_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_obstacle_observation_audit_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_obstacle_observation_audit_seed2026"
)

PROBE_SEED = 2026
PROBE_OBSTACLE_DISTANCE_M = 6.0
PROBE_OBSTACLE_RADIUS_M = 1.0
PROBE_UAV_POSITION = np.asarray([50.0, 100.0, 20.0], dtype=float)
PROBE_GOAL_POSITION = np.asarray([80.0, 100.0, 20.0], dtype=float)
PROBE_VELOCITY = np.zeros(3, dtype=float)

# The audit convention is +y = left and -y = right when the UAV faces the
# +x goal direction.  All centers are six metres from the UAV; only the
# direction changes.
PROBE_CASES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("front", (6.0, 0.0, 0.0)),
    ("left_front", (4.0, 4.47213595499958, 0.0)),
    ("right_front", (4.0, -4.47213595499958, 0.0)),
    ("left", (0.0, 6.0, 0.0)),
    ("right", (0.0, -6.0, 0.0)),
    ("behind", (-6.0, 0.0, 0.0)),
)


def _array_hash(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _probe_scenario(case_name: str, relative_offset: Sequence[float]) -> dict[str, Any]:
    offset = np.asarray(relative_offset, dtype=float).reshape(3)
    center = PROBE_UAV_POSITION + offset
    return {
        "scenario_version": "single-uav-obstacle-observation-audit-v1",
        "seed": PROBE_SEED,
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": PROBE_UAV_POSITION.tolist()},
        "initial_velocities": {"uav_00": PROBE_VELOCITY.tolist()},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": PROBE_GOAL_POSITION.tolist()},
        "obstacle_specs": [
            {
                "obstacle_id": "audit_static_obstacle",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": center.tolist(),
                "radius_m": PROBE_OBSTACLE_RADIUS_M,
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "observation_audit": {
            "case": str(case_name),
            "relative_offset_m": offset.tolist(),
            "uav_position_fixed": True,
            "goal_fixed": True,
            "velocity_fixed": True,
            "obstacle_center_distance_m": PROBE_OBSTACLE_DISTANCE_M,
            "direction_only_change": True,
            "left_is_positive_y": True,
        },
    }


def probe_scenarios() -> dict[str, dict[str, Any]]:
    """Return the six deterministic direction-only probe scenarios."""

    return {
        name: _probe_scenario(name, offset) for name, offset in PROBE_CASES
    }


def actor_node_feature_layout(node_feature_dim: int) -> list[dict[str, Any]]:
    """Describe the runtime node-feature slots used by the current actor."""

    state_names = (
        "estimated_position_x",
        "estimated_position_y",
        "estimated_position_z",
        "estimated_velocity_x",
        "estimated_velocity_y",
        "estimated_velocity_z",
        "estimated_attitude_roll",
        "estimated_attitude_pitch",
        "estimated_attitude_yaw",
        "estimated_accel_bias_x",
        "estimated_accel_bias_y",
        "estimated_accel_bias_z",
        "estimated_gyro_bias_x",
        "estimated_gyro_bias_y",
        "estimated_gyro_bias_z",
    )
    covariance_names = (
        "covariance_position_x",
        "covariance_position_y",
        "covariance_position_z",
        "covariance_velocity_x",
        "covariance_velocity_y",
        "covariance_velocity_z",
        "covariance_attitude_roll",
        "covariance_attitude_pitch",
        "covariance_attitude_yaw",
        "covariance_accel_bias_x",
        "covariance_accel_bias_y",
        "covariance_accel_bias_z",
        "covariance_gyro_bias_x",
        "covariance_gyro_bias_y",
        "covariance_gyro_bias_z",
    )
    labels = [
        *state_names,
        *covariance_names,
        "goal_relative_dx",
        "goal_relative_dy",
        "goal_relative_dz",
        "minimum_obstacle_clearance",
        "obstacle_box_count",
        "obstacle_sphere_count",
        "environment_timestep",
        "braking_target_speed_normalized",
        "braking_margin_normalized",
    ]
    frames = [
        *(["world_estimated_state"] * len(state_names)),
        *(["state_covariance"] * len(covariance_names)),
        *(["world_relative_goal"] * 3),
        *(["scalar_aggregate"] * 4),
        *(["scalar_braking_descriptor"] * 2),
    ]
    sources = [
        *(["LocalEKF.state"] * len(state_names)),
        *(["LocalEKF.covariance diagonal"] * len(covariance_names)),
        *(["goal - EKF estimate"] * 3),
        *(["environment._local_obstacle_features"] * 4),
        *(["environment._build_graph_batch braking features"] * 2),
    ]
    rows: list[dict[str, Any]] = []
    for index, (label, frame, source) in enumerate(zip(labels, frames, sources)):
        rows.append(
            {
                "layer": "actor_local_node_features",
                "feature": label,
                "index": index,
                "dimension": 1,
                "coordinate_frame": frame,
                "source": source,
                "present_in_observation": True,
                "enters_actor": index < int(node_feature_dim),
                "notes": "normalized by frozen ObservationNormalizer before policy use",
            }
        )
    for index in range(len(labels), int(node_feature_dim)):
        rows.append(
            {
                "layer": "actor_local_node_features",
                "feature": f"unused_or_padded_node_feature_{index}",
                "index": index,
                "dimension": 1,
                "coordinate_frame": "none",
                "source": "runtime padding",
                "present_in_observation": False,
                "enters_actor": True,
                "notes": "not populated by current graph_interface node_row",
            }
        )
    return rows


def _required_feature_audit_rows(config: Any) -> list[dict[str, Any]]:
    """Record required/absent features that are not individual node slots."""

    dynamic_enabled = bool(config.dynamic_obstacle_motion_features_enabled)
    rows = [
        ("uav_speed_separate_scalar", False, "derived from velocity", "Actor can compute speed from velocity; no separate slot"),
        ("uav_acceleration", False, "not exposed", "Estimated state contains acceleration/gyro bias, not current acceleration"),
        ("goal_absolute_position", True, "local observation dictionary", "Present as metadata but not consumed by _local_context"),
        ("goal_distance_separate_scalar", False, "derived from goal_relative", "No separate goal-distance slot"),
        ("goal_direction_normalized", False, "derived from goal_relative", "Sign/direction remains recoverable from dx/dy/dz"),
        ("obstacle_count", True, "local obstacle aggregate", "Box and sphere counts are separate slots"),
        ("obstacle_minimum_clearance", True, "local obstacle aggregate", "Scalar only; no obstacle identity or direction"),
        ("obstacle_nearest_distance", False, "not separately exposed", "Minimum clearance is exposed, nearest identity is not"),
        ("obstacle_absolute_position", False, "critic/safety only", "Not in actor node features"),
        ("obstacle_relative_dx_dy_dz", False, "not exposed to actor", "Direction-only probe tests this directly"),
        ("obstacle_bearing_or_direction", False, "not exposed to actor", "No signed lateral/longitudinal obstacle feature"),
        ("obstacle_size_or_bbox", False, "critic/safety only", "Not in actor node features"),
        ("obstacle_velocity", dynamic_enabled, "optional local dynamic aggregate", "Current checkpoint setting is recorded in summary"),
        ("obstacle_swept_clearance", False, "diagnostics/safety only", "Stage reward uses it before projection; actor does not"),
        ("obstacle_collision_risk_flag", False, "diagnostics/safety only", "No pre-projection unsafe flag in actor input"),
        ("boundary_feature", False, "not exposed", "Position is visible, but no explicit boundary distance/normal"),
        ("remaining_time_fraction", bool(config.include_remaining_time), "optional actor context", "Appended only when include_remaining_time is enabled"),
    ]
    return [
        {
            "layer": "feature_presence_audit",
            "feature": feature,
            "index": "",
            "dimension": 1 if present else 0,
            "coordinate_frame": frame,
            "source": "runtime/config audit",
            "present_in_observation": bool(present),
            "enters_actor": bool(present and feature in {"remaining_time_fraction"}),
            "notes": notes,
        }
        for feature, present, frame, notes in rows
    ]


def _obstacle_runtime_payload(environment: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for obstacle in environment.obstacles:
        result.append(
            {
                "obstacle_id": str(obstacle.obstacle_id),
                "obstacle_type": str(obstacle.obstacle_type),
                "geometry_type": type(obstacle).__name__,
                "center_m": np.asarray(obstacle.center_m, dtype=float).tolist(),
                "velocity_mps": (
                    None
                    if obstacle.velocity_mps is None
                    else np.asarray(obstacle.velocity_mps, dtype=float).tolist()
                ),
                "radius_m": float(getattr(obstacle, "radius_m", 0.0)),
                "size_m": (
                    np.asarray(obstacle.size_m, dtype=float).tolist()
                    if hasattr(obstacle, "size_m")
                    else None
                ),
                "physical_collision_enabled": bool(
                    getattr(obstacle, "physical_collision_enabled", True)
                ),
            }
        )
    return result


def _select_observation_payload(local: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "estimate",
        "covariance",
        "goal",
        "obstacle_features",
        "node_ids",
        "node_features",
        "effective_edge_index",
        "effective_edge_features",
        "candidate_edge_index",
        "action_mask",
        "bandwidth_mask",
        "remaining_time_fraction",
    )
    return {key: _jsonable(local[key]) for key in keys if key in local}


def _probe_one_checkpoint(
    *,
    role: str,
    checkpoint: Path,
    output_root: Path,
    scenarios: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, checkpoint_sha256_before = _read_checkpoint(checkpoint)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=False)
    config = replace(
        config,
        output_dir=output_root,
        force=True,
        progress=False,
        command_line="single-UAV obstacle observation audit; read-only",
        scenario_path=None,
        scenario_pool_root=None,
    )
    trainer, identity, _checkpoint_identity = _load_frozen_trainer(
        payload,
        config,
        metric_scales=config.metric_scales,
    )
    trainer.policy.eval()
    policy_hash_before = _state_dict_hash(trainer.policy.state_dict())
    expected_policy_hash = _state_dict_hash(payload.get("policy_state_dict", {}))
    if policy_hash_before != expected_policy_hash:
        raise RuntimeError(f"{role} did not load the checkpoint policy exactly")

    records: list[dict[str, Any]] = []
    for case_name, scenario in scenarios.items():
        environment = _make_environment(
            config,
            PROBE_SEED,
            safety_projection_enabled=True,
        )
        observations, info = environment.reset(
            seed=PROBE_SEED,
            scenario_config=copy.deepcopy(dict(scenario)),
        )
        raw_local = observations["uav_00"]
        decision = trainer.act(
            observations,
            info["central_state"],
            deterministic=True,
        )
        normalized_local = decision.observations["uav_00"]
        context, action_mask, bandwidth_mask, effective, potential, embedding, node_ids = (
            trainer.policy._local_context(normalized_local)
        )
        raw_node = np.asarray(raw_local["node_features"], dtype=float)
        normalized_node = np.asarray(normalized_local["node_features"], dtype=float)
        actor_context = context.detach().cpu().numpy().astype(np.float32)
        actor_embedding = embedding.detach().cpu().numpy().astype(np.float32)
        delta_v = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=float).reshape(3)
        uav_position = np.asarray(environment.states["uav_00"].position_m, dtype=float)
        obstacle = _obstacle_runtime_payload(environment)[0]
        obstacle_center = np.asarray(obstacle["center_m"], dtype=float)
        relative = obstacle_center - uav_position
        records.append(
            {
                "role": str(role),
                "case": str(case_name),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256_before,
                "policy_hash": policy_hash_before,
                "scenario": _jsonable(scenario),
                "environment_state": {
                    "uav_position_m": uav_position.tolist(),
                    "uav_velocity_mps": np.asarray(
                        environment.states["uav_00"].velocity_mps, dtype=float
                    ).tolist(),
                    "goal_m": np.asarray(environment.goals["uav_00"], dtype=float).tolist(),
                    "obstacles": _obstacle_runtime_payload(environment),
                },
                "obstacle_relative_position_m": relative.tolist(),
                "obstacle_center_distance_m": float(np.linalg.norm(relative)),
                "obstacle_surface_clearance_m": float(
                    raw_local["obstacle_features"][0]
                ),
                "raw_environment_observation": _select_observation_payload(raw_local),
                "normalized_actor_observation": _select_observation_payload(normalized_local),
                "raw_node_features": raw_node.tolist(),
                "normalized_node_features": normalized_node.tolist(),
                "actor_context": actor_context.tolist(),
                "actor_embedding": actor_embedding.tolist(),
                "actor_context_dim": int(actor_context.size),
                "actor_node_shape": list(raw_node.shape),
                "node_ids": [str(value) for value in node_ids],
                "effective_edge_mean": effective.detach().cpu().numpy().tolist(),
                "potential_edge_mean": potential.detach().cpu().numpy().tolist(),
                "action_mask": action_mask.detach().cpu().numpy().astype(int).tolist(),
                "bandwidth_mask": bandwidth_mask.detach().cpu().numpy().astype(int).tolist(),
                "actor_action": {
                    "measurement_uav_id": str(
                        decision.actions["uav_00"].get("measurement_uav_id", "")
                    ),
                    "delta_v_mps": delta_v.tolist(),
                    "delta_v_norm_mps": float(np.linalg.norm(delta_v)),
                },
                "hashes": {
                    "raw_node_features": _array_hash(raw_node),
                    "normalized_node_features": _array_hash(normalized_node),
                    "actor_context": _array_hash(actor_context),
                    "actor_action_delta_v": _array_hash(delta_v),
                },
            }
        )

    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    checkpoint_sha256_after = _sha256(checkpoint)
    integrity = {
        **read_only_integrity_report(
            policy_hash_before=policy_hash_before,
            policy_hash_after=policy_hash_after,
            checkpoint_sha256_before=checkpoint_sha256_before,
            checkpoint_sha256_after=checkpoint_sha256_after,
        ),
        "trainer_update_calls": 0,
        "loaded_policy_hash_matches_checkpoint": policy_hash_before == expected_policy_hash,
        "identity": identity,
    }
    if not integrity["policy_unchanged"] or not integrity["checkpoint_unchanged"]:
        raise RuntimeError(f"read-only integrity check failed for {role}: {integrity}")
    return records, integrity


def _difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "l1": float(np.abs(delta).sum()),
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def read_only_integrity_report(
    *,
    policy_hash_before: str,
    policy_hash_after: str,
    checkpoint_sha256_before: str,
    checkpoint_sha256_after: str,
) -> dict[str, Any]:
    """Return the immutable-state checks used by the read-only audit."""

    return {
        "policy_hash_before": str(policy_hash_before),
        "policy_hash_after": str(policy_hash_after),
        "checkpoint_sha256_before": str(checkpoint_sha256_before),
        "checkpoint_sha256_after": str(checkpoint_sha256_after),
        "policy_unchanged": str(policy_hash_before) == str(policy_hash_after),
        "checkpoint_unchanged": str(checkpoint_sha256_before)
        == str(checkpoint_sha256_after),
    }


def _pairwise_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_labels: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_role_case = {
        (str(row["role"]), str(row["case"])): row for row in records
    }
    pairs = (
        ("left_front", "right_front", "left_front_vs_right_front"),
        ("left", "right", "left_vs_right"),
        ("front", "behind", "front_vs_behind"),
    )
    difference_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for role in sorted({str(row["role"]) for row in records}):
        for case_a, case_b, pair_name in pairs:
            first = by_role_case[(role, case_a)]
            second = by_role_case[(role, case_b)]
            context_a = np.asarray(first["actor_context"], dtype=float)
            context_b = np.asarray(second["actor_context"], dtype=float)
            node_a = np.asarray(first["normalized_node_features"], dtype=float)
            node_b = np.asarray(second["normalized_node_features"], dtype=float)
            action_a = np.asarray(first["actor_action"]["delta_v_mps"], dtype=float)
            action_b = np.asarray(second["actor_action"]["delta_v_mps"], dtype=float)
            context_diff = _difference(context_a, context_b)
            node_diff = _difference(node_a, node_b)
            action_diff = _difference(action_a, action_b)
            changed_indices = np.flatnonzero(
                np.max(np.abs(node_a - node_b), axis=0) > 1.0e-7
            ).tolist()
            changed_features = [
                str(feature_labels[index])
                for index in changed_indices
                if index < len(feature_labels)
            ]
            difference_rows.append(
                {
                    "role": role,
                    "pair": pair_name,
                    "case_a": case_a,
                    "case_b": case_b,
                    "obstacle_distance_a_m": first["obstacle_center_distance_m"],
                    "obstacle_distance_b_m": second["obstacle_center_distance_m"],
                    "surface_clearance_a_m": first["obstacle_surface_clearance_m"],
                    "surface_clearance_b_m": second["obstacle_surface_clearance_m"],
                    "actor_context_l1": context_diff["l1"],
                    "actor_context_l2": context_diff["l2"],
                    "actor_context_max_abs": context_diff["max_abs"],
                    "normalized_node_l1": node_diff["l1"],
                    "normalized_node_l2": node_diff["l2"],
                    "normalized_node_max_abs": node_diff["max_abs"],
                    "action_l1": action_diff["l1"],
                    "action_l2": action_diff["l2"],
                    "action_max_abs": action_diff["max_abs"],
                    "changed_node_feature_indices": json.dumps(changed_indices),
                    "changed_node_features": json.dumps(changed_features, ensure_ascii=False),
                    "actor_observation_equal_within_tolerance": bool(
                        context_diff["max_abs"] <= 1.0e-7
                    ),
                }
            )
            for index, label in enumerate(feature_labels):
                if index >= node_a.shape[-1] or index >= node_b.shape[-1]:
                    continue
                value_a = float(node_a.reshape(-1, node_a.shape[-1])[0, index])
                value_b = float(node_b.reshape(-1, node_b.shape[-1])[0, index])
                feature_rows.append(
                    {
                        "role": role,
                        "pair": pair_name,
                        "feature_index": index,
                        "feature": str(label),
                        "value_a": value_a,
                        "value_b": value_b,
                        "absolute_difference": abs(value_a - value_b),
                    }
                )
    return difference_rows, feature_rows


def _write_direction_plot(
    *,
    plot_root: Path,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    feature_labels: Sequence[str],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_records = [
        row for row in records if str(row["role"]) == "stage1_static_trained"
    ]
    if not stage_records:
        stage_records = list(records)
    stage_records = sorted(stage_records, key=lambda row: list(dict.fromkeys(
        str(item[0]) for item in PROBE_CASES
    )).index(str(row["case"])))
    matrix = np.asarray(
        [np.asarray(row["normalized_node_features"], dtype=float).reshape(-1) for row in stage_records],
        dtype=float,
    )
    labels = [str(row["case"]) for row in stage_records]
    plot_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True)
    image = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("Actor node-feature index")
    axes[0].set_ylabel("Probe case")
    axes[0].set_title("Stage-1 normalized actor node features")
    figure.colorbar(image, ax=axes[0], shrink=0.85, label="normalized value")
    action_y = [
        float(np.asarray(row["actor_action"]["delta_v_mps"], dtype=float)[1])
        for row in stage_records
    ]
    axes[1].bar(np.arange(len(labels)), action_y, color="#2563eb")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_ylabel("deterministic actor Δv_y (m/s)")
    axes[1].set_title("Actor lateral action under direction-only obstacle changes")
    axes[1].grid(axis="y", alpha=0.25)
    max_pair = max(
        (float(row["actor_context_max_abs"]) for row in pair_rows if row["role"] == "stage1_static_trained"),
        default=0.0,
    )
    figure.suptitle(
        f"Obstacle-direction identifiability probe | max actor-context difference={max_pair:.3g}",
        fontsize=13,
    )
    path = plot_root / "actor_observation_direction_probe.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return str(path.resolve())


def _write_csv_outputs(
    logs_root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> None:
    probe_columns = (
        "role", "case", "checkpoint", "checkpoint_sha256", "policy_hash",
        "obstacle_relative_x_m", "obstacle_relative_y_m", "obstacle_relative_z_m",
        "obstacle_center_distance_m", "obstacle_surface_clearance_m",
        "actor_node_shape", "actor_context_dim", "raw_node_hash",
        "normalized_node_hash", "actor_context_hash", "actor_action_hash",
        "delta_v_x_mps", "delta_v_y_mps", "delta_v_z_mps", "delta_v_norm_mps",
        "measurement_uav_id",
    )
    probe_rows: list[dict[str, Any]] = []
    for row in records:
        relative = list(row["obstacle_relative_position_m"])
        action = row["actor_action"]
        hashes = row["hashes"]
        probe_rows.append(
            {
                "role": row["role"],
                "case": row["case"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "policy_hash": row["policy_hash"],
                "obstacle_relative_x_m": relative[0],
                "obstacle_relative_y_m": relative[1],
                "obstacle_relative_z_m": relative[2],
                "obstacle_center_distance_m": row["obstacle_center_distance_m"],
                "obstacle_surface_clearance_m": row["obstacle_surface_clearance_m"],
                "actor_node_shape": json.dumps(row["actor_node_shape"]),
                "actor_context_dim": row["actor_context_dim"],
                "raw_node_hash": hashes["raw_node_features"],
                "normalized_node_hash": hashes["normalized_node_features"],
                "actor_context_hash": hashes["actor_context"],
                "actor_action_hash": hashes["actor_action_delta_v"],
                "delta_v_x_mps": action["delta_v_mps"][0],
                "delta_v_y_mps": action["delta_v_mps"][1],
                "delta_v_z_mps": action["delta_v_mps"][2],
                "delta_v_norm_mps": action["delta_v_norm_mps"],
                "measurement_uav_id": action["measurement_uav_id"],
            }
        )
    _write_csv(logs_root / "direction_probe.csv", probe_columns, probe_rows)
    _write_csv(
        logs_root / "actor_input_difference_matrix.csv",
        tuple(pair_rows[0].keys()) if pair_rows else (
            "role", "pair", "case_a", "case_b", "actor_context_l1", "actor_context_l2", "actor_context_max_abs"
        ),
        list(pair_rows),
    )
    _write_csv(
        logs_root / "actor_feature_direction_differences.csv",
        tuple(feature_rows[0].keys()) if feature_rows else (
            "role", "pair", "feature_index", "feature", "value_a", "value_b", "absolute_difference"
        ),
        list(feature_rows),
    )


def _feature_table_rows(config: Any, node_feature_dim: int, context_dim: int) -> list[dict[str, Any]]:
    rows = actor_node_feature_layout(node_feature_dim)
    rows.extend(_required_feature_audit_rows(config))
    # FormalExperimentConfig derives the actor node dimension but intentionally
    # does not duplicate MAPPO's graph-edge width.  The runtime MAPPO contract
    # currently uses 11 edge features; honor an explicitly attached value when
    # present and otherwise use that project-level compatibility default.
    edge_feature_dim = int(getattr(config, "edge_feature_dim", 11))
    context_labels: list[str] = []
    context_labels.extend([f"encoder_embedding_{index}" for index in range(int(config.gat_output_dim))])
    for block in ("effective_edge_mean", "candidate_edge_mean", "candidate_incoming_edge_mean", "potential_edge_mean"):
        context_labels.extend([f"{block}_{index}" for index in range(edge_feature_dim)])
    context_labels.extend(["action_mask_0", "action_mask_1", "bandwidth_mask_0"])
    if config.include_remaining_time:
        context_labels.append("remaining_time_fraction")
    for index in range(int(context_dim)):
        label = context_labels[index] if index < len(context_labels) else f"context_feature_{index}"
        rows.append(
            {
                "layer": "actor_body_context",
                "feature": label,
                "index": index,
                "dimension": 1,
                "coordinate_frame": "encoded_or_mask",
                "source": "PolicyNetwork._local_context",
                "present_in_observation": True,
                "enters_actor": True,
                "notes": "final context tensor passed to actor_body",
            }
        )
    return rows


def run_obstacle_observation_audit(
    *,
    stage1_checkpoint: str | Path = DEFAULT_STAGE1_CHECKPOINT,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the read-only runtime observation/direction audit."""

    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    for root in (logs_root, plots_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty audit root: {root.resolve()}")
        root.mkdir(parents=True, exist_ok=True)
    stage1_checkpoint = Path(stage1_checkpoint).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    for checkpoint in (stage1_checkpoint, baseline_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"audit checkpoint does not exist: {checkpoint}")

    scenarios = probe_scenarios()
    # Use a fresh output placeholder only for checkpoint-derived config
    # construction.  No episode data or checkpoint is written there.
    config_payload, _ = _read_checkpoint(stage1_checkpoint)
    config = _config_from_checkpoint(config_payload, output_dir=logs_root, progress=False)
    all_records: list[dict[str, Any]] = []
    checkpoint_integrity: dict[str, Any] = {}
    for role, checkpoint in (
        ("stage1_static_trained", stage1_checkpoint),
        ("baseline", baseline_checkpoint),
    ):
        if progress:
            print(f"[observation-audit] probing {role}", flush=True)
        records, integrity = _probe_one_checkpoint(
            role=role,
            checkpoint=checkpoint,
            output_root=logs_root,
            scenarios=scenarios,
        )
        all_records.extend(records)
        checkpoint_integrity[role] = integrity

    primary = [row for row in all_records if row["role"] == "stage1_static_trained"]
    primary_context_dim = int(primary[0]["actor_context_dim"])
    primary_node_dim = int(np.asarray(primary[0]["normalized_node_features"]).shape[-1])
    labels = [str(row["feature"]) for row in actor_node_feature_layout(primary_node_dim)]
    pair_rows, feature_rows = _pairwise_differences(
        all_records,
        feature_labels=labels,
    )
    _write_csv_outputs(
        logs_root,
        records=all_records,
        pair_rows=pair_rows,
        feature_rows=feature_rows,
    )
    feature_rows_for_table = _feature_table_rows(config, primary_node_dim, primary_context_dim)
    _write_csv(
        logs_root / "actor_feature_table.csv",
        (
            "layer", "feature", "index", "dimension", "coordinate_frame", "source",
            "present_in_observation", "enters_actor", "notes",
        ),
        feature_rows_for_table,
    )
    plot_path = _write_direction_plot(
        plot_root=plots_root,
        records=all_records,
        pair_rows=pair_rows,
        feature_labels=labels,
    )

    primary_pairs = [row for row in pair_rows if row["role"] == "stage1_static_trained"]
    max_context_difference = max(
        (float(row["actor_context_max_abs"]) for row in primary_pairs),
        default=0.0,
    )
    max_node_difference = max(
        (float(row["normalized_node_max_abs"]) for row in primary_pairs),
        default=0.0,
    )
    actor_observation_contract = {
        "actor_node_feature_dim": primary_node_dim,
        "actor_body_context_dim": primary_context_dim,
        "node_feature_layout": "15 estimated EKF state + 15 covariance diagonal + 3 goal-relative + 4 obstacle aggregate + 2 braking descriptors",
        "obstacle_directional_relative_xyz_in_actor": False,
        "obstacle_bearing_in_actor": False,
        "obstacle_absolute_geometry_in_actor": False,
        "static_obstacle_visible_as_aggregate": True,
        "dynamic_obstacle_motion_features_enabled": bool(
            config.dynamic_obstacle_motion_features_enabled
        ),
        "direction_information_lost_by_aggregation": False,
        "direction_information_absent_before_aggregation": True,
        "left_right_identifiable": bool(max_context_difference > 1.0e-7),
        "front_behind_identifiable": bool(
            next(
                (
                    float(row["actor_context_max_abs"]) > 1.0e-7
                    for row in primary_pairs
                    if row["pair"] == "front_vs_behind"
                ),
                False,
            )
        ),
    }
    action_contract = MotionConfig().action_contract()
    summary = {
        "status": "completed_read_only",
        "audit": "single_uav_obstacle_observation_identifiability",
        "probe": {
            "case_count": len(PROBE_CASES),
            "cases": [name for name, _offset in PROBE_CASES],
            "uav_position_m": PROBE_UAV_POSITION.tolist(),
            "goal_m": PROBE_GOAL_POSITION.tolist(),
            "velocity_mps": PROBE_VELOCITY.tolist(),
            "obstacle_center_distance_m": PROBE_OBSTACLE_DISTANCE_M,
            "obstacle_radius_m": PROBE_OBSTACLE_RADIUS_M,
            "surface_clearance_m": PROBE_OBSTACLE_DISTANCE_M - PROBE_OBSTACLE_RADIUS_M,
            "direction_convention": "+y is left when facing +x goal direction",
            "same_uav_goal_velocity_and_obstacle_shape": True,
            "only_obstacle_direction_changes": True,
        },
        "actor_observation_contract": actor_observation_contract,
        "actor_vs_critic_vs_safety_vs_diagnostics": {
            "actor": {
                "sees": [
                    "estimated state and covariance",
                    "relative goal vector",
                    "minimum static-obstacle clearance",
                    "static obstacle type counts",
                    "timestep",
                    "braking target-speed/margin descriptors",
                ],
                "does_not_see": [
                    "obstacle relative x/y/z",
                    "obstacle bearing",
                    "obstacle absolute center/size/identity",
                    "pre-projection swept clearance/conflict flag",
                ],
            },
            "critic": {
                "sees": [
                    "central graph node features",
                    "obstacle table with center, velocity, active/type and size/radius",
                ],
                "note": "central_state_to_vector is critic input, not actor input",
            },
            "safety_layer": {
                "sees": [
                    "full physical obstacle geometry",
                    "predicted/swept candidate motion",
                    "safety distance and bounds",
                ],
                "information_more_than_actor": True,
            },
            "diagnostics": {
                "sees": [
                    "raw candidate segment",
                    "pre-projection swept clearance/conflict/unsafe",
                    "raw versus applied action",
                ],
                "not_forwarded_to_actor": True,
            },
        },
        "direction_probe": {
            "primary_role": "stage1_static_trained",
            "actor_context_pair_max_abs_difference": max_context_difference,
            "normalized_node_pair_max_abs_difference": max_node_difference,
            "pairs": primary_pairs,
            "all_primary_direction_pairs_equal_within_tolerance": all(
                bool(row["actor_observation_equal_within_tolerance"])
                for row in primary_pairs
            ),
        },
        "action_space": {
            **action_contract,
            "lateral_control_theoretically_available": bool(
                action_contract["vector_dimension"] >= 2
                and action_contract["negative_delta_v_allowed"]
            ),
            "single_uav_measurement_action": "none is the only valid measurement choice; no inter-UAV edge",
        },
        "stage1_reward_observation_relation": {
            "stage1_static_reward_source": "raw pre-projection candidate segment / swept clearance",
            "actor_receives_same_swept_clearance": False,
            "actor_receives_risk_direction": False,
            "interpretation": "reward supplies obstacle-risk magnitude during static training, while actor observation lacks the direction needed to choose a bypass side",
        },
        "multi_obstacle_aggregation": {
            "actor": "minimum clearance plus box/sphere counts; no nearest identity or direction-preserving obstacle list",
            "critic": "fixed obstacle table up to max_obstacles with absolute center/velocity/type/size fields",
            "left_and_right_obstacles": "actor aggregate remains unable to preserve which side each obstacle occupies",
        },
        "checkpoint_integrity": checkpoint_integrity,
        "blocker": {
            "classification": "A",
            "statement": "Actor observation does not satisfy active left/right obstacle-bypass identifiability; action space itself supports lateral delta-v",
            "current_static_test_consistency": "大量 raw candidate risk followed by safety projection is consistent with obstacle-risk magnitude being visible but bypass direction being unavailable",
        },
        "minimum_observation_enhancement_recommendation": {
            "do_not_implement_in_this_audit": True,
            "features": [
                "UAV-local relative obstacle dx/dy/dz",
                "obstacle distance or clearance",
                "obstacle size/type embedding",
                "optional relative velocity for dynamic obstacles",
            ],
            "placement": "environment._local_obstacle_features -> graph_interface node features, behind a new opt-in observation-contract/config flag",
            "additional_dimensions_minimum": 4,
            "old_checkpoint_compatibility": "directly adding node dimensions changes GAT input shape; old checkpoints remain compatible only when the flag is disabled or features are padded to the historical dimension",
            "recommended_frame": "UAV-centered local frame, with a documented yaw/body-frame convention; avoid absolute world obstacle coordinates",
            "future_gat": "for multiple obstacles, prefer obstacle nodes or UAV-obstacle edge features carrying relative position, distance, clearance, size/type and relative velocity rather than a single scalar aggregate",
        },
        "files": {
            "actor_feature_table": str((logs_root / "actor_feature_table.csv").resolve()),
            "direction_probe": str((logs_root / "direction_probe.csv").resolve()),
            "actor_input_difference_matrix": str((logs_root / "actor_input_difference_matrix.csv").resolve()),
            "actor_feature_direction_differences": str((logs_root / "actor_feature_direction_differences.csv").resolve()),
            "raw_records": str((logs_root / "observation_probe_records.json").resolve()),
            "plot": plot_path,
        },
    }
    _write_json(logs_root / "observation_probe_records.json", all_records)
    _write_json(logs_root / "observation_audit_summary.json", summary)
    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run_obstacle_observation_audit(
        stage1_checkpoint=args.stage1_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "DEFAULT_BASELINE_CHECKPOINT",
    "DEFAULT_STAGE1_CHECKPOINT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "PROBE_CASES",
    "actor_node_feature_layout",
    "probe_scenarios",
    "read_only_integrity_report",
    "run_obstacle_observation_audit",
]
