"""Hierarchical DAgger waypoint-generalization and two-UAV smoke audits.

This module adds an evaluation-only ``hierarchical-dagger-v1`` protocol.  It
does not replace the direct MAPPO ``delta_v`` action path and it does not
contain an optimizer.  A high-level command is installed as a temporary goal
reference, then a frozen single-UAV DAgger Actor produces the legal low-level
motion increment.  In the two-UAV smoke, a freshly constructed MAPPO actor is
used only to execute the existing communication/resource heads and to record
the GAT trace; its motion output is deliberately discarded in favour of the
scripted waypoint + frozen DAgger path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.mappo import MAPPOTrainer
from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.environment import MultiUAVJointEnv
from ..simulation.motion_model import (
    MotionConfig,
    braking_distance_target_speed,
    stopping_distance_m,
)
from ..simulation.scenario import generate_scenario
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _scenario_for,
    _scale_seed,
    _write_scale_manifest,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_oracle_behavior_cloning import actor_physical_mean
from .single_uav_rl_equivalent_oracle_audit import (
    _extract_rl_visible_state,
    _unit,
)


HIERARCHICAL_PROTOCOL = "hierarchical-dagger-v1"
FROZEN_POLICY_ROLE = "frozen_low_level_dagger_policy"
SINGLE_WAYPOINT_POLICY = "hierarchical_dagger_waypoint"
TWO_UAV_POLICY = "hierarchical_dagger_two_uav_smoke"

WAYPOINT_DISTANCE_BANDS: tuple[tuple[str, float, float, float], ...] = (
    ("2-5m", 2.0, 5.0, 3.5),
    ("5-8m", 5.0, 8.0, 6.5),
    ("8-10m", 8.0, 10.0, 9.0),
)

# The diagnostic directions are expressed in the fixed ENU world frame.  The
# initial attitude is varied independently so the controller cannot rely on a
# single orientation.  The controller itself only receives the goal-relative
# state carried by the Actor observation.
WAYPOINT_DIRECTIONS: tuple[tuple[str, np.ndarray], ...] = (
    ("front", np.asarray((1.0, 0.0, 0.0), dtype=float)),
    ("back", np.asarray((-1.0, 0.0, 0.0), dtype=float)),
    ("left", np.asarray((0.0, 1.0, 0.0), dtype=float)),
    ("right", np.asarray((0.0, -1.0, 0.0), dtype=float)),
    ("up", np.asarray((0.0, 0.0, 1.0), dtype=float)),
    ("down", np.asarray((0.0, 0.0, -1.0), dtype=float)),
    ("front_left", np.asarray((1.0, 1.0, 0.0), dtype=float)),
    ("front_right", np.asarray((1.0, -1.0, 0.0), dtype=float)),
    ("back_left", np.asarray((-1.0, 1.0, 0.0), dtype=float)),
    ("back_right", np.asarray((-1.0, -1.0, 0.0), dtype=float)),
)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean(values: Iterable[Any]) -> float:
    numbers = [number for number in (_finite(value) for value in values) if np.isfinite(number)]
    return float(np.mean(numbers)) if numbers else float("nan")


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                output[column] = value
            writer.writerow(output)


def _copy_for_json(value: Any) -> Any:
    return _jsonable(copy.deepcopy(value))


class FrozenDaggerLowLevelPolicy:
    """A checkpoint-backed DAgger Actor with no optimizer or mutable state."""

    def __init__(self, checkpoint: str | Path, *, output_dir: str | Path) -> None:
        import torch

        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        self.payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(self.payload, Mapping):
            raise ValueError("DAgger checkpoint must contain a mapping payload")
        self.config = _config_from_checkpoint(
            self.payload,
            output_dir=Path(output_dir),
            progress=False,
        )
        self.trainer, self.policy_hashes, _ = _load_frozen_trainer(
            self.payload,
            self.config,
            metric_scales=self.config.metric_scales,
        )
        self.trainer.policy.eval()
        for parameter in self.trainer.policy.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.trainer.policy.parameters()):
            raise AssertionError("frozen DAgger policy has trainable parameters")
        self.optimizer = None
        trainer_state = self.payload.get("trainer_config", {})
        self.node_feature_dim = int(
            trainer_state.get("node_feature_dim", self.config.node_feature_dim)
        )
        self.edge_feature_dim = int(trainer_state.get("edge_feature_dim", 11))
        self.max_delta_v_mps = float(self.trainer.policy.config.max_delta_v)
        self.policy_hash_before = _state_dict_hash(self.trainer.policy.state_dict())
        self.normalizer_hash = _canonical_hash(self.trainer.normalizer.state_dict())

    def action(self, local_observation: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the deterministic Actor-mean physical ``delta_v``.

        Only the provided local observation is read.  In particular, this
        method never accesses an environment, simulator truth, or a mission
        goal outside the observation.
        """

        import torch

        if not isinstance(local_observation, Mapping):
            raise ValueError("frozen DAgger requires a local mapping observation")
        raw = copy.deepcopy(dict(local_observation))
        normalized = self.trainer.normalizer.transform({"uav_00": raw})["uav_00"]
        normalized["uav_id"] = "uav_00"
        with torch.no_grad():
            physical = actor_physical_mean(self.trainer.policy, normalized)
        delta = physical.detach().cpu().numpy().astype(float).reshape(3)
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or norm > self.max_delta_v_mps + 1.0e-6:
            raise AssertionError("frozen DAgger produced an illegal delta_v")
        return delta, {
            "actor_observation_state_source": "estimated_local_observation_only",
            "truth_state_read_for_control": False,
            "normalizer_hash": self.normalizer_hash,
            "policy_hash": self.policy_hash_before,
            "predicted_delta_v_mps": delta.tolist(),
            "predicted_delta_v_norm_mps": norm,
            "max_delta_v_mps": self.max_delta_v_mps,
        }

    def self_only_observation(
        self,
        local_observation: Mapping[str, Any],
        *,
        source_uav_id: str,
    ) -> dict[str, Any]:
        """Project a multi-UAV local graph to the frozen single-UAV contract.

        The low-level checkpoint was trained with one self node.  In the
        two-UAV smoke, GAT receives the full graph at the high level, while
        each independent DAgger instance receives only its own EKF-estimated
        node and the corresponding temporary waypoint.  No neighbour truth is
        fabricated or passed into the low-level Actor.
        """

        if not isinstance(local_observation, Mapping):
            raise ValueError("local observation must be a mapping")
        result = copy.deepcopy(dict(local_observation))
        node = np.asarray(local_observation.get("node_features"), dtype=float)
        if node.ndim != 2 or node.shape[0] < 1:
            raise ValueError("local observation has no node_features")
        node_ids = [str(value) for value in local_observation.get("node_ids", [])]
        own_index = node_ids.index(source_uav_id) if source_uav_id in node_ids else 0
        if own_index >= node.shape[0]:
            raise ValueError("source UAV is absent from local node_features")
        row = node[own_index].reshape(-1)
        if row.size < self.node_feature_dim:
            row = self._append_missing_single_uav_features(row, local_observation)
        if row.size < self.node_feature_dim:
            row = np.pad(row, (0, self.node_feature_dim - row.size))
        elif row.size > self.node_feature_dim:
            row = row[: self.node_feature_dim]
        result["node_features"] = np.asarray([row], dtype=float)
        result["node_ids"] = ["uav_00"]
        result["uav_id"] = "uav_00"
        result["effective_edge_index"] = np.empty((2, 0), dtype=np.int64)
        result["effective_edge_features"] = np.empty((0, self.edge_feature_dim), dtype=float)
        result["candidate_edge_index"] = np.empty((2, 0), dtype=np.int64)
        result["candidate_edge_features"] = {}
        result["potential_edge_index"] = np.empty((2, 0), dtype=np.int64)
        result["potential_edge_features"] = {}
        result["candidate_incoming_edge_features"] = {}
        result["candidate_targets"] = []
        result["candidate_measurement_sources"] = []
        result["candidate_incoming_sources"] = []
        result["effective_sources"] = []
        result["effective_targets"] = []
        result["potential_targets"] = []
        result["neighbor_messages"] = {}
        result["candidate_links"] = {}
        result["candidate_incoming_links"] = {}
        result["effective_links"] = {}
        result["potential_links"] = {}
        result["action_mask"] = np.asarray((False, True), dtype=bool)
        result["bandwidth_mask"] = np.asarray((False,), dtype=bool)
        result["uwb_available"] = {}
        result["uwb_ranges"] = {}
        result["uwb_available_outgoing"] = {}
        result["uwb_ranges_outgoing"] = {}
        return result

    def _append_missing_single_uav_features(
        self,
        row: np.ndarray,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Append the two braking-distance node features absent in a 2-UAV graph."""

        if row.size < 33:
            return row
        estimate = np.asarray(row[:15], dtype=float)
        goal_delta = np.asarray(row[30:33], dtype=float)
        distance = float(np.linalg.norm(goal_delta))
        speed = float(np.linalg.norm(estimate[3:6]))
        reward = self.config
        # Recover the exact physical max speed from the existing motion
        # contract; this is a configuration constant, not simulator truth.
        max_speed = float(MotionConfig().max_speed_mps)
        target = braking_distance_target_speed(
            distance,
            goal_threshold_m=float(reward.single_uav_goal_threshold_m),
            near_goal_radius_m=float(reward.single_uav_near_goal_radius_m),
            stable_speed_mps=float(reward.single_uav_goal_speed_threshold_mps),
            braking_acceleration_mps2=float(reward.single_uav_braking_acceleration_mps2),
            braking_margin_m=float(reward.single_uav_braking_margin_m),
            far_target_speed_mps=float(reward.single_uav_target_speed_far_mps),
            max_speed_mps=max_speed,
            speed_profile=str(reward.single_uav_braking_distance_speed_profile),
        )
        max_stopping_distance = max(
            stopping_distance_m(
                max_speed,
                float(reward.single_uav_goal_speed_threshold_mps),
                float(reward.single_uav_braking_acceleration_mps2),
            ),
            1.0e-9,
        )
        actual_stopping_distance = stopping_distance_m(
            speed,
            float(reward.single_uav_goal_speed_threshold_mps),
            float(reward.single_uav_braking_acceleration_mps2),
        )
        brake_margin = (
            distance
            - float(reward.single_uav_goal_threshold_m)
            - float(reward.single_uav_braking_margin_m)
            - actual_stopping_distance
        )
        return np.concatenate(
            [
                row,
                np.asarray(
                    [
                        np.clip(target / max(max_speed, 1.0e-9), 0.0, 1.0),
                        np.clip(brake_margin / max_stopping_distance, -1.0, 1.0),
                    ],
                    dtype=float,
                ),
            ]
        )

    def audit(self) -> dict[str, Any]:
        return {
            "role": FROZEN_POLICY_ROLE,
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_hash": self.policy_hash_before,
            "observation_normalizer_hash": self.normalizer_hash,
            "uav_count_in_checkpoint": int(self.config.uav_count),
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "action_transform": "Actor mean -> tanh -> max_delta_v/sqrt(3) per component",
            "max_delta_v_mps": self.max_delta_v_mps,
            "stable_criterion": {
                "estimated_goal_distance_m_at_most": float(self.config.single_uav_goal_threshold_m),
                "actual_speed_mps_at_most": float(self.config.single_uav_goal_speed_threshold_mps),
                "consecutive_steps": int(self.config.single_uav_stable_hold_steps),
            },
            "optimizer_present": False,
            "all_policy_parameters_frozen": True,
            "truth_state_read_for_control": False,
        }


def _make_waypoint_scenarios(
    config: FormalExperimentConfig,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Create seeded, direction-balanced, evaluation-only waypoint cases."""

    no_curriculum = replace(
        config,
        scenario_seed=None,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        progress=False,
        force=True,
        command_line=f"{HIERARCHICAL_PROTOCOL} waypoint generalization scenario generation",
    )
    directions = WAYPOINT_DIRECTIONS[:4] if smoke else WAYPOINT_DIRECTIONS
    cases: list[dict[str, Any]] = []
    serial = 0
    for band_name, low, high, nominal_distance in WAYPOINT_DISTANCE_BANDS:
        for direction_name, direction in directions:
            serial += 1
            geometry_seed = 2026 + 97 * serial
            noise_seed = 702026 + 113 * serial
            scenario = _scenario_for(
                no_curriculum,
                geometry_seed,
                split="test",
                episode_index=serial - 1,
                phase="test",
            )
            initial_position = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)
            bounds_min = np.asarray((0.0, 0.0, 2.0), dtype=float)
            bounds_max = np.asarray((200.0, 200.0, 60.0), dtype=float)
            waypoint = np.clip(
                initial_position + nominal_distance * _unit(direction),
                bounds_min + 0.25,
                bounds_max - 0.25,
            )
            # Three initial-state regimes are paired across the direction
            # matrix: rest, small motion toward the waypoint, and small
            # tangential motion.  The state remains within the normal action
            # contract and does not use simulator truth in the controller.
            state_mode = (serial - 1) % 3
            unit_direction = _unit(direction)
            if state_mode == 0:
                initial_velocity = np.zeros(3, dtype=float)
            elif state_mode == 1:
                initial_velocity = 0.25 * unit_direction
            else:
                reference = np.asarray((0.0, 0.0, 1.0), dtype=float)
                if abs(float(np.dot(reference, unit_direction))) > 0.9:
                    reference = np.asarray((1.0, 0.0, 0.0), dtype=float)
                tangent = _unit(reference - float(np.dot(reference, unit_direction)) * unit_direction)
                initial_velocity = 0.18 * tangent
            yaw = float((serial % 8) * np.pi / 4.0)
            scenario["initial_velocities"]["uav_00"] = initial_velocity.tolist()
            scenario["initial_attitudes"] = {"uav_00": [0.0, 0.0, yaw]}
            scenario["hierarchical_waypoint_diagnostic"] = {
                "protocol": HIERARCHICAL_PROTOCOL,
                "waypoint_distance_band": band_name,
                "requested_distance_m": nominal_distance,
                "requested_distance_range_m": [low, high],
                "direction": direction_name,
                "direction_frame": "ENU_world_frame",
                "geometry_seed": geometry_seed,
                "noise_seed": noise_seed,
                "initial_state_mode": ["rest", "toward_waypoint", "tangential"][state_mode],
                "controller_truth_state_read": False,
                "curriculum_used": False,
            }
            cases.append(
                {
                    "case_index": serial,
                    "waypoint_distance_band": band_name,
                    "distance_low_m": low,
                    "distance_high_m": high,
                    "requested_waypoint_distance_m": nominal_distance,
                    "direction": direction_name,
                    "direction_vector": _unit(direction).tolist(),
                    "geometry_seed": geometry_seed,
                    "noise_seed": noise_seed,
                    "scenario": scenario,
                    "scenario_hash": _canonical_hash(scenario),
                    "waypoint": waypoint.tolist(),
                    "initial_state_mode": ["rest", "toward_waypoint", "tangential"][state_mode],
                }
            )
    return cases


def _step_metric_row(
    *,
    case: Mapping[str, Any],
    episode_id: str,
    step: int,
    environment: MultiUAVJointEnv,
    before_visible: Mapping[str, Any],
    after_visible: Mapping[str, Any],
    action: np.ndarray,
    execution: Mapping[str, Any],
    next_info: Mapping[str, Any],
    control: Mapping[str, Any],
    stable_steps: int,
    stable_condition: bool,
) -> dict[str, Any]:
    goal = np.asarray(environment.goals["uav_00"], dtype=float)
    after_position = np.asarray(after_visible["position_m"], dtype=float)
    after_velocity = np.asarray(after_visible["velocity_mps"], dtype=float)
    true_position = np.asarray(environment.states["uav_00"].position_m, dtype=float)
    true_velocity = np.asarray(environment.states["uav_00"].velocity_mps, dtype=float)
    true_goal_delta = goal - true_position
    true_distance = float(np.linalg.norm(true_goal_delta))
    true_direction = _unit(true_goal_delta)
    true_radial = float(np.dot(true_velocity, true_direction))
    true_tangent = true_velocity - true_radial * true_direction
    pre_velocity = np.asarray(before_visible["velocity_mps"], dtype=float)
    velocity_direction = _unit(pre_velocity)
    radial_action = float(np.dot(action, np.asarray(before_visible["goal_direction"], dtype=float)))
    goal_direction = np.asarray(before_visible["goal_direction"], dtype=float)
    tangential_action = action - radial_action * goal_direction
    braking = max(0.0, -float(np.dot(action, velocity_direction)))
    learning = next_info.get("learning_reward_components", {})
    if not isinstance(learning, Mapping):
        learning = {}
    target_speed = _finite(learning.get("single_uav_target_speed_mps"))
    if not np.isfinite(target_speed):
        reward = environment.config.reward
        target_speed = braking_distance_target_speed(
            float(after_visible["distance_m"]),
            goal_threshold_m=float(reward.single_uav_goal_threshold_m),
            near_goal_radius_m=float(reward.single_uav_near_goal_radius_m),
            stable_speed_mps=float(reward.single_uav_goal_speed_threshold_mps),
            braking_acceleration_mps2=float(reward.single_uav_braking_acceleration_mps2),
            braking_margin_m=float(reward.single_uav_braking_margin_m),
            far_target_speed_mps=float(reward.single_uav_target_speed_far_mps),
            max_speed_mps=float(environment.config.motion.max_speed_mps),
            speed_profile=str(reward.single_uav_braking_distance_speed_profile),
        )
    return {
        "policy": SINGLE_WAYPOINT_POLICY,
        "protocol": HIERARCHICAL_PROTOCOL,
        "episode_id": episode_id,
        "case_index": int(case["case_index"]),
        "waypoint_distance_band": str(case["waypoint_distance_band"]),
        "direction": str(case["direction"]),
        "initial_state_mode": str(case["initial_state_mode"]),
        "geometry_seed": int(case["geometry_seed"]),
        "noise_seed": int(case["noise_seed"]),
        "scenario_hash": str(case["scenario_hash"]),
        "step_index": int(step),
        "timestamp_s": float((step + 1) * environment.config.motion.dt_s),
        "waypoint_x_m": float(goal[0]),
        "waypoint_y_m": float(goal[1]),
        "waypoint_z_m": float(goal[2]),
        "estimated_goal_distance_m": float(after_visible["distance_m"]),
        "estimated_goal_distance_before_m": float(before_visible["distance_m"]),
        "estimated_goal_progress_m": float(before_visible["distance_m"] - after_visible["distance_m"]),
        "true_goal_distance_m": true_distance,
        "estimated_speed_mps": float(np.linalg.norm(after_visible["velocity_mps"])),
        "true_speed_mps": float(np.linalg.norm(true_velocity)),
        "radial_velocity_toward_mps": float(np.dot(after_velocity, _unit(np.asarray(after_visible["goal_delta_m"], dtype=float)))),
        "true_radial_velocity_toward_mps": true_radial,
        "tangential_speed_mps": float(np.linalg.norm(after_velocity - float(np.dot(after_velocity, _unit(np.asarray(after_visible["goal_delta_m"], dtype=float)))) * _unit(np.asarray(after_visible["goal_delta_m"], dtype=float)))),
        "true_tangential_speed_mps": float(np.linalg.norm(true_tangent)),
        "target_speed_mps": float(target_speed),
        "desired_delta_v_norm_mps": float(control.get("predicted_delta_v_norm_mps", np.linalg.norm(action))),
        "commanded_delta_v_norm_mps": float(np.linalg.norm(action)),
        "applied_delta_v_norm_mps": _finite(execution.get("delta_v_applied_norm_mps"), np.linalg.norm(action)),
        "radial_action_projection_mps": radial_action,
        "tangential_action_mps": float(np.linalg.norm(tangential_action)),
        "braking_magnitude_mps": braking,
        "applied_braking_magnitude_mps": _finite(execution.get("braking_delta_v_mps"), 0.0),
        "braking_action": int(bool(execution.get("braking_action", 0))),
        "effective_braking": int(bool(execution.get("brake_effective", 0))),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "single_uav_goal_progress_signal": _finite(learning.get("single_uav_goal_progress_signal")),
        "single_uav_goal_progress_reward": _finite(learning.get("single_uav_goal_progress_reward")),
        "single_uav_braking_progress_signal": _finite(learning.get("single_uav_braking_progress_signal")),
        "single_uav_braking_progress_reward": _finite(learning.get("single_uav_braking_progress_reward")),
        "single_uav_target_velocity_action_reward": _finite(learning.get("single_uav_target_velocity_action_reward")),
        "inside_8m": int(float(after_visible["distance_m"]) <= 8.0),
        "inside_5m": int(float(after_visible["distance_m"]) <= 5.0),
        "inside_3m": int(float(after_visible["distance_m"]) <= 3.0),
        "inside_1m": int(float(after_visible["distance_m"]) <= 1.0),
        "stable_condition": int(stable_condition),
        "consecutive_stable_steps": int(stable_steps),
        "truth_state_read_for_control": False,
        "temporary_waypoint_in_observation": True,
        "temporary_waypoint_matches_environment_goal": bool(
            np.allclose(np.asarray(after_visible["goal_delta_m"]) + after_position, goal, atol=1.0e-6)
        ),
        "termination_reason": str(next_info.get("termination_reason", "running")),
    }


def _waypoint_episode_summary(
    case: Mapping[str, Any],
    episode_summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    stable_hold_steps: int,
) -> dict[str, Any]:
    distances = [_finite(row.get("estimated_goal_distance_m")) for row in rows]
    distances = [value for value in distances if np.isfinite(value)]
    speeds = [_finite(row.get("true_speed_mps")) for row in rows]
    speeds = [value for value in speeds if np.isfinite(value)]
    estimated = [_finite(row.get("estimated_goal_distance_m")) for row in rows]
    estimated = [value for value in estimated if np.isfinite(value)]
    stable_values = [int(row.get("consecutive_stable_steps", 0) or 0) for row in rows]
    max_hold = max(stable_values, default=0)

    def first_speed(threshold: float) -> float:
        for row in rows:
            if _finite(row.get("estimated_goal_distance_m")) <= threshold:
                return _finite(row.get("true_speed_mps"))
        return float("nan")

    return {
        "policy": SINGLE_WAYPOINT_POLICY,
        "protocol": HIERARCHICAL_PROTOCOL,
        "episode_id": str(episode_summary.get("episode_id", "")),
        "case_index": int(case["case_index"]),
        "waypoint_distance_band": str(case["waypoint_distance_band"]),
        "direction": str(case["direction"]),
        "initial_state_mode": str(case["initial_state_mode"]),
        "geometry_seed": int(case["geometry_seed"]),
        "noise_seed": int(case["noise_seed"]),
        "scenario_hash": str(case["scenario_hash"]),
        "requested_waypoint_distance_m": float(case["requested_waypoint_distance_m"]),
        "actual_initial_waypoint_distance_m": float(
            _finite(rows[0].get("estimated_goal_distance_before_m")) if rows else float("nan")
        ),
        "steps": len(rows),
        "entered_8m": int(bool(estimated) and min(estimated) <= 8.0),
        "entered_5m": int(bool(estimated) and min(estimated) <= 5.0),
        "entered_3m": int(bool(estimated) and min(estimated) <= 3.0),
        "entered_1m": int(bool(estimated) and min(estimated) <= 1.0),
        "waypoint_reached": int(bool(estimated) and min(estimated) <= 1.0),
        "stable_goal": int(max_hold >= int(stable_hold_steps)),
        "minimum_goal_distance_m": float(min(distances)) if distances else float("nan"),
        "mean_speed_mps": _mean(speeds),
        "mean_speed_within_1m_mps": _mean(
            row.get("true_speed_mps") for row in rows if _finite(row.get("estimated_goal_distance_m")) <= 1.0
        ),
        "mean_speed_at_8m_entry_mps": first_speed(8.0),
        "mean_speed_at_5m_entry_mps": first_speed(5.0),
        "mean_speed_at_3m_entry_mps": first_speed(3.0),
        "mean_speed_at_1m_entry_mps": first_speed(1.0),
        "mean_radial_velocity_toward_mps": _mean(
            row.get("radial_velocity_toward_mps") for row in rows
        ),
        "mean_tangential_speed_mps": _mean(
            row.get("tangential_speed_mps") for row in rows
        ),
        "mean_target_speed_mps": _mean(row.get("target_speed_mps") for row in rows),
        "maximum_consecutive_goal_hold_steps": max_hold,
        "hold_steps_ge_1": int(max_hold >= 1),
        "hold_steps_ge_3": int(max_hold >= 3),
        "hold_steps_ge_5": int(max_hold >= 5),
        "mean_action_norm_mps": _mean(row.get("commanded_delta_v_norm_mps") for row in rows),
        "mean_radial_action_projection_mps": _mean(row.get("radial_action_projection_mps") for row in rows),
        "negative_radial_velocity_fraction": _mean(
            int(_finite(row.get("radial_velocity_toward_mps"), 0.0) < -1.0e-4) for row in rows
        ),
        "mean_tangential_action_mps": _mean(
            row.get("tangential_action_mps") for row in rows
        ),
        "mean_braking_magnitude_mps": _mean(
            row.get("braking_magnitude_mps") for row in rows
        ),
        "mean_applied_braking_magnitude_mps": _mean(
            row.get("applied_braking_magnitude_mps") for row in rows
        ),
        "braking_action_fraction": _mean(row.get("braking_action") for row in rows),
        "effective_braking_fraction": _mean(
            row.get("effective_braking") for row in rows
        ),
        "action_clipped_steps": int(
            sum(int(row.get("action_clipped", 0) or 0) for row in rows)
        ),
        "action_clipping_fraction": _mean(row.get("action_clipped") for row in rows),
        "collision_occurred": bool(episode_summary.get("collision_occurred", False)),
        "timeout": int(str(episode_summary.get("termination_reason", "")) == "time_limit"),
        "termination_reason": str(episode_summary.get("termination_reason", "running")),
    }


def _aggregate_waypoint_metrics(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    groups: list[str] = []
    for row in episode_rows:
        value = str(row.get(group_key, "overall"))
        if value not in groups:
            groups.append(value)
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = [row for row in episode_rows if str(row.get(group_key)) == group]
        count = len(selected)

        def count_field(key: str) -> int:
            return int(sum(int(row.get(key, 0) or 0) for row in selected))

        result: dict[str, Any] = {
            group_key: group,
            "episodes": count,
            "entered_8m_episodes": count_field("entered_8m"),
            "entered_8m_rate": _rate(count_field("entered_8m"), count),
            "entered_5m_episodes": count_field("entered_5m"),
            "entered_5m_rate": _rate(count_field("entered_5m"), count),
            "entered_3m_episodes": count_field("entered_3m"),
            "entered_3m_rate": _rate(count_field("entered_3m"), count),
            "entered_1m_episodes": count_field("entered_1m"),
            "entered_1m_rate": _rate(count_field("entered_1m"), count),
            "waypoint_reached_episodes": count_field("waypoint_reached"),
            "waypoint_reach_rate": _rate(count_field("waypoint_reached"), count),
            "stable_goal_episodes": count_field("stable_goal"),
            "stable_goal_rate": _rate(count_field("stable_goal"), count),
            "minimum_goal_distance_m": min(
                (_finite(row.get("minimum_goal_distance_m")) for row in selected),
                default=float("nan"),
            ),
            "mean_minimum_goal_distance_m": _mean(
                row.get("minimum_goal_distance_m") for row in selected
            ),
            "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in selected),
            "mean_speed_within_1m_mps": _mean(
                row.get("mean_speed_within_1m_mps") for row in selected
            ),
            "mean_speed_at_8m_entry_mps": _mean(
                row.get("mean_speed_at_8m_entry_mps") for row in selected
            ),
            "mean_speed_at_5m_entry_mps": _mean(
                row.get("mean_speed_at_5m_entry_mps") for row in selected
            ),
            "mean_speed_at_3m_entry_mps": _mean(
                row.get("mean_speed_at_3m_entry_mps") for row in selected
            ),
            "mean_speed_at_1m_entry_mps": _mean(
                row.get("mean_speed_at_1m_entry_mps") for row in selected
            ),
            "mean_radial_velocity_toward_mps": _mean(
                row.get("mean_radial_velocity_toward_mps") for row in selected
            ),
            "mean_tangential_speed_mps": _mean(
                row.get("mean_tangential_speed_mps") for row in selected
            ),
            "mean_target_speed_mps": _mean(
                row.get("mean_target_speed_mps") for row in selected
            ),
            "mean_maximum_consecutive_goal_hold_steps": _mean(
                row.get("maximum_consecutive_goal_hold_steps") for row in selected
            ),
            "maximum_consecutive_goal_hold_steps_max": max(
                (int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in selected),
                default=0,
            ),
            "episodes_with_hold_steps_ge_1": count_field("hold_steps_ge_1"),
            "episodes_with_hold_steps_ge_1_rate": _rate(count_field("hold_steps_ge_1"), count),
            "episodes_with_hold_steps_ge_3": count_field("hold_steps_ge_3"),
            "episodes_with_hold_steps_ge_3_rate": _rate(count_field("hold_steps_ge_3"), count),
            "episodes_with_hold_steps_ge_5": count_field("hold_steps_ge_5"),
            "episodes_with_hold_steps_ge_5_rate": _rate(count_field("hold_steps_ge_5"), count),
            "mean_action_norm_mps": _mean(row.get("mean_action_norm_mps") for row in selected),
            "mean_radial_action_projection_mps": _mean(
                row.get("mean_radial_action_projection_mps") for row in selected
            ),
            "negative_radial_velocity_fraction": _mean(
                row.get("negative_radial_velocity_fraction") for row in selected
            ),
            "mean_tangential_action_mps": _mean(
                row.get("mean_tangential_action_mps") for row in selected
            ),
            "mean_braking_magnitude_mps": _mean(
                row.get("mean_braking_magnitude_mps") for row in selected
            ),
            "mean_applied_braking_magnitude_mps": _mean(
                row.get("mean_applied_braking_magnitude_mps") for row in selected
            ),
            "braking_action_fraction": _mean(
                row.get("braking_action_fraction") for row in selected
            ),
            "effective_braking_fraction": _mean(
                row.get("effective_braking_fraction") for row in selected
            ),
            "action_clipped_steps": count_field("action_clipped_steps"),
            "action_clipping_fraction": _mean(
                row.get("action_clipping_fraction") for row in selected
            ),
            "collision_episodes": count_field("collision_occurred"),
            "collision_rate": _rate(count_field("collision_occurred"), count),
            "timeout_episodes": count_field("timeout"),
            "timeout_rate": _rate(count_field("timeout"), count),
        }
        output.append(result)
    return output


def _plot_waypoint_diagnostics(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    plots_root: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    bands = [band[0] for band in WAYPOINT_DISTANCE_BANDS]
    summaries = _aggregate_waypoint_metrics(episode_rows, group_key="waypoint_distance_band")
    by_band = {str(row["waypoint_distance_band"]): row for row in summaries}
    positions = np.arange(len(bands))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for key, label, color in (
        ("waypoint_reach_rate", "enter 1 m", "#2563eb"),
        ("stable_goal_rate", "stable", "#16a34a"),
        ("entered_8m_rate", "enter 8 m", "#7c3aed"),
    ):
        axes[0].bar(
            positions + (list(("waypoint_reach_rate", "stable_goal_rate", "entered_8m_rate")).index(key) - 1) * 0.23,
            [float(by_band.get(band, {}).get(key, 0.0)) for band in bands],
            width=0.22,
            label=label,
            color=color,
        )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("rate")
    axes[0].set_title("Waypoint success by distance")
    axes[0].set_xticks(positions, bands)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        positions - 0.18,
        [float(by_band.get(band, {}).get("mean_minimum_goal_distance_m", np.nan)) for band in bands],
        width=0.36,
        label="mean minimum distance",
        color="#0891b2",
    )
    axes[1].bar(
        positions + 0.18,
        [float(by_band.get(band, {}).get("mean_maximum_consecutive_goal_hold_steps", np.nan)) for band in bands],
        width=0.36,
        label="mean max hold",
        color="#f59e0b",
    )
    axes[1].set_title("Terminal precision and hold")
    axes[1].set_xticks(positions, bands)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    for band in bands:
        values = [
            _finite(row.get("estimated_goal_distance_m"))
            for row in step_rows
            if str(row.get("waypoint_distance_band")) == band
        ]
        values = [value for value in values if np.isfinite(value)]
        axes[2].hist(values, bins=20, alpha=0.5, label=band)
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="goal radius")
    axes[2].set_xlabel("estimated waypoint distance (m)")
    axes[2].set_ylabel("step count")
    axes[2].set_title("Distance distribution")
    axes[2].legend(fontsize=8)
    overview = plots_root / "waypoint_generalization_overview.png"
    figure.savefig(overview, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    colors = {"2-5m": "#2563eb", "5-8m": "#16a34a", "8-10m": "#dc2626"}
    for band in bands:
        selected = [row for row in step_rows if str(row.get("waypoint_distance_band")) == band]
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for row in selected:
            grouped.setdefault(int(row.get("step_index", 0)), []).append(row)
        x = sorted(grouped)
        distance = [_mean(row.get("estimated_goal_distance_m") for row in grouped[index]) for index in x]
        speed = [_mean(row.get("true_speed_mps") for row in grouped[index]) for index in x]
        axes[0].plot(x, distance, label=band, color=colors[band], linewidth=2.0)
        axes[1].plot(x, speed, label=band, color=colors[band], linewidth=2.0)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("estimated distance (m)")
    axes[0].set_title("Frozen DAgger waypoint approach")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="stable speed threshold")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("true speed (m/s)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    approach = plots_root / "waypoint_distance_speed_by_band.png"
    figure.savefig(approach, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"overview": str(overview.resolve()), "distance_speed": str(approach.resolve())}


def _run_single_waypoint_episode(
    *,
    config: FormalExperimentConfig,
    low_level: FrozenDaggerLowLevelPolicy,
    case: Mapping[str, Any],
    scale_root: Path,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    seed = int(case["noise_seed"])
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(
        seed=seed,
        scenario_config=copy.deepcopy(dict(case["scenario"])),
    )
    waypoint = np.asarray(case["waypoint"], dtype=float)
    environment.set_temporary_goal("uav_00", waypoint)
    observations = environment._observations()
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="hierarchical_waypoint_generalization",
        environment=environment,
        normalization_fitted=bool(low_level.trainer.normalizer.fitted),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=SINGLE_WAYPOINT_POLICY,
    )
    rows: list[dict[str, Any]] = []
    stable_steps = 0
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        visible_before = _extract_rl_visible_state(
            observations["uav_00"],
            expected_node_dim=low_level.node_feature_dim,
        )
        delta, control = low_level.action(observations["uav_00"])
        action = {
            "uav_00": {
                "delta_v": delta.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }
        _assert_action_feasible(environment, action)
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        visible_after = _extract_rl_visible_state(
            next_observations["uav_00"],
            expected_node_dim=low_level.node_feature_dim,
        )
        speed = float(np.linalg.norm(environment.states["uav_00"].velocity_mps))
        stable_condition = bool(
            float(visible_after["distance_m"]) <= float(config.single_uav_goal_threshold_m)
            and speed <= float(config.single_uav_goal_speed_threshold_mps)
        )
        stable_steps = stable_steps + 1 if stable_condition else 0
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        row = _step_metric_row(
            case=case,
            episode_id=episode_id,
            step=step,
            environment=environment,
            before_visible=visible_before,
            after_visible=visible_after,
            action=delta,
            execution=execution,
            next_info=next_info,
            control=control,
            stable_steps=stable_steps,
            stable_condition=stable_condition,
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=SINGLE_WAYPOINT_POLICY,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    writer_summary.update(
        {
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "waypoint_case": _copy_for_json(case),
            "truth_state_read_for_control": False,
            "frozen_low_level_checkpoint_sha256": low_level.checkpoint_sha256,
            "frozen_low_level_policy_hash": low_level.policy_hash_before,
        }
    )
    return _waypoint_episode_summary(
        case,
        writer_summary,
        rows,
        stable_hold_steps=int(config.single_uav_stable_hold_steps),
    ), rows


def _write_single_manifest(
    *,
    output_root: Path,
    scale_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    low_level: FrozenDaggerLowLevelPolicy,
    waypoint_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _write_scale_manifest(scale_root=scale_root, config=config, episodes=episode_summaries)
    manifest_path = scale_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "single_uav_hierarchical_waypoint_generalization",
            "protocol": HIERARCHICAL_PROTOCOL,
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "policy_source": FROZEN_POLICY_ROLE,
            "curriculum_used": False,
            "validation_test_curriculum": False,
            "frozen_low_level_policy": low_level.audit(),
            "waypoint_contract": {
                "temporary_goal_replaces_current_goal_reference": True,
                "observation_goal_delta_rebuilt": True,
                "estimated_state_only": True,
                "goal_threshold_m": float(config.single_uav_goal_threshold_m),
                "stable_speed_threshold_mps": float(config.single_uav_goal_speed_threshold_mps),
                "stable_hold_steps": int(config.single_uav_stable_hold_steps),
                "direction_frame": "ENU_world_frame",
                "distance_bands_m": [[band[1], band[2]] for band in WAYPOINT_DISTANCE_BANDS],
            },
            "waypoint_summary": _copy_for_json(waypoint_summary),
            "read_only_contract": [
                "no MAPPOTrainer.update call",
                "no PPO/DAgger optimizer",
                "no reward or termination modification",
                "no direct position or velocity assignment by controller",
                "no simulator truth read for low-level control",
            ],
        }
    )
    _write_json(manifest_path, _jsonable(manifest))
    root_manifest = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "dataset_id": "single_uav_hierarchical_waypoint_generalization",
        "protocol": HIERARCHICAL_PROTOCOL,
        "scale": "scale_1",
        "scale_manifest": str(manifest_path.resolve()),
        "evaluation_only": True,
        "frozen_low_level_checkpoint": low_level.audit(),
        "summary": _copy_for_json(waypoint_summary),
    }
    _write_json(output_root / "manifest.json", root_manifest)
    return manifest


def _write_single_summary(
    *,
    output_root: Path,
    logs_root: Path,
    plots_root: Path,
    config: FormalExperimentConfig,
    cases: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    low_level: FrozenDaggerLowLevelPolicy,
    plot_paths: Mapping[str, str],
    smoke: bool,
) -> dict[str, Any]:
    by_band = _aggregate_waypoint_metrics(episode_rows, group_key="waypoint_distance_band")
    by_direction = _aggregate_waypoint_metrics(episode_rows, group_key="direction")
    overall_rows = [dict(row, waypoint_distance_band="overall") for row in episode_rows]
    overall = _aggregate_waypoint_metrics(overall_rows, group_key="waypoint_distance_band")[0]
    summary = {
        "dataset_id": "single_uav_hierarchical_waypoint_generalization",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "protocol": HIERARCHICAL_PROTOCOL,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "smoke": bool(smoke),
        "frozen_low_level_policy": low_level.audit(),
        "experiment_config": _copy_for_json(config.__dict__),
        "design": {
            "waypoint_cases": len(cases),
            "directions": [name for name, _ in WAYPOINT_DIRECTIONS[:4] if smoke] if smoke else [name for name, _ in WAYPOINT_DIRECTIONS],
            "distance_bands_m": [[band[1], band[2]] for band in WAYPOINT_DISTANCE_BANDS],
            "scenario_seeds": sorted({int(case["geometry_seed"]) for case in cases}),
            "noise_seeds": sorted({int(case["noise_seed"]) for case in cases}),
            "curriculum_used": False,
            "train_data_used": False,
        },
        "aggregate": overall,
        "by_distance_band": by_band,
        "by_direction": by_direction,
        "thresholds": {
            "reach_rate_at_least": 0.95,
            "stable_rate_at_least": 0.90,
            "reach_pass": bool(float(overall.get("waypoint_reach_rate", 0.0)) >= 0.95),
            "stable_pass": bool(float(overall.get("stable_goal_rate", 0.0)) >= 0.90),
        },
        "plots": dict(plot_paths),
        "outputs": {
            "data": str(output_root.resolve()),
            "logs": str(logs_root.resolve()),
            "plots": str(plots_root.resolve()),
            "episode_metrics": str((logs_root / "waypoint_episode_metrics.csv").resolve()),
            "step_metrics": str((logs_root / "waypoint_step_metrics.csv").resolve()),
        },
        "claim_boundary": "evaluation-only waypoint generalization audit; not an RL training result",
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(logs_root / "summary.json", _jsonable(summary))
    _write_rows_csv(logs_root / "waypoint_episode_metrics.csv", episode_rows)
    _write_rows_csv(logs_root / "waypoint_step_metrics.csv", step_rows)
    _write_rows_csv(logs_root / "waypoint_by_distance_band.csv", by_band)
    _write_rows_csv(logs_root / "waypoint_by_direction.csv", by_direction)
    return summary


def _make_two_uav_environment(
    config: FormalExperimentConfig,
    *,
    seed: int,
    max_steps: int,
) -> MultiUAVJointEnv:
    one_uav = _make_environment(config, seed, metric_scales=config.metric_scales)
    two_config = replace(one_uav.config, uav_count=2, max_steps=int(max_steps), seed=int(seed))
    return MultiUAVJointEnv(two_config)


def _two_uav_scenario(config: FormalExperimentConfig, seed: int) -> dict[str, Any]:
    scenario = generate_scenario(
        int(seed),
        2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    scenario["hierarchical_two_uav_smoke"] = {
        "protocol": HIERARCHICAL_PROTOCOL,
        "geometry_seed": int(seed),
        "noise_seed": int(seed + 100000),
        "curriculum_used": False,
        "high_level_waypoint_source": "scripted_waypoint_sequence",
    }
    return scenario


def _two_uav_self_only_observation(
    low_level: FrozenDaggerLowLevelPolicy,
    local_observation: Mapping[str, Any],
    uav_id: str,
) -> dict[str, Any]:
    return low_level.self_only_observation(local_observation, source_uav_id=uav_id)


def _run_two_uav_smoke(
    *,
    config: FormalExperimentConfig,
    low_level_checkpoint: str | Path,
    output_root: Path,
    logs_root: Path,
    plots_root: Path,
    force: bool,
    progress: bool,
) -> dict[str, Any]:
    import torch

    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"two-UAV smoke output is not empty: {output_root}")
    if force and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    for root in (logs_root, plots_root):
        if force and root.exists():
            import shutil

            shutil.rmtree(root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    scale_root = output_root / "scale_2"
    _copy_calibration_files(scale_root, config)
    geometry_seed = 812026
    noise_seed = 912026
    scenario = _two_uav_scenario(config, geometry_seed)
    environment = _make_two_uav_environment(config, seed=noise_seed, max_steps=30)
    observations, info = environment.reset(seed=noise_seed, scenario_config=copy.deepcopy(scenario))
    initial_estimates = {
        uav_id: np.asarray(observations[uav_id]["estimate"], dtype=float).copy()
        for uav_id in environment.uav_ids
    }
    waypoints = {
        "uav_00": initial_estimates["uav_00"][:3] + np.asarray((4.0, 0.0, 0.0)),
        "uav_01": initial_estimates["uav_01"][:3] + np.asarray((-4.0, 0.0, 0.0)),
    }
    for uav_id, waypoint in waypoints.items():
        environment.set_temporary_goal(uav_id, waypoint)
    observations = environment._observations()
    low_policies = {
        uav_id: FrozenDaggerLowLevelPolicy(low_level_checkpoint, output_dir=output_root / "frozen")
        for uav_id in environment.uav_ids
    }
    first_low = low_policies[environment.uav_ids[0]]
    checkpoint_hashes_before = {
        uav_id: policy.checkpoint_sha256 for uav_id, policy in low_policies.items()
    }
    policy_hashes_before = {
        uav_id: policy.policy_hash_before for uav_id, policy in low_policies.items()
    }
    high_level = MAPPOTrainer.from_environment(
        environment,
        seed=2026,
        node_feature_dim=int(observations["uav_00"]["node_features"].shape[1]),
        edge_feature_dim=11,
        gat_hidden_dim=int(first_low.trainer.config.gat_hidden_dim),
        gat_output_dim=int(first_low.trainer.config.gat_output_dim),
        gat_heads=int(first_low.trainer.config.gat_heads),
        gat_layers=int(first_low.trainer.config.gat_layers),
        actor_hidden_dim=int(first_low.trainer.config.actor_hidden_dim),
        critic_hidden_dim=int(first_low.trainer.config.critic_hidden_dim),
        max_obstacles=int(first_low.trainer.config.max_obstacles),
        max_delta_v=float(first_low.trainer.config.max_delta_v),
        include_remaining_time=False,
    )
    high_level.policy.eval()
    for parameter in high_level.policy.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in high_level.policy.parameters()):
        raise AssertionError("two-UAV smoke high-level diagnostic actor is not frozen")
    # MAPPOTrainer historically creates an optimizer during construction. The
    # smoke never updates this diagnostic high-level actor, so discard that
    # unused object before the rollout and report its status explicitly.
    high_level_optimizer_created = getattr(high_level, "optimizer", None) is not None
    high_level.optimizer = None
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / "episode_000001",
        episode_id="episode_000001",
        scenario_id="scenario_000001",
        split="test",
        seed=noise_seed,
        phase="hierarchical_two_uav_smoke",
        environment=environment,
        normalization_fitted=False,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=TWO_UAV_POLICY,
    )
    step_rows: list[dict[str, Any]] = []
    hold = {uav_id: 0 for uav_id in environment.uav_ids}
    communication_preserved = True
    legal_delta_v = True
    local_estimate_ids: list[dict[str, Any]] = []
    for step in range(environment.config.max_steps):
        _assert_observation_finite(observations)
        high_decision = high_level.act(observations, info.get("central_state"), deterministic=True)
        actions: dict[str, dict[str, Any]] = {}
        per_uav_control: dict[str, Any] = {}
        for uav_id in environment.uav_ids:
            high_action = copy.deepcopy(high_decision.actions[uav_id])
            comm_measurement = high_action.get("measurement_uav_id", "none")
            comm_bandwidth = copy.deepcopy(high_action.get("bandwidth", {}))
            low_observation = _two_uav_self_only_observation(
                low_policies[uav_id], observations[uav_id], uav_id
            )
            delta, control = low_policies[uav_id].action(low_observation)
            if np.linalg.norm(delta) > low_policies[uav_id].max_delta_v_mps + 1.0e-6:
                legal_delta_v = False
            composed = {
                "delta_v": delta.tolist(),
                "measurement_uav_id": comm_measurement,
                "bandwidth": comm_bandwidth,
            }
            communication_preserved &= bool(
                composed["measurement_uav_id"] == comm_measurement
                and composed["bandwidth"] == comm_bandwidth
            )
            actions[uav_id] = composed
            per_uav_control[uav_id] = control
            local_estimate_ids.append(
                {
                    "step": step,
                    "uav_id": uav_id,
                    "observation_uav_id": str(observations[uav_id].get("uav_id", uav_id)),
                    "self_node_id": str(_two_uav_self_only_observation(low_policies[uav_id], observations[uav_id], uav_id)["node_ids"][0]),
                    "estimated_position": np.asarray(observations[uav_id]["estimate"][:3], dtype=float).tolist(),
                }
            )
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        for uav_id in environment.uav_ids:
            visible_after = _extract_rl_visible_state(
                _two_uav_self_only_observation(low_policies[uav_id], next_observations[uav_id], uav_id),
                expected_node_dim=low_policies[uav_id].node_feature_dim,
            )
            speed = float(np.linalg.norm(environment.states[uav_id].velocity_mps))
            stable_condition = bool(
                visible_after["distance_m"] <= float(config.single_uav_goal_threshold_m)
                and speed <= float(config.single_uav_goal_speed_threshold_mps)
            )
            hold[uav_id] = hold[uav_id] + 1 if stable_condition else 0
            step_rows.append(
                {
                    "protocol": HIERARCHICAL_PROTOCOL,
                    "policy": TWO_UAV_POLICY,
                    "episode_id": "episode_000001",
                    "step_index": step,
                    "uav_id": uav_id,
                    "temporary_waypoint": waypoints[uav_id].tolist(),
                    "estimated_goal_distance_m": float(visible_after["distance_m"]),
                    "speed_mps": speed,
                    "stable_condition": int(stable_condition),
                    "consecutive_stable_steps": hold[uav_id],
                    "delta_v_mps": actions[uav_id]["delta_v"],
                    "delta_v_norm_mps": float(np.linalg.norm(actions[uav_id]["delta_v"])),
                    "measurement_uav_id": actions[uav_id]["measurement_uav_id"],
                    "bandwidth": actions[uav_id]["bandwidth"],
                    "high_level_motion_action_discarded": True,
                    "communication_action_preserved": True,
                    "local_ekf_instance": uav_id,
                    "dagger_instance": uav_id,
                    "truth_state_read_for_control": False,
                }
            )
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=high_decision.graph_traces,
            policy_source=TWO_UAV_POLICY,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    checkpoint_after = {uav_id: _sha256(policy.checkpoint) for uav_id, policy in low_policies.items()}
    policy_after = {uav_id: _state_dict_hash(policy.trainer.policy.state_dict()) for uav_id, policy in low_policies.items()}
    high_level_parameter_hash = _state_dict_hash(high_level.policy.state_dict())
    per_uav_summary: dict[str, Any] = {}
    for uav_id in environment.uav_ids:
        selected = [row for row in step_rows if row["uav_id"] == uav_id]
        distances = [float(row["estimated_goal_distance_m"]) for row in selected]
        max_hold = max((int(row["consecutive_stable_steps"]) for row in selected), default=0)
        per_uav_summary[uav_id] = {
            "entered_8m": int(bool(distances) and min(distances) <= 8.0),
            "entered_1m": int(bool(distances) and min(distances) <= 1.0),
            "stable_goal": int(max_hold >= int(config.single_uav_stable_hold_steps)),
            "minimum_goal_distance_m": min(distances) if distances else float("nan"),
            "maximum_consecutive_goal_hold_steps": max_hold,
            "hold_steps_ge_1": int(max_hold >= 1),
            "hold_steps_ge_3": int(max_hold >= 3),
            "hold_steps_ge_5": int(max_hold >= 5),
            "mean_speed_mps": _mean(row["speed_mps"] for row in selected),
        }
    manifest_payload = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "dataset_id": "two_uav_hierarchical_dagger_smoke",
        "protocol": HIERARCHICAL_PROTOCOL,
        "uav_count": 2,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "curriculum_used": False,
        "validation_test_curriculum": False,
        "high_level_controller": "scripted_waypoint_sequence",
        "communication_controller": "frozen_diagnostic_mappo_action_heads",
        "mappo_motion_action_discarded": True,
        "local_ekf_independent": True,
        "gat_neighbor_graph_used": True,
        "communication_action_preserved": communication_preserved,
        "legal_delta_v": legal_delta_v,
        "low_level_checkpoint_hashes_before": checkpoint_hashes_before,
        "low_level_checkpoint_hashes_after": checkpoint_after,
        "low_level_policy_hashes_before": policy_hashes_before,
        "low_level_policy_hashes_after": policy_after,
        "low_level_checkpoint_unchanged": checkpoint_hashes_before == checkpoint_after,
        "low_level_policy_unchanged": policy_hashes_before == policy_after,
        "high_level_policy_parameter_hash": high_level_parameter_hash,
        "high_level_optimizer_created": high_level_optimizer_created,
        "high_level_optimizer_present_after_setup": getattr(high_level, "optimizer", None) is not None,
        "high_level_optimizer_used": False,
        "truth_state_read_for_control": False,
        "writer_summary": writer_summary,
        "per_uav_summary": per_uav_summary,
        "local_estimate_identity_records": local_estimate_ids,
        "graph_trace_contract": {
            "physical_neighbor_features": True,
            "attention_trace_recorded": True,
            "communication_action_not_overwritten_by_dagger": communication_preserved,
        },
        "episodes": [writer_summary],
    }
    _write_json(scale_root / "manifest.json", _jsonable(manifest_payload))
    _write_json(output_root / "manifest.json", _jsonable(manifest_payload))
    _write_rows_csv(logs_root / "two_uav_step_metrics.csv", step_rows)
    _write_json(
        logs_root / "two_uav_summary.json",
        _jsonable(
            {
                "protocol": HIERARCHICAL_PROTOCOL,
                "policy": TWO_UAV_POLICY,
                "episodes": 1,
                "per_uav": per_uav_summary,
                "communication_action_preserved": communication_preserved,
                "legal_delta_v": legal_delta_v,
                "low_level_checkpoint_unchanged": checkpoint_hashes_before == checkpoint_after,
                "low_level_policy_unchanged": policy_hashes_before == policy_after,
                "gat_neighbor_graph_used": True,
                "local_ekf_independent": True,
            }
        ),
    )
    validation = validate_episode_benchmark(scale_root)
    result = {
        "dataset_id": "two_uav_hierarchical_dagger_smoke",
        "protocol": HIERARCHICAL_PROTOCOL,
        "evaluation_only": True,
        "training_performed": False,
        "schema_validation": validation,
        "writer_summary": writer_summary,
        "per_uav_summary": per_uav_summary,
        "communication_action_preserved": communication_preserved,
        "legal_delta_v": legal_delta_v,
        "local_ekf_independent": True,
        "gat_neighbor_graph_used": True,
        "low_level_checkpoint_unchanged": checkpoint_hashes_before == checkpoint_after,
        "low_level_policy_unchanged": policy_hashes_before == policy_after,
        "high_level_optimizer_created": high_level_optimizer_created,
        "high_level_optimizer_present_after_setup": getattr(high_level, "optimizer", None) is not None,
        "high_level_optimizer_used": False,
        "outputs": {
            "data": str(output_root.resolve()),
            "logs": str(logs_root.resolve()),
            "plots": str(plots_root.resolve()),
        },
    }
    _write_json(output_root / "summary.json", _jsonable(result))
    if progress:
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), flush=True)
    return result


def run_hierarchical_dagger_waypoint_audit(
    *,
    checkpoint: str | Path = "data/synthetic/single_uav_dagger_actor_seed2026/scale_1/checkpoints/best_validation.pt",
    output: str | Path = "data/synthetic/single_uav_hierarchical_waypoint_generalization_seed2026",
    logs_output: str | Path | None = None,
    plots_output: str | Path | None = None,
    smoke: bool = False,
    force: bool = False,
    progress: bool = True,
    run_two_uav: bool = True,
    two_uav_output: str | Path = "data/synthetic/two_uav_hierarchical_dagger_smoke_seed2026",
    two_uav_logs_output: str | Path | None = None,
    two_uav_plots_output: str | Path | None = None,
) -> dict[str, Any]:
    """Run single-UAV waypoint generalization, then conditionally 2-UAV smoke."""

    output_root = Path(output)
    logs_root = Path(logs_output) if logs_output is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_output) if plots_output is not None else Path("plots/preliminary") / output_root.name
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"waypoint audit output is not empty: {output_root}")
    if force:
        import shutil

        for root in (output_root, logs_root, plots_root):
            if root.exists():
                shutil.rmtree(root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    low_level = FrozenDaggerLowLevelPolicy(checkpoint, output_dir=output_root / "frozen")
    if low_level.config.uav_count != 1:
        raise ValueError("hierarchical waypoint audit requires a single-UAV DAgger checkpoint")
    if low_level.config.single_uav_stable_hold_steps < 1:
        raise ValueError("stable_hold_steps must be positive")
    cases = _make_waypoint_scenarios(low_level.config, smoke=smoke)
    scale_root = output_root / "scale_1"
    _copy_calibration_files(scale_root, low_level.config)
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    for serial, case in enumerate(cases, start=1):
        if progress:
            print(
                f"[{HIERARCHICAL_PROTOCOL}] waypoint case {serial}/{len(cases)} "
                f"{case['waypoint_distance_band']} {case['direction']}",
                flush=True,
            )
        summary, rows = _run_single_waypoint_episode(
            config=low_level.config,
            low_level=low_level,
            case=case,
            scale_root=scale_root,
            serial=serial,
        )
        episode_rows.append(summary)
        step_rows.extend(rows)
        episode_summaries.append(
            {
                **summary,
                "split": "test",
                "scenario_id": f"scenario_{serial:06d}",
                "seed": int(case["noise_seed"]),
                "scenario_hash": str(case["scenario_hash"]),
            }
        )
    plot_paths = _plot_waypoint_diagnostics(episode_rows, step_rows, plots_root)
    waypoint_summary = _write_single_summary(
        output_root=output_root,
        logs_root=logs_root,
        plots_root=plots_root,
        config=low_level.config,
        cases=cases,
        episode_rows=episode_rows,
        step_rows=step_rows,
        low_level=low_level,
        plot_paths=plot_paths,
        smoke=smoke,
    )
    _write_single_manifest(
        output_root=output_root,
        scale_root=scale_root,
        config=low_level.config,
        episode_summaries=episode_summaries,
        low_level=low_level,
        waypoint_summary=waypoint_summary,
    )
    single_validation = validate_episode_benchmark(scale_root)
    waypoint_summary["schema_validation"] = single_validation
    waypoint_summary["checkpoint_sha256_after"] = _sha256(low_level.checkpoint)
    waypoint_summary["policy_hash_after"] = _state_dict_hash(low_level.trainer.policy.state_dict())
    waypoint_summary["checkpoint_unchanged"] = waypoint_summary["checkpoint_sha256_after"] == low_level.checkpoint_sha256
    waypoint_summary["policy_unchanged"] = waypoint_summary["policy_hash_after"] == low_level.policy_hash_before
    _write_json(output_root / "summary.json", _jsonable(waypoint_summary))
    _write_json(logs_root / "summary.json", _jsonable(waypoint_summary))

    single_pass = bool(
        waypoint_summary["thresholds"]["reach_pass"]
        and waypoint_summary["thresholds"]["stable_pass"]
    )
    two_uav_result: dict[str, Any] | None = None
    if run_two_uav and single_pass and not smoke:
        two_output = Path(two_uav_output)
        two_logs = Path(two_uav_logs_output) if two_uav_logs_output is not None else Path("logs/preliminary") / two_output.name
        two_plots = Path(two_uav_plots_output) if two_uav_plots_output is not None else Path("plots/preliminary") / two_output.name
        two_uav_result = _run_two_uav_smoke(
            config=low_level.config,
            low_level_checkpoint=low_level.checkpoint,
            output_root=two_output,
            logs_root=two_logs,
            plots_root=two_plots,
            force=force,
            progress=progress,
        )
    elif run_two_uav and not single_pass:
        two_uav_result = {
            "skipped": True,
            "reason": "single-UAV waypoint generalization did not meet reach/stable development thresholds",
        }
        _write_json(output_root / "two_uav_skipped.json", _jsonable(two_uav_result))

    final = {
        "protocol": HIERARCHICAL_PROTOCOL,
        "single_uav_waypoint": waypoint_summary,
        "single_uav_waypoint_pass": single_pass,
        "two_uav_smoke": two_uav_result,
        "next_step": (
            "ready_for_two_uav_mappo_training_after_review"
            if single_pass and isinstance(two_uav_result, Mapping) and not two_uav_result.get("skipped", False) and bool(two_uav_result.get("schema_validation", {}).get("ok", False))
            else "train_goal_conditioned_randomized_goal_dagger_before_hierarchical_mappo"
        ),
    }
    _write_json(output_root / "hierarchical_summary.json", _jsonable(final))
    if progress:
        print(json.dumps(_jsonable(final), ensure_ascii=False, indent=2), flush=True)
    return final


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation-only hierarchical DAgger waypoint audit")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/synthetic/single_uav_dagger_actor_seed2026/scale_1/checkpoints/best_validation.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/single_uav_hierarchical_waypoint_generalization_seed2026"))
    parser.add_argument("--logs-output", type=Path, default=None)
    parser.add_argument("--plots-output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-two-uav", action="store_true")
    parser.add_argument("--two-uav-output", type=Path, default=Path("data/synthetic/two_uav_hierarchical_dagger_smoke_seed2026"))
    parser.add_argument("--two-uav-logs-output", type=Path, default=None)
    parser.add_argument("--two-uav-plots-output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_hierarchical_dagger_waypoint_audit(
        checkpoint=args.checkpoint,
        output=args.output,
        logs_output=args.logs_output,
        plots_output=args.plots_output,
        smoke=bool(args.smoke),
        force=bool(args.force),
        progress=not args.quiet,
        run_two_uav=not bool(args.skip_two_uav),
        two_uav_output=args.two_uav_output,
        two_uav_logs_output=args.two_uav_logs_output,
        two_uav_plots_output=args.two_uav_plots_output,
    )
    single = result.get("single_uav_waypoint", {})
    return 0 if bool(single.get("schema_validation", {}).get("ok", False)) else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke
    raise SystemExit(main())


__all__ = [
    "FROZEN_POLICY_ROLE",
    "HIERARCHICAL_PROTOCOL",
    "FrozenDaggerLowLevelPolicy",
    "WAYPOINT_DIRECTIONS",
    "WAYPOINT_DISTANCE_BANDS",
    "run_hierarchical_dagger_waypoint_audit",
]
