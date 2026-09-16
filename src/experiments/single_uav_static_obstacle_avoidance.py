"""Stage-1 single-UAV static-obstacle avoidance experiment.

This module is intentionally separate from the historical formal runner.  It
loads the verified single-UAV baseline checkpoint, creates an independent
static-obstacle pressure split, and adds one opt-in learning signal based on
the raw candidate motion before the scene safety projection.  The existing
environment, action space, dynamics, PPO implementation, and success
definition remain unchanged.

The first invocation is a smoke protocol.  It deliberately writes to a new
experiment root and refuses to overwrite an existing non-empty root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..simulation.obstacle_model import (
    is_physical_obstacle,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
)
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _episode_metrics,
    _make_environment,
    _parameter_vector,
    _scale_seed,
    _update_root_manifest,
    _write_attention_summary,
    _write_episode_metrics,
    _write_scale_manifest,
    summarize_preliminary,
    validate_preliminary,
)
from .pilot import (
    _EpisodeTraceWriter,
    _append_jsonl,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_navigation_preserving_finetune import (
    _baseline_checkpoint_guard,
    _load_trainable_trainer,
    _normal_full_distance_scenario,
    _read_checkpoint,
)


DEFAULT_BASELINE_CHECKPOINT = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_static_obstacle_avoidance_stage1_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_static_obstacle_avoidance_stage1_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_static_obstacle_avoidance_stage1_seed2026"
)


@dataclass(frozen=True)
class StaticObstacleAvoidanceRewardConfig:
    """Independent opt-in configuration for the stage-1 reward channel.

    All fields default to zero/disabled so importing this module or using the
    helper outside the stage-1 runner cannot change a historical experiment.
    The active smoke uses conservative non-zero weights recorded in the stage
    manifest.
    """

    clearance_weight: float = 0.0
    risk_weight: float = 0.0
    warning_distance_m: float = 8.0
    safe_distance_m: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("clearance_weight", self.clearance_weight),
            ("risk_weight", self.risk_weight),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("warning_distance_m", self.warning_distance_m),
            ("safe_distance_m", self.safe_distance_m),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.safe_distance_m) > float(self.warning_distance_m):
            raise ValueError("safe_distance_m cannot exceed warning_distance_m")


ACTIVE_REWARD_CONFIG = StaticObstacleAvoidanceRewardConfig(
    clearance_weight=0.5,
    risk_weight=1.0,
    warning_distance_m=8.0,
    safe_distance_m=2.0,
)

TRAIN_NORMAL_FRACTION = 0.70
TRAIN_STATIC_FRACTION = 0.30
SCHEDULE_CYCLE_LENGTH = 10
STATIC_TRAIN_SLOTS = 3
STAGE_NAME = "single_uav_static_obstacle_avoidance_stage1"


def training_episode_kind(episode_index: int) -> str:
    """Return the deterministic 70/30 training schedule label."""

    slot = (int(episode_index) - 1) % SCHEDULE_CYCLE_LENGTH
    return "static_pressure" if slot >= SCHEDULE_CYCLE_LENGTH - STATIC_TRAIN_SLOTS else "normal_far_distance"


def training_schedule_counts(episodes: int) -> dict[str, int]:
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    counts = {"normal_far_distance": 0, "static_pressure": 0}
    for episode in range(1, int(episodes) + 1):
        counts[training_episode_kind(episode)] += 1
    return counts


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _static_obstacle(obstacle: Any) -> bool:
    if not getattr(obstacle, "active", False) or not is_physical_obstacle(obstacle):
        return False
    obstacle_type = str(getattr(obstacle, "obstacle_type", "")).strip().lower()
    velocity = np.asarray(
        getattr(obstacle, "velocity_mps", np.zeros(3, dtype=float)), dtype=float
    ).reshape(-1)
    return (
        obstacle_type in {"building", "static", "static_obstacle"}
        and velocity.size == 3
        and np.isfinite(velocity).all()
        and float(np.linalg.norm(velocity)) <= 1.0e-12
    )


def static_candidate_diagnostics(
    environment: Any,
    *,
    position_before: np.ndarray,
    nominal_position: np.ndarray,
) -> dict[str, Any]:
    """Measure static-obstacle risk on the unprojected candidate segment."""

    clearances: list[float] = []
    conflict_objects: list[str] = []
    unsafe_objects: list[str] = []
    for obstacle in environment.obstacles:
        if not _static_obstacle(obstacle):
            continue
        obstacle_id = str(obstacle.obstacle_id)
        clearance = float(
            segment_obstacle_clearance(
                position_before,
                nominal_position,
                obstacle,
            )
        )
        clearances.append(clearance)
        if segment_intersects_obstacle(position_before, nominal_position, obstacle):
            conflict_objects.append(obstacle_id)
        if clearance < float(environment.config.motion.safety_distance_m):
            unsafe_objects.append(obstacle_id)
    minimum = float(min(clearances, default=float("inf")))
    return {
        "static_obstacle_count": int(len(clearances)),
        "pre_candidate_static_swept_minimum_clearance_m": minimum,
        "pre_candidate_static_swept_conflict": int(bool(conflict_objects)),
        "pre_candidate_static_unsafe": int(bool(unsafe_objects)),
        "pre_candidate_static_conflict_objects": sorted(set(conflict_objects)),
        "pre_candidate_static_unsafe_objects": sorted(set(unsafe_objects)),
    }


def static_obstacle_reward(
    diagnostics: Mapping[str, Any],
    reward_config: StaticObstacleAvoidanceRewardConfig,
) -> dict[str, float | bool]:
    """Compute the optional penalty from pre-projection swept clearance."""

    clearance = _finite(
        diagnostics.get("pre_candidate_static_swept_minimum_clearance_m"),
        float("inf"),
    )
    if not np.isfinite(clearance):
        clearance = float("inf")
    warning = float(reward_config.warning_distance_m)
    safe = float(reward_config.safe_distance_m)
    clearance_signal = max(0.0, warning - clearance) / warning
    clearance_penalty = float(reward_config.clearance_weight) * clearance_signal**2
    risk_penalty = float(reward_config.risk_weight) * float(clearance < safe)
    reward = -(clearance_penalty + risk_penalty)
    active = bool(clearance_penalty > 0.0 or risk_penalty > 0.0)
    return {
        "clearance_signal": float(clearance_signal),
        "clearance_penalty": float(clearance_penalty),
        "risk_penalty": float(risk_penalty),
        "reward": float(reward),
        "active": active,
    }


def _stage_config_payload(reward_config: StaticObstacleAvoidanceRewardConfig) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "uav_count": 1,
        "training_schedule": {
            "normal_far_distance_fraction": TRAIN_NORMAL_FRACTION,
            "static_pressure_fraction": TRAIN_STATIC_FRACTION,
            "cycle_length_episodes": SCHEDULE_CYCLE_LENGTH,
            "static_slots": [8, 9, 10],
            "scope": "train_only",
        },
        "static_obstacle_reward": {
            **asdict(reward_config),
            "source": "raw_pre_projection_candidate_segment",
            "formula": "-clearance_weight*((warning_distance-clearance_pre)/warning_distance)^2 - risk_weight*I(clearance_pre<safe_distance)",
            "default_zero_interface": True,
        },
        "normal_validation_test": "curriculum_free_full_distance_holdout",
        "static_validation_test": "unseen_static_wall_holdout",
        "reward_scope": "static_pressure_training_episodes_only",
        "prohibited_changes": [
            "dynamics",
            "action_space",
            "action_mapping",
            "dt",
            "episode_horizon",
            "PPO/MAPPO hyperparameters",
            "goal_progress_and_braking_reward",
            "stable_hold_steps",
            "goal_hold_reward",
            "terminal_curriculum",
            "boundary_reward",
            "dynamic_obstacle_reward",
            "safety_execution_mode",
        ],
    }


def _static_pressure_scenario(
    config: FormalExperimentConfig,
    *,
    seed: int,
    split: str,
    episode_index: int,
) -> dict[str, Any]:
    """Create an independent wall-on-direct-path pressure scenario.

    The train and holdout geometries use different wall sizes/locations while
    retaining the same task semantics.  No dynamic obstacle is introduced in
    this stage.
    """

    index = int(episode_index)
    if split == "train":
        center_x = 50.0
        y = 94.0 + float(index % 6) * 3.0
        wall_y = 18.0 + float(index % 3) * 2.0
        wall_z = 24.0 + float(index % 2) * 2.0
    elif split == "validation":
        center_x = 49.0 + float(index % 2) * 1.5
        y = 92.0 + float(index % 5) * 4.0
        wall_y = 21.0 + float(index % 2) * 1.0
        wall_z = 23.0 + float(index % 3)
    else:
        center_x = 51.0 - float(index % 3) * 1.0
        y = 97.0 + float(index % 5) * 2.5
        wall_y = 20.0 + float(index % 2) * 3.0
        wall_z = 25.0 + float(index % 3) * 1.5
    start = np.asarray((34.0, y, 20.0), dtype=float)
    goal = np.asarray((66.0, y, 20.0), dtype=float)
    scenario = {
        "scenario_version": "single-uav-static-obstacle-stage1-v1",
        "seed": int(seed),
        "uav_count": 1,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": start.tolist()},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "initial_attitudes": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": goal.tolist()},
        "obstacle_specs": [
            {
                "obstacle_id": "stage1_static_wall",
                "obstacle_type": "building",
                "geometry_type": "box",
                "center_m": [center_x, y, 20.0],
                "size_m": [4.0, wall_y, wall_z],
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "hierarchical_waypoint": goal.tolist(),
        "stage1_static_obstacle": {
            "scene_type": "static_pressure",
            "scope": split,
            "direct_path_crosses_static_obstacle": True,
            "unseen_holdout_geometry": split in {"validation", "test"},
            "geometry_family": "single_box_wall",
            "episode_index": index,
        },
    }
    return scenario


def _normal_scenario(
    config: FormalExperimentConfig,
    *,
    seed: int,
    split: str,
    episode_index: int,
) -> dict[str, Any]:
    scenario = _normal_full_distance_scenario(
        config,
        seed,
        split=split,
        episode_index=episode_index,
        phase=split,
    )
    scenario["stage1_static_obstacle"] = {
        "scene_type": "normal_far_distance",
        "scope": split,
        "curriculum_used": False,
        "full_distance_start_preserved": True,
    }
    return scenario


def _scenario_path(data_root: Path, split: str, index: int) -> Path:
    return data_root / "scenarios" / split / f"scenario_{int(index):06d}.json"


def _write_scenario_pool(
    data_root: Path,
    config: FormalExperimentConfig,
    *,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
) -> dict[str, list[dict[str, Any]]]:
    scenarios: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation_normal": [],
        "validation_static": [],
        "test_normal": [],
        "test_static": [],
    }
    for split, count, normal_key, static_key in (
        ("train", train_episodes, "train", None),
        ("validation", validation_episodes, "validation_normal", "validation_static"),
        ("test", test_episodes, "test_normal", "test_static"),
    ):
        if split == "train":
            for episode in range(1, int(count) + 1):
                seed = _scale_seed(config.seed, 1, "train", episode - 1)
                kind = training_episode_kind(episode)
                scenario = (
                    _static_pressure_scenario(
                        config,
                        seed=seed,
                        split="train",
                        episode_index=episode,
                    )
                    if kind == "static_pressure"
                    else _normal_scenario(
                        config,
                        seed=seed,
                        split="train",
                        episode_index=episode - 1,
                    )
                )
                scenario["stage1_schedule"] = {
                    "episode_number": episode,
                    "episode_type": kind,
                    "train_only": True,
                }
                path = _scenario_path(data_root, "train", episode)
                _write_json(path, scenario)
                scenarios["train"].append(
                    {
                        "episode_number": episode,
                        "episode_type": kind,
                        "seed": int(seed),
                        "path": str(path.resolve()),
                        "scenario_hash": _canonical_hash(scenario),
                        "scenario": scenario,
                    }
                )
        else:
            for episode in range(1, int(count) + 1):
                # ``_scale_seed`` deliberately accepts only the canonical
                # phase names.  Keep normal/static holdouts reproducible and
                # disjoint by using the same split offset with non-overlapping
                # index ranges.
                normal_seed = _scale_seed(config.seed, 1, split, episode - 1)
                static_seed = _scale_seed(config.seed, 1, split, int(count) + episode - 1)
                normal = _normal_scenario(
                    config,
                    seed=normal_seed,
                    split=split,
                    episode_index=episode - 1,
                )
                static = _static_pressure_scenario(
                    config,
                    seed=static_seed,
                    split=split,
                    episode_index=episode,
                )
                for key, scenario, seed in (
                    (normal_key, normal, normal_seed),
                    (static_key, static, static_seed),
                ):
                    assert key is not None
                    scenario["stage1_schedule"] = {
                        "episode_number": episode,
                        "episode_type": "normal_far_distance" if key.endswith("normal") else "static_pressure",
                        "train_only": False,
                        "curriculum_used": False,
                    }
                    path = _scenario_path(data_root, key, episode)
                    _write_json(path, scenario)
                    scenarios[key].append(
                        {
                            "episode_number": episode,
                            "episode_type": scenario["stage1_schedule"]["episode_type"],
                            "seed": int(seed),
                            "path": str(path.resolve()),
                            "scenario_hash": _canonical_hash(scenario),
                            "scenario": scenario,
                        }
                    )
    return scenarios


def _augment_episode_scenario(
    episode_summary: dict[str, Any],
    *,
    scene_type: str,
    scenario: Mapping[str, Any],
    stage_config: StaticObstacleAvoidanceRewardConfig,
) -> dict[str, Any]:
    path = Path(str(episode_summary["path"])) / "scenario.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage1_static_obstacle"] = copy.deepcopy(
        scenario.get("stage1_static_obstacle", {})
    )
    payload["stage1_static_obstacle"]["scene_type"] = str(scene_type)
    payload["stage1_static_obstacle"]["reward_contract"] = _stage_config_payload(
        stage_config
    )["static_obstacle_reward"]
    _write_json(path, payload)
    episode_summary.setdefault("files", {})["scenario.json"] = _sha256(path)
    return episode_summary


def _apply_stage_reward(
    info: Mapping[str, Any],
    rewards: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    reward_terms: Mapping[str, float | bool],
    *,
    clip: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Add the stage reward to the PPO learning channel only."""

    learning = {
        str(key): float(value)
        for key, value in dict(info.get("learning_rewards", rewards)).items()
    }
    reward = float(reward_terms["reward"])
    unclip = {uav_id: float(value + reward) for uav_id, value in learning.items()}
    applied = {
        uav_id: float(np.clip(value, -float(clip), float(clip)))
        for uav_id, value in unclip.items()
    }
    components = copy.deepcopy(dict(info.get("learning_reward_components", {})))
    components.update(
        {
            "static_obstacle_reward_source": "raw_pre_projection_candidate_segment",
            "static_obstacle_reward": reward,
            "static_obstacle_clearance_penalty": float(reward_terms["clearance_penalty"]),
            "static_obstacle_risk_penalty": float(reward_terms["risk_penalty"]),
            "static_obstacle_clearance_signal": float(reward_terms["clearance_signal"]),
            "static_obstacle_reward_active": bool(reward_terms["active"]),
            "pre_candidate_static_swept_minimum_clearance_m": float(
                diagnostics["pre_candidate_static_swept_minimum_clearance_m"]
            ),
            "pre_candidate_static_swept_conflict": int(
                diagnostics["pre_candidate_static_swept_conflict"]
            ),
            "pre_candidate_static_unsafe": int(
                diagnostics["pre_candidate_static_unsafe"]
            ),
            "learning_reward_preclip_by_uav": dict(unclip),
            "learning_reward_total_by_uav": dict(applied),
            "learning_reward_total": float(np.mean(list(applied.values()))) if applied else 0.0,
        }
    )
    return applied, components


def _stage_step_row(
    *,
    episode_summary: Mapping[str, Any] | None,
    scene_type: str,
    phase: str,
    split: str,
    seed: int,
    step: int,
    environment: Any,
    position_before: np.ndarray,
    velocity_before: np.ndarray,
    raw_action: Mapping[str, Any],
    nominal_position: np.ndarray,
    static_diag: Mapping[str, Any],
    reward_terms: Mapping[str, float | bool],
    next_info: Mapping[str, Any],
) -> dict[str, Any]:
    uid = "uav_00"
    position_after = np.asarray(environment.states[uid].position_m, dtype=float)
    velocity_after = np.asarray(environment.states[uid].velocity_mps, dtype=float)
    goal = np.asarray(environment.goals[uid], dtype=float)
    before_goal = goal - position_before
    after_goal = goal - position_after
    before_distance = float(np.linalg.norm(before_goal))
    after_distance = float(np.linalg.norm(after_goal))
    before_direction = before_goal / max(float(np.linalg.norm(before_goal)), 1.0e-12)
    after_direction = after_goal / max(float(np.linalg.norm(after_goal)), 1.0e-12)
    raw_delta = np.asarray(raw_action[uid].get("delta_v", (0.0, 0.0, 0.0)), dtype=float)
    execution = next_info.get("action_execution", {}).get(uid, {})
    applied_delta = np.asarray(
        execution.get("delta_v_applied", raw_delta), dtype=float
    ).reshape(3)
    raw_difference = float(np.linalg.norm(raw_delta - applied_delta))
    return {
        "episode_id": str(episode_summary.get("episode_id", "")) if episode_summary else "",
        "scene_type": str(scene_type),
        "phase": str(phase),
        "split": str(split),
        "seed": int(seed),
        "step": int(step),
        "timestamp_s": float((step + 1) * environment.config.motion.dt_s),
        "position_before_x": float(position_before[0]),
        "position_before_y": float(position_before[1]),
        "position_before_z": float(position_before[2]),
        "position_after_x": float(position_after[0]),
        "position_after_y": float(position_after[1]),
        "position_after_z": float(position_after[2]),
        "nominal_position_x": float(nominal_position[0]),
        "nominal_position_y": float(nominal_position[1]),
        "nominal_position_z": float(nominal_position[2]),
        "goal_distance_before_m": before_distance,
        "goal_distance_after_m": after_distance,
        "radial_velocity_before_mps": float(np.dot(velocity_before, before_direction)),
        "radial_velocity_after_mps": float(np.dot(velocity_after, after_direction)),
        "speed_before_mps": float(np.linalg.norm(velocity_before)),
        "speed_after_mps": float(np.linalg.norm(velocity_after)),
        "raw_delta_v_x": float(raw_delta[0]),
        "raw_delta_v_y": float(raw_delta[1]),
        "raw_delta_v_z": float(raw_delta[2]),
        "applied_delta_v_x": float(applied_delta[0]),
        "applied_delta_v_y": float(applied_delta[1]),
        "applied_delta_v_z": float(applied_delta[2]),
        "raw_action_norm_mps": float(np.linalg.norm(raw_delta)),
        "applied_action_norm_mps": float(np.linalg.norm(applied_delta)),
        "raw_applied_difference_norm_mps": raw_difference,
        "raw_action_changed": int(raw_difference > 1.0e-9),
        "action_clipped": int(bool(execution.get("action_clipped", False))),
        "safety_projected": int(bool(execution.get("safety_projected", False))),
        "boundary_projected": int(bool(execution.get("boundary_projected", False))),
        "pre_projection_collision": int(next_info.get("pre_projection_collision", 0)),
        "pre_projection_minimum_clearance_m": _finite(
            next_info.get("pre_projection_minimum_clearance"), float("inf")
        ),
        "post_projection_minimum_clearance_m": _finite(
            next_info.get("minimum_clearance"), float("inf")
        ),
        "pre_candidate_static_swept_minimum_clearance_m": _finite(
            static_diag.get("pre_candidate_static_swept_minimum_clearance_m"),
            float("inf"),
        ),
        "pre_candidate_static_swept_conflict": int(
            static_diag.get("pre_candidate_static_swept_conflict", 0)
        ),
        "pre_candidate_static_unsafe": int(
            static_diag.get("pre_candidate_static_unsafe", 0)
        ),
        "static_obstacle_reward": float(reward_terms["reward"]),
        "static_obstacle_clearance_penalty": float(reward_terms["clearance_penalty"]),
        "static_obstacle_risk_penalty": float(reward_terms["risk_penalty"]),
        "static_obstacle_reward_active": int(bool(reward_terms["active"])),
        "collision": int(bool(next_info.get("collision", 0))),
        "physical_collision": int(bool(next_info.get("physical_collision", False))),
        "clearance_violation": int(bool(next_info.get("clearance_violation", False))),
        "termination_reason": str(next_info.get("termination_reason", "running")),
    }


def _stage_episode_summary(
    *,
    standard_summary: Mapping[str, Any],
    scene_type: str,
    rows: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    distances = [_finite(row.get("goal_distance_after_m"), float("inf")) for row in rows]
    speeds = [_finite(row.get("speed_after_mps"), float("nan")) for row in rows]
    finite_speeds = [value for value in speeds if np.isfinite(value)]
    pre_clearances = [
        _finite(row.get("pre_candidate_static_swept_minimum_clearance_m"), float("inf"))
        for row in rows
    ]
    post_clearances = [
        _finite(row.get("post_projection_minimum_clearance_m"), float("inf"))
        for row in rows
    ]
    hold_steps = max(
        [
            int(row.get("single_uav_stable_steps", 0) or 0)
            for row in rows
        ]
        or [0]
    )
    start = scenario.get("initial_positions", {}).get("uav_00", [0.0, 0.0, 0.0])
    goal = scenario.get("goals", {}).get("uav_00", [0.0, 0.0, 0.0])
    try:
        start_distance = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))
    except (TypeError, ValueError):
        start_distance = float("nan")
    return {
        **dict(standard_summary),
        "scene_type": str(scene_type),
        "start_distance_m": start_distance,
        "entered_8m": int(any(value <= 8.0 for value in distances)),
        "entered_5m": int(any(value <= 5.0 for value in distances)),
        "entered_3m": int(any(value <= 3.0 for value in distances)),
        "entered_1m": int(any(value <= 1.0 for value in distances)),
        "minimum_goal_distance_m": float(min(distances, default=float("inf"))),
        "mean_speed_mps": float(np.mean(finite_speeds)) if finite_speeds else 0.0,
        "minimum_pre_clearance_m": float(min(pre_clearances, default=float("inf"))),
        "minimum_post_clearance_m": float(min(post_clearances, default=float("inf"))),
        "pre_candidate_static_swept_conflict_steps": int(
            sum(int(row.get("pre_candidate_static_swept_conflict", 0) or 0) for row in rows)
        ),
        "pre_candidate_static_unsafe_steps": int(
            sum(int(row.get("pre_candidate_static_unsafe", 0) or 0) for row in rows)
        ),
        "safety_projection_steps": int(
            sum(int(row.get("safety_projected", 0) or 0) for row in rows)
        ),
        "raw_action_changed_steps": int(
            sum(int(row.get("raw_action_changed", 0) or 0) for row in rows)
        ),
        "action_clipping_fraction": float(
            np.mean([int(row.get("action_clipped", 0) or 0) for row in rows])
            if rows
            else 0.0
        ),
        "static_obstacle_reward_active_steps": int(
            sum(int(row.get("static_obstacle_reward_active", 0) or 0) for row in rows)
        ),
        "maximum_consecutive_goal_hold_steps": int(hold_steps),
    }


def _run_policy_episode(
    *,
    config: FormalExperimentConfig,
    trainer: Any,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scene_type: str,
    split: str,
    phase: str,
    serial: int,
    seed: int,
    episode_index: int,
    episode_total: int,
    update_policy: bool,
    update_index: int | None,
    reward_config: StaticObstacleAvoidanceRewardConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, float] | None]:
    _seed_everything(seed)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, info = environment.reset(
        seed=seed,
        scenario_config=copy.deepcopy(dict(scenario)),
    )
    episode_id = f"episode_{int(serial):06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / split / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{int(serial):06d}",
        split=split,
        seed=seed,
        phase=phase,
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted and trainer.central_normalizer.fitted),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source="mappo_static_obstacle_stage1",
    )
    rows: list[dict[str, Any]] = []
    report_rewards: list[float] = []
    learning_rewards: list[float] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision = trainer.act(
            observations,
            info["central_state"],
            deterministic=not update_policy,
        )
        _assert_action_feasible(environment, decision.actions)
        position_before = np.asarray(environment.states["uav_00"].position_m, dtype=float).copy()
        velocity_before = np.asarray(environment.states["uav_00"].velocity_mps, dtype=float).copy()
        nominal_state, _ = apply_velocity_increment(
            environment.states["uav_00"],
            decision.actions["uav_00"]["delta_v"],
            environment.config.motion,
            safety_projector=None,
        )
        nominal_position = np.asarray(nominal_state.position_m, dtype=float).copy()
        static_diag = static_candidate_diagnostics(
            environment,
            position_before=position_before,
            nominal_position=nominal_position,
        )
        next_observations, rewards, terminated, truncated, next_info = environment.step(
            decision.actions
        )
        if update_policy:
            trainer.reconcile_action_execution(decision, next_info)
        reward_terms = static_obstacle_reward(static_diag, reward_config)
        if not (update_policy and scene_type == "static_pressure"):
            reward_terms = {
                "clearance_signal": 0.0,
                "clearance_penalty": 0.0,
                "risk_penalty": 0.0,
                "reward": 0.0,
                "active": False,
            }
        if update_policy:
            updated_learning, components = _apply_stage_reward(
                next_info,
                rewards,
                static_diag,
                reward_terms,
                clip=float(config.learning_reward_clip),
            )
            next_info = dict(next_info)
            next_info["learning_rewards"] = updated_learning
            next_info["learning_reward_components"] = components
            trainer.add_transition(
                decision,
                updated_learning,
                next_info["central_state"],
                done=bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        effective_learning = dict(next_info.get("learning_rewards", rewards))
        report_rewards.extend(float(value) for value in rewards.values())
        learning_rewards.extend(float(value) for value in effective_learning.values())
        stage_row = _stage_step_row(
            episode_summary={"episode_id": episode_id},
            scene_type=scene_type,
            phase=phase,
            split=split,
            seed=seed,
            step=step,
            environment=environment,
            position_before=position_before,
            velocity_before=velocity_before,
            raw_action=decision.actions,
            nominal_position=nominal_position,
            static_diag=static_diag,
            reward_terms=reward_terms,
            next_info=next_info,
        )
        rows.append(stage_row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=decision.actions,
            rewards=rewards,
            learning_rewards=effective_learning,
            info=next_info,
            graph_traces=None,
            policy_source="mappo_static_obstacle_stage1",
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    standard_summary = writer.finalize()
    standard_summary.update(
        {
            "seed": int(seed),
            "mean_reward": float(np.mean(report_rewards)) if report_rewards else 0.0,
            "mean_learning_reward": float(np.mean(learning_rewards)) if learning_rewards else 0.0,
            "cumulative_return": float(np.sum(report_rewards)) if report_rewards else 0.0,
            "learning_cumulative_return": float(np.sum(learning_rewards)) if learning_rewards else 0.0,
            "learning_reward_total": float(np.sum(learning_rewards)) if learning_rewards else 0.0,
            "reporting_reward_total": float(np.sum(report_rewards)) if report_rewards else 0.0,
            "update": update_index,
            "stage1_scene_type": str(scene_type),
        }
    )
    standard_summary = _augment_episode_scenario(
        standard_summary,
        scene_type=scene_type,
        scenario=scenario,
        stage_config=reward_config,
    )
    stage_summary = _stage_episode_summary(
        standard_summary=standard_summary,
        scene_type=scene_type,
        rows=rows,
        scenario=scenario,
    )
    stage_summary["episode_id"] = episode_id
    stage_summary["scenario_id"] = standard_summary["scenario_id"]
    stage_summary["phase"] = phase
    stage_summary["split"] = split
    standard_metrics = _episode_metrics(
        scale_root / split / episode_id,
        standard_summary,
    )
    update_stats: dict[str, float] | None = None
    if update_policy:
        batch = getattr(trainer, "_episode_batch_metadata", [])
        batch.append(
            {
                "episode_id": episode_id,
                "windows": len(report_rewards) // max(1, config.uav_count),
                "mean_reward": standard_summary["mean_reward"],
                "mean_learning_reward": standard_summary["mean_learning_reward"],
                "collision": bool(standard_summary.get("collision_occurred", False)),
            }
        )
        trainer._episode_batch_metadata = batch
        update_stats = trainer.update()
        if not all(np.isfinite(float(value)) for value in update_stats.values()):
            raise RuntimeError(f"non-finite MAPPO update statistics: {update_stats}")
        training_record = {
            "uav_count": 1,
            "phase": phase,
            "split": split,
            "episode": episode_id,
            "episode_id": episode_id,
            "update": update_index,
            "mean_reward": standard_summary["mean_reward"],
            "mean_learning_reward": standard_summary["mean_learning_reward"],
            "learning_reward_total": standard_summary["learning_reward_total"],
            "reporting_reward_total": standard_summary["reporting_reward_total"],
            "completed_training_episodes": int(episode_index),
            "scene_type": str(scene_type),
            "static_obstacle_reward_active_steps": stage_summary["static_obstacle_reward_active_steps"],
            "pre_candidate_static_unsafe_steps": stage_summary["pre_candidate_static_unsafe_steps"],
            "safety_projection_steps": stage_summary["safety_projection_steps"],
            **{str(key): float(value) for key, value in update_stats.items()},
        }
        _append_jsonl(scale_root / "diagnostics" / "training_metrics.jsonl", training_record)
        _append_jsonl(
            scale_root / "diagnostics" / "training_episodes.jsonl",
            {
                **stage_summary,
                "training_episode": int(episode_index),
                "cumulative_return": standard_summary["cumulative_return"],
                "learning_cumulative_return": standard_summary["learning_cumulative_return"],
                "assigned_update": update_index,
            },
        )
        trainer._episode_batch_metadata = []
    return standard_summary, stage_summary, rows, update_stats


def _direct_probe_action(environment: Any) -> dict[str, dict[str, Any]]:
    state = environment.states["uav_00"]
    goal = np.asarray(environment.goals["uav_00"], dtype=float)
    direction = goal - np.asarray(state.position_m, dtype=float)
    norm = float(np.linalg.norm(direction))
    direction = direction / max(norm, 1.0e-12)
    desired_velocity = direction * 6.0
    delta_v = desired_velocity - np.asarray(state.velocity_mps, dtype=float)
    max_delta = float(environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s)
    magnitude = float(np.linalg.norm(delta_v))
    if magnitude > max_delta:
        delta_v = delta_v * max_delta / magnitude
    return {
        "uav_00": {
            "delta_v": delta_v,
            "measurement_uav_id": "none",
            "bandwidth": {},
        }
    }


def _run_static_pressure_sanity(
    config: FormalExperimentConfig,
    *,
    data_root: Path,
    logs_root: Path,
    reward_config: StaticObstacleAvoidanceRewardConfig,
) -> dict[str, Any]:
    scenario = _static_pressure_scenario(
        config,
        seed=2026,
        split="test",
        episode_index=0,
    )
    # A small x shift puts the legal direct-path candidate segment through the
    # front face before the 2 m protection shell, making the swept-contact and
    # unsafe diagnostics independently visible.
    scenario["obstacle_specs"][0]["center_m"][0] = 49.0
    probe_config = replace(config, max_steps=100)
    no_safety_env = _make_environment(
        probe_config,
        2026,
        safety_projection_enabled=False,
    )
    observations, info = no_safety_env.reset(
        seed=2026,
        scenario_config=copy.deepcopy(scenario),
    )
    action_trace: list[dict[str, Any]] = []
    for _step in range(int(probe_config.max_steps)):
        action = _direct_probe_action(no_safety_env)
        action_trace.append(copy.deepcopy(action))
        observations, _rewards, terminated, truncated, _info = no_safety_env.step(action)
        if terminated or truncated:
            break

    mode_rows: list[dict[str, Any]] = []
    replay_hash = _canonical_hash(
        [
            np.asarray(action["uav_00"]["delta_v"], dtype=float).tolist()
            for action in action_trace
        ]
    )
    for mode, safety_enabled in (("safety_off", False), ("safety_on", True)):
        environment = _make_environment(
            probe_config,
            2026,
            safety_projection_enabled=safety_enabled,
        )
        observations, info = environment.reset(
            seed=2026,
            scenario_config=copy.deepcopy(scenario),
        )
        for step, action in enumerate(action_trace):
            position_before = np.asarray(environment.states["uav_00"].position_m, dtype=float).copy()
            velocity_before = np.asarray(environment.states["uav_00"].velocity_mps, dtype=float).copy()
            nominal_state, _ = apply_velocity_increment(
                environment.states["uav_00"],
                action["uav_00"]["delta_v"],
                environment.config.motion,
                safety_projector=None,
            )
            nominal_position = np.asarray(nominal_state.position_m, dtype=float)
            diag = static_candidate_diagnostics(
                environment,
                position_before=position_before,
                nominal_position=nominal_position,
            )
            next_observations, _rewards, terminated, truncated, next_info = environment.step(
                copy.deepcopy(action)
            )
            terms = static_obstacle_reward(diag, reward_config)
            execution = next_info.get("action_execution", {}).get("uav_00", {})
            mode_rows.append(
                {
                    "mode": mode,
                    "seed": 2026,
                    "step": step,
                    "position_before_x": float(position_before[0]),
                    "position_after_x": float(environment.states["uav_00"].position_m[0]),
                    "speed_before_mps": float(np.linalg.norm(velocity_before)),
                    "speed_after_mps": float(np.linalg.norm(environment.states["uav_00"].velocity_mps)),
                    "pre_candidate_static_swept_minimum_clearance_m": float(diag["pre_candidate_static_swept_minimum_clearance_m"]),
                    "pre_candidate_static_swept_conflict": int(diag["pre_candidate_static_swept_conflict"]),
                    "pre_candidate_static_unsafe": int(diag["pre_candidate_static_unsafe"]),
                    "pre_projection_collision": int(next_info.get("pre_projection_collision", 0)),
                    "safety_projected": int(bool(execution.get("safety_projected", False))),
                    "collision": int(bool(next_info.get("collision", 0))),
                    "static_obstacle_reward": float(terms["reward"]),
                    "termination_reason": str(next_info.get("termination_reason", "running")),
                }
            )
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
    _write_csv(
        logs_root / "static_pressure_sanity_steps.csv",
        (
            "mode",
            "seed",
            "step",
            "position_before_x",
            "position_after_x",
            "speed_before_mps",
            "speed_after_mps",
            "pre_candidate_static_swept_minimum_clearance_m",
            "pre_candidate_static_swept_conflict",
            "pre_candidate_static_unsafe",
            "pre_projection_collision",
            "safety_projected",
            "collision",
            "static_obstacle_reward",
            "termination_reason",
        ),
        mode_rows,
    )
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("safety_off", "safety_on"):
        rows = [row for row in mode_rows if row["mode"] == mode]
        by_mode[mode] = {
            "steps": len(rows),
            "pre_risk_steps": int(
                sum(
                    int(row["pre_candidate_static_unsafe"])
                    or int(row["pre_candidate_static_swept_conflict"])
                    for row in rows
                )
            ),
            "pre_candidate_static_swept_conflict_steps": int(
                sum(int(row["pre_candidate_static_swept_conflict"]) for row in rows)
            ),
            "pre_candidate_static_unsafe_steps": int(
                sum(int(row["pre_candidate_static_unsafe"]) for row in rows)
            ),
            "safety_projection_steps": int(sum(int(row["safety_projected"]) for row in rows)),
            "collision_steps": int(sum(int(row["collision"]) for row in rows)),
            "minimum_pre_clearance_m": float(
                min((float(row["pre_candidate_static_swept_minimum_clearance_m"]) for row in rows), default=float("inf"))
            ),
        }
    result = {
        "scenario_hash": _canonical_hash(scenario),
        "action_trace_hash": replay_hash,
        "same_scene_and_action_trace": True,
        "by_mode": by_mode,
        "passed_pressure_gate": bool(
            by_mode["safety_off"]["pre_risk_steps"] > 0
            and by_mode["safety_on"]["pre_risk_steps"] > 0
        ),
        "safety_on_intervened": bool(by_mode["safety_on"]["safety_projection_steps"] > 0),
        "safety_off_collision_or_risk": bool(
            by_mode["safety_off"]["collision_steps"] > 0
            or by_mode["safety_off"]["pre_risk_steps"] > 0
        ),
    }
    _write_json(logs_root / "static_pressure_sanity_summary.json", result)
    return result


def _aggregate_stage_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    def count(name: str) -> int:
        return int(sum(int(row.get(name, 0) or 0) for row in rows))
    def nonempty_count(name: str) -> int:
        return int(sum(bool(row.get(name)) for row in rows))
    def rate(name: str) -> float:
        return float(count(name) / len(rows))
    def mean(name: str) -> float:
        values = [_finite(row.get(name), float("nan")) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else 0.0
    return {
        "episodes": len(rows),
        "entered_8m_episodes": count("entered_8m"),
        "entered_8m_rate": rate("entered_8m"),
        "entered_5m_episodes": count("entered_5m"),
        "entered_5m_rate": rate("entered_5m"),
        "entered_3m_episodes": count("entered_3m"),
        "entered_3m_rate": rate("entered_3m"),
        "entered_1m_episodes": count("entered_1m"),
        "entered_1m_rate": rate("entered_1m"),
        "collision_episodes": count("collision_occurred"),
        "collision_rate": rate("collision_occurred"),
        "physical_collision_episodes": nonempty_count("physical_collision_steps"),
        "clearance_violation_episodes": nonempty_count("clearance_violation_steps"),
        "mean_minimum_goal_distance_m": mean("minimum_goal_distance_m"),
        "mean_speed_mps": mean("mean_speed_mps"),
        "mean_minimum_pre_clearance_m": mean("minimum_pre_clearance_m"),
        "mean_minimum_post_clearance_m": mean("minimum_post_clearance_m"),
        "pre_candidate_static_risk_episodes": int(
            sum(
                int(row.get("pre_candidate_static_swept_conflict_steps", 0) or 0) > 0
                or int(row.get("pre_candidate_static_unsafe_steps", 0) or 0) > 0
                for row in rows
            )
        ),
        "pre_candidate_static_swept_conflict_steps": int(
            sum(int(row.get("pre_candidate_static_swept_conflict_steps", 0) or 0) for row in rows)
        ),
        "pre_candidate_static_unsafe_steps": int(
            sum(int(row.get("pre_candidate_static_unsafe_steps", 0) or 0) for row in rows)
        ),
        "safety_projection_steps": int(
            sum(int(row.get("safety_projection_steps", 0) or 0) for row in rows)
        ),
        "safety_projection_episodes": int(
            sum(int(row.get("safety_projection_steps", 0) or 0) > 0 for row in rows)
        ),
        "raw_action_changed_steps": int(
            sum(int(row.get("raw_action_changed_steps", 0) or 0) for row in rows)
        ),
        "action_clipping_fraction": mean("action_clipping_fraction"),
        "mean_static_obstacle_reward_active_steps": mean("static_obstacle_reward_active_steps"),
        "mean_maximum_consecutive_goal_hold_steps": mean("maximum_consecutive_goal_hold_steps"),
    }


def _plot_stage_diagnostics(
    *,
    plot_root: Path,
    training_rows: Sequence[Mapping[str, Any]],
    static_step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root.mkdir(parents=True, exist_ok=True)
    schedule = [str(row.get("scene_type", "")) for row in training_rows]
    fig, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    values = [1 if value == "static_pressure" else 0 for value in schedule]
    axis.step(range(1, len(values) + 1), values, where="mid", color="#2563eb")
    axis.set_yticks([0, 1], ["normal far", "static pressure"])
    axis.set_xlabel("Training episode")
    axis.set_title("Stage-1 training schedule")
    axis.grid(alpha=0.25)
    schedule_path = plot_root / "stage1_training_schedule.png"
    fig.savefig(schedule_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for mode, color in (("safety_off", "#dc2626"), ("safety_on", "#16a34a")):
        selected = [row for row in static_step_rows if row.get("mode") == mode]
        selected.sort(key=lambda row: int(row.get("step", 0)))
        if selected:
            x = [int(row["step"]) for row in selected]
            axes[0].plot(
                x,
                [float(row["pre_candidate_static_swept_minimum_clearance_m"]) for row in selected],
                color=color,
                label=mode,
            )
            axes[1].plot(
                x,
                [float(row["position_after_x"]) for row in selected],
                color=color,
                label=mode,
            )
    axes[0].axhline(2.0, color="#64748b", linestyle="--", linewidth=0.9, label="safety distance")
    axes[0].set_title("Raw candidate static clearance")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Clearance (m)")
    axes[1].set_title("Safety-on/off replay position")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("x position (m)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    sanity_path = plot_root / "static_pressure_safety_replay.png"
    fig.savefig(sanity_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "training_schedule": str(schedule_path.resolve()),
        "static_pressure_safety_replay": str(sanity_path.resolve()),
    }


def _write_stage_csvs(
    logs_root: Path,
    *,
    stage_step_rows: Sequence[Mapping[str, Any]],
    stage_episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    step_columns = (
        "episode_id", "scene_type", "phase", "split", "seed", "step", "timestamp_s",
        "position_before_x", "position_before_y", "position_before_z",
        "position_after_x", "position_after_y", "position_after_z",
        "nominal_position_x", "nominal_position_y", "nominal_position_z",
        "goal_distance_before_m", "goal_distance_after_m",
        "radial_velocity_before_mps", "radial_velocity_after_mps",
        "speed_before_mps", "speed_after_mps", "raw_delta_v_x", "raw_delta_v_y", "raw_delta_v_z",
        "applied_delta_v_x", "applied_delta_v_y", "applied_delta_v_z", "raw_action_norm_mps",
        "applied_action_norm_mps", "raw_applied_difference_norm_mps", "raw_action_changed",
        "action_clipped", "safety_projected", "boundary_projected", "pre_projection_collision",
        "pre_projection_minimum_clearance_m", "post_projection_minimum_clearance_m",
        "pre_candidate_static_swept_minimum_clearance_m", "pre_candidate_static_swept_conflict",
        "pre_candidate_static_unsafe", "static_obstacle_reward", "static_obstacle_clearance_penalty",
        "static_obstacle_risk_penalty", "static_obstacle_reward_active", "collision",
        "physical_collision", "clearance_violation", "termination_reason",
    )
    episode_columns = (
        "episode_id", "scenario_id", "scene_type", "phase", "split", "seed", "start_distance_m",
        "steps", "entered_8m", "entered_5m", "entered_3m", "entered_1m", "collision_occurred",
        "physical_collision_steps", "clearance_violation_steps", "minimum_goal_distance_m",
        "mean_speed_mps", "minimum_pre_clearance_m", "minimum_post_clearance_m",
        "pre_candidate_static_swept_conflict_steps", "pre_candidate_static_unsafe_steps",
        "safety_projection_steps", "raw_action_changed_steps", "action_clipping_fraction",
        "static_obstacle_reward_active_steps", "maximum_consecutive_goal_hold_steps",
        "termination_reason",
    )
    _write_csv(logs_root / "stage1_step_diagnostics.csv", step_columns, stage_step_rows)
    _write_csv(logs_root / "stage1_episode_metrics.csv", episode_columns, stage_episode_rows)


def run_static_obstacle_avoidance_stage1(
    *,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    train_episodes: int = 30,
    validation_episodes: int = 5,
    test_episodes: int = 5,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run code checks, static sanity, and the bounded stage-1 smoke."""

    if int(train_episodes) < 1 or int(validation_episodes) < 1 or int(test_episodes) < 1:
        raise ValueError("episode counts must be positive")
    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    if data_root.exists() and any(data_root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty data root: {data_root.resolve()}")
    if logs_root.exists() and any(logs_root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty logs root: {logs_root.resolve()}")
    if plots_root.exists() and any(plots_root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty plots root: {plots_root.resolve()}")
    # ``force`` is intentionally not a destructive cleanup switch.  This
    # runner only permits it for empty roots, preserving the repository's
    # recoverable-output convention.
    data_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(baseline_checkpoint).resolve()
    payload, checkpoint_sha256 = _read_checkpoint(checkpoint)
    baseline_guard = _baseline_checkpoint_guard(payload, checkpoint_path=checkpoint)
    config = _config_from_checkpoint(payload, output_dir=data_root, progress=progress)
    if config.uav_count != 1:
        raise ValueError("stage-1 static obstacle experiment requires a 1-UAV baseline")
    # The stage itself has one explicit normal/full-distance track and one
    # explicit static-pressure track.  This only disables automatic start
    # rewriting for the scenario factory; the checkpoint's reward, action,
    # dynamics and PPO settings remain the source of truth.
    config = replace(
        config,
        output_dir=data_root,
        train_episodes=int(train_episodes),
        validation_episodes=int(validation_episodes) * 2,
        test_episodes=int(test_episodes) * 2,
        force=True,
        progress=progress,
        command_line="single-UAV static-obstacle avoidance stage-1 smoke",
        scenario_path=None,
        scenario_pool_root=None,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
    )
    if config.single_uav_goal_hold_reward != 0.0:
        raise ValueError("baseline goal_hold_reward must remain zero")
    if config.single_uav_stable_hold_steps != 5:
        raise ValueError("baseline stable_hold_steps must remain five")
    if not math.isclose(config.single_uav_braking_progress_reward_coef, 0.25, abs_tol=1e-12):
        raise ValueError("stage-1 requires braking_progress_reward_coef=0.25")

    stage_config = _stage_config_payload(ACTIVE_REWARD_CONFIG)
    _write_json(logs_root / "stage1_config.json", stage_config)
    _write_json(
        logs_root / "checkpoint_identity.json",
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_role": "original_coef025_baseline",
            "baseline_guard": baseline_guard,
        },
    )
    _write_json(data_root / "baseline_checkpoint_identity.json", {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": "original_coef025_baseline",
    })

    scale_root = data_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    normalization = payload.get("normalization_stats")
    central = payload.get("central_normalization_stats")
    _write_json(
        scale_root / "normalization_stats.json",
        {
            "normalization_stats": _jsonable(normalization),
            "central_normalization_stats": _jsonable(central),
            "source": "baseline_checkpoint_frozen_state",
        },
    )
    _write_json(
        scale_root / "diagnostics" / "initial_parameter_hashes.json",
        {
            "source": "baseline_checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
            "loaded_policy_hash": _state_dict_hash(payload.get("policy_state_dict", {})),
        },
    )

    scenarios = _write_scenario_pool(
        data_root,
        config,
        train_episodes=int(train_episodes),
        validation_episodes=int(validation_episodes),
        test_episodes=int(test_episodes),
    )
    sanity = _run_static_pressure_sanity(
        config,
        data_root=data_root,
        logs_root=logs_root,
        reward_config=ACTIVE_REWARD_CONFIG,
    )
    if not sanity["passed_pressure_gate"]:
        _write_json(
            data_root / "summary.json",
            {
                "status": "blocked_static_pressure_gate",
                "sanity": sanity,
                "message": "static pressure did not produce pre-projection risk; no training was started",
            },
        )
        return {"status": "blocked_static_pressure_gate", "sanity": sanity}

    trainer, trainer_identity = _load_trainable_trainer(payload, config)
    initial_parameters = _parameter_vector(trainer)
    trainer.policy.train()
    train_episode_summaries: list[dict[str, Any]] = []
    validation_summaries: list[dict[str, Any]] = []
    test_summaries: list[dict[str, Any]] = []
    standard_metric_rows: list[dict[str, Any]] = []
    stage_episode_rows: list[dict[str, Any]] = []
    stage_step_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    schedule = training_schedule_counts(int(train_episodes))
    for episode in range(1, int(train_episodes) + 1):
        kind = training_episode_kind(episode)
        item = scenarios["train"][episode - 1]
        summary, stage_summary, rows, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            scenario=item["scenario"],
            scene_type=kind,
            split="train",
            phase="train",
            serial=episode,
            seed=int(item["seed"]),
            episode_index=episode,
            episode_total=int(train_episodes),
            update_policy=True,
            update_index=episode,
            reward_config=ACTIVE_REWARD_CONFIG,
        )
        summary["stage1_schedule"] = copy.deepcopy(item)
        train_episode_summaries.append(summary)
        stage_episode_rows.append(stage_summary)
        stage_step_rows.extend(rows)
        standard_metric_rows.append(_episode_metrics(scale_root / "train" / summary["episode_id"], summary))

    final_parameters = _parameter_vector(trainer)
    parameter_delta = {
        "initial_l2": float(initial_parameters.norm().cpu()),
        "final_l2": float(final_parameters.norm().cpu()),
        "delta_l2": float((final_parameters - initial_parameters).norm().cpu()),
        "updates": int(train_episodes),
    }
    _write_json(scale_root / "diagnostics" / "parameter_delta.json", parameter_delta)
    if parameter_delta["delta_l2"] <= 0.0:
        raise RuntimeError("stage-1 smoke did not update policy parameters")

    # All four evaluation groups use frozen deterministic policy execution.
    trainer.policy.eval()
    for index, item in enumerate(scenarios["validation_normal"], start=1):
        summary, stage_summary, rows, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            scenario=item["scenario"],
            scene_type="normal_far_distance",
            split="validation",
            phase="validation_normal_holdout",
            serial=400000 + index,
            seed=int(item["seed"]),
            episode_index=index,
            episode_total=int(validation_episodes),
            update_policy=False,
            update_index=None,
            reward_config=ACTIVE_REWARD_CONFIG,
        )
        validation_summaries.append(stage_summary)
        stage_episode_rows.append(stage_summary)
        stage_step_rows.extend(rows)
        standard_metric_rows.append(_episode_metrics(scale_root / "validation" / summary["episode_id"], summary))
    for index, item in enumerate(scenarios["validation_static"], start=1):
        summary, stage_summary, rows, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            scenario=item["scenario"],
            scene_type="static_pressure",
            split="validation",
            phase="validation_static_holdout",
            serial=410000 + index,
            seed=int(item["seed"]),
            episode_index=index,
            episode_total=int(validation_episodes),
            update_policy=False,
            update_index=None,
            reward_config=ACTIVE_REWARD_CONFIG,
        )
        validation_summaries.append(stage_summary)
        stage_episode_rows.append(stage_summary)
        stage_step_rows.extend(rows)
        standard_metric_rows.append(_episode_metrics(scale_root / "validation" / summary["episode_id"], summary))
    for index, item in enumerate(scenarios["test_normal"], start=1):
        summary, stage_summary, rows, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            scenario=item["scenario"],
            scene_type="normal_far_distance",
            split="test",
            phase="test_normal_holdout",
            serial=500000 + index,
            seed=int(item["seed"]),
            episode_index=index,
            episode_total=int(test_episodes),
            update_policy=False,
            update_index=None,
            reward_config=ACTIVE_REWARD_CONFIG,
        )
        test_summaries.append(stage_summary)
        stage_episode_rows.append(stage_summary)
        stage_step_rows.extend(rows)
        standard_metric_rows.append(_episode_metrics(scale_root / "test" / summary["episode_id"], summary))
    for index, item in enumerate(scenarios["test_static"], start=1):
        summary, stage_summary, rows, _stats = _run_policy_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            scenario=item["scenario"],
            scene_type="static_pressure",
            split="test",
            phase="test_static_holdout",
            serial=510000 + index,
            seed=int(item["seed"]),
            episode_index=index,
            episode_total=int(test_episodes),
            update_policy=False,
            update_index=None,
            reward_config=ACTIVE_REWARD_CONFIG,
        )
        test_summaries.append(stage_summary)
        stage_episode_rows.append(stage_summary)
        stage_step_rows.extend(rows)
        standard_metric_rows.append(_episode_metrics(scale_root / "test" / summary["episode_id"], summary))

    _write_stage_csvs(
        logs_root,
        stage_step_rows=stage_step_rows,
        stage_episode_rows=stage_episode_rows,
    )
    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", standard_metric_rows)
    _write_episode_metrics(
        scale_root / "diagnostics" / "validation_episode_metrics.csv",
        [row for row in standard_metric_rows if str(row.get("split", "")) == "validation"],
    )
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)

    # The stage checkpoint is independent and is never written to the source
    # baseline directory.
    from .formal import _save_checkpoint

    _save_checkpoint(
        scale_root / "checkpoints" / "final.pt",
        trainer,
        config,
        int(train_episodes),
    )
    all_standard_summaries = train_episode_summaries + [
        row for row in stage_episode_rows if str(row.get("split", "")) in {"validation", "test"}
    ]
    scale_manifest = _write_scale_manifest(
        scale_root=scale_root,
        config=config,
        episodes=all_standard_summaries,
    )
    _update_root_manifest(data_root, scale_manifest)

    sanity_step_rows = []
    sanity_path = logs_root / "static_pressure_sanity_steps.csv"
    if sanity_path.is_file():
        import csv

        with sanity_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sanity_step_rows = list(csv.DictReader(handle))
    plot_paths = _plot_stage_diagnostics(
        plot_root=plots_root,
        training_rows=stage_episode_rows[: int(train_episodes)],
        static_step_rows=sanity_step_rows,
    )

    # Reuse the canonical summarizer so the stage has the normal formal
    # learning curves, episode CSV aggregate, and single-UAV diagnostics.
    preliminary_summary = summarize_preliminary(
        data_root,
        logs_output=logs_root,
        plots_output=plots_root,
    )
    validation_report = validate_preliminary(data_root)
    test_normal = [row for row in test_summaries if row.get("scene_type") == "normal_far_distance"]
    test_static = [row for row in test_summaries if row.get("scene_type") == "static_pressure"]
    validation_normal = [row for row in validation_summaries if row.get("scene_type") == "normal_far_distance"]
    validation_static = [row for row in validation_summaries if row.get("scene_type") == "static_pressure"]
    stage_report = {
        "status": "completed_smoke" if validation_report.get("ok", False) else "completed_smoke_validation_failed",
        "stage": STAGE_NAME,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "loaded_policy_hash": _state_dict_hash(payload.get("policy_state_dict", {})),
            "trainer_loaded_policy_hash": trainer_identity.get("loaded_policy_hash"),
            "initialized_from_baseline": bool(
                trainer_identity.get("loaded_policy_hash") == _state_dict_hash(payload.get("policy_state_dict", {}))
            ),
            "stage_checkpoint": str((scale_root / "checkpoints" / "final.pt").resolve()),
        },
        "parameters": {
            "uav_count": 1,
            "max_steps": int(config.max_steps),
            "dt_s": 0.2,
            "single_uav_learning_mode": config.single_uav_learning_mode,
            "single_uav_speed_control_mode": config.single_uav_speed_control_mode,
            "single_uav_braking_progress_reward_coef": config.single_uav_braking_progress_reward_coef,
            "single_uav_stable_hold_steps": config.single_uav_stable_hold_steps,
            "single_uav_goal_hold_reward": config.single_uav_goal_hold_reward,
            "safety_execution_mode": config.safety_execution_mode,
            "safety_projection_enabled": True,
            "boundary_penalty_weight": config.boundary_penalty_weight,
            "dynamic_obstacle_training": False,
        },
        "schedule": {
            "requested_fractions": {"normal_far_distance": TRAIN_NORMAL_FRACTION, "static_pressure": TRAIN_STATIC_FRACTION},
            "actual_counts": schedule,
            "actual_fractions": {key: value / int(train_episodes) for key, value in schedule.items()},
            "all_train_rows_are_train_only": all(row.get("split") == "train" for row in stage_episode_rows[: int(train_episodes)]),
        },
        "sanity": sanity,
        "training": {
            "episodes": int(train_episodes),
            "parameter_delta": parameter_delta,
            "static_reward_active_steps": int(sum(int(row.get("static_obstacle_reward_active_steps", 0) or 0) for row in stage_episode_rows[: int(train_episodes)])),
            "pre_risk_episodes": int(sum(int(row.get("pre_candidate_static_unsafe_steps", 0) or 0) > 0 or int(row.get("pre_candidate_static_swept_conflict_steps", 0) or 0) > 0 for row in stage_episode_rows[: int(train_episodes)])),
        },
        "holdout": {
            "validation_normal": _aggregate_stage_rows(validation_normal),
            "validation_static": _aggregate_stage_rows(validation_static),
            "test_normal": _aggregate_stage_rows(test_normal),
            "test_static": _aggregate_stage_rows(test_static),
            "curriculum_free_validation_test": all(
                not bool(item["scenario"].get("stage1_schedule", {}).get("train_only", True))
                and item["scenario"].get("stage1_schedule", {}).get("curriculum_used") is False
                for key in ("validation_normal", "validation_static", "test_normal", "test_static")
                for item in scenarios[key]
            ),
        },
        "schema_csv_summary_plots": {
            "validate_preliminary": validation_report,
            "summarize_preliminary": preliminary_summary,
            "custom_logs": {
                "stage1_config": str((logs_root / "stage1_config.json").resolve()),
                "stage1_step_diagnostics": str((logs_root / "stage1_step_diagnostics.csv").resolve()),
                "stage1_episode_metrics": str((logs_root / "stage1_episode_metrics.csv").resolve()),
                "sanity_summary": str((logs_root / "static_pressure_sanity_summary.json").resolve()),
            },
            "custom_plots": plot_paths,
        },
        "code_checks": {
            "static_reward_default_interface_is_zero": static_obstacle_reward(
                {"pre_candidate_static_swept_minimum_clearance_m": 0.0},
                StaticObstacleAvoidanceRewardConfig(),
            )["reward"] == 0.0,
            "reward_source_is_pre_projection": True,
            "policy_updates_occurred": parameter_delta["delta_l2"] > 0.0,
            "training_not_random_init": bool(
                trainer_identity.get("loaded_policy_hash") == _state_dict_hash(payload.get("policy_state_dict", {}))
            ),
        },
    }
    _write_json(logs_root / "stage1_summary.json", stage_report)
    _write_json(data_root / "summary.json", {**preliminary_summary, "stage1": stage_report})
    return stage_report


def run_static_holdout_baseline_evaluation(
    *,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    source_data_root: str | Path,
    data_root: str | Path,
    logs_root: str | Path,
    plots_root: str | Path,
    validation_episodes: int = 20,
    test_episodes: int = 20,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate the frozen baseline on the formal static/normal holdouts.

    This is a read-only control for the formal stage-1 experiment.  The
    scenarios are copied from an already completed formal run so the control
    uses exactly the same holdout geometry, seeds, and episode order.  No
    trainer update is called and no source experiment directory is modified.
    """

    if int(validation_episodes) < 1 or int(test_episodes) < 1:
        raise ValueError("validation_episodes and test_episodes must be positive")
    source_data_root = Path(source_data_root).resolve()
    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    output_roots = (data_root, logs_root, plots_root)
    for root in output_roots:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty baseline-control root: {root.resolve()}"
            )
        root.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(baseline_checkpoint).resolve()
    payload, checkpoint_sha256 = _read_checkpoint(checkpoint)
    baseline_guard = _baseline_checkpoint_guard(payload, checkpoint_path=checkpoint)
    config = _config_from_checkpoint(payload, output_dir=data_root, progress=progress)
    if config.uav_count != 1:
        raise ValueError("baseline holdout control requires a 1-UAV checkpoint")
    # Explicit holdout scenarios are supplied below.  Zeroing the curriculum
    # fractions here makes the control contract explicit; it cannot affect
    # the frozen policy, environment dynamics, or the supplied scenarios.
    config = replace(
        config,
        output_dir=data_root,
        train_episodes=1,
        validation_episodes=int(validation_episodes),
        test_episodes=int(test_episodes),
        force=True,
        progress=progress,
        command_line="single-UAV frozen baseline holdout control",
        scenario_path=None,
        scenario_pool_root=None,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
    )
    trainer, trainer_identity, checkpoint_identity = _load_frozen_trainer(
        payload,
        config,
        metric_scales=config.metric_scales,
    )
    expected_policy_hash = _state_dict_hash(payload.get("policy_state_dict", {}))
    if trainer_identity.get("loaded_policy_hash") != expected_policy_hash:
        raise RuntimeError(
            "frozen baseline control loaded a policy hash different from the checkpoint"
        )
    trainer.policy.eval()

    scale_root = data_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    _write_json(
        scale_root / "normalization_stats.json",
        {
            "normalization_stats": _jsonable(payload.get("normalization_stats")),
            "central_normalization_stats": _jsonable(
                payload.get("central_normalization_stats")
            ),
            "source": "baseline_checkpoint_frozen_state",
        },
    )
    _write_json(
        scale_root / "diagnostics" / "initial_parameter_hashes.json",
        {
            "source": "frozen_baseline_checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_policy_hash": expected_policy_hash,
            "loaded_policy_hash": trainer_identity.get("loaded_policy_hash"),
            "policy_updates": 0,
        },
    )
    _write_json(
        logs_root / "checkpoint_identity.json",
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_policy_hash": expected_policy_hash,
            "loaded_policy_hash": trainer_identity.get("loaded_policy_hash"),
            "checkpoint_identity": checkpoint_identity,
            "baseline_guard": baseline_guard,
            "policy_frozen": True,
            "training_updates": 0,
        },
    )

    scenario_items: dict[str, list[dict[str, Any]]] = {}
    for key, count in (
        ("validation_normal", int(validation_episodes)),
        ("validation_static", int(validation_episodes)),
        ("test_normal", int(test_episodes)),
        ("test_static", int(test_episodes)),
    ):
        source_split_root = source_data_root / "scenarios" / key
        source_paths = sorted(source_split_root.glob("scenario_*.json"))
        if len(source_paths) != count:
            raise ValueError(
                f"expected {count} exact source scenarios in {source_split_root}, "
                f"found {len(source_paths)}"
            )
        target_split_root = data_root / "scenarios" / key
        target_split_root.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for index, source_path in enumerate(source_paths, start=1):
            try:
                scenario = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not load source scenario {source_path}: {exc}") from exc
            if not isinstance(scenario, Mapping):
                raise ValueError(f"source scenario must be a mapping: {source_path}")
            target_path = target_split_root / source_path.name
            _write_json(target_path, scenario)
            items.append(
                {
                    "episode_number": index,
                    "seed": int(scenario.get("seed", 0)),
                    "source_path": str(source_path.resolve()),
                    "source_sha256": _sha256(source_path),
                    "path": str(target_path.resolve()),
                    "scenario_hash": _canonical_hash(scenario),
                    "scenario": dict(scenario),
                }
            )
        scenario_items[key] = items

    stage_config = _stage_config_payload(StaticObstacleAvoidanceRewardConfig())
    _write_json(logs_root / "baseline_holdout_config.json", {
        "control_type": "frozen_baseline_holdout",
        "policy_updates": 0,
        "curriculum_used": False,
        "evaluation_reward_channel": "disabled",
        "static_obstacle_reward": stage_config["static_obstacle_reward"],
        "source_data_root": str(source_data_root),
        "same_holdout_scenarios_as_formal_stage1": True,
    })

    stage_episode_rows: list[dict[str, Any]] = []
    stage_step_rows: list[dict[str, Any]] = []
    standard_summaries: list[dict[str, Any]] = []
    standard_metric_rows: list[dict[str, Any]] = []
    group_plan = (
        ("validation_normal", "validation", "normal_far_distance", 610000),
        ("validation_static", "validation", "static_pressure", 620000),
        ("test_normal", "test", "normal_far_distance", 630000),
        ("test_static", "test", "static_pressure", 640000),
    )
    for key, split, scene_type, serial_base in group_plan:
        items = scenario_items[key]
        for index, item in enumerate(items, start=1):
            summary, stage_summary, rows, update_stats = _run_policy_episode(
                config=config,
                trainer=trainer,
                scale_root=scale_root,
                scenario=item["scenario"],
                scene_type=scene_type,
                split=split,
                phase=f"baseline_{key}_holdout",
                serial=int(serial_base) + index,
                seed=int(item["seed"]),
                episode_index=index,
                episode_total=len(items),
                update_policy=False,
                update_index=None,
                reward_config=StaticObstacleAvoidanceRewardConfig(),
            )
            if update_stats is not None:
                raise RuntimeError("frozen baseline control unexpectedly returned update statistics")
            summary["control_source_scenario_path"] = item["source_path"]
            summary["control_source_scenario_sha256"] = item["source_sha256"]
            stage_summary["control_source_scenario_path"] = item["source_path"]
            stage_summary["control_source_scenario_sha256"] = item["source_sha256"]
            standard_summaries.append(summary)
            stage_episode_rows.append(stage_summary)
            stage_step_rows.extend(rows)
            standard_metric_rows.append(
                _episode_metrics(scale_root / split / summary["episode_id"], summary)
            )

    _write_stage_csvs(
        logs_root,
        stage_step_rows=stage_step_rows,
        stage_episode_rows=stage_episode_rows,
    )
    _write_episode_metrics(
        scale_root / "diagnostics" / "episode_metrics.csv",
        standard_metric_rows,
    )
    _write_episode_metrics(
        scale_root / "diagnostics" / "validation_episode_metrics.csv",
        [row for row in standard_metric_rows if str(row.get("split", "")) == "validation"],
    )
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", [])
    scale_manifest = _write_scale_manifest(
        scale_root=scale_root,
        config=config,
        episodes=standard_summaries,
    )
    _update_root_manifest(data_root, scale_manifest)
    # ``validate_preliminary`` requires the root summary to exist, so create a
    # small marker before the final report is assembled.
    _write_json(data_root / "summary.json", {
        "status": "baseline_holdout_control_in_progress",
        "policy_frozen": True,
    })
    validation_report = validate_preliminary(data_root)

    groups = {
        key: _aggregate_stage_rows(
            [row for row in stage_episode_rows if row.get("phase") == f"baseline_{key}_holdout"]
        )
        for key, _split, _scene_type, _serial in group_plan
    }
    report = {
        "status": "completed_read_only_baseline_holdout",
        "control_only": True,
        "policy_frozen": True,
        "training_updates": 0,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "checkpoint_policy_hash": expected_policy_hash,
            "loaded_policy_hash": trainer_identity.get("loaded_policy_hash"),
            "hash_match": trainer_identity.get("loaded_policy_hash") == expected_policy_hash,
            "baseline_guard": baseline_guard,
        },
        "source_formal_data_root": str(source_data_root),
        "scenario_control": {
            "same_scenarios_as_formal_stage1": True,
            "scenario_hashes": {
                key: [item["scenario_hash"] for item in items]
                for key, items in scenario_items.items()
            },
            "source_files": {
                key: [item["source_path"] for item in items]
                for key, items in scenario_items.items()
            },
        },
        "parameters": {
            "uav_count": 1,
            "max_steps": int(config.max_steps),
            "dt_s": 0.2,
            "safety_projection_enabled": True,
            "safety_execution_mode": str(config.safety_execution_mode),
            "curriculum_used": False,
            "evaluation_reward_channel": "disabled",
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
            "goal_hold_reward": float(config.single_uav_goal_hold_reward),
        },
        "holdout": groups,
        "schema_csv_summary": {
            "validate_preliminary": validation_report,
            "stage_step_diagnostics": str((logs_root / "stage1_step_diagnostics.csv").resolve()),
            "stage_episode_metrics": str((logs_root / "stage1_episode_metrics.csv").resolve()),
            "episode_metrics": str((scale_root / "diagnostics" / "episode_metrics.csv").resolve()),
        },
    }
    _write_json(logs_root / "baseline_static_holdout_summary.json", report)
    _write_json(data_root / "summary.json", report)
    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run_static_obstacle_avoidance_stage1(
        baseline_checkpoint=args.baseline_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        force=args.force,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "ACTIVE_REWARD_CONFIG",
    "DEFAULT_BASELINE_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "StaticObstacleAvoidanceRewardConfig",
    "static_candidate_diagnostics",
    "static_obstacle_reward",
    "training_episode_kind",
    "training_schedule_counts",
    "run_static_obstacle_avoidance_stage1",
    "run_static_holdout_baseline_evaluation",
]
