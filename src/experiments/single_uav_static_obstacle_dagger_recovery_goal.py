"""Static-obstacle DAgger/BC for bypass recovery and goal arrival.

This entry point is intentionally independent from the PPO and Stage-1
static-obstacle experiments.  It uses the original single-UAV checkpoint as
the policy source, migrates its 39-dimensional observation contract to the
already validated Stage-2A 42-dimensional contract, and trains only the
Actor mean path with an estimated-state, action-constrained geometric Oracle.

The data protocol has three safeguards that are important for the current
failure mode: normal no-obstacle rollouts preserve the original navigation
skill; static mirror pairs provide both bypass sides; and a second closed-loop
DAgger recollection round exposes the candidate to its own post-bypass drift
states.  No PPO update, reward change, dynamics change, teleport, or new
termination rule is performed here.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import validate_episode_benchmark
from ..simulation.motion_model import apply_velocity_increment
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _scale_seed,
    _write_scale_manifest,
    validate_preliminary,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)
from .single_uav_bc_v2_dagger import (
    _batch_actor_physical_mean,
    _train_imitation,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_actor_constrained_oracle import actor_action_transform
from .single_uav_obstacle_dagger_stage2b import (
    _episode_step_row,
    _physical_obstacles,
    _static_candidate_diagnostics,
    _summary_from_rows,
    _write_rows_csv,
)
from .single_uav_obstacle_dagger_stage2br import (
    BoundaryAwareGeometricObstacleOracleController,
    _actor_decision,
    _normal_no_obstacle_scenario,
    _stage2br_config_from_payload,
    mirror_static_scenario,
)
from .single_uav_obstacle_direction_observation_stage2a import run_stage2a
from .single_uav_obstacle_stage2br2_followup import (
    _action_breakdown,
    _build_followup_row,
    _boundary_info,
    _obstacle_center_radius,
    _phase_for_step,
    _scenario_category,
    _scenario_side,
)
from .single_uav_oracle_behavior_cloning import (
    _canonical_hash,
    _copy_observation_for_json,
)
from .single_uav_rl_equivalent_oracle_audit import _extract_rl_visible_state


STAGE_NAME = "single_uav_static_obstacle_dagger_recovery_goal"
DEFAULT_BASELINE_CHECKPOINT = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_FINE_TUNED_CHECKPOINT = Path(
    "data/synthetic/single_uav_navigation_preserving_finetune_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_DAGGER_CHECKPOINT = Path(
    "data/synthetic/single_uav_dagger_actor_seed2026/"
    "scale_1/checkpoints/best_validation.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_static_obstacle_dagger_recovery_goal_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026"
)

OLD_NODE_DIM = 39
NEW_NODE_DIM = 42
DEFAULT_TRAIN_NORMAL = 24
DEFAULT_TRAIN_STATIC = 24
DEFAULT_VALIDATION_NORMAL = 8
DEFAULT_VALIDATION_STATIC = 8
DEFAULT_DAGGER_NORMAL = 12
DEFAULT_DAGGER_STATIC = 12
SMOKE_TRAIN_NORMAL = 4
SMOKE_TRAIN_STATIC = 4
SMOKE_VALIDATION_NORMAL = 2
SMOKE_VALIDATION_STATIC = 2
SMOKE_DAGGER_NORMAL = 2
SMOKE_DAGGER_STATIC = 2
PHASE_CAP_PER_EPISODE = 24
STATIC_SAFE_DISTANCE_M = 2.0
LATERAL_NONDEGENERATE_MPS = 0.01


class ActorConstrainedBoundaryAwareGeometricObstacleOracleController(
    BoundaryAwareGeometricObstacleOracleController
):
    """Obstacle-aware Oracle whose labels are reachable by the frozen Actor.

    The geometric controller's internal action is useful for route planning,
    but its historical Euclidean-norm projection is not the final MAPPO Actor
    action set.  The Actor emits a componentwise box action after tanh and
    scaling.  This adapter keeps the existing estimated-state route logic and
    applies that exact, already-existing Actor transform to the teacher label.
    It changes neither the environment action space nor the dynamics.
    """

    def __init__(self, environment: Any) -> None:
        super().__init__(environment)
        self.actor_max_delta_v_mps = float(
            self.motion.max_acceleration_mps2 * self.motion.dt_s
        )

    def action(
        self, observations: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_payload, base_control = super().action(observations)
        base_control = dict(base_control)
        base_delta = np.asarray(
            base_control.get(
                "raw_delta_v_mps",
                base_payload[self.uav_id]["delta_v"],
            ),
            dtype=float,
        ).reshape(3)
        transformed = actor_action_transform(
            base_delta,
            self.actor_max_delta_v_mps,
        )
        constrained = np.asarray(
            transformed["constrained_delta_v_mps"],
            dtype=float,
        ).reshape(3)
        # Preserve the geometric controller diagnostics, but make the fields
        # describing the action sent to the learner/environment unambiguous.
        control = {
            **base_control,
            "oracle_action_source": (
                "geometric_near_obstacle_bypass_or_goal_tracking_"
                "actor_component_constrained"
            ),
            "actor_action_constraint_applied": True,
            "actor_component_limit_mps": float(
                transformed["actor_component_limit_mps"]
            ),
            "actor_desired_delta_v_mps": transformed["desired_delta_v_mps"],
            "actor_component_limited_delta_v_mps": transformed[
                "actor_component_limited_delta_v_mps"
            ],
            "actor_scaled_delta_v_mps": transformed[
                "actor_scaled_delta_v_mps"
            ],
            "actor_constraint_delta_v_difference_norm_mps": float(
                transformed["actor_constraint_delta_v_difference_norm_mps"]
            ),
            "actor_constraint_affected": int(
                transformed["actor_constraint_affected"]
            ),
            "actor_any_axis_saturated": int(
                transformed["actor_any_axis_saturated"]
            ),
            "raw_controller_delta_v_mps": base_delta.tolist(),
            "raw_controller_delta_v_norm_mps": float(np.linalg.norm(base_delta)),
            "legal_delta_v_mps": constrained.tolist(),
            "legal_delta_v_norm_mps": float(np.linalg.norm(constrained)),
            "controller_delta_v_capped": int(
                np.linalg.norm(base_delta) > self.actor_max_delta_v_mps + 1.0e-9
            ),
            "teacher_action_componentwise_legal": int(
                bool(
                    np.all(
                        np.abs(constrained)
                        <= float(transformed["actor_component_limit_mps"])
                        + 1.0e-9
                    )
                )
            ),
        }
        self.last_control = control
        return {
            self.uav_id: {
                "delta_v": constrained.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }, control


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output root: {root.resolve()}")
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean(values: Iterable[Any]) -> float:
    numbers = [_finite(value) for value in values]
    numbers = [value for value in numbers if np.isfinite(value)]
    return float(np.mean(numbers)) if numbers else float("nan")


def _unit(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1.0e-12 else np.zeros(3, dtype=float)


def _visible_any(observation: Mapping[str, Any], config: FormalExperimentConfig) -> dict[str, Any]:
    """Extract only actor-visible state for either the 39D or 42D contract."""

    visible = _extract_rl_visible_state(
        observation,
        expected_node_dim=int(config.node_feature_dim),
    )
    node = np.asarray(observation["node_features"], dtype=float)
    ids = list(observation.get("node_ids", []))
    own_index = ids.index("uav_00") if "uav_00" in ids else 0
    row = node[own_index]
    yaw = float(row[8]) if row.size > 8 and np.isfinite(row[8]) else 0.0
    direction = row[39:42].copy() if row.size >= NEW_NODE_DIM else np.zeros(3, dtype=float)
    return {**visible, "yaw_rad": yaw, "obstacle_direction_body_m": direction}


def _normalizer_actor_observation(trainer: Any, observations: Mapping[str, Any]) -> dict[str, Any]:
    normalized = trainer.normalizer.transform(observations)
    result = copy.deepcopy(dict(normalized["uav_00"]))
    result["uav_id"] = "uav_00"
    return result


def _bc_action(trainer: Any, observations: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    actor_observation = _normalizer_actor_observation(trainer, observations)
    with torch.no_grad():
        action = _batch_actor_physical_mean(trainer, [actor_observation])[0]
    return action.detach().cpu().numpy().astype(float), actor_observation


def _scenario_metadata(
    scenario: Mapping[str, Any],
    *,
    split: str,
    kind: str,
    scenario_seed: int,
    noise_seed: int,
    round_index: int,
    pair_id: str,
) -> dict[str, Any]:
    return {
        "scenario_hash": _canonical_hash(scenario),
        "split": split,
        "kind": kind,
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "round_index": int(round_index),
        "pair_id": str(pair_id),
        "category": _scenario_category(scenario),
        "side": _scenario_side(scenario),
    }


def _static_variant(
    config: FormalExperimentConfig,
    *,
    split: str,
    ordinal: int,
    scenario_seed: int,
    pair_id: str,
    side: str,
    round_index: int,
) -> dict[str, Any]:
    """Create varied but paired direct-path static obstacle scenes."""

    category = "left_front" if ordinal % 4 in (2, 3) else "left"
    category = category if side == "left" else ("right_front" if category == "left_front" else "right")
    scenario = mirror_static_scenario(
        config,
        category=category,
        pair_index=7000 + int(ordinal),
        side=side,
        seed=int(scenario_seed),
        round_index=int(round_index),
    )
    # Vary the start/goal ray and the obstacle geometry together.  The left
    # and right members of a pair remain exact mirrors in body-frame lateral
    # sign, while train and validation use disjoint geometry/seed ranges.
    y_shift = float(((ordinal * 5) % 9) - 4)
    start_x = float(32.0 + (ordinal % 3) * 2.0)
    goal_x = float(66.0 + ((ordinal // 3) % 3 - 1) * 2.0)
    start_y = 100.0 + y_shift
    goal_y = start_y
    is_front = "front" in category
    longitudinal = 12.0 if is_front else 8.0
    lateral = 4.5 if is_front else 4.0
    radius = 4.8 if is_front else 4.5
    sign = 1.0 if side == "left" else -1.0
    scenario["initial_positions"]["uav_00"] = [start_x, start_y, 20.0]
    scenario["goals"]["uav_00"] = [goal_x, goal_y, 20.0]
    spec = scenario["obstacle_specs"][0]
    spec["center_m"] = [start_x + longitudinal, start_y + sign * lateral, 20.0]
    spec["radius_m"] = radius
    scenario["hierarchical_waypoint"] = [goal_x, goal_y, 20.0]
    scenario["stage2br"].update(
        {
            "scene_type": "static_mirror_pressure_recovery_goal",
            "split": split,
            "pair_id": str(pair_id),
            "scenario_seed": int(scenario_seed),
            "noise_seed": int(scenario_seed + 10000000),
            "round": int(round_index),
            "varied_start_goal": True,
            "holdout_geometry": split != "train",
        }
    )
    return scenario


def _normal_variant(
    config: FormalExperimentConfig,
    *,
    split: str,
    ordinal: int,
    scenario_seed: int,
    noise_seed: int,
    round_index: int,
    pair_id: str,
) -> dict[str, Any]:
    # Historical checkpoints may carry a fixed ``scenario_seed`` for their
    # formal run.  Explicit recollection pools must still vary the generated
    # scene across seeds, so disable only that fixed-seed override here.
    pool_config = config
    if config.scenario_seed is not None:
        from dataclasses import replace

        pool_config = replace(config, scenario_seed=None)
    scenario = _normal_no_obstacle_scenario(
        pool_config,
        seed=int(scenario_seed),
        round_index=int(round_index),
        episode_index=int(ordinal),
    )
    scenario["recovery_goal"] = _scenario_metadata(
        scenario,
        split=split,
        kind="normal_far_distance",
        scenario_seed=scenario_seed,
        noise_seed=noise_seed,
        round_index=round_index,
        pair_id=pair_id,
    )
    return scenario


def _build_pool(
    config: FormalExperimentConfig,
    *,
    split: str,
    normal_count: int,
    static_count: int,
    seed_base: int,
    round_index: int,
) -> list[dict[str, Any]]:
    if static_count % 2:
        raise ValueError("static_count must be even so left/right mirror pairs remain balanced")
    normal_items: list[dict[str, Any]] = []
    for index in range(int(normal_count)):
        scenario_seed = int(seed_base + 17 * index + 1)
        noise_seed = int(seed_base + 100000 + 31 * index + 7)
        pair_id = f"{split}_normal_{round_index}_{index:04d}"
        normal_items.append(
            {
                "scenario": _normal_variant(
                    config,
                    split=split,
                    ordinal=index,
                    scenario_seed=scenario_seed,
                    noise_seed=noise_seed,
                    round_index=round_index,
                    pair_id=pair_id,
                ),
                "kind": "normal_far_distance",
                "split": split,
                "scenario_seed": scenario_seed,
                "noise_seed": noise_seed,
                "pair_id": pair_id,
                "ordinal": index,
                "round_index": round_index,
            }
        )
    static_items: list[dict[str, Any]] = []
    pair_count = int(static_count) // 2
    for pair_index in range(pair_count):
        pair_id = f"{split}_static_{round_index}_{pair_index:04d}"
        for side_offset, side in enumerate(("left", "right")):
            ordinal = pair_index * 2 + side_offset
            scenario_seed = int(seed_base + 1000 + 43 * ordinal)
            noise_seed = int(seed_base + 200000 + 59 * ordinal + side_offset)
            scenario = _static_variant(
                config,
                split=split,
                ordinal=ordinal,
                scenario_seed=scenario_seed,
                pair_id=pair_id,
                side=side,
                round_index=round_index,
            )
            scenario["stage2br"]["noise_seed"] = noise_seed
            scenario["recovery_goal"] = _scenario_metadata(
                scenario,
                split=split,
                kind="static_obstacle",
                scenario_seed=scenario_seed,
                noise_seed=noise_seed,
                round_index=round_index,
                pair_id=pair_id,
            )
            static_items.append(
                {
                    "scenario": scenario,
                    "kind": "static_obstacle",
                    "split": split,
                    "scenario_seed": scenario_seed,
                    "noise_seed": noise_seed,
                    "pair_id": pair_id,
                    "ordinal": ordinal,
                    "round_index": round_index,
                }
            )
    # Blocked schedule: two normal episodes followed by a complete mirror
    # pair.  The block order is seeded, but pair members are never separated.
    rng = np.random.default_rng(int(seed_base) + int(round_index))
    rng.shuffle(normal_items)
    rng.shuffle(static_items)
    blocks: list[list[dict[str, Any]]] = []
    for index in range(pair_count):
        block = normal_items[index * 2 : (index + 1) * 2] + static_items[index * 2 : (index + 1) * 2]
        blocks.append(block)
    remainder = normal_items[pair_count * 2 :]
    if remainder:
        blocks.append(remainder)
    output: list[dict[str, Any]] = []
    for block in blocks:
        output.extend(block)
    for episode_index, item in enumerate(output, start=1):
        item["episode_index"] = episode_index
        item["episode_id"] = f"{split}_r{round_index}_episode_{episode_index:04d}"
        item["scenario"]["recovery_goal"]["episode_id"] = item["episode_id"]
    return output


def _static_or_normal_row_phase(
    *,
    item: Mapping[str, Any],
    phase: str,
    base_phase: str,
    student_closed_loop: bool,
) -> str:
    if item["kind"] == "normal_far_distance":
        return "normal"
    if not student_closed_loop and phase == "approach":
        return "teacher_bootstrap_approach"
    return str(phase)


def _post_bypass_metrics(rows: Sequence[Mapping[str, Any]], item: Mapping[str, Any]) -> dict[str, Any]:
    if item["kind"] == "normal_far_distance":
        return {
            "passed_obstacle": 0,
            "post_bypass_recovery": 0,
            "post_bypass_max_same_sign_lateral_steps": 0,
            "post_bypass_positive_radial_progress_steps": 0,
        }
    passed_indices = [
        index for index, row in enumerate(rows) if int(row.get("passed_obstacle_after", 0) or 0)
    ]
    if not passed_indices:
        return {
            "passed_obstacle": 0,
            "post_bypass_recovery": 0,
            "post_bypass_max_same_sign_lateral_steps": 0,
            "post_bypass_positive_radial_progress_steps": 0,
        }
    start = min(passed_indices)
    side = _scenario_side(item["scenario"])
    bypass_sign = -1.0 if side == "left" else 1.0
    current = 0
    maximum = 0
    positive = 0
    for row in rows[start:]:
        lateral = _finite(row.get("raw_lateral_action_mps"), 0.0)
        same_side = abs(lateral) >= LATERAL_NONDEGENERATE_MPS and lateral * bypass_sign > 0.0
        current = current + 1 if same_side else 0
        maximum = max(maximum, current)
        positive += int(_finite(row.get("true_radial_progress_mps"), 0.0) > 0.0)
    recovery = int(
        any(str(row.get("base_phase", "")) == "recovery" for row in rows[start:])
    )
    return {
        "passed_obstacle": 1,
        "post_bypass_recovery": recovery,
        "post_bypass_max_same_sign_lateral_steps": int(maximum),
        "post_bypass_positive_radial_progress_steps": int(positive),
    }


def _demo_record(
    *,
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    actor_observation: Mapping[str, Any],
    teacher_action: np.ndarray,
    student_action: np.ndarray,
    collection_mode: str,
    student_closed_loop: bool,
    teacher_query: bool,
    teacher_control: Mapping[str, Any],
) -> dict[str, Any]:
    visible_direction = [
        _finite(row.get("obstacle_relative_x_body_m"), 0.0),
        _finite(row.get("obstacle_relative_y_body_m"), 0.0),
        _finite(row.get("obstacle_relative_z_body_m"), 0.0),
    ]
    return {
        "dataset_id": "single_uav_static_obstacle_dagger_recovery_goal",
        "split": str(item["split"]),
        "episode_id": str(item["episode_id"]),
        "episode_index": int(item["episode_index"]),
        "step_index": int(row.get("step", 0)),
        "scenario_hash": _canonical_hash(item["scenario"]),
        "scenario_seed": int(item["scenario_seed"]),
        "noise_seed": int(item["noise_seed"]),
        "round_index": int(item["round_index"]),
        "pair_id": str(item["pair_id"]),
        "kind": str(item["kind"]),
        "category": _scenario_category(item["scenario"]),
        "side": _scenario_side(item["scenario"]),
        "phase": str(row.get("phase", "")),
        "base_phase": str(row.get("base_phase", "")),
        "collection_mode": collection_mode,
        "student_closed_loop": bool(student_closed_loop),
        "teacher_query": bool(teacher_query),
        "teacher_state_source": "Actor-visible estimated state + static map geometry",
        "truth_state_read_for_control": False,
        "observation": _copy_observation_for_json(actor_observation),
        "teacher_action_delta_v_mps": np.asarray(teacher_action, dtype=float).reshape(3).tolist(),
        "teacher_action_norm_mps": float(np.linalg.norm(np.asarray(teacher_action, dtype=float).reshape(3))),
        "teacher_actor_component_limit_mps": _finite(
            teacher_control.get("actor_component_limit_mps")
        ),
        "teacher_actor_constraint_affected": int(
            teacher_control.get("actor_constraint_affected", 0) or 0
        ),
        "teacher_actor_any_axis_saturated": int(
            teacher_control.get("actor_any_axis_saturated", 0) or 0
        ),
        "student_action_delta_v_mps": np.asarray(student_action, dtype=float).reshape(3).tolist(),
        "goal_distance_m": _finite(row.get("true_goal_distance_before_m")),
        "speed_mps": _finite(row.get("total_speed_before_mps")),
        "radial_velocity_mps": _finite(row.get("true_radial_velocity_before_mps")),
        "tangential_velocity_mps": _finite(row.get("tangential_speed_before_mps")),
        "goal_direction": [
            _finite(row.get("goal_direction_x"), 0.0),
            _finite(row.get("goal_direction_y"), 0.0),
            _finite(row.get("goal_direction_z"), 0.0),
        ],
        "obstacle_direction_body_m": visible_direction,
        "braking_component_mps": _finite(row.get("raw_braking_magnitude_mps"), 0.0),
        "action_norm_mps": _finite(row.get("raw_action_norm_mps"), 0.0),
        "radial_action_projection_mps": _finite(row.get("raw_radial_action_projection_mps"), 0.0),
        "tangential_action_mps": _finite(row.get("raw_tangential_action_mps"), 0.0),
        "boundary_projected": int(row.get("boundary_projected", 0) or 0),
        "safety_projected": int(row.get("safety_projected", 0) or 0),
        "action_clipped": int(row.get("action_clipped", 0) or 0),
        "raw_action_changed": int(row.get("raw_action_changed", 0) or 0),
    }


def _rollout_episode(
    *,
    student_trainer: Any,
    preservation_trainer: Any | None,
    config: FormalExperimentConfig,
    item: Mapping[str, Any],
    trace_root: Path,
    policy: str,
    collection_mode: str,
    include_demo: bool,
    evaluation_stage: str,
    oracle_factory: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reset_seed = int(item["noise_seed"])
    _seed_everything(reset_seed)
    environment = _make_environment(
        config,
        reset_seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    scenario = copy.deepcopy(dict(item["scenario"]))
    observations, info = environment.reset(seed=reset_seed, scenario_config=scenario)
    episode_id = str(item["episode_id"])
    writer = _EpisodeTraceWriter(
        episode_root=trace_root / str(item["split"]) / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{_canonical_hash(scenario)[:16]}",
        split=str(item["split"]),
        seed=reset_seed,
        phase=evaluation_stage,
        environment=environment,
        normalization_fitted=bool(student_trainer.normalizer.fitted),
        diagnostics_path=trace_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=policy,
    )
    oracle_constructor = (
        oracle_factory
        if oracle_factory is not None
        else ActorConstrainedBoundaryAwareGeometricObstacleOracleController
    )
    oracle = (
        oracle_constructor(environment)
        if item["kind"] == "static_obstacle"
        and bool(config.single_uav_obstacle_direction_features_enabled)
        else None
    )
    center, radius = _obstacle_center_radius(scenario)
    start_position = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)
    phase_state: dict[str, Any] = {
        "obstacle_center": center,
        "decision_seen": False,
        "decision_lateral_abs": 0.0,
    }
    passed_before = False
    first_pass_step: int | None = None
    first_boundary_step: int | None = None
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        state_before = environment.states["uav_00"].copy()
        visible = _visible_any(observations["uav_00"], config)
        if collection_mode == "teacher_bootstrap":
            if oracle is not None:
                teacher_payload, teacher_control = oracle.action(observations)
                teacher_action = np.asarray(teacher_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
            else:
                _decision, teacher_action, actor_observation = _actor_decision(
                    preservation_trainer or student_trainer, observations, info
                )
                teacher_control = {
                    "route_phase": "normal_preservation",
                    "desired_speed_mps": float("nan"),
                    "truth_state_read_for_control": False,
                }
            student_action = teacher_action.copy()
            actor_observation = _normalizer_actor_observation(student_trainer, observations)
            teacher_query = True
            student_closed_loop = False
        else:
            _decision, student_action, actor_observation = _actor_decision(
                student_trainer, observations, info
            )
            if item["kind"] == "static_obstacle":
                if oracle is not None:
                    teacher_payload, teacher_control = oracle.action(observations)
                    teacher_action = np.asarray(teacher_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
                    teacher_query = True
                else:
                    teacher_action = student_action.copy()
                    teacher_control = {
                        "route_phase": "static_evaluation_without_direction_teacher",
                        "desired_speed_mps": float("nan"),
                        "truth_state_read_for_control": False,
                    }
                    teacher_query = False
            elif preservation_trainer is not None:
                _preservation_decision, teacher_action, _ = _actor_decision(
                    preservation_trainer, observations, info
                )
                teacher_control = {
                    "route_phase": "normal_preservation",
                    "desired_speed_mps": float("nan"),
                    "truth_state_read_for_control": False,
                }
                teacher_query = True
            else:
                teacher_action = student_action.copy()
                teacher_control = {
                    "route_phase": "normal_student",
                    "desired_speed_mps": float("nan"),
                    "truth_state_read_for_control": False,
                }
                teacher_query = False
            student_closed_loop = True
        _assert_action_feasible(environment, {"uav_00": {"delta_v": student_action.tolist()}})
        nominal_state, _ = apply_velocity_increment(
            state_before,
            student_action,
            environment.config.motion,
            safety_projector=None,
        )
        pre_diag = _static_candidate_diagnostics(
            environment,
            np.asarray(state_before.position_m, dtype=float),
            np.asarray(nominal_state.position_m, dtype=float),
        )
        lateral_world = np.asarray(
            [-math.sin(float(visible["yaw_rad"])), math.cos(float(visible["yaw_rad"])), 0.0],
            dtype=float,
        )
        raw_lateral = float(np.dot(student_action, lateral_world))
        phase, base_phase = _phase_for_step(
            phase_state=phase_state,
            step=step,
            position_before=np.asarray(state_before.position_m, dtype=float),
            visible=visible,
            pre_diag=pre_diag,
            raw_lateral=raw_lateral,
            radial_progress=0.0,
            passed_before=passed_before,
            first_pass_step=first_pass_step,
            first_boundary_step=first_boundary_step,
        )
        next_observations, rewards, terminated, truncated, next_info = environment.step(
            {"uav_00": {"delta_v": student_action.tolist(), "measurement_uav_id": "none", "bandwidth": {}}}
        )
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        breakdown = _action_breakdown(state_before, student_action, environment, execution)
        base_row = _episode_step_row(
            policy=policy,
            scenario_name="static_obstacle" if item["kind"] == "static_obstacle" else "normal_far_distance",
            safety_mode="on",
            episode_index=int(item["episode_index"]),
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=student_action,
            control=teacher_control,
            nominal_position=np.asarray(nominal_state.position_m, dtype=float),
            pre_diag=pre_diag,
            next_info=next_info,
        )
        position_after = np.asarray(environment.states["uav_00"].position_m, dtype=float)
        passed_after = passed_before
        if item["kind"] == "static_obstacle" and np.isfinite(radius):
            passed_after = bool(position_after[0] > center[0] + radius + 0.5)
        if passed_after and not passed_before:
            first_pass_step = int(step)
        if bool(execution.get("boundary_projected", False)) and first_boundary_step is None:
            first_boundary_step = int(step)
        goal = np.asarray(environment.goals["uav_00"], dtype=float)
        before_distance = float(np.linalg.norm(goal - np.asarray(state_before.position_m, dtype=float)))
        after_distance = float(np.linalg.norm(goal - position_after))
        radial_progress = (before_distance - after_distance) / max(float(environment.config.motion.dt_s), 1.0e-12)
        if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
            phase = "boundary_onset"
        elif passed_before and radial_progress > 0.0 and abs(raw_lateral) <= max(
            LATERAL_NONDEGENERATE_MPS,
            float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5,
        ):
            phase = "recovery"
            base_phase = "recovery"
        phase = _static_or_normal_row_phase(
            item=item,
            phase=phase,
            base_phase=base_phase,
            student_closed_loop=student_closed_loop,
        )
        row = _build_followup_row(
            base_row=base_row,
            environment=environment,
            scenario=scenario,
            state_before=state_before,
            visible=visible,
            actions=breakdown,
            execution=execution,
            phase=phase,
            base_phase=base_phase,
            passed_before=passed_before,
            passed_after=passed_after,
            first_pass_step=first_pass_step,
            first_boundary_step=first_boundary_step,
            start_position=start_position,
        )
        row.update(
            {
                "evaluation_stage": evaluation_stage,
                "collection_mode": collection_mode,
                "student_closed_loop": int(student_closed_loop),
                "teacher_query": int(teacher_query),
                "episode_id": episode_id,
                "scenario_seed": int(item["scenario_seed"]),
                "noise_seed": int(item["noise_seed"]),
                "round_index": int(item["round_index"]),
                "pair_id": str(item["pair_id"]),
                "kind": str(item["kind"]),
                "teacher_delta_v_x_mps": float(teacher_action[0]),
                "teacher_delta_v_y_mps": float(teacher_action[1]),
                "teacher_delta_v_z_mps": float(teacher_action[2]),
                "teacher_action_norm_mps": float(np.linalg.norm(teacher_action)),
                "teacher_actor_component_limit_mps": _finite(
                    teacher_control.get("actor_component_limit_mps")
                ),
                "teacher_actor_constraint_affected": int(
                    teacher_control.get("actor_constraint_affected", 0) or 0
                ),
                "teacher_actor_any_axis_saturated": int(
                    teacher_control.get("actor_any_axis_saturated", 0) or 0
                ),
                "teacher_action_componentwise_legal": int(
                    teacher_control.get("teacher_action_componentwise_legal", 1) or 0
                ),
                "oracle_truth_state_read": int(bool(teacher_control.get("truth_state_read_for_control", False))),
                "oracle_route_phase": str(teacher_control.get("route_phase", "")),
                "oracle_desired_speed_mps": _finite(teacher_control.get("desired_speed_mps")),
                "raw_radial_action_projection_mps": float(np.dot(student_action, _unit(goal - np.asarray(state_before.position_m, dtype=float)))),
                "raw_braking_magnitude_mps": max(0.0, -float(np.dot(student_action, _unit(np.asarray(state_before.velocity_mps, dtype=float))))),
                "raw_tangential_action_mps": float(np.linalg.norm(student_action - np.dot(student_action, _unit(goal - np.asarray(state_before.position_m, dtype=float))) * _unit(goal - np.asarray(state_before.position_m, dtype=float)))),
                "negative_radial_velocity": int(float(base_row.get("true_radial_velocity_before_mps", 0.0)) < 0.0),
                "truth_state_read_for_control": False,
            }
        )
        # Preserve method-specific Teacher diagnostics without changing the
        # common environment/reward schema.  The external recovery study uses
        # these fields for soft-fusion and post-bypass release plots.
        for control_key, control_value in teacher_control.items():
            if str(control_key).startswith("soft_fusion_"):
                row[str(control_key)] = control_value
        rows.append(row)
        if include_demo:
            records.append(
                _demo_record(
                    item=item,
                    row=row,
                    actor_observation=actor_observation,
                    teacher_action=teacher_action,
                    student_action=student_action,
                    collection_mode=collection_mode,
                    student_closed_loop=student_closed_loop,
                    teacher_query=teacher_query,
                    teacher_control=teacher_control,
                )
            )
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions={"uav_00": {"delta_v": student_action.tolist(), "measurement_uav_id": "none", "bandwidth": {}}},
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=policy,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        passed_before = passed_after
        if terminated or truncated:
            break
    if first_pass_step is not None:
        for row in rows:
            row["first_obstacle_pass_step"] = int(first_pass_step)
    if first_boundary_step is not None:
        for row in rows:
            row["first_boundary_projection_step"] = int(first_boundary_step)
    writer_summary = writer.finalize()
    summary = _summary_from_rows(
        rows,
        policy=policy,
        scenario_name="static_obstacle" if item["kind"] == "static_obstacle" else "normal_far_distance",
        safety_mode="on",
        episode_index=int(item["episode_index"]),
    )
    summary.update(
        {
            "split": str(item["split"]),
            "kind": str(item["kind"]),
            "episode_id": episode_id,
            "scenario_hash": _canonical_hash(scenario),
            "scenario_seed": int(item["scenario_seed"]),
            "noise_seed": int(item["noise_seed"]),
            "round_index": int(item["round_index"]),
            "pair_id": str(item["pair_id"]),
            "collection_mode": collection_mode,
            "student_closed_loop": bool(student_closed_loop),
            "teacher_query": bool(teacher_query),
            "truth_state_read_for_control": False,
            "writer_summary": writer_summary,
            **_post_bypass_metrics(rows, item),
            "training_performed": False,
            "policy_updates_performed": False,
            "ppo_updates": 0,
        }
    )
    return summary, rows, records


def _evenly_cap(records: Sequence[Mapping[str, Any]], cap: int = PHASE_CAP_PER_EPISODE) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("episode_id", "")), str(record.get("phase", "")))].append(record)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= int(cap):
            selected.extend(dict(item) for item in group)
            continue
        indices = np.linspace(0, len(group) - 1, int(cap), dtype=int)
        selected.extend(dict(group[int(index)]) for index in sorted(set(indices.tolist())))
    selected.sort(key=lambda item: (str(item.get("episode_id", "")), int(item.get("step_index", 0))))
    return selected


def _stratified_episode_cap(
    records: Sequence[Mapping[str, Any]], cap: int
) -> list[dict[str, Any]]:
    """Cap one episode while retaining its phase strata deterministically."""

    if int(cap) < 0:
        raise ValueError("cap must be non-negative")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("phase", ""))].append(record)
    if not grouped or int(cap) == 0:
        return []
    total = sum(len(group) for group in grouped.values())
    if total <= int(cap):
        output = [dict(record) for record in records]
        output.sort(key=lambda item: int(item.get("step_index", 0)))
        return output

    phase_names = sorted(grouped)
    quotas = {phase: 0 for phase in phase_names}
    remaining = int(cap)
    # Keep one sample from every available phase whenever the cap permits it.
    if int(cap) >= len(phase_names):
        for phase in phase_names:
            quotas[phase] = 1
            remaining -= 1
    # Allocate the remaining slots proportionally, with deterministic largest
    # remainders.  This keeps approach/bypass/recovery evidence without
    # allowing a long phase to crowd every other phase out.
    if remaining > 0:
        weights = {
            phase: max(0, len(grouped[phase]) - quotas[phase]) for phase in phase_names
        }
        weight_total = sum(weights.values())
        if weight_total > 0:
            exact = {
                phase: remaining * weights[phase] / weight_total for phase in phase_names
            }
            for phase in phase_names:
                add = min(
                    weights[phase],
                    int(math.floor(exact[phase])),
                )
                quotas[phase] += add
            allocated = sum(quotas.values())
            leftovers = int(cap) - allocated
            order = sorted(
                phase_names,
                key=lambda phase: (
                    -(exact[phase] - math.floor(exact[phase])),
                    phase,
                ),
            )
            for phase in order:
                if leftovers <= 0:
                    break
                if quotas[phase] < len(grouped[phase]):
                    quotas[phase] += 1
                    leftovers -= 1

    selected: list[dict[str, Any]] = []
    for phase in phase_names:
        group = sorted(grouped[phase], key=lambda item: int(item.get("step_index", 0)))
        quota = min(len(group), int(quotas[phase]))
        if quota <= 0:
            continue
        indices = np.linspace(0, len(group) - 1, quota, dtype=int)
        selected.extend(dict(group[int(index)]) for index in sorted(set(indices.tolist())))
    selected.sort(key=lambda item: int(item.get("step_index", 0)))
    return selected


def _balance_demo_kinds(
    records: Sequence[Mapping[str, Any]], cap: int = PHASE_CAP_PER_EPISODE
) -> list[dict[str, Any]]:
    """Balance normal/static demonstration steps without mixing episodes.

    ``_evenly_cap`` intentionally caps every episode/phase independently.  A
    static episode has several phases, whereas a normal episode has one, so
    that operation otherwise creates a large static-sample majority.  This
    second, deterministic cap gives each episode at most ``cap`` rows with
    phase-stratified selection, then aligns the total sample count of each
    ``kind``.  It changes only which BC examples are presented; it does not
    alter rollout dynamics, rewards, or any evaluation distribution.
    """

    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_kind[str(record.get("kind", ""))].append(record)
    if len(by_kind) <= 1:
        return [dict(record) for record in records]

    capped_by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind, kind_records in by_kind.items():
        episodes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in kind_records:
            episodes[str(record.get("episode_id", ""))].append(record)
        capped: list[dict[str, Any]] = []
        for episode_id in sorted(episodes):
            capped.extend(_stratified_episode_cap(episodes[episode_id], int(cap)))
        capped_by_kind[kind] = capped
    target = min((len(rows) for rows in capped_by_kind.values()), default=0)

    balanced: list[dict[str, Any]] = []
    for kind in sorted(capped_by_kind):
        kind_rows = capped_by_kind[kind]
        episodes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in kind_rows:
            episodes[str(record.get("episode_id", ""))].append(record)
        episode_ids = sorted(episodes)
        if len(kind_rows) <= target or not episode_ids:
            balanced.extend(dict(record) for record in kind_rows)
            continue
        base = target // len(episode_ids)
        extra = target % len(episode_ids)
        for index, episode_id in enumerate(episode_ids):
            quota = base + int(index < extra)
            balanced.extend(
                _stratified_episode_cap(episodes[episode_id], quota)
            )
    balanced.sort(
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("episode_id", "")),
            int(item.get("step_index", 0)),
        )
    )
    return balanced


def _write_demo_dataset(
    data_root: Path,
    logs_root: Path,
    *,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    train_phase_capped = _evenly_cap(train_records)
    validation_phase_capped = _evenly_cap(validation_records)
    train = _balance_demo_kinds(train_phase_capped)
    validation = _balance_demo_kinds(validation_phase_capped)
    train_ids = {(str(row.get("scenario_hash")), str(row.get("episode_id"))) for row in train}
    validation_ids = {(str(row.get("scenario_hash")), str(row.get("episode_id"))) for row in validation}
    train_pairs = {str(row.get("pair_id")) for row in train}
    validation_pairs = {str(row.get("pair_id")) for row in validation}
    if train_ids & validation_ids or train_pairs & validation_pairs:
        raise AssertionError("demonstration train/validation leakage detected")
    phase_counts = Counter(str(row.get("phase", "")) for row in all_records)
    collection_counts = Counter(str(row.get("collection_mode", "")) for row in all_records)
    split_counts = Counter(str(row.get("split", "")) for row in all_records)
    manifest = {
        "dataset_id": STAGE_NAME,
        "source": "estimated-state Actor-constrained BoundaryAwareGeometricObstacleOracle plus baseline normal-preservation labels",
        "teacher_truth_state_read_for_control": False,
        "train_samples_raw": len(train_records),
        "validation_samples_raw": len(validation_records),
        "train_samples_balanced": len(train),
        "validation_samples_balanced": len(validation),
        "train_episode_count": len({str(row.get("episode_id")) for row in train}),
        "validation_episode_count": len({str(row.get("episode_id")) for row in validation}),
        "all_samples_before_balance": len(all_records),
        "phase_counts_raw": dict(phase_counts),
        "collection_mode_counts": dict(collection_counts),
        "split_counts": dict(split_counts),
        "train_episode_ids": sorted({str(row.get("episode_id")) for row in train}),
        "validation_episode_ids": sorted({str(row.get("episode_id")) for row in validation}),
        "train_pair_ids": sorted(train_pairs),
        "validation_pair_ids": sorted(validation_pairs),
        "trajectory_leakage": bool(train_ids & validation_ids),
        "pair_leakage": bool(train_pairs & validation_pairs),
        "phase_cap_per_episode": int(PHASE_CAP_PER_EPISODE),
        "kind_balance": {
            "enabled": True,
            "policy": "phase_stratified_per_episode_cap_then_equal_total_steps",
            "kind_counts_balanced": dict(Counter(str(row.get("kind", "")) for row in train)),
            "validation_kind_counts_balanced": dict(Counter(str(row.get("kind", "")) for row in validation)),
        },
        "pool": pool_manifest,
        "files": {
            "train": str((data_root / "demonstrations" / "train.jsonl").resolve()),
            "validation": str((data_root / "demonstrations" / "validation.jsonl").resolve()),
        },
    }
    _write_jsonl(data_root / "demonstrations" / "train.jsonl", train)
    _write_jsonl(data_root / "demonstrations" / "validation.jsonl", validation)
    _write_rows_csv(data_root / "demonstrations" / "train.csv", train)
    _write_rows_csv(data_root / "demonstrations" / "validation.csv", validation)
    _write_rows_csv(logs_root / "demonstration_step_audit.csv", list(all_records))
    _write_json(data_root / "demonstrations" / "manifest.json", _jsonable(manifest))
    return manifest


def _load_checkpoint(path: Path, output_dir: Path) -> tuple[Any, FormalExperimentConfig, dict[str, Any], dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(
        payload,
        output_dir=output_dir,
        command_line=f"{STAGE_NAME}: frozen policy",
    )
    trainer, hashes, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    identity = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
        "loaded_policy_hash": hashes.get("loaded_policy_hash", ""),
        "node_feature_dim": int(config.node_feature_dim),
    }
    return trainer, config, dict(payload), identity


def _migrate_baseline(
    baseline_checkpoint: Path,
    *,
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    progress: bool,
) -> tuple[Path, dict[str, Any]]:
    migration_data = data_root / "baseline_observation_migration"
    migration_logs = logs_root / "baseline_observation_migration"
    migration_plots = plots_root / "baseline_observation_migration"
    result = run_stage2a(
        stage1_checkpoint=baseline_checkpoint,
        baseline_checkpoint=baseline_checkpoint,
        data_root=migration_data,
        logs_root=migration_logs,
        plots_root=migration_plots,
        progress=progress,
    )
    migrated = migration_data / "checkpoints" / "stage2a_observation_compatible_init.pt"
    if not migrated.is_file():
        raise FileNotFoundError(f"Stage 2A migration did not create {migrated}")
    import torch

    payload = torch.load(migrated, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=migration_data, progress=False)
    if int(config.node_feature_dim) != NEW_NODE_DIM or not bool(config.single_uav_obstacle_direction_features_enabled):
        raise AssertionError("migrated baseline does not have the required 42D direction-observation contract")
    return migrated, {
        "source_checkpoint": str(baseline_checkpoint.resolve()),
        "source_sha256": _sha256(baseline_checkpoint),
        "source_node_feature_dim": OLD_NODE_DIM,
        "migrated_checkpoint": str(migrated.resolve()),
        "migrated_sha256": _sha256(migrated),
        "migrated_node_feature_dim": int(config.node_feature_dim),
        "migration_summary": result,
        "source_not_modified": True,
    }


def _aggregate(summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(summaries)

    def count_field(field: str) -> int:
        return int(sum(int(row.get(field, 0) or 0) for row in summaries))

    return {
        "episodes": count,
        "entered_8m": count_field("entered_8m"),
        "entered_8m_rate": count_field("entered_8m") / count if count else 0.0,
        "entered_5m": count_field("entered_5m"),
        "entered_5m_rate": count_field("entered_5m") / count if count else 0.0,
        "entered_3m": count_field("entered_3m"),
        "entered_3m_rate": count_field("entered_3m") / count if count else 0.0,
        "entered_1m": count_field("entered_1m"),
        "entered_1m_rate": count_field("entered_1m") / count if count else 0.0,
        "stable_goal": count_field("stable_goal"),
        "stable_goal_rate": count_field("stable_goal") / count if count else 0.0,
        "passed_obstacle": count_field("passed_obstacle"),
        "post_bypass_recovery": count_field("post_bypass_recovery"),
        "post_bypass_recovery_rate": count_field("post_bypass_recovery") / count if count else 0.0,
        "mean_minimum_goal_distance_m": _mean(row.get("minimum_goal_distance_m") for row in summaries),
        "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in summaries),
        "mean_speed_within_1m_mps": _mean(row.get("mean_speed_within_1m_mps") for row in summaries),
        "mean_speed_at_8m_entry_mps": _mean(row.get("mean_speed_at_8m_entry_mps") for row in summaries),
        "mean_speed_at_5m_entry_mps": _mean(row.get("mean_speed_at_5m_entry_mps") for row in summaries),
        "mean_speed_at_3m_entry_mps": _mean(row.get("mean_speed_at_3m_entry_mps") for row in summaries),
        "mean_speed_at_1m_entry_mps": _mean(row.get("mean_speed_at_1m_entry_mps") for row in summaries),
        "mean_maximum_consecutive_goal_hold_steps": _mean(row.get("maximum_consecutive_goal_hold_steps") for row in summaries),
        "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in summaries), default=0),
        "hold_steps_ge_1": count_field("hold_steps_ge_1"),
        "hold_steps_ge_3": count_field("hold_steps_ge_3"),
        "hold_steps_ge_5": count_field("hold_steps_ge_5"),
        "hold_steps_ge_1_rate": count_field("hold_steps_ge_1") / count if count else 0.0,
        "hold_steps_ge_3_rate": count_field("hold_steps_ge_3") / count if count else 0.0,
        "hold_steps_ge_5_rate": count_field("hold_steps_ge_5") / count if count else 0.0,
        "mean_pre_candidate_clearance_m": _mean(row.get("minimum_pre_clearance_m") for row in summaries),
        "mean_post_clearance_m": _mean(row.get("minimum_post_clearance_m") for row in summaries),
        "pre_candidate_static_risk_steps": int(sum(int(row.get("pre_candidate_static_swept_conflict_steps", 0) or 0) + int(row.get("pre_candidate_static_unsafe_steps", 0) or 0) for row in summaries)),
        "safety_projection_steps": int(sum(int(row.get("safety_projection_steps", 0) or 0) for row in summaries)),
        "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in summaries)),
        "action_clipping_fraction_mean": _mean(row.get("action_clipping_fraction") for row in summaries),
        "mean_post_bypass_max_same_sign_lateral_steps": _mean(row.get("post_bypass_max_same_sign_lateral_steps") for row in summaries),
        "maximum_post_bypass_same_sign_lateral_steps": max((int(row.get("post_bypass_max_same_sign_lateral_steps", 0) or 0) for row in summaries), default=0),
        "negative_radial_velocity_fraction": _mean(row.get("negative_radial_velocity") for row in rows),
        "mean_raw_action_norm_mps": _mean(row.get("raw_action_norm_mps") for row in rows),
        "mean_raw_radial_action_projection_mps": _mean(row.get("raw_radial_action_projection_mps") for row in rows),
        "mean_raw_tangential_action_mps": _mean(row.get("raw_tangential_action_mps") for row in rows),
        "mean_raw_braking_magnitude_mps": _mean(row.get("raw_braking_magnitude_mps") for row in rows),
        "collision": count_field("collision"),
        "timeout": int(sum(str(row.get("termination_reason", "")) == "time_limit" for row in summaries)),
    }


def _evaluation_row_plot(
    plot_root: Path,
    aggregates: Mapping[str, Mapping[str, Any]],
    static_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root.mkdir(parents=True, exist_ok=True)
    roles = list(aggregates)
    labels = [str(role).replace("_", "\n") for role in roles]
    x = np.arange(len(roles), dtype=float)
    width = 0.18
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for offset, field, label, color in (
        (-1.5 * width, "entered_8m_rate", "8 m", "#0072B2"),
        (-0.5 * width, "entered_3m_rate", "3 m", "#E69F00"),
        (0.5 * width, "entered_1m_rate", "1 m", "#CC79A7"),
        (1.5 * width, "stable_goal_rate", "stable", "#009E73"),
    ):
        axis.bar(x + offset, [100.0 * float(aggregates[role].get(field, 0.0)) for role in roles], width=width, label=label, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Episodes (%)")
    axis.set_title("Static-obstacle bypass and goal-arrival comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = plot_root / "entry_and_stable_comparison.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    selected = [row for row in static_rows if int(row.get("episode_index", 0) or 0) == 1]
    for role in roles:
        rows = [row for row in selected if row.get("policy") == role]
        if not rows:
            continue
        axis.plot(
            [float(row.get("position_after_x_m", np.nan)) for row in rows],
            [float(row.get("position_after_y_m", np.nan)) for row in rows],
            linewidth=1.6,
            label=str(role),
        )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Static holdout trajectories, paired episode 1")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    path_xy = plot_root / "static_holdout_trajectories.png"
    figure.savefig(path_xy, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"entry_and_stable": str((plot_root / "entry_and_stable_comparison.png").resolve()), "static_xy": str(path_xy.resolve())}


def _load_optional_roles(
    *,
    output_dir: Path,
    baseline_path: Path,
    bc_path: Path | None = None,
    dagger_path: Path | None = None,
    fine_tuned_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, FormalExperimentConfig], dict[str, Any], dict[str, Any]]:
    paths: dict[str, Path] = {"baseline_rl": baseline_path}
    for role, path in (("dagger_round1", dagger_path), ("navigation_preserving_fine_tuned", fine_tuned_path), ("bc_actor", bc_path)):
        if path is not None and Path(path).is_file():
            paths[role] = Path(path)
    trainers: dict[str, Any] = {}
    configs: dict[str, FormalExperimentConfig] = {}
    payloads: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for role, path in paths.items():
        trainer, config, payload, identity = _load_checkpoint(path.resolve(), output_dir / role)
        trainers[role] = trainer
        configs[role] = config
        payloads[role] = payload
        identities[role] = identity
    return trainers, configs, payloads, identities


def _evaluate_roles(
    *,
    roles: Mapping[str, Any],
    configs: Mapping[str, FormalExperimentConfig],
    oracle_trainer: Any,
    oracle_config: FormalExperimentConfig,
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    progress: bool,
    oracle_factory: Any | None = None,
) -> dict[str, Any]:
    all_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    validation_reports: dict[str, Any] = {}
    aggregates: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in pools}
    for split, pool in pools.items():
        for role, trainer in roles.items():
            role_root = data_root / "evaluation" / split / role / "scale_1"
            role_root.mkdir(parents=True, exist_ok=True)
            role_summaries: list[dict[str, Any]] = []
            role_rows: list[dict[str, Any]] = []
            for item in pool:
                eval_item = dict(item)
                eval_item["split"] = "test" if split == "static_holdout" else "test"
                # The project schema validator discovers benchmark episodes by
                # the canonical ``episode_*`` directory prefix.
                # Keep the per-role episode name short enough for Windows
                # MAX_PATH while retaining the schema-required prefix.
                eval_item["episode_id"] = f"episode_{int(item['episode_index']):06d}"
                if role == "oracle_teacher":
                    trainer_for_rollout = oracle_trainer
                    config_for_rollout = oracle_config
                    mode = "teacher_bootstrap"
                else:
                    trainer_for_rollout = trainer
                    config_for_rollout = configs[role]
                    mode = "evaluation"
                summary, rows, _ = _rollout_episode(
                    student_trainer=trainer_for_rollout,
                    preservation_trainer=None,
                    config=config_for_rollout,
                    item=eval_item,
                    trace_root=role_root,
                    policy=role,
                    collection_mode=mode,
                    include_demo=False,
                    evaluation_stage=f"bc_only_{split}",
                    oracle_factory=oracle_factory,
                )
                summary["evaluation_family"] = split
                role_summaries.append(summary)
                # Keep split/role metadata on every step row as well as on the
                # episode summary.  The plotting and CSV audit paths use the
                # step table to select paired static-holdout trajectories.
                for row in rows:
                    row["evaluation_family"] = split
                    row["evaluation_role"] = role
                role_rows.extend(rows)
                if progress and (int(item["episode_index"]) == 1 or int(item["episode_index"]) == len(pool)):
                    print(f"[{STAGE_NAME}] evaluation {split} {role} {item['episode_index']}/{len(pool)}", flush=True)
            _copy_calibration_files(role_root, config_for_rollout)
            manifest = _write_scale_manifest(scale_root=role_root, config=config_for_rollout, episodes=[row["writer_summary"] for row in role_summaries])
            manifest.update({"dataset_id": STAGE_NAME, "evaluation_only": True, "training_performed": False, "policy_updates_performed": False, "policy_source": role, "curriculum_used": False})
            _write_json(role_root / "manifest.json", _jsonable(manifest))
            validation_reports.setdefault(split, {})[role] = validate_episode_benchmark(role_root)
            aggregates[split][role] = _aggregate(role_summaries, role_rows)
            all_summaries.extend(role_summaries)
            all_rows.extend(role_rows)
    _write_rows_csv(logs_root / "evaluation_episode_metrics.csv", all_summaries)
    _write_rows_csv(logs_root / "evaluation_step_metrics.csv", all_rows)
    overall = {
        split: {role: value for role, value in split_values.items()}
        for split, split_values in aggregates.items()
    }
    static_rows = [row for row in all_rows if row.get("evaluation_family") == "static_holdout"]
    plot_aggregates = {role: aggregates["static_holdout"].get(role, {}) for role in aggregates.get("static_holdout", {})}
    plots = _evaluation_row_plot(plots_root, plot_aggregates, static_rows)
    return {
        "aggregates": overall,
        "overall_static_holdout": plot_aggregates,
        "schema_validation": validation_reports,
        "plots": plots,
        "files": {
            "episode_metrics": str((logs_root / "evaluation_episode_metrics.csv").resolve()),
            "step_metrics": str((logs_root / "evaluation_step_metrics.csv").resolve()),
        },
    }


def _reference_normal_pool(
    config: FormalExperimentConfig,
    *,
    count: int,
    reference_root: Path,
    round_index: int,
) -> list[dict[str, Any]]:
    """Reuse the original formal normal-test physical scenes for paired eval.

    Demonstration collection intentionally varies scenes.  Evaluation of the
    normal branch has a different contract: it must be directly comparable to
    the historical baseline, so load the baseline episode ``scenario_generation``
    payloads and their recorded reset/noise seeds without changing the policy or
    environment.
    """

    scenario_files = sorted(Path(reference_root).glob("episode_*/scenario.json"))
    if len(scenario_files) < int(count):
        raise FileNotFoundError(
            f"normal evaluation requires {count} baseline test scenarios under {reference_root}; "
            f"found {len(scenario_files)}"
        )
    output: list[dict[str, Any]] = []
    for index, path in enumerate(scenario_files[: int(count)]):
        payload = json.loads(path.read_text(encoding="utf-8"))
        generation = payload.get("scenario_generation")
        if not isinstance(generation, Mapping):
            raise ValueError(f"baseline scenario lacks scenario_generation: {path}")
        scenario = copy.deepcopy(dict(generation))
        scenario_seed = int(generation.get("seed", config.scenario_seed or config.seed))
        noise_seed = int(payload.get("seed", config.seed))
        pair_id = f"reference_normal_test_{index:04d}"
        episode_id = f"test_reference_episode_{index + 1:04d}"
        scenario["recovery_goal"] = _scenario_metadata(
            scenario,
            split="test",
            kind="normal_far_distance",
            scenario_seed=scenario_seed,
            noise_seed=noise_seed,
            round_index=round_index,
            pair_id=pair_id,
        )
        scenario["recovery_goal"]["reference_scenario_file"] = str(path.resolve())
        output.append(
            {
                "scenario": scenario,
                "kind": "normal_far_distance",
                "split": "test",
                "scenario_seed": scenario_seed,
                "noise_seed": noise_seed,
                "pair_id": pair_id,
                "ordinal": index,
                "round_index": int(round_index),
                "episode_index": index + 1,
                "episode_id": episode_id,
            }
        )
    return output


def _make_eval_pool(
    config: FormalExperimentConfig,
    *,
    normal_count: int,
    static_count: int,
    normal_reference_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    normal = _reference_normal_pool(
        config,
        count=normal_count,
        reference_root=normal_reference_root,
        round_index=99,
    )
    static = _build_pool(config, split="test", normal_count=0, static_count=static_count, seed_base=4020000, round_index=99)
    return {"normal_far_distance": normal, "static_holdout": static}


def _summary_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {STAGE_NAME}",
        "",
        "本实验只进行估计状态 Oracle 数据采集、Actor-only BC 和冻结评估；没有 PPO 更新。",
        "",
        "## 关键约束",
        "",
        "- 未修改 reward、braking_progress_reward_coef、target-speed、terminal curriculum、goal hold、stable 判据、动作空间、动作映射、动力学、horizon 或 PPO 参数。",
        "- teacher 使用 Actor 可见的估计状态与静态地图几何；`truth_state_read_for_control=false`。",
        "- train/validation 按独立 scenario/pair 划分；step 只用于诊断，不作为独立重复单位。",
        "- validation/test 不使用 curriculum；输出目录与历史结果独立。",
        "",
        "## 数据与训练",
        "",
    ]
    dataset = summary.get("dataset", {})
    lines.extend([
        f"- 训练轨迹={dataset.get('train_episode_count', 0)}，验证轨迹={dataset.get('validation_episode_count', 0)}，train samples={dataset.get('train_samples_balanced', 0)}，validation samples={dataset.get('validation_samples_balanced', 0)}。",
        f"- phase raw counts={json.dumps(dataset.get('phase_counts_raw', {}), ensure_ascii=False)}。",
        f"- collection modes={json.dumps(dataset.get('collection_mode_counts', {}), ensure_ascii=False)}。",
        f"- trajectory leakage={dataset.get('trajectory_leakage')}，pair leakage={dataset.get('pair_leakage')}。",
        f"- BC checkpoint={summary.get('bc_checkpoint', '')}。",
        "",
        "## 冻结评估",
        "",
    ])
    for split, values in dict(summary.get("evaluation", {}).get("aggregates", {})).items():
        lines.append(f"### {split}")
        lines.append("")
        for role, aggregate in values.items():
            lines.append(
                f"- {role}: 8m={aggregate.get('entered_8m', 0)}/{aggregate.get('episodes', 0)}, 3m={aggregate.get('entered_3m', 0)}/{aggregate.get('episodes', 0)}, 1m={aggregate.get('entered_1m', 0)}/{aggregate.get('episodes', 0)}, stable={aggregate.get('stable_goal', 0)}/{aggregate.get('episodes', 0)}, passed={aggregate.get('passed_obstacle', 0)}/{aggregate.get('episodes', 0)}, recovery={aggregate.get('post_bypass_recovery', 0)}/{aggregate.get('episodes', 0)}, boundary_steps={aggregate.get('boundary_projection_steps', 0)}"
            )
    lines.extend([
        "",
        "## 结论边界",
        "",
        "BC-only 结果用于判断原始 Actor 学习/状态分布问题；即使 collision=0，也必须结合 pre-risk、raw action、绕障通过和 post-bypass recovery 才能认定主动避障。该入口不会自动进入 PPO+BC anchor。",
        "",
    ])
    return "\n".join(lines)


def run_static_obstacle_dagger_recovery_goal(
    *,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    dagger_checkpoint: str | Path = DEFAULT_DAGGER_CHECKPOINT,
    fine_tuned_checkpoint: str | Path = DEFAULT_FINE_TUNED_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    smoke: bool = False,
    progress: bool = True,
    oracle_factory: Any | None = None,
) -> dict[str, Any]:
    """Run recollection, Actor-only BC and frozen normal/static evaluation."""

    data_root, logs_root, plots_root = Path(data_root), Path(logs_root), Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    baseline_path = Path(baseline_checkpoint).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    migrated_path, migration = _migrate_baseline(
        baseline_path,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
        progress=progress,
    )
    baseline_trainer, config, baseline_payload, baseline_identity = _load_checkpoint(
        migrated_path, data_root / "baseline_42_source"
    )
    if config.node_feature_dim != NEW_NODE_DIM:
        raise AssertionError("recovery DAgger requires the migrated 42D Actor contract")
    train_normal = SMOKE_TRAIN_NORMAL if smoke else DEFAULT_TRAIN_NORMAL
    train_static = SMOKE_TRAIN_STATIC if smoke else DEFAULT_TRAIN_STATIC
    validation_normal = SMOKE_VALIDATION_NORMAL if smoke else DEFAULT_VALIDATION_NORMAL
    validation_static = SMOKE_VALIDATION_STATIC if smoke else DEFAULT_VALIDATION_STATIC
    dagger_normal = SMOKE_DAGGER_NORMAL if smoke else DEFAULT_DAGGER_NORMAL
    dagger_static = SMOKE_DAGGER_STATIC if smoke else DEFAULT_DAGGER_STATIC

    initial_train_pool = _build_pool(
        config,
        split="train",
        normal_count=train_normal,
        static_count=train_static,
        seed_base=3100000,
        round_index=1,
    )
    validation_pool = _build_pool(
        config,
        split="validation",
        normal_count=validation_normal,
        static_count=validation_static,
        seed_base=3200000,
        round_index=1,
    )
    # The first static half is teacher-bootstrap coverage; the other half is
    # a real baseline-student closed-loop DAgger rollout.  Normal trajectories
    # always carry baseline-preservation labels.
    bootstrap_static_ids = {item["pair_id"] for item in initial_train_pool if item["kind"] == "static_obstacle"}
    bootstrap_static_ids = set(sorted(bootstrap_static_ids)[: max(1, len(bootstrap_static_ids) // 2)])
    train_summaries: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    validation_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for item in initial_train_pool:
        mode = "teacher_bootstrap" if item["kind"] == "static_obstacle" and item["pair_id"] in bootstrap_static_ids else "student_closed_loop"
        summary, rows, records = _rollout_episode(
            student_trainer=baseline_trainer,
            preservation_trainer=baseline_trainer,
            config=config,
            item=item,
            trace_root=data_root / "demonstrations" / "initial_traces",
            policy="baseline_student" if mode != "teacher_bootstrap" else "oracle_teacher",
            collection_mode=mode,
            include_demo=True,
            evaluation_stage="dagger_initial_collection",
            oracle_factory=oracle_factory,
        )
        train_summaries.append(summary)
        train_rows.extend(rows)
        train_records.extend(records)
        all_records.extend(records)
    for item in validation_pool:
        mode = "teacher_bootstrap" if item["kind"] == "static_obstacle" else "student_closed_loop"
        summary, rows, records = _rollout_episode(
            student_trainer=baseline_trainer,
            preservation_trainer=baseline_trainer,
            config=config,
            item=item,
            trace_root=data_root / "demonstrations" / "validation_traces",
            policy="validation_teacher" if mode == "teacher_bootstrap" else "validation_baseline",
            collection_mode=mode,
            include_demo=True,
            evaluation_stage="dagger_validation_collection",
            oracle_factory=oracle_factory,
        )
        validation_summaries.append(summary)
        validation_records.extend(records)
        all_records.extend(records)
    pool_manifest = {
        "initial_train_counts": dict(Counter(str(item["kind"]) for item in initial_train_pool)),
        "validation_counts": dict(Counter(str(item["kind"]) for item in validation_pool)),
        "initial_train_episode_ids": [str(item["episode_id"]) for item in initial_train_pool],
        "validation_episode_ids": [str(item["episode_id"]) for item in validation_pool],
        "train_static_bootstrap_pair_ids": sorted(bootstrap_static_ids),
        "train_student_closed_loop": True,
    }
    dataset_manifest_initial = _write_demo_dataset(
        data_root,
        logs_root,
        train_records=train_records,
        validation_records=validation_records,
        all_records=all_records,
        pool_manifest=pool_manifest,
    )
    _write_rows_csv(logs_root / "initial_train_episode_metrics.csv", train_summaries)
    _write_rows_csv(logs_root / "validation_demo_episode_metrics.csv", validation_summaries)

    # Train a temporary BC candidate only to create the next student-state
    # distribution.  The final BC is retrained from the same migrated baseline
    # after the new DAgger recollection has been appended.
    round0_root = data_root / "bc_round0"
    round0_logs = logs_root / "bc_round0"
    round0_plots = plots_root / "bc_round0"
    round0 = _train_imitation(
        migrated_path,
        data_root,
        output_dir=round0_root,
        logs_dir=round0_logs,
        plots_dir=round0_plots,
        max_epochs=8 if smoke else 30,
        patience=3 if smoke else 6,
        batch_size=32,
        learning_rate=1.0e-4,
        seed=20262701,
        stage="recovery_goal_dagger_round0",
        force=False,
        progress=progress,
    )
    round0_checkpoint = Path(str(round0["best_validation_checkpoint"]))
    candidate_trainer, candidate_config, _candidate_payload, candidate_identity = _load_checkpoint(
        round0_checkpoint, data_root / "bc_round0_source"
    )
    # New scenarios are disjoint from both initial and validation pools.  The
    # candidate is now rolled out closed-loop; Oracle labels are queried at
    # the states it actually visits, which is the key post-bypass correction.
    dagger_pool = _build_pool(
        config,
        split="train",
        normal_count=dagger_normal,
        static_count=dagger_static,
        seed_base=3300000,
        round_index=2,
    )
    dagger_summaries: list[dict[str, Any]] = []
    dagger_rows: list[dict[str, Any]] = []
    dagger_records: list[dict[str, Any]] = []
    for item in dagger_pool:
        summary, rows, records = _rollout_episode(
            student_trainer=candidate_trainer,
            preservation_trainer=baseline_trainer,
            config=candidate_config,
            item=item,
            trace_root=data_root / "dagger_round2_traces",
            policy="bc_round0_student",
            collection_mode="student_closed_loop",
            include_demo=True,
            evaluation_stage="dagger_recollection_round2",
            oracle_factory=oracle_factory,
        )
        dagger_summaries.append(summary)
        dagger_rows.extend(rows)
        dagger_records.extend(records)
        all_records.extend(records)
    _write_rows_csv(logs_root / "dagger_round2_episode_metrics.csv", dagger_summaries)
    _write_rows_csv(logs_root / "dagger_round2_step_audit.csv", dagger_rows)

    final_train_records = train_records + dagger_records
    final_dataset_manifest = _write_demo_dataset(
        data_root,
        logs_root,
        train_records=final_train_records,
        validation_records=validation_records,
        all_records=all_records,
        pool_manifest={
            **pool_manifest,
            "dagger_round2_counts": dict(Counter(str(item["kind"]) for item in dagger_pool)),
            "dagger_round2_episode_ids": [str(item["episode_id"]) for item in dagger_pool],
            "dagger_round2_student_closed_loop": True,
        },
    )
    final_root = data_root / "bc_final"
    final_logs = logs_root / "bc_final"
    final_plots = plots_root / "bc_final"
    final_bc = _train_imitation(
        migrated_path,
        data_root,
        output_dir=final_root,
        logs_dir=final_logs,
        plots_dir=final_plots,
        max_epochs=12 if smoke else 45,
        patience=4 if smoke else 8,
        batch_size=32,
        learning_rate=1.0e-4,
        seed=20262702,
        stage="recovery_goal_dagger_final_bc",
        force=False,
        progress=progress,
    )
    final_checkpoint = Path(str(final_bc["best_validation_checkpoint"]))
    final_trainer, final_config, final_payload, final_identity = _load_checkpoint(
        final_checkpoint, data_root / "bc_final_source"
    )

    eval_config = final_config
    normal_eval_count = 4 if smoke else 20
    static_eval_count = 4 if smoke else 20
    normal_reference_root = baseline_path.parents[2] / "scale_1" / "test"
    eval_pools = _make_eval_pool(
        eval_config,
        normal_count=normal_eval_count,
        static_count=static_eval_count,
        normal_reference_root=normal_reference_root,
    )
    optional_dagger = Path(dagger_checkpoint).resolve()
    optional_fine = Path(fine_tuned_checkpoint).resolve()
    role_trainers, role_configs, role_payloads, role_identities = _load_optional_roles(
        output_dir=data_root / "evaluation_sources",
        baseline_path=baseline_path,
        bc_path=final_checkpoint,
        dagger_path=optional_dagger if optional_dagger.is_file() else None,
        fine_tuned_path=optional_fine if optional_fine.is_file() else None,
    )
    # The migrated baseline is the exact BC/teacher observation contract.  It
    # is also evaluated as a separate role so the 39->42 migration is auditable.
    role_trainers["baseline_42_migrated"] = baseline_trainer
    role_configs["baseline_42_migrated"] = config
    role_identities["baseline_42_migrated"] = baseline_identity
    role_trainers["bc_actor"] = final_trainer
    role_configs["bc_actor"] = final_config
    oracle_trainer = baseline_trainer
    role_trainers["oracle_teacher"] = oracle_trainer
    role_configs["oracle_teacher"] = config
    evaluation = _evaluate_roles(
        roles=role_trainers,
        configs=role_configs,
        oracle_trainer=oracle_trainer,
        oracle_config=config,
        pools=eval_pools,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
        progress=progress,
        oracle_factory=oracle_factory,
    )
    source_hashes = {
        "baseline_original": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        "baseline_migrated": migration,
        "bc_final": {"path": str(final_checkpoint.resolve()), "sha256": _sha256(final_checkpoint), "policy_hash": final_identity["policy_hash"]},
        "optional_roles": role_identities,
    }
    summary = {
        "stage": STAGE_NAME,
        "status": "completed",
        "smoke": bool(smoke),
        "training_performed": False,
        "ppo_updates": 0,
        "reward_or_environment_modified": False,
        "dynamic_obstacles_used": False,
        "checkpoint_provenance": source_hashes,
        "migration": migration,
        "dataset": final_dataset_manifest,
        "initial_train_episode_summary": _aggregate(train_summaries, train_rows),
        "validation_demo_episode_summary": _aggregate(validation_summaries, []),
        "dagger_round2_episode_summary": _aggregate(dagger_summaries, dagger_rows),
        "round0_bc": round0,
        "final_bc": final_bc,
        "bc_checkpoint": str(final_checkpoint.resolve()),
        "bc_policy_identity": final_identity,
        "evaluation": evaluation,
        "contracts": {
            "baseline_source_not_modified": True,
            "observation_normalizer_frozen": True,
            "critic_frozen": True,
            "logstd_frozen": True,
            "ppo_updates": 0,
            "train_validation_pair_leakage": False,
            "teacher_truth_state_read_for_control": False,
            "validation_test_curriculum": False,
            "static_only": True,
            "stable_criterion_reused": True,
        },
        "files": {
            "dataset_manifest": str((data_root / "demonstrations" / "manifest.json").resolve()),
            "evaluation_episode_metrics": evaluation["files"]["episode_metrics"],
            "evaluation_step_metrics": evaluation["files"]["step_metrics"],
            "report": str((logs_root / "stage_report.md").resolve()),
        },
    }
    _write_json(logs_root / "summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    (logs_root / "stage_report.md").write_text(_summary_report(summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--dagger-checkpoint", type=Path, default=DEFAULT_DAGGER_CHECKPOINT)
    parser.add_argument("--fine-tuned-checkpoint", type=Path, default=DEFAULT_FINE_TUNED_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_static_obstacle_dagger_recovery_goal(
        baseline_checkpoint=args.baseline_checkpoint,
        dagger_checkpoint=args.dagger_checkpoint,
        fine_tuned_checkpoint=args.fine_tuned_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        smoke=args.smoke,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASELINE_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "STAGE_NAME",
    "_build_pool",
    "_evenly_cap",
    "_static_variant",
    "run_static_obstacle_dagger_recovery_goal",
]
