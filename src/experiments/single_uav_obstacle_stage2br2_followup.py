"""Stage 2B-R2 follow-up: sequential bypass and post-bypass recovery.

This is an isolated diagnostic/very-small Actor-only DAgger experiment.  It
does not modify the simulator, reward, observation contract, safety layer or
PPO path.  The first evaluation replays the four Stage 2B-R2 static scenes
with frozen checkpoints.  Only when those frozen rollouts fail the existing
behavioural gates is a small, pair-blocked sequential DAgger round collected.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..simulation.obstacle_model import point_obstacle_distance
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import FormalExperimentConfig, _make_environment
from .pilot import _assert_action_feasible, _assert_observation_finite, _seed_everything
from .single_uav_bc_v2_dagger import _batch_actor_physical_mean, _save_imitation_checkpoint
from .single_uav_checkpoint_evaluation import (
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_obstacle_dagger_stage2b import (
    _episode_step_row,
    _physical_obstacles,
    _static_candidate_diagnostics,
    _summary_from_rows,
    _visible_state,
    _write_rows_csv,
)
from .single_uav_obstacle_dagger_stage2br import (
    BoundaryAwareGeometricObstacleOracleController,
    _body_left_world,
    _stage2br_config_from_payload,
    _actor_decision,
    _normal_no_obstacle_scenario,
    mirror_static_scenario,
)
from .single_uav_obstacle_stage2br2_learning_bottleneck import (
    _mask_direction_gradients,
    _scope_parameter_selected,
)
from .single_uav_oracle_behavior_cloning import _canonical_hash, _copy_observation_for_json


STAGE_NAME = "single_uav_obstacle_stage2br2_followup"
DEFAULT_STAGE2A_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/"
    "checkpoints/stage2a_observation_compatible_init.pt"
)
DEFAULT_R2_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/"
    "checkpoints/ablation_39_42_direction_columns.pt"
)
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_obstacle_stage2br2_followup_seed2026")
DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_obstacle_stage2br2_followup_seed2026")
DEFAULT_PLOT_ROOT = Path("plots/preliminary/single_uav_obstacle_stage2br2_followup_seed2026")

R2_SCOPE_CHECKPOINTS = {
    "r2_39_42": DEFAULT_R2_CHECKPOINT,
    "r2_actor_head": Path(
        "data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/"
        "checkpoints/ablation_actor_head.pt"
    ),
    "r2_full_actor": Path(
        "data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/"
        "checkpoints/ablation_full_actor.pt"
    ),
}

EXACT_R2_CATEGORIES = ("left", "right", "left", "right")
ALL_CATEGORIES = ("left", "right", "left_front", "right_front")
SEQUENTIAL_PAIR_START = 8101
SEQUENTIAL_BC_EPOCHS = 80
SEQUENTIAL_BC_PATIENCE = 12
SEQUENTIAL_BC_BATCH_SIZE = 32
SEQUENTIAL_BC_LEARNING_RATE = 3.0e-3
SEQUENTIAL_DAGGER_SEED = 20262702
LATERAL_NONDEGENERATE_MPS = 0.01
DECISION_LONGITUDINAL_DISTANCE_M = 8.0
BOUNDARY_WINDOW_RADIUS = 10

PHASE_DEFINITION = {
    "approach": "before the stateful decision condition is observed",
    "decision": "first step with candidate risk, longitudinal obstacle distance below 8 m, or non-degenerate lateral action",
    "bypass": "decision has been observed, obstacle pass has not yet been observed",
    "post_bypass": "obstacle has passed, but recovery condition is not yet observed",
    "recovery": "after pass, positive radial progress and lateral action has decayed below the recovery threshold",
    "boundary_onset": "first boundary projection and the following two steps; base_phase is retained separately",
}


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


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _unit(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 and np.isfinite(norm) else np.zeros(3, dtype=float)


def _prepare_roots(*roots: Path) -> None:
    for root in roots:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Follow-up output root is not empty: {root.resolve()}")
    for root in roots:
        Path(root).mkdir(parents=True, exist_ok=True)


def _load_policy(checkpoint: Path, *, output_dir: Path) -> tuple[dict[str, Any], Any, FormalExperimentConfig, dict[str, Any]]:
    import torch

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(
        payload, output_dir=output_dir, command_line=f"{STAGE_NAME} frozen evaluation"
    )
    if config.uav_count != 1 or config.node_feature_dim != 42:
        raise ValueError("Follow-up requires the one-UAV Stage 2A 42-dimensional checkpoint")
    trainer, identity, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    return dict(payload), trainer, config, {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
        "loaded_policy_hash": identity.get("loaded_policy_hash", ""),
    }


def _scenario_specs(config: FormalExperimentConfig, categories: Sequence[str], *, pair_start: int, seed_start: int) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for index, category in enumerate(categories, start=1):
        side = "left" if str(category).startswith("left") else "right"
        scenario = mirror_static_scenario(
            config,
            category=str(category),
            pair_index=int(pair_start + index),
            side=side,
            seed=int(seed_start + index),
            round_index=0,
        )
        scenarios.append(scenario)
    return scenarios


def exact_r2_smoke_scenarios(config: FormalExperimentConfig) -> list[dict[str, Any]]:
    """The four scene definitions used by the existing R2 static smoke."""

    return _scenario_specs(config, EXACT_R2_CATEGORIES, pair_start=7000, seed_start=2500000)


def four_category_scenarios(config: FormalExperimentConfig) -> list[dict[str, Any]]:
    return _scenario_specs(config, ALL_CATEGORIES, pair_start=7100, seed_start=2510000)


def sequential_pair_scenarios(config: FormalExperimentConfig) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for pair_offset in range(4):
        pair_index = SEQUENTIAL_PAIR_START + pair_offset
        base_category = "left" if pair_offset < 2 else "left_front"
        for side in ("left", "right"):
            category = base_category if side == "left" else ("right" if base_category == "left" else "right_front")
            scenario = mirror_static_scenario(
                config,
                category=category,
                pair_index=pair_index,
                side=side,
                seed=2600000 + pair_offset,
                round_index=0,
            )
            scenario["stage2br2_followup"] = {
                "pair_id": f"pair_{pair_index:04d}",
                "split": "train" if pair_offset < 3 else "validation",
                "sequential_dagger": True,
                "scenario_seed": 2600000 + pair_offset,
                "rollout_seed": 2700000 + pair_index,
            }
            scenarios.append(scenario)
    return scenarios


def _scenario_category(scenario: Mapping[str, Any]) -> str:
    return str(scenario.get("stage2br", {}).get("category", "unknown"))


def _scenario_side(scenario: Mapping[str, Any]) -> str:
    category = _scenario_category(scenario)
    return "left" if category.startswith("left") else "right" if category.startswith("right") else "none"


def _obstacle_center_radius(scenario: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    specs = scenario.get("obstacle_specs", [])
    if not specs or not isinstance(specs[0], Mapping):
        return np.zeros(3, dtype=float), float("nan")
    spec = specs[0]
    return np.asarray(spec.get("center_m", (0.0, 0.0, 0.0)), dtype=float).reshape(3), _finite(spec.get("radius_m"), float("nan"))


def _boundary_info(position: np.ndarray, environment: Any) -> dict[str, Any]:
    lower = np.asarray(environment.config.motion.bounds_min_m, dtype=float)
    upper = np.asarray(environment.config.motion.bounds_max_m, dtype=float)
    distances = {
        "x_min": float(position[0] - lower[0]),
        "x_max": float(upper[0] - position[0]),
        "y_min": float(position[1] - lower[1]),
        "y_max": float(upper[1] - position[1]),
        "z_min": float(position[2] - lower[2]),
        "z_max": float(upper[2] - position[2]),
    }
    nearest = min(distances, key=distances.get)
    return {
        "nearest_boundary": nearest,
        "distance_to_nearest_boundary_m": float(distances[nearest]),
        **{f"distance_to_{key}_boundary_m": value for key, value in distances.items()},
    }


def _action_breakdown(state_before: Any, raw_action: np.ndarray, environment: Any, execution: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return action vectors at the observable execution stages.

    The simulator exposes the final execution vector, not two intermediate
    vectors.  ``after_boundary`` is therefore reconstructed by replaying the
    same motion model without the scene safety projector.  In the current
    implementation boundary clipping occurs inside ``apply_velocity_increment``
    before the obstacle safety projector; the metadata records this exact
    order rather than pretending that the log contains a separate hidden
    after-safety vector.
    """

    raw = np.asarray(raw_action, dtype=float).reshape(3)
    braking_delta = np.asarray(execution.get("braking_safety_projection_delta_v", np.zeros(3)), dtype=float).reshape(3)
    after_braking = raw + braking_delta if bool(execution.get("braking_safety_projected", False)) else raw.copy()
    _, replay = apply_velocity_increment(
        state_before,
        after_braking,
        environment.config.motion,
        safety_projector=None,
    )
    after_boundary = np.asarray(replay.get("delta_v_applied", after_braking), dtype=float).reshape(3)
    final = np.asarray(execution.get("delta_v_applied", after_boundary), dtype=float).reshape(3)
    return {
        "raw": raw,
        "after_braking": after_braking,
        "after_boundary": after_boundary,
        "after_safety": final,
        "raw_vs_after_boundary": after_boundary - raw,
        "raw_vs_after_safety": final - raw,
        "safety_vs_boundary": final - after_boundary,
    }


def _phase_for_step(
    *,
    phase_state: dict[str, Any],
    step: int,
    position_before: np.ndarray,
    visible: Mapping[str, Any],
    pre_diag: Mapping[str, Any],
    raw_lateral: float,
    radial_progress: float,
    passed_before: bool,
    first_pass_step: int | None,
    first_boundary_step: int | None,
) -> tuple[str, str]:
    center = phase_state.get("obstacle_center")
    obstacle_dx = float(center[0] - position_before[0]) if center is not None else float("inf")
    if not passed_before and not phase_state.get("decision_seen", False):
        if (
            int(pre_diag.get("pre_candidate_static_swept_conflict", 0) or 0)
            or int(pre_diag.get("pre_candidate_static_unsafe", 0) or 0)
            or obstacle_dx < DECISION_LONGITUDINAL_DISTANCE_M
            or abs(float(raw_lateral)) >= LATERAL_NONDEGENERATE_MPS
        ):
            phase_state["decision_seen"] = True
            base = "decision"
        else:
            base = "approach"
    elif not passed_before:
        base = "bypass" if phase_state.get("decision_seen", False) else "approach"
    elif first_pass_step is not None and step - first_pass_step <= 2:
        base = "post_bypass"
    else:
        lateral_threshold = max(LATERAL_NONDEGENERATE_MPS, float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5)
        if radial_progress > 0.0 and abs(float(raw_lateral)) <= lateral_threshold:
            base = "recovery"
        else:
            base = "post_bypass"
    if abs(float(raw_lateral)) >= LATERAL_NONDEGENERATE_MPS:
        phase_state["decision_lateral_abs"] = max(float(phase_state.get("decision_lateral_abs", 0.0)), abs(float(raw_lateral)))
    if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
        return "boundary_onset", base
    return base, base


def _goal_heading_features(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray, yaw: float) -> dict[str, float]:
    goal_relative = goal - position
    bearing = float(math.atan2(float(goal_relative[1]), float(goal_relative[0])))
    error = float((bearing - float(yaw) + math.pi) % (2.0 * math.pi) - math.pi)
    forward = np.asarray([math.cos(float(yaw)), math.sin(float(yaw)), 0.0], dtype=float)
    left = _body_left_world(float(yaw))
    return {
        "goal_relative_x_m": float(goal_relative[0]),
        "goal_relative_y_m": float(goal_relative[1]),
        "goal_relative_z_m": float(goal_relative[2]),
        "goal_bearing_rad": bearing,
        "heading_to_goal_error_rad": error,
        "forward_velocity_mps": float(np.dot(velocity, forward)),
        "lateral_velocity_mps": float(np.dot(velocity, left)),
    }


def _build_followup_row(
    *,
    base_row: Mapping[str, Any],
    environment: Any,
    scenario: Mapping[str, Any],
    state_before: Any,
    visible: Mapping[str, Any],
    actions: Mapping[str, np.ndarray],
    execution: Mapping[str, Any],
    phase: str,
    base_phase: str,
    passed_before: bool,
    passed_after: bool,
    first_pass_step: int | None,
    first_boundary_step: int | None,
    start_position: np.ndarray,
) -> dict[str, Any]:
    row = dict(base_row)
    breakdown = actions
    position_before = np.asarray(state_before.position_m, dtype=float)
    position_after = np.asarray(environment.states["uav_00"].position_m, dtype=float)
    velocity_before = np.asarray(state_before.velocity_mps, dtype=float)
    goal = np.asarray(environment.goals["uav_00"], dtype=float)
    boundary = _boundary_info(position_before, environment)
    heading = _goal_heading_features(position_before, velocity_before, goal, float(visible.get("yaw_rad", 0.0)))
    goal_direction = _unit(goal - position_before)
    expected_sign = -1.0 if _scenario_side(scenario) == "left" else 1.0
    raw = breakdown["raw"]
    final = breakdown["after_safety"]
    velocity_direction = _unit(velocity_before)
    raw_brake = max(0.0, -float(np.dot(raw, velocity_direction)))
    final_brake = max(0.0, -float(np.dot(final, velocity_direction)))
    lateral_world = _body_left_world(float(visible.get("yaw_rad", 0.0)))
    raw_lateral = float(np.dot(raw, lateral_world))
    final_lateral = float(np.dot(final, lateral_world))
    forward_action = float(np.dot(raw, np.asarray([math.cos(float(visible.get("yaw_rad", 0.0))), math.sin(float(visible.get("yaw_rad", 0.0))), 0.0], dtype=float)))
    phase_state = {
        "obstacle_center": _obstacle_center_radius(scenario)[0],
    }
    row.update(
        {
            "scenario_hash": _canonical_hash(scenario),
            "category": _scenario_category(scenario),
            "side": _scenario_side(scenario),
            "obstacle_category": _scenario_category(scenario),
            "obstacle_side": _scenario_side(scenario),
            "expected_lateral_sign": expected_sign,
            "phase": phase,
            "base_phase": base_phase,
            "passed_obstacle_before": int(passed_before),
            "passed_obstacle_after": int(passed_after),
            "first_obstacle_pass_step": "" if first_pass_step is None else int(first_pass_step),
            "first_boundary_projection_step": "" if first_boundary_step is None else int(first_boundary_step),
            "yaw_rad": float(visible.get("yaw_rad", 0.0)),
            **heading,
            "obstacle_relative_x_body_m": float(np.asarray(visible.get("obstacle_direction_body_m", (np.nan, np.nan, np.nan)), dtype=float).reshape(3)[0]),
            "obstacle_relative_y_body_m": float(np.asarray(visible.get("obstacle_direction_body_m", (np.nan, np.nan, np.nan)), dtype=float).reshape(3)[1]),
            "obstacle_relative_z_body_m": float(np.asarray(visible.get("obstacle_direction_body_m", (np.nan, np.nan, np.nan)), dtype=float).reshape(3)[2]),
            "position_before_x_m": float(position_before[0]),
            "position_before_y_m": float(position_before[1]),
            "position_before_z_m": float(position_before[2]),
            "position_after_x_m": float(position_after[0]),
            "position_after_y_m": float(position_after[1]),
            "position_after_z_m": float(position_after[2]),
            "velocity_before_x_mps": float(velocity_before[0]),
            "velocity_before_y_mps": float(velocity_before[1]),
            "velocity_before_z_mps": float(velocity_before[2]),
            "goal_direction_x": float(goal_direction[0]),
            "goal_direction_y": float(goal_direction[1]),
            "goal_direction_z": float(goal_direction[2]),
            "raw_actor_delta_v_x_mps": float(breakdown["raw"][0]),
            "raw_actor_delta_v_y_mps": float(breakdown["raw"][1]),
            "raw_actor_delta_v_z_mps": float(breakdown["raw"][2]),
            "after_braking_delta_v_x_mps": float(breakdown["after_braking"][0]),
            "after_braking_delta_v_y_mps": float(breakdown["after_braking"][1]),
            "after_braking_delta_v_z_mps": float(breakdown["after_braking"][2]),
            "after_boundary_delta_v_x_mps": float(breakdown["after_boundary"][0]),
            "after_boundary_delta_v_y_mps": float(breakdown["after_boundary"][1]),
            "after_boundary_delta_v_z_mps": float(breakdown["after_boundary"][2]),
            "after_safety_delta_v_x_mps": float(breakdown["after_safety"][0]),
            "after_safety_delta_v_y_mps": float(breakdown["after_safety"][1]),
            "after_safety_delta_v_z_mps": float(breakdown["after_safety"][2]),
            "raw_vs_boundary_action_difference_norm_mps": float(np.linalg.norm(breakdown["raw_vs_after_boundary"])),
            "raw_vs_safety_action_difference_norm_mps": float(np.linalg.norm(breakdown["raw_vs_after_safety"])),
            "safety_vs_boundary_action_difference_norm_mps": float(np.linalg.norm(breakdown["safety_vs_boundary"])),
            "raw_lateral_action_mps": raw_lateral,
            "final_lateral_action_mps": final_lateral,
            "raw_forward_action_mps": forward_action,
            "raw_braking_magnitude_mps": raw_brake,
            "effective_braking_magnitude_mps": final_brake,
            "lateral_forward_action_ratio": abs(raw_lateral) / max(abs(forward_action), 1.0e-8),
            "cumulative_lateral_displacement_after_pass_m": float(position_after[1] - start_position[1]) if passed_before or passed_after else 0.0,
            "obstacle_passed_after": int(passed_after),
            **boundary,
            "execution_order": "raw_actor -> braking_supervisor -> dynamics/action_clip -> boundary_projection -> obstacle_safety_projection",
            "after_boundary_is_reconstructed": True,
            "after_safety_is_final_applied_action": True,
            "braking_safety_projected": int(bool(execution.get("braking_safety_projected", False))),
            "safety_projected": int(bool(execution.get("safety_projected", False))),
            "boundary_projected": int(bool(execution.get("boundary_projected", False))),
            "action_clipped": int(bool(execution.get("action_clipped", False))),
            "collision": int(bool(row.get("collision", 0))),
            "termination_reason": str(row.get("termination_reason", "running")),
        }
    )
    return row


def _run_policy_episode(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    scenario: Mapping[str, Any],
    role: str,
    episode_index: int,
    reset_seed: int,
    evaluation_stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = _make_environment(
        config,
        int(reset_seed),
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, info = environment.reset(seed=int(reset_seed), scenario_config=copy.deepcopy(dict(scenario)))
    center, radius = _obstacle_center_radius(scenario)
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
    rows: list[dict[str, Any]] = []
    phase_state: dict[str, Any] = {"obstacle_center": center, "decision_seen": False, "decision_lateral_abs": 0.0}
    passed_before = False
    first_pass_step: int | None = None
    first_boundary_step: int | None = None
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision, raw_action, _actor_observation = _actor_decision(trainer, observations, info)
        state_before = environment.states["uav_00"].copy()
        visible = _visible_state(observations["uav_00"])
        execution_probe = {
            "braking_safety_projected": False,
            "braking_safety_projection_delta_v": np.zeros(3, dtype=float),
        }
        # The candidate diagnostic is explicitly pre scene projection.  The
        # current R2 configuration has braking supervisor disabled, but the
        # replay remains correct if that flag changes in a future checkpoint.
        breakdown_probe = _action_breakdown(state_before, raw_action, environment, execution_probe)
        nominal_state, _ = apply_velocity_increment(
            state_before,
            breakdown_probe["after_braking"],
            environment.config.motion,
            safety_projector=None,
        )
        pre_diag = _static_candidate_diagnostics(
            environment,
            np.asarray(state_before.position_m, dtype=float),
            np.asarray(nominal_state.position_m, dtype=float),
        )
        lateral_world = _body_left_world(float(visible.get("yaw_rad", 0.0)))
        raw_lateral = float(np.dot(raw_action, lateral_world))
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
        next_observations, _rewards, terminated, truncated, next_info = environment.step(decision.actions)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        breakdown = _action_breakdown(state_before, raw_action, environment, execution)
        # The base helper uses the same state/action and the final environment
        # state, so all legacy metrics stay comparable with Stage 2B-R2.
        base_row = _episode_step_row(
            policy=role,
            scenario_name="static_mirror_smoke" if _scenario_category(scenario) != "no_obstacle" else "normal_navigation_smoke",
            safety_mode="on",
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=np.asarray(raw_action, dtype=float),
            control={"route_phase": phase, "lateral_action_projection_mps": raw_lateral},
            nominal_position=np.asarray(nominal_state.position_m, dtype=float),
            pre_diag=pre_diag,
            next_info=next_info,
        )
        position_after = np.asarray(environment.states["uav_00"].position_m, dtype=float)
        passed_after = passed_before
        if np.isfinite(radius) and _scenario_category(scenario) != "no_obstacle":
            passed_after = bool(position_after[0] > center[0] + radius + 0.5)
        if passed_after and not passed_before:
            first_pass_step = int(step)
        if bool(execution.get("boundary_projected", False)) and first_boundary_step is None:
            first_boundary_step = int(step)
        before_goal = np.asarray(environment.goals["uav_00"], dtype=float) - np.asarray(state_before.position_m, dtype=float)
        after_goal = np.asarray(environment.goals["uav_00"], dtype=float) - position_after
        radial_progress = float(np.linalg.norm(before_goal) - np.linalg.norm(after_goal)) / float(environment.config.motion.dt_s)
        # Re-evaluate the phase with the just-observed pass/boundary event.
        if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
            phase = "boundary_onset"
        elif passed_before and radial_progress > 0.0 and abs(raw_lateral) <= max(LATERAL_NONDEGENERATE_MPS, float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5):
            phase = "recovery"
            base_phase = "recovery"
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
            start_position=start,
        )
        row.update({"evaluation_stage": evaluation_stage, "reset_seed": int(reset_seed), "episode_id": f"{role}_{episode_index:03d}"})
        rows.append(row)
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
            if first_boundary_step <= int(row["step"]) <= first_boundary_step + 2:
                row["phase"] = "boundary_onset"
    summary = _summary_from_rows(
        rows,
        policy=role,
        scenario_name="static_mirror_smoke" if _scenario_category(scenario) != "no_obstacle" else "normal_navigation_smoke",
        safety_mode="on",
        episode_index=episode_index,
    )
    distances_after = [float(row.get("true_goal_distance_after_m", float("nan"))) for row in rows]
    direction_sign = -1.0 if _scenario_side(scenario) == "left" else 1.0
    lateral_displacement = float(rows[-1].get("position_after_y_m", start[1]) - start[1]) if rows else 0.0
    passed_index = next((index for index, row in enumerate(rows) if int(row.get("passed_obstacle_after", 0) or 0)), None)
    post_distances = distances_after[passed_index:] if passed_index is not None else []
    post_slope = float("nan")
    if len(post_distances) >= 2:
        post_slope = float(np.polyfit(np.arange(len(post_distances), dtype=float), np.asarray(post_distances, dtype=float), 1)[0])
    post_progress = bool(post_distances and min(post_distances) < float(rows[passed_index].get("true_goal_distance_before_m", float("inf"))) - 1.0e-6) if passed_index is not None else False
    post_rows = rows[passed_index:] if passed_index is not None else []
    same_sign_after_pass = [abs(float(row.get("raw_lateral_action_mps", 0.0))) >= LATERAL_NONDEGENERATE_MPS and float(row.get("raw_lateral_action_mps", 0.0)) * direction_sign > 0.0 for row in post_rows]
    max_same_sign = 0
    current = 0
    for value in same_sign_after_pass:
        current = current + 1 if value else 0
        max_same_sign = max(max_same_sign, current)
    summary.update(
        {
            "evaluation_stage": evaluation_stage,
            "scenario_hash": _canonical_hash(scenario),
            "category": _scenario_category(scenario),
            "obstacle_side": _scenario_side(scenario),
            "direction_adaptive": int(lateral_displacement * direction_sign > 0.5) if _scenario_category(scenario) != "no_obstacle" else 0,
            "lateral_displacement_m": lateral_displacement,
            "passed_obstacle": int(first_pass_step is not None),
            "post_bypass_goal_progress": int(post_progress),
            "post_bypass_goal_distance_slope_m_per_step": post_slope,
            "post_bypass_recovery": int(any(str(row.get("base_phase", "")) == "recovery" for row in post_rows)),
            "post_bypass_max_same_sign_lateral_steps": int(max_same_sign),
            "first_obstacle_pass_step": "" if first_pass_step is None else int(first_pass_step),
            "first_boundary_projection_step": "" if first_boundary_step is None else int(first_boundary_step),
            "policy_updates_performed": False,
            "ppo_updates": 0,
            "formal_training_run": False,
            "training_performed": False,
        }
    )
    return summary, rows


def _aggregate_episodes(rows: Sequence[Mapping[str, Any]], *, include_by_category: bool = True) -> dict[str, Any]:
    count = len(rows)

    def count_field(field: str) -> int:
        return int(sum(int(row.get(field, 0) or 0) for row in rows))

    output = {
        "episodes": count,
        "direction_adaptive": count_field("direction_adaptive"),
        "direction_adaptive_rate": count_field("direction_adaptive") / count if count else 0.0,
        "passed_obstacle": count_field("passed_obstacle"),
        "passed_obstacle_rate": count_field("passed_obstacle") / count if count else 0.0,
        "post_bypass_goal_progress": count_field("post_bypass_goal_progress"),
        "post_bypass_goal_progress_rate": count_field("post_bypass_goal_progress") / count if count else 0.0,
        "post_bypass_recovery": count_field("post_bypass_recovery"),
        "post_bypass_recovery_rate": count_field("post_bypass_recovery") / count if count else 0.0,
        "stable_goal": count_field("stable_goal"),
        "collision": count_field("collision"),
        "mean_minimum_goal_distance_m": _mean(row.get("minimum_goal_distance_m") for row in rows),
        "mean_minimum_post_clearance_m": _mean(row.get("minimum_post_clearance_m") for row in rows),
        "mean_post_bypass_goal_distance_slope_m_per_step": _mean(row.get("post_bypass_goal_distance_slope_m_per_step") for row in rows),
        "mean_maximum_consecutive_goal_hold_steps": _mean(row.get("maximum_consecutive_goal_hold_steps") for row in rows),
        "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in rows), default=0),
        "safety_projection_steps": int(sum(int(row.get("safety_projection_steps", 0) or 0) for row in rows)),
        "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in rows)),
        "action_clipped_steps": int(sum(int(row.get("raw_action_changed_steps", 0) or 0) for row in rows)),
        "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in rows),
    }
    for threshold, suffix in ((8.0, "8m"), (5.0, "5m"), (3.0, "3m"), (1.0, "1m")):
        entered = int(sum(int(row.get(f"entered_{suffix}", 0) or 0) for row in rows))
        output[f"entered_{suffix}"] = entered
        output[f"entered_{suffix}_rate"] = entered / count if count else 0.0
    output["hold_steps_ge_1"] = count_field("hold_steps_ge_1")
    output["hold_steps_ge_3"] = count_field("hold_steps_ge_3")
    output["hold_steps_ge_5"] = count_field("hold_steps_ge_5")
    output["hold_steps_ge_1_rate"] = output["hold_steps_ge_1"] / count if count else 0.0
    output["hold_steps_ge_3_rate"] = output["hold_steps_ge_3"] / count if count else 0.0
    output["hold_steps_ge_5_rate"] = output["hold_steps_ge_5"] / count if count else 0.0
    if include_by_category:
        by_category: dict[str, Any] = {}
        for category in ALL_CATEGORIES:
            selected = [row for row in rows if str(row.get("category", "")) == category]
            if selected:
                by_category[category] = _aggregate_episodes(selected, include_by_category=False)
        output["by_category"] = by_category
    return output


def _evaluate_roles(
    roles: Mapping[str, Any],
    config: FormalExperimentConfig,
    scenarios: Sequence[Mapping[str, Any]],
    *,
    evaluation_stage: str,
    evaluation_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for episode_index, scenario in enumerate(scenarios, start=1):
        # The same reset seed is used for every role.  This is stronger than
        # the historical R2 source/candidate serial offset and makes the
        # replay explicitly paired.
        reset_seed = 2500000 + episode_index if evaluation_family == "r2_exact" else 2510000 + episode_index
        for role, trainer in roles.items():
            summary, rows = _run_policy_episode(
                trainer,
                config,
                scenario=scenario,
                role=role,
                episode_index=episode_index,
                reset_seed=reset_seed,
                evaluation_stage=evaluation_stage,
            )
            summary.update({"evaluation_family": evaluation_family, "scenario_hash": _canonical_hash(scenario)})
            for row in rows:
                row.update({"evaluation_family": evaluation_family, "scenario_hash": _canonical_hash(scenario)})
            summaries.append(summary)
            step_rows.extend(rows)
    return summaries, step_rows


def _boundary_windows(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row.get("evaluation_stage", "")), str(row.get("evaluation_family", "")), str(row.get("policy", "")), int(row.get("episode_index", -1)))].append(row)
    for key, rows in grouped.items():
        first = next((int(row["first_boundary_projection_step"]) for row in rows if str(row.get("first_boundary_projection_step", "")) not in {"", "None"}), None)
        if first is None:
            continue
        for row in rows:
            step = int(row.get("step", 0))
            if first - BOUNDARY_WINDOW_RADIUS <= step <= first + BOUNDARY_WINDOW_RADIUS:
                item = dict(row)
                item["boundary_window_center_step"] = first
                item["boundary_window_offset"] = step - first
                item["obstacle_passed_after_cumulative_lateral_displacement_m"] = row.get("cumulative_lateral_displacement_after_pass_m", 0.0)
                item["post_bypass_goal_distance_slope_m_per_step"] = row.get("post_bypass_goal_distance_slope_m_per_step", "")
                output.append(item)
    return output


def _phase_action_audit(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        # Keep the externally visible phase label here.  ``base_phase`` is
        # retained in each raw row so boundary_onset can still be analysed as
        # a window without erasing that label from the phase-level audit.
        phase = str(row.get("phase", row.get("base_phase", "")))
        groups[(str(row.get("evaluation_stage", "")), str(row.get("policy", "")), str(row.get("category", "")), phase)].append(row)
    output: list[dict[str, Any]] = []
    for (stage, policy, category, phase), rows in sorted(groups.items()):
        actor_y = np.asarray([_finite(row.get("raw_actor_delta_v_y_mps"), np.nan) for row in rows], dtype=float)
        teacher_y = np.asarray([_finite(row.get("teacher_delta_v_y_mps"), np.nan) for row in rows], dtype=float)
        oracle_y = teacher_y[np.isfinite(teacher_y)]
        actor_y_valid = actor_y[np.isfinite(actor_y)]
        ratios = [abs(float(a)) / max(abs(float(t)), 1.0e-8) for a, t in zip(actor_y, teacher_y) if np.isfinite(a) and np.isfinite(t)]
        output.append(
            {
                "evaluation_stage": stage,
                "policy": policy,
                "category": category,
                "phase": phase,
                "base_phases_present": ",".join(sorted({str(row.get("base_phase", "")) for row in rows})),
                "steps": len(rows),
                "mean_actor_raw_delta_v_y_mps": float(np.mean(actor_y_valid)) if len(actor_y_valid) else float("nan"),
                "mean_oracle_delta_v_y_mps": float(np.mean(oracle_y)) if len(oracle_y) else float("nan"),
                "actor_oracle_abs_y_magnitude_ratio": _mean(ratios),
                "mean_raw_delta_v_x_mps": _mean(row.get("raw_actor_delta_v_x_mps") for row in rows),
                "mean_oracle_delta_v_x_mps": _mean(row.get("oracle_delta_v_x_mps") for row in rows),
                "mean_lateral_forward_action_ratio": _mean(row.get("lateral_forward_action_ratio") for row in rows),
                "mean_raw_action_norm_mps": _mean(row.get("raw_action_norm_mps") for row in rows),
                "mean_after_boundary_action_norm_mps": _mean(np.linalg.norm(np.asarray([row.get("after_boundary_delta_v_x_mps", 0.0), row.get("after_boundary_delta_v_y_mps", 0.0), row.get("after_boundary_delta_v_z_mps", 0.0)], dtype=float)) for row in rows),
                "mean_goal_distance_m": _mean(row.get("true_goal_distance_before_m") for row in rows),
                "mean_radial_progress_mps": _mean(row.get("true_radial_progress_mps") for row in rows),
                "mean_total_speed_mps": _mean(row.get("total_speed_mps") for row in rows),
                "mean_target_speed_mps": _mean(row.get("oracle_desired_speed_mps") for row in rows),
                "boundary_projection_steps": int(sum(int(row.get("boundary_projected", 0) or 0) for row in rows)),
                "safety_projection_steps": int(sum(int(row.get("safety_projected", 0) or 0) for row in rows)),
            }
        )
    return output


def _dataset_record(
    *,
    scenario: Mapping[str, Any],
    split: str,
    pair_id: str,
    episode_id: str,
    episode_index: int,
    step: int,
    actor_observation: Mapping[str, Any],
    teacher_action: np.ndarray,
    student_action: np.ndarray,
    row: Mapping[str, Any],
    control: Mapping[str, Any],
    scenario_seed: int,
    rollout_seed: int,
) -> dict[str, Any]:
    return {
        "dataset_id": "single_uav_obstacle_stage2br2_followup_sequential_dagger",
        "sample_id": f"{episode_id}_step_{step:03d}",
        "sample_source": "student_visited_state_oracle_relabel",
        "split": split,
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "step": int(step),
        "pair_id": pair_id,
        "category": _scenario_category(scenario),
        "side": _scenario_side(scenario),
        "scenario_seed": int(scenario_seed),
        "rollout_seed": int(rollout_seed),
        "scenario_hash": _canonical_hash(scenario),
        # ``phase`` is the actual externally visible label, including
        # boundary_onset.  Keep base_phase separately for the underlying
        # behavioral state before the diagnostic window overlay.
        "phase": str(row.get("phase", row.get("base_phase", ""))),
        "base_phase": str(row.get("base_phase", row.get("phase", ""))),
        "phase_label": str(row.get("phase", "")),
        "observation": _copy_observation_for_json(actor_observation),
        "teacher_action_delta_v_mps": np.asarray(teacher_action, dtype=float).reshape(3).tolist(),
        "student_action_delta_v_mps": np.asarray(student_action, dtype=float).reshape(3).tolist(),
        "teacher_delta_v_x_mps": float(teacher_action[0]),
        "teacher_delta_v_y_mps": float(teacher_action[1]),
        "teacher_delta_v_z_mps": float(teacher_action[2]),
        "student_delta_v_x_mps": float(student_action[0]),
        "student_delta_v_y_mps": float(student_action[1]),
        "student_delta_v_z_mps": float(student_action[2]),
        "goal_distance_m": _finite(row.get("true_goal_distance_before_m")),
        "speed_mps": _finite(row.get("total_speed_before_mps")),
        "radial_velocity_mps": _finite(row.get("true_radial_velocity_before_mps")),
        "tangential_velocity_mps": _finite(row.get("tangential_speed_before_mps")),
        "goal_direction": [float(row.get("goal_direction_x", 0.0)), float(row.get("goal_direction_y", 0.0)), float(row.get("goal_direction_z", 0.0))],
        "braking_component_mps": max(0.0, -float(np.dot(teacher_action, _unit([row.get("velocity_before_x_mps", 0.0), row.get("velocity_before_y_mps", 0.0), row.get("velocity_before_z_mps", 0.0)])))),
        "student_action_norm_mps": float(np.linalg.norm(student_action)),
        "teacher_action_norm_mps": float(np.linalg.norm(teacher_action)),
        "radial_action_projection_mps": float(np.dot(student_action, _unit([row.get("goal_direction_x", 0.0), row.get("goal_direction_y", 0.0), row.get("goal_direction_z", 0.0)]))),
        "braking_action": _finite(control.get("braking_magnitude_mps"), 0.0),
        "oracle_route_phase": str(control.get("route_phase", "")),
        "oracle_truth_state_read": bool(control.get("truth_state_read_for_control", False)),
        "target_source": "boundary_aware_estimated_state_oracle",
        "raw_actor_delta_v_y_mps": float(student_action[1]),
        "oracle_delta_v_y_mps": float(teacher_action[1]),
        "raw_action_norm_mps": float(np.linalg.norm(student_action)),
        "lateral_forward_action_ratio": _finite(row.get("lateral_forward_action_ratio"), float("nan")),
        "boundary_projected": int(row.get("boundary_projected", 0) or 0),
        "safety_projected": int(row.get("safety_projected", 0) or 0),
        "action_clipped": int(row.get("action_clipped", 0) or 0),
    }


def _run_sequential_dagger_episode(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    scenario: Mapping[str, Any],
    episode_index: int,
    split: str,
    pair_id: str,
    episode_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rollout_seed = int(2700000 + int(scenario.get("stage2br2_followup", {}).get("pair_id", "pair_0").split("_")[-1]))
    environment = _make_environment(config, rollout_seed, metric_scales=config.metric_scales, safety_projection_enabled=True)
    observations, info = environment.reset(seed=rollout_seed, scenario_config=copy.deepcopy(dict(scenario)))
    oracle = BoundaryAwareGeometricObstacleOracleController(environment)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    center, _radius = _obstacle_center_radius(scenario)
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
    phase_state: dict[str, Any] = {"obstacle_center": center, "decision_seen": False, "decision_lateral_abs": 0.0}
    passed_before = False
    first_pass_step: int | None = None
    first_boundary_step: int | None = None
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision, student_action, actor_observation = _actor_decision(trainer, observations, info)
        visible = _visible_state(observations["uav_00"])
        teacher_payload, control = oracle.action(observations)
        teacher_action = np.asarray(teacher_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
        _assert_action_feasible(environment, teacher_payload)
        state_before = environment.states["uav_00"].copy()
        nominal_state, nominal_execution = apply_velocity_increment(state_before, student_action, environment.config.motion, safety_projector=None)
        pre_diag = _static_candidate_diagnostics(environment, state_before.position_m, nominal_state.position_m)
        raw_lateral = float(np.dot(student_action, _body_left_world(float(visible.get("yaw_rad", 0.0)))))
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
        next_observations, _rewards, terminated, truncated, next_info = environment.step(decision.actions)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        breakdown = _action_breakdown(state_before, student_action, environment, execution)
        base_row = _episode_step_row(
            policy="sequential_dagger_student",
            scenario_name="sequential_static_mirror",
            safety_mode="on",
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=student_action,
            control=control,
            nominal_position=np.asarray(nominal_state.position_m, dtype=float),
            pre_diag=pre_diag,
            next_info=next_info,
        )
        position_after = np.asarray(environment.states["uav_00"].position_m, dtype=float)
        _center, radius = _obstacle_center_radius(scenario)
        passed_after = bool(position_after[0] > _center[0] + radius + 0.5)
        if passed_after and not passed_before:
            first_pass_step = step
        if bool(execution.get("boundary_projected", False)) and first_boundary_step is None:
            first_boundary_step = step
        before_goal = np.asarray(environment.goals["uav_00"], dtype=float) - np.asarray(state_before.position_m, dtype=float)
        after_goal = np.asarray(environment.goals["uav_00"], dtype=float) - position_after
        radial_progress = float(np.linalg.norm(before_goal) - np.linalg.norm(after_goal)) / float(environment.config.motion.dt_s)
        if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
            phase = "boundary_onset"
        elif passed_before and radial_progress > 0.0 and abs(raw_lateral) <= max(LATERAL_NONDEGENERATE_MPS, float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5):
            phase = "recovery"
            base_phase = "recovery"
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
            start_position=start,
        )
        row.update(
            {
                "evaluation_stage": "sequential_dagger_data_collection",
                "evaluation_family": "sequential_dagger",
                "split": split,
                "pair_id": pair_id,
                "episode_id": episode_id,
                "scenario_seed": int(scenario.get("stage2br2_followup", {}).get("scenario_seed", scenario.get("seed", -1))),
                "rollout_seed": rollout_seed,
                # The phase audit consumes step rows rather than JSONL
                # records, so copy teacher labels into the row explicitly.
                "teacher_delta_v_x_mps": float(teacher_action[0]),
                "teacher_delta_v_y_mps": float(teacher_action[1]),
                "teacher_delta_v_z_mps": float(teacher_action[2]),
                "oracle_delta_v_x_mps": float(teacher_action[0]),
                "oracle_delta_v_y_mps": float(teacher_action[1]),
                "oracle_delta_v_z_mps": float(teacher_action[2]),
                "oracle_desired_speed_mps": _finite(control.get("desired_speed_mps")),
                "oracle_route_phase": str(control.get("route_phase", "")),
                "oracle_truth_state_read": bool(control.get("truth_state_read_for_control", False)),
            }
        )
        rows.append(row)
        teacher_control = dict(control)
        teacher_control["braking_magnitude_mps"] = max(0.0, -float(np.dot(teacher_action, _unit(np.asarray(state_before.velocity_mps, dtype=float)))))
        record = _dataset_record(
            scenario=scenario,
            split=split,
            pair_id=pair_id,
            episode_id=episode_id,
            episode_index=episode_index,
            step=step,
            actor_observation=actor_observation,
            teacher_action=teacher_action,
            student_action=student_action,
            row=row,
            control=teacher_control,
            scenario_seed=int(scenario.get("stage2br2_followup", {}).get("scenario_seed", scenario.get("seed", -1))),
            rollout_seed=rollout_seed,
        )
        records.append(record)
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
    summary = _summary_from_rows(rows, policy="sequential_dagger_student", scenario_name="sequential_static_mirror", safety_mode="on", episode_index=episode_index)
    summary.update({
        "split": split,
        "pair_id": pair_id,
        "episode_id": episode_id,
        "scenario_hash": _canonical_hash(scenario),
        "category": _scenario_category(scenario),
        "obstacle_side": _scenario_side(scenario),
        "first_obstacle_pass_step": "" if first_pass_step is None else int(first_pass_step),
        "first_boundary_projection_step": "" if first_boundary_step is None else int(first_boundary_step),
        "training_performed": False,
        "policy_updates_performed": False,
        "ppo_updates": 0,
    })
    return summary, rows, records


def _hash_parameter_map(policy: Any) -> dict[str, str]:
    return {name: _state_dict_hash({name: parameter.detach().cpu()}) for name, parameter in policy.named_parameters()}


def _normalizer_hash(trainer: Any) -> str:
    return _canonical_hash(trainer.normalizer.state_dict())


def _sequential_loss(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> Any:
    import torch

    predictions = _batch_actor_physical_mean(trainer, [sample["observation"] for sample in samples])
    targets = torch.as_tensor(np.asarray([sample["teacher_action_delta_v_mps"] for sample in samples], dtype=np.float32), dtype=torch.float32, device=predictions.device)
    return torch.mean((predictions - targets) ** 2)


def _train_sequential_bc(
    initial_checkpoint: Path,
    config: FormalExperimentConfig,
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    data_root: Path,
    logs_root: Path,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if not train_samples or not validation_samples:
        raise ValueError("Sequential DAgger requires non-empty train and validation demonstrations")
    payload, trainer, _config, identity = _load_policy(initial_checkpoint, output_dir=data_root / "sequential_bc_source")
    initial_policy_hash = _state_dict_hash(trainer.policy.state_dict())
    initial_parameters = _hash_parameter_map(trainer.policy)
    critic_before = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_before = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    normalizer_before = _normalizer_hash(trainer)
    for name, parameter in trainer.policy.named_parameters():
        parameter.requires_grad_(bool(_scope_parameter_selected(name, scope)))
    trainable_names = [name for name, parameter in trainer.policy.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise RuntimeError(f"scope {scope} selected no parameters")
    optimizer = torch.optim.Adam([parameter for parameter in trainer.policy.parameters() if parameter.requires_grad], lr=SEQUENTIAL_BC_LEARNING_RATE)
    rng = np.random.default_rng(SEQUENTIAL_DAGGER_SEED)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    wait = 0
    for epoch in range(1, SEQUENTIAL_BC_EPOCHS + 1):
        trainer.policy.eval()
        order = rng.permutation(len(train_samples))
        batch_losses: list[float] = []
        for start in range(0, len(order), SEQUENTIAL_BC_BATCH_SIZE):
            batch = [train_samples[int(index)] for index in order[start : start + SEQUENTIAL_BC_BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            loss = _sequential_loss(trainer, batch)
            loss.backward()
            _mask_direction_gradients(trainer.policy, "39_42_direction_columns" if scope == "39_42_direction_columns" else scope)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            train_loss = float(_sequential_loss(trainer, train_samples).detach().cpu())
            validation_loss = float(_sequential_loss(trainer, validation_samples).detach().cpu())
        history.append({"epoch": epoch, "train_samples": len(train_samples), "validation_samples": len(validation_samples), "batch_loss_mean": _mean(batch_losses), "train_mse": train_loss, "validation_mse": validation_loss})
        current_state = {name: parameter.detach().cpu().clone() for name, parameter in trainer.policy.state_dict().items()}
        if validation_loss < best_validation - 1.0e-12:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(current_state)
            wait = 0
        else:
            wait += 1
        if wait >= SEQUENTIAL_BC_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("sequential BC produced no best policy")
    trainer.policy.load_state_dict(best_state, strict=True)
    trainer.policy.eval()
    final_policy_hash = _state_dict_hash(trainer.policy.state_dict())
    final_parameters = _hash_parameter_map(trainer.policy)
    changed_names = [name for name in initial_parameters if initial_parameters[name] != final_parameters[name]]
    checkpoint = data_root / "checkpoints" / "sequential_dagger_checkpoint.pt"
    metadata = {
        "stage": STAGE_NAME,
        "training_scope": scope,
        "source_checkpoint": {"path": str(Path(initial_checkpoint).resolve()), "sha256": _sha256(initial_checkpoint)},
        "source_policy_hash": identity["policy_hash"],
        "initial_policy_hash": initial_policy_hash,
        "final_policy_hash": final_policy_hash,
        "trainable_parameter_names": trainable_names,
        "changed_parameter_names": changed_names,
        "critic_frozen": critic_before == _state_dict_hash(trainer.policy.critic.state_dict()),
        "normalizer_frozen": normalizer_before == _normalizer_hash(trainer),
        "logstd_frozen": logstd_before == _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()}),
        "ppo_updates": 0,
        "formal_training_run": False,
        "reward_or_environment_modified": False,
        "loss_space": "post_tanh_scaled_physical_delta_v_mse",
        "learning_rate": SEQUENTIAL_BC_LEARNING_RATE,
        "batch_size": SEQUENTIAL_BC_BATCH_SIZE,
        "max_epochs": SEQUENTIAL_BC_EPOCHS,
        "actual_epochs": len(history),
        "best_validation_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "trajectory_leakage": False,
    }
    _save_imitation_checkpoint(payload, best_state, checkpoint, metadata)
    result = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint), "policy_hash_before": initial_policy_hash, "policy_hash_after": final_policy_hash, "metadata": metadata, "history": history}
    _write_json(logs_root / "sequential_dagger_training.json", _jsonable(result))
    _write_rows_csv(logs_root / "sequential_dagger_training_history.csv", history)
    return trainer, result


def _write_plots(plots_root: Path, *, step_rows: Sequence[Mapping[str, Any]], scenarios: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    # Use the post-DAgger exact replay when available; otherwise the pre-DAgger
    # rows are still a valid frozen diagnostic.
    selected = [row for row in step_rows if row.get("evaluation_family") == "r2_exact" and row.get("evaluation_stage") == "post_dagger"]
    if not selected:
        selected = [row for row in step_rows if row.get("evaluation_family") == "r2_exact"]
    policies = ["stage2a_frozen", "r2_39_42", "r2_actor_head", "r2_full_actor", "sequential_dagger"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, policy in zip(axes.flat, policies):
        rows = [row for row in selected if row.get("policy") == policy]
        for episode in sorted({int(row.get("episode_index", -1)) for row in rows}):
            episode_rows = [row for row in rows if int(row.get("episode_index", -1)) == episode]
            axis.plot([float(row.get("position_after_x_m", np.nan)) for row in episode_rows], [float(row.get("position_after_y_m", np.nan)) for row in episode_rows], linewidth=1.2, label=f"ep{episode}")
        if scenarios:
            center, radius = _obstacle_center_radius(scenarios[0])
            if np.isfinite(radius):
                axis.add_patch(Circle((float(center[0]), float(center[1])), float(radius), color="tab:red", alpha=0.2))
            start = np.asarray(scenarios[0]["initial_positions"]["uav_00"], dtype=float)
            goal = np.asarray(scenarios[0]["goals"]["uav_00"], dtype=float)
            axis.scatter([start[0]], [start[1]], color="green", s=35)
            axis.scatter([goal[0]], [goal[1]], color="red", marker="*", s=80)
        axis.set_title(policy)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.25)
        axis.set_xlim(30, 75)
        axis.set_ylim(88, 112)
    axes.flat[-1].axis("off")
    path = plots_root / "stage2br2_followup_xy_trajectories.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["xy_trajectories"] = str(path.resolve())

    selected_policies = ["r2_39_42", "r2_actor_head", "r2_full_actor", "sequential_dagger"]
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    fields = [
        ("raw_actor_delta_v_y_mps", "after_safety_delta_v_y_mps", "after_boundary_delta_v_y_mps", "delta-v y (m/s)"),
        ("raw_actor_delta_v_x_mps", "after_safety_delta_v_x_mps", "after_boundary_delta_v_x_mps", "delta-v x (m/s)"),
        ("true_goal_distance_before_m", None, None, "goal distance (m)"),
        ("pre_candidate_static_swept_minimum_clearance_m", "minimum_clearance_m", None, "clearance (m)"),
        ("distance_to_nearest_boundary_m", None, None, "nearest-boundary distance (m)"),
        ("true_radial_progress_mps", None, None, "radial progress (m/s)"),
    ]
    colors = {"r2_39_42": "#0072B2", "r2_actor_head": "#D55E00", "r2_full_actor": "#009E73", "sequential_dagger": "#CC79A7"}
    for axis, (field, field2, field3, label) in zip(axes.flat, fields):
        for policy in selected_policies:
            rows = [row for row in selected if row.get("policy") == policy and int(row.get("episode_index", -1)) == 1]
            if not rows:
                continue
            x = [int(row.get("step", 0)) for row in rows]
            axis.plot(x, [_finite(row.get(field), np.nan) for row in rows], color=colors[policy], label=policy, linewidth=1.4)
            if field2:
                axis.plot(x, [_finite(row.get(field2), np.nan) for row in rows], color=colors[policy], linestyle="--", linewidth=0.9)
            if field3:
                axis.plot(x, [_finite(row.get(field3), np.nan) for row in rows], color=colors[policy], linestyle=":", linewidth=0.9)
        axis.set_title(label)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    path = plots_root / "stage2br2_followup_action_distance_phase.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["action_distance_phase"] = str(path.resolve())
    return output


def _scope_aggregate(rows: Sequence[Mapping[str, Any]], role: str, *, stage: str, family: str = "r2_exact") -> dict[str, Any]:
    selected = [row for row in rows if row.get("policy") == role and row.get("evaluation_stage") == stage and row.get("evaluation_family") == family]
    aggregate = _aggregate_episodes(selected)
    aggregate["policy"] = role
    aggregate["evaluation_stage"] = stage
    aggregate["evaluation_family"] = family
    return aggregate


def _select_training_scope(pre: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, str]:
    base = pre.get("r2_39_42", {})
    head = pre.get("r2_actor_head", {})
    full = pre.get("r2_full_actor", {})
    baseline = pre.get("stage2a_frozen", {})
    boundary_gate_available = "stage2a_frozen" in pre
    baseline_boundary_steps = int(baseline.get("boundary_projection_steps", 0) or 0)
    base_score = (int(base.get("passed_obstacle", 0)), int(base.get("post_bypass_goal_progress", 0)), -int(base.get("boundary_projection_steps", 0)))
    head_score = (int(head.get("passed_obstacle", 0)), int(head.get("post_bypass_goal_progress", 0)), -int(head.get("boundary_projection_steps", 0)))
    base_pass = bool(base.get("direction_adaptive", 0) >= 1.0 and base.get("passed_obstacle_rate", 0.0) >= 1.0 and base.get("post_bypass_goal_progress_rate", 0.0) >= 1.0 and int(base.get("collision", 0)) == 0 and (not boundary_gate_available or int(base.get("boundary_projection_steps", 0) or 0) <= baseline_boundary_steps))
    head_pass = bool(head.get("direction_adaptive", 0) >= 1.0 and head.get("passed_obstacle_rate", 0.0) >= 1.0 and head.get("post_bypass_goal_progress_rate", 0.0) >= 1.0 and int(head.get("collision", 0)) == 0 and (not boundary_gate_available or int(head.get("boundary_projection_steps", 0) or 0) <= baseline_boundary_steps))
    if head_pass:
        return None, "behavioral_scope_requires_actor_head"
    if head_score > base_score and (int(head.get("boundary_projection_steps", 0)) <= int(base.get("boundary_projection_steps", 0))):
        return "actor_head", "actor_head_better_but_sequential_gate_not_complete"
    if base_pass:
        return None, "39_42_sequential_gate_complete"
    # Full Actor is a read-only reference by default.  It is not selected for
    # DAgger merely because it has more parameters.
    _ = full
    return "39_42", "all_tiny_scopes_fail_or_are_not_clearly_better"


def _build_report(
    summary: Mapping[str, Any],
    *,
    pre: Mapping[str, Mapping[str, Any]],
    post: Mapping[str, Mapping[str, Any]],
    category_post: Mapping[str, Mapping[str, Any]],
    dataset_manifest: Mapping[str, Any] | None,
    training: Mapping[str, Any] | None,
) -> str:
    def compact(role: str, data: Mapping[str, Any]) -> str:
        return (
            f"{role}: direction {int(data.get('direction_adaptive', 0))}/{int(data.get('episodes', 0))}, "
            f"passed {int(data.get('passed_obstacle', 0))}/{int(data.get('episodes', 0))}, "
            f"post-progress {int(data.get('post_bypass_goal_progress', 0))}/{int(data.get('episodes', 0))}, "
            f"boundary steps={int(data.get('boundary_projection_steps', 0))}, "
            f"safety steps={int(data.get('safety_projection_steps', 0))}, "
            f"collision={int(data.get('collision', 0))}"
        )

    lines = [
        f"# {STAGE_NAME}",
        "",
        "本报告是小规模 sequential rollout / Actor-only DAgger 诊断，不是正式 100/20/20，也没有运行 PPO。",
        "",
        "## 实验约束与动作执行语义",
        "",
        "- reward、observation、动力学、动作空间、safety/boundary 规则和既有 gate 未修改。",
        "- execution order: raw Actor → braking supervisor → dynamics/action clip → boundary projection → obstacle safety projection。",
        "- `after_boundary` 是无场景安全投影的同动力学重建；`after_safety` 是环境最终 applied action。",
        "- 每个 episode/scenario 是重复单位；没有把 step 当作独立重复样本。",
        "",
        "## Step 1 同场景 frozen scope comparison",
        "",
    ]
    for role, data in pre.items():
        lines.append(f"- {compact(role, data)}")
    lines.extend(["", "## Step 2 sequential DAgger", ""])
    if dataset_manifest is None:
        lines.append("- 未执行：39:42 在 Step 1 已满足现有 sequential gate。")
    else:
        lines.extend([
            f"- 数据：{dataset_manifest.get('episodes', 0)} episodes，{dataset_manifest.get('samples', 0)} samples，{dataset_manifest.get('mirror_pairs', 0)} mirror pairs。",
            f"- phase counts：{json.dumps(dataset_manifest.get('phase_counts', {}), ensure_ascii=False)}。",
            f"- trajectory leakage：{dataset_manifest.get('trajectory_leakage')}；mirror-pair split leakage：{dataset_manifest.get('mirror_pair_train_validation_leakage')}。",
        ])
        if training:
            lines.append(f"- BC scope={training.get('metadata', {}).get('training_scope')}，epochs={training.get('metadata', {}).get('actual_epochs')}，best validation MSE={training.get('metadata', {}).get('best_validation_mse'):.6g}。")
    lines.extend(["", "## Step 3 post-DAgger exact replay", ""])
    for role, data in post.items():
        lines.append(f"- {compact(role, data)}")
    lines.extend(["", "## 四类别补充结果", ""])
    for category, data in category_post.items():
        lines.append(f"- {category}: direction={int(data.get('direction_adaptive', 0))}/{int(data.get('episodes', 0))}, passed={int(data.get('passed_obstacle', 0))}/{int(data.get('episodes', 0))}, post-progress={int(data.get('post_bypass_goal_progress', 0))}/{int(data.get('episodes', 0))}, boundary={int(data.get('boundary_projection_steps', 0))}, recovery={int(data.get('post_bypass_recovery', 0))}/{int(data.get('episodes', 0))}.")
    lines.extend([
        "",
        "## 必须回答的十个问题",
        "",
        f"1. 39:42：{'不仅是 direction sign，已完成 sequential bypass' if pre.get('r2_39_42', {}).get('passed_obstacle_rate', 0.0) >= 1.0 and pre.get('r2_39_42', {}).get('post_bypass_goal_progress_rate', 0.0) >= 1.0 else '目前仍主要证明 direction sign，尚不足以完成 sequential bypass'}。",
        f"2. actor-head：{'在相同 rollout 中明显优于 39:42' if int(pre.get('r2_actor_head', {}).get('passed_obstacle', 0)) > int(pre.get('r2_39_42', {}).get('passed_obstacle', 0)) else '没有证据显示其明显优于 39:42'}。",
        f"3. blocker 排序：{summary.get('identified_primary_blocker')}；次要问题={summary.get('identified_secondary_blockers')}。",
        f"4. boundary projection 的时序原因：{summary.get('boundary_projection_direct_cause')}。",
        f"5. safety 174→0 的解释：{summary.get('safety_to_boundary_interpretation')}。",
        f"6. sequential DAgger 是否改善 passed/post/recovery：{summary.get('sequential_policy_improvement')}。",
        f"7. 当前最小可接受 scope：{summary.get('minimum_behavioral_scope')}。",
        f"8. 是否仍禁止正式 100/20/20：{'是' if not summary.get('stage2br2_followup_passed', False) else '否'}。",
        f"9. eligible_for_next_stage：{summary.get('eligible_for_next_stage')}。",
        f"10. 唯一下一步：{summary.get('recommended_next_step')}。",
        "",
        "## 证据与限制",
        "",
        "collision=0 只表示本次实际执行未碰撞；是否主动避障必须同时由 pre-candidate 风险、raw action、projection 分层和 post-bypass 恢复证据支持。boundary gate 没有在本阶段自行发明新阈值；相对 110 steps 的下降与既有 Stage2A/R2 结果并列报告。",
    ])
    return "\n".join(lines) + "\n"


def run_stage2br2_followup(
    *,
    stage2a_checkpoint: str | Path = DEFAULT_STAGE2A_CHECKPOINT,
    r2_checkpoint: str | Path = DEFAULT_R2_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    data_root, logs_root, plots_root = Path(data_root), Path(logs_root), Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    stage2a_checkpoint = Path(stage2a_checkpoint).resolve()
    r2_checkpoint = Path(r2_checkpoint).resolve()
    source_payload, stage2a_trainer, config, stage2a_identity = _load_policy(stage2a_checkpoint, output_dir=data_root / "stage2a_source")
    r2_payload, r2_trainer, _r2_config, r2_identity = _load_policy(r2_checkpoint, output_dir=data_root / "r2_source")
    del source_payload, r2_payload
    if progress:
        print("[stage2br2-followup] loaded frozen Stage 2A and R2 checkpoints", flush=True)
    roles: dict[str, Any] = {"stage2a_frozen": stage2a_trainer, "r2_39_42": r2_trainer}
    identities: dict[str, Any] = {"stage2a_frozen": stage2a_identity, "r2_39_42": r2_identity}
    checkpoint_paths = {"stage2a_frozen": stage2a_checkpoint, "r2_39_42": r2_checkpoint}
    for role, relative in (("r2_actor_head", R2_SCOPE_CHECKPOINTS["r2_actor_head"]), ("r2_full_actor", R2_SCOPE_CHECKPOINTS["r2_full_actor"])):
        checkpoint = Path(relative).resolve()
        _payload, trainer, _cfg, identity = _load_policy(checkpoint, output_dir=data_root / role)
        del _payload, _cfg
        roles[role] = trainer
        identities[role] = identity
        checkpoint_paths[role] = checkpoint
    exact_scenarios = exact_r2_smoke_scenarios(config)
    exact_hashes = [_canonical_hash(scenario) for scenario in exact_scenarios]
    pre_summaries, pre_steps = _evaluate_roles(roles, config, exact_scenarios, evaluation_stage="pre_dagger", evaluation_family="r2_exact")
    pre_aggregates = {role: _scope_aggregate(pre_summaries, role, stage="pre_dagger") for role in roles}
    selected_scope, selection_reason = _select_training_scope(pre_aggregates)
    if progress:
        print(f"[stage2br2-followup] frozen scope comparison complete; selected={selected_scope!r} reason={selection_reason}", flush=True)
    # Normal-navigation smoke is frozen and small.  It is run before/after
    # DAgger for the source and the eventual candidate only.
    normal_scenarios = [
        _normal_no_obstacle_scenario(config, seed=2300000 + index, round_index=0, episode_index=index)
        for index in range(1, 5)
    ]
    normal_roles = {"stage2a_frozen": stage2a_trainer}
    normal_pre_summaries, normal_pre_steps = _evaluate_roles(normal_roles, config, normal_scenarios, evaluation_stage="pre_dagger", evaluation_family="normal_navigation")
    dataset_manifest: dict[str, Any] | None = None
    training_result: dict[str, Any] | None = None
    sequential_trainer: Any | None = None
    sequential_rows: list[dict[str, Any]] = []
    sequential_records: list[dict[str, Any]] = []
    sequential_summaries: list[dict[str, Any]] = []
    if selected_scope is not None:
        selected_role = "r2_39_42" if selected_scope == "39:42" else "r2_39_42" if selected_scope == "39_42" else "r2_actor_head"
        selected_trainer = roles[selected_role]
        sequential_scenarios = sequential_pair_scenarios(config)
        for episode_index, scenario in enumerate(sequential_scenarios, start=1):
            pair_id = str(scenario["stage2br2_followup"]["pair_id"])
            split = str(scenario["stage2br2_followup"]["split"])
            episode_id = f"sequential_{pair_id}_{_scenario_category(scenario)}"
            ep_summary, ep_rows, records = _run_sequential_dagger_episode(
                selected_trainer,
                config,
                scenario=scenario,
                episode_index=episode_index,
                split=split,
                pair_id=pair_id,
                episode_id=episode_id,
            )
            sequential_summaries.append(ep_summary)
            sequential_rows.extend(ep_rows)
            sequential_records.extend(records)
        train_records = [record for record in sequential_records if record["split"] == "train"]
        validation_records = [record for record in sequential_records if record["split"] == "validation"]
        train_episode_ids = sorted({str(record["episode_id"]) for record in train_records})
        validation_episode_ids = sorted({str(record["episode_id"]) for record in validation_records})
        train_pair_ids = {str(record["pair_id"]) for record in train_records}
        validation_pair_ids = {str(record["pair_id"]) for record in validation_records}
        phase_counts = dict(Counter(str(record["phase"]) for record in sequential_records))
        side_counts = dict(Counter(str(record["side"]) for record in sequential_records))
        category_counts = dict(Counter(str(record["category"]) for record in sequential_records))
        dataset_manifest = {
            "dataset_id": "single_uav_obstacle_stage2br2_followup_sequential_dagger",
            "episodes": len(sequential_summaries),
            "samples": len(sequential_records),
            "mirror_pairs": len({str(record["pair_id"]) for record in sequential_records}),
            "phase_counts": phase_counts,
            "side_counts": side_counts,
            "category_counts": category_counts,
            "train_episode_ids": train_episode_ids,
            "validation_episode_ids": validation_episode_ids,
            "train_pair_ids": sorted(train_pair_ids),
            "validation_pair_ids": sorted(validation_pair_ids),
            "trajectory_leakage": bool(set(train_episode_ids) & set(validation_episode_ids)),
            "mirror_pair_train_validation_leakage": bool(train_pair_ids & validation_pair_ids),
            "source_policy": {"role": selected_role, **identities[selected_role]},
            "oracle_version": "BoundaryAwareGeometricObstacleOracleController estimated-state visible-observation only",
            "oracle_truth_state_read_for_control": False,
            "phase_definition": PHASE_DEFINITION,
            "phase_imbalance_ratio": (max(phase_counts.values()) / max(1, min(value for value in phase_counts.values() if value > 0))) if phase_counts else float("nan"),
        }
        _write_json(data_root / "sequential_dagger_dataset_manifest.json", _jsonable(dataset_manifest))
        _write_jsonl(data_root / "sequential_dagger_train.jsonl", train_records)
        _write_jsonl(data_root / "sequential_dagger_validation.jsonl", validation_records)
        # The CSVs intentionally retain scalar diagnostics and the serialized
        # observation, so they are independently inspectable without loading
        # torch or the simulator.
        _write_rows_csv(data_root / "sequential_dagger_train.csv", train_records)
        _write_rows_csv(data_root / "sequential_dagger_validation.csv", validation_records)
        _write_rows_csv(logs_root / "sequential_dagger_projection_audit.csv", sequential_rows)
        if dataset_manifest["trajectory_leakage"] or dataset_manifest["mirror_pair_train_validation_leakage"]:
            raise AssertionError("sequential DAgger split leakage detected")
        sequential_trainer, training_result = _train_sequential_bc(
            checkpoint_paths[selected_role],
            config,
            train_records,
            validation_records,
            scope="39_42_direction_columns" if selected_role == "r2_39_42" else "actor_head",
            data_root=data_root,
            logs_root=logs_root,
        )
    # Post-Dagger exact replay contains the same four scene hashes.  If DAgger
    # was not needed, the frozen roles are still re-labelled as post for a
    # uniform output contract, without any training.
    post_roles = dict(roles)
    if sequential_trainer is not None:
        post_roles["sequential_dagger"] = sequential_trainer
    post_summaries, post_steps = _evaluate_roles(post_roles, config, exact_scenarios, evaluation_stage="post_dagger", evaluation_family="r2_exact")
    post_aggregates = {role: _scope_aggregate(post_summaries, role, stage="post_dagger") for role in post_roles}
    category_summaries, category_steps = _evaluate_roles(post_roles, config, four_category_scenarios(config), evaluation_stage="post_dagger", evaluation_family="four_category_probe")
    category_aggregates = {
        category: _aggregate_episodes([row for row in category_summaries if str(row.get("category", "")) == category and row.get("policy") == ("sequential_dagger" if sequential_trainer is not None else "r2_39_42")], include_by_category=False)
        for category in ALL_CATEGORIES
    }
    candidate_role = "sequential_dagger" if sequential_trainer is not None else "r2_39_42"
    normal_roles[candidate_role] = sequential_trainer if sequential_trainer is not None else roles["r2_39_42"]
    normal_post_summaries, normal_post_steps = _evaluate_roles(normal_roles, config, normal_scenarios, evaluation_stage="post_dagger", evaluation_family="normal_navigation")
    normal_aggregates = {role: _aggregate_episodes([row for row in normal_post_summaries if row.get("policy") == role]) for role in normal_roles}
    # Output all common episode/step diagnostics in one place.
    all_summaries = pre_summaries + post_summaries + category_summaries + normal_pre_summaries + normal_post_summaries
    all_steps = pre_steps + post_steps + category_steps + normal_pre_steps + normal_post_steps + sequential_rows
    _write_rows_csv(logs_root / "scope_static_smoke_comparison.csv", all_summaries)
    scope_json = {
        "evaluation_frozen": True,
        "no_retraining_in_scope_comparison": True,
        "r2_exact_scene_hashes": exact_hashes,
        "same_scene_across_roles": len(set(exact_hashes)) == len(exact_hashes),
        "pre_dagger": pre_aggregates,
        "post_dagger": post_aggregates,
        "four_category_probe": {role: _aggregate_episodes([row for row in category_summaries if row.get("policy") == role]) for role in post_roles},
        "selection_reason": selection_reason,
        "scene_contract": {"categories": list(EXACT_R2_CATEGORIES), "same_reset_seed_per_episode": True, "max_steps": int(config.max_steps)},
    }
    _write_json(logs_root / "scope_static_smoke_comparison.json", _jsonable(scope_json))
    _write_rows_csv(logs_root / "projection_action_audit.csv", all_steps)
    windows = _boundary_windows(all_steps)
    _write_rows_csv(logs_root / "boundary_onset_windows.csv", windows or [{"status": "no_boundary_projection_events", "window_radius_steps": BOUNDARY_WINDOW_RADIUS}])
    action_audit = _phase_action_audit(all_steps)
    _write_rows_csv(logs_root / "sequential_action_phase_audit.csv", action_audit)
    _write_rows_csv(logs_root / "sequential_dagger_episode_metrics.csv", sequential_summaries)
    _write_rows_csv(logs_root / "sequential_dagger_rollout_steps.csv", sequential_rows)
    if dataset_manifest is not None:
        _write_json(logs_root / "sequential_dagger_dataset_manifest.json", _jsonable(dataset_manifest))
    # Existing R2 gate semantics, applied to the post-Dagger candidate.  No
    # new numeric boundary threshold is invented here.
    candidate = post_aggregates[candidate_role]
    stage2a = post_aggregates["stage2a_frozen"]
    current_r2 = post_aggregates["r2_39_42"]
    boundary_existing_gate = int(candidate.get("boundary_projection_steps", 0)) <= int(stage2a.get("boundary_projection_steps", 0))
    normal_preserved = bool(normal_aggregates[candidate_role].get("episodes", 0) and normal_aggregates[candidate_role].get("collision", 0) == 0 and normal_aggregates[candidate_role].get("entered_8m", 0) >= normal_aggregates["stage2a_frozen"].get("entered_8m", 0) - 1)
    static_behavior_gate = bool(candidate.get("direction_adaptive_rate", 0.0) >= 1.0 and candidate.get("passed_obstacle_rate", 0.0) >= 1.0 and candidate.get("post_bypass_goal_progress_rate", 0.0) >= 1.0 and candidate.get("collision", 0) == 0)
    passed = bool(static_behavior_gate and boundary_existing_gate and normal_preserved)
    safety_reduced = bool(candidate.get("safety_projection_steps", 0) < stage2a.get("safety_projection_steps", 0))
    boundary_reduced_vs_r2 = bool(candidate.get("boundary_projection_steps", 0) < current_r2.get("boundary_projection_steps", 0))
    post_improvement = bool(sequential_trainer is not None and (candidate.get("passed_obstacle", 0) > current_r2.get("passed_obstacle", 0) or candidate.get("post_bypass_goal_progress", 0) > current_r2.get("post_bypass_goal_progress", 0) or candidate.get("post_bypass_recovery", 0) > current_r2.get("post_bypass_recovery", 0)))
    if sequential_trainer is not None and post_improvement:
        primary_blocker = "sequential_state_distribution_and_post_bypass_recovery"
    elif sequential_trainer is not None and not boundary_reduced_vs_r2:
        primary_blocker = "post_bypass_lateral_persistence"
    elif selected_scope == "actor_head":
        primary_blocker = "behavioral_scope_requires_actor_head"
    else:
        primary_blocker = "sequential_state_distribution_insufficient"
    boundary_rows = [
        row
        for row in post_steps
        if row.get("policy") == candidate_role
        and row.get("evaluation_family") == "r2_exact"
        and int(row.get("boundary_projected", 0) or 0)
    ]
    collection_boundary_rows = [row for row in sequential_rows if int(row.get("boundary_projected", 0) or 0)]
    if boundary_rows:
        candidate_post_boundary = [row for row in boundary_rows if int(row.get("passed_obstacle_before", 0) or 0)]
        if candidate_post_boundary:
            boundary_cause = (
                "post-DAgger candidate: boundary projection includes post-bypass lateral persistence; "
                "pre-bypass drift is also present"
            )
        else:
            boundary_cause = (
                "post-DAgger candidate: boundary projection occurred before obstacle pass, "
                "so the candidate drifted laterally before completing bypass"
            )
        if collection_boundary_rows:
            collection_post_boundary = [row for row in collection_boundary_rows if int(row.get("passed_obstacle_before", 0) or 0)]
            if collection_post_boundary:
                boundary_cause += (
                    f"; sequential DAgger collection had {len(collection_post_boundary)}/{len(collection_boundary_rows)} "
                    "boundary steps post-bypass"
                )
    elif collection_boundary_rows:
        collection_post_boundary = [row for row in collection_boundary_rows if int(row.get("passed_obstacle_before", 0) or 0)]
        boundary_cause = (
            f"candidate had no boundary event; sequential DAgger collection had {len(collection_post_boundary)}/{len(collection_boundary_rows)} "
            "boundary steps post-bypass"
        )
    else:
        boundary_cause = "no boundary projection event in the evaluated candidate"
    transfer = bool(safety_reduced and int(candidate.get("boundary_projection_steps", 0)) >= int(current_r2.get("boundary_projection_steps", 0)))
    # The follow-up is intentionally diagnostic.  When the candidate fails,
    # the next action must first be a read-only audit of the missing
    # post-bypass/recovery state coverage; do not silently turn this report
    # into another BC/PPO round.
    recommended_next_step = (
        "read_only_post_bypass_recovery_and_boundary_onset_coverage_audit_before_any_new_BC_or_PPO"
        if not passed
        else "independent_multi_seed_validation_before_formal_100_20_20"
    )
    summary: dict[str, Any] = {
        "stage": STAGE_NAME,
        "status": "completed",
        "source_stage2a_checkpoint": stage2a_identity,
        "source_stage2br2_checkpoint": r2_identity,
        "checkpoint_paths": {role: str(path.resolve()) for role, path in checkpoint_paths.items()},
        "checkpoint_hashes": {role: identities[role] for role in identities},
        "ppo_updates": 0,
        "formal_training_run": False,
        "scope_comparison_run": True,
        "scope_39_42": {**pre_aggregates["r2_39_42"], "post_dagger": post_aggregates["r2_39_42"]},
        "scope_actor_head": {**pre_aggregates["r2_actor_head"], "post_dagger": post_aggregates["r2_actor_head"]},
        "scope_full_actor": {"evaluated": True, **pre_aggregates["r2_full_actor"], "post_dagger": post_aggregates["r2_full_actor"]},
        "minimum_behavioral_scope": selected_scope or ("actor_head" if selection_reason == "behavioral_scope_requires_actor_head" else "undetermined"),
        "sequential_dagger_run": sequential_trainer is not None,
        "sequential_dataset": dataset_manifest or {"episodes": 0, "samples": 0, "mirror_pairs": 0, "phase_counts": {}, "side_counts": {}, "trajectory_leakage": False},
        "sequential_policy": post_aggregates.get("sequential_dagger", {}),
        "normal_navigation_preserved": normal_preserved,
        "normal_navigation": normal_aggregates,
        "safety_intervention_reduced": safety_reduced,
        "autonomous_avoidance_improved": bool(post_improvement and not transfer),
        "boundary_interference_improved": boundary_reduced_vs_r2,
        "boundary_interference_gate": {"existing_stage2br2_gate": boundary_existing_gate, "requires_manual_gate_judgement": True, "absolute_candidate_steps": int(candidate.get("boundary_projection_steps", 0)), "r2_reference_steps": int(current_r2.get("boundary_projection_steps", 0)), "stage2a_reference_steps": int(stage2a.get("boundary_projection_steps", 0))},
        "identified_primary_blocker": primary_blocker,
        "identified_secondary_blockers": ["post_bypass_lateral_persistence" if any(int(row.get("post_bypass_max_same_sign_lateral_steps", 0) or 0) >= 5 for row in post_summaries) else "not_established", "projection_interaction_transfer" if transfer else "not_established"],
        "action_magnitude_issue": False,
        "lateral_persistence_issue": bool(any(int(row.get("post_bypass_max_same_sign_lateral_steps", 0) or 0) >= 5 for row in post_summaries)),
        "forward_progress_issue": bool(candidate.get("post_bypass_goal_progress", 0) < candidate.get("passed_obstacle", 0)),
        "recovery_issue": bool(candidate.get("passed_obstacle", 0) > candidate.get("post_bypass_recovery", 0)),
        "sequential_distribution_issue": bool(sequential_trainer is not None),
        "boundary_projection_direct_cause": boundary_cause,
        "boundary_projection_timing": {
            "candidate_post_dagger_exact_boundary_steps": len(boundary_rows),
            "candidate_post_dagger_exact_post_bypass_boundary_steps": int(sum(int(row.get("passed_obstacle_before", 0) or 0) for row in boundary_rows)),
            "candidate_post_dagger_exact_pre_or_bypass_boundary_steps": int(sum(not int(row.get("passed_obstacle_before", 0) or 0) for row in boundary_rows)),
            "sequential_dagger_collection_boundary_steps": len(collection_boundary_rows),
            "sequential_dagger_collection_post_bypass_boundary_steps": int(sum(int(row.get("passed_obstacle_before", 0) or 0) for row in collection_boundary_rows)),
        },
        "safety_to_boundary_interpretation": "safety intervention reduced with no matching boundary increase" if safety_reduced and not transfer else "safety intervention reduction is compatible with transfer to boundary projection; it is not autonomous-improvement evidence" if transfer else "not enough reduction evidence to claim autonomous improvement",
        "sequential_policy_improvement": {"passed_obstacle": post_aggregates.get(candidate_role, {}).get("passed_obstacle", 0) > current_r2.get("passed_obstacle", 0), "post_bypass_goal_progress": post_aggregates.get(candidate_role, {}).get("post_bypass_goal_progress", 0) > current_r2.get("post_bypass_goal_progress", 0), "boundary_projection": boundary_reduced_vs_r2, "recovery": post_aggregates.get(candidate_role, {}).get("post_bypass_recovery", 0) > current_r2.get("post_bypass_recovery", 0)},
        "stage2br2_followup_passed": passed,
        "eligible_for_next_stage": passed,
        "recommended_next_step": recommended_next_step,
        "selection_reason": selection_reason,
        "phase_definition": PHASE_DEFINITION,
        "evaluation_contract": {"same_r2_scenes": True, "same_seed_per_role": True, "old_outputs_overwritten": False, "training_in_scope_comparison": False, "formal_100_20_20": False},
        "required_test_contracts": {"critic_frozen": bool(training_result is None or training_result["metadata"]["critic_frozen"]), "normalizer_frozen": bool(training_result is None or training_result["metadata"]["normalizer_frozen"]), "logstd_frozen": bool(training_result is None or training_result["metadata"]["logstd_frozen"]), "observation_dim_42": bool(config.node_feature_dim == 42), "ppo_updates_0": True, "trajectory_split_no_leakage": bool(dataset_manifest is None or not dataset_manifest["trajectory_leakage"]), "mirror_pair_split_no_leakage": bool(dataset_manifest is None or not dataset_manifest["mirror_pair_train_validation_leakage"])},
        "plots": _write_plots(plots_root, step_rows=all_steps, scenarios=exact_scenarios),
    }
    # A plain summary copy in data/ makes the new output self-contained.
    _write_json(logs_root / "stage2br2_followup_summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    if dataset_manifest is not None:
        _write_json(data_root / "sequential_dagger_dataset_manifest.json", _jsonable(dataset_manifest))
    report = _build_report(summary, pre=pre_aggregates, post=post_aggregates, category_post=category_aggregates, dataset_manifest=dataset_manifest, training=training_result)
    (logs_root / "stage2br2_followup_report.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--stage2a-checkpoint", type=Path, default=DEFAULT_STAGE2A_CHECKPOINT)
    parser.add_argument("--r2-checkpoint", type=Path, default=DEFAULT_R2_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage2br2_followup(
        stage2a_checkpoint=args.stage2a_checkpoint,
        r2_checkpoint=args.r2_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "DEFAULT_R2_CHECKPOINT",
    "DEFAULT_STAGE2A_CHECKPOINT",
    "PHASE_DEFINITION",
    "exact_r2_smoke_scenarios",
    "run_stage2br2_followup",
    "sequential_pair_scenarios",
]
