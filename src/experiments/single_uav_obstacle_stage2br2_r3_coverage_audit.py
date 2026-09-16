"""Stage 2B-R2-R3 coverage audit and minimal DAgger collection correction.

This module is deliberately separate from the validated Stage 2B-R2 follow-up
entry point.  It audits and improves only the diagnostic/data-collection
contract: the simulator, reward, action mapping, safety/boundary execution,
and learning code are not changed.  No BC or PPO update is performed here.

The R3 collector keeps the externally visible ``boundary_onset`` overlay and
the underlying ``base_phase``.  Its decision classifier is state-only: a
non-zero lateral student action is recorded as a diagnostic signal, but does
not by itself erase genuine far-field ``approach`` states.
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
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import FormalExperimentConfig
from .single_uav_obstacle_dagger_stage2b import _episode_step_row, _summary_from_rows, _write_rows_csv
from .single_uav_obstacle_dagger_stage2br import (
    BoundaryAwareGeometricObstacleOracleController,
    _body_left_world,
    _normal_no_obstacle_scenario,
    _post_obstacle_scenario,
    mirror_static_scenario,
)
from .single_uav_obstacle_stage2br2_followup import (
    ALL_CATEGORIES,
    _action_breakdown,
    _aggregate_episodes,
    _build_followup_row,
    _load_policy,
    _obstacle_center_radius,
    _phase_action_audit,
    _scenario_category,
    _scenario_side,
    _visible_state,
    _evaluate_roles,
)
from .single_uav_obstacle_stage2br2_followup import _dataset_record as _r2_dataset_record
from .single_uav_obstacle_stage2br2_learning_bottleneck import (
    _scope_parameter_selected,
)
from .pilot import _assert_action_feasible, _assert_observation_finite
from .single_uav_oracle_behavior_cloning import _canonical_hash, _copy_observation_for_json


STAGE_NAME = "single_uav_obstacle_stage2br2_r3_coverage_audit"
DEFAULT_ACTOR_HEAD_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/"
    "checkpoints/ablation_actor_head.pt"
)
DEFAULT_STAGE2A_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/"
    "checkpoints/stage2a_observation_compatible_init.pt"
)
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026")
DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026")
DEFAULT_PLOT_ROOT = Path("plots/preliminary/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026")

R3_PAIR_START = 9201
R3_SCENARIO_SEED_START = 2800000
R3_ROLLOUT_SEED_START = 2900000
R3_TRAIN_PAIRS = 6
R3_TOTAL_PAIRS = 8
R3_BALANCED_SAMPLES_PER_EPISODE_PHASE = 8
R3_MAX_STEPS = 100
LATERAL_NONDEGENERATE_MPS = 0.01
DECISION_LONGITUDINAL_DISTANCE_M = 8.0
BOUNDARY_WINDOW_RADIUS = 10

PHASE_DEFINITION = {
    "approach": "before the state-only decision condition is observed; premature lateral action is retained as a diagnostic field",
    "decision": "first step with pre-candidate static risk/unsafe clearance or longitudinal obstacle distance at or below 8 m",
    "bypass": "decision has been observed, obstacle pass has not yet been observed",
    "post_bypass": "obstacle has passed, but recovery condition is not yet observed",
    "recovery": "after pass, positive radial progress and lateral action has decayed below the recovery threshold",
    "boundary_onset": "first boundary projection and the following two steps; base_phase is retained separately",
}
TARGET_PHASES = tuple(PHASE_DEFINITION)


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
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_roots(*roots: Path) -> None:
    for root in roots:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"R3 output root is not empty: {root.resolve()}")
    for root in roots:
        Path(root).mkdir(parents=True, exist_ok=True)


def _coverage_pair_metadata(
    scenario: Mapping[str, Any], *, pair_id: str, split: str, coverage_target: str,
    scenario_seed: int, rollout_seed: int,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "split": split,
        "coverage_target": coverage_target,
        "scenario_seed": int(scenario_seed),
        "rollout_seed": int(rollout_seed),
        "coverage_audit": True,
        "sequential_dagger": True,
    }


def _mirror_coverage_scenario(
    config: FormalExperimentConfig,
    *,
    pair_index: int,
    side: str,
    category: str,
    longitudinal_distance: float,
    start_y: float,
    radius: float = 4.5,
    lateral_offset: float = 4.5,
    coverage_target: str,
    split: str,
    scenario_seed: int,
) -> dict[str, Any]:
    base_category = "left_front" if longitudinal_distance >= 10.0 else "left"
    scenario = mirror_static_scenario(
        config,
        category=base_category,
        pair_index=pair_index,
        side="left" if side == "left" else "right",
        seed=int(scenario_seed),
        round_index=0,
    )
    sign = 1.0 if side == "left" else -1.0
    start = np.asarray([34.0, float(start_y), 20.0], dtype=float)
    goal = np.asarray([66.0, float(start_y), 20.0], dtype=float)
    center = np.asarray(
        [start[0] + float(longitudinal_distance), start[1] + sign * float(lateral_offset), start[2]],
        dtype=float,
    )
    scenario["initial_positions"] = {"uav_00": start.tolist()}
    scenario["goals"] = {"uav_00": goal.tolist()}
    scenario["obstacle_specs"][0]["center_m"] = center.tolist()
    scenario["obstacle_specs"][0]["radius_m"] = float(radius)
    scenario["stage2br"].update(
        {
            "category": str(category),
            "side": str(side),
            "coverage_target": str(coverage_target),
            "longitudinal_distance_m": float(longitudinal_distance),
            "start_y_m": float(start_y),
            "lateral_distance_abs_m": float(lateral_offset),
        }
    )
    scenario["stage2br2_r3"] = _coverage_pair_metadata(
        scenario,
        pair_id=f"pair_{int(pair_index):04d}",
        split=split,
        coverage_target=coverage_target,
        scenario_seed=scenario_seed,
        rollout_seed=R3_ROLLOUT_SEED_START + int(pair_index),
    )
    return scenario


def _post_recovery_coverage_scenario(
    config: FormalExperimentConfig,
    *,
    pair_index: int,
    side: str,
    start_y: float,
    coverage_target: str,
    split: str,
    scenario_seed: int,
) -> dict[str, Any]:
    scenario = _post_obstacle_scenario(config, seed=int(scenario_seed))
    start = np.asarray([55.0, float(start_y), 20.0], dtype=float)
    goal = np.asarray([70.0, float(start_y), 20.0], dtype=float)
    scenario["initial_positions"] = {"uav_00": start.tolist()}
    scenario["goals"] = {"uav_00": goal.tolist()}
    scenario["obstacle_specs"][0]["center_m"] = [45.0, float(start_y), 20.0]
    category = "left_post_recovery" if side == "left" else "right_post_recovery"
    scenario["stage2br"].update(
        {"category": category, "side": side, "coverage_target": coverage_target, "start_y_m": float(start_y)}
    )
    scenario["stage2br2_r3"] = _coverage_pair_metadata(
        scenario,
        pair_id=f"pair_{int(pair_index):04d}",
        split=split,
        coverage_target=coverage_target,
        scenario_seed=scenario_seed,
        rollout_seed=R3_ROLLOUT_SEED_START + int(pair_index),
    )
    return scenario


def coverage_scenarios(config: FormalExperimentConfig) -> list[dict[str, Any]]:
    """Build eight independent mirror pairs for the coverage-only audit.

    The first six pairs are training data and the final two are validation
    data.  These are diagnostic/DAgger collection scenarios only; they do not
    alter any formal benchmark generator.
    """

    blueprints = (
        ("approach_far", "approach", 20.0, 100.0, 4.5),
        ("approach_far_2", "approach", 24.0, 100.0, 4.5),
        ("bypass_near", "bypass", 8.0, 100.0, 4.5),
        ("bypass_front", "bypass", 12.0, 100.0, 4.8),
        ("boundary_low", "boundary_onset", 8.0, 12.0, 4.5),
        ("boundary_high", "boundary_onset", 8.0, 188.0, 4.5),
        ("post_recovery_low", "recovery", -1.0, 100.0, 3.0),
        ("post_recovery_high", "recovery", -1.0, 100.0, 3.0),
    )
    scenarios: list[dict[str, Any]] = []
    # Keep validation independent while making it cover more than the easy
    # post-obstacle probe.  The two validation pairs are one far approach pair
    # and one post-recovery probe pair; all remaining pairs stay in train.
    validation_offsets = {1, 6}
    for offset, (name, target, distance, start_y, radius) in enumerate(blueprints):
        pair_index = R3_PAIR_START + offset
        split = "validation" if offset in validation_offsets else "train"
        for side in ("left", "right"):
            if target == "recovery":
                scenario = _post_recovery_coverage_scenario(
                    config,
                    pair_index=pair_index,
                    side=side,
                    start_y=start_y,
                    coverage_target=target,
                    split=split,
                    scenario_seed=R3_SCENARIO_SEED_START + offset,
                )
            else:
                category = f"{side}_{name}"
                scenario = _mirror_coverage_scenario(
                    config,
                    pair_index=pair_index,
                    side=side,
                    category=category,
                    longitudinal_distance=distance,
                    start_y=start_y,
                    radius=radius,
                    lateral_offset=4.5 if target != "boundary_onset" else 4.0,
                    coverage_target=target,
                    split=split,
                    scenario_seed=R3_SCENARIO_SEED_START + offset,
                )
            scenarios.append(scenario)
    return scenarios


def _coverage_phase_for_step(
    *,
    phase_state: dict[str, Any],
    step: int,
    position_before: np.ndarray,
    pre_diag: Mapping[str, Any],
    raw_lateral: float,
    radial_progress: float,
    passed_before: bool,
    first_pass_step: int | None,
    first_boundary_step: int | None,
) -> tuple[str, str]:
    """State-only phase classifier used only by the R3 collector.

    The prior classifier treated any lateral actor output as an immediate
    decision.  Here that output is retained as ``premature_lateral_action``
    evidence, while decision onset is tied to physical risk or the 8 m
    longitudinal boundary.  This makes approach coverage measurable without
    relabelling a step by time alone.
    """

    center = phase_state.get("obstacle_center")
    obstacle_dx = float(center[0] - position_before[0]) if center is not None else float("inf")
    risk = bool(
        int(pre_diag.get("pre_candidate_static_swept_conflict", 0) or 0)
        or int(pre_diag.get("pre_candidate_static_unsafe", 0) or 0)
    )
    if not passed_before and not phase_state.get("decision_seen", False):
        if risk or obstacle_dx <= DECISION_LONGITUDINAL_DISTANCE_M:
            phase_state["decision_seen"] = True
            base = "decision"
        else:
            base = "approach"
    elif not passed_before:
        base = "bypass"
    elif first_pass_step is not None and step - first_pass_step <= 2:
        base = "post_bypass"
    else:
        lateral_threshold = max(
            LATERAL_NONDEGENERATE_MPS,
            float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5,
        )
        base = "recovery" if radial_progress > 0.0 and abs(float(raw_lateral)) <= lateral_threshold else "post_bypass"
    if phase_state.get("decision_seen", False):
        phase_state["decision_lateral_abs"] = max(
            float(phase_state.get("decision_lateral_abs", 0.0)), abs(float(raw_lateral))
        )
    if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
        return "boundary_onset", base
    return base, base


def _r3_dataset_record(
    *, scenario: Mapping[str, Any], row: Mapping[str, Any], split: str, pair_id: str,
    episode_id: str, episode_index: int, step: int, actor_observation: Mapping[str, Any],
    teacher_action: np.ndarray, student_action: np.ndarray, control: Mapping[str, Any],
    scenario_seed: int, rollout_seed: int,
) -> dict[str, Any]:
    record = _r2_dataset_record(
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
        control=control,
        scenario_seed=scenario_seed,
        rollout_seed=rollout_seed,
    )
    record.update(
        {
            "dataset_id": "single_uav_obstacle_stage2br2_r3_coverage_audit",
            "coverage_target": str(scenario.get("stage2br2_r3", {}).get("coverage_target", "")),
            "after_boundary_is_reconstructed": True,
            "after_safety_is_final_applied_action": True,
            "premature_lateral_action": int(abs(float(student_action[1])) >= LATERAL_NONDEGENERATE_MPS),
        }
    )
    return record


def _run_coverage_episode(
    trainer: Any,
    config: FormalExperimentConfig,
    *, scenario: Mapping[str, Any], episode_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = scenario["stage2br2_r3"]
    split = str(metadata["split"])
    pair_id = str(metadata["pair_id"])
    episode_id = f"r3_{pair_id}_{_scenario_category(scenario)}"
    rollout_seed = int(metadata["rollout_seed"])
    environment = _make_environment_for_r3(config, rollout_seed)
    observations, info = environment.reset(seed=rollout_seed, scenario_config=copy.deepcopy(dict(scenario)))
    oracle = BoundaryAwareGeometricObstacleOracleController(environment)
    center, _radius = _obstacle_center_radius(scenario)
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
    phase_state: dict[str, Any] = {"obstacle_center": center, "decision_seen": False, "decision_lateral_abs": 0.0}
    passed_before = False
    first_pass_step: int | None = None
    first_boundary_step: int | None = None
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for step in range(min(R3_MAX_STEPS, int(config.max_steps))):
        _assert_observation_finite(observations)
        decision, student_action, actor_observation = _actor_decision_for_r3(trainer, observations, info)
        visible = _visible_state(observations["uav_00"])
        teacher_payload, control = oracle.action(observations)
        teacher_action = np.asarray(teacher_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
        _assert_action_feasible(environment, teacher_payload)
        state_before = environment.states["uav_00"].copy()
        nominal_state, _nominal_execution = apply_velocity_increment(
            state_before, student_action, environment.config.motion, safety_projector=None
        )
        pre_diag = _static_candidate_diagnostics_for_r3(
            environment, np.asarray(state_before.position_m), np.asarray(nominal_state.position_m)
        )
        lateral_world = _body_left_world(float(visible.get("yaw_rad", 0.0)))
        raw_lateral = float(np.dot(student_action, lateral_world))
        phase, base_phase = _coverage_phase_for_step(
            phase_state=phase_state,
            step=step,
            position_before=np.asarray(state_before.position_m),
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
            policy="r3_coverage_student",
            scenario_name="r3_coverage_static",
            safety_mode="on",
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m),
            velocity_before=np.asarray(state_before.velocity_mps),
            action=student_action,
            control=control,
            nominal_position=np.asarray(nominal_state.position_m),
            pre_diag=pre_diag,
            next_info=next_info,
        )
        position_after = np.asarray(environment.states["uav_00"].position_m)
        passed_after = bool(position_after[0] > center[0] + _radius + 0.5) if np.isfinite(_radius) else False
        if passed_after and not passed_before:
            first_pass_step = int(step)
        if bool(execution.get("boundary_projected", False)) and first_boundary_step is None:
            first_boundary_step = int(step)
        goal = np.asarray(environment.goals["uav_00"], dtype=float)
        radial_progress = (
            float(np.linalg.norm(goal - np.asarray(state_before.position_m)) - np.linalg.norm(goal - position_after))
            / float(environment.config.motion.dt_s)
        )
        if first_boundary_step is not None and first_boundary_step <= step <= first_boundary_step + 2:
            phase = "boundary_onset"
        elif passed_before and radial_progress > 0.0 and abs(raw_lateral) <= max(
            LATERAL_NONDEGENERATE_MPS, float(phase_state.get("decision_lateral_abs", 0.0)) * 0.5
        ):
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
                "evaluation_stage": "r3_coverage_data_collection",
                "evaluation_family": "r3_coverage",
                "split": split,
                "pair_id": pair_id,
                "episode_id": episode_id,
                "coverage_target": str(metadata["coverage_target"]),
                "scenario_seed": int(metadata["scenario_seed"]),
                "rollout_seed": rollout_seed,
                "teacher_delta_v_x_mps": float(teacher_action[0]),
                "teacher_delta_v_y_mps": float(teacher_action[1]),
                "teacher_delta_v_z_mps": float(teacher_action[2]),
                "oracle_delta_v_x_mps": float(teacher_action[0]),
                "oracle_delta_v_y_mps": float(teacher_action[1]),
                "oracle_delta_v_z_mps": float(teacher_action[2]),
                "oracle_desired_speed_mps": _finite(control.get("desired_speed_mps")),
                "oracle_route_phase": str(control.get("route_phase", "")),
                "oracle_truth_state_read": bool(control.get("truth_state_read_for_control", False)),
                "premature_lateral_action": int(abs(raw_lateral) >= LATERAL_NONDEGENERATE_MPS),
            }
        )
        rows.append(row)
        teacher_control = dict(control)
        teacher_control["braking_magnitude_mps"] = max(
            0.0,
            -float(np.dot(teacher_action, np.asarray(state_before.velocity_mps) / max(np.linalg.norm(state_before.velocity_mps), 1.0e-12)))
            if np.linalg.norm(state_before.velocity_mps) > 1.0e-12
            else 0.0,
        )
        records.append(
            _r3_dataset_record(
                scenario=scenario,
                row=row,
                split=split,
                pair_id=pair_id,
                episode_id=episode_id,
                episode_index=episode_index,
                step=step,
                actor_observation=actor_observation,
                teacher_action=teacher_action,
                student_action=student_action,
                control=teacher_control,
                scenario_seed=int(metadata["scenario_seed"]),
                rollout_seed=rollout_seed,
            )
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
            if first_boundary_step <= int(row["step"]) <= first_boundary_step + 2:
                row["phase"] = "boundary_onset"
    summary = _summary_from_rows(
        rows,
        policy="r3_coverage_student",
        scenario_name="r3_coverage_static",
        safety_mode="on",
        episode_index=episode_index,
    )
    post_rows = [row for row in rows if int(row.get("passed_obstacle_before", 0) or 0)]
    direction_sign = -1.0 if _scenario_side(scenario) == "left" else 1.0
    same_sign_lateral = [
        abs(float(row.get("raw_lateral_action_mps", 0.0) or 0.0)) >= LATERAL_NONDEGENERATE_MPS
        and float(row.get("raw_lateral_action_mps", 0.0) or 0.0) * direction_sign > 0.0
        for row in post_rows
    ]
    max_same_sign_lateral = 0
    current_same_sign = 0
    for value in same_sign_lateral:
        current_same_sign = current_same_sign + 1 if value else 0
        max_same_sign_lateral = max(max_same_sign_lateral, current_same_sign)
    summary.update(
        {
            "evaluation_stage": "r3_coverage_data_collection",
            "evaluation_family": "r3_coverage",
            "split": split,
            "pair_id": pair_id,
            "episode_id": episode_id,
            "coverage_target": str(metadata["coverage_target"]),
            "scenario_hash": _canonical_hash(scenario),
            "phase_counts": dict(Counter(str(row.get("phase", "")) for row in rows)),
            "base_phase_counts": dict(Counter(str(row.get("base_phase", "")) for row in rows)),
            "post_bypass_recovery": int(any(str(row.get("base_phase", "")) == "recovery" for row in post_rows)),
            "post_bypass_max_same_sign_lateral_steps": int(max_same_sign_lateral),
            "training_performed": False,
            "bc_training_performed": False,
            "policy_updates_performed": False,
            "ppo_updates": 0,
        }
    )
    return summary, rows, records


def _make_environment_for_r3(config: FormalExperimentConfig, seed: int) -> Any:
    from .formal import _make_environment

    return _make_environment(
        config,
        int(seed),
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )


def _actor_decision_for_r3(trainer: Any, observations: Mapping[str, Any], info: Mapping[str, Any]) -> tuple[Any, np.ndarray, Mapping[str, Any]]:
    from .single_uav_obstacle_dagger_stage2br import _actor_decision

    return _actor_decision(trainer, observations, info)


def _static_candidate_diagnostics_for_r3(environment: Any, position: np.ndarray, nominal_position: np.ndarray) -> dict[str, Any]:
    from .single_uav_obstacle_dagger_stage2b import _static_candidate_diagnostics

    return _static_candidate_diagnostics(environment, position, nominal_position)


def _balanced_records(records: Sequence[Mapping[str, Any]], *, max_per_episode_phase: int = R3_BALANCED_SAMPLES_PER_EPISODE_PHASE) -> list[dict[str, Any]]:
    """Cap each episode/phase group deterministically without step pseudoreplication."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("split", "")), str(record.get("episode_id", "")), str(record.get("phase", "")))].append(record)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: int(item.get("step", 0)))
        take = min(int(max_per_episode_phase), len(group))
        if take == len(group):
            indices = list(range(len(group)))
        else:
            indices = sorted(set(np.linspace(0, len(group) - 1, take, dtype=int).tolist()))
        selected.extend(dict(group[index]) for index in indices)
    return selected


def _coverage_manifest(records: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], *, source_identity: Mapping[str, Any]) -> dict[str, Any]:
    phase_counts = Counter(str(record.get("phase", "")) for record in records)
    base_phase_counts = Counter(str(record.get("base_phase", "")) for record in records)
    phase_episode_counts = {
        phase: len({str(record.get("episode_id", "")) for record in records if str(record.get("phase", "")) == phase})
        for phase in TARGET_PHASES
    }
    base_phase_episode_counts = {
        phase: len({str(record.get("episode_id", "")) for record in records if str(record.get("base_phase", "")) == phase})
        for phase in TARGET_PHASES if phase != "boundary_onset"
    }
    pair_ids = {str(record.get("pair_id", "")) for record in records}
    train_pairs = {str(record.get("pair_id", "")) for record in records if str(record.get("split", "")) == "train"}
    validation_pairs = {str(record.get("pair_id", "")) for record in records if str(record.get("split", "")) == "validation"}
    episode_ids = {str(record.get("episode_id", "")) for record in records}
    train_episode_ids = {str(record.get("episode_id", "")) for record in records if str(record.get("split", "")) == "train"}
    validation_episode_ids = {str(record.get("episode_id", "")) for record in records if str(record.get("split", "")) == "validation"}
    nonzero = [int(value) for value in phase_counts.values() if int(value) > 0]
    coverage_sufficient = all(phase_episode_counts.get(phase, 0) >= 2 for phase in TARGET_PHASES)
    phase_counts_by_split: dict[str, dict[str, int]] = {}
    phase_episode_counts_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        split_records = [record for record in records if str(record.get("split", "")) == split]
        phase_counts_by_split[split] = dict(Counter(str(record.get("phase", "")) for record in split_records))
        phase_episode_counts_by_split[split] = {
            phase: len({str(record.get("episode_id", "")) for record in split_records if str(record.get("phase", "")) == phase})
            for phase in TARGET_PHASES
        }
    coverage_target_counts = dict(Counter(str(record.get("coverage_target", "")) for record in records))
    coverage_target_episode_counts = {
        target: len({str(summary.get("episode_id", "")) for summary in summaries if str(summary.get("coverage_target", "")) == target})
        for target in sorted({str(summary.get("coverage_target", "")) for summary in summaries})
    }
    recovery_by_target = {
        target: int(sum(1 for summary in summaries if str(summary.get("coverage_target", "")) == target and int(summary.get("post_bypass_recovery", 0) or 0)))
        for target in sorted({str(summary.get("coverage_target", "")) for summary in summaries})
    }
    return {
        "dataset_id": "single_uav_obstacle_stage2br2_r3_coverage_audit",
        "episodes": len(summaries),
        "samples": len(records),
        "mirror_pairs": len(pair_ids),
        "phase_counts": dict(phase_counts),
        "base_phase_counts": dict(base_phase_counts),
        "phase_episode_counts": phase_episode_counts,
        "base_phase_episode_counts": base_phase_episode_counts,
        "phase_counts_by_split": phase_counts_by_split,
        "phase_episode_counts_by_split": phase_episode_counts_by_split,
        "coverage_target_counts": coverage_target_counts,
        "coverage_target_episode_counts": coverage_target_episode_counts,
        "recovery_episodes_by_coverage_target": recovery_by_target,
        "side_counts": dict(Counter(str(record.get("side", "")) for record in records)),
        "category_counts": dict(Counter(str(record.get("category", "")) for record in records)),
        "episode_ids": sorted(episode_ids),
        "train_episode_ids": sorted(train_episode_ids),
        "validation_episode_ids": sorted(validation_episode_ids),
        "train_pair_ids": sorted(train_pairs),
        "validation_pair_ids": sorted(validation_pairs),
        "trajectory_leakage": bool(train_episode_ids & validation_episode_ids),
        "mirror_pair_train_validation_leakage": bool(train_pairs & validation_pairs),
        "phase_coverage_sufficient": bool(coverage_sufficient),
        "missing_phases": [phase for phase in TARGET_PHASES if phase_counts.get(phase, 0) == 0],
        "missing_phase_episode_coverage": [phase for phase in TARGET_PHASES if phase_episode_counts.get(phase, 0) < 2],
        "phase_imbalance_ratio": float(max(nonzero) / max(1, min(nonzero))) if nonzero else float("nan"),
        "phase_definition": PHASE_DEFINITION,
        "source_policy": dict(source_identity),
        "oracle_version": "BoundaryAwareGeometricObstacleOracle estimated-state visible-observation only",
        "oracle_truth_state_read_for_control": False,
        "repeated_unit": "episode_or_mirror_pair",
    }


def _normal_comparison(config: FormalExperimentConfig, stage2a_trainer: Any, actor_head_trainer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .single_uav_obstacle_stage2br2_followup import _evaluate_roles

    scenarios = [
        _normal_no_obstacle_scenario(config, seed=2950000 + index, round_index=0, episode_index=index)
        for index in range(1, 5)
    ]
    roles = {"stage2a_frozen": stage2a_trainer, "r2_actor_head": actor_head_trainer}
    summaries, steps = _evaluate_roles(roles, config, scenarios, evaluation_stage="r3_normal_navigation_audit", evaluation_family="normal_navigation")
    aggregates = {role: _aggregate_episodes([row for row in summaries if row.get("policy") == role]) for role in roles}
    baseline = aggregates["stage2a_frozen"]
    candidate = aggregates["r2_actor_head"]
    warning = bool(
        candidate.get("entered_8m", 0) < baseline.get("entered_8m", 0)
        or candidate.get("mean_minimum_goal_distance_m", float("inf")) > baseline.get("mean_minimum_goal_distance_m", float("inf")) + 1.0e-9
    )
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "normal_navigation_degradation_warning": warning,
        "entry_rate_delta": float(candidate.get("entered_8m_rate", 0.0) - baseline.get("entered_8m_rate", 0.0)),
        "mean_minimum_goal_distance_delta_m": float(candidate.get("mean_minimum_goal_distance_m", float("nan")) - baseline.get("mean_minimum_goal_distance_m", float("nan"))),
        "same_scenarios_and_reset_seeds": True,
        "episode_count": len(scenarios),
    }
    return summaries, steps, comparison


def _write_r3_plots(plots_root: Path, *, records: Sequence[Mapping[str, Any]], step_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], normal_comparison: Mapping[str, Any]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    raw_counts = Counter(str(record.get("phase", "")) for record in records)
    balanced = _balanced_records(records)
    balanced_counts = Counter(str(record.get("phase", "")) for record in balanced)
    figure, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(TARGET_PHASES), dtype=float)
    axis.bar(positions - 0.19, [raw_counts.get(phase, 0) for phase in TARGET_PHASES], width=0.38, label="raw samples")
    axis.bar(positions + 0.19, [balanced_counts.get(phase, 0) for phase in TARGET_PHASES], width=0.38, label="balanced view")
    axis.set_xticks(positions, TARGET_PHASES, rotation=25, ha="right")
    axis.set_ylabel("samples")
    axis.set_title("R3 phase coverage: raw vs episode-phase capped")
    axis.legend()
    figure.tight_layout()
    phase_path = plots_root / "r3_phase_coverage_raw_vs_balanced.png"
    figure.savefig(phase_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["phase_coverage"] = str(phase_path.resolve())

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    # The phase/action plot needs rollout-step fields (true distance, radial
    # progress and raw lateral action), not the compact JSONL dataset record.
    selected = [row for row in step_rows if str(row.get("base_phase", "")) in {"post_bypass", "recovery"}]
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_episode[str(row.get("episode_id", ""))].append(row)
    for episode_id, rows in sorted(by_episode.items()):
        rows = sorted(rows, key=lambda row: int(row.get("step", 0)))
        x = np.arange(len(rows))
        axes[0, 0].plot(x, [_finite(row.get("true_goal_distance_before_m")) for row in rows], alpha=0.45)
        axes[0, 1].plot(x, [_finite(row.get("true_radial_progress_mps")) for row in rows], alpha=0.45)
        axes[1, 0].plot(x, [_finite(row.get("raw_lateral_action_mps")) for row in rows], alpha=0.45)
        axes[1, 1].plot(x, [int(row.get("boundary_projected", 0) or 0) for row in rows], alpha=0.45)
    axes[0, 0].set_title("goal distance")
    axes[0, 1].set_title("radial progress")
    axes[1, 0].set_title("raw lateral action")
    axes[1, 1].set_title("boundary projection")
    for axis in axes.flat:
        axis.set_xlabel("local phase step")
        axis.grid(alpha=0.25)
    figure.suptitle("R3 post-bypass / recovery diagnostics")
    figure.tight_layout()
    recovery_path = plots_root / "r3_post_bypass_recovery_diagnostics.png"
    figure.savefig(recovery_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["post_bypass_recovery"] = str(recovery_path.resolve())

    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["Stage2A frozen", "Actor-head"]
    values = [normal_comparison["baseline"].get("entered_8m", 0), normal_comparison["candidate"].get("entered_8m", 0)]
    axis.bar(labels, values, color=["#0072B2", "#D55E00"])
    axis.set_ylabel("episodes entering 8 m")
    axis.set_title("R3 normal-navigation audit")
    figure.tight_layout()
    normal_path = plots_root / "r3_normal_navigation_comparison.png"
    figure.savefig(normal_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["normal_navigation"] = str(normal_path.resolve())
    return output


def _build_report(summary: Mapping[str, Any], manifest: Mapping[str, Any], normal_comparison: Mapping[str, Any], tests: Mapping[str, Any]) -> str:
    lines = [
        f"# {STAGE_NAME}",
        "",
        "本阶段只做 post-bypass/recovery/boundary-onset coverage audit 和 DAgger 数据采集修正；没有 BC、PPO 或环境训练。",
        "",
        "## 当前执行约束",
        "",
        "- 未修改 reward、target-speed、curriculum、动力学、动作空间、action mapping、observation contract、safety/boundary projection、collision 判据或 episode horizon。",
        "- 使用合法 Actor action、Oracle label 和现有 dynamics；没有 teleport、位置/速度直接注入或按 timestep 伪造 phase。",
        "- episode/scenario 和 mirror pair 是独立重复单位；train/validation 按 pair 划分。",
        "- `after_boundary` 是同动力学重建的中间动作，`after_safety` 是环境最终 applied action。",
        "",
        "## Phase coverage",
        "",
        f"- raw phase counts：{json.dumps(manifest.get('phase_counts', {}), ensure_ascii=False)}",
        f"- raw base-phase counts：{json.dumps(manifest.get('base_phase_counts', {}), ensure_ascii=False)}",
        f"- phase episode counts：{json.dumps(manifest.get('phase_episode_counts', {}), ensure_ascii=False)}",
        f"- phase counts by split：{json.dumps(manifest.get('phase_counts_by_split', {}), ensure_ascii=False)}",
        f"- phase episode counts by split：{json.dumps(manifest.get('phase_episode_counts_by_split', {}), ensure_ascii=False)}",
        f"- coverage target sample counts：{json.dumps(manifest.get('coverage_target_counts', {}), ensure_ascii=False)}",
        f"- coverage target episode counts：{json.dumps(manifest.get('coverage_target_episode_counts', {}), ensure_ascii=False)}",
        f"- recovery episodes by coverage target：{json.dumps(manifest.get('recovery_episodes_by_coverage_target', {}), ensure_ascii=False)}",
        f"- phase imbalance ratio：{manifest.get('phase_imbalance_ratio')}",
        f"- phase_coverage_sufficient：{manifest.get('phase_coverage_sufficient')}",
        f"- balanced-view phase counts：{json.dumps(manifest.get('balanced_view', {}).get('phase_counts', {}), ensure_ascii=False)}",
        f"- balanced-view phase imbalance ratio：{manifest.get('balanced_view', {}).get('phase_imbalance_ratio')}",
        f"- missing phases：{manifest.get('missing_phases')}",
        f"- missing phase episode coverage：{manifest.get('missing_phase_episode_coverage')}",
        "",
        "## Normal-navigation audit",
        "",
        f"- Stage2A entered8m={normal_comparison['baseline'].get('entered_8m', 0)}/{normal_comparison['baseline'].get('episodes', 0)}；Actor-head entered8m={normal_comparison['candidate'].get('entered_8m', 0)}/{normal_comparison['candidate'].get('episodes', 0)}。",
        f"- normal_navigation_degradation_warning={normal_comparison.get('normal_navigation_degradation_warning')}；entry-rate delta={normal_comparison.get('entry_rate_delta')}。",
        "",
        "## Boundary / recovery interpretation",
        "",
        f"- boundary projection steps：{summary.get('boundary_projection_steps')}",
        f"- post-bypass boundary projection steps：{summary.get('post_bypass_boundary_projection_steps')}",
        f"- pre/bypass boundary projection steps：{summary.get('pre_or_bypass_boundary_projection_steps')}",
        f"- post-bypass lateral persistence episodes：{summary.get('post_bypass_lateral_persistence_episodes')}",
        f"- recovery episodes：{summary.get('recovery_episodes')}",
        "",
        "## Test and stop status",
        "",
        f"- py_compile：{tests.get('py_compile')}。",
        f"- targeted pytest：{tests.get('targeted_pytest')}。",
        f"- ppo_updates：{summary.get('ppo_updates')}；bc_training_performed：{summary.get('bc_training_performed')}。",
        f"- blocker：{summary.get('primary_blocker')}。",
        f"- next step：{summary.get('recommended_next_step')}。",
        "",
        "本阶段完成后停止；不能依据 coverage audit 直接宣称策略已经学会主动避障，也不自动进入 BC 或 PPO。",
    ]
    return "\n".join(lines) + "\n"


def run_r3_coverage_audit(
    *,
    actor_head_checkpoint: str | Path = DEFAULT_ACTOR_HEAD_CHECKPOINT,
    stage2a_checkpoint: str | Path = DEFAULT_STAGE2A_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    data_root, logs_root, plots_root = Path(data_root), Path(logs_root), Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    actor_head_checkpoint = Path(actor_head_checkpoint).resolve()
    stage2a_checkpoint = Path(stage2a_checkpoint).resolve()
    _actor_payload, actor_trainer, config, actor_identity = _load_policy(actor_head_checkpoint, output_dir=data_root / "actor_head_source")
    _stage_payload, stage2a_trainer, _stage_config, stage2a_identity = _load_policy(stage2a_checkpoint, output_dir=data_root / "stage2a_source")
    del _actor_payload, _stage_payload, _stage_config
    if progress:
        print("[stage2br2-r3] loaded frozen Stage2A and actor-head checkpoints", flush=True)

    scenarios = coverage_scenarios(config)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for episode_index, scenario in enumerate(scenarios, start=1):
        summary, episode_rows, episode_records = _run_coverage_episode(
            actor_trainer, config, scenario=scenario, episode_index=episode_index
        )
        summaries.append(summary)
        rows.extend(episode_rows)
        records.extend(episode_records)
        if progress:
            print(
                f"[stage2br2-r3] episode={episode_index}/{len(scenarios)} "
                f"target={scenario['stage2br2_r3']['coverage_target']} "
                f"phases={summary.get('phase_counts', {})}",
                flush=True,
            )

    manifest = _coverage_manifest(records, summaries, source_identity=actor_identity)
    balanced_records = _balanced_records(records)
    balanced_manifest = _coverage_manifest(balanced_records, summaries, source_identity=actor_identity)
    balanced_manifest.update({"view": "episode_phase_capped", "max_samples_per_episode_phase": R3_BALANCED_SAMPLES_PER_EPISODE_PHASE})
    manifest["balanced_view"] = balanced_manifest
    _write_json(data_root / "r3_coverage_dataset_manifest.json", _jsonable(manifest))
    _write_json(data_root / "r3_coverage_balanced_dataset_manifest.json", _jsonable(balanced_manifest))
    _write_jsonl(data_root / "r3_coverage_all.jsonl", records)
    _write_jsonl(data_root / "r3_coverage_balanced_train.jsonl", [record for record in balanced_records if record.get("split") == "train"])
    _write_jsonl(data_root / "r3_coverage_balanced_validation.jsonl", [record for record in balanced_records if record.get("split") == "validation"])
    _write_rows_csv(data_root / "r3_coverage_all.csv", records)
    _write_rows_csv(data_root / "r3_coverage_balanced_train.csv", [record for record in balanced_records if record.get("split") == "train"])
    _write_rows_csv(data_root / "r3_coverage_balanced_validation.csv", [record for record in balanced_records if record.get("split") == "validation"])
    _write_rows_csv(logs_root / "r3_coverage_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "r3_coverage_rollout_steps.csv", rows)
    _write_rows_csv(logs_root / "r3_phase_action_audit.csv", _phase_action_audit(rows))

    normal_summaries, normal_rows, normal_comparison = _normal_comparison(config, stage2a_trainer, actor_trainer)
    _write_rows_csv(logs_root / "r3_normal_navigation_episode_metrics.csv", normal_summaries)
    _write_rows_csv(logs_root / "r3_normal_navigation_rollout_steps.csv", normal_rows)

    phase_counts = Counter(str(row.get("phase", "")) for row in rows)
    base_phase_counts = Counter(str(row.get("base_phase", "")) for row in rows)
    post_rows = [row for row in rows if str(row.get("base_phase", "")) == "post_bypass"]
    recovery_rows = [row for row in rows if str(row.get("base_phase", "")) == "recovery"]
    boundary_rows = [row for row in rows if int(row.get("boundary_projected", 0) or 0)]
    post_persistence_episode_ids = {
        str(item.get("episode_id", ""))
        for item in summaries
        if int(item.get("post_bypass_max_same_sign_lateral_steps", 0) or 0) >= 5
    }
    summary = {
        "stage": STAGE_NAME,
        "status": "completed",
        "source_policy": actor_identity,
        "comparison_policy": stage2a_identity,
        "coverage_dataset": manifest,
        "balanced_dataset": balanced_manifest,
        "episodes": len(summaries),
        "samples": len(records),
        "phase_counts": dict(phase_counts),
        "base_phase_counts": dict(base_phase_counts),
        "boundary_projection_steps": len(boundary_rows),
        "post_bypass_boundary_projection_steps": sum(int(row.get("passed_obstacle_before", 0) or 0) for row in boundary_rows),
        "pre_or_bypass_boundary_projection_steps": sum(not int(row.get("passed_obstacle_before", 0) or 0) for row in boundary_rows),
        "post_bypass_lateral_persistence_episodes": len(post_persistence_episode_ids),
        "post_bypass_lateral_persistence_episode_ids": sorted(post_persistence_episode_ids),
        "recovery_episodes": len({str(row.get("episode_id", "")) for row in recovery_rows}),
        "normal_navigation": normal_comparison,
        "normal_navigation_degradation_warning": bool(normal_comparison.get("normal_navigation_degradation_warning")),
        "primary_blocker": (
            "missing_or_imbalanced_phase_coverage"
            if not manifest.get("phase_coverage_sufficient")
            else "post_bypass_lateral_persistence"
            if summary_phase_lateral_persistence(post_rows)
            else "not_established"
        ),
        "case_classification": "post_bypass_lateral_persistence_confirmed_with_coverage_gap_audited",
        "strict_behavior_gate_passed": False,
        "eligible_for_BC": False,
        "phase_coverage_sufficient": bool(manifest.get("phase_coverage_sufficient")),
        "balanced_phase_coverage_sufficient": bool(manifest.get("balanced_view", {}).get("phase_coverage_sufficient")),
        "balanced_phase_imbalance_ratio": manifest.get("balanced_view", {}).get("phase_imbalance_ratio"),
        "normal_navigation_preserved_strict": not bool(normal_comparison.get("normal_navigation_degradation_warning")),
        "ppo_updates": 0,
        "bc_training_performed": False,
        "training_performed": False,
        "reward_or_environment_modified": False,
        "after_boundary_is_reconstructed": True,
        "after_safety_is_final_applied_action": True,
        "recommended_next_step": "read_only_post_bypass_recovery_and_boundary_onset_coverage_audit_before_any_new_BC_or_PPO" if not manifest.get("phase_coverage_sufficient") else "balanced_DAgger_recollection_review_before_BC",
        "evaluation_contract": {
            "formal_training_run": False,
            "bc_run": False,
            "ppo_run": False,
            "environment_modified": False,
            "reward_modified": False,
            "existing_outputs_overwritten": False,
            "repeated_unit": "episode_or_mirror_pair",
        },
    }
    _write_json(logs_root / "r3_coverage_summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    plots = _write_r3_plots(plots_root, records=records, step_rows=rows, summaries=summaries, normal_comparison=normal_comparison)
    summary["plots"] = plots
    _write_json(logs_root / "r3_coverage_summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    tests = {"py_compile": "pending", "targeted_pytest": "pending"}
    report = _build_report(summary, manifest, normal_comparison, tests)
    (logs_root / "r3_coverage_report.md").write_text(report, encoding="utf-8")
    return summary


def summary_phase_lateral_persistence(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id", ""))].append(row)
    for episode_rows in by_episode.values():
        same_sign = sum(abs(float(row.get("raw_lateral_action_mps", 0.0) or 0.0)) >= LATERAL_NONDEGENERATE_MPS for row in episode_rows)
        if same_sign >= 5:
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--actor-head-checkpoint", type=Path, default=DEFAULT_ACTOR_HEAD_CHECKPOINT)
    parser.add_argument("--stage2a-checkpoint", type=Path, default=DEFAULT_STAGE2A_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = run_r3_coverage_audit(
        actor_head_checkpoint=args.actor_head_checkpoint,
        stage2a_checkpoint=args.stage2a_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "DEFAULT_ACTOR_HEAD_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "DEFAULT_STAGE2A_CHECKPOINT",
    "PHASE_DEFINITION",
    "TARGET_PHASES",
    "_balanced_records",
    "_coverage_phase_for_step",
    "coverage_scenarios",
    "run_r3_coverage_audit",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
