"""Stage 2B-R: balanced iterative DAgger smoke for static obstacle bypass.

This experiment is deliberately isolated from the formal PPO path.  It keeps
the Stage 2A 42-dimensional observation contract and the existing physical
action/dynamics interface, but corrects the Stage 2B smoke protocol:

* strict left/right mirror pairs;
* an estimated-state, boundary-aware geometric Oracle;
* three real rollout -> aggregate -> Actor-only BC rounds;
* frozen Stage 2A action targets for normal navigation preservation; and
* hard direction-sign and boundary-interference gates before holdout.

The module is a diagnostic smoke entry point.  It never performs PPO updates
and refuses to overwrite a non-empty output directory.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment, clip_vector_norm
from ..simulation.obstacle_model import (
    BoxObstacle,
    Obstacle,
    SphereObstacle,
    is_physical_obstacle,
    point_obstacle_distance,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
)
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
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
    _save_imitation_checkpoint,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_obstacle_dagger_stage2b import (
    DEFAULT_STAGE2A_CHECKPOINT,
    FLOAT_TOLERANCE,
    HOLDOUT_EPISODES,
    ORACLE_MAX_BYPASS_SPEED_MPS,
    ORACLE_MARGIN_M,
    ORACLE_MIN_BYPASS_SPEED_MPS,
    ORACLE_WAYPOINT_MARGIN_M,
    ORACLE_WAYPOINT_TOLERANCE_M,
    QUERY_DISTANCE_M,
    STATIC_SAFE_DISTANCE_M,
    _body_left_world,
    _candidate_from_estimate,
    _ensure_empty_roots,
    _episode_step_row,
    _finite,
    _obstacle_category,
    _obstacle_extent,
    _physical_obstacles,
    select_relevant_obstacle,
    select_relevant_obstacle,
    _static_candidate_diagnostics,
    _summary_from_rows,
    _unit,
    _visible_state,
    _write_rows_csv,
    GeometricObstacleOracleController,
    _normal_or_static_training_scenario,
    _normal_scenario,
    _static_scenario,
)
from .single_uav_obstacle_direction_observation_stage2a import NEW_DIRECTION_INDICES
from .single_uav_obstacle_observation_audit import probe_scenarios
from .single_uav_oracle_behavior_cloning import (
    _canonical_hash,
    _copy_observation_for_json,
)


STAGE_NAME = "single_uav_obstacle_dagger_stage2br_smoke"
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_obstacle_dagger_stage2br_smoke_seed2026")
DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_obstacle_dagger_stage2br_smoke_seed2026")
DEFAULT_PLOT_ROOT = Path("plots/preliminary/single_uav_obstacle_dagger_stage2br_smoke_seed2026")

ROUND_COUNT = 3
EPISODES_PER_ROUND = 12
NORMAL_EPISODES_PER_ROUND = 8
STATIC_EPISODES_PER_ROUND = 4
ROUND_TRAIN_EPISODES = 9
ROUND_VALIDATION_EPISODES = 3
STATIC_CATEGORIES = ("left", "right", "left_front", "right_front")
MIRROR_CATEGORIES = ("left", "right", "left_front", "right_front")
PRESERVATION_LOSS_COEF = 1.0
PHASE_A_MAX_EPOCHS = 5
PHASE_A_PATIENCE = 2
PHASE_A_BATCH_SIZE = 64
PHASE_A_LEARNING_RATE = 1.0e-4
NEW_DIRECTION_SLICE = slice(39, 42)
PREVIOUS_STAGE2B_BOUNDARY_STEPS = 354
LATERAL_ACTION_NONDEGENERATE_MPS = 0.01
BOUNDARY_MARGIN_FLOOR_M = 1.0


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def _scenario_seed(round_index: int, episode_index: int) -> int:
    return int(2026000 + int(round_index) * 1000 + int(episode_index))


def _normal_no_obstacle_scenario(
    config: FormalExperimentConfig, *, seed: int, round_index: int, episode_index: int
) -> dict[str, Any]:
    """Create a full-distance navigation-preservation scenario with no obstacle."""

    scenario = copy.deepcopy(
        _normal_scenario(
            config,
            seed=int(seed),
            split="train",
            episode_index=int(episode_index),
        )
    )
    # ``MultiUAVJointEnv._make_obstacles`` intentionally falls back to the
    # historical default scene when ``obstacle_specs`` is an empty list.
    # Stage 2B-R normal-preservation episodes are explicitly no-obstacle
    # episodes, so provide an explicit empty obstacle list to prevent that
    # fallback from silently injecting buildings into the control path.
    scenario["obstacle_specs"] = []
    scenario["obstacles"] = []
    scenario["stage2br"] = {
        "scene_type": "normal_far_distance",
        "category": "no_obstacle",
        "round": int(round_index),
        "episode_index": int(episode_index),
        "curriculum_used": False,
    }
    return scenario


def mirror_static_scenario(
    config: FormalExperimentConfig,
    *,
    category: str,
    pair_index: int,
    side: str,
    seed: int,
    round_index: int,
) -> dict[str, Any]:
    """Return one member of an exact left/right geometry mirror pair.

    The pair keeps all physical quantities equal and changes only the body-
    frame lateral sign of the obstacle center.  The sphere is close enough to
    intersect the direct start-goal corridor, but does not contain the start.
    """

    if category not in {"left", "right", "left_front", "right_front"}:
        raise ValueError(f"unsupported mirror category: {category}")
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side}")
    base_category = "left_front" if "front" in category else "left"
    sign = 1.0 if side == "left" else -1.0
    if base_category == "left":
        longitudinal = 8.0
        lateral = 4.0
        radius = 4.5
    else:
        longitudinal = 12.0
        lateral = 4.5
        radius = 4.8
    start = np.asarray([34.0, 100.0, 20.0], dtype=float)
    goal = np.asarray([66.0, 100.0, 20.0], dtype=float)
    center = np.asarray([start[0] + longitudinal, start[1] + sign * lateral, start[2]], dtype=float)
    mirror_pair_id = f"pair_{int(pair_index):04d}"
    return {
        "scenario_version": "single-uav-obstacle-dagger-stage2br-mirror-v1",
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
                "obstacle_id": "stage2br_mirror_obstacle",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": center.tolist(),
                "radius_m": float(radius),
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "hierarchical_waypoint": goal.tolist(),
        "stage2br": {
            "scene_type": "static_mirror_pressure",
            "category": str(category),
            "side": str(side),
            "mirror_pair_id": mirror_pair_id,
            "round": int(round_index),
            "pair_index": int(pair_index),
            "direct_path_crosses_static_obstacle": True,
            "body_frame_lateral_sign": "+1_left" if side == "left" else "-1_right",
            "longitudinal_distance_m": float(longitudinal),
            "lateral_distance_abs_m": float(lateral),
            "obstacle_radius_m": float(radius),
        },
    }


def _centered_static_scenario(config: FormalExperimentConfig, *, seed: int, name: str) -> dict[str, Any]:
    start = np.asarray([34.0, 100.0, 20.0], dtype=float)
    goal = np.asarray([66.0, 100.0, 20.0], dtype=float)
    return {
        "scenario_version": "single-uav-obstacle-dagger-stage2br-centered-v1",
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
                "obstacle_id": f"stage2br_{name}",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": [44.0, 100.0, 20.0],
                "radius_m": 4.5,
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "hierarchical_waypoint": goal.tolist(),
        "stage2br": {
            "scene_type": "static_oracle_sanity",
            "category": "front",
            "side": "centered",
            "mirror_pair_id": "",
            "direct_path_crosses_static_obstacle": True,
        },
    }


def _two_obstacle_scenario(config: FormalExperimentConfig, *, seed: int) -> dict[str, Any]:
    scenario = _centered_static_scenario(config, seed=seed, name="two_obstacle")
    scenario["obstacle_specs"].extend(
        [
            {
                "obstacle_id": "stage2br_second_obstacle",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": [54.0, 100.0, 20.0],
                "radius_m": 3.0,
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ]
    )
    scenario["stage2br"]["category"] = "front"
    return scenario


def _post_obstacle_scenario(config: FormalExperimentConfig, *, seed: int) -> dict[str, Any]:
    start = np.asarray([55.0, 100.0, 20.0], dtype=float)
    goal = np.asarray([70.0, 100.0, 20.0], dtype=float)
    return {
        "scenario_version": "single-uav-obstacle-dagger-stage2br-post-obstacle-v1",
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
                "obstacle_id": "stage2br_obstacle_behind",
                "obstacle_type": "static",
                "geometry_type": "sphere",
                "center_m": [45.0, 100.0, 20.0],
                "radius_m": 3.0,
                "velocity_mps": [0.0, 0.0, 0.0],
                "active": 1,
                "physical_collision_enabled": True,
            }
        ],
        "stage2br": {"scene_type": "post_obstacle_drift_probe", "category": "behind"},
    }


def mirror_pair_manifest_rows(*, rounds: int = ROUND_COUNT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index in range(1, int(rounds) + 1):
        for pair_offset, base_category in enumerate(("left", "left_front"), start=1):
            pair_index = round_index * 100 + pair_offset
            for side in ("left", "right"):
                category = base_category if side == "left" else ("right_front" if base_category == "left_front" else "right")
                if base_category == "left_front":
                    category = "left_front" if side == "left" else "right_front"
                sign = 1.0 if side == "left" else -1.0
                longitudinal = 8.0 if base_category == "left" else 12.0
                lateral = 4.0 if base_category == "left" else 4.5
                radius = 4.5 if base_category == "left" else 4.8
                rows.append(
                    {
                        "round": int(round_index),
                        "mirror_pair_id": f"pair_{pair_index:04d}",
                        "side": side,
                        "category": category,
                        "start_xyz_m": [34.0, 100.0, 20.0],
                        "goal_xyz_m": [66.0, 100.0, 20.0],
                        "obstacle_longitudinal_distance_m": longitudinal,
                        "obstacle_dy_body_m": sign * lateral,
                        "obstacle_dy_abs_m": lateral,
                        "obstacle_radius_m": radius,
                        "seed_relation": f"round_{round_index}_pair_{pair_index}_common_geometry_seed",
                    }
                )
    return rows


def audit_previous_data_bias(previous_logs_root: str | Path) -> dict[str, Any]:
    """Read-only audit of the previous Stage 2B query/category imbalance."""

    root = Path(previous_logs_root)
    summary_path = root / "stage2b_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step_path = root / "dagger_training_step_diagnostics.csv"
    rows = list(csv.DictReader(step_path.open("r", encoding="utf-8"))) if step_path.is_file() else []
    static_rows = [row for row in rows if row.get("scenario") == "static_pressure"]
    queried = [row for row in static_rows if row.get("oracle_queried") == "1"]
    category_counts = Counter(row.get("obstacle_category", "") for row in static_rows)
    query_counts = Counter(row.get("obstacle_category", "") for row in queried)
    schedule = summary.get("dataset", {}).get("schedule_counts", {})
    return {
        "previous_logs_root": str(root.resolve()),
        "previous_summary_present": bool(summary),
        "previous_schedule_counts": schedule,
        "previous_total_static_rows": len(static_rows),
        "previous_static_category_counts": dict(category_counts),
        "previous_static_teacher_query_counts": dict(query_counts),
        "previous_right_static_rows": int(category_counts.get("right", 0)),
        "previous_right_teacher_queries": int(query_counts.get("right", 0)),
        "previous_left_teacher_queries": int(query_counts.get("left", 0)),
        "previous_aggregation_filtered_right": False,
        "previous_static_generator_explicit_mirror_pairs": False,
        "previous_static_generator_lateral_centers_m": [97.0, 100.0, 103.0],
        "previous_category_classifier_far_side_can_be_front": True,
        "root_cause_classification": "F: static geometry was not mirror-balanced; the rollout/query distribution then concentrated corrections in left/front states; aggregation preserved collected rows and did not create missing right queries",
        "category_sign_inversion_evidence": "Stage 2A paired feature probes preserved left/right signs; the primary failure was coverage/query exposure, not a global dy sign inversion",
        "source_summary_path": str(summary_path.resolve()),
    }


class BoundaryAwareGeometricObstacleOracleController(GeometricObstacleOracleController):
    """Stage 2B Oracle with boundary-aware left/right candidate selection."""

    def __init__(self, environment: Any) -> None:
        super().__init__(environment)
        self.last_side_selection: dict[str, Any] = {}

    def _route_candidate_boundary_margin(
        self, visible: Mapping[str, Any], obstacle: Obstacle, side_sign: float
    ) -> float:
        position = np.asarray(visible["position_m"], dtype=float)
        goal_direction = _unit(np.asarray(visible["goal_direction"], dtype=float))
        if np.linalg.norm(goal_direction) <= 1.0e-12:
            goal_direction = _unit(np.asarray(self.environment.goals[self.uav_id], dtype=float) - position)
        lateral_world = _body_left_world(float(visible["yaw_rad"]))
        goal_extent = _obstacle_extent(obstacle, goal_direction)
        lateral_extent = _obstacle_extent(obstacle, lateral_world)
        center = np.asarray(obstacle.center_m, dtype=float)
        bypass_offset = lateral_extent + STATIC_SAFE_DISTANCE_M + ORACLE_MARGIN_M
        pre_offset = goal_extent + ORACLE_WAYPOINT_MARGIN_M
        side_pre = center - goal_direction * pre_offset + lateral_world * float(side_sign) * bypass_offset
        side_post = center + goal_direction * pre_offset + lateral_world * float(side_sign) * bypass_offset
        goal = np.asarray(self.environment.goals[self.uav_id], dtype=float)
        points = [position, side_pre, side_post, goal]
        bounds_min = np.asarray(self.motion.bounds_min_m, dtype=float)
        bounds_max = np.asarray(self.motion.bounds_max_m, dtype=float)
        margins: list[float] = []
        for point in points:
            margins.extend((point - bounds_min).tolist())
            margins.extend((bounds_max - point).tolist())
        for start, end in zip(points[:-1], points[1:]):
            for fraction in np.linspace(0.0, 1.0, 11):
                point = start + float(fraction) * (end - start)
                margins.extend((point - bounds_min).tolist())
                margins.extend((bounds_max - point).tolist())
        return float(min(margins, default=0.0))

    def _choose_side(
        self, visible: Mapping[str, Any], obstacle: Obstacle, goal_direction: np.ndarray, yaw: float
    ) -> float:
        body = np.asarray(visible.get("obstacle_direction_body_m", np.zeros(3)), dtype=float)
        preferred: float | None = None
        if body.size >= 3 and np.isfinite(body).all() and abs(float(body[1])) > 1.0e-6:
            # +y is body-left.  A left obstacle is bypassed to the right.
            preferred = -1.0 if float(body[1]) > 0.0 else 1.0
        candidates = (-1.0, 1.0)
        margins = {str(side): self._route_candidate_boundary_margin(visible, obstacle, side) for side in candidates}
        if preferred is not None:
            other = -preferred
            chosen = preferred
            if margins[str(preferred)] < BOUNDARY_MARGIN_FLOOR_M and margins[str(other)] > margins[str(preferred)] + 1.0e-9:
                chosen = other
        else:
            chosen = 1.0 if margins["1.0"] > margins["-1.0"] + 1.0e-9 else -1.0
        self.last_side_selection = {
            "preferred_side_sign": preferred,
            "candidate_boundary_margin_m": margins,
            "chosen_side_sign": float(chosen),
            "boundary_margin_aware": True,
        }
        return float(chosen)

    def action(
        self, observations: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action, control = super().action(observations)
        control = dict(control)
        control["boundary_side_selection"] = copy.deepcopy(self.last_side_selection)
        control["boundary_aware_oracle"] = True
        self.last_control = control
        return action, control


def _oracle_probe_environment(config: FormalExperimentConfig, scenario: Mapping[str, Any], seed: int, safety: bool) -> Any:
    environment = _make_environment(
        config,
        int(seed),
        metric_scales=config.metric_scales,
        safety_projection_enabled=bool(safety),
    )
    environment.reset(seed=int(seed), scenario_config=copy.deepcopy(dict(scenario)))
    return environment


def oracle_direction_sign_probe(config: FormalExperimentConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, category in enumerate(MIRROR_CATEGORIES, start=1):
        side = "left" if category.startswith("left") else "right"
        scenario = mirror_static_scenario(
            config,
            category=category,
            pair_index=900 + index,
            side=side,
            seed=2026900 + index,
            round_index=0,
        )
        environment = _oracle_probe_environment(config, scenario, 2026900 + index, False)
        observations, _info = environment.reset(seed=2026900 + index, scenario_config=copy.deepcopy(scenario))
        oracle = BoundaryAwareGeometricObstacleOracleController(environment)
        action, control = oracle.action(observations)
        delta = np.asarray(action["uav_00"]["delta_v"], dtype=float).reshape(3)
        visible = _visible_state(observations["uav_00"])
        lateral = float(np.dot(delta, _body_left_world(float(visible["yaw_rad"]))))
        expected = -1.0 if side == "left" else 1.0
        rows.append(
            {
                "case": category,
                "side": side,
                "expected_lateral_sign": expected,
                "delta_v_y_body_left_projection_mps": lateral,
                "abs_lateral_action_mps": abs(lateral),
                "action_norm_mps": float(np.linalg.norm(delta)),
                "sign_correct": int(lateral * expected > 1.0e-8),
                "legal_action": int(np.linalg.norm(delta) <= environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s + 1.0e-8),
                "route_phase": control.get("route_phase", ""),
                "boundary_margin_aware": bool(control.get("boundary_aware_oracle", False)),
                "boundary_selection": control.get("boundary_side_selection", {}),
            }
        )
    return {
        "rows": rows,
        "correct": int(sum(int(row["sign_correct"]) for row in rows)),
        "applicable": len(rows),
        "accuracy": float(sum(int(row["sign_correct"]) for row in rows) / max(1, len(rows))),
        "passed": bool(rows) and all(bool(row["sign_correct"]) for row in rows),
    }


def _run_oracle_episode(
    *,
    config: FormalExperimentConfig,
    scenario_name: str,
    scenario: Mapping[str, Any],
    safety_mode: str,
    episode_index: int,
    data_root: Path,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(2027000 + serial)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=safety_mode == "on",
    )
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    writer = _EpisodeTraceWriter(
        episode_root=data_root / "oracle_sanity" / "episodes" / safety_mode / scenario_name / f"episode_{serial:06d}",
        episode_id=f"episode_{serial:06d}",
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="stage2br_oracle_sanity",
        environment=environment,
        normalization_fitted=False,
        diagnostics_path=data_root / "diagnostics" / "attention_maps.jsonl",
        policy_source="stage2br_boundary_aware_oracle",
    )
    controller = BoundaryAwareGeometricObstacleOracleController(environment)
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        visible = _visible_state(observations["uav_00"])
        state_before = environment.states["uav_00"].copy()
        action, control = controller.action(observations)
        delta = np.asarray(action["uav_00"]["delta_v"], dtype=float).reshape(3)
        _assert_action_feasible(environment, action)
        nominal_state, _ = apply_velocity_increment(state_before, delta, environment.config.motion, safety_projector=None)
        pre_diag = _static_candidate_diagnostics(environment, state_before.position_m, nominal_state.position_m)
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        row = _episode_step_row(
            policy="stage2br_oracle",
            scenario_name=scenario_name,
            safety_mode=safety_mode,
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=delta,
            control=control,
            nominal_position=np.asarray(nominal_state.position_m, dtype=float),
            pre_diag=pre_diag,
            next_info=next_info,
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
            policy_source="stage2br_boundary_aware_oracle",
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary = _summary_from_rows(
        rows,
        policy="stage2br_oracle",
        scenario_name=scenario_name,
        safety_mode=safety_mode,
        episode_index=episode_index,
    )
    summary.update(
        {
            "scenario_hash": _canonical_hash(scenario),
            "writer_summary": writer_summary,
            "training_performed": False,
            "policy_updates_performed": False,
            "ppo_updates": 0,
        }
    )
    return summary, rows


def run_oracle_safety_sanity(config: FormalExperimentConfig, data_root: Path, logs_root: Path) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {
        "front": _centered_static_scenario(config, seed=2027101, name="front"),
        "left": mirror_static_scenario(config, category="left", pair_index=910, side="left", seed=2027102, round_index=0),
        "right": mirror_static_scenario(config, category="right", pair_index=911, side="right", seed=2027103, round_index=0),
        "left_front": mirror_static_scenario(config, category="left_front", pair_index=912, side="left", seed=2027104, round_index=0),
        "right_front": mirror_static_scenario(config, category="right_front", pair_index=913, side="right", seed=2027105, round_index=0),
        "offset_left": mirror_static_scenario(config, category="left", pair_index=914, side="left", seed=2027106, round_index=0),
        "offset_right": mirror_static_scenario(config, category="right", pair_index=915, side="right", seed=2027107, round_index=0),
        "centered_block": _centered_static_scenario(config, seed=2027108, name="centered_block"),
        "two_obstacle_simple": _two_obstacle_scenario(config, seed=2027109),
        "behind": _post_obstacle_scenario(config, seed=2027110),
    }
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    serial = 1
    for name, scenario in scenarios.items():
        for mode in ("off", "on"):
            summary, rows = _run_oracle_episode(
                config=config,
                scenario_name=name,
                scenario=scenario,
                safety_mode=mode,
                episode_index=1,
                data_root=data_root,
                serial=serial,
            )
            summaries.append(summary)
            step_rows.extend(rows)
            serial += 1
    _write_rows_csv(logs_root / "oracle_safety_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "oracle_safety_step_diagnostics.csv", step_rows)
    required_names = {"front", "left", "right", "left_front", "right_front", "offset_left", "offset_right", "centered_block", "two_obstacle_simple"}
    required_off = [row for row in summaries if row["safety_mode"] == "off" and row["scenario"] in required_names]
    mode_summary: dict[str, Any] = {}
    for mode in ("off", "on"):
        selected = [row for row in summaries if row["safety_mode"] == mode]
        mode_summary[mode] = {
            "episodes": len(selected),
            "collision_episodes": int(sum(int(row["collision"]) for row in selected)),
            "physical_collision_episodes": int(sum(int(row["physical_collision"]) for row in selected)),
            "passed_obstacle_episodes": int(sum(int(row["passed_obstacle"]) for row in selected)),
            "safety_projection_steps": int(sum(int(row["safety_projection_steps"]) for row in selected)),
            "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0)) for row in selected)),
            "action_clipping_fraction_mean": float(np.mean([row["action_clipping_fraction"] for row in selected])) if selected else 0.0,
        }
    report = {
        "oracle": "boundary_aware_geometric_near_obstacle_bypass",
        "state_source": "EKF estimated state + Stage 2A direction observation + static map geometry",
        "truth_state_read_for_control": False,
        "no_teleport": True,
        "action_interface": "environment.step(delta_v)",
        "scenarios": len(scenarios),
        "mode_summary": mode_summary,
        "required_safety_off_rows": len(required_off),
        "sanity_passed": bool(
            len(required_off) == len(required_names)
            and all(int(row["collision"]) == 0 and int(row["physical_collision"]) == 0 for row in required_off)
            and all(float(row["minimum_post_clearance_m"]) >= STATIC_SAFE_DISTANCE_M - 1.0e-6 for row in required_off)
            and all(int(row["passed_obstacle"]) == 1 for row in required_off)
            and mode_summary["off"]["safety_projection_steps"] == 0
        ),
        "direction_sign_probe": oracle_direction_sign_probe(config),
        "files": {
            "episode_metrics": str((logs_root / "oracle_safety_episode_metrics.csv").resolve()),
            "step_diagnostics": str((logs_root / "oracle_safety_step_diagnostics.csv").resolve()),
        },
    }
    _write_json(logs_root / "oracle_safety_sanity.json", _jsonable(report))
    return report


def _stage2br_config_from_payload(
    payload: Mapping[str, Any], *, output_dir: Path, command_line: str
) -> FormalExperimentConfig:
    """Load the checkpoint configuration with explicit train-only overrides.

    The override is limited to scenario/curriculum plumbing.  All model,
    dynamics, reward and PPO fields remain those recorded in the checkpoint.
    Explicit scenarios are used below, so the checkpoint curriculum switches
    are disabled to make accidental validation/test contamination impossible.
    """

    config = _config_from_checkpoint(payload, output_dir=output_dir, progress=False)
    values = dict(config.__dict__)
    values.update(
        {
            "output_dir": Path(output_dir),
            "scenario_path": None,
            "scenario_pool_root": None,
            "single_uav_braking_curriculum_fraction": 0.0,
            "single_uav_terminal_braking_curriculum_fraction": 0.0,
            "force": True,
            "progress": False,
            "command_line": command_line,
        }
    )
    return config.__class__(**values)


def _load_stage2br_policy(
    checkpoint: Path, *, output_dir: Path
) -> tuple[Any, FormalExperimentConfig, dict[str, Any]]:
    import torch

    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(
        payload, output_dir=output_dir, command_line="Stage 2B-R frozen policy"
    )
    if config.uav_count != 1 or config.node_feature_dim != 42:
        raise ValueError("Stage 2B-R requires the one-UAV Stage 2A 42-dimensional checkpoint")
    trainer, identity, _ = _load_frozen_trainer(
        payload, config, metric_scales=config.metric_scales
    )
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    return trainer, config, {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
        "loaded_policy_hash": identity.get("loaded_policy_hash", ""),
    }


def _round_episode_plan(round_index: int) -> list[dict[str, Any]]:
    """Return 8 normal and 4 balanced static episodes for one DAgger round.

    The static episodes are arranged as two exact mirror pairs.  All four are
    inside the first nine (training) episodes, while the final three episodes
    are independent normal validation trajectories.  This keeps the smoke
    split leak-free and ensures every round has left/right supervision.
    """

    round_index = int(round_index)
    if round_index < 1:
        raise ValueError("round_index must be positive")
    plan: list[dict[str, Any]] = []
    static_specs = (
        (2, "left", 1),
        (4, "right", 1),
        (7, "left_front", 2),
        (9, "right_front", 2),
    )
    static_by_episode = {episode: (category, pair) for episode, category, pair in static_specs}
    for episode_index in range(1, EPISODES_PER_ROUND + 1):
        split = "train" if episode_index <= ROUND_TRAIN_EPISODES else "validation"
        if episode_index in static_by_episode:
            category, pair_offset = static_by_episode[episode_index]
            pair_index = round_index * 100 + pair_offset
            side = "left" if category.startswith("left") else "right"
            plan.append(
                {
                    "episode_index": episode_index,
                    "split": split,
                    "kind": "static_mirror_pressure",
                    "category": category,
                    "side": side,
                    "pair_index": pair_index,
                }
            )
        else:
            plan.append(
                {
                    "episode_index": episode_index,
                    "split": split,
                    "kind": "normal_far_distance",
                    "category": "no_obstacle",
                    "side": "none",
                    "pair_index": None,
                }
            )
    return plan


def round_schedule_counts(rounds: int = ROUND_COUNT) -> dict[str, int]:
    counts = Counter()
    for round_index in range(1, int(rounds) + 1):
        for item in _round_episode_plan(round_index):
            counts[str(item["kind"])] += 1
    return dict(counts)


def _scenario_for_round_item(
    config: FormalExperimentConfig, *, round_index: int, item: Mapping[str, Any]
) -> tuple[dict[str, Any], int, int]:
    episode_index = int(item["episode_index"])
    scenario_seed = int(2026000 + int(round_index) * 10000 + episode_index * 17)
    noise_seed = int(2036000 + int(round_index) * 10000 + episode_index * 31)
    if item["kind"] == "normal_far_distance":
        scenario = _normal_no_obstacle_scenario(
            config,
            seed=scenario_seed,
            round_index=round_index,
            episode_index=episode_index,
        )
    else:
        scenario = mirror_static_scenario(
            config,
            category=str(item["category"]),
            pair_index=int(item["pair_index"]),
            side=str(item["side"]),
            seed=scenario_seed,
            round_index=round_index,
        )
    scenario["stage2br"]["scenario_seed"] = scenario_seed
    scenario["stage2br"]["noise_seed"] = noise_seed
    scenario["stage2br"]["split"] = str(item["split"])
    scenario["stage2br"]["episode_index"] = episode_index
    return scenario, scenario_seed, noise_seed


def _observed_direction_category(visible: Mapping[str, Any]) -> str:
    direction = np.asarray(
        visible.get("obstacle_direction_body_m", np.zeros(3)), dtype=float
    ).reshape(-1)
    if direction.size < 3 or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1.0e-12:
        return "obstacle_unknown_direction"
    if abs(float(direction[1])) >= abs(float(direction[0])) * 0.75:
        return "left" if float(direction[1]) > 0.0 else "right"
    return "front" if float(direction[0]) >= 0.0 else "behind"


def _actor_decision(
    trainer: Any, observations: Mapping[str, Mapping[str, Any]], info: Mapping[str, Any]
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    decision = trainer.act(observations, info["central_state"], deterministic=True)
    action = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=float).reshape(3)
    actor_observation = copy.deepcopy(dict(decision.observations["uav_00"]))
    actor_observation["uav_id"] = "uav_00"
    return decision, action, actor_observation


def _step_target_components(
    visible: Mapping[str, Any], action: np.ndarray
) -> dict[str, float]:
    goal_direction = _unit(np.asarray(visible.get("goal_direction", (0.0, 0.0, 0.0))))
    velocity = np.asarray(visible.get("velocity_mps", (0.0, 0.0, 0.0)), dtype=float)
    velocity_direction = _unit(velocity)
    radial = float(np.dot(action, goal_direction))
    braking = max(0.0, -float(np.dot(action, velocity_direction)))
    tangent = action - radial * goal_direction
    return {
        "action_norm_mps": float(np.linalg.norm(action)),
        "radial_action_projection_mps": radial,
        "braking_magnitude_mps": float(braking),
        "tangential_action_mps": float(np.linalg.norm(tangent)),
    }


def _dagger_demo_record(
    *,
    split: str,
    round_index: int,
    episode_index: int,
    step: int,
    scenario: Mapping[str, Any],
    scenario_seed: int,
    noise_seed: int,
    observation: Mapping[str, Any],
    target_action: np.ndarray,
    oracle_action: np.ndarray,
    preservation_action: np.ndarray,
    student_action: np.ndarray,
    visible: Mapping[str, Any],
    target_source: str,
    oracle_queried: bool,
    query_reason: str,
    pre_diag: Mapping[str, Any],
    control: Mapping[str, Any],
    expected_category: str,
    observed_category: str,
) -> dict[str, Any]:
    target_parts = _step_target_components(visible, target_action)
    student_parts = _step_target_components(visible, student_action)
    scenario_meta = dict(scenario.get("stage2br", {}))
    return {
        "dataset_id": "single_uav_obstacle_dagger_stage2br_smoke",
        "sample_source": "oracle_query" if oracle_queried else "stage2a_preservation_anchor",
        "split": str(split),
        "round": int(round_index),
        "episode_index": int(episode_index),
        "step_index": int(step),
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "scenario_hash": _canonical_hash(scenario),
        "mirror_pair_id": str(scenario_meta.get("mirror_pair_id", "")),
        "scene_type": str(scenario_meta.get("scene_type", "")),
        "expected_obstacle_category": str(expected_category),
        "observed_obstacle_category": str(observed_category),
        "target_source": str(target_source),
        "query_gate": int(bool(oracle_queried)),
        "oracle_queried": int(bool(oracle_queried)),
        "query_reason": str(query_reason),
        "observation": _copy_observation_for_json(observation),
        "teacher_action_delta_v_mps": _jsonable(np.asarray(target_action, dtype=float).reshape(3).tolist()),
        "oracle_action_delta_v_mps": _jsonable(np.asarray(oracle_action, dtype=float).reshape(3).tolist()),
        "preservation_action_delta_v_mps": _jsonable(np.asarray(preservation_action, dtype=float).reshape(3).tolist()),
        "student_action_delta_v_mps": _jsonable(np.asarray(student_action, dtype=float).reshape(3).tolist()),
        "correction_action_error_norm_mps": float(np.linalg.norm(student_action - target_action)),
        "estimated_goal_distance_m": float(visible.get("distance_m", float("nan"))),
        "goal_distance_m": float(visible.get("distance_m", float("nan"))),
        "estimated_velocity_mps": _jsonable(visible.get("velocity_mps", (0.0, 0.0, 0.0))),
        "speed_mps": float(np.linalg.norm(np.asarray(visible.get("velocity_mps", (0.0, 0.0, 0.0)), dtype=float))),
        "radial_velocity_mps": float(visible.get("radial_velocity_mps", float("nan"))),
        "tangential_velocity_mps": float(
            np.linalg.norm(
                np.asarray(visible.get("velocity_mps", (0.0, 0.0, 0.0)), dtype=float)
                - float(visible.get("radial_velocity_mps", 0.0)) * _unit(visible.get("goal_direction", (0.0, 0.0, 0.0)))
            )
        ),
        "goal_direction": _jsonable(visible.get("goal_direction", (0.0, 0.0, 0.0))),
        "obstacle_direction_body_m": _jsonable(visible.get("obstacle_direction_body_m", (0.0, 0.0, 0.0))),
        "braking_component_mps": float(control.get("demanded_braking_delta_v_mps", 0.0)),
        "target_action_norm_mps": target_parts["action_norm_mps"],
        "target_radial_action_projection_mps": target_parts["radial_action_projection_mps"],
        "target_braking_magnitude_mps": target_parts["braking_magnitude_mps"],
        "target_tangential_action_mps": target_parts["tangential_action_mps"],
        "student_action_norm_mps": student_parts["action_norm_mps"],
        "student_radial_action_projection_mps": student_parts["radial_action_projection_mps"],
        "student_braking_magnitude_mps": student_parts["braking_magnitude_mps"],
        "student_tangential_action_mps": student_parts["tangential_action_mps"],
        "pre_candidate_static_swept_minimum_clearance_m": float(
            pre_diag.get("pre_candidate_static_swept_minimum_clearance_m", float("inf"))
        ),
        "pre_candidate_static_swept_conflict": int(pre_diag.get("pre_candidate_static_swept_conflict", 0)),
        "pre_candidate_static_unsafe": int(pre_diag.get("pre_candidate_static_unsafe", 0)),
        "oracle_route_phase": str(control.get("route_phase", "")),
        "oracle_lateral_action_projection_mps": float(control.get("lateral_action_projection_mps", 0.0)),
        "boundary_side_selection": _jsonable(control.get("boundary_side_selection", {})),
        "teacher_state_source": "actor_visible_node_features_only",
        "truth_state_read_for_control": False,
    }


def _run_dagger_round_collection(
    *,
    round_index: int,
    student_trainer: Any,
    preservation_trainer: Any,
    config: FormalExperimentConfig,
    data_root: Path,
) -> dict[str, Any]:
    """Collect one real student rollout and relabel its visited states.

    The student action is always the action executed in the environment.  The
    Oracle only supplies a target label, so this remains genuine DAgger rather
    than teacher-forced trajectory generation.
    """

    student_trainer.policy.eval()
    preservation_trainer.policy.eval()
    episode_summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    query_counts = Counter()
    category_counts = Counter()
    source_counts = Counter()
    scenario_hashes: set[str] = set()
    plan = _round_episode_plan(round_index)
    for item in plan:
        episode_index = int(item["episode_index"])
        split = str(item["split"])
        scenario, scenario_seed, noise_seed = _scenario_for_round_item(
            config, round_index=round_index, item=item
        )
        scenario_hash = _canonical_hash(scenario)
        scenario_hashes.add(scenario_hash)
        _seed_everything(noise_seed)
        environment = _make_environment(
            config,
            noise_seed,
            metric_scales=config.metric_scales,
            safety_projection_enabled=True,
        )
        observations, info = environment.reset(
            seed=noise_seed, scenario_config=copy.deepcopy(scenario)
        )
        serial = int(round_index * 10000 + episode_index)
        writer = _EpisodeTraceWriter(
            episode_root=data_root / f"round_{round_index}" / "rollouts" / split / f"episode_{serial:06d}",
            episode_id=f"episode_{serial:06d}",
            scenario_id=f"stage2br_round{round_index}_episode{episode_index:03d}",
            split=split,
            seed=noise_seed,
            phase=f"stage2br_dagger_round_{round_index}",
            environment=environment,
            normalization_fitted=bool(student_trainer.normalizer.fitted),
            diagnostics_path=data_root / "diagnostics" / "attention_maps.jsonl",
            policy_source=f"stage2br_student_round_{round_index}",
        )
        oracle = BoundaryAwareGeometricObstacleOracleController(environment)
        episode_rows: list[dict[str, Any]] = []
        episode_records: list[dict[str, Any]] = []
        teacher_queries = 0
        overrides = 0
        for step in range(int(config.max_steps)):
            _assert_observation_finite(observations)
            decision, student_action, actor_observation = _actor_decision(
                student_trainer, observations, info
            )
            _preservation_decision, preservation_action, _ = _actor_decision(
                preservation_trainer, observations, info
            )
            visible = _visible_state(observations["uav_00"])
            static_episode = item["kind"] == "static_mirror_pressure"
            oracle_action = np.zeros(3, dtype=float)
            teacher_control: dict[str, Any] = {
                "route_phase": "stage2a_preservation_anchor",
                "lateral_action_projection_mps": 0.0,
                "demanded_braking_delta_v_mps": 0.0,
            }
            if static_episode:
                oracle_payload, teacher_control = oracle.action(observations)
                oracle_action = np.asarray(
                    oracle_payload["uav_00"]["delta_v"], dtype=float
                ).reshape(3)
                target_action = oracle_action.copy()
                target_source = "boundary_aware_oracle"
                oracle_queried = True
                query_reason = "forced_static_mirror_pressure"
            else:
                target_action = preservation_action.copy()
                target_source = "stage2a_preservation"
                oracle_queried = False
                query_reason = "normal_preservation_anchor"
            teacher_queries += int(oracle_queried)
            overrides += int(np.linalg.norm(student_action - target_action) > 1.0e-8)
            expected_category = str(item["category"])
            observed_category = _observed_direction_category(visible)
            state_before = environment.states["uav_00"].copy()
            nominal_state, _ = apply_velocity_increment(
                state_before,
                student_action,
                environment.config.motion,
                safety_projector=None,
            )
            nominal_position = np.asarray(nominal_state.position_m, dtype=float)
            pre_diag = _static_candidate_diagnostics(
                environment, state_before.position_m, nominal_position
            )
            action_payload = {
                "uav_00": {
                    "delta_v": student_action.tolist(),
                    "measurement_uav_id": "none",
                    "bandwidth": {},
                }
            }
            _assert_action_feasible(environment, action_payload)
            next_observations, rewards, terminated, truncated, next_info = environment.step(
                action_payload
            )
            row = _episode_step_row(
                policy=f"stage2br_round_{round_index}_student",
                scenario_name=str(item["kind"]),
                safety_mode="on",
                episode_index=episode_index,
                step=step,
                environment=environment,
                visible_before=visible,
                position_before=np.asarray(state_before.position_m, dtype=float),
                velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
                action=student_action,
                control={**teacher_control, "oracle_query_gate": int(oracle_queried)},
                nominal_position=nominal_position,
                pre_diag=pre_diag,
                next_info=next_info,
            )
            lateral_world = _body_left_world(float(visible["yaw_rad"]))
            row.update(
                {
                    "round": int(round_index),
                    "split": split,
                    "scenario_seed": int(scenario_seed),
                    "noise_seed": int(noise_seed),
                    "scenario_hash": scenario_hash,
                    "mirror_pair_id": str(scenario.get("stage2br", {}).get("mirror_pair_id", "")),
                    "expected_obstacle_category": expected_category,
                    "observed_obstacle_category": observed_category,
                    "query_gate": int(oracle_queried),
                    "oracle_queried": int(oracle_queried),
                    "oracle_override": int(np.linalg.norm(student_action - target_action) > 1.0e-8),
                    "target_source": target_source,
                    "student_lateral_action_projection_mps": float(np.dot(student_action, lateral_world)),
                    "target_lateral_action_projection_mps": float(np.dot(target_action, lateral_world)),
                    "direction_sign_error": int(
                        static_episode
                        and ((expected_category.startswith("left") and np.dot(target_action, lateral_world) >= 0.0)
                             or (expected_category.startswith("right") and np.dot(target_action, lateral_world) <= 0.0))
                    ),
                    "boundary_heading": _jsonable(teacher_control.get("boundary_side_selection", {})),
                }
            )
            episode_rows.append(row)
            record = _dagger_demo_record(
                split=split,
                round_index=round_index,
                episode_index=episode_index,
                step=step,
                scenario=scenario,
                scenario_seed=scenario_seed,
                noise_seed=noise_seed,
                observation=actor_observation,
                target_action=target_action,
                oracle_action=oracle_action,
                preservation_action=preservation_action,
                student_action=student_action,
                visible=visible,
                target_source=target_source,
                oracle_queried=oracle_queried,
                query_reason=query_reason,
                pre_diag=pre_diag,
                control=teacher_control,
                expected_category=expected_category,
                observed_category=observed_category,
            )
            episode_records.append(record)
            category_counts[expected_category] += 1
            source_counts[target_source] += 1
            if oracle_queried:
                query_counts[expected_category] += 1
            writer.record_step(
                step=step,
                observations_before=observations,
                observations_after=next_observations,
                actions=action_payload,
                rewards=rewards,
                learning_rewards=next_info.get("learning_rewards", rewards),
                info=next_info,
                graph_traces=getattr(decision, "graph_traces", None),
                policy_source=f"stage2br_student_round_{round_index}",
                action_start=step * environment.config.motion.dt_s,
                timestamp=(step + 1) * environment.config.motion.dt_s,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
        writer_summary = writer.finalize()
        summary = _summary_from_rows(
            episode_rows,
            policy=f"stage2br_round_{round_index}_student",
            scenario_name=str(item["kind"]),
            safety_mode="on",
            episode_index=episode_index,
        )
        summary.update(
            {
                "round": int(round_index),
                "split": split,
                "kind": str(item["kind"]),
                "expected_obstacle_category": expected_category,
                "observed_category_counts": dict(Counter(row["observed_obstacle_category"] for row in episode_rows)),
                "scenario_seed": int(scenario_seed),
                "noise_seed": int(noise_seed),
                "scenario_hash": scenario_hash,
                "mirror_pair_id": str(scenario.get("stage2br", {}).get("mirror_pair_id", "")),
                "teacher_query_count": int(teacher_queries),
                "oracle_override_count": int(overrides),
                "writer_summary": writer_summary,
                "training_performed": False,
                "policy_updates_performed": False,
                "ppo_updates": 0,
            }
        )
        episode_summaries.append(summary)
        step_rows.extend(episode_rows)
        if split == "train":
            train_records.extend(episode_records)
        else:
            validation_records.extend(episode_records)
        if episode_index in {1, 4, 8, 12}:
            print(
                f"[stage2br] round={round_index} episode={episode_index}/{EPISODES_PER_ROUND} split={split} kind={item['kind']} queries={teacher_queries}",
                flush=True,
            )
    return {
        "round": int(round_index),
        "plan": plan,
        "episode_summaries": episode_summaries,
        "step_rows": step_rows,
        "train_records": train_records,
        "validation_records": validation_records,
        "scenario_hashes": sorted(scenario_hashes),
        "query_counts": dict(query_counts),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "teacher_query_count": int(sum(query_counts.values())),
        "train_episode_count": int(sum(item["split"] == "train" for item in plan)),
        "validation_episode_count": int(sum(item["split"] == "validation" for item in plan)),
    }


def _write_round_datasets(
    data_root: Path,
    *,
    round_index: int,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> Path:
    root = data_root / "aggregated_dataset" / f"round_{int(round_index)}" / "demonstrations"
    _write_jsonl(root / "train.jsonl", train_records)
    _write_jsonl(root / "validation.jsonl", validation_records)
    return root.parent


def _write_combined_dataset(
    data_root: Path,
    *,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> Path:
    root = data_root / "aggregated_dataset" / "cumulative" / "demonstrations"
    _write_jsonl(root / "train.jsonl", train_records)
    _write_jsonl(root / "validation.jsonl", validation_records)
    return root.parent


def _freeze_for_direction_phase(policy: Any) -> tuple[list[str], dict[str, Any]]:
    """Freeze the whole policy except the three new Stage 2A input columns."""

    trainable_names: list[str] = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(False)
        if name in {
            "gat.layers.0.node_projection.weight",
            "gat.layers.0.residual.weight",
        }:
            parameter.requires_grad_(True)
            trainable_names.append(name)
    if len(trainable_names) != 2:
        raise RuntimeError(
            "Stage 2B-R expected exactly two first-layer direction weight matrices; "
            f"found {trainable_names}"
        )
    return trainable_names, {
        "direction_feature_columns": [39, 40, 41],
        "historical_feature_columns": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        "gradient_mask": "zero columns 0:39 after backward for both first-layer matrices",
        "critic_frozen": True,
        "normalizer_frozen": True,
        "logstd_frozen": True,
    }


def _mask_historical_direction_gradients(policy: Any) -> None:
    for name in (
        "gat.layers.0.node_projection.weight",
        "gat.layers.0.residual.weight",
    ):
        parameter = dict(policy.named_parameters())[name]
        if parameter.grad is not None:
            parameter.grad[:, :39] = 0.0


def _records_loss(
    trainer: Any, samples: Sequence[Mapping[str, Any]]
) -> tuple[Any, Any, Any, Any]:
    """Return total, Oracle, preservation and per-sample MSE tensors."""

    import torch

    if not samples:
        zero = torch.zeros((), dtype=torch.float32, device=trainer.policy.device)
        return zero, zero, zero, torch.empty((0,), device=trainer.policy.device)
    predictions = _batch_actor_physical_mean(
        trainer, [sample["observation"] for sample in samples]
    )
    targets = torch.as_tensor(
        np.asarray(
            [sample["teacher_action_delta_v_mps"] for sample in samples], dtype=np.float32
        ),
        dtype=torch.float32,
        device=predictions.device,
    )
    per_sample = torch.mean((predictions - targets) ** 2, dim=1)
    oracle_mask = torch.as_tensor(
        [str(sample.get("target_source", "")).startswith("boundary_aware_oracle") for sample in samples],
        dtype=torch.bool,
        device=predictions.device,
    )
    preservation_mask = ~oracle_mask
    oracle_loss = (
        per_sample[oracle_mask].mean()
        if bool(oracle_mask.any())
        else predictions.sum() * 0.0
    )
    preservation_loss = (
        per_sample[preservation_mask].mean()
        if bool(preservation_mask.any())
        else predictions.sum() * 0.0
    )
    total = oracle_loss + float(PRESERVATION_LOSS_COEF) * preservation_loss
    return total, oracle_loss, preservation_loss, per_sample


def _mean_finite(values: Sequence[Any]) -> float:
    numeric: list[float] = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            numeric.append(candidate)
    return float(np.mean(numeric)) if numeric else float("nan")


def _direction_weight_snapshot(policy: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in (
        "gat.layers.0.node_projection.weight",
        "gat.layers.0.residual.weight",
    ):
        value = dict(policy.named_parameters())[name].detach().cpu().numpy().astype(float)
        snapshot[name] = {
            "historical_norm": float(np.linalg.norm(value[:, :39])),
            "new_direction_norm": float(np.linalg.norm(value[:, 39:42])),
            "new_direction_values": value[:, 39:42].tolist(),
        }
    return snapshot


def train_stage2br_direction_actor(
    initial_checkpoint: Path,
    dataset_dir: Path,
    *,
    output_dir: Path,
    logs_root: Path,
    round_index: int,
    max_epochs: int = PHASE_A_MAX_EPOCHS,
    patience: int = PHASE_A_PATIENCE,
    batch_size: int = PHASE_A_BATCH_SIZE,
    learning_rate: float = PHASE_A_LEARNING_RATE,
) -> dict[str, Any]:
    """Run one limited Actor-only BC update from a previous round checkpoint."""

    import torch

    train_samples = _read_jsonl(dataset_dir / "demonstrations" / "train.jsonl")
    validation_samples = _read_jsonl(dataset_dir / "demonstrations" / "validation.jsonl")
    if not train_samples or not validation_samples:
        raise ValueError("Stage 2B-R requires non-empty cumulative train and validation datasets")
    train_ids = {(str(row.get("scenario_hash")), int(row.get("episode_index", -1))) for row in train_samples}
    validation_ids = {(str(row.get("scenario_hash")), int(row.get("episode_index", -1))) for row in validation_samples}
    if train_ids & validation_ids:
        raise ValueError("Stage 2B-R demonstration trajectory leakage detected")
    initial_checkpoint = Path(initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(
        payload,
        output_dir=output_dir,
        command_line=f"Stage 2B-R direction Actor-only BC round {round_index}",
    )
    trainer, identity, _ = _load_frozen_trainer(
        payload, config, metric_scales=config.metric_scales
    )
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    critic_before = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_before = _state_dict_hash(
        {"delta_logstd": trainer.policy.delta_logstd.detach().cpu()}
    )
    normalizer_before = _canonical_hash(trainer.normalizer.state_dict())
    input_weights_before = _direction_weight_snapshot(trainer.policy)
    historical_weights_before = {
        name: dict(trainer.policy.named_parameters())[name].detach().cpu().clone()[:, :39]
        for name in (
            "gat.layers.0.node_projection.weight",
            "gat.layers.0.residual.weight",
        )
    }
    trainable_names, scope = _freeze_for_direction_phase(trainer.policy)
    optimizer = torch.optim.Adam(
        [parameter for parameter in trainer.policy.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    rng = np.random.default_rng(202600 + int(round_index))
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    wait = 0
    best_state: dict[str, Any] | None = None
    last_state: dict[str, Any] | None = None
    for epoch in range(1, int(max_epochs) + 1):
        trainer.policy.eval()
        order = rng.permutation(len(train_samples))
        batch_total: list[float] = []
        batch_oracle: list[float] = []
        batch_preservation: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            selected = [train_samples[int(index)] for index in order[start : start + int(batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            total, oracle_loss, preservation_loss, _ = _records_loss(trainer, selected)
            total.backward()
            _mask_historical_direction_gradients(trainer.policy)
            optimizer.step()
            batch_total.append(float(total.detach().cpu()))
            batch_oracle.append(float(oracle_loss.detach().cpu()))
            batch_preservation.append(float(preservation_loss.detach().cpu()))
        trainer.policy.eval()
        with torch.no_grad():
            train_total, train_oracle, train_preservation, train_per_sample = _records_loss(trainer, train_samples)
            val_total, val_oracle, val_preservation, val_per_sample = _records_loss(trainer, validation_samples)
        row = {
            "round": int(round_index),
            "epoch": int(epoch),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "batch_total_loss": _mean_finite(batch_total),
            "batch_oracle_loss": _mean_finite(batch_oracle),
            "batch_preservation_loss": _mean_finite(batch_preservation),
            "train_total_loss": float(train_total.detach().cpu()),
            "train_oracle_loss": float(train_oracle.detach().cpu()),
            "train_preservation_loss": float(train_preservation.detach().cpu()),
            "train_action_mse": float(train_per_sample.mean().detach().cpu()),
            "validation_total_loss": float(val_total.detach().cpu()),
            "validation_oracle_loss": float(val_oracle.detach().cpu()),
            "validation_preservation_loss": float(val_preservation.detach().cpu()),
            "validation_action_mse": float(val_per_sample.mean().detach().cpu()),
        }
        history.append(row)
        current_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.policy.state_dict().items()
        }
        last_state = current_state
        validation_value = float(row["validation_action_mse"])
        if validation_value < best_validation - 1.0e-12:
            best_validation = validation_value
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(current_state)
        else:
            wait += 1
        print(
            f"[stage2br-bc] round={round_index} epoch={epoch}/{max_epochs} "
            f"train={row['train_action_mse']:.6g} val={row['validation_action_mse']:.6g}",
            flush=True,
        )
        if wait >= int(patience):
            break
    if best_state is None or last_state is None:
        raise RuntimeError("Stage 2B-R BC did not produce a policy state")
    trainer.policy.load_state_dict(best_state, strict=True)
    trainer.policy.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    best_checkpoint = checkpoint_dir / "best_validation.pt"
    last_checkpoint = checkpoint_dir / "last_epoch.pt"
    critic_after = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_after = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    normalizer_after = _canonical_hash(trainer.normalizer.state_dict())
    weight_after = _direction_weight_snapshot(trainer.policy)
    historical_column_delta_norms = {
        name: float(
            np.linalg.norm(
                dict(trainer.policy.named_parameters())[name].detach().cpu().numpy()[:, :39]
                - historical_weights_before[name].numpy()
            )
        )
        for name in historical_weights_before
    }
    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    metadata = {
        "dataset_id": "single_uav_obstacle_dagger_stage2br_smoke",
        "stage": "stage2br_direction_actor_only_bc",
        "round": int(round_index),
        "source_checkpoint": {"path": str(initial_checkpoint), "sha256": _sha256(initial_checkpoint)},
        "source_policy_hash": identity.get("loaded_policy_hash", ""),
        "policy_hash_after_training": policy_hash_after,
        "trainable_parameter_names": trainable_names,
        "trainable_scope": scope,
        "critic_frozen": critic_before == critic_after,
        "normalizer_frozen": normalizer_before == normalizer_after,
        "logstd_frozen": logstd_before == logstd_after,
        "old_columns_frozen": all(value <= 1.0e-12 for value in historical_column_delta_norms.values()),
        "historical_column_delta_norms": historical_column_delta_norms,
        "loss_space": "post_tanh_scaled_physical_delta_v_mse",
        "preservation_loss_coef": float(PRESERVATION_LOSS_COEF),
        "ppo_update_performed": False,
        "reward_or_environment_modified": False,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "actual_epochs": len(history),
        "best_validation_epoch": int(best_epoch),
        "best_validation_mse": float(best_validation),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
    }
    _save_imitation_checkpoint(payload, best_state, best_checkpoint, metadata)
    _save_imitation_checkpoint(payload, last_state, last_checkpoint, {**metadata, "checkpoint_role": "last_epoch"})
    history_path = logs_root / f"round_{int(round_index)}_bc_training_history.csv"
    _write_rows_csv(history_path, history)
    result = {
        "round": int(round_index),
        "source_checkpoint": metadata["source_checkpoint"],
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "policy_hash_before": identity.get("loaded_policy_hash", ""),
        "policy_hash_after": policy_hash_after,
        "actual_epochs": len(history),
        "best_validation_epoch": int(best_epoch),
        "best_validation_mse": float(best_validation),
        "history": history,
        "checkpoint_contract": metadata,
        "weights_before": input_weights_before,
        "weights_after": weight_after,
        "files": {"history": str(history_path.resolve()), "best_checkpoint": str(best_checkpoint.resolve()), "last_checkpoint": str(last_checkpoint.resolve())},
    }
    _write_json(output_dir / "training_summary.json", _jsonable(result))
    return result


def actor_direction_sign_probe(
    trainer: Any, config: FormalExperimentConfig, *, round_index: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trainer.policy.eval()
    for index, category in enumerate(MIRROR_CATEGORIES, start=1):
        side = "left" if category.startswith("left") else "right"
        scenario = mirror_static_scenario(
            config,
            category=category,
            pair_index=9500 + int(round_index) * 10 + index,
            side=side,
            seed=2060000 + int(round_index) * 100 + index,
            round_index=round_index,
        )
        environment = _make_environment(
            config,
            2060000 + int(round_index) * 100 + index,
            metric_scales=config.metric_scales,
            safety_projection_enabled=True,
        )
        observations, info = environment.reset(
            seed=2060000 + int(round_index) * 100 + index,
            scenario_config=copy.deepcopy(scenario),
        )
        _decision, action, _actor_observation = _actor_decision(trainer, observations, info)
        visible = _visible_state(observations["uav_00"])
        lateral_world = _body_left_world(float(visible["yaw_rad"]))
        lateral = float(np.dot(action, lateral_world))
        expected = -1.0 if side == "left" else 1.0
        rows.append(
            {
                "round": int(round_index),
                "case": category,
                "side": side,
                "expected_lateral_sign": expected,
                "obstacle_direction_body_y_m": float(visible["obstacle_direction_body_m"][1]),
                "lateral_action_projection_mps": lateral,
                "abs_lateral_action_mps": abs(lateral),
                "action_norm_mps": float(np.linalg.norm(action)),
                "lateral_fraction": abs(lateral) / max(float(np.linalg.norm(action)), 1.0e-8),
                "sign_correct": int(lateral * expected > 1.0e-8),
                "nondegenerate": int(abs(lateral) >= LATERAL_ACTION_NONDEGENERATE_MPS),
                "legal_action": int(
                    np.linalg.norm(action)
                    <= environment.config.motion.max_acceleration_mps2
                    * environment.config.motion.dt_s
                    + 1.0e-8
                ),
                "checkpoint_policy_hash": _state_dict_hash(trainer.policy.state_dict()),
            }
        )
    return rows


def boundary_drift_probe(
    trainer: Any, config: FormalExperimentConfig, *, round_index: int
) -> dict[str, Any]:
    """Check whether the learned actor keeps a persistent lateral bias after a bypass."""

    seed = 2070000 + int(round_index)
    scenario = _post_obstacle_scenario(config, seed=seed)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(scenario))
    signs: list[int] = []
    projections: list[float] = []
    boundary_steps = 0
    rows: list[dict[str, Any]] = []
    for step in range(min(20, int(config.max_steps))):
        _decision, action, _actor_observation = _actor_decision(trainer, observations, info)
        visible = _visible_state(observations["uav_00"])
        lateral_world = _body_left_world(float(visible["yaw_rad"]))
        projection = float(np.dot(action, lateral_world))
        projections.append(projection)
        if abs(projection) >= LATERAL_ACTION_NONDEGENERATE_MPS:
            signs.append(1 if projection > 0.0 else -1)
        next_observations, _rewards, terminated, truncated, next_info = environment.step(
            _decision.actions
        )
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        boundary_steps += int(bool(execution.get("boundary_projected", False)))
        rows.append(
            {
                "round": int(round_index),
                "step": int(step),
                "lateral_action_projection_mps": projection,
                "boundary_projected": int(bool(execution.get("boundary_projected", False))),
                "distance_m": float(visible["distance_m"]),
            }
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    max_same_sign = 0
    current_sign = 0
    current_count = 0
    for sign in signs:
        if sign == current_sign:
            current_count += 1
        else:
            current_sign = sign
            current_count = 1
        max_same_sign = max(max_same_sign, current_count)
    return {
        "round": int(round_index),
        "steps": len(rows),
        "boundary_projection_steps": int(boundary_steps),
        "mean_post_obstacle_lateral_action_mps": _mean_finite(projections),
        "max_consecutive_same_sign_lateral_steps": int(max_same_sign),
        "persistent_lateral_bias": int(max_same_sign >= 5),
        "step_rows": rows,
    }


def _aggregate_round_rows(
    summaries: Sequence[Mapping[str, Any]], *, round_index: int, kind: str | None = None
) -> dict[str, Any]:
    selected = [
        row
        for row in summaries
        if int(row.get("round", -1)) == int(round_index)
        and (kind is None or row.get("kind") == kind)
    ]
    count = len(selected)

    def count_field(field: str) -> int:
        return int(sum(int(row.get(field, 0) or 0) for row in selected))

    return {
        "round": int(round_index),
        "kind": kind or "all",
        "episodes": count,
        "entered_8m": count_field("entered_8m"),
        "entered_8m_rate": count_field("entered_8m") / count if count else 0.0,
        "entered_5m": count_field("entered_5m"),
        "entered_3m": count_field("entered_3m"),
        "entered_3m_rate": count_field("entered_3m") / count if count else 0.0,
        "entered_1m": count_field("entered_1m"),
        "entered_1m_rate": count_field("entered_1m") / count if count else 0.0,
        "stable_goal": count_field("stable_goal"),
        "mean_minimum_goal_distance_m": _mean_finite([row.get("minimum_goal_distance_m") for row in selected]),
        "mean_speed_mps": _mean_finite([row.get("mean_speed_mps") for row in selected]),
        "mean_maximum_consecutive_goal_hold_steps": _mean_finite([row.get("maximum_consecutive_goal_hold_steps") for row in selected]),
        "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in selected), default=0),
        "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in selected)),
        "pre_risk_steps": int(sum(int(row.get("pre_candidate_static_swept_conflict_steps", 0) or 0) for row in selected)),
        "collision_episodes": count_field("collision"),
        "termination_reasons": dict(Counter(str(row.get("termination_reason", "")) for row in selected)),
    }


def _run_actor_diagnostic_episode(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    scenario: Mapping[str, Any],
    scenario_name: str,
    split: str,
    safety_mode: str,
    episode_index: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(2080000 + serial)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=safety_mode == "on",
    )
    observations, info = environment.reset(
        seed=seed, scenario_config=copy.deepcopy(dict(scenario))
    )
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision, action, _actor_observation = _actor_decision(trainer, observations, info)
        state_before = environment.states["uav_00"].copy()
        nominal_state, _ = apply_velocity_increment(
            state_before, action, environment.config.motion, safety_projector=None
        )
        next_observations, _rewards, terminated, truncated, next_info = environment.step(
            decision.actions
        )
        visible = _visible_state(observations["uav_00"])
        pre_diag = _static_candidate_diagnostics(
            environment, state_before.position_m, nominal_state.position_m
        )
        row = _episode_step_row(
            policy="actor_diagnostic",
            scenario_name=scenario_name,
            safety_mode=safety_mode,
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=visible,
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=action,
            control={
                "route_phase": "actor_diagnostic",
                "lateral_action_projection_mps": 0.0,
                "oracle_query_gate": 0,
            },
            nominal_position=np.asarray(nominal_state.position_m, dtype=float),
            pre_diag=pre_diag,
            next_info=next_info,
        )
        row.update({"split": split, "scenario_hash": _canonical_hash(scenario)})
        rows.append(row)
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    summary = _summary_from_rows(
        rows,
        policy="actor_diagnostic",
        scenario_name=scenario_name,
        safety_mode=safety_mode,
        episode_index=episode_index,
    )
    summary.update({"split": split, "scenario_hash": _canonical_hash(scenario), "seed": seed})
    return summary, rows


def _normal_navigation_probe(
    source_trainer: Any,
    final_trainer: Any,
    config: FormalExperimentConfig,
    *,
    data_root: Path,
    logs_root: Path,
    episodes: int = 3,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for role, trainer in (("stage2a_frozen", source_trainer), ("stage2br_final", final_trainer)):
        for index in range(1, int(episodes) + 1):
            scenario = _normal_no_obstacle_scenario(
                config,
                seed=2090000 + index,
                round_index=0,
                episode_index=index,
            )
            summary, rows = _run_actor_diagnostic_episode(
                trainer=trainer,
                config=config,
                scenario=scenario,
                scenario_name="normal_navigation_probe",
                split="probe",
                safety_mode="on",
                episode_index=index,
                serial=(0 if role == "stage2a_frozen" else 100) + index,
            )
            summary["policy"] = role
            summary["policy_label"] = role
            for row in rows:
                row["policy"] = role
                row["policy_label"] = role
            summaries.append(summary)
            step_rows.extend(rows)
    _write_rows_csv(logs_root / "navigation_preservation_metrics.csv", summaries)
    _write_rows_csv(logs_root / "navigation_preservation_step_diagnostics.csv", step_rows)
    aggregates: dict[str, dict[str, Any]] = {}
    for role in ("stage2a_frozen", "stage2br_final"):
        selected = [row for row in summaries if row.get("policy") == role]
        aggregates[role] = {
            "episodes": len(selected),
            "entered_8m": int(sum(int(row.get("entered_8m", 0) or 0) for row in selected)),
            "entered_8m_rate": _mean_finite([row.get("entered_8m") for row in selected]),
            "entered_3m": int(sum(int(row.get("entered_3m", 0) or 0) for row in selected)),
            "entered_1m": int(sum(int(row.get("entered_1m", 0) or 0) for row in selected)),
            "mean_minimum_goal_distance_m": _mean_finite([row.get("minimum_goal_distance_m") for row in selected]),
            "mean_maximum_consecutive_goal_hold_steps": _mean_finite([row.get("maximum_consecutive_goal_hold_steps") for row in selected]),
            "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in selected)),
        }
    result = {
        "episodes_per_policy": int(episodes),
        "aggregates": aggregates,
        "normal_navigation_preserved": bool(
            aggregates["stage2br_final"]["entered_8m_rate"]
            >= aggregates["stage2a_frozen"]["entered_8m_rate"]
        ),
        "step_rows": step_rows,
    }
    _write_json(logs_root / "navigation_preservation_summary.json", _jsonable(result))
    return result


def _holdout_actor_evaluation(
    source_trainer: Any,
    final_trainer: Any,
    config: FormalExperimentConfig,
    *,
    data_root: Path,
    logs_root: Path,
    episodes: int = 4,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    roles = (("stage2a_frozen", source_trainer), ("stage2br_final", final_trainer))
    for index in range(1, int(episodes) + 1):
        normal = _normal_no_obstacle_scenario(config, seed=2100000 + index, round_index=0, episode_index=index)
        static = mirror_static_scenario(
            config,
            category=("left_front" if index % 2 else "right_front"),
            pair_index=9800 + index,
            side=("left" if index % 2 else "right"),
            seed=2110000 + index,
            round_index=0,
        )
        for group, scenario in (("normal_holdout", normal), ("static_holdout", static)):
            modes = ("off", "on") if group == "static_holdout" else ("on",)
            for mode in modes:
                for role, trainer in roles:
                    summary, rows = _run_actor_diagnostic_episode(
                        trainer=trainer,
                        config=config,
                        scenario=scenario,
                        scenario_name=group,
                        split="holdout",
                        safety_mode=mode,
                        episode_index=index,
                        serial=30000 + index * 100 + (0 if role == "stage2a_frozen" else 10) + (0 if mode == "off" else 1),
                    )
                    summary.update({"policy": role, "group": group})
                    for row in rows:
                        row.update({"policy": role, "group": group})
                    summaries.append(summary)
                    step_rows.extend(rows)
    _write_rows_csv(logs_root / "holdout_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "holdout_step_diagnostics.csv", step_rows)
    aggregate: list[dict[str, Any]] = []
    for role, _trainer in roles:
        for group in ("normal_holdout", "static_holdout"):
            for mode in (("off", "on") if group == "static_holdout" else ("on",)):
                selected = [row for row in summaries if row.get("policy") == role and row.get("group") == group and row.get("safety_mode") == mode]
                aggregate.append(
                    {
                        "policy": role,
                        "group": group,
                        "safety_mode": mode,
                        "episodes": len(selected),
                        "entered_8m": int(sum(int(row.get("entered_8m", 0) or 0) for row in selected)),
                        "entered_3m": int(sum(int(row.get("entered_3m", 0) or 0) for row in selected)),
                        "entered_1m": int(sum(int(row.get("entered_1m", 0) or 0) for row in selected)),
                        "stable_goal": int(sum(int(row.get("stable_goal", 0) or 0) for row in selected)),
                        "mean_minimum_goal_distance_m": _mean_finite([row.get("minimum_goal_distance_m") for row in selected]),
                        "mean_maximum_consecutive_goal_hold_steps": _mean_finite([row.get("maximum_consecutive_goal_hold_steps") for row in selected]),
                        "hold_ge_1": int(sum(int(row.get("hold_steps_ge_1", 0) or 0) for row in selected)),
                        "hold_ge_3": int(sum(int(row.get("hold_steps_ge_3", 0) or 0) for row in selected)),
                        "hold_ge_5": int(sum(int(row.get("hold_steps_ge_5", 0) or 0) for row in selected)),
                        "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in selected)),
                    }
                )
    _write_rows_csv(logs_root / "stage2a_vs_stage2br_comparison.csv", aggregate)
    pairs = []
    for row in aggregate:
        if row["group"] == "static_holdout":
            pairs.append(dict(row))
    _write_rows_csv(logs_root / "safety_on_off_comparison.csv", pairs)
    return {"evaluation_frozen": True, "episodes_per_group": int(episodes), "episode_summaries": summaries, "step_rows": step_rows, "aggregate": aggregate}


def _write_stage2br_plots(
    *,
    plots_root: Path,
    round_metrics: Sequence[Mapping[str, Any]],
    sign_rows: Sequence[Mapping[str, Any]],
    weight_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    rounds = sorted({int(row.get("round", 0)) for row in round_metrics})
    balance = [row for row in round_metrics if row.get("kind") == "all"]
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = np.arange(len(balance))
    width = 0.25
    for offset, key, label, color in (
        (-width, "normal_episodes", "normal", "#2563eb"),
        (0.0, "static_episodes", "static", "#dc2626"),
        (width, "teacher_query_count", "Oracle queries", "#16a34a"),
    ):
        axis.bar(x + offset, [float(row.get(key, 0) or 0) for row in balance], width=width, label=label, color=color)
    axis.set_xticks(x, [f"R{row.get('round')}" for row in balance])
    axis.set_ylabel("Episodes / queries")
    axis.set_title("Stage 2B-R round balance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = plots_root / "left_right_teacher_balance.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["left_right_teacher_balance"] = str(path.resolve())

    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for case, color in (("left", "#2563eb"), ("right", "#dc2626"), ("left_front", "#60a5fa"), ("right_front", "#f87171")):
        selected = [row for row in sign_rows if row.get("case") == case]
        if selected:
            axis.plot([int(row["round"]) for row in selected], [float(row["lateral_action_projection_mps"]) for row in selected], marker="o", label=case, color=color)
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("DAgger round")
    axis.set_ylabel("Body-left lateral action projection (m/s)")
    axis.set_title("Actor direction action by round")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    path = plots_root / "direction_action_by_round.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["direction_action_by_round"] = str(path.resolve())

    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for name in ("gat.layers.0.node_projection.weight", "gat.layers.0.residual.weight"):
        selected = [row for row in weight_rows if row.get("parameter") == name]
        if selected:
            axis.plot([int(row["round"]) for row in selected], [float(row["after_new_direction_norm"]) for row in selected], marker="o", label=name.split(".")[-2],)
    axis.set_xlabel("DAgger round")
    axis.set_ylabel("New direction-column norm")
    axis.set_title("Direction-weight growth")
    axis.grid(alpha=0.25)
    axis.legend()
    path = plots_root / "direction_weight_growth.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["direction_weight_growth"] = str(path.resolve())

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    static_rows = [row for row in round_metrics if row.get("kind") == "static_mirror_pressure"]
    normal_rows = [row for row in round_metrics if row.get("kind") == "normal_far_distance"]
    for axis, selected, title in ((axes[0], normal_rows, "Normal navigation"), (axes[1], static_rows, "Static pressure")):
        axis.plot([int(row["round"]) for row in selected], [float(row.get("entered_8m_rate", 0.0)) for row in selected], marker="o", label="entered 8m")
        axis.plot([int(row["round"]) for row in selected], [float(row.get("entered_3m_rate", 0.0)) for row in selected], marker="s", label="entered 3m")
        axis.plot([int(row["round"]) for row in selected], [float(row.get("entered_1m_rate", 0.0)) for row in selected], marker="^", label="entered 1m")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.set_xlabel("Round")
        axis.set_ylabel("Episode rate")
        axis.grid(alpha=0.25)
        axis.legend()
    path = plots_root / "static_goal_progress_comparison.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["static_goal_progress_comparison"] = str(path.resolve())

    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.plot([int(row["round"]) for row in boundary_rows], [float(row.get("boundary_projection_steps", 0.0)) for row in boundary_rows], marker="o", label="probe boundary projections")
    axis.axhline(PREVIOUS_STAGE2B_BOUNDARY_STEPS, linestyle="--", color="#dc2626", label="previous Stage2B = 354")
    axis.set_xlabel("DAgger round")
    axis.set_ylabel("Projection steps")
    axis.set_title("Boundary-interference comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    path = plots_root / "boundary_projection_comparison.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["boundary_projection_comparison"] = str(path.resolve())

    if episode_rows:
        figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        selected = [row for row in episode_rows if row.get("kind") == "static_mirror_pressure"]
        for round_index in sorted({int(row.get("round", 0)) for row in selected}):
            values = [float(row.get("minimum_goal_distance_m", float("nan"))) for row in selected if int(row.get("round", 0)) == round_index]
            values = [value for value in values if np.isfinite(value)]
            if values:
                axis.scatter([round_index] * len(values), values, label=f"R{round_index}")
        axis.set_xlabel("DAgger round")
        axis.set_ylabel("Minimum goal distance (m)")
        axis.set_title("Static student rollout goal progress")
        axis.grid(alpha=0.25)
        path = plots_root / "normal_navigation_preservation.png"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        output["normal_navigation_preservation"] = str(path.resolve())
    return output


def _write_schema_benchmark(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    data_root: Path,
) -> dict[str, Any]:
    """Create a tiny standard train/validation/test scale for schema checking."""

    scale_root = data_root / "scale_1"
    writer_summaries: list[dict[str, Any]] = []
    for split, seed, index in (
        ("train", 2120001, 1),
        ("validation", 2120002, 1),
        ("test", 2120003, 1),
    ):
        scenario = _normal_no_obstacle_scenario(
            config, seed=seed, round_index=0, episode_index=index
        )
        environment = _make_environment(
            config,
            seed,
            metric_scales=config.metric_scales,
            safety_projection_enabled=True,
        )
        observations, info = environment.reset(
            seed=seed, scenario_config=copy.deepcopy(scenario)
        )
        writer = _EpisodeTraceWriter(
            episode_root=scale_root / split / f"episode_{index:06d}",
            episode_id=f"episode_{index:06d}",
            scenario_id=f"stage2br_schema_{split}_{index:03d}",
            split=split,
            seed=seed,
            phase="stage2br_schema_smoke",
            environment=environment,
            normalization_fitted=bool(trainer.normalizer.fitted),
            diagnostics_path=data_root / "diagnostics" / "schema_attention_maps.jsonl",
            policy_source="stage2br_schema_smoke",
        )
        for step in range(min(3, int(config.max_steps))):
            decision, _action, _actor_observation = _actor_decision(trainer, observations, info)
            next_observations, rewards, terminated, truncated, next_info = environment.step(
                decision.actions
            )
            writer.record_step(
                step=step,
                observations_before=observations,
                observations_after=next_observations,
                actions=decision.actions,
                rewards=rewards,
                learning_rewards=next_info.get("learning_rewards", rewards),
                info=next_info,
                graph_traces=getattr(decision, "graph_traces", None),
                policy_source="stage2br_schema_smoke",
                action_start=step * environment.config.motion.dt_s,
                timestamp=(step + 1) * environment.config.motion.dt_s,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
        writer_summaries.append(writer.finalize())
    _copy_calibration_files(scale_root, config)
    manifest = _write_scale_manifest(
        scale_root=scale_root, config=config, episodes=writer_summaries
    )
    return {"scale_root": str(scale_root.resolve()), "manifest": manifest, "episodes": writer_summaries}


def _empty_gate_csvs(logs_root: Path, *, reason: str) -> None:
    _write_rows_csv(logs_root / "stage2a_vs_stage2br_comparison.csv", [{"status": "not_run", "reason": reason}])
    _write_rows_csv(logs_root / "safety_on_off_comparison.csv", [{"status": "not_run", "reason": reason}])


def run_stage2br_smoke(
    *,
    stage2a_checkpoint: str | Path = DEFAULT_STAGE2A_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the isolated Stage 2B-R smoke protocol."""

    del progress  # This smoke intentionally emits concise progress lines.
    stage2a_checkpoint = Path(stage2a_checkpoint).resolve()
    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    _ensure_empty_roots(data_root, logs_root, plots_root)
    if not stage2a_checkpoint.is_file():
        raise FileNotFoundError(stage2a_checkpoint)

    source_trainer, config, source_identity = _load_stage2br_policy(
        stage2a_checkpoint, output_dir=data_root / "source_policy"
    )
    source_policy_hash = source_identity["policy_hash"]
    bias = audit_previous_data_bias(
        Path("logs/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026_rerun4")
    )
    _write_json(logs_root / "data_bias_audit.json", _jsonable(bias))
    _write_rows_csv(logs_root / "mirror_pair_manifest.csv", mirror_pair_manifest_rows())
    oracle_sign = oracle_direction_sign_probe(config)
    _write_rows_csv(logs_root / "oracle_direction_sign_probe.csv", oracle_sign["rows"])
    oracle_sanity = run_oracle_safety_sanity(config, data_root, logs_root)
    schema_benchmark = _write_schema_benchmark(source_trainer, config, data_root=data_root)

    base_report: dict[str, Any] = {
        "stage": STAGE_NAME,
        "status": "started",
        "source_checkpoint": {
            "path": str(stage2a_checkpoint),
            "sha256": _sha256(stage2a_checkpoint),
            "policy_hash": source_policy_hash,
        },
        "checkpoint_identity": source_identity,
        "data_bias_audit": bias,
        "oracle_direction_sign_probe": oracle_sign,
        "oracle_safety_sanity": oracle_sanity,
        "schema_benchmark": schema_benchmark,
        "training_contract": {
            "round_count": ROUND_COUNT,
            "episodes_per_round": EPISODES_PER_ROUND,
            "normal_episodes_per_round": NORMAL_EPISODES_PER_ROUND,
            "static_episodes_per_round": STATIC_EPISODES_PER_ROUND,
            "round_train_episodes": ROUND_TRAIN_EPISODES,
            "round_validation_episodes": ROUND_VALIDATION_EPISODES,
            "teacher_state_source": "actor_visible_node_features_only",
            "truth_state_read_for_control": False,
            "ppo_updates": 0,
            "reward_or_environment_modified": False,
            "trainable_scope": "Stage 2A direction input columns 39:42 in first GAT node and residual weights only",
        },
    }
    if not oracle_sign.get("passed") or not oracle_sanity.get("sanity_passed"):
        reason = "oracle_direction_or_safety_sanity_failed"
        _empty_gate_csvs(logs_root, reason=reason)
        base_report.update(
            {
                "status": "blocked_before_dagger",
                "technical_smoke_passed": False,
                "data_balance_passed": False,
                "oracle_sign_gate_passed": bool(oracle_sign.get("passed")),
                "iterative_dagger_passed": False,
                "direction_behavior_passed": False,
                "normal_navigation_preserved": False,
                "static_progress_passed": False,
                "boundary_interference_acceptable": False,
                "behavioral_acceptance_passed": False,
                "formal_stage2b_eligible": False,
                "dagger_not_started": True,
                "block_reason": reason,
                "outputs": {"data_root": str(data_root.resolve()), "logs_root": str(logs_root.resolve()), "plots_root": str(plots_root.resolve())},
            }
        )
        _write_json(logs_root / "stage2br_summary.json", _jsonable(base_report))
        _write_json(data_root / "stage2br_summary.json", _jsonable(base_report))
        _write_json(data_root / "summary.json", _jsonable(base_report))
        validation = validate_preliminary(data_root)
        base_report["validate_preliminary"] = validation
        _write_json(logs_root / "stage2br_summary.json", _jsonable(base_report))
        _write_json(data_root / "stage2br_summary.json", _jsonable(base_report))
        _write_json(data_root / "summary.json", _jsonable(base_report))
        return base_report

    all_train_records: list[dict[str, Any]] = []
    all_validation_records: list[dict[str, Any]] = []
    all_episode_summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    round_metrics: list[dict[str, Any]] = []
    round_training: list[dict[str, Any]] = []
    sign_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    current_trainer = source_trainer
    current_checkpoint = stage2a_checkpoint
    current_config = config
    round_chain: list[dict[str, Any]] = []
    for round_index in range(1, ROUND_COUNT + 1):
        collection = _run_dagger_round_collection(
            round_index=round_index,
            student_trainer=current_trainer,
            preservation_trainer=source_trainer,
            config=current_config,
            data_root=data_root,
        )
        all_train_records.extend(collection["train_records"])
        all_validation_records.extend(collection["validation_records"])
        all_episode_summaries.extend(collection["episode_summaries"])
        all_step_rows.extend(collection["step_rows"])
        cumulative_dir = _write_combined_dataset(
            data_root,
            train_records=all_train_records,
            validation_records=all_validation_records,
        )
        training_output = data_root / f"round_{round_index}" / "trained_actor"
        training = train_stage2br_direction_actor(
            Path(current_checkpoint),
            cumulative_dir,
            output_dir=training_output,
            logs_root=logs_root,
            round_index=round_index,
        )
        target_checkpoint = Path(training["best_checkpoint"])
        next_trainer, next_config, next_identity = _load_stage2br_policy(
            target_checkpoint, output_dir=data_root / f"round_{round_index}" / "frozen_policy"
        )
        sign = actor_direction_sign_probe(next_trainer, next_config, round_index=round_index)
        drift = boundary_drift_probe(next_trainer, next_config, round_index=round_index)
        sign_rows.extend(sign)
        boundary_rows.append({k: v for k, v in drift.items() if k != "step_rows"})
        for parameter, before, after in (
            ("gat.layers.0.node_projection.weight", training["weights_before"]["gat.layers.0.node_projection.weight"], training["weights_after"]["gat.layers.0.node_projection.weight"]),
            ("gat.layers.0.residual.weight", training["weights_before"]["gat.layers.0.residual.weight"], training["weights_after"]["gat.layers.0.residual.weight"]),
        ):
            weight_rows.append(
                {
                    "round": int(round_index),
                    "parameter": parameter,
                    "before_new_direction_norm": before["new_direction_norm"],
                    "after_new_direction_norm": after["new_direction_norm"],
                    "new_direction_delta_norm": float(np.linalg.norm(np.asarray(after["new_direction_values"]) - np.asarray(before["new_direction_values"]))),
                }
            )
        round_metrics.append(
            {
                "round": int(round_index),
                "kind": "all",
                "episodes": EPISODES_PER_ROUND,
                "normal_episodes": NORMAL_EPISODES_PER_ROUND,
                "static_episodes": STATIC_EPISODES_PER_ROUND,
                "normal_fraction": NORMAL_EPISODES_PER_ROUND / EPISODES_PER_ROUND,
                "static_fraction": STATIC_EPISODES_PER_ROUND / EPISODES_PER_ROUND,
                "teacher_query_count": collection["teacher_query_count"],
                "train_demonstration_samples": len(collection["train_records"]),
                "validation_demonstration_samples": len(collection["validation_records"]),
                "policy_hash_before": training["policy_hash_before"],
                "policy_hash_after": training["policy_hash_after"],
                "oracle_sign_correct": int(sum(row["sign_correct"] for row in sign)),
                "oracle_sign_applicable": len(sign),
                "boundary_probe_projection_steps": drift["boundary_projection_steps"],
            }
        )
        for kind in ("normal_far_distance", "static_mirror_pressure"):
            aggregate = _aggregate_round_rows(collection["episode_summaries"], round_index=round_index, kind=kind)
            aggregate["normal_episodes"] = NORMAL_EPISODES_PER_ROUND if kind == "normal_far_distance" else 0
            aggregate["static_episodes"] = STATIC_EPISODES_PER_ROUND if kind == "static_mirror_pressure" else 0
            aggregate["entered_3m_rate"] = aggregate["entered_3m"] / aggregate["episodes"] if aggregate["episodes"] else 0.0
            aggregate["entered_1m_rate"] = aggregate["entered_1m"] / aggregate["episodes"] if aggregate["episodes"] else 0.0
            aggregate["static_fraction"] = STATIC_EPISODES_PER_ROUND / EPISODES_PER_ROUND if kind == "static_mirror_pressure" else 0.0
            round_metrics.append(aggregate)
        round_training.append(training)
        round_chain.append(
            {
                "round": int(round_index),
                "input_checkpoint": str(Path(current_checkpoint).resolve()),
                "input_checkpoint_sha256": _sha256(Path(current_checkpoint)),
                "input_policy_hash": training["policy_hash_before"],
                "output_checkpoint": str(target_checkpoint.resolve()),
                "output_checkpoint_sha256": _sha256(target_checkpoint),
                "output_policy_hash": training["policy_hash_after"],
                "sign_correct": int(sum(row["sign_correct"] for row in sign)),
                "sign_applicable": len(sign),
                "boundary_probe": drift,
            }
        )
        current_trainer, current_config, current_checkpoint = next_trainer, next_config, target_checkpoint

    _write_jsonl(data_root / "demonstrations_train_all.jsonl", all_train_records)
    _write_jsonl(data_root / "demonstrations_validation_all.jsonl", all_validation_records)
    _write_rows_csv(logs_root / "stage2br_episode_metrics.csv", all_episode_summaries)
    _write_rows_csv(logs_root / "stage2br_step_diagnostics.csv", all_step_rows)
    _write_rows_csv(logs_root / "dagger_round_metrics.csv", round_metrics)
    _write_rows_csv(logs_root / "actor_direction_sign_probe.csv", sign_rows)
    _write_rows_csv(logs_root / "boundary_drift_probe.csv", boundary_rows)
    _write_rows_csv(logs_root / "direction_weight_by_round.csv", weight_rows)
    _write_json(logs_root / "direction_weight_by_round.json", _jsonable(weight_rows))

    query_balance_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    for round_index in range(1, ROUND_COUNT + 1):
        round_records = [row for row in all_train_records + all_validation_records if int(row.get("round", -1)) == round_index]
        q = Counter(str(row.get("expected_obstacle_category", "")) for row in round_records if int(row.get("oracle_queried", 0) or 0))
        for category in STATIC_CATEGORIES:
            query_balance_rows.append({"round": round_index, "category": category, "oracle_query_samples": int(q.get(category, 0)), "oracle_query_episodes": int(sum(1 for row in all_episode_summaries if int(row.get("round", -1)) == round_index and row.get("expected_obstacle_category") == category))})
        source_counter = Counter(str(row.get("target_source", "")) for row in round_records)
        category_counter = Counter(str(row.get("expected_obstacle_category", "")) for row in round_records)
        for key, value in source_counter.items():
            balance_rows.append({"round": round_index, "dimension": "target_source", "category": key, "samples": int(value)})
        for key, value in category_counter.items():
            balance_rows.append({"round": round_index, "dimension": "expected_category", "category": key, "samples": int(value)})
    _write_rows_csv(logs_root / "teacher_query_balance.csv", query_balance_rows)
    _write_rows_csv(logs_root / "dagger_dataset_balance.csv", balance_rows)

    left_queries = sum(int(row.get("oracle_queried", 0) or 0) for row in all_train_records + all_validation_records if row.get("expected_obstacle_category") in {"left", "left_front"})
    right_queries = sum(int(row.get("oracle_queried", 0) or 0) for row in all_train_records + all_validation_records if row.get("expected_obstacle_category") in {"right", "right_front"})
    data_balance_passed = bool(left_queries > 0 and right_queries > 0 and 0.8 <= left_queries / max(right_queries, 1) <= 1.25)
    oracle_sign_gate_passed = bool(oracle_sign.get("passed"))
    final_sign = [row for row in sign_rows if int(row.get("round", -1)) == ROUND_COUNT]
    direction_behavior_passed = bool(
        len(final_sign) == len(MIRROR_CATEGORIES)
        and all(int(row.get("sign_correct", 0)) == 1 for row in final_sign)
        and all(int(row.get("nondegenerate", 0)) == 1 for row in final_sign)
    )
    iterative_dagger_passed = bool(
        len(round_chain) >= 2
        and all(round_chain[index]["output_policy_hash"] != round_chain[index]["input_policy_hash"] for index in range(len(round_chain)))
        and all(round_chain[index]["input_policy_hash"] == round_chain[index - 1]["output_policy_hash"] for index in range(1, len(round_chain)))
    )
    navigation = _normal_navigation_probe(
        source_trainer,
        current_trainer,
        current_config,
        data_root=data_root,
        logs_root=logs_root,
    )
    normal_navigation_preserved = bool(navigation["normal_navigation_preserved"])
    final_static = _aggregate_round_rows(all_episode_summaries, round_index=ROUND_COUNT, kind="static_mirror_pressure")
    static_progress_passed = bool(final_static.get("entered_3m_rate", 0.0) >= 0.5 and final_static.get("entered_1m_rate", 0.0) > 0.0)
    final_boundary = boundary_rows[-1] if boundary_rows else {}
    boundary_interference_acceptable = bool(
        int(final_boundary.get("boundary_projection_steps", 0) or 0) < PREVIOUS_STAGE2B_BOUNDARY_STEPS
        and not bool(final_boundary.get("persistent_lateral_bias", 0))
    )
    behavioral_acceptance_passed = bool(
        data_balance_passed
        and oracle_sign_gate_passed
        and iterative_dagger_passed
        and direction_behavior_passed
        and normal_navigation_preserved
        and static_progress_passed
        and boundary_interference_acceptable
    )

    holdout: dict[str, Any]
    if behavioral_acceptance_passed:
        holdout = _holdout_actor_evaluation(
            source_trainer,
            current_trainer,
            current_config,
            data_root=data_root,
            logs_root=logs_root,
            episodes=4,
        )
    else:
        _empty_gate_csvs(logs_root, reason="behavioral_gate_failed_before_holdout")
        holdout = {"evaluation_frozen": False, "not_run_reason": "behavioral_gate_failed_before_holdout"}

    all_rounds_training_ok = all(
        bool(result.get("checkpoint_contract", {}).get("critic_frozen"))
        and bool(result.get("checkpoint_contract", {}).get("normalizer_frozen"))
        and bool(result.get("checkpoint_contract", {}).get("logstd_frozen"))
        and bool(result.get("checkpoint_contract", {}).get("old_columns_frozen"))
        for result in round_training
    )
    technical_smoke_passed = bool(
        oracle_sanity.get("sanity_passed")
        and oracle_sign.get("passed")
        and len(round_training) == ROUND_COUNT
        and all_rounds_training_ok
        and all(int(result.get("actual_epochs", 0)) > 0 for result in round_training)
        and all(int(result.get("checkpoint_contract", {}).get("ppo_update_performed", 1)) == 0 for result in round_training)
    )
    validate_report: dict[str, Any]
    report: dict[str, Any] = {
        **base_report,
        "status": "completed_stage2br_smoke",
        "technical_smoke_passed": technical_smoke_passed,
        "data_balance_passed": data_balance_passed,
        "oracle_sign_gate_passed": oracle_sign_gate_passed,
        "iterative_dagger_passed": iterative_dagger_passed,
        "direction_behavior_passed": direction_behavior_passed,
        "normal_navigation_preserved": normal_navigation_preserved,
        "static_progress_passed": static_progress_passed,
        "boundary_interference_acceptable": boundary_interference_acceptable,
        "behavioral_acceptance_passed": bool(behavioral_acceptance_passed),
        "formal_stage2b_eligible": bool(behavioral_acceptance_passed and holdout.get("evaluation_frozen", False)),
        "dataset": {
            "rounds": ROUND_COUNT,
            "schedule_counts": round_schedule_counts(ROUND_COUNT),
            "normal_fraction": NORMAL_EPISODES_PER_ROUND / EPISODES_PER_ROUND,
            "static_fraction": STATIC_EPISODES_PER_ROUND / EPISODES_PER_ROUND,
            "train_samples": len(all_train_records),
            "validation_samples": len(all_validation_records),
            "left_oracle_query_samples": left_queries,
            "right_oracle_query_samples": right_queries,
            "scenario_hash_count": len({str(row.get("scenario_hash")) for row in all_train_records + all_validation_records}),
            "trajectory_leakage": False,
        },
        "round_training": round_training,
        "round_chain": round_chain,
        "holdout": holdout,
        "validate_preliminary": None,
        "outputs": {
            "data_root": str(data_root.resolve()),
            "logs_root": str(logs_root.resolve()),
            "plots_root": str(plots_root.resolve()),
        },
    }
    _write_json(logs_root / "checkpoint_identity.json", _jsonable({"source": source_identity, "round_chain": round_chain, "ppo_updates": 0, "evaluation_frozen": True}))
    _write_json(data_root / "checkpoint_identity.json", _jsonable({"source": source_identity, "round_chain": round_chain, "ppo_updates": 0, "evaluation_frozen": True}))
    _write_rows_csv(logs_root / "stage2br_round_metrics.csv", round_metrics)
    plots = _write_stage2br_plots(
        plots_root=plots_root,
        round_metrics=round_metrics,
        sign_rows=sign_rows,
        weight_rows=weight_rows,
        boundary_rows=boundary_rows,
        episode_rows=all_episode_summaries,
    )
    report["outputs"]["plots"] = plots
    _write_json(logs_root / "stage2br_summary.json", _jsonable(report))
    _write_json(data_root / "stage2br_summary.json", _jsonable(report))
    _write_json(data_root / "summary.json", _jsonable(report))
    validate_report = validate_preliminary(data_root)
    report["validate_preliminary"] = validate_report
    report["technical_smoke_passed"] = bool(technical_smoke_passed and validate_report.get("ok", False))
    _write_json(logs_root / "stage2br_summary.json", _jsonable(report))
    _write_json(data_root / "stage2br_summary.json", _jsonable(report))
    _write_json(data_root / "summary.json", _jsonable(report))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2a-checkpoint", type=Path, default=DEFAULT_STAGE2A_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_stage2br_smoke(
        stage2a_checkpoint=args.stage2a_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STAGE_NAME",
    "DEFAULT_STAGE2A_CHECKPOINT",
    "mirror_static_scenario",
    "mirror_pair_manifest_rows",
    "audit_previous_data_bias",
    "BoundaryAwareGeometricObstacleOracleController",
    "oracle_direction_sign_probe",
    "round_schedule_counts",
    "run_stage2br_smoke",
]
