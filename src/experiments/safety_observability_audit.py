"""Read-only audit of dynamic-obstacle observability and safety envelopes.

This module intentionally does not change the simulator.  It consumes the
frozen, deterministic rows produced by ``safety_timing_counterfactual`` and
adds an offline observability and multi-horizon geometry audit.  Reusing that
frozen replay is important: the four collision events and their random
streams remain exactly the same, while the new calculations cannot feed back
into an actor, critic, reward, or subsequent environment state.

The actor-visible obstacle representation in the current Python environment
is an aggregate four-vector::

    [minimum current obstacle distance, box count, sphere count, timestep].

It is deliberately treated as an aggregate observation.  The audit never
claims that this vector identifies a particular collision obstacle, and it
records the absence of obstacle velocity, relative closing-rate, and
time-to-collision fields explicitly.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.obstacle_model import Obstacle, point_obstacle_distance, segment_intersects_obstacle
from ..synthetic.episode_generator import _jsonable
from .safety_mappo_evaluation import (
    BASELINE_BRANCH,
    DEFAULT_POLICY_SEEDS,
    DEFAULT_SCENARIO_SEEDS,
    FULL_BRANCH,
    _bootstrap_ci,
    _csv_value,
    _finite_or,
    _write_csv,
    _write_json,
)
from .safety_mechanism_audit import _obstacle_snapshot, _scene_clearance_detail
from .safety_timing_counterfactual import (
    _active_flag,
    _obstacles_from_payload,
    _payload,
    _shadow_project,
)


OBSERVABILITY_CAUSES = (
    "not_observable",
    "observable_but_no_motion_feature",
    "prediction_horizon_insufficient",
    "projection_coverage_insufficient",
    "post_update_ordering",
    "unclassified",
)
_SCHEMA_VERSION = "safety-observability-audit-v1.0"
_PLOT_FILE = "safety_observability_audit.png"
_TOLERANCE = 1e-7


@dataclass(frozen=True)
class SafetyObservabilityAuditConfig:
    """Configuration for the validation-only observability audit."""

    model_root: Path = Path(
        "data/synthetic/"
        "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
    )
    scenario_root: Path = Path("data/synthetic/safety_risk_screening_seed2026/scenarios")
    source_timing_audit_dir: Path = Path(
        "logs/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_timing_counterfactual"
    )
    output_dir: Path = Path(
        "logs/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_observability_audit"
    )
    plots_dir: Path = Path(
        "plots/preliminary/safety_mappo_evaluation_seed2026/"
        "safety_observability_audit"
    )
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    branches: tuple[str, ...] = (BASELINE_BRANCH, FULL_BRANCH)
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    deterministic: bool = True
    safety_projection_enabled: bool = True
    prediction_horizons: tuple[int, ...] = (1, 2, 3)
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("safety observability audit is restricted to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("the deterministic observability audit supports device='cpu' only")
        if not self.deterministic:
            raise ValueError("safety observability audit must use deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("observability audit requires safety projection enabled")
        if not self.policy_seeds or not self.scenario_seeds:
            raise ValueError("at least one policy seed and scenario seed are required")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if len(self.branches) != 2 or set(self.branches) != {BASELINE_BRANCH, FULL_BRANCH}:
            raise ValueError(
                "branches must be exactly baseline_mappo and "
                "budgeted_quality_aware_mappo"
            )
        if tuple(self.prediction_horizons) != (1, 2, 3):
            raise ValueError("prediction_horizons must be exactly (1, 2, 3)")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: SafetyObservabilityAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-observability-audit] {message}", flush=True)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _prepare_outputs(config: SafetyObservabilityAuditConfig) -> tuple[Path, Path]:
    output = Path(config.output_dir).resolve()
    plots = Path(config.plots_dir).resolve()
    sources = [
        Path(config.model_root).resolve(),
        Path(config.scenario_root).resolve(),
        Path(config.source_timing_audit_dir).resolve(),
    ]
    if output == plots:
        raise ValueError("output_dir and plots_dir must be distinct")
    for path in (output, plots):
        if any(_paths_overlap(path, source) for source in sources):
            raise ValueError("audit output cannot replace a source directory")
        if path.exists() and any(path.iterdir()) and not config.force:
            raise FileExistsError(
                f"audit output is not empty: {path}; use --force to replace it"
            )
    if config.force:
        for path in (output, plots):
            if path.exists():
                shutil.rmtree(path)
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return output, plots


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing frozen replay file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_payload(value: Any) -> Any:
    return _payload(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    decoded = _as_payload(value)
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _as_sequence(value: Any) -> list[Any]:
    decoded = _as_payload(value)
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)):
        return list(decoded)
    return []


def _vector(value: Any, size: int = 3) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(size, dtype=float)
    if result.size != size:
        return np.resize(result, size).astype(float)
    return result.astype(float)


def _finite_vector(value: Any, size: int = 3) -> bool:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        return bool(array.size == size and np.isfinite(array).all())
    except (TypeError, ValueError):
        return False


def _finite_value(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _contains_motion_key(value: Any) -> bool:
    """Find obstacle-motion fields without treating the UAV state as one."""

    motion_tokens = (
        "velocity",
        "relative_velocity",
        "closing",
        "time_to_collision",
        "ttc",
        "prediction",
        "horizon",
        "distance_rate",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in motion_tokens):
                return True
            if _contains_motion_key(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_motion_key(child) for child in value)
    return False


def _feature_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = _as_mapping(row.get("actor_observation_json", {}))
    feature_payload = observation.get(
        "obstacle_features",
        _as_payload(row.get("actor_obstacle_features_json", [])),
    )
    try:
        features = np.asarray(feature_payload, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        features = np.empty(0, dtype=float)
    present = int(features.size > 0)
    finite = int(present and np.isfinite(features).all())
    dynamic_motion = int(_contains_motion_key(feature_payload))
    explicit_ids = int(
        isinstance(feature_payload, Mapping)
        and any("id" in str(key).lower() for key in feature_payload)
    )
    return {
        "actor_obstacle_feature_present": present,
        "actor_obstacle_feature_finite": finite,
        "actor_obstacle_feature_dimension": int(features.size),
        "actor_obstacle_feature_min_distance_m": (
            float(features[0]) if finite else -1.0
        ),
        "actor_obstacle_motion_feature_present": dynamic_motion,
        "actor_obstacle_id_feature_present": explicit_ids,
        "actor_obstacle_feature_semantics": (
            "aggregate_min_distance_box_count_sphere_count_timestep"
            if features.size == 4
            else "unknown_or_missing"
        ),
        "actor_observation_truth_free": _int(
            row.get("actor_observation_truth_free", 0)
        ),
    }


def _dynamic_items(payload: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for item in _as_sequence(payload):
        if not isinstance(item, Mapping):
            continue
        obstacle_type = _text(item.get("obstacle_type", ""), "").lower()
        velocity = _vector(item.get("velocity_mps", (0.0, 0.0, 0.0)))
        if obstacle_type not in {"building", "static", "static_obstacle"} or np.linalg.norm(velocity) > _TOLERANCE:
            result.append(item)
    return result


def _predict_objects(
    before_payload: Any,
    horizon: int,
    dt_s: float,
    bounds_min_m: Sequence[float],
    bounds_max_m: Sequence[float],
) -> list[Obstacle]:
    objects = _obstacles_from_payload(before_payload)
    lower = _vector(bounds_min_m)
    upper = _vector(bounds_max_m)
    for _ in range(int(horizon)):
        for obstacle in objects:
            obstacle.update(float(dt_s), lower, upper)
    return objects


def _dynamic_min_clearance(
    positions: Mapping[str, np.ndarray],
    obstacles: Iterable[Obstacle],
    safety_distance_m: float,
) -> tuple[float, int, str]:
    candidates: list[tuple[float, str]] = []
    dynamic_ids = {
        str(item.obstacle_id)
        for item in obstacles
        if item.active
        and (
            str(item.obstacle_type).lower() not in {"building", "static", "static_obstacle"}
            or np.linalg.norm(_vector(item.velocity_mps)) > _TOLERANCE
        )
    }
    for uav_id, position in positions.items():
        for obstacle in obstacles:
            if obstacle.active and str(obstacle.obstacle_id) in dynamic_ids:
                candidates.append(
                    (float(point_obstacle_distance(position, obstacle)), str(obstacle.obstacle_id))
                )
    if not candidates:
        return float(safety_distance_m), 0, "none"
    clearance, source_id = min(candidates, key=lambda item: (item[0], item[1]))
    return float(clearance), int(clearance < float(safety_distance_m)), source_id


def _dynamic_segment_risk(
    current_positions: Mapping[str, np.ndarray],
    nominal_positions: Mapping[str, np.ndarray],
    obstacles: Iterable[Obstacle],
) -> tuple[int, str]:
    for uav_id in sorted(current_positions):
        for obstacle in sorted(obstacles, key=lambda item: str(item.obstacle_id)):
            if obstacle.active and segment_intersects_obstacle(
                current_positions[uav_id], nominal_positions[uav_id], obstacle
            ):
                return 1, str(obstacle.obstacle_id)
    return 0, "none"


def _unique_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, int], list[Mapping[str, Any]]]:
    groups: dict[tuple[int, int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            _int(row.get("policy_seed"), -1),
            _int(row.get("scenario_seed"), -1),
            _text(row.get("branch")),
            _int(row.get("episode_step"), -1),
        )
        groups.setdefault(key, []).append(row)
    return groups


def _horizon_context(
    group: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int],
    dt_s: float = 0.2,
    bounds_min_m: Sequence[float] = (0.0, 0.0, 2.0),
    bounds_max_m: Sequence[float] = (200.0, 200.0, 60.0),
    safety_distance_m: float = 2.0,
) -> dict[str, Any]:
    first = group[0]
    before_payload = _as_payload(first.get("before_obstacle_state_json", []))
    current_positions = {
        _text(row.get("uav_id")): _vector(_as_payload(row.get("current_position_json")))
        for row in group
    }
    nominal_positions = {
        _text(row.get("uav_id")): _vector(_as_payload(row.get("nominal_position_json")))
        for row in group
    }
    actual_positions = {
        _text(row.get("uav_id")): _vector(_as_payload(row.get("actual_position_json")))
        for row in group
    }
    horizon: dict[str, Any] = {}
    for h in horizons:
        objects = _predict_objects(
            before_payload,
            int(h),
            dt_s,
            bounds_min_m,
            bounds_max_m,
        )
        scene_detail = _scene_clearance_detail(
            actual_positions,
            objects,
            safety_distance_m,
        )
        dynamic_clearance, dynamic_collision, dynamic_source = _dynamic_min_clearance(
            actual_positions,
            objects,
            safety_distance_m,
        )
        segment_risk, segment_source = _dynamic_segment_risk(
            current_positions,
            nominal_positions,
            objects,
        )
        shadow_positions: dict[str, np.ndarray] = {}
        shadow_projections: dict[str, dict[str, Any]] = {}
        for uav_id in sorted(current_positions):
            shadow_position, projection = _shadow_project(
                uav_id,
                current_positions[uav_id],
                nominal_positions[uav_id],
                current_positions,
                objects,
                safety_distance_m,
            )
            shadow_positions[uav_id] = shadow_position
            shadow_projections[uav_id] = projection
        shadow_detail = _scene_clearance_detail(shadow_positions, objects, safety_distance_m)
        shadow_triggered = int(any(int(item.get("triggered", 0)) for item in shadow_projections.values()))
        shadow_dynamic_clearance, shadow_dynamic_collision, shadow_source = _dynamic_min_clearance(
            shadow_positions,
            objects,
            safety_distance_m,
        )
        horizon[str(int(h))] = {
            "predicted_scene_min_clearance_m": _finite_or(
                scene_detail.get("minimum_clearance_m"), safety_distance_m
            ),
            "predicted_dynamic_min_clearance_m": float(dynamic_clearance),
            "predicted_dynamic_collision": int(dynamic_collision),
            "predicted_dynamic_source_id": dynamic_source,
            "predicted_dynamic_segment_risk": int(segment_risk),
            "predicted_dynamic_segment_source_id": segment_source,
            "predicted_projection_triggered": shadow_triggered,
            "predicted_projection_collision": int(shadow_detail.get("collision", 0)),
            "predicted_projection_dynamic_min_clearance_m": float(shadow_dynamic_clearance),
            "predicted_projection_dynamic_collision": int(shadow_dynamic_collision),
            "predicted_projection_source_id": shadow_source,
            "predicted_projection_detail_json": shadow_projections,
        }
    return {
        "horizon": horizon,
        "before_payload": before_payload,
        "current_positions": current_positions,
        "nominal_positions": nominal_positions,
        "actual_positions": actual_positions,
        "safety_distance_m": float(safety_distance_m),
    }


def _classify_collision(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[str, str]:
    """Return a primary cause and a secondary geometry finding."""

    if not _int(row.get("actual_collision"), 0):
        return "unclassified", "none"
    if not profile.get("actor_obstacle_feature_finite", 0):
        return "not_observable", "missing_or_nonfinite_aggregate_obstacle_feature"
    horizon = context["horizon"]
    h1 = horizon["1"]
    h2 = horizon["2"]
    h3 = horizon["3"]
    if not profile.get("actor_obstacle_motion_feature_present", 0):
        secondary = (
            "projection_coverage_insufficient"
            if h1["predicted_projection_collision"]
            else "none"
        )
        return "observable_but_no_motion_feature", secondary
    if (
        not h1["predicted_dynamic_collision"]
        and (h2["predicted_dynamic_collision"] or h3["predicted_dynamic_collision"])
    ):
        return "prediction_horizon_insufficient", "none"
    if h1["predicted_dynamic_collision"]:
        if h1["predicted_projection_collision"]:
            return "projection_coverage_insufficient", "none"
        return "post_update_ordering", "none"
    return "unclassified", "insufficient_dynamic_geometry_evidence"


def _collision_event_from_group(
    group: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    collision_rows = [row for row in group if _int(row.get("actual_collision"), 0)]
    if not collision_rows:
        return None
    row = sorted(collision_rows, key=lambda item: _text(item.get("uav_id")))[0]
    profile = _feature_profile(row)
    before = _as_sequence(row.get("before_obstacle_state_json", []))
    moved_ids = _as_sequence(row.get("dynamic_obstacle_moved_ids_json", []))
    cause, secondary = _classify_collision(row, profile, context)
    h = context["horizon"]
    collision_source_id = _text(row.get("actual_collision_source_id"), "none")
    actor_text = _text(row.get("actor_observation_json", ""))
    source_explicitly_present = int(
        collision_source_id != "none" and collision_source_id in actor_text
    )
    return {
        "policy_seed": _int(row.get("policy_seed"), -1),
        "scenario_seed": _int(row.get("scenario_seed"), -1),
        "branch": _text(row.get("branch")),
        "scenario_hash": _text(row.get("scenario_hash")),
        "episode_step": _int(row.get("episode_step"), -1),
        "time_s": _finite_or(row.get("time_s"), 0.0),
        "collision_uav_id": _text(row.get("uav_id")),
        "collision_source_id": collision_source_id,
        "collision_source_type": _text(row.get("actual_collision_source_type")),
        "collision_obstacle_type": _text(row.get("actual_collision_obstacle_type")),
        "dynamic_obstacle_count": len(_dynamic_items(before)),
        "dynamic_obstacle_moved": _int(row.get("dynamic_obstacle_moved"), 0),
        "dynamic_obstacle_moved_ids_json": moved_ids,
        "actor_obstacle_feature_present": profile["actor_obstacle_feature_present"],
        "actor_obstacle_feature_finite": profile["actor_obstacle_feature_finite"],
        "actor_obstacle_feature_dimension": profile["actor_obstacle_feature_dimension"],
        "actor_obstacle_feature_min_distance_m": profile["actor_obstacle_feature_min_distance_m"],
        "actor_obstacle_feature_semantics": profile["actor_obstacle_feature_semantics"],
        "actor_obstacle_geometry_observable": int(
            profile["actor_obstacle_feature_present"]
            and profile["actor_obstacle_feature_finite"]
        ),
        "collision_source_identifiable_from_actor_observation": source_explicitly_present,
        "collision_source_observation_status": (
            "identified" if source_explicitly_present else "aggregate_only_not_identifiable"
        ),
        "actor_obstacle_motion_feature_present": profile["actor_obstacle_motion_feature_present"],
        "actor_relative_closing_rate_present": 0,
        "actor_time_to_collision_present": 0,
        "actor_future_obstacle_prediction_present": 0,
        "observation_is_current_aggregate_only": int(
            profile["actor_obstacle_feature_dimension"] == 4
            and not profile["actor_obstacle_motion_feature_present"]
        ),
        "pre_dynamic_actual_clearance_m": _finite_or(
            row.get("pre_dynamic_actual_minimum_clearance_m"), 0.0
        ),
        "post_dynamic_actual_clearance_m": _finite_or(
            row.get("actual_minimum_clearance_m"), 0.0
        ),
        "dynamic_clearance_change_m": _finite_or(
            row.get("dynamic_clearance_change_m"), 0.0
        ),
        "pre_dynamic_collision": _int(row.get("pre_dynamic_actual_collision"), 0),
        "post_dynamic_collision": _int(row.get("actual_collision"), 0),
        "dynamic_update_order": _text(
            row.get("dynamic_obstacle_update_order"),
            "motion -> obstacle_update -> collision_check",
        ),
        "current_projection_geometry": (
            "current-obstacle segment-intersection and endpoint-clearance checks"
        ),
        "current_projection_uses_swept_envelope": 0,
        "current_projection_triggered": _int(row.get("projection_triggered"), 0),
        "current_projection_reason": _text(row.get("pre_motion_projection_reason")),
        "nominal_position_json": _as_payload(row.get("nominal_position_json", [])),
        "actual_position_json": _as_payload(row.get("actual_position_json", [])),
        "delta_v_commanded_json": _as_payload(row.get("delta_v_commanded_json", [])),
        "delta_v_applied_json": _as_payload(row.get("delta_v_applied_json", [])),
        "nominal_actual_position_closure": _int(
            row.get("nominal_actual_position_closure"), 0
        ),
        "h1_predicted_dynamic_min_clearance_m": h["1"]["predicted_dynamic_min_clearance_m"],
        "h2_predicted_dynamic_min_clearance_m": h["2"]["predicted_dynamic_min_clearance_m"],
        "h3_predicted_dynamic_min_clearance_m": h["3"]["predicted_dynamic_min_clearance_m"],
        "h1_predicted_dynamic_collision": h["1"]["predicted_dynamic_collision"],
        "h2_predicted_dynamic_collision": h["2"]["predicted_dynamic_collision"],
        "h3_predicted_dynamic_collision": h["3"]["predicted_dynamic_collision"],
        "h1_predicted_projection_triggered": h["1"]["predicted_projection_triggered"],
        "h2_predicted_projection_triggered": h["2"]["predicted_projection_triggered"],
        "h3_predicted_projection_triggered": h["3"]["predicted_projection_triggered"],
        "h1_predicted_projection_collision": h["1"]["predicted_projection_collision"],
        "h2_predicted_projection_collision": h["2"]["predicted_projection_collision"],
        "h3_predicted_projection_collision": h["3"]["predicted_projection_collision"],
        "h1_predicted_segment_risk": h["1"]["predicted_dynamic_segment_risk"],
        "h2_predicted_segment_risk": h["2"]["predicted_dynamic_segment_risk"],
        "h3_predicted_segment_risk": h["3"]["predicted_dynamic_segment_risk"],
        "one_step_counterfactual_collision_avoided": int(
            _int(row.get("actual_collision"), 0)
            and not h["1"]["predicted_projection_collision"]
        ),
        "primary_cause": cause,
        "secondary_geometry_finding": secondary,
        "terminated": _int(row.get("terminated"), 0),
        "termination_reason": _text(row.get("termination_reason")),
        "task_complete": _int(row.get("task_complete"), 0),
        "early_termination_censored": _int(
            row.get("early_termination_censored"), 0
        ),
        "offline_geometry_only": 1,
        "counterfactual_state_unchanged": _int(
            row.get("counterfactual_state_unchanged"), 0
        ),
    }


def _envelope_row(
    group: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    row = sorted(group, key=lambda item: _text(item.get("uav_id")))[0]
    profile = _feature_profile(row)
    before_payload = context["before_payload"]
    after_payload = _as_payload(row.get("after_obstacle_state_json", []))
    before_dynamic = len(_dynamic_items(before_payload))
    after_dynamic = len(_dynamic_items(after_payload))
    h = context["horizon"]
    actual_collision = max(_int(item.get("actual_collision"), 0) for item in group)
    collision_row = next(
        (item for item in group if _int(item.get("actual_collision"), 0)),
        row,
    )
    return {
        "policy_seed": _int(row.get("policy_seed"), -1),
        "scenario_seed": _int(row.get("scenario_seed"), -1),
        "branch": _text(row.get("branch")),
        "scenario_hash": _text(row.get("scenario_hash")),
        "episode_step": _int(row.get("episode_step"), -1),
        "time_s": _finite_or(row.get("time_s"), 0.0),
        "actor_obstacle_feature_present": profile["actor_obstacle_feature_present"],
        "actor_obstacle_feature_finite": profile["actor_obstacle_feature_finite"],
        "actor_obstacle_feature_min_distance_m": profile["actor_obstacle_feature_min_distance_m"],
        "actor_obstacle_feature_dimension": profile["actor_obstacle_feature_dimension"],
        "actor_obstacle_motion_feature_present": profile["actor_obstacle_motion_feature_present"],
        "dynamic_obstacle_count_before": before_dynamic,
        "dynamic_obstacle_count_after": after_dynamic,
        "dynamic_obstacle_moved_count": len(_as_sequence(row.get("dynamic_obstacle_moved_ids_json", []))),
        "pre_dynamic_actual_minimum_clearance_m": _finite_or(
            row.get("pre_dynamic_actual_minimum_clearance_m"), 0.0
        ),
        "post_dynamic_actual_minimum_clearance_m": _finite_or(
            row.get("actual_minimum_clearance_m"), 0.0
        ),
        "dynamic_clearance_change_m": _finite_or(
            row.get("dynamic_clearance_change_m"), 0.0
        ),
        "nominal_position_json": _as_payload(row.get("nominal_position_json", [])),
        "actual_position_json": _as_payload(row.get("actual_position_json", [])),
        "projection_triggered": _int(row.get("projection_triggered"), 0),
        "safety_projected": _int(row.get("safety_projected"), 0),
        "pre_motion_projection_risk": _int(row.get("pre_motion_projection_risk"), 0),
        "pre_dynamic_nominal_collision": _int(row.get("pre_dynamic_nominal_collision"), 0),
        "pre_dynamic_actual_collision": _int(row.get("pre_dynamic_actual_collision"), 0),
        "post_dynamic_nominal_collision": _int(row.get("post_dynamic_nominal_collision"), 0),
        "actual_collision": actual_collision,
        "actual_collision_source_id": _text(collision_row.get("actual_collision_source_id")),
        "actual_collision_obstacle_type": _text(collision_row.get("actual_collision_obstacle_type")),
        "dynamic_obstacle_update_order": _text(
            row.get("dynamic_obstacle_update_order"),
            "motion -> obstacle_update -> collision_check",
        ),
        "current_projection_geometry": "current-obstacle segment/intersection plus endpoint clearance",
        "current_projection_uses_swept_envelope": 0,
        "h1_predicted_dynamic_min_clearance_m": h["1"]["predicted_dynamic_min_clearance_m"],
        "h2_predicted_dynamic_min_clearance_m": h["2"]["predicted_dynamic_min_clearance_m"],
        "h3_predicted_dynamic_min_clearance_m": h["3"]["predicted_dynamic_min_clearance_m"],
        "h1_predicted_dynamic_collision": h["1"]["predicted_dynamic_collision"],
        "h2_predicted_dynamic_collision": h["2"]["predicted_dynamic_collision"],
        "h3_predicted_dynamic_collision": h["3"]["predicted_dynamic_collision"],
        "h1_predicted_projection_triggered": h["1"]["predicted_projection_triggered"],
        "h2_predicted_projection_triggered": h["2"]["predicted_projection_triggered"],
        "h3_predicted_projection_triggered": h["3"]["predicted_projection_triggered"],
        "h1_predicted_projection_collision": h["1"]["predicted_projection_collision"],
        "h2_predicted_projection_collision": h["2"]["predicted_projection_collision"],
        "h3_predicted_projection_collision": h["3"]["predicted_projection_collision"],
        "h1_predicted_segment_risk": h["1"]["predicted_dynamic_segment_risk"],
        "h2_predicted_segment_risk": h["2"]["predicted_dynamic_segment_risk"],
        "h3_predicted_segment_risk": h["3"]["predicted_dynamic_segment_risk"],
        "nominal_actual_position_closure": _int(
            row.get("nominal_actual_position_closure"), 0
        ),
        "state_finite": _int(row.get("state_finite"), 0),
        "covariance_finite": _int(row.get("covariance_finite"), 0),
        "counterfactual_state_unchanged": _int(
            row.get("counterfactual_state_unchanged"), 0
        ),
        "offline_geometry_only": 1,
    }


def _episode_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (_int(row.get("policy_seed"), -1), _int(row.get("scenario_seed"), -1), _text(row.get("branch")))


def _scenario_summary(
    envelope_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: SafetyObservabilityAuditConfig,
) -> list[dict[str, Any]]:
    by_scenario: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in envelope_rows:
        by_scenario.setdefault((_int(row.get("scenario_seed"), -1), _text(row.get("branch"))), []).append(row)
    event_by_scenario: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in events:
        event_by_scenario.setdefault((_int(row.get("scenario_seed"), -1), _text(row.get("branch"))), []).append(row)
    result: list[dict[str, Any]] = []
    for scenario_seed in config.scenario_seeds:
        for branch in config.branches:
            rows = by_scenario.get((int(scenario_seed), branch), [])
            collision_episode_keys = {
                _episode_key(row)
                for row in rows
                if _int(row.get("actual_collision"), 0)
            }
            branch_events = event_by_scenario.get((int(scenario_seed), branch), [])
            causes = [_text(row.get("primary_cause")) for row in branch_events]
            cause = (
                max(sorted(set(causes)), key=lambda item: (causes.count(item), item))
                if causes
                else "none"
            )
            def mean(key: str, source: Sequence[Mapping[str, Any]] = branch_events) -> float:
                values = [_finite_or(item.get(key), 0.0) for item in source]
                return float(np.mean(values)) if values else 0.0
            result.append(
                {
                    "scenario_seed": int(scenario_seed),
                    "branch": branch,
                    "scenario_hash": _text(rows[0].get("scenario_hash")) if rows else "",
                    "policy_seed_episode_count": len({_int(row.get("policy_seed"), -1) for row in rows}),
                    "collision_episode_count": len(collision_episode_keys),
                    "scenario_collision_rate": float(
                        len(collision_episode_keys) / max(1, len(config.policy_seeds))
                    ),
                    "collision_event_count": len(branch_events),
                    "aggregate_obstacle_observable_collision_count": sum(
                        int(_int(row.get("actor_obstacle_geometry_observable"), 0))
                        for row in branch_events
                    ),
                    "motion_feature_collision_count": sum(
                        int(_int(row.get("actor_obstacle_motion_feature_present"), 0))
                        for row in branch_events
                    ),
                    "projection_coverage_finding_count": sum(
                        int(_text(row.get("primary_cause")) == "projection_coverage_insufficient")
                        for row in branch_events
                    ),
                    "post_update_ordering_finding_count": sum(
                        int(_text(row.get("primary_cause")) == "post_update_ordering")
                        for row in branch_events
                    ),
                    "prediction_horizon_finding_count": sum(
                        int(_text(row.get("primary_cause")) == "prediction_horizon_insufficient")
                        for row in branch_events
                    ),
                    "mean_pre_dynamic_clearance_m": mean("pre_dynamic_actual_clearance_m"),
                    "mean_post_dynamic_clearance_m": mean("post_dynamic_actual_clearance_m"),
                    "mean_dynamic_clearance_change_m": mean("dynamic_clearance_change_m"),
                    "mean_h1_predicted_dynamic_clearance_m": mean(
                        "h1_predicted_dynamic_min_clearance_m"
                    ),
                    "mean_h2_predicted_dynamic_clearance_m": mean(
                        "h2_predicted_dynamic_min_clearance_m"
                    ),
                    "mean_h3_predicted_dynamic_clearance_m": mean(
                        "h3_predicted_dynamic_min_clearance_m"
                    ),
                    "actor_obstacle_observable_rate": float(
                        np.mean([_int(row.get("actor_obstacle_feature_finite"), 0) for row in rows])
                    ) if rows else 0.0,
                    "actor_motion_feature_rate": float(
                        np.mean([_int(row.get("actor_obstacle_motion_feature_present"), 0) for row in rows])
                    ) if rows else 0.0,
                    "actual_projection_rate": float(
                        np.mean([_int(row.get("projection_triggered"), 0) for row in rows])
                    ) if rows else 0.0,
                    "primary_cause": cause,
                    "collision_energy_censoring_present": int(bool(collision_episode_keys)),
                    "independent_unit": "scenario_seed",
                }
            )
    return result


def _paired_contrasts(
    scenario_rows: Sequence[Mapping[str, Any]],
    config: SafetyObservabilityAuditConfig,
) -> list[dict[str, Any]]:
    metrics = (
        "scenario_collision_rate",
        "mean_pre_dynamic_clearance_m",
        "mean_post_dynamic_clearance_m",
        "mean_dynamic_clearance_change_m",
        "mean_h1_predicted_dynamic_clearance_m",
        "mean_h2_predicted_dynamic_clearance_m",
        "mean_h3_predicted_dynamic_clearance_m",
        "actor_obstacle_observable_rate",
        "actor_motion_feature_rate",
        "actual_projection_rate",
    )
    baseline = {
        _int(row.get("scenario_seed"), -1): row
        for row in scenario_rows
        if _text(row.get("branch")) == BASELINE_BRANCH
    }
    full = {
        _int(row.get("scenario_seed"), -1): row
        for row in scenario_rows
        if _text(row.get("branch")) == FULL_BRANCH
    }
    result: list[dict[str, Any]] = []
    for metric in metrics:
        differences = [
            _finite_or(full[seed].get(metric), 0.0)
            - _finite_or(baseline[seed].get(metric), 0.0)
            for seed in config.scenario_seeds
            if int(seed) in baseline and int(seed) in full
        ]
        # Do not use Python's process-randomized ``hash`` here: audit output
        # must be byte-for-byte reproducible across separate PyCharm runs.
        metric_index = metrics.index(metric)
        ci_low, ci_high = _bootstrap_ci(
            differences,
            config.bootstrap_samples,
            config.bootstrap_seed + 10_007 * metric_index,
        )
        result.append(
            {
                "metric": metric,
                "comparison": f"{FULL_BRANCH} - {BASELINE_BRANCH}",
                "mean_difference": float(np.mean(differences)) if differences else 0.0,
                "ci95_lower": float(ci_low),
                "ci95_upper": float(ci_high),
                "n_scenarios": len(differences),
                "independent_unit": "scenario_seed",
            }
        )
    return result


def _finite_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        numeric_keys = (
            "time_s",
            "actor_obstacle_feature_min_distance_m",
            "dynamic_clearance_change_m",
            "h1_predicted_dynamic_min_clearance_m",
            "h2_predicted_dynamic_min_clearance_m",
            "h3_predicted_dynamic_min_clearance_m",
        )
        if "pre_dynamic_actual_clearance_m" in row:
            numeric_keys += (
                "pre_dynamic_actual_clearance_m",
                "post_dynamic_actual_clearance_m",
            )
        else:
            numeric_keys += (
                "pre_dynamic_actual_minimum_clearance_m",
                "post_dynamic_actual_minimum_clearance_m",
            )
        for key in numeric_keys:
            if not _finite_value(row.get(key)):
                return False
        for key in (
            "nominal_actual_position_closure",
            "counterfactual_state_unchanged",
            "offline_geometry_only",
        ):
            if _int(row.get(key), 0) not in (0, 1):
                return False
    return True


def _event_key(row: Mapping[str, Any]) -> tuple[int, int, str, int]:
    step_value = row.get("episode_step")
    if step_value in (None, ""):
        step_value = row.get("collision_step")
    return (
        _int(row.get("policy_seed"), -1),
        _int(row.get("scenario_seed"), -1),
        _text(row.get("branch")),
        _int(step_value, -1),
    )


def _collision_keys_from_timing_events(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, int, str, int]]:
    return {_event_key(row) for row in rows}


def _plot(
    path: Path,
    events: Sequence[Mapping[str, Any]],
    envelope_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    config: SafetyObservabilityAuditConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {BASELINE_BRANCH: "#0072B2", FULL_BRANCH: "#D55E00"}
    labels = {
        BASELINE_BRANCH: "baseline MAPPO",
        FULL_BRANCH: "budgeted quality-aware MAPPO",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    if events:
        for index, row in enumerate(events, start=1):
            color = colors.get(_text(row.get("branch")), "#666666")
            ax.plot(
                [index, index],
                [
                    _finite_or(row.get("pre_dynamic_actual_clearance_m"), 0.0),
                    _finite_or(row.get("post_dynamic_actual_clearance_m"), 0.0),
                ],
                marker="o",
                color=color,
                linewidth=1.8,
            )
        ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="safety threshold")
        ax.set_xlabel("Collision event (frozen replay order)")
        ax.set_ylabel("Clearance (m)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No collision events", ha="center", va="center")
    ax.set_title("Clearance before and after obstacle update")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    cause_counts = {cause: sum(_text(row.get("primary_cause")) == cause for row in events) for cause in OBSERVABILITY_CAUSES}
    shown = [(key, value) for key, value in cause_counts.items() if value]
    if shown:
        ax.bar([item[0] for item in shown], [item[1] for item in shown], color="#56B4E9")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    else:
        ax.text(0.5, 0.5, "No classified collisions", ha="center", va="center")
    ax.set_ylabel("Collision-event count")
    ax.set_title("Primary observability cause")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 2]
    for row in events:
        color = colors.get(_text(row.get("branch")), "#666666")
        values = [
            _finite_or(row.get("pre_dynamic_actual_clearance_m"), 0.0),
            _finite_or(row.get("h1_predicted_dynamic_min_clearance_m"), 0.0),
            _finite_or(row.get("h2_predicted_dynamic_min_clearance_m"), 0.0),
            _finite_or(row.get("h3_predicted_dynamic_min_clearance_m"), 0.0),
        ]
        ax.plot([0, 1, 2, 3], values, marker="o", alpha=0.65, color=color)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1, 2, 3], ["pre", "1-step", "2-step", "3-step"])
    ax.set_ylabel("Dynamic-obstacle clearance (m)")
    ax.set_title("Obstacle-only prediction horizon")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    x = np.arange(len(config.scenario_seeds))
    width = 0.36
    for offset, branch in enumerate(config.branches):
        values = [
            _finite_or(
                next(
                    (
                        row.get("scenario_collision_rate")
                        for row in scenario_rows
                        if _int(row.get("scenario_seed"), -1) == int(seed)
                        and _text(row.get("branch")) == branch
                    ),
                    0.0,
                ),
            )
            for seed in config.scenario_seeds
        ]
        ax.bar(x + (offset - 0.5) * width, values, width, label=labels[branch], color=colors[branch])
    ax.set_xticks(x, [str(seed) for seed in config.scenario_seeds], rotation=45, ha="right")
    ax.set_ylabel("Collision fraction within scenario")
    ax.set_title("Scenario-level collision rate")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    bars = [
        "actor_obstacle_observable_rate",
        "actor_motion_feature_rate",
        "actual_projection_rate",
    ]
    x2 = np.arange(len(bars))
    for offset, branch in enumerate(config.branches):
        values = [
            float(np.mean([
                _finite_or(row.get(metric), 0.0)
                for row in scenario_rows
                if _text(row.get("branch")) == branch
            ]))
            if any(_text(row.get("branch")) == branch for row in scenario_rows)
            else 0.0
            for metric in bars
        ]
        ax.bar(x2 + (offset - 0.5) * width, values, width, label=labels[branch], color=colors[branch])
    ax.set_xticks(x2, ["aggregate\nobstacle", "motion\nfeature", "projection\ntrigger"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Observation and projection coverage")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 2]
    for branch in config.branches:
        values = [
            _finite_or(row.get("dynamic_clearance_change_m"), 0.0)
            for row in envelope_rows
            if _text(row.get("branch")) == branch
        ]
        if values:
            ax.hist(values, bins=min(20, max(5, len(values) // 20)), alpha=0.45, color=colors[branch], label=labels[branch])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Post-update − pre-update clearance (m)")
    ax.set_ylabel("Decision-window count")
    ax.set_title("Dynamic-obstacle clearance change")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle("Dynamic-obstacle observability and safety-envelope audit", fontsize=14)
    fig.text(
        0.5,
        0.005,
        "Frozen validation replay; geometry is offline only; scenario seed is the independent unit; horizons freeze the current UAV endpoint.",
        ha="center",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_observability_audit(
    config: SafetyObservabilityAuditConfig,
) -> dict[str, Any]:
    """Audit actor observability and dynamic safety-envelope coverage."""

    output, plots = _prepare_outputs(config)
    source = Path(config.source_timing_audit_dir).resolve()
    step_rows_all = _read_csv(source / "timing_step_diagnostics.csv")
    timing_events_all = _read_csv(source / "timing_counterfactual_events.csv")
    timing_report_path = source / "timing_audit_report.json"
    timing_report_available = timing_report_path.exists()
    timing_report = _read_json(timing_report_path) if timing_report_available else {}
    requested_rows = [
        row
        for row in step_rows_all
        if _int(row.get("policy_seed"), -1) in {int(seed) for seed in config.policy_seeds}
        and _int(row.get("scenario_seed"), -1) in {int(seed) for seed in config.scenario_seeds}
        and _text(row.get("branch")) in set(config.branches)
    ]
    _progress(config, f"loaded {len(requested_rows)} frozen timing rows")
    groups = _unique_groups(requested_rows)
    envelope_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    for index, group in enumerate(groups.values(), start=1):
        context = _horizon_context(group, horizons=config.prediction_horizons)
        envelope_rows.append(_envelope_row(group, context))
        event = _collision_event_from_group(group, context)
        if event is not None:
            event_rows.append(event)
        if config.progress and (index == 1 or index % 1000 == 0):
            _progress(config, f"processed {index}/{len(groups)} decision windows")
    envelope_rows.sort(key=lambda row: (_int(row.get("policy_seed"), -1), _text(row.get("branch")), _int(row.get("scenario_seed"), -1), _int(row.get("episode_step"), -1)))
    event_rows.sort(key=lambda row: (_int(row.get("policy_seed"), -1), _text(row.get("branch")), _int(row.get("scenario_seed"), -1), _int(row.get("episode_step"), -1)))
    scenario_rows = _scenario_summary(envelope_rows, event_rows, config)
    paired_rows = _paired_contrasts(scenario_rows, config)

    requested_keys = {
        (int(policy), int(scenario), str(branch))
        for policy in config.policy_seeds
        for scenario in config.scenario_seeds
        for branch in config.branches
    }
    observed_episode_keys = {
        (_int(row.get("policy_seed"), -1), _int(row.get("scenario_seed"), -1), _text(row.get("branch")))
        for row in envelope_rows
    }
    # Each frozen timing group contains one row for every UAV and one episode
    # can terminate early.  Episode coverage is therefore checked by the
    # requested policy/scenario/branch key, not by assuming max_steps rows.
    episode_key_coverage = observed_episode_keys == requested_keys
    collision_keys = {_event_key(row) for row in event_rows}
    source_collision_keys = _collision_keys_from_timing_events(
        [
            row
            for row in timing_events_all
            if _int(row.get("policy_seed"), -1) in {int(seed) for seed in config.policy_seeds}
            and _int(row.get("scenario_seed"), -1) in {int(seed) for seed in config.scenario_seeds}
            and _text(row.get("branch")) in set(config.branches)
        ]
    )
    collision_replay_match = collision_keys == source_collision_keys
    scenario_hashes = {
        (_int(row.get("scenario_seed"), -1), _text(row.get("branch"))): _text(row.get("scenario_hash"))
        for row in envelope_rows
        if _text(row.get("scenario_hash"))
    }
    same_scenario_hash = all(
        _text(row.get("scenario_hash")) == scenario_hashes.get((_int(row.get("scenario_seed"), -1), _text(row.get("branch"))), "")
        for row in envelope_rows
    )
    actor_truth_free = all(_int(row.get("actor_observation_truth_free"), 0) == 1 for row in requested_rows)
    source_design = timing_report.get("design", {})
    trainer_update_zero = not bool(source_design.get("trainer_update_called", False))
    if any("trainer_update_called" in row for row in requested_rows):
        trainer_update_zero = trainer_update_zero and all(
            _int(row.get("trainer_update_called"), 1) == 0
            for row in requested_rows
        )
    source_state_unchanged = all(
        _int(row.get("counterfactual_state_unchanged"), 0) == 1
        and _int(row.get("offline_geometry_only"), 0) == 1
        for row in requested_rows
    )
    closure = all(_int(row.get("nominal_actual_position_closure"), 0) == 1 for row in envelope_rows)
    finite = _finite_rows(envelope_rows) and _finite_rows(event_rows)
    termination = all(
        not _int(row.get("actual_collision"), 0)
        or (_int(row.get("terminated"), 0) == 1 and _text(row.get("termination_reason")) == "collision")
        for row in requested_rows
    )
    task_complete_false = all(_int(row.get("task_complete"), 0) == 0 for row in requested_rows)
    source_report_ok = timing_report_available and bool(timing_report.get("audit_ok", False))
    test_data_read_zero = True
    no_source_overwrite = not any(
        _paths_overlap(path.resolve(), source_path)
        for path in (Path(config.output_dir).resolve(), Path(config.plots_dir).resolve())
        for source_path in (
            Path(config.model_root).resolve(),
            Path(config.scenario_root).resolve(),
            source,
        )
    )
    checks = {
        "model_root_available": Path(config.model_root).exists(),
        "scenario_root_available": Path(config.scenario_root).exists(),
        "source_timing_replay_available": bool(requested_rows),
        "source_timing_report_available": timing_report_available,
        "source_timing_report_ok": source_report_ok,
        "episode_key_coverage": episode_key_coverage,
        "same_scenario_hash": same_scenario_hash,
        "current_collision_events_reproduced": collision_replay_match,
        "actor_observations_truth_free": actor_truth_free,
        "trainer_update_called_zero": trainer_update_zero,
        "test_data_read_zero": test_data_read_zero,
        "counterfactual_state_unchanged": source_state_unchanged,
        "nominal_actual_position_closure": closure,
        "collision_termination": termination,
        "collision_task_complete_false": task_complete_false,
        "finite_diagnostics": finite,
        "no_source_overwrite": no_source_overwrite,
        "scenario_independent_units": True,
    }
    if len(event_rows) != len(source_collision_keys):
        audit_errors.append(
            f"collision event count mismatch: derived={len(event_rows)} source={len(source_collision_keys)}"
        )
    cause_counts = {cause: sum(_text(row.get("primary_cause")) == cause for row in event_rows) for cause in OBSERVABILITY_CAUSES}
    if event_rows:
        primary_cause = max(
            OBSERVABILITY_CAUSES,
            key=lambda cause: (cause_counts.get(cause, 0), -OBSERVABILITY_CAUSES.index(cause)),
        )
    else:
        primary_cause = "unclassified"
    # A useful secondary check: all current collision events have a dynamic
    # obstacle record and the one-step/three-step calculations are finite.
    dynamic_collision_events = [
        row for row in event_rows if _text(row.get("collision_obstacle_type")) == "dynamic_vehicle"
    ]
    prediction_horizon_status = (
        "prediction_horizon_insufficient"
        if any(
            not _int(row.get("h1_predicted_dynamic_collision"), 0)
            and (
                _int(row.get("h2_predicted_dynamic_collision"), 0)
                or _int(row.get("h3_predicted_dynamic_collision"), 0)
            )
            for row in event_rows
        )
        else "not_triggered_by_horizon_evidence"
    )
    projection_status = (
        "projection_coverage_insufficient"
        if any(_int(row.get("h1_predicted_projection_collision"), 0) for row in event_rows)
        else "projection_avoids_current_event"
    )
    audit_ok = bool(requested_rows and all(checks.values()) and not audit_errors)
    report = {
        "schema_version": _SCHEMA_VERSION,
        "audit_ok": audit_ok,
        "decision": primary_cause,
        "primary_cause": primary_cause,
        "cause_counts": cause_counts,
        "design": {
            "uav_count": int(config.uav_count),
            "max_steps": int(config.max_steps),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "branches": list(config.branches),
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "mean within scenario; no event-level inference",
            "deterministic_inference": True,
            "trainer_update_called": False,
            "test_data_read": False,
            "actual_update_order": "motion -> current-obstacle projection -> dynamic_obstacle_update -> collision_check",
            "projection_geometry": "current obstacle segment intersection and endpoint clearance; no swept envelope",
            "prediction_horizons": [1, 2, 3],
            "prediction_horizon_interpretation": "obstacle-only prediction with the current UAV endpoint frozen; diagnostic only",
            "censoring_policy": "collision episodes terminate early; safety and energy claims are not inferred from censored length",
        },
        "source": {
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "source_timing_audit_dir": str(source),
            "source_type": "frozen_timing_counterfactual_replay",
            "source_episode_schema": "episode-v1.3",
            "replay_data_reused": True,
            "test_data_role": "sealed; no test paths or files were read",
        },
        "counts": {
            "decision_windows": len(envelope_rows),
            "observability_events": len(event_rows),
            "dynamic_vehicle_collision_events": len(dynamic_collision_events),
            "scenario_summary_rows": len(scenario_rows),
            "paired_contrast_rows": len(paired_rows),
            "policy_seed_count": len(config.policy_seeds),
            "scenario_seed_count": len(config.scenario_seeds),
            "branch_count": len(config.branches),
        },
        "checks": checks,
        "mechanism_findings": {
            "actor_observability": {
                "aggregate_obstacle_feature_is_available": int(
                    all(_int(row.get("actor_obstacle_feature_finite"), 0) for row in envelope_rows)
                ),
                "collision_source_is_identifiable": int(
                    any(_int(row.get("collision_source_identifiable_from_actor_observation"), 0) for row in event_rows)
                ),
                "motion_feature_collision_count": cause_counts.get("observable_but_no_motion_feature", 0),
                "aggregate_semantics": "minimum distance plus obstacle type counts and timestep; source-specific identity unavailable",
            },
            "safety_envelope": {
                "current_projection_uses_swept_envelope": False,
                "one_step_shadow_collision_count": sum(
                    _int(row.get("h1_predicted_projection_collision"), 0) for row in event_rows
                ),
                "two_step_predicted_collision_count": sum(
                    _int(row.get("h2_predicted_dynamic_collision"), 0) for row in event_rows
                ),
                "three_step_predicted_collision_count": sum(
                    _int(row.get("h3_predicted_dynamic_collision"), 0) for row in event_rows
                ),
                "prediction_horizon_status": prediction_horizon_status,
                "projection_status": projection_status,
            },
            "dynamic_obstacle_timing": {
                "collision_events_after_dynamic_update": sum(
                    _int(row.get("dynamic_obstacle_moved"), 0)
                    and not _int(row.get("pre_dynamic_collision"), 0)
                    and _int(row.get("post_dynamic_collision"), 0)
                    for row in event_rows
                ),
                "update_order": "motion -> obstacle_update -> collision_check",
            },
            "censoring": {
                "collision_censored_event_count": len(event_rows),
                "energy_path_rmse_not_used_as_safety_evidence": True,
            },
        },
        "audit_errors": audit_errors,
        "interpretation": {
            "claim_boundary": "Frozen validation mechanism audit only; no new training result or algorithm superiority claim.",
            "primary_cause_semantics": "The cause is a diagnostic classification, not an actor input or a simulator branch.",
            "offline_truth_boundary": "Obstacle geometry and motion are read only from the frozen replay for offline diagnosis; they are never fed to actor, critic, reward, or actual control.",
            "next_fix_gate": "Choose at most one correction after reviewing this audit; do not change core models in this audit.",
        },
    }

    _write_csv(output / "observability_events.csv", event_rows)
    _write_csv(output / "safety_envelope_diagnostics.csv", envelope_rows)
    _write_csv(output / "scenario_summary.csv", scenario_rows)
    _write_csv(output / "paired_contrasts.csv", paired_rows)
    _write_json(output / "audit_report.json", report)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "source_timing_audit_dir": str(source),
            "output_dir": str(output),
            "plots_dir": str(plots),
        },
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "output_dir": str(output),
            "plots_dir": str(plots),
            "audit_report": str((output / "audit_report.json").resolve()),
            "source_episode_schema": "episode-v1.3",
            "validation_only": True,
            "test_data_read": False,
            "source_type": "frozen_timing_counterfactual_replay",
            "prediction_horizons": [1, 2, 3],
        },
    )
    plot_path = plots / _PLOT_FILE
    _plot(plot_path, event_rows, envelope_rows, scenario_rows, paired_rows, config)
    _progress(config, f"audit complete: audit_ok={audit_ok}, primary_cause={primary_cause}")
    return {
        "audit_ok": audit_ok,
        "decision": primary_cause,
        "primary_cause": primary_cause,
        "observability_events": len(event_rows),
        "decision_windows": len(envelope_rows),
        "logs": str(output),
        "plot": str(plot_path),
        "audit_report": str(output / "audit_report.json"),
    }


__all__ = [
    "OBSERVABILITY_CAUSES",
    "SafetyObservabilityAuditConfig",
    "run_safety_observability_audit",
]
