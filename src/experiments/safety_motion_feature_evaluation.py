"""Frozen single-policy evaluation of the 41-dimensional safety observation.

This experiment is deliberately diagnostic.  It evaluates the two fresh
single-seed smoke checkpoints on the same frozen safety scenarios without
calling a trainer update.  The independent units are the ten scenario seeds;
the result is not a multi-seed algorithm claim.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..simulation.obstacle_model import (
    collision_event,
    point_obstacle_distance,
    segment_intersects_obstacle,
)
from ..synthetic.episode_generator import _jsonable
from .formal import _make_environment
from .policy_mechanism_audit import _formal_config, _load_trainer
from .safety_mappo_evaluation import (
    BASELINE_BRANCH,
    FULL_BRANCH,
    DEFAULT_SCENARIO_SEEDS,
    _action_metrics,
    _actor_observation_truth_free,
    _bootstrap_ci,
    _evaluation_seed,
    _load_scenario,
    _measurement_metrics,
    _positions_metrics,
    _state_and_covariance_finite,
)


BRANCHES = (BASELINE_BRANCH, FULL_BRANCH)
EXPECTED_NODE_FEATURE_DIM = 41
EXPECTED_DYNAMIC_FEATURE_FLAG = True
EXPECTED_OVERHEAD_FLAG = True


@dataclass(frozen=True)
class SafetyMotionFeatureEvaluationConfig:
    """Configuration for the single-seed 41-dimensional safety evaluation."""

    checkpoint_root: Path
    scenario_root: Path
    output_dir: Path = Path("data/synthetic/safety_motion_feature_evaluation_seed2026")
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    branches: tuple[str, ...] = BRANCHES
    policy_seeds: tuple[int, ...] = (2026,)
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    deterministic: bool = True
    safety_projection_enabled: bool = True
    measurement_overhead_enabled: bool = True
    quality_logit_penalty: float = 1.0
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True
    command_line: str = ""

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("safety motion-feature evaluation is fixed to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("safety motion-feature evaluation supports CPU only")
        if not self.deterministic:
            raise ValueError("safety motion-feature evaluation requires deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("the paired safety evaluation requires projection_on")
        if tuple(self.branches) != BRANCHES:
            raise ValueError("the diagnostic comparison requires the two smoke branches")
        if tuple(int(seed) for seed in self.policy_seeds) != (2026,):
            raise ValueError("this diagnostic evaluates the frozen seed=2026 smoke checkpoints")
        if not self.scenario_seeds:
            raise ValueError("at least one scenario seed is required")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not math.isfinite(float(self.quality_logit_penalty)) or self.quality_logit_penalty < 0.0:
            raise ValueError("quality_logit_penalty must be finite and non-negative")
        if not self.measurement_overhead_enabled:
            raise ValueError("this evaluation is fixed to measurement_overhead_enabled=True")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        column_list: list[str] = []
        for row in rows:
            for key in row:
                if str(key) not in column_list:
                    column_list.append(str(key))
    else:
        column_list = [str(column) for column in columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in column_list})


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_or(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _finite_flag(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(all(_finite_flag(item) for item in value.values()))
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return int(np.isfinite(np.asarray(value, dtype=float)).all())
        except (TypeError, ValueError):
            return int(all(_finite_flag(item) for item in value))
    if isinstance(value, (float, np.floating)):
        return int(math.isfinite(float(value)))
    return 1


def _prepare_outputs(config: SafetyMotionFeatureEvaluationConfig) -> tuple[Path, Path, Path]:
    output = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / output.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / output.name
    paths = (output, logs, plots)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output, logs, and plots directories must be distinct")
    nonempty = [path for path in paths if path.exists() and any(path.iterdir())]
    if nonempty and not config.force:
        raise FileExistsError(
            "evaluation output is not empty: "
            + ", ".join(str(path) for path in nonempty)
            + "; use --force"
        )
    if config.force:
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return output, logs, plots


def _progress(config: SafetyMotionFeatureEvaluationConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-motion-eval] {message}", flush=True)


def _extract_motion_features(local: Mapping[str, Any]) -> tuple[np.ndarray, int, int]:
    """Return the four physical-unit motion aggregates and their validity."""

    values = np.asarray(local.get("obstacle_features", []), dtype=float).reshape(-1)
    if values.size >= 8:
        selected = values[4:8].copy()
        return selected, int(np.isfinite(selected).all()), 1
    node = np.asarray(local.get("node_features", []), dtype=float)
    if node.ndim == 2 and node.shape[1] >= EXPECTED_NODE_FEATURE_DIM:
        selected = node[0, 37:41].copy()
        return selected, int(np.isfinite(selected).all()), 1
    return np.zeros(4, dtype=float), 0, 0


def _actor_node_feature_status(local: Mapping[str, Any]) -> tuple[int, int]:
    node = np.asarray(local.get("node_features", []), dtype=float)
    if node.ndim != 2 or node.shape[1] < 1:
        return 0, 0
    return int(node.shape[1]), int(node.shape[1] >= EXPECTED_NODE_FEATURE_DIM and np.isfinite(node).all())


def _is_dynamic_obstacle(obstacle: Any) -> bool:
    if not getattr(obstacle, "active", False):
        return False
    velocity = getattr(obstacle, "velocity_mps", None)
    obstacle_type = str(getattr(obstacle, "obstacle_type", "")).lower()
    if velocity is None:
        return obstacle_type not in {"building", "static", "static_obstacle"}
    velocity_array = np.asarray(velocity, dtype=float).reshape(-1)
    return bool(
        velocity_array.size == 3
        and np.isfinite(velocity_array).all()
        and (
            obstacle_type not in {"building", "static", "static_obstacle"}
            or np.linalg.norm(velocity_array) > 1e-12
        )
    )


def _dynamic_clearance(point: np.ndarray, obstacles: Iterable[Any]) -> tuple[float, int]:
    dynamic = [obstacle for obstacle in obstacles if _is_dynamic_obstacle(obstacle)]
    if not dynamic:
        return 0.0, 0
    values = [point_obstacle_distance(np.asarray(point, dtype=float), obstacle) for obstacle in dynamic]
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return (min(finite) if finite else 0.0), int(bool(finite))


def _collision_source(positions: Mapping[str, np.ndarray], obstacles: Iterable[Any], safety_distance: float) -> str:
    causes: list[str] = []
    if any(
        obstacle.active
        and point_obstacle_distance(np.asarray(position, dtype=float), obstacle) < safety_distance
        for position in positions.values()
        for obstacle in obstacles
    ):
        causes.append("obstacle")
    ids = sorted(positions)
    if any(
        float(np.linalg.norm(np.asarray(positions[left]) - np.asarray(positions[right]))) < safety_distance
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    ):
        causes.append("uav_spacing")
    return "+".join(causes) if causes else "none"


def _projection_causes(
    uav_id: str,
    previous_states: Mapping[str, Any],
    nominal_state: Any,
    previous_obstacles: Iterable[Any],
    safety_distance: float,
) -> list[str]:
    current = np.asarray(previous_states[uav_id].position_m, dtype=float)
    proposed = np.asarray(nominal_state.position_m, dtype=float)
    causes: list[str] = []
    if any(
        obstacle.active
        and (
            segment_intersects_obstacle(current, proposed, obstacle)
            or point_obstacle_distance(proposed, obstacle) < safety_distance
        )
        for obstacle in previous_obstacles
    ):
        causes.append("obstacle")
    if any(
        other_id != uav_id
        and float(np.linalg.norm(proposed - np.asarray(state.position_m, dtype=float))) < safety_distance
        for other_id, state in previous_states.items()
    ):
        causes.append("uav_spacing")
    return causes


def _load_direct_bundle(
    config: SafetyMotionFeatureEvaluationConfig,
    policy_seed: int,
    branch: str,
) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]:
    scale_root = Path(config.checkpoint_root) / branch / f"scale_{config.uav_count}"
    manifest_path = scale_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing direct Smoke manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    base_config = _formal_config(manifest, branch)
    if base_config.node_feature_dim != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError(f"{branch} is not a 41-dimensional checkpoint")
    if not base_config.dynamic_obstacle_motion_features_enabled:
        raise ValueError(f"{branch} does not enable dynamic obstacle motion features")
    if not base_config.measurement_overhead_enabled:
        raise ValueError(f"{branch} does not enable measurement overhead")
    eval_config = base_config.__class__(
        **{
            **asdict(base_config),
            "max_steps": int(config.max_steps),
            "device": config.device,
            "progress": False,
            "force": False,
            "mappo_variant": branch,
        }
    )
    environment = _make_environment(
        eval_config,
        seed=_evaluation_seed(int(policy_seed)),
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=True,
        measurement_budget_enabled=eval_config.measurement_budget_enabled,
        measurement_budget_by_uav=eval_config.measurement_budget_by_uav,
    )
    trainer, quality_model = _load_trainer(
        eval_config,
        scale_root,
        manifest,
        environment,
        seed=int(policy_seed),
    )
    trainer.policy.eval()
    metadata = {
        "scale_root": str(scale_root.resolve()),
        "manifest_node_feature_dim": int(base_config.node_feature_dim),
        "dynamic_obstacle_motion_features_enabled": int(
            base_config.dynamic_obstacle_motion_features_enabled
        ),
        "measurement_overhead_enabled": int(base_config.measurement_overhead_enabled),
        "trainer_node_feature_dim": int(getattr(trainer.config, "node_feature_dim", -1)),
        "quality_feature_enabled": int(
            getattr(trainer.config, "use_measurement_quality_features", False)
        ),
        "budget_feature_enabled": int(
            getattr(trainer.config, "use_measurement_budget_features", False)
        ),
        "measurement_quality_logit_penalty": _finite_or(
            getattr(trainer.config, "measurement_quality_logit_penalty", 0.0)
        ),
        "quality_model_source_split": str(quality_model.get("source_split", "")),
    }
    return trainer, eval_config, manifest, scale_root, metadata


def _episode(
    config: SafetyMotionFeatureEvaluationConfig,
    trainer: Any,
    eval_config: Any,
    scenario: Mapping[str, Any],
    scenario_seed: int,
    scenario_hash: str,
    policy_seed: int,
    branch: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = _make_environment(
        eval_config,
        seed=_evaluation_seed(int(scenario_seed)),
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=True,
        measurement_budget_enabled=eval_config.measurement_budget_enabled,
        measurement_budget_by_uav=eval_config.measurement_budget_by_uav,
    )
    observations, reset_info = environment.reset(
        seed=_evaluation_seed(int(scenario_seed)),
        scenario_config=scenario,
    )
    initial_positions = {
        uav_id: np.asarray(state.position_m, dtype=float).copy()
        for uav_id, state in environment.states.items()
    }
    initial_collision, initial_clearance = collision_event(
        initial_positions,
        environment.obstacles,
        environment.config.motion.safety_distance_m,
    )
    previous_states = {uav_id: state.copy() for uav_id, state in environment.states.items()}
    previous_obstacles = copy.deepcopy(environment.obstacles)
    step_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    all_position_errors: list[float] = []
    trajectory_length = 0.0
    energy_values: list[float] = []
    action_means: list[float] = []
    action_maxima: list[float] = []
    saturation_values: list[float] = []
    clearances = [float(initial_clearance)]
    pre_clearances = [float(initial_clearance)]
    projection_count = 0
    obstacle_projection_count = 0
    uav_projection_count = 0
    projection_unclassified_count = 0
    collision_episode = int(initial_collision)
    pre_projection_collision_episode = 0
    collision_step: int | None = None
    measurement_count_total = 0
    overhead_packets_total = 0.0
    finite_state = int(_finite_flag(initial_positions) and _finite_flag(initial_clearance))
    finite_covariance = 1
    truth_free = 1
    dynamic_feature_finite = 1
    dynamic_feature_present = 1
    trainer_update_called = 0

    for step in range(int(config.max_steps)):
        truth_free &= int(
            all(
                _actor_observation_truth_free(observations.get(uav_id, {}))
                for uav_id in environment.uav_ids
            )
        )
        raw_motion: dict[str, tuple[np.ndarray, int, int]] = {
            uav_id: _extract_motion_features(observations.get(uav_id, {}))
            for uav_id in environment.uav_ids
        }
        dynamic_feature_finite &= int(all(value[1] for value in raw_motion.values()))
        dynamic_feature_present &= int(all(value[2] for value in raw_motion.values()))
        decision = trainer.act(
            observations,
            reset_info.get("central_state") if reset_info else None,
            deterministic=True,
        )
        truth_free &= int(
            all(
                _actor_observation_truth_free(decision.observations.get(uav_id, {}))
                for uav_id in environment.uav_ids
            )
        )
        nominal_states = {
            uav_id: apply_velocity_increment(
                previous_states[uav_id],
                decision.actions[uav_id].get("delta_v", np.zeros(3)),
                environment.config.motion,
                safety_projector=None,
            )[0]
            for uav_id in environment.uav_ids
        }
        next_observations, rewards, terminated, truncated, info = environment.step(
            decision.actions
        )
        trainer_update_called = 0
        state_finite, covariance_finite = _state_and_covariance_finite(environment)
        finite_state &= state_finite
        finite_covariance &= covariance_finite
        rmse, p95, errors = _positions_metrics(environment)
        all_position_errors.extend(errors)
        execution = info.get("action_execution", {}) or {}
        displacements: dict[str, float] = {}
        for uav_id in environment.uav_ids:
            displacements[uav_id] = float(
                np.linalg.norm(
                    np.asarray(environment.states[uav_id].position_m, dtype=float)
                    - np.asarray(previous_states[uav_id].position_m, dtype=float)
                )
            )
        path_step = float(sum(value for value in displacements.values() if math.isfinite(value)))
        trajectory_length += path_step
        energy, action_mean, action_max, action_finite = _action_metrics(info, environment)
        finite_state &= action_finite
        energy_values.append(energy)
        action_means.append(action_mean)
        action_maxima.append(action_max)
        saturation_values.append(_finite_or(info.get("action_saturation_rate", 0.0)))
        minimum = _finite_or(info.get("minimum_clearance", 0.0))
        pre_minimum = _finite_or(info.get("pre_projection_minimum_clearance", 0.0))
        clearances.append(minimum)
        pre_clearances.append(pre_minimum)
        projection_step = int(_finite_or(info.get("safety_projection_count", 0.0)))
        projection_count += projection_step
        collision = int(bool(info.get("collision", 0)))
        pre_collision = int(bool(info.get("pre_projection_collision", 0)))
        collision_episode = int(collision_episode or collision)
        pre_projection_collision_episode = int(pre_projection_collision_episode or pre_collision)
        if collision and collision_step is None:
            collision_step = int(step + 1)
        count_step, packets_step = _measurement_metrics(info)
        measurement_count_total += count_step
        overhead_packets_total += packets_step
        objective = dict(info.get("objective_components", {}) or {})
        metrics = dict(info.get("global_metrics", {}) or {})
        team_reward = _finite_or(
            info.get("team_reward", np.mean(list(rewards.values())) if rewards else 0.0)
        )
        actual_positions = {
            uav_id: np.asarray(state.position_m, dtype=float).copy()
            for uav_id, state in environment.states.items()
        }
        collision_source = _collision_source(
            actual_positions,
            environment.obstacles,
            environment.config.motion.safety_distance_m,
        )
        next_motion = {
            uav_id: _extract_motion_features(next_observations.get(uav_id, {}))
            for uav_id in environment.uav_ids
        }
        for uav_id in environment.uav_ids:
            execution_row = execution.get(uav_id, {}) or {}
            cause = _projection_causes(
                uav_id,
                previous_states,
                nominal_states[uav_id],
                previous_obstacles,
                environment.config.motion.safety_distance_m,
            )
            safety_projected = int(bool(execution_row.get("safety_projected", False)))
            boundary_projected = int(bool(execution_row.get("boundary_projected", False)))
            if safety_projected:
                if "obstacle" in cause:
                    obstacle_projection_count += 1
                if "uav_spacing" in cause:
                    uav_projection_count += 1
                if not cause:
                    projection_unclassified_count += 1
            actor_dim, actor_motion_present = _actor_node_feature_status(
                decision.observations.get(uav_id, {})
            )
            before_dynamic_clearance, before_dynamic_valid = _dynamic_clearance(
                actual_positions[uav_id], previous_obstacles
            )
            after_dynamic_clearance, after_dynamic_valid = _dynamic_clearance(
                actual_positions[uav_id], environment.obstacles
            )
            commanded = np.asarray(
                execution_row.get("delta_v_commanded", np.zeros(3)), dtype=float
            ).reshape(3)
            applied = np.asarray(
                execution_row.get("delta_v_applied", np.zeros(3)), dtype=float
            ).reshape(3)
            projection_delta = applied - commanded
            step_rows.append(
                {
                    "policy_seed": int(policy_seed),
                    "scenario_seed": int(scenario_seed),
                    "branch": branch,
                    "uav_id": uav_id,
                    "step": int(step + 1),
                    "time_s": float((step + 1) * environment.config.motion.dt_s),
                    "scenario_hash": scenario_hash,
                    "dynamic_min_clearance_m": float(raw_motion[uav_id][0][0]),
                    "dynamic_closing_rate_mps": float(raw_motion[uav_id][0][1]),
                    "dynamic_predicted_clearance_m": float(raw_motion[uav_id][0][2]),
                    "dynamic_motion_valid": int(raw_motion[uav_id][0][3]),
                    "next_dynamic_min_clearance_m": float(next_motion[uav_id][0][0]),
                    "next_dynamic_closing_rate_mps": float(next_motion[uav_id][0][1]),
                    "next_dynamic_predicted_clearance_m": float(next_motion[uav_id][0][2]),
                    "next_dynamic_motion_valid": int(next_motion[uav_id][0][3]),
                    "actor_node_feature_dim": int(actor_dim),
                    "actor_motion_feature_present": int(actor_motion_present),
                    "actor_observation_truth_free": int(truth_free),
                    "delta_v_policy": np.asarray(
                        decision.actions[uav_id].get("delta_v", np.zeros(3)), dtype=float
                    ),
                    "delta_v_commanded": commanded,
                    "delta_v_applied": applied,
                    "delta_v_projection": projection_delta,
                    "projection_delta_norm": float(np.linalg.norm(projection_delta)),
                    "safety_projected": safety_projected,
                    "boundary_projected": boundary_projected,
                    "projection_cause": "+".join(cause) if cause else "none",
                    "projection_pre_minimum_clearance_m": float(
                        collision_event(
                            {
                                key: value.position_m
                                for key, value in nominal_states.items()
                            },
                            previous_obstacles,
                            environment.config.motion.safety_distance_m,
                        )[1]
                    ),
                    "minimum_clearance_m": minimum,
                    "pre_projection_minimum_clearance_m": pre_minimum,
                    "dynamic_clearance_before_update_m": float(before_dynamic_clearance),
                    "dynamic_clearance_after_update_m": float(after_dynamic_clearance),
                    "dynamic_clearance_drop_m": float(
                        before_dynamic_clearance - after_dynamic_clearance
                    ),
                    "dynamic_clearance_before_valid": int(before_dynamic_valid),
                    "dynamic_clearance_after_valid": int(after_dynamic_valid),
                    "collision": collision,
                    "pre_projection_collision": pre_collision,
                    "collision_source": collision_source,
                    "terminated": int(bool(terminated)),
                    "truncated": int(bool(truncated)),
                    "termination_reason": str(info.get("termination_reason", "running")),
                    "action_saturation_rate": _finite_or(
                        info.get("action_saturation_rate", 0.0)
                    ),
                    "energy_proxy_step": float(energy),
                    "path_length_step_m": float(displacements[uav_id]),
                    "measurement_count_step": int(count_step),
                    "measurement_overhead_packets_step": float(packets_step),
                    "j_loc": _finite_or(objective.get("j_loc", 0.0)),
                    "j_com": _finite_or(objective.get("j_com", 0.0)),
                    "joint_objective": _finite_or(
                        objective.get("joint_objective", 0.0)
                    ),
                    "team_reward": team_reward,
                    "packet_loss": _finite_or(metrics.get("packet_loss", 0.0)),
                    "delay_s": _finite_or(metrics.get("delay", 0.0)),
                    "aoi_s": _finite_or(metrics.get("aoi", 0.0)),
                    "connectivity": _finite_or(metrics.get("connectivity", 0.0)),
                    "position_rmse_m": rmse,
                    "position_error_p95_m": p95,
                    "state_finite": int(state_finite),
                    "covariance_finite": int(covariance_finite),
                    "trainer_update_called": int(trainer_update_called),
                    "measurement_overhead_enabled": int(
                        bool(info.get("measurement_overhead_enabled", False))
                    ),
                    "dynamic_obstacle_update_order": (
                        "motion_projection_then_dynamic_update_then_collision_check"
                    ),
                }
            )
        aggregate_rows.append(
            {
                "step": int(step + 1),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "termination_reason": str(info.get("termination_reason", "running")),
                "collision": collision,
                "pre_projection_collision": pre_collision,
                "minimum_clearance_m": minimum,
                "pre_projection_minimum_clearance_m": pre_minimum,
                "safety_projection_count": projection_step,
                "action_saturation_rate": _finite_or(
                    info.get("action_saturation_rate", 0.0)
                ),
                "energy_proxy_step": float(energy),
                "path_length_step_m": path_step,
                "measurement_count_step": int(count_step),
                "measurement_overhead_packets_step": float(packets_step),
                "j_loc": _finite_or(objective.get("j_loc", 0.0)),
                "j_com": _finite_or(objective.get("j_com", 0.0)),
                "joint_objective": _finite_or(objective.get("joint_objective", 0.0)),
                "team_reward": team_reward,
                "packet_loss": _finite_or(metrics.get("packet_loss", 0.0)),
                "delay_s": _finite_or(metrics.get("delay", 0.0)),
                "aoi_s": _finite_or(metrics.get("aoi", 0.0)),
                "connectivity": _finite_or(metrics.get("connectivity", 0.0)),
                "position_rmse_m": rmse,
                "position_error_p95_m": p95,
                "measurement_overhead_enabled": int(
                    bool(info.get("measurement_overhead_enabled", False))
                ),
            }
        )
        previous_states = {uav_id: state.copy() for uav_id, state in environment.states.items()}
        previous_obstacles = copy.deepcopy(environment.obstacles)
        reset_info = info
        observations = next_observations
        if terminated or truncated:
            break

    final_reason = str(aggregate_rows[-1]["termination_reason"] if aggregate_rows and "termination_reason" in aggregate_rows[-1] else ("collision" if collision_episode else "time_limit"))
    summary = {
        "policy_seed": int(policy_seed),
        "scenario_seed": int(scenario_seed),
        "branch": branch,
        "evaluation_seed": int(_evaluation_seed(scenario_seed)),
        "scenario_hash": scenario_hash,
        "episode_steps": int(len(aggregate_rows)),
        "initial_collision": int(initial_collision),
        "initial_minimum_clearance_m": float(initial_clearance),
        "collision_episode": int(collision_episode),
        "pre_projection_collision_episode": int(pre_projection_collision_episode),
        "collision_step": int(collision_step) if collision_step is not None else -1,
        "minimum_clearance_m": float(min(clearances)),
        "pre_projection_minimum_clearance_m": float(min(pre_clearances)),
        "safety_projection_count": int(projection_count),
        "obstacle_projection_count": int(obstacle_projection_count),
        "uav_spacing_projection_count": int(uav_projection_count),
        "projection_unclassified_count": int(projection_unclassified_count),
        "action_saturation_rate": float(np.mean(saturation_values)) if saturation_values else 0.0,
        "action_magnitude_mean": float(np.mean(action_means)) if action_means else 0.0,
        "action_magnitude_max": float(max(action_maxima, default=0.0)),
        "energy_proxy": float(np.mean(energy_values)) if energy_values else 0.0,
        "trajectory_length_m": float(trajectory_length),
        "position_rmse_m": float(
            np.sqrt(np.mean(np.square(all_position_errors))) if all_position_errors else 0.0
        ),
        "position_error_p95_m": float(
            np.percentile(all_position_errors, 95.0) if all_position_errors else 0.0
        ),
        "j_loc": float(np.mean([row["j_loc"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "j_com": float(np.mean([row["j_com"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "joint_objective": float(
            np.mean([row["joint_objective"] for row in aggregate_rows])
        ) if aggregate_rows else 0.0,
        "mean_reward": float(np.mean([row["team_reward"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "packet_loss": float(np.mean([row["packet_loss"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "delay_s": float(np.mean([row["delay_s"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "aoi_s": float(np.mean([row["aoi_s"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "connectivity": float(np.mean([row["connectivity"] for row in aggregate_rows])) if aggregate_rows else 0.0,
        "measurement_count_total": int(measurement_count_total),
        "measurement_overhead_packets_total": float(overhead_packets_total),
        "dynamic_motion_feature_valid_rate": float(
            np.mean([row["dynamic_motion_valid"] for row in step_rows])
        ) if step_rows else 0.0,
        "dynamic_motion_feature_present": int(dynamic_feature_present),
        "dynamic_motion_feature_finite": int(dynamic_feature_finite),
        "actor_node_feature_dim_41": int(
            bool(step_rows) and all(int(row["actor_node_feature_dim"]) == EXPECTED_NODE_FEATURE_DIM for row in step_rows)
        ),
        "actor_motion_feature_present": int(
            bool(step_rows) and all(int(row["actor_motion_feature_present"]) for row in step_rows)
        ),
        "termination_reason": final_reason,
        "terminated": int(bool(aggregate_rows and aggregate_rows[-1]["terminated"])),
        "truncated": int(bool(aggregate_rows and aggregate_rows[-1]["truncated"])),
        "task_complete": 0,
        "collision_censored": int(bool(collision_episode and aggregate_rows and aggregate_rows[-1]["terminated"])),
        "actor_observation_truth_free": int(truth_free),
        "trainer_update_called": int(trainer_update_called),
        "state_finite": int(finite_state),
        "covariance_finite": int(finite_covariance),
        "test_data_read": 0,
    }
    return step_rows, summary


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    return {
        key: float(np.mean([_finite_or(row.get(key, 0.0)) for row in rows])) if rows else 0.0
        for key in keys
    }


def _plot(path: Path, scenario_rows: Sequence[Mapping[str, Any]], scenario_seeds: Sequence[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("collision_episode", "Collision rate", "fraction"),
        ("minimum_clearance_m", "Minimum clearance", "m"),
        ("energy_proxy", "Energy proxy", "normalized"),
        ("trajectory_length_m", "Trajectory length", "m"),
        ("safety_projection_count", "Safety projections", "count"),
        ("position_rmse_m", "Position RMSE", "m"),
    )
    colors = {BASELINE_BRANCH: "#0072B2", FULL_BRANCH: "#D55E00"}
    labels = {
        BASELINE_BRANCH: "baseline MAPPO",
        FULL_BRANCH: "budgeted quality-aware MAPPO",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(scenario_seeds), dtype=float)
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        for branch in BRANCHES:
            values = [
                _finite_or(
                    next(
                        (
                            row.get(metric, 0.0)
                            for row in scenario_rows
                            if int(row["scenario_seed"]) == int(seed)
                            and str(row["branch"]) == branch
                        ),
                        0.0,
                    )
                )
                for seed in scenario_seeds
            ]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color=colors[branch],
                label=labels[branch],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(
        "41-dimensional dynamic-obstacle observation evaluation "
        "(single policy seed; diagnostic only)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Frozen checkpoints; deterministic CPU inference; projection on; "
        "scenario seed is the independent unit",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_motion_feature_evaluation(
    config: SafetyMotionFeatureEvaluationConfig,
) -> dict[str, Any]:
    """Evaluate the two direct Smoke checkpoints on frozen safety scenes."""

    output, logs, plots = _prepare_outputs(config)
    scenario_payloads: dict[int, tuple[dict[str, Any], str]] = {
        int(seed): _load_scenario(Path(config.scenario_root), int(seed), config.uav_count)
        for seed in config.scenario_seeds
    }
    scenario_hashes = {
        str(seed): scenario_payloads[int(seed)][1] for seed in config.scenario_seeds
    }
    bundles: dict[str, tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]] = {}
    branch_checks: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        _progress(config, f"loading branch={branch}")
        bundle = _load_direct_bundle(config, int(config.policy_seeds[0]), branch)
        bundles[branch] = bundle
        branch_checks[branch] = bundle[4]

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for branch in BRANCHES:
        trainer, eval_config, _manifest, _scale_root, _metadata = bundles[branch]
        for scenario_seed in config.scenario_seeds:
            _progress(config, f"branch={branch} scenario_seed={scenario_seed} start")
            scenario, scenario_hash = scenario_payloads[int(scenario_seed)]
            rows, summary = _episode(
                config,
                trainer,
                eval_config,
                scenario,
                int(scenario_seed),
                scenario_hash,
                int(config.policy_seeds[0]),
                branch,
            )
            step_rows.extend(rows)
            episode_rows.append(summary)
            if summary["initial_collision"]:
                errors.append(f"initial collision: {branch}/{scenario_seed}")
            if summary["collision_episode"] and not summary["terminated"]:
                errors.append(f"collision was not terminated: {branch}/{scenario_seed}")
            if summary["task_complete"]:
                errors.append(f"collision marked complete: {branch}/{scenario_seed}")

    metric_keys = (
        "collision_episode",
        "pre_projection_collision_episode",
        "minimum_clearance_m",
        "pre_projection_minimum_clearance_m",
        "safety_projection_count",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "j_com",
        "joint_objective",
        "mean_reward",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "measurement_count_total",
        "measurement_overhead_packets_total",
    )
    scenario_rows: list[dict[str, Any]] = []
    for scenario_seed in config.scenario_seeds:
        for branch in BRANCHES:
            matching = [
                row
                for row in episode_rows
                if int(row["scenario_seed"]) == int(scenario_seed)
                and str(row["branch"]) == branch
            ]
            means = _mean_rows(matching, metric_keys)
            scenario_rows.append(
                {
                    "scenario_seed": int(scenario_seed),
                    "branch": branch,
                    "policy_seed_count": int(len(matching)),
                    "scenario_hash": scenario_hashes[str(int(scenario_seed))],
                    **means,
                    "all_state_finite": int(all(int(row["state_finite"]) for row in matching)) if matching else 0,
                    "all_covariance_finite": int(all(int(row["covariance_finite"]) for row in matching)) if matching else 0,
                    "all_actor_observations_truth_free": int(
                        all(int(row["actor_observation_truth_free"]) for row in matching)
                    ) if matching else 0,
                    "all_actor_node_feature_dim_41": int(
                        all(int(row["actor_node_feature_dim_41"]) for row in matching)
                    ) if matching else 0,
                    "all_dynamic_feature_finite": int(
                        all(int(row["dynamic_motion_feature_finite"]) for row in matching)
                    ) if matching else 0,
                    "all_trainer_updates_zero": int(
                        all(int(row["trainer_update_called"]) == 0 for row in matching)
                    ) if matching else 0,
                }
            )

    contrasts: list[dict[str, Any]] = []
    baseline = {
        int(row["scenario_seed"]): row
        for row in scenario_rows
        if row["branch"] == BASELINE_BRANCH
    }
    full = {
        int(row["scenario_seed"]): row
        for row in scenario_rows
        if row["branch"] == FULL_BRANCH
    }
    for metric_index, metric in enumerate(metric_keys):
        differences = [
            _finite_or(full[seed].get(metric, 0.0))
            - _finite_or(baseline[seed].get(metric, 0.0))
            for seed in config.scenario_seeds
            if seed in baseline and seed in full
        ]
        ci_low, ci_high = _bootstrap_ci(
            differences,
            config.bootstrap_samples,
            config.bootstrap_seed + metric_index,
        )
        contrasts.append(
            {
                "comparison": f"{FULL_BRANCH} - {BASELINE_BRANCH}",
                "metric": metric,
                "independent_unit": "scenario_seed",
                "scenario_count": int(len(differences)),
                "mean_difference": float(np.mean(differences)) if differences else 0.0,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "bootstrap_samples": int(config.bootstrap_samples),
            }
        )

    def contrast(metric: str) -> dict[str, Any]:
        return next(row for row in contrasts if row["metric"] == metric)

    collision_base = float(
        np.mean([baseline[seed]["collision_episode"] for seed in baseline])
    ) if baseline else 0.0
    collision_full = float(
        np.mean([full[seed]["collision_episode"] for seed in full])
    ) if full else 0.0
    both_zero = collision_base == 0.0 and collision_full == 0.0
    collision_contrast = contrast("collision_episode")
    clearance_contrast = contrast("minimum_clearance_m")
    energy_contrast = contrast("energy_proxy")
    audit_ok = bool(
        len(episode_rows) == len(config.scenario_seeds) * len(BRANCHES)
        and len(scenario_rows) == len(config.scenario_seeds) * len(BRANCHES)
        and not errors
        and all(int(row["initial_collision"]) == 0 for row in episode_rows)
        and all(int(row["state_finite"]) == 1 for row in episode_rows)
        and all(int(row["covariance_finite"]) == 1 for row in episode_rows)
        and all(int(row["actor_observation_truth_free"]) == 1 for row in episode_rows)
        and all(int(row["actor_node_feature_dim_41"]) == 1 for row in episode_rows)
        and all(int(row["dynamic_motion_feature_finite"]) == 1 for row in episode_rows)
        and all(int(row["dynamic_motion_feature_present"]) == 1 for row in episode_rows)
        and all(int(row["trainer_update_called"]) == 0 for row in episode_rows)
        and all(int(row["test_data_read"]) == 0 for row in episode_rows)
        and all(
            str(row["scenario_hash"]) == scenario_hashes[str(int(row["scenario_seed"]))]
            for row in episode_rows
        )
        and all(
            math.isfinite(float(row["measurement_overhead_packets_total"]))
            and float(row["measurement_overhead_packets_total"]) >= 0.0
            for row in episode_rows
        )
        and all(
            int(metadata["trainer_node_feature_dim"]) == EXPECTED_NODE_FEATURE_DIM
            and int(metadata["manifest_node_feature_dim"]) == EXPECTED_NODE_FEATURE_DIM
            and int(metadata["dynamic_obstacle_motion_features_enabled"]) == 1
            and int(metadata["measurement_overhead_enabled"]) == 1
            for metadata in branch_checks.values()
        )
        and not any("test" in str(path).lower() for path in (config.checkpoint_root, config.scenario_root))
    )
    if both_zero:
        decision = "no_safety_discrimination_zero_collisions"
    elif collision_full < collision_base:
        decision = "motion_feature_single_seed_safety_signal"
    elif collision_full > collision_base:
        decision = "motion_feature_single_seed_branch_worse"
    else:
        decision = "single_seed_safety_difference_inconclusive"

    report = {
        "schema_version": "safety-motion-feature-evaluation-v1.0",
        "audit_ok": audit_ok,
        "decision_supported": False,
        "decision": decision,
        "diagnostic_only": True,
        "design": {
            "uav_count": int(config.uav_count),
            "max_steps": int(config.max_steps),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "branches": list(BRANCHES),
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "single frozen policy seed=2026",
            "deterministic_inference": True,
            "trainer_update_called": False,
            "safety_projection_enabled": True,
            "measurement_overhead_enabled": True,
            "test_data_read": False,
            "test_data_role": "sealed; no test files are opened",
        },
        "source": {
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "scenario_hashes": scenario_hashes,
            "checkpoint_layout": "direct Smoke root / branch / scale_3",
            "checkpoint_branches": list(BRANCHES),
        },
        "branch_checks": branch_checks,
        "counts": {
            "scenario_seed_count": len(config.scenario_seeds),
            "branch_count": len(BRANCHES),
            "episode_rows": len(episode_rows),
            "step_rows": len(step_rows),
            "collision_episode_count_baseline": int(
                sum(int(row["collision_episode"]) for row in episode_rows if row["branch"] == BASELINE_BRANCH)
            ),
            "collision_episode_count_full": int(
                sum(int(row["collision_episode"]) for row in episode_rows if row["branch"] == FULL_BRANCH)
            ),
        },
        "safety_summary": {
            "baseline_collision_rate": collision_base,
            "full_collision_rate": collision_full,
            "collision_difference_full_minus_baseline": collision_contrast,
            "minimum_clearance_difference_full_minus_baseline": clearance_contrast,
            "energy_difference_full_minus_baseline": energy_contrast,
        },
        "audit_errors": errors,
        "interpretation": {
            "claim_boundary": (
                "This is a single-policy-seed diagnostic of the new 41-dimensional "
                "observation. It cannot establish multi-seed algorithm superiority."
            ),
            "collision_semantics": "collision terminates the episode and task_complete remains false",
            "censoring": "energy, path length, and RMSE are censored descriptions when collision terminates early",
            "next_step_if_signal": "retrain five independent 41-dimensional policy seeds before a formal safety claim",
        },
    }
    _write_csv(logs / "episode_metrics.csv", episode_rows)
    _write_csv(logs / "step_metrics.csv", step_rows)
    _write_csv(logs / "scenario_summary.csv", scenario_rows)
    _write_csv(logs / "paired_contrasts.csv", contrasts)
    _write_json(logs / "audit_report.json", report)
    _write_json(output / "scenario_hashes.json", scenario_hashes)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "output_dir": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
        },
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "safety-motion-feature-evaluation-v1.0",
            "data_root": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
            "audit_report": str((logs / "audit_report.json").resolve()),
            "episode_schema": "diagnostic replay; source episode-v1.3 data is read-only",
            "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
            "dynamic_obstacle_motion_features_enabled": True,
            "test_data_read": False,
        },
    )
    plot_path = plots / "safety_motion_feature_comparison.png"
    _plot(plot_path, scenario_rows, config.scenario_seeds)
    return {
        "audit_ok": bool(audit_ok),
        "decision_supported": False,
        "decision": decision,
        "diagnostic_only": True,
        "independent_unit": "scenario_seed",
        "policy_seed_count": len(config.policy_seeds),
        "scenario_seed_count": len(config.scenario_seeds),
        "data": str(output),
        "logs": str(logs),
        "plot": str(plot_path),
        "audit_report": str(logs / "audit_report.json"),
    }


__all__ = [
    "BRANCHES",
    "EXPECTED_NODE_FEATURE_DIM",
    "SafetyMotionFeatureEvaluationConfig",
    "run_safety_motion_feature_evaluation",
]
