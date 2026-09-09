"""Paired safety--energy evaluation of frozen MAPPO policies.

This module evaluates already-trained policies on the frozen safety-risk
scenarios.  It intentionally does not collect rollouts for training and never
calls ``MAPPOTrainer.update``.  The independent statistical unit is the
scenario seed: the five policy seeds are averaged within each scenario before
paired contrasts and bootstrap intervals are computed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.mappo import MAPPOConfig
from ..synthetic.episode_generator import _jsonable
from ..simulation.obstacle_model import collision_event
from .formal import _make_environment
from .policy_mechanism_audit import (
    _formal_config,
    _load_quality_model,
    _load_trainer,
    _manifest_for,
)


DEFAULT_POLICY_SEEDS = (7071, 8080, 9099, 10118, 11127)
DEFAULT_SCENARIO_SEEDS = (
    2026,
    3035,
    4044,
    5053,
    6062,
    7071,
    8080,
    9089,
    10098,
    11107,
)
DEFAULT_BRANCHES = ("baseline_mappo", "budgeted_quality_aware_mappo")
SUPPORTED_BRANCHES = (
    "baseline_mappo",
    "quality_aware_mappo",
    "budgeted_quality_aware_mappo",
)
FULL_BRANCH = "budgeted_quality_aware_mappo"
BASELINE_BRANCH = "baseline_mappo"


@dataclass(frozen=True)
class SafetyMAPPOEvaluationConfig:
    """Configuration for deterministic evaluation on frozen safety scenes."""

    model_root: Path
    scenario_root: Path
    output_dir: Path
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    deterministic: bool = True
    safety_projection_enabled: bool = True
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("safety MAPPO evaluation is restricted to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("deterministic safety evaluation currently supports device='cpu' only")
        if not self.deterministic:
            raise ValueError("safety MAPPO evaluation must use deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("algorithm comparison must use safety_projection_enabled=True")
        if not self.policy_seeds or not self.scenario_seeds:
            raise ValueError("at least one policy seed and scenario seed are required")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        unknown = sorted(set(self.branches) - set(SUPPORTED_BRANCHES))
        if unknown:
            raise ValueError("unsupported branch(es): " + ", ".join(unknown))
        if BASELINE_BRANCH not in self.branches or FULL_BRANCH not in self.branches:
            raise ValueError(
                "branches must include baseline_mappo and budgeted_quality_aware_mappo"
            )
        if len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


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


def _finite_or(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_outputs(config: SafetyMAPPOEvaluationConfig) -> tuple[Path, Path, Path]:
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
            "safety MAPPO evaluation output is not empty: "
            + ", ".join(str(path) for path in nonempty)
            + "; use --force to replace these exact directories"
        )
    if config.force:
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return output, logs, plots


def _progress(config: SafetyMAPPOEvaluationConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-mappo] {message}", flush=True)


def _load_scenario(scenario_root: Path, scenario_seed: int, uav_count: int) -> tuple[dict[str, Any], str]:
    path = Path(scenario_root) / f"scenario_{int(scenario_seed)}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen safety scenario: {path}")
    scenario = _read_json(path)
    if int(scenario.get("seed", -1)) != int(scenario_seed):
        raise ValueError(f"scenario seed mismatch in {path}")
    if int(scenario.get("uav_count", -1)) != int(uav_count):
        raise ValueError(f"scenario UAV count mismatch in {path}")
    return scenario, _canonical_hash(scenario)


def _evaluation_seed(scenario_seed: int) -> int:
    """Map each frozen scenario to a reproducible environment RNG seed."""

    return int(1_000_000 + int(scenario_seed))


def _load_bundle(
    config: SafetyMAPPOEvaluationConfig,
    policy_seed: int,
    branch: str,
) -> tuple[Any, Any, dict[str, Any], Path]:
    scale_root, manifest = _manifest_for(Path(config.model_root), int(policy_seed), branch)
    base_config = _formal_config(manifest, branch)
    eval_config = replace(
        base_config,
        max_steps=int(config.max_steps),
        device=config.device,
        progress=False,
        force=False,
        mappo_variant=branch,
    )
    # Build a compatible environment only to derive dimensions for the frozen
    # trainer.  No episode is advanced by this dummy environment.
    environment = _make_environment(
        eval_config,
        seed=_evaluation_seed(int(policy_seed)),
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=eval_config.measurement_overhead_enabled,
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
    return trainer, eval_config, manifest, scale_root


def _actor_observation_truth_free(value: Any) -> bool:
    """Check local actor observations for truth/future labels by field name."""

    forbidden = (
        "ground_truth",
        "true_range",
        "true_position",
        "position_error",
        "future",
        "los_truth",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in forbidden):
                return False
            if not _actor_observation_truth_free(item):
                return False
    elif isinstance(value, (list, tuple)):
        return all(_actor_observation_truth_free(item) for item in value)
    return True


def _state_and_covariance_finite(environment: Any) -> tuple[int, int]:
    state_finite = 1
    covariance_finite = 1
    for state in environment.states.values():
        state_finite &= _finite_flag(state.position_m)
        state_finite &= _finite_flag(state.velocity_mps)
        state_finite &= _finite_flag(state.attitude_rad)
    for ekf in environment.ekfs.values():
        state_finite &= _finite_flag(ekf.state)
        covariance_finite &= _finite_flag(ekf.covariance)
    return int(state_finite), int(covariance_finite)


def _positions_metrics(environment: Any) -> tuple[float, float, list[float]]:
    errors: list[float] = []
    for uav_id in environment.uav_ids:
        truth = np.asarray(environment.states[uav_id].position_m, dtype=float)
        estimate = np.asarray(environment.ekfs[uav_id].position, dtype=float)
        error = float(np.linalg.norm(truth - estimate))
        errors.append(error if math.isfinite(error) else 0.0)
    if not errors:
        return 0.0, 0.0, []
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return rmse, float(np.percentile(errors, 95.0)), errors


def _action_metrics(info: Mapping[str, Any], environment: Any) -> tuple[float, float, float, int]:
    execution = info.get("action_execution", {})
    norms: list[float] = []
    for uav_id in environment.uav_ids:
        delta = np.asarray(
            execution.get(uav_id, {}).get("delta_v_applied", np.zeros(3)), dtype=float
        ).reshape(-1)
        norm = float(np.linalg.norm(delta)) if delta.size else 0.0
        norms.append(norm if math.isfinite(norm) else 0.0)
    denominator = max(
        1e-12,
        float(environment.config.motion.max_acceleration_mps2)
        * float(environment.config.motion.dt_s),
    )
    energy = float(np.mean([norm / denominator for norm in norms])) if norms else 0.0
    return (
        energy,
        float(np.mean(norms)) if norms else 0.0,
        float(max(norms, default=0.0)),
        int(_finite_flag(norms)),
    )


def _measurement_metrics(info: Mapping[str, Any]) -> tuple[int, float]:
    events = list(info.get("measurement_overhead_events", []) or [])
    measurement_count = len(events)
    packets = sum(_finite_or(event.get("load_packets_applied", 0.0)) for event in events)
    if measurement_count == 0:
        actions = info.get("actions", {}) or {}
        measurement_count = sum(
            str(command.get("measurement_uav_id", "none")) != "none"
            for command in actions.values()
            if isinstance(command, Mapping)
        )
    return int(measurement_count), float(max(0.0, packets))


def _episode(
    config: SafetyMAPPOEvaluationConfig,
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
        measurement_overhead_enabled=eval_config.measurement_overhead_enabled,
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
    previous_positions = {key: value.copy() for key, value in initial_positions.items()}
    rows: list[dict[str, Any]] = []
    all_position_errors: list[float] = []
    trajectory_length = 0.0
    energy_values: list[float] = []
    action_means: list[float] = []
    action_maxima: list[float] = []
    saturation_values: list[float] = []
    clearances: list[float] = [float(initial_clearance)]
    pre_clearances: list[float] = [float(initial_clearance)]
    projection_count = 0
    collision_episode = int(initial_collision)
    pre_projection_collision_episode = 0
    collision_step: int | None = None
    measurement_count_total = 0
    overhead_packets_total = 0.0
    finite_state = int(initial_collision == 0 and _finite_flag(initial_clearance))
    finite_covariance = 1
    truth_free = 1
    trainer_update_called = 0

    for step in range(int(config.max_steps)):
        truth_free &= int(
            all(_actor_observation_truth_free(observations.get(uav_id, {})) for uav_id in environment.uav_ids)
        )
        decision = trainer.act(
            observations,
            reset_info.get("central_state") if reset_info else None,
            deterministic=True,
        )
        # This evaluator has no learning path by construction.  Keep the
        # explicit flag in every row so accidental future updates are visible.
        trainer_update_called = 0
        next_observations, rewards, terminated, truncated, info = environment.step(
            decision.actions
        )
        state_finite, covariance_finite = _state_and_covariance_finite(environment)
        finite_state &= state_finite
        finite_covariance &= covariance_finite
        rmse, p95, errors = _positions_metrics(environment)
        all_position_errors.extend(errors)
        execution = info.get("action_execution", {}) or {}
        displacement = [
            float(
                np.linalg.norm(
                    np.asarray(environment.states[uav_id].position_m, dtype=float)
                    - previous_positions[uav_id]
                )
            )
            for uav_id in environment.uav_ids
        ]
        path_step = float(sum(value for value in displacement if math.isfinite(value)))
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
        projection_count += int(_finite_or(info.get("safety_projection_count", 0.0)))
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
        team_reward = _finite_or(info.get("team_reward", np.mean(list(rewards.values())) if rewards else 0.0))
        row = {
            "policy_seed": int(policy_seed),
            "scenario_seed": int(scenario_seed),
            "branch": branch,
            "evaluation_seed": int(_evaluation_seed(scenario_seed)),
            "step": int(step + 1),
            "time_s": float((step + 1) * environment.config.motion.dt_s),
            "scenario_hash": scenario_hash,
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
            "termination_reason": str(info.get("termination_reason", "running")),
            "collision": collision,
            "pre_projection_collision": pre_collision,
            "minimum_clearance_m": minimum,
            "pre_projection_minimum_clearance_m": pre_minimum,
            "safety_projection_count": int(_finite_or(info.get("safety_projection_count", 0.0))),
            "action_saturation_rate": _finite_or(info.get("action_saturation_rate", 0.0)),
            "action_magnitude_mean": action_mean,
            "action_magnitude_max": action_max,
            "energy_proxy_step": energy,
            "path_length_step_m": path_step,
            "trajectory_length_m": trajectory_length,
            "measurement_count_step": int(count_step),
            "measurement_overhead_packets_step": packets_step,
            "j_loc": _finite_or(objective.get("j_loc", 0.0)),
            "j_com": _finite_or(objective.get("j_com", 0.0)),
            "joint_objective": _finite_or(objective.get("joint_objective", 0.0)),
            "team_reward": team_reward,
            "packet_loss": _finite_or(metrics.get("packet_loss", 0.0)),
            "delay_s": _finite_or(metrics.get("delay", 0.0)),
            "aoi_s": _finite_or(metrics.get("aoi", 0.0)),
            "connectivity": _finite_or(metrics.get("connectivity", 0.0)),
            "position_rmse_m": rmse,
            "actor_observation_truth_free": int(truth_free),
            "trainer_update_called": int(trainer_update_called),
            "state_finite": int(state_finite),
            "covariance_finite": int(covariance_finite),
        }
        rows.append(row)
        previous_positions = {
            uav_id: np.asarray(state.position_m, dtype=float).copy()
            for uav_id, state in environment.states.items()
        }
        reset_info = info
        observations = next_observations
        if terminated or truncated:
            break

    steps = max(1, len(rows))
    final_reason = str(rows[-1]["termination_reason"] if rows else "running")
    summary = {
        "policy_seed": int(policy_seed),
        "scenario_seed": int(scenario_seed),
        "branch": branch,
        "evaluation_seed": int(_evaluation_seed(scenario_seed)),
        "scenario_hash": scenario_hash,
        "episode_steps": int(len(rows)),
        "initial_collision": int(initial_collision),
        "initial_minimum_clearance_m": float(initial_clearance),
        "collision_episode": int(collision_episode),
        "pre_projection_collision_episode": int(pre_projection_collision_episode),
        "collision_step": int(collision_step) if collision_step is not None else -1,
        "minimum_clearance_m": float(min(clearances)),
        "pre_projection_minimum_clearance_m": float(min(pre_clearances)),
        "safety_projection_count": int(projection_count),
        "action_saturation_rate": float(np.mean(saturation_values)) if saturation_values else 0.0,
        "action_magnitude_mean": float(np.mean(action_means)) if action_means else 0.0,
        "action_magnitude_max": float(max(action_maxima, default=0.0)),
        "energy_proxy": float(np.mean(energy_values)) if energy_values else 0.0,
        "trajectory_length_m": float(trajectory_length),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(all_position_errors)))) if all_position_errors else 0.0,
        "position_error_p95_m": float(np.percentile(all_position_errors, 95.0)) if all_position_errors else 0.0,
        "j_loc": float(np.mean([row["j_loc"] for row in rows])) if rows else 0.0,
        "j_com": float(np.mean([row["j_com"] for row in rows])) if rows else 0.0,
        "joint_objective": float(np.mean([row["joint_objective"] for row in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([row["team_reward"] for row in rows])) if rows else 0.0,
        "packet_loss": float(np.mean([row["packet_loss"] for row in rows])) if rows else 0.0,
        "delay_s": float(np.mean([row["delay_s"] for row in rows])) if rows else 0.0,
        "aoi_s": float(np.mean([row["aoi_s"] for row in rows])) if rows else 0.0,
        "connectivity": float(np.mean([row["connectivity"] for row in rows])) if rows else 0.0,
        "measurement_count_total": int(measurement_count_total),
        "measurement_overhead_packets_total": float(overhead_packets_total),
        "termination_reason": final_reason,
        "terminated": int(bool(rows and rows[-1]["terminated"])),
        "truncated": int(bool(rows and rows[-1]["truncated"])),
        "task_complete": 0,
        "actor_observation_truth_free": int(truth_free),
        "trainer_update_called": int(trainer_update_called),
        "state_finite": int(finite_state),
        "covariance_finite": int(finite_covariance),
        "test_data_read": 0,
    }
    return rows, summary


def _bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in keys:
        values = [_finite_or(row.get(key, 0.0)) for row in rows]
        result[key] = float(np.mean(values)) if values else 0.0
    return result


def _plot(
    path: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        BASELINE_BRANCH: "#0072B2",
        FULL_BRANCH: "#D55E00",
        "quality_aware_mappo": "#009E73",
    }
    labels = {
        BASELINE_BRANCH: "baseline MAPPO",
        FULL_BRANCH: "budgeted quality-aware MAPPO",
        "quality_aware_mappo": "quality-aware MAPPO",
    }
    metrics = (
        ("collision_episode", "Collision rate", "episode fraction"),
        ("minimum_clearance_m", "Minimum clearance", "m"),
        ("energy_proxy", "Energy proxy", "normalized"),
        ("trajectory_length_m", "Trajectory length", "m"),
        ("safety_projection_count", "Safety projections", "count"),
        ("position_rmse_m", "Position RMSE", "m"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(scenario_seeds), dtype=float)
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        for branch in branches:
            values = []
            for scenario_seed in scenario_seeds:
                match = [
                    row for row in scenario_rows
                    if int(row.get("scenario_seed", -1)) == int(scenario_seed)
                    and str(row.get("branch", "")) == branch
                ]
                values.append(_finite_or(match[0].get(metric, 0.0)) if match else 0.0)
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color=colors.get(branch, "#666666"),
                label=labels.get(branch, branch.replace("_", " ")),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(
        "Safety--energy evaluation: 10 scenario seeds; policy seeds averaged within scenario",
        fontsize=13,
    )
    fig.text(0.5, 0.01, "Deterministic frozen-checkpoint inference; projection on; diagnostic only", ha="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_mappo_evaluation(config: SafetyMAPPOEvaluationConfig) -> dict[str, Any]:
    """Evaluate frozen baseline and full policies on paired safety scenes."""

    output, logs, plots = _prepare_outputs(config)
    scenario_payloads: dict[int, tuple[dict[str, Any], str]] = {
        int(seed): _load_scenario(Path(config.scenario_root), int(seed), config.uav_count)
        for seed in config.scenario_seeds
    }
    scenario_hashes = {str(seed): value[1] for seed, value in scenario_payloads.items()}
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    bundle_cache: dict[tuple[int, str], tuple[Any, Any, dict[str, Any], Path]] = {}
    errors: list[str] = []

    for branch in config.branches:
        for policy_seed in config.policy_seeds:
            _progress(config, f"policy_seed={policy_seed} branch={branch} checkpoint loading")
            bundle_cache[(int(policy_seed), branch)] = _load_bundle(config, int(policy_seed), branch)
            trainer, eval_config, _manifest, _scale_root = bundle_cache[(int(policy_seed), branch)]
            for scenario_seed in config.scenario_seeds:
                scenario, scenario_hash = scenario_payloads[int(scenario_seed)]
                _progress(
                    config,
                    f"policy_seed={policy_seed} branch={branch} "
                    f"scenario_seed={scenario_seed} start",
                )
                rows, summary = _episode(
                    config,
                    trainer,
                    eval_config,
                    scenario,
                    int(scenario_seed),
                    scenario_hash,
                    int(policy_seed),
                    branch,
                )
                step_rows.extend(rows)
                episode_rows.append(summary)
                if summary["collision_episode"] and not summary["terminated"]:
                    errors.append(
                        f"collision episode was not terminated: {policy_seed}/{branch}/{scenario_seed}"
                    )
                if summary["task_complete"]:
                    errors.append(
                        f"collision/task semantics invalid: {policy_seed}/{branch}/{scenario_seed}"
                    )

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
        for branch in config.branches:
            matching = [
                row for row in episode_rows
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
                    "all_trainer_updates_zero": int(
                        all(int(row["trainer_update_called"]) == 0 for row in matching)
                    ) if matching else 0,
                }
            )

    sensitivity_rows: list[dict[str, Any]] = []
    for policy_seed in config.policy_seeds:
        for branch in config.branches:
            matching = [
                row for row in episode_rows
                if int(row["policy_seed"]) == int(policy_seed)
                and str(row["branch"]) == branch
            ]
            sensitivity_rows.append(
                {
                    "policy_seed": int(policy_seed),
                    "branch": branch,
                    "scenario_count": int(len(matching)),
                    **_mean_rows(matching, metric_keys),
                }
            )

    contrasts: list[dict[str, Any]] = []
    baseline_scenario = {
        int(row["scenario_seed"]): row
        for row in scenario_rows if row["branch"] == BASELINE_BRANCH
    }
    full_scenario = {
        int(row["scenario_seed"]): row
        for row in scenario_rows if row["branch"] == FULL_BRANCH
    }
    for metric_index, metric in enumerate(metric_keys):
        differences = [
            _finite_or(full_scenario[seed].get(metric, 0.0))
            - _finite_or(baseline_scenario[seed].get(metric, 0.0))
            for seed in config.scenario_seeds
            if seed in baseline_scenario and seed in full_scenario
        ]
        ci_low, ci_high = _bootstrap_ci(
            differences,
            config.bootstrap_samples,
            # Python's built-in hash is intentionally process-randomized.  A
            # stable metric index keeps bootstrap intervals reproducible across
            # separate PyCharm/Python processes.
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

    collision_base = float(np.mean([baseline_scenario[seed]["collision_episode"] for seed in baseline_scenario])) if baseline_scenario else 0.0
    collision_full = float(np.mean([full_scenario[seed]["collision_episode"] for seed in full_scenario])) if full_scenario else 0.0
    both_zero = collision_base == 0.0 and collision_full == 0.0
    collision_contrast = contrast("collision_episode")
    clearance_contrast = contrast("minimum_clearance_m")
    energy_contrast = contrast("energy_proxy")
    audit_ok = bool(
        len(episode_rows) == len(config.policy_seeds) * len(config.scenario_seeds) * len(config.branches)
        and len(scenario_rows) == len(config.scenario_seeds) * len(config.branches)
        and not errors
        and all(int(row["initial_collision"]) == 0 for row in episode_rows)
        and all(int(row["state_finite"]) == 1 for row in episode_rows)
        and all(int(row["covariance_finite"]) == 1 for row in episode_rows)
        and all(int(row["actor_observation_truth_free"]) == 1 for row in episode_rows)
        and all(int(row["trainer_update_called"]) == 0 for row in episode_rows)
        and all(int(row["task_complete"]) == 0 for row in episode_rows)
        and all(int(row["scenario_hash"] == scenario_hashes[str(int(row["scenario_seed"]))]) for row in episode_rows)
        and not any("test" in str(path).lower() for path in (config.scenario_root, config.model_root))
    )
    decision_supported = bool(
        audit_ok
        and not both_zero
        and collision_full < collision_base
        and collision_contrast["bootstrap_95_ci_high"] < 0.0
        and clearance_contrast["bootstrap_95_ci_low"] >= 0.0
        and energy_contrast["bootstrap_95_ci_high"] <= 0.0
    )
    if both_zero:
        decision = "no_safety_discrimination_zero_collisions"
    elif decision_supported:
        decision = "full_mappo_safety_energy_improvement_supported"
    elif collision_full < collision_base and energy_contrast["bootstrap_95_ci_high"] > 0.0:
        decision = "safety_energy_tradeoff"
    elif collision_full > collision_base:
        decision = "full_mappo_collision_rate_worse"
    else:
        decision = "safety_improvement_not_supported"
    report: dict[str, Any] = {
        "schema_version": "safety-mappo-evaluation-v1.0",
        "audit_ok": audit_ok,
        "decision_supported": decision_supported,
        "decision": decision,
        "design": {
            "uav_count": int(config.uav_count),
            "max_steps": int(config.max_steps),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "branches": list(config.branches),
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "mean over five policy seeds within each scenario",
            "deterministic_inference": True,
            "trainer_update_called": False,
            "safety_projection_enabled": True,
            "test_data_read": False,
            "test_data_role": "sealed; not read during evaluation",
        },
        "source": {
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "scenario_hashes": scenario_hashes,
            "evaluation_seed_rule": "1000000 + scenario_seed",
            "checkpoint_branches": list(config.branches),
        },
        "counts": {
            "policy_seed_count": len(config.policy_seeds),
            "scenario_seed_count": len(config.scenario_seeds),
            "branch_count": len(config.branches),
            "episode_rows": len(episode_rows),
            "step_rows": len(step_rows),
            "collision_episode_count_baseline": int(
                sum(
                    int(row["collision_episode"])
                    for row in episode_rows
                    if row["branch"] == BASELINE_BRANCH
                )
            ),
            "collision_episode_count_full": int(
                sum(
                    int(row["collision_episode"])
                    for row in episode_rows
                    if row["branch"] == FULL_BRANCH
                )
            ),
            "collision_scenario_count_baseline": int(
                sum(int(row["collision_episode"] > 0.0) for row in scenario_rows if row["branch"] == BASELINE_BRANCH)
            ),
            "collision_scenario_count_full": int(
                sum(int(row["collision_episode"] > 0.0) for row in scenario_rows if row["branch"] == FULL_BRANCH)
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
                "This is a paired safety--energy evaluation of frozen checkpoints; "
                "it is not a new training result and does not establish final algorithm superiority."
            ),
            "collision_semantics": "collision terminates the episode and task_complete remains false",
            "primary_statistical_unit": "scenario_seed",
        },
    }
    _write_csv(logs / "episode_metrics.csv", episode_rows)
    _write_csv(logs / "step_metrics.csv", step_rows)
    _write_csv(logs / "scenario_summary.csv", scenario_rows)
    _write_csv(logs / "paired_contrasts.csv", contrasts)
    _write_csv(logs / "policy_seed_sensitivity.csv", sensitivity_rows)
    _write_json(logs / "audit_report.json", report)
    _write_json(output / "scenario_hashes.json", scenario_hashes)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "model_root": str(Path(config.model_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "output_dir": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
        },
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "safety-mappo-evaluation-v1.0",
            "data_root": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
            "audit_report": str((logs / "audit_report.json").resolve()),
            "episode_schema": "episode-v1.3 source episodes are read-only",
            "test_data_read": False,
        },
    )
    plot_path = plots / "safety_energy_comparison.png"
    _plot(plot_path, scenario_rows, config.scenario_seeds, config.branches)
    return {
        "audit_ok": audit_ok,
        "decision_supported": decision_supported,
        "decision": decision,
        "independent_unit": "scenario_seed",
        "policy_seed_count": len(config.policy_seeds),
        "scenario_seed_count": len(config.scenario_seeds),
        "data": str(output),
        "logs": str(logs),
        "plot": str(plot_path),
        "audit_report": str(logs / "audit_report.json"),
    }


__all__ = [
    "BASELINE_BRANCH",
    "DEFAULT_BRANCHES",
    "DEFAULT_POLICY_SEEDS",
    "DEFAULT_SCENARIO_SEEDS",
    "FULL_BRANCH",
    "SafetyMAPPOEvaluationConfig",
    "run_safety_mappo_evaluation",
]
