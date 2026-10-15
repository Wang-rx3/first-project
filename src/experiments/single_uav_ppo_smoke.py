"""Paired, bounded online PPO smoke; historical experiments remain untouched.

Run with ``python -m src.experiments.single_uav_ppo_smoke --config ...``.
The parking reporter is read-only. All control/reward/termination decisions are
made by the frozen StrictSingleUAVPPOEnv, including in the no-obstacle stage.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..learning.mappo import central_state_to_vector
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..simulation.motion_model import clip_vector_norm
from ..simulation.obstacle_model import is_physical_obstacle, segment_obstacle_clearance, segment_intersects_obstacle
from .single_uav_ppo_baseline import (
    FEATURE_INDEX_MAP, create_single_uav_ppo_environment,
    create_single_uav_ppo_trainer, load_single_uav_ppo_config,
    verify_actor_observation_contract,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("non-finite JSON artifact")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class CSVLog:
    def __init__(self, path: Path):
        self.path = path
        self.fields: list[str] | None = None

    def append(self, row: Mapping[str, Any]) -> None:
        record = {k: ("" if v is None else json.dumps(json_value(v), separators=(",", ":")) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()}
        if self.fields is None:
            self.fields = list(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.fields).writeheader()
        if set(record) != set(self.fields):
            raise ValueError(f"CSV schema drift: {self.path}")
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(record)


@dataclass
class ParkingReporter:
    """Report actual-state five-step parking without changing the environment."""
    hold: int = 0
    reached: bool = False
    collided: bool = False
    first_reached_step: int | None = None

    def observe(self, distance: float, speed: float, collision: bool, step: int) -> bool:
        self.collided |= bool(collision)
        eligible = not self.collided and distance <= 1.0 and speed <= 0.5
        self.hold = self.hold + 1 if eligible else 0
        if self.hold >= 5 and self.first_reached_step is None:
            self.first_reached_step = step
        self.reached = (self.reached or self.hold >= 5) and not self.collided
        return self.reached


def raw_action_diagnostics(environment: Any, sampled_delta: Any) -> dict[str, Any]:
    """Pure candidate segment before safety/boundary projection, after limits."""
    state = environment.states[environment.uav_ids[0]]
    motion = environment.config.motion
    delta = clip_vector_norm(np.asarray(sampled_delta), motion.max_acceleration_mps2 * motion.dt_s)
    velocity = clip_vector_norm(state.velocity_mps + delta, motion.max_speed_mps)
    candidate = state.position_m + state.velocity_mps * motion.dt_s + 0.5 * (velocity - state.velocity_mps) * motion.dt_s
    obstacles = [o for o in environment.obstacles if o.active and is_physical_obstacle(o)]
    clearance = min((segment_obstacle_clearance(state.position_m, candidate, o) for o in obstacles), default=None)
    contact = any(segment_intersects_obstacle(state.position_m, candidate, o) for o in obstacles)
    lower, upper = np.asarray(motion.bounds_min_m), np.asarray(motion.bounds_max_m)
    return {
        "raw_action_physical_danger": None if not obstacles else bool(contact or clearance < motion.safety_distance_m),
        "raw_action_contact": None if not obstacles else bool(contact),
        "raw_action_clearance_m": clearance,
        "raw_action_boundary_danger": bool(np.any(candidate < lower) or np.any(candidate > upper)),
    }


def _reset(environment: Any, scene: Mapping[str, Any], episode_id: str):
    scenario = copy.deepcopy(scene["scenario"])
    scenario["episode_id"] = episode_id
    # Sensor streams use the reset seed; geometry is an explicit fixed manifest.
    return environment.reset(seed=int(scene["noise_seed"]), scenario_config=scenario)


def _central(info: Mapping[str, Any], base: Any) -> np.ndarray:
    return central_state_to_vector(info["central_state"], uav_count=1, node_feature_dim=42, edge_feature_dim=11, max_obstacles=base.max_obstacles, include_remaining_time=False)


def calibrate_training_only(base: Any, pool: list[dict[str, Any]], group: str, data_dir: Path, log_dir: Path):
    if len(pool) != 18 or any(s["split"] != "train" for s in pool):
        raise ValueError("calibration requires exactly the training pool")
    placeholder = {"u_center": 0.0, "u_scale": 1.0, "c_center": 0.0, "c_scale": 1.0, "delta_u_scale": 1.0, "delta_goal_scale": 1.0}
    # These constants only permit reward-independent random calibration. They
    # are discarded before PPO construction and never produce PPO transitions.
    env = create_single_uav_ppo_environment(replace(base, train_potential_statistics=placeholder))
    rng = np.random.default_rng(base.seed + 8100)
    observations, central, records = [], [], []
    u_values, c_values, du_values, dg_values = [], [], [], []
    csv_log = CSVLog(log_dir / f"calibration_{group}.csv")
    resets = 0
    for scene in pool:
        obs, info = _reset(env, scene, f"calibration_{group}_{scene['scenario_id']}")
        resets += 1
        observations.append(copy.deepcopy(obs))
        central.append(_central(info, base))
        for local_step in range(16):
            direction = rng.normal(size=3)
            delta = direction / max(np.linalg.norm(direction), 1e-12) * rng.uniform(0, base.max_delta_v)
            previous_u, previous_goal = env._position_logdet_uncertainty(), env.estimated_goal_distance()
            action = {env.uav_ids[0]: {"delta_v": delta, "measurement_uav_id": "none", "bandwidth": {}}}
            next_obs, _rewards, terminated, truncated, next_info = env.step(action)
            current_u, current_goal = env._position_logdet_uncertainty(), env.estimated_goal_distance()
            u_values.append(current_u)
            communication_objective = float(next_info["objective_components"]["j_com"])
            c_values.append(communication_objective)
            du_values.append(previous_u - current_u)
            dg_values.append(previous_goal - current_goal)
            row = {"source": "train_only_random_calibration", "scenario_id": scene["scenario_id"], "noise_seed": scene["noise_seed"], "step": local_step, "delta_v": delta.tolist(), "node_features": np.asarray(next_obs[env.uav_ids[0]]["node_features"]).tolist(), "central_vector": _central(next_info, base).tolist(), "position_logdet": current_u, "communication_objective": communication_objective, "estimated_goal_distance_m": current_goal, "delta_u": previous_u-current_u, "delta_goal": previous_goal-current_goal, "terminated": bool(terminated), "truncated": bool(truncated)}
            records.append(row)
            csv_log.append({"group": group, **row})
            observations.append(copy.deepcopy(next_obs))
            central.append(_central(next_info, base))
            if terminated or truncated:
                obs, info = _reset(env, scene, f"calibration_{group}_{scene['scenario_id']}_reset_{local_step}")
                resets += 1
                observations.append(copy.deepcopy(obs))
                central.append(_central(info, base))
            else:
                obs, info = next_obs, next_info
    if len(records) != 288:
        raise AssertionError("calibration budget mismatch")
    write_json(data_dir / f"calibration_{group}.json", {"group": group, "actual_interaction_steps": 288, "resets": resets, "ppo_samples": 0, "reward_independent_placeholder": placeholder, "stream_sha256": canonical_hash(records), "records": records})
    return {"observations": observations, "central": central, "u": u_values, "c": c_values, "du": du_values, "dg": dg_values, "records": records, "resets": resets, "stream_sha256": canonical_hash(records)}


def fit_common_statistics(calibration: Mapping[str, Any], base: Any, artifact: Path):
    normalizer = ObservationNormalizer().fit(calibration["observations"])
    node = normalizer.normalizers["node_features"]
    # Keep velocity scaling tied to the frozen maximum speed rather than the
    # low-velocity random calibration trajectory.
    node.scale[3:6] = np.maximum(node.scale[3:6], base.max_speed_mps / 2.0)
    central = FeatureNormalizer().fit(np.asarray(calibration["central"]))
    central.scale[3:6] = np.maximum(central.scale[3:6], base.max_speed_mps / 2.0)
    def location_scale(values):
        arr = np.asarray(values, dtype=float)
        center = float(np.median(arr))
        return center, max(float(np.median(np.abs(arr-center))), 1e-3)
    u_center, u_scale = location_scale(calibration["u"])
    c_center, c_scale = location_scale(calibration["c"])
    _, du_scale = location_scale(calibration["du"])
    _, dg_scale = location_scale(calibration["dg"])
    potential = {"u_center": u_center, "u_scale": u_scale, "c_center": c_center, "c_scale": c_scale, "delta_u_scale": du_scale, "delta_goal_scale": dg_scale, "fit_split": "train", "sample_count": 288, "method": "median/MAD; scale floor 0.001", "source_stream_sha256": calibration["stream_sha256"]}
    state = {"observation_normalizer": normalizer.state_dict(), "central_normalizer": central.state_dict(), "potential_statistics": potential, "fit_split": "train", "frozen_during_ppo_and_validation": True, "calibration_steps_per_arm": 288, "velocity_scale_floor_mps": 6.0}
    write_json(artifact, state)
    provenance = {"artifact_path": str(artifact.resolve()), "source_sha256": sha256_file(artifact), "fit_split": "train", "feature_index_map": FEATURE_INDEX_MAP, "calibration_stream_sha256": calibration["stream_sha256"]}
    return normalizer, central, potential, provenance


def _parameter_group(name: str) -> str:
    return "critic" if name.startswith("critic.") else "gat" if name.startswith("gat.") else "actor"


def probe_actor_features(trainer, environment, scenes):
    """Reset-only checks on actual geometry plus encoder/Actor sensitivity."""
    records = []
    for scene in scenes:
        obs, _info = _reset(environment, scene, f"actor_contract_{scene['scenario_id']}")
        record = verify_actor_observation_contract(environment, obs, normalizer=trainer.normalizer)
        record["scenario_id"] = scene["scenario_id"]
        record["physical_obstacle_count"] = sum(is_physical_obstacle(o) for o in environment.obstacles)
        local = trainer.normalizer.transform(obs)[environment.uav_ids[0]]
        with torch.no_grad():
            context = trainer.policy._local_context(local)[0]
            mean = trainer.policy.delta_mean_head(trainer.policy.actor_body(context))
            responses = {}
            for name, lo, hi in (("estimated_velocity", 3, 6), ("goal_delta", 30, 33), ("obstacle_body_direction", 39, 42)):
                altered = copy.deepcopy(local)
                altered["node_features"][0, lo:hi] += 0.1
                changed_context = trainer.policy._local_context(altered)[0]
                changed_mean = trainer.policy.delta_mean_head(trainer.policy.actor_body(changed_context))
                responses[name] = {"context_change_l2": float(torch.linalg.vector_norm(changed_context-context)), "pre_tanh_mean_change_l2": float(torch.linalg.vector_norm(changed_mean-mean))}
                if any(not math.isfinite(v) or v <= 0 for v in responses[name].values()):
                    raise AssertionError(f"Actor does not respond to normalized {name}")
        record["normalized_feature_sensitivity"] = responses
        records.append(record)
    return {"scenes": records, "environment_steps": 0, "reset_count": len(scenes), "training_or_optimizer_steps": 0, "real_encoder_input": trainer._single_uav_actor_contract_probe}


def _parameter_snapshot(policy: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in policy.named_parameters()}


def parameter_deltas(before: Mapping[str, Any], policy: Any) -> dict[str, float]:
    squares = {"actor": 0.0, "gat": 0.0, "critic": 0.0}
    for name, value in policy.named_parameters():
        squares[_parameter_group(name)] += float((value.detach().cpu().double()-before[name].double()).square().sum())
    return {f"{key}_parameter_delta_l2": math.sqrt(value) for key, value in squares.items()}


class OptimizerAudit:
    """Observe actual optimizer calls and gradients; do not modify updates."""
    def __init__(self, trainer: Any):
        self.trainer = trainer
        self.steps = 0
        self.gradients: list[dict[str, float]] = []
        original = trainer.optimizer.step
        def observed_step(*args, **kwargs):
            sums = {"actor": 0.0, "gat": 0.0, "critic": 0.0}
            for name, parameter in trainer.policy.named_parameters():
                if parameter.grad is not None:
                    sums[_parameter_group(name)] += float(parameter.grad.detach().double().square().sum())
            norms = {key: math.sqrt(v) for key, v in sums.items()}
            if not all(math.isfinite(v) for v in norms.values()):
                raise FloatingPointError("non-finite optimizer gradient")
            result = original(*args, **kwargs)
            self.steps += 1
            self.gradients.append(norms)
            return result
        trainer.optimizer.step = observed_step


def assert_online_transition(decision: Any, sampled_delta: np.ndarray, sampled_logprob: float) -> None:
    sample = next(iter(decision.samples.values()))
    if not np.array_equal(np.asarray(sample.delta_v), sampled_delta) or float(sample.old_log_prob) != sampled_logprob:
        raise AssertionError("execution corrupted sampled PPO action/logprob")
    def inspect(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in {"teacher_action", "bc_target", "expert_action", "dagger_sample", "teacher_sample"}:
                    raise AssertionError(f"teacher sample in online transition: {key}")
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
    inspect(decision.observations)
    inspect(decision.actions)


def summarize_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(selected):
        count = len(selected)
        physical = [r for r in selected if r["kind"] != "normal"]
        steps = sum(r["steps"] for r in selected)
        physical_steps = sum(r["raw_action_physical_denominator"] for r in selected)
        fraction = lambda n, d: None if not d else n / d
        return {"episodes": count, "steps": steps, "stable_arrivals": sum(r["stable_arrival"] for r in selected), "stable_arrival_rate": fraction(sum(r["stable_arrival"] for r in selected), count), "task_successes": sum(r["task_success"] for r in selected), "task_success_rate": fraction(sum(r["task_success"] for r in selected), count), "physical_bypass_episodes": len(physical), "physical_bypasses": sum(bool(r["physical_bypass"]) for r in physical), "physical_bypass_rate": fraction(sum(bool(r["physical_bypass"]) for r in physical), len(physical)), "collisions": sum(r["collision"] for r in selected), "collision_rate": fraction(sum(r["collision"] for r in selected), count), "timeout_failures": sum(r["timeout_failure"] for r in selected), "timeout_rate": fraction(sum(r["timeout_failure"] for r in selected), count), "raw_time_limits": sum(r["time_limit_reached_raw"] for r in selected), "rollout_window_cuts": sum(r["rollout_window_cut"] for r in selected), "raw_action_physical_danger_steps": sum(r["raw_action_danger_steps"] for r in selected), "raw_action_physical_denominator": physical_steps, "raw_action_danger_rate": fraction(sum(r["raw_action_danger_steps"] for r in selected), physical_steps), "safety_projection_steps": sum(r["safety_projection_steps"] for r in selected), "safety_projection_rate": fraction(sum(r["safety_projection_steps"] for r in selected), steps), "boundary_projection_steps": sum(r["boundary_projection_steps"] for r in selected)}
    return {"all": aggregate(rows), **{kind: aggregate([r for r in rows if r["kind"] == kind]) for kind in ("normal", "single", "full")}}


def _finish_episode(group, phase, stage, scene, episode_id, tracker, counts, info, terminated, truncated, cut):
    physical = scene["kind"] != "normal"
    collision = bool(tracker.collided)
    strict = bool(info.get("strict_success", False)) if physical else None
    task_success = bool(strict if physical else tracker.reached)
    return {"group": group, "phase": phase, "stage": stage, "scenario_id": scene["scenario_id"], "episode_id": episode_id, "kind": scene["kind"], "side": scene["side"], "noise_seed": scene["noise_seed"], "geometry_sha256": scene["geometry_sha256"], "steps": counts["steps"], "stable_arrival": task_success, "parking_diagnostic": bool(tracker.reached), "first_parking_step": tracker.first_reached_step, "strict_success": strict, "task_success": task_success, "physical_bypass": bool(info.get("strict_physical_pass_complete", False) and not collision) if physical else None, "collision": collision, "physical_collision": bool(counts["physical_collision"]), "terminated": bool(terminated), "truncated": bool(truncated), "time_limit_reached_raw": bool(truncated), "timeout_failure": bool(truncated and not task_success), "rollout_window_cut": bool(cut), "termination_reason": "rollout_window_cut" if cut else info.get("termination_reason", "unknown"), "actual_goal_distance_m_final": info["strict_goal_distance_m_actual"], "actual_speed_mps_final": info["strict_speed_mps_actual"], "learning_return": counts["learning_return"], "raw_action_danger_steps": counts["raw_action_danger_steps"], "raw_action_physical_denominator": counts["raw_action_physical_denominator"], "safety_projection_steps": counts["safety_projection_steps"], "boundary_projection_steps": counts["boundary_projection_steps"]}


def collect_decisions(trainer, environment, base, scene, *, group, phase, stage, block_id, step_budget, online, step_log, episode_log):
    if online and trainer.rollout:
        raise AssertionError("old rollout at block start")
    all_episodes = []
    resets = 0
    terminated = truncated = False
    episode_id = ""
    tracker = None
    for block_step in range(step_budget):
        if tracker is None or terminated or truncated:
            episode_id = f"{group}_{phase}_stage{stage}_block{block_id}_{scene['scenario_id']}_ep{resets}"
            obs, info = _reset(environment, scene, episode_id)
            resets += 1
            tracker = ParkingReporter()
            counts = Counter(steps=0, physical_collision=0, learning_return=0.0, raw_action_danger_steps=0, raw_action_physical_denominator=0, safety_projection_steps=0, boundary_projection_steps=0)
        decision = trainer.act(obs, info.get("central_state"), deterministic=not online)
        uid = environment.uav_ids[0]
        sample = decision.samples[uid]
        sampled = np.asarray(sample.delta_v).copy()
        logprob = float(sample.old_log_prob)
        danger = raw_action_diagnostics(environment, sampled)
        next_obs, _rewards, terminated, truncated, next_info = environment.step(decision.actions)
        trainer.reconcile_action_execution(decision, next_info)
        assert_online_transition(decision, sampled, logprob)
        if online:
            trainer.add_transition(decision, next_info["learning_rewards"], next_info["central_state"], done=terminated or truncated, terminated=terminated, truncated=truncated)
        collision = bool(next_info.get("collision", False))
        counts["steps"] += 1
        tracker.observe(next_info["strict_goal_distance_m_actual"], next_info["strict_speed_mps_actual"], collision, counts["steps"])
        execution = next_info["action_execution"][uid]
        counts["physical_collision"] += bool(next_info.get("physical_collision", False))
        counts["learning_return"] += float(next_info["learning_rewards"][uid])
        counts["raw_action_danger_steps"] += bool(danger["raw_action_physical_danger"])
        counts["raw_action_physical_denominator"] += danger["raw_action_physical_danger"] is not None
        counts["safety_projection_steps"] += bool(execution.get("safety_projected", False))
        counts["boundary_projection_steps"] += bool(execution.get("boundary_projected", False))
        executed = np.asarray(execution["delta_v_applied"])
        step_log.append({"group": group, "phase": phase, "stage": stage, "block_id": block_id, "block_step": block_step, "episode_id": episode_id, "episode_step": counts["steps"], "scenario_id": scene["scenario_id"], "kind": scene["kind"], "side": scene["side"], "sample_source": "online_policy_rollout" if online else "deterministic_validation", "old_logprob": logprob, "logprob_action": "sampled", "sampled_delta_v": sampled.tolist(), "executed_delta_v": executed.tolist(), "total_correction_delta_v": (executed-sampled).tolist(), "safety_projected": bool(execution.get("safety_projected", False)), "boundary_projected": bool(execution.get("boundary_projected", False)), "actual_position_m": environment.states[uid].position_m.tolist(), "actual_velocity_mps": environment.states[uid].velocity_mps.tolist(), "goal_distance_m_actual": next_info["strict_goal_distance_m_actual"], "speed_mps_actual": next_info["strict_speed_mps_actual"], "parking_hold": tracker.hold, "parking_diagnostic": tracker.reached, "physical_bypass": None if scene["kind"] == "normal" else bool(next_info["strict_physical_pass_complete"]), "strict_success": None if scene["kind"] == "normal" else bool(next_info["strict_success"]), "collision": collision, "physical_collision": bool(next_info.get("physical_collision", False)), "learning_reward": float(next_info["learning_rewards"][uid]), "terminated": bool(terminated), "truncated": bool(truncated), **danger})
        obs, info = next_obs, next_info
        cut = block_step == step_budget - 1 and not (terminated or truncated)
        if terminated or truncated or cut:
            row = _finish_episode(group, phase, stage, scene, episode_id, tracker, counts, next_info, terminated, truncated, cut)
            all_episodes.append(row)
            episode_log.append(row)
            if not online:
                # Evaluation is one physical episode per held-out scene, not
                # a fixed interaction budget filled by repeats.
                break
    return {"steps": block_step + 1, "resets": resets, "episodes": all_episodes}


def _save_checkpoint(path, trainer, group, updates, online_steps, initialization, manifest_sha):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"policy_state_dict": trainer.policy.state_dict(), "optimizer_state_dict": trainer.optimizer.state_dict(), "trainer_config": trainer.config.to_dict() if hasattr(trainer.config, "to_dict") else vars(trainer.config), "normalization_stats": trainer.normalizer.state_dict(), "central_normalization_stats": trainer.central_normalizer.state_dict(), "uav_count": 1, "group": group, "update_count": updates, "ppo_updates_this_run": updates, "ppo_online_steps_this_run": online_steps, "source_update_count_ignored": initialization.get("source_update_count"), "initialization": initialization, "smoke_manifest_sha256": manifest_sha, "online_sample_source": "fresh_policy_rollout", "teacher_online_samples": 0, "smoke_only": True}, path)


def _plot_diagnostics(updates, evaluations, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for group in ("A", "B"):
        rows = [r for r in updates if r["group"] == group]
        x = [r["ppo_update_this_run"] for r in rows]
        axes[0, 0].plot(x, [r["actor_loss"] for r in rows], label=group)
        axes[0, 1].plot(x, [r["critic_loss"] for r in rows], label=group)
        axes[1, 0].plot(x, [r["actor_parameter_delta_l2"] for r in rows], label=group)
        stages = [r for r in evaluations if r["group"] == group]
        axes[1, 1].plot([r["stage"] for r in stages], [r["metrics"]["all"]["stable_arrival_rate"] for r in stages], marker="o", label=group)
    for ax, title in zip(axes.flat, ("Online PPO actor loss", "Online PPO critic loss", "Actor parameter change per update", "Held-out stable arrival (changing stage mix)")):
        ax.set_title(title)
        ax.set_xlabel("PPO update" if ax is not axes[1, 1] else "Stage (0 = init)")
        ax.grid(alpha=0.2)
        ax.legend(title="Initialization arm")
    axes[1, 1].set_ylim(-0.03, 1.03)
    fig.suptitle("Single-seed smoke diagnostics; no generalization claim")
    fig.savefig(plot_dir / "training_diagnostics.png", dpi=180)
    plt.close(fig)


def run_smoke(config_path: str | Path) -> dict[str, Any]:
    from .single_uav_ppo_smoke_curriculum import build_smoke_curriculum
    from .single_uav_ppo_smoke_initialization import initialize_smoke_trainer, b_pretraining_cost
    started = time.perf_counter()
    path = Path(config_path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "smoke_id", "seed", "baseline_config", "baseline_config_sha256", "sources", "data_root", "log_root", "plot_root", "calibration_steps_per_arm", "ppo_steps_per_arm", "updates_per_arm", "rollout_steps", "curriculum_version"}
    if set(spec) != required or spec["schema_version"] != "single_uav_static_ppo_smoke_v1":
        raise ValueError("unsupported smoke specification")
    if (spec["calibration_steps_per_arm"], spec["ppo_steps_per_arm"], spec["updates_per_arm"], spec["rollout_steps"]) != (288, 2160, 18, 120):
        raise ValueError("smoke budget is frozen")
    base_path = Path(spec["baseline_config"])
    if sha256_file(base_path) != spec["baseline_config_sha256"]:
        raise ValueError("frozen baseline template changed")
    base = replace(load_single_uav_ppo_config(base_path), seed=int(spec["seed"]))
    roots = [Path(spec[k]).resolve() for k in ("data_root", "log_root", "plot_root")]
    for directory in roots:
        if directory.exists():
            raise FileExistsError(f"independent output already exists: {directory}")
    for directory in roots:
        directory.mkdir(parents=True)
    data_dir, log_dir, plot_dir = roots
    protected = {str(base_path.resolve()): sha256_file(base_path), str(Path(__file__).with_name("single_uav_ppo_baseline.py").resolve()): sha256_file(Path(__file__).with_name("single_uav_ppo_baseline.py"))}
    for source in spec["sources"].values():
        if sha256_file(source["checkpoint"]) != source["sha256"]:
            raise ValueError("source checkpoint SHA mismatch")
        protected[str(Path(source["checkpoint"]).resolve())] = source["sha256"]
    old_manifest = Path("logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/workspace_baseline.json")
    if old_manifest.exists():
        prior = json.loads(old_manifest.read_text(encoding="utf-8"))
        for name in prior["protected_file_sha256"]:
            protected[str(Path(name).resolve())] = sha256_file(name)
    for root in (Path("logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/preflight_final"), Path("logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/offline_recovery_action_decomposition_fix_v1_20261010")):
        for artifact in root.rglob("*"):
            if artifact.is_file():
                protected[str(artifact.resolve())] = sha256_file(artifact)
    write_json(log_dir / "protected_inputs_before.json", protected)
    curriculum = build_smoke_curriculum(base.seed)
    write_json(data_dir / "curriculum.json", curriculum)
    manifest = {"smoke_specification": spec, "smoke_config_sha256": sha256_file(path), "baseline_template": base.to_dict(), "curriculum_sha256": canonical_hash(curriculum), "environment_contract": "unchanged StrictSingleUAVPPOEnv", "baseline_from_scratch_field_scope": "fresh trainer construction only; compatible Actor blocks subsequently warm-started", "historical_20_used_for_training_or_validation": False, "sampling_mode": "sampled", "ppo_online_teacher_samples": 0, "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1}, "reward_note": "Normal parking is report-only; strict no-obstacle environment cannot terminate with strict success or receive its bonus.", "window_cut_bootstrap": "nonterminal final transition done=False; next-state value; buffer cleared after update; never label window cut timeout"}
    write_json(data_dir / "config.lock.json", manifest)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    calibration_a = calibrate_training_only(base, curriculum["train_pool"], "A", data_dir, log_dir)
    calibration_b = calibrate_training_only(base, curriculum["train_pool"], "B", data_dir, log_dir)
    if calibration_a["stream_sha256"] != calibration_b["stream_sha256"]:
        raise AssertionError("paired calibration streams differ")
    normalizer, central, potential, provenance = fit_common_statistics(calibration_a, base, data_dir / "shared_train_statistics.json")
    del calibration_b
    resolved = replace(base, train_potential_statistics=potential)
    write_json(data_dir / "resolved_environment_config.json", resolved.to_dict())
    write_json(log_dir / "b_pretraining_cost.json", b_pretraining_cost(Path(spec["sources"]["B"]["checkpoint"])))
    updates, evaluations, summaries = [], [], {}
    update_log = CSVLog(log_dir / "ppo_updates.csv")
    evaluation_log = CSVLog(log_dir / "evaluation_summary.csv")
    for group in ("A", "B"):
        group_data, group_logs = data_dir / group, log_dir / group
        group_data.mkdir()
        group_logs.mkdir()
        env = create_single_uav_ppo_environment(resolved)
        _reset(env, curriculum["train_pool"][0], f"{group}_constructor_contract_probe")
        trainer = create_single_uav_ppo_trainer(env, resolved, normalizer=normalizer, central_normalizer=central, normalizer_provenance=provenance, action_log_path=group_logs / "sampled_execution.csv")
        init = initialize_smoke_trainer(trainer, Path(spec["sources"][group]["checkpoint"]), group=group)
        write_json(group_logs / "compatibility.json", init)
        probes = probe_actor_features(trainer, env, [curriculum["train_pool"][i] for i in (0, 6, 7)])
        write_json(group_logs / "actor_observation_contract.json", probes)
        frozen_state_hash = canonical_hash({"node": trainer.normalizer.state_dict(), "central": trainer.central_normalizer.state_dict()})
        initial_parameters = _parameter_snapshot(trainer.policy)
        audit = OptimizerAudit(trainer)
        step_log, episode_log = CSVLog(group_logs / "steps.csv"), CSVLog(group_logs / "episodes.csv")
        _save_checkpoint(group_data / "initial.pt", trainer, group, 0, 0, init, canonical_hash(manifest))
        eval_cost = {"steps": 0, "resets": 0, "evaluations": 0}
        train_cost = {"steps": 0, "resets": 0, "updates": 0}
        def evaluate(stage: int):
            kinds = {"normal", "single", "full"} if stage in (0, 3) else {"normal"} if stage == 1 else {"normal", "single"}
            selected = [s for s in curriculum["validation_pool"] if s["kind"] in kinds]
            records = []
            for index, scene in enumerate(selected):
                if scene["split"] != "validation":
                    raise AssertionError("evaluation split leakage")
                result = collect_decisions(trainer, env, resolved, scene, group=group, phase="validation", stage=stage, block_id=index, step_budget=base.max_steps, online=False, step_log=step_log, episode_log=episode_log)
                eval_cost["steps"] += result["steps"]
                eval_cost["resets"] += result["resets"]
                records.extend(result["episodes"])
            if trainer.rollout:
                raise AssertionError("validation entered PPO buffer")
            metrics = summarize_episodes(records)
            row = {"group": group, "stage": stage, "metrics": metrics, "validation_scene_ids": [s["scenario_id"] for s in selected], "ppo_updates_this_run": train_cost["updates"], "ppo_online_steps_this_run": train_cost["steps"]}
            evaluations.append(row)
            evaluation_log.append(row)
            write_json(group_logs / f"validation_stage{stage}.json", row)
            eval_cost["evaluations"] += 1
            return metrics
        print(f"{group}: initialization verified; starting held-out init evaluation", flush=True)
        evaluate(0)
        train_rows = []
        scene_by_id = {s["scenario_id"]: s for s in curriculum["train_pool"]}
        for block in curriculum["schedule"]:
            stage = int(block["stage"])
            update_number = train_cost["updates"] + 1
            scene = scene_by_id[block["scenario_id"]]
            if scene["split"] != "train":
                raise AssertionError("non-train scene used online")
            before = _parameter_snapshot(trainer.policy)
            start_steps, grad_start = audit.steps, len(audit.gradients)
            result = collect_decisions(trainer, env, resolved, scene, group=group, phase="train", stage=stage, block_id=update_number, step_budget=120, online=True, step_log=step_log, episode_log=episode_log)
            if len(trainer.rollout) != 120 or result["steps"] != 120:
                raise AssertionError("real PPO rollout budget mismatch")
            nonterminal_cut = bool(result["episodes"][-1]["rollout_window_cut"])
            if nonterminal_cut and trainer.rollout[-1].done:
                raise AssertionError("partial rollout incorrectly terminal")
            stats = trainer.update()
            if trainer.rollout:
                raise AssertionError("rollout not cleared by update")
            delta = parameter_deltas(before, trainer.policy)
            actual_steps = audit.steps - start_steps
            gradients = audit.gradients[grad_start:]
            if int(stats["samples"]) != 120 or actual_steps != int(stats["optimizer_steps"]) or actual_steps != 8:
                raise AssertionError("PPO optimizer calls/counts differ from frozen budget")
            if not all(math.isfinite(float(v)) for v in stats.values()) or not all(v > 0 for v in delta.values()):
                raise AssertionError("PPO loss/parameter change invalid")
            if any(max(g[k] for g in gradients) <= 0 for k in ("actor", "gat", "critic")):
                raise AssertionError("Actor/GAT/critic not all receiving gradients")
            train_cost["steps"] += result["steps"]
            train_cost["resets"] += result["resets"]
            train_cost["updates"] += 1
            train_rows.extend(result["episodes"])
            row = {"group": group, "stage": stage, "ppo_update_this_run": update_number, "scenario_id": scene["scenario_id"], "kind": scene["kind"], "side": scene["side"], "rollout_steps": 120, "sample_source": "fresh_online_policy_rollout", "teacher_online_samples": 0, "sampled_action_logprob_preserved": True, "buffer_size_after_update": 0, "nonterminal_window_cut": nonterminal_cut, "actual_optimizer_steps": actual_steps, "optimizer_steps_cumulative": audit.steps, **delta, **{f"{key}_gradient_norm_max": max(g[key] for g in gradients) for key in ("actor", "gat", "critic")}, **stats}
            updates.append(row)
            update_log.append(row)
            print(f"{group}: stage {stage}, PPO update {update_number}/18, real steps {train_cost['steps']}/2160, optimizer steps {audit.steps}", flush=True)
            if update_number in (6, 12, 18):
                metrics = evaluate(stage)
                _save_checkpoint(group_data / f"stage{stage}.pt", trainer, group, update_number, train_cost["steps"], init, canonical_hash(manifest))
                if canonical_hash({"node": trainer.normalizer.state_dict(), "central": trainer.central_normalizer.state_dict()}) != frozen_state_hash:
                    raise AssertionError("training statistics changed during PPO/evaluation")
                print(f"{group}: stage {stage} held-out success {metrics['all']['task_successes']}/{metrics['all']['episodes']}", flush=True)
        final_delta = parameter_deltas(initial_parameters, trainer.policy)
        _save_checkpoint(group_data / "final.pt", trainer, group, 18, 2160, init, canonical_hash(manifest))
        if train_cost["steps"] != 2160 or train_cost["updates"] != 18 or audit.steps != 144:
            raise AssertionError("completed smoke budget mismatch")
        summaries[group] = {"calibration_steps": 288, "calibration_resets": calibration_a["resets"], "ppo_online_steps": train_cost["steps"], "total_training_interactions": 288+train_cost["steps"], "ppo_updates": train_cost["updates"], "actual_optimizer_steps": audit.steps, "ppo_resets": train_cost["resets"], "evaluation_cost": eval_cost, "constructor_probe_resets": 1, "actor_feature_probe_resets": probes["reset_count"], "source_update_count_is_not_new_updates": True, "teacher_online_samples": 0, "final_parameter_change": final_delta, "train_metrics": summarize_episodes(train_rows), "final_validation": evaluations[-1]["metrics"], "normalization_unchanged": True, "all_trainable": all(p.requires_grad for p in trainer.policy.parameters())}
        write_json(group_logs / "arm_summary.json", summaries[group])
    protected_checks = {name: sha256_file(name) == digest for name, digest in protected.items()}
    write_json(log_dir / "protected_inputs_checks.json", protected_checks)
    if not all(protected_checks.values()):
        raise AssertionError("historical/frozen input modified")
    _plot_diagnostics(updates, evaluations, plot_dir)
    report = {"schema_version": "single_uav_static_ppo_smoke_result_v1", "smoke_id": spec["smoke_id"], "status": "engineering_smoke_completed", "seed": base.seed, "wall_seconds": time.perf_counter()-started, "arms": summaries, "calibration_streams_identical": True, "shared_statistics": provenance, "protected_input_files_unchanged": len(protected_checks), "same_training_schedule": True, "same_ppo_budget": True, "same_validation_rule": True, "historical_20_training_or_validation_samples": 0, "teacher_online_samples": 0, "multiseed_engineering_gate": "passed", "performance_or_generalization_gate": "not established by this smoke", "multiseed_entry_conditions": ["keep the frozen physical/success/action contract, disjoint validation and retained balanced curriculum", "preregister independent seeds, larger equal per-arm online budgets and separate pretraining/calibration/evaluation costs", "reuse train-only normalization per seed, audit Actor/GAT/critic updates and sampled logprobs", "review failures, collision and raw-action danger by scene type before interpreting safety or performance; zero collisions with projection is not proof of raw-policy safety"], "config_path": str(path), "data_root": str(data_dir), "log_root": str(log_dir), "plot_root": str(plot_dir), "no_generalization_claim": True}
    write_json(log_dir / "smoke_report.json", report)
    print(json.dumps({"status": report["status"], "report": str(log_dir / 'smoke_report.json')}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/configs/single_uav_static_ppo_smoke_v1_seed2026.json"))
    args = parser.parse_args()
    run_smoke(args.config)


if __name__ == "__main__":
    main()
