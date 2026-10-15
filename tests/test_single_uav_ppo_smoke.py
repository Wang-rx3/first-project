"""Budget, reporter and actual online update tests for the opt-in smoke runner."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.experiments.single_uav_ppo_baseline import create_single_uav_ppo_environment, create_single_uav_ppo_trainer, load_single_uav_ppo_config
from src.experiments.single_uav_ppo_smoke_curriculum import build_smoke_curriculum
from src.experiments.single_uav_ppo_smoke import (
    CSVLog, OptimizerAudit, ParkingReporter, _central, _parameter_snapshot,
    _reset, assert_online_transition, collect_decisions, fit_common_statistics,
    parameter_deltas, probe_actor_features, raw_action_diagnostics, summarize_episodes,
)


def test_parking_is_actual_continuous_and_collision_permanent():
    tracker = ParkingReporter()
    for step in range(4):
        assert not tracker.observe(0.9, 0.4, False, step)
    assert not tracker.observe(0.9, 0.51, False, 5)
    for step in range(5):
        result = tracker.observe(1.0, 0.5, False, step+6)
        assert result == (step == 4)
    assert not tracker.observe(0, 0, True, 12)
    for step in range(6):
        assert not tracker.observe(0, 0, False, step+13)


def test_sampled_pair_and_teacher_rejection():
    sample = SimpleNamespace(delta_v=np.array([0.1, 0.2, -0.1]), old_log_prob=-0.4)
    decision = SimpleNamespace(samples={"uav_00": sample}, observations={}, actions={})
    assert_online_transition(decision, sample.delta_v.copy(), -0.4)
    with pytest.raises(AssertionError):
        assert_online_transition(decision, sample.delta_v.copy(), -0.5)
    decision.observations = {"uav_00": {"teacher_action": [1, 2, 3]}}
    with pytest.raises(AssertionError, match="teacher sample"):
        assert_online_transition(decision, sample.delta_v.copy(), -0.4)


def test_real_online_rollout_cut_bootstraps_and_updates_all_modules(tmp_path):
    torch.set_num_threads(1)
    base = load_single_uav_ppo_config("data/configs/single_uav_ppo_baseline_v1.json")
    pool = build_smoke_curriculum()["train_pool"]
    placeholder = {"u_center": 0., "u_scale": 1., "c_center": 0., "c_scale": 1., "delta_u_scale": 1., "delta_goal_scale": 1.}
    env = create_single_uav_ppo_environment(replace(base, train_potential_statistics=placeholder))
    obs, info = _reset(env, pool[0], "unit_probe")
    calibration = {"observations": [], "central": [], "u": [0., 1.], "c": [5., 6.], "du": [0.1, 0.2], "dg": [0.1, 0.2], "stream_sha256": "unit"}
    for scene in pool:
        observations, state = _reset(env, scene, "unit_statistics")
        calibration["observations"].append(copy.deepcopy(observations))
        calibration["central"].append(_central(state, base))
    norm, central, potential, provenance = fit_common_statistics(calibration, base, tmp_path / "train.json")
    config = replace(base, train_potential_statistics=potential)
    env = create_single_uav_ppo_environment(config)
    _reset(env, pool[0], "unit_initial")
    trainer = create_single_uav_ppo_trainer(env, config, normalizer=norm, central_normalizer=central, normalizer_provenance=provenance, action_log_path=tmp_path / "actions.csv")
    probes = probe_actor_features(trainer, env, [pool[i] for i in (0, 6, 7)])
    assert probes["environment_steps"] == 0
    assert probes["scenes"][0]["physical_obstacle_count"] == 0
    assert probes["scenes"][1]["physical_obstacle_count"] == 1
    before = _parameter_snapshot(trainer.policy)
    audit = OptimizerAudit(trainer)
    steps, episodes = CSVLog(tmp_path / "steps.csv"), CSVLog(tmp_path / "episodes.csv")
    result = collect_decisions(trainer, env, config, pool[0], group="A", phase="train", stage=1, block_id=1, step_budget=8, online=True, step_log=steps, episode_log=episodes)
    assert result["steps"] == len(trainer.rollout) == 8
    assert result["episodes"][-1]["rollout_window_cut"]
    assert not result["episodes"][-1]["timeout_failure"]
    assert not trainer.rollout[-1].done
    assert np.isfinite(trainer._next_value(trainer.rollout[-1]))
    stats = trainer.update()
    assert stats["samples"] == 8
    assert audit.steps == stats["optimizer_steps"] == 4
    assert not trainer.rollout
    assert all(v > 0 for v in parameter_deltas(before, trainer.policy).values())
    assert all(np.isfinite(float(v)) for v in stats.values())
    result = collect_decisions(trainer, env, config, pool[0], group="A", phase="validation", stage=1, block_id=2, step_budget=6, online=False, step_log=steps, episode_log=episodes)
    assert not trainer.rollout
    assert result["steps"] == 6
    assert audit.steps == 4


def test_raw_candidate_diagnostics_are_pure_and_normal_na():
    base = load_single_uav_ppo_config("data/configs/single_uav_ppo_baseline_v1.json")
    env = create_single_uav_ppo_environment(base)
    pool = build_smoke_curriculum()["train_pool"]
    _reset(env, pool[0], "normal")
    assert raw_action_diagnostics(env, [0, 0, 0])["raw_action_physical_danger"] is None
    _reset(env, pool[6], "physical")
    uid = env.uav_ids[0]
    obstacle = env.obstacles[0]
    env.states[uid].position_m = np.array([obstacle.center_m[0]-obstacle.size_m[0]/2-2.1, obstacle.center_m[1], obstacle.center_m[2]])
    env.states[uid].velocity_mps = np.array([2., 0., 0.])
    state_before = env.states[uid].copy()
    assert raw_action_diagnostics(env, [0.6, 0, 0])["raw_action_physical_danger"] is True
    np.testing.assert_array_equal(env.states[uid].position_m, state_before.position_m)
    np.testing.assert_array_equal(env.states[uid].velocity_mps, state_before.velocity_mps)


def test_physical_metric_denominator_excludes_normal():
    def row(kind, bypass):
        return dict(kind=kind, steps=10, stable_arrival=False, task_success=False, physical_bypass=bypass, collision=False, timeout_failure=True, time_limit_reached_raw=True, rollout_window_cut=False, raw_action_danger_steps=1 if kind != "normal" else 0, raw_action_physical_denominator=10 if kind != "normal" else 0, safety_projection_steps=2, boundary_projection_steps=0)
    result = summarize_episodes([row("normal", None), row("single", True)])
    assert result["all"]["physical_bypass_rate"] == 1
    assert result["all"]["raw_action_danger_rate"] == 0.1
    assert result["normal"]["physical_bypass_rate"] is None
    assert result["normal"]["raw_action_danger_rate"] is None
