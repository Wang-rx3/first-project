"""Integration checks with reset fixtures and a mocked transition boundary."""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.single_uav_ppo_baseline import (
    FEATURE_INDEX_MAP, SingleUAVPPOConfig, FrozenObservationNormalizer,
    FrozenFeatureNormalizer, create_single_uav_ppo_environment,
    create_single_uav_ppo_trainer, evaluate_strict_success,
    _controlled_reset_perturbation_probe,
)
from src.learning.mappo import central_state_to_vector
from src.learning.normalization import ObservationNormalizer, FeatureNormalizer
from src.simulation.environment import MultiUAVJointEnv


def scene(physical=True):
    return {
        "episode_id": "unit_fixture_1",
        "initial_positions": {"uav_00": [30.0, 100.0, 10.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [170.0, 100.0, 10.0]},
        "obstacle_specs": [{
            "geometry_type": "box", "obstacle_id": "wall",
            "center_m": [100.0, 100.0, 10.0], "size_m": [20.0, 20.0, 20.0],
            "physical_collision_enabled": physical,
        }],
    }


@pytest.mark.parametrize("name,value", [
    ("dt_s", 0.25), ("max_speed_mps", 10.0), ("max_acceleration_mps2", 4.0),
    ("single_uav_goal_threshold_m", 1.1), ("single_uav_goal_speed_threshold_mps", 0.6),
    ("strict_success_hold_steps", 4), ("pass_commit_enabled", True),
    ("lateral_exit_enabled", True), ("radial_intent_enabled", True),
    ("goal_recovery_correction_enabled", True), ("goal_recovery_terminal_capture_enabled", True),
    ("ppo_action_logprob_mode", "executed"),
])
def test_baseline_rejects_contract_drift(name, value):
    with pytest.raises(ValueError):
        replace(SingleUAVPPOConfig(), **{name: value})


@pytest.mark.parametrize("distance,speed", [(1.001, 0.5), (1.0, 0.501), (float("nan"), 0.0)])
def test_hold_resets_and_next_qualifying_decision_starts_at_one(distance, speed):
    broken = evaluate_strict_success(collision=False, physical_pass_complete=True,
        actual_goal_distance_m=distance, actual_speed_mps=speed, previous_hold_steps=4)
    assert not broken.success and broken.hold_steps == 0
    next_step = evaluate_strict_success(collision=False, physical_pass_complete=True,
        actual_goal_distance_m=1.0, actual_speed_mps=0.5, previous_hold_steps=broken.hold_steps)
    assert next_step.hold_steps == 1 and not next_step.success


def test_pass_uses_entity_geometry_and_does_not_require_historical_clearance_floor():
    env = create_single_uav_ppo_environment(SingleUAVPPOConfig())
    env.reset(seed=2026, scenario_config=scene())
    env._update_physical_pass(np.array([112.1, 100, 10]), np.array([114, 100, 10]))
    assert not env.strict_status()["strict_physical_pass_complete"]
    env._update_physical_pass(np.array([114, 100, 10]), np.array([116, 100, 10]))
    assert env.strict_status()["strict_physical_pass_complete"]
    assert env.strict_status()["strict_trajectory_minimum_clearance_m"] < 2.5
    env.reset(seed=2026, scenario_config=scene(False))
    env._update_physical_pass(np.array([160, 100, 10]), np.array([170, 100, 10]))
    assert not env.strict_status()["strict_physical_pass_complete"]


def mock_transition(monkeypatch):
    def stub(self, action):
        self.timestep += 1
        info = {"collision": bool(getattr(self, "unit_collision", False)),
            "task_complete": False, "terminated": False, "truncated": self.timestep >= 120,
            "termination_reason": "time_limit" if self.timestep >= 120 else "running",
            "learning_rewards": {"uav_00": 0.2},
            "learning_reward_components": {"learning_reward_preclip_by_uav": {"uav_00": 0.2}}}
        return {}, {"uav_00": 1.0}, False, info["truncated"], info
    monkeypatch.setattr(MultiUAVJointEnv, "step", stub)


def test_truth_speed_timeout_collision_and_reset_use_isolated_gate(monkeypatch):
    env = create_single_uav_ppo_environment(SingleUAVPPOConfig())
    env.reset(seed=2026, scenario_config=scene())
    mock_transition(monkeypatch)
    env._update_physical_pass(np.array([160, 100, 10]), np.array([170, 100, 10]))
    env.states["uav_00"].position_m[:] = [172, 100, 10]
    env.ekfs["uav_00"].state[:3] = [170, 100, 10]
    env._strict_hold_steps = 4
    _, _, terminated, _, info = env.step({})
    assert not terminated and info["single_uav_stable_steps"] == 0
    env.states["uav_00"].position_m[:] = [170, 100, 10]
    env.states["uav_00"].velocity_mps[:] = [0.6, 0, 0]
    env.ekfs["uav_00"].state[3:6] = 0
    env._strict_hold_steps = 4
    _, _, terminated, _, info = env.step({})
    assert not terminated and info["single_uav_stable_steps"] == 0
    env.states["uav_00"].velocity_mps[:] = 0
    env.timestep = 119
    env._strict_hold_steps = 3
    _, _, terminated, truncated, info = env.step({})
    assert not terminated and truncated and not info["strict_success"]
    env.unit_collision = True
    _, _, terminated, truncated, info = env.step({})
    assert terminated and not truncated and not info["task_complete"]
    env.unit_collision = False
    env._strict_hold_steps = 4
    _, _, _, _, info = env.step({})
    assert not info["strict_success"] and info["strict_hold_steps"] == 0
    env.reset(seed=2026, scenario_config=scene())
    assert not env.strict_status()["strict_collision_permanent_failure"]
    assert env.strict_status()["strict_hold_steps"] == 0
    assert not env.strict_status()["strict_physical_pass_complete"]


def fitted_fixture(tmp_path):
    stats = {"u_center": 0.0, "u_scale": 1.0, "c_center": 0.0,
        "c_scale": 1.0, "delta_u_scale": 1.0, "delta_goal_scale": 1.0}
    config = replace(SingleUAVPPOConfig(), train_potential_statistics=stats)
    env = create_single_uav_ppo_environment(config)
    obs, info = env.reset(seed=2026, scenario_config=scene())
    normalizer = ObservationNormalizer().fit(obs)
    central = FeatureNormalizer().fit(central_state_to_vector(info["central_state"],
        uav_count=1, node_feature_dim=42, edge_feature_dim=11, max_obstacles=8, include_remaining_time=False))
    payload = normalizer.state_dict()
    payload["central_normalizer"] = central.state_dict()
    artifact = tmp_path / "unit_normalization.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    provenance = {"fit_split": "train", "purpose": "synthetic_unit_fixture_only",
        "feature_index_map": FEATURE_INDEX_MAP, "artifact_path": str(artifact),
        "source_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    return config, env, obs, info, normalizer, central, provenance


def test_frozen_statistics_are_bound_to_artifact_and_do_not_refit(tmp_path):
    config, env, obs, info, normalizer, central, provenance = fitted_fixture(tmp_path)
    frozen = FrozenObservationNormalizer(normalizer, provenance)
    frozen_central = FrozenFeatureNormalizer(central, provenance, central.feature_dim)
    saved = frozen.state_dict()
    normalizer.normalizers["node_features"].mean += 10
    assert frozen.state_dict() == saved
    with pytest.raises(RuntimeError):
        frozen.fit(obs)
    with pytest.raises(RuntimeError):
        frozen.update(obs)
    with pytest.raises(RuntimeError):
        frozen_central.fit([1])
    with pytest.raises(ValueError):
        FrozenObservationNormalizer(normalizer, provenance)
    with pytest.raises(ValueError):
        FrozenObservationNormalizer(ObservationNormalizer.from_state_dict(saved), dict(provenance, fit_split="test"))
    with pytest.raises(ValueError):
        FrozenObservationNormalizer(ObservationNormalizer.from_state_dict(saved), dict(provenance, source_sha256="wrong"))


def test_production_trainer_automatically_logs_sampled_execution_without_updates(tmp_path):
    config, env, obs, info, normalizer, central, provenance = fitted_fixture(tmp_path)
    before_goal = env.goals["uav_00"].copy()
    action_path = tmp_path / "sampled_execution.csv"
    trainer = create_single_uav_ppo_trainer(env, config, normalizer=normalizer,
        central_normalizer=central, normalizer_provenance=provenance, action_log_path=action_path)
    np.testing.assert_array_equal(env.goals["uav_00"], before_goal)
    decision = trainer.act(obs, info["central_state"], deterministic=False)
    sample = decision.samples["uav_00"]
    sampled, old = sample.delta_v.copy(), sample.old_log_prob
    previous = env.states["uav_00"]
    payload = {"single_uav_ppo_episode_id": "unit_fixture_1", "single_uav_ppo_step": 0,
        "single_uav_ppo_pre_action_state": {name: getattr(previous, name).tolist() for name in (
            "position_m", "velocity_mps", "attitude_rad", "acceleration_mps2", "angular_velocity_rps")},
        "actions": decision.actions, "action_execution": {"uav_00": {
            "delta_v_applied": [0, 0, 0], "safety_projected": True, "boundary_projected": False}}}
    trainer.reconcile_action_execution(decision, payload)
    np.testing.assert_array_equal(sample.delta_v, sampled)
    assert sample.old_log_prob == old and len(trainer.optimizer.state) == 0
    with action_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["logprob_action"] == "sampled"
    assert float(rows[0]["old_logprob"]) == old
    assert float(rows[0]["safety_correction_norm_mps"]) > 0
    assert not list(tmp_path.glob("*.pt"))


def test_upstream_features_survive_normalization():
    env = create_single_uav_ppo_environment(SingleUAVPPOConfig())
    obs, _ = env.reset(seed=2026, scenario_config=scene())
    result = _controlled_reset_perturbation_probe(env, obs)
    assert all(value for key, value in result.items() if key.endswith("_changed"))
