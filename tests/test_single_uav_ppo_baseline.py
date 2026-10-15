from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.single_uav_ppo_baseline import (
    BASELINE_ID,
    EXPECTED_NODE_FEATURE_DIM,
    ActionExecutionCSVLogger,
    SingleUAVPPOConfig,
    create_single_uav_ppo_environment,
    evaluate_strict_success,
    load_single_uav_ppo_config,
    prepare_single_uav_ppo,
    verify_actor_observation_contract,
)
from src.simulation.environment import MultiUAVJointEnv
from src.simulation.motion_model import UAVKinematicState


def _config() -> SingleUAVPPOConfig:
    return SingleUAVPPOConfig()


def _fixture() -> dict[str, object]:
    return {
        "initial_positions": {"uav_00": [30.0, 100.0, 10.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [170.0, 100.0, 10.0]},
        "obstacle_specs": [
            {
                "geometry_type": "box",
                "obstacle_id": "blocking_initial_path",
                "center_m": [100.0, 100.0, 10.0],
                "size_m": [20.0, 20.0, 20.0],
                "physical_collision_enabled": True,
            }
        ],
    }


def test_json_loader_is_explicit_and_from_scratch() -> None:
    config = load_single_uav_ppo_config(Path("data/configs/single_uav_ppo_baseline_v1.json"))
    assert config.baseline_id == BASELINE_ID
    assert config.node_feature_dim == EXPECTED_NODE_FEATURE_DIM == 42
    assert config.ppo_action_logprob_mode == "sampled"
    assert config.from_scratch is True
    assert config.historical_checkpoint is None
    assert config.pass_commit_enabled is False
    assert config.lateral_exit_enabled is False
    assert config.radial_intent_enabled is False
    assert config.goal_recovery_correction_enabled is False
    assert config.goal_recovery_terminal_capture_enabled is False


def test_actual_reset_graph_contains_velocity_goal_direction_and_42_features() -> None:
    environment = create_single_uav_ppo_environment(_config())
    observations, info = environment.reset(seed=2026, scenario_config=_fixture())
    report = verify_actor_observation_contract(environment, observations)
    assert report["actual_actor_node_feature_dim"] == 42
    assert observations["uav_00"]["node_features"].shape[-1] == 42
    assert info["actor_node_feature_dim"] == 42


def test_strict_gate_distinguishes_hold_four_hold_five_and_exact_boundaries() -> None:
    result = evaluate_strict_success(
        collision=False,
        physical_pass_complete=True,
        actual_goal_distance_m=1.0,
        actual_speed_mps=0.5,
        previous_hold_steps=3,
    )
    assert result.success is False and result.hold_steps == 4
    result = evaluate_strict_success(
        collision=False,
        physical_pass_complete=True,
        actual_goal_distance_m=1.0,
        actual_speed_mps=0.5,
        previous_hold_steps=4,
    )
    assert result.success is True and result.hold_steps == 5
    result = evaluate_strict_success(
        collision=False,
        physical_pass_complete=True,
        actual_goal_distance_m=np.nextafter(1.0, np.inf),
        actual_speed_mps=0.5,
        previous_hold_steps=4,
    )
    assert result.success is False and result.hold_steps == 0


def test_strict_gate_collision_is_permanent_and_requires_physical_pass() -> None:
    first = evaluate_strict_success(
        collision=True,
        physical_pass_complete=True,
        actual_goal_distance_m=0.0,
        actual_speed_mps=0.0,
        previous_hold_steps=4,
    )
    assert first.collision_permanent_failure and not first.success
    later = evaluate_strict_success(
        collision=False,
        physical_pass_complete=True,
        actual_goal_distance_m=0.0,
        actual_speed_mps=0.0,
        previous_hold_steps=4,
        collision_permanent_failure=first.collision_permanent_failure,
    )
    assert later.collision_permanent_failure and not later.success
    unpassed = evaluate_strict_success(
        collision=False,
        physical_pass_complete=False,
        actual_goal_distance_m=0.0,
        actual_speed_mps=0.0,
        previous_hold_steps=4,
    )
    assert unpassed.hold_steps == 0 and not unpassed.success


def test_isolated_env_path_uses_truth_state_and_one_shot_preclip_bonus(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = create_single_uav_ppo_environment(_config())
    environment.reset(seed=2026, scenario_config=_fixture())
    uid = "uav_00"
    environment._strict_passed_obstacles = set(environment._strict_relevant_obstacles)
    environment._strict_hold_steps = 3
    environment.ekfs[uid].state[:3] = np.asarray([30.0, 100.0, 10.0])

    def fake_base_step(self, action):
        info = {
            "collision": False,
            "task_complete": False,
            "terminated": False,
            "truncated": True,
            "learning_rewards": {uid: 0.2},
            "learning_reward_components": {
                "learning_reward_preclip_by_uav": {uid: 0.2},
                "learning_reward_total_by_uav": {uid: 0.2},
            },
        }
        return {}, {uid: 1.0}, False, True, info

    monkeypatch.setattr(MultiUAVJointEnv, "step", fake_base_step)
    environment.states[uid].position_m = environment._strict_goal + np.asarray([1.0, 0.0, 0.0])
    environment.states[uid].velocity_mps = np.asarray([0.5, 0.0, 0.0])
    _obs, _rewards, terminated, truncated, info = environment.step({uid: {}})
    assert not terminated and truncated
    assert info["single_uav_goal_condition"] is True
    assert info["single_uav_goal_reached"] is False
    assert info["single_uav_stable_steps"] == 4
    assert info["single_uav_goal_in_radius"] is True
    assert info["strict_goal_distance_m_actual"] == pytest.approx(1.0)
    assert info["learning_rewards"][uid] == pytest.approx(0.2)

    environment._strict_hold_steps = 4
    _obs, _rewards, terminated, truncated, info = environment.step({uid: {}})
    assert terminated and not truncated
    assert info["termination_reason"] == "strict_goal_reached"
    assert info["single_uav_goal_reached"] is True
    assert info["single_uav_goal_bonus_triggered"] is True
    # Base preclip 0.2 plus strict bonus 5 is clipped once by the learning
    # channel's configured bound; it is not added to reporting reward.
    assert info["learning_rewards"][uid] == pytest.approx(1.0)
    assert info["strict_success_bonus"] == pytest.approx(5.0)


def test_prepare_is_reset_only_and_refuses_overwrite(tmp_path: Path) -> None:
    config_path = Path("data/configs/single_uav_ppo_baseline_v1.json")
    output = tmp_path / "preflight"
    result = prepare_single_uav_ppo(config_path, output)
    assert result["summary"]["environment_steps"] == 0
    assert result["summary"]["formal_training"] is False
    assert (output / "config.lock").exists()
    payload = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert payload["normalizer_status"] == "pending_train_fit"
    with pytest.raises(FileExistsError):
        prepare_single_uav_ppo(config_path, output)


def test_action_logger_separates_safety_from_base_execution(tmp_path: Path) -> None:
    from src.simulation.motion_model import MotionConfig

    previous = UAVKinematicState(
        position_m=np.array([10.0, 10.0, 10.0]),
        velocity_mps=np.array([1.0, 0.0, 0.0]),
        attitude_rad=np.zeros(3),
        acceleration_mps2=np.zeros(3),
        angular_velocity_rps=np.zeros(3),
    )
    path = tmp_path / "actions.csv"
    logger = ActionExecutionCSVLogger(path)
    row = logger.append(
        step=0,
        uav_id="uav_00",
        previous_state=previous,
        sampled_delta_v=[0.6, 0.0, 0.0],
        execution={
            "delta_v_applied": np.array([0.4, 0.0, 0.0]),
            "safety_projected": False,
            "boundary_projected": False,
        },
        motion=MotionConfig(),
    )
    assert row["safety_correction_norm_mps"] == 0.0
    assert row["total_correction_norm_mps"] > 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1
