from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.single_uav_scripted_feasibility import ScriptedOracleController
from src.simulation.environment import RewardConfig
from src.simulation.motion_model import MotionConfig, UAVKinematicState


def _environment(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 2.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    goal: tuple[float, float, float] = (10.0, 0.0, 2.0),
) -> SimpleNamespace:
    motion = MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0, max_speed_mps=12.0)
    reward = RewardConfig(
        single_uav_goal_threshold_m=1.0,
        single_uav_goal_speed_threshold_mps=0.5,
        single_uav_near_goal_radius_m=8.0,
        single_uav_target_speed_far_mps=6.0,
        single_uav_braking_acceleration_mps2=2.0,
    )
    state = UAVKinematicState(
        position_m=np.asarray(position, dtype=float),
        velocity_mps=np.asarray(velocity, dtype=float),
        attitude_rad=np.zeros(3, dtype=float),
        acceleration_mps2=np.zeros(3, dtype=float),
        angular_velocity_rps=np.zeros(3, dtype=float),
    )
    return SimpleNamespace(
        uav_ids=["uav_00"],
        config=SimpleNamespace(motion=motion, reward=reward),
        states={"uav_00": state},
        goals={"uav_00": np.asarray(goal, dtype=float)},
    )


def test_scripted_speed_schedule_is_positive_before_goal_and_damped_inside() -> None:
    controller = ScriptedOracleController(_environment())

    assert controller.desired_speed(9.0) == pytest.approx(6.0)
    assert 0.5 < controller.desired_speed(4.0) < 6.0
    assert 0.5 < controller.desired_speed(2.0) < controller.desired_speed(3.0)
    assert 0.0 <= controller.desired_speed(0.5) < 0.5


def test_scripted_action_uses_goal_direction_and_respects_delta_v_limit() -> None:
    environment = _environment(
        position=(0.0, 0.0, 2.0),
        velocity=(5.0, 0.0, 0.0),
        goal=(4.0, 0.0, 2.0),
    )
    action, diagnostics = ScriptedOracleController(environment).action()
    delta_v = np.asarray(action["uav_00"]["delta_v"], dtype=float)

    assert delta_v[0] < 0.0
    assert np.linalg.norm(delta_v) <= 0.6 + 1.0e-12
    assert diagnostics["controller_delta_v_capped"] == 1
    assert diagnostics["demanded_braking_delta_v_mps"] > 0.0
    assert action["uav_00"]["measurement_uav_id"] == "none"


def test_scripted_action_is_forward_before_the_braking_region() -> None:
    environment = _environment(
        position=(0.0, 0.0, 2.0),
        velocity=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 2.0),
    )
    action, diagnostics = ScriptedOracleController(environment).action()

    assert action["uav_00"]["delta_v"][0] > 0.0
    assert diagnostics["desired_speed_mps"] == pytest.approx(6.0)
    assert diagnostics["demanded_braking_delta_v_mps"] == pytest.approx(0.0)


def test_scripted_controller_requires_one_uav() -> None:
    environment = _environment()
    environment.uav_ids = ["uav_00", "uav_01"]

    with pytest.raises(ValueError, match="one UAV"):
        ScriptedOracleController(environment)
