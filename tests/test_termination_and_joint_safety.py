"""Tests for versioned termination attribution and joint safety execution."""

from __future__ import annotations

import unittest

import numpy as np

from src.experiments.termination import (
    classify_episode_outcome,
    outcome_marker,
)
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.motion_model import MotionConfig
from src.simulation.obstacle_model import (
    SphereObstacle,
    joint_motion_collision_diagnostics,
    segment_minimum_distance,
)


class TerminationAttributionTests(unittest.TestCase):
    def test_collision_short_episode_is_not_also_early_termination(self) -> None:
        row = {
            "termination_reason": "collision",
            "terminated": True,
            "truncated": False,
            "collision_occurred": True,
            "window_count": 20,
        }
        self.assertEqual(classify_episode_outcome(row, max_steps=50), "collision")
        self.assertEqual(
            outcome_marker(classify_episode_outcome(row, max_steps=50)),
            "X",
        )

    def test_time_limit_is_normal_and_unmarked(self) -> None:
        row = {
            "termination_reason": "time_limit",
            "terminated": False,
            "truncated": True,
            "collision_occurred": False,
            "window_count": 50,
        }
        self.assertEqual(classify_episode_outcome(row, max_steps=50), "time_limit")
        self.assertIsNone(
            outcome_marker(classify_episode_outcome(row, max_steps=50))
        )

    def test_non_collision_termination_has_triangle_marker(self) -> None:
        row = {
            "termination_reason": "task_complete",
            "terminated": True,
            "truncated": False,
            "collision_occurred": False,
            "window_count": 12,
        }
        self.assertEqual(
            classify_episode_outcome(row, max_steps=50),
            "early_termination",
        )
        self.assertEqual(
            outcome_marker(classify_episode_outcome(row, max_steps=50)),
            "^",
        )

    def test_unexplained_short_episode_is_not_inferred(self) -> None:
        row = {
            "termination_reason": "",
            "terminated": False,
            "truncated": False,
            "collision_occurred": False,
            "window_count": 12,
        }
        self.assertEqual(
            classify_episode_outcome(row, max_steps=50),
            "unknown_short",
        )
        self.assertIsNone(
            outcome_marker(classify_episode_outcome(row, max_steps=50))
        )


class JointSafetyGeometryTests(unittest.TestCase):
    def test_segment_distance_detects_crossing_paths(self) -> None:
        distance = segment_minimum_distance(
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(distance, 0.0, places=8)

    def test_joint_projection_is_symmetric_for_simultaneous_collision(self) -> None:
        motion = MotionConfig(
            dt_s=1.0,
            max_speed_mps=12.0,
            max_acceleration_mps2=12.0,
            safety_distance_m=2.0,
        )
        common = dict(
            uav_count=3,
            max_steps=2,
            gnss_mode="denied",
            motion=motion,
        )
        legacy = MultiUAVJointEnv(SimulationConfig(**common))
        joint = MultiUAVJointEnv(
            SimulationConfig(**common, safety_execution_mode="joint_swept_v1")
        )
        for environment in (legacy, joint):
            environment.reset(seed=777, scenario_config={"obstacle_specs": []})
            environment.obstacles = []
            environment.states["uav_00"].position_m[:] = [40.0, 40.0, 10.0]
            environment.states["uav_01"].position_m[:] = [50.0, 40.0, 10.0]
            environment.states["uav_00"].velocity_mps[:] = [12.0, 0.0, 0.0]
            environment.states["uav_01"].velocity_mps[:] = [-12.0, 0.0, 0.0]

        action = {
            f"uav_{index:02d}": {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [0.0, 0.0, 0.0],
            }
            for index in range(3)
        }
        _legacy_obs, _legacy_rewards, legacy_terminated, _legacy_truncated, legacy_info = legacy.step(action)
        _joint_obs, _joint_rewards, joint_terminated, _joint_truncated, joint_info = joint.step(action)
        self.assertFalse(legacy_terminated)
        self.assertFalse(joint_terminated)
        self.assertTrue(joint_info["pre_projection_collision"])
        self.assertGreaterEqual(joint_info["safety_projection_count"], 2)
        self.assertEqual(legacy_info["safety_projection_count"], 0)

    def test_joint_diagnostic_handles_moving_obstacle(self) -> None:
        obstacle = SphereObstacle(
            "moving",
            np.array([0.0, -4.0, 10.0]),
            1.0,
            np.array([0.0, 20.0, 0.0]),
        )
        result = joint_motion_collision_diagnostics(
            {"uav_00": np.array([0.0, 0.0, 10.0])},
            {"uav_00": np.array([1.0, 0.0, 10.0])},
            [obstacle],
            [SphereObstacle(
                "moving",
                np.array([0.0, 0.0, 10.0]),
                1.0,
                np.array([0.0, 20.0, 0.0]),
            )],
            dt_s=0.2,
            bounds_min=np.array([0.0, 0.0, 2.0]),
            bounds_max=np.array([200.0, 200.0, 60.0]),
            safety_distance_m=2.0,
        )
        self.assertTrue(result["collision"])
        self.assertTrue(result["obstacle_conflicts"])


if __name__ == "__main__":
    unittest.main()
