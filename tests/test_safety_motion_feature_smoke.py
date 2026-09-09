"""Tests for execution-visible dynamic-obstacle motion features and smoke wiring."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.motion_model import UAVKinematicState
from src.simulation.obstacle_model import BoxObstacle, SphereObstacle


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SCENARIO = Path(
    "data/synthetic/safety_risk_screening_seed2026/scenarios/scenario_2026.json"
)


def _scenario() -> dict[str, object]:
    return json.loads(SCENARIO.read_text(encoding="utf-8"))


def _state(position: tuple[float, float, float]) -> UAVKinematicState:
    return UAVKinematicState(
        position_m=np.asarray(position, dtype=float),
        velocity_mps=np.zeros(3, dtype=float),
        attitude_rad=np.zeros(3, dtype=float),
        acceleration_mps2=np.zeros(3, dtype=float),
        angular_velocity_rps=np.zeros(3, dtype=float),
    )


class SafetyMotionFeatureTests(unittest.TestCase):
    def test_safety_scenario_exposes_eight_obstacle_values_and_41_nodes(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=2,
                gnss_mode="denied",
                dynamic_obstacle_motion_features_enabled=True,
            )
        )
        observations, _info = environment.reset(seed=2026, scenario_config=_scenario())
        for observation in observations.values():
            self.assertEqual(observation["obstacle_features"].shape, (8,))
            self.assertEqual(observation["node_features"].shape[1], 41)
            self.assertTrue(np.isfinite(observation["obstacle_features"]).all())
            self.assertTrue(np.isfinite(observation["node_features"]).all())
        self.assertTrue(
            any(observation["obstacle_features"][7] == 1.0 for observation in observations.values())
        )

    def test_approaching_dynamic_obstacle_has_positive_closing_rate_and_lower_prediction(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=2,
                dynamic_obstacle_motion_features_enabled=True,
            )
        )
        estimate = np.zeros(15, dtype=float)
        estimate[:3] = [0.0, 2.0, 2.0]
        approaching = SphereObstacle(
            "approaching",
            np.array([0.0, 10.0, 2.0]),
            1.0,
            np.array([0.0, -2.0, 0.0]),
            "dynamic_vehicle",
            True,
        )
        receding = SphereObstacle(
            "receding",
            np.array([0.0, 10.0, 2.0]),
            1.0,
            np.array([0.0, 2.0, 0.0]),
            "dynamic_vehicle",
            True,
        )
        environment.obstacles = [approaching]
        approaching_features = environment._local_obstacle_features(estimate)
        environment.obstacles = [receding]
        receding_features = environment._local_obstacle_features(estimate)
        self.assertGreater(approaching_features[5], 0.0)
        self.assertEqual(receding_features[5], 0.0)
        self.assertLess(approaching_features[6], receding_features[6])
        self.assertTrue(np.isfinite(approaching_features).all())

    def test_missing_dynamic_motion_is_finite_and_invalid_not_safe(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=2,
                dynamic_obstacle_motion_features_enabled=True,
            )
        )
        environment.obstacles = [
            BoxObstacle(
                "building",
                np.array([20.0, 20.0, 10.0]),
                np.array([10.0, 10.0, 10.0]),
                np.zeros(3),
                "building",
                True,
            )
        ]
        features = environment._local_obstacle_features(np.r_[0.0, 0.0, 5.0, np.zeros(12)])
        self.assertEqual(features.shape, (8,))
        self.assertEqual(features[7], 0.0)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[4], 0.0)
        self.assertEqual(features[6], 0.0)

    def test_actor_observation_does_not_expose_truth_fields(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=2,
                gnss_mode="denied",
                dynamic_obstacle_motion_features_enabled=True,
            )
        )
        observations, _info = environment.reset(seed=2026, scenario_config=_scenario())
        forbidden = {
            "true_position",
            "true_error",
            "true_range",
            "collision",
            "future_state",
            "obstacle_coordinates",
        }
        for observation in observations.values():
            self.assertTrue(forbidden.isdisjoint(observation))

    def test_same_seed_produces_same_extended_observation(self) -> None:
        config = SimulationConfig(
            uav_count=3,
            max_steps=2,
            gnss_mode="denied",
            dynamic_obstacle_motion_features_enabled=True,
        )
        first = MultiUAVJointEnv(config)
        second = MultiUAVJointEnv(config)
        first_obs, _first_info = first.reset(seed=2026, scenario_config=_scenario())
        second_obs, _second_info = second.reset(seed=2026, scenario_config=_scenario())
        for uav_id in first.uav_ids:
            np.testing.assert_array_equal(
                first_obs[uav_id]["node_features"], second_obs[uav_id]["node_features"]
            )
        action = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in first.uav_ids
        }
        first_next, _r1, _t1, _tr1, _i1 = first.step(action)
        second_next, _r2, _t2, _tr2, _i2 = second.step(action)
        for uav_id in first.uav_ids:
            np.testing.assert_array_equal(
                first_next[uav_id]["node_features"], second_next[uav_id]["node_features"]
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the smoke run")
class SafetyMotionFeatureSmokeRunTests(unittest.TestCase):
    def test_small_smoke_runs_both_41_dimensional_branches(self) -> None:
        from src.experiments.safety_motion_feature_smoke import (
            SafetyMotionFeatureSmokeConfig,
            run_safety_motion_feature_smoke,
        )

        with tempfile.TemporaryDirectory(prefix="safety_motion_feature_smoke_") as temporary:
            root = Path(temporary)
            summary = run_safety_motion_feature_smoke(
                SafetyMotionFeatureSmokeConfig(
                    scenario_path=SCENARIO,
                    output_dir=root / "data",
                    logs_output=root / "logs",
                    plots_output=root / "plots",
                    max_steps=1,
                    warmup_episodes=1,
                    train_episodes=1,
                    validation_episodes=1,
                    test_episodes=1,
                    progress=False,
                )
            )
            self.assertTrue(summary["audit_ok"], summary)
            self.assertTrue((root / "logs" / "audit_report.json").exists())
            for branch in ("baseline_mappo", "budgeted_quality_aware_mappo"):
                self.assertTrue(
                    (root / "data" / branch / "scale_3" / "checkpoints" / "final.pt").exists()
                )


if __name__ == "__main__":
    unittest.main()
