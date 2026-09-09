"""Regression tests for the optional per-link UWB timing mask."""

from __future__ import annotations

import unittest

from src.simulation.environment import MultiUAVJointEnv, SimulationConfig


def _scenario() -> dict[str, object]:
    return {
        "initial_positions": {
            "uav_00": [100.0, 100.0, 50.0],
            "uav_01": [110.0, 100.0, 50.0],
            "uav_02": [120.0, 100.0, 50.0],
        },
        "initial_velocities": {
            "uav_00": [0.0, 0.0, 0.0],
            "uav_01": [0.0, 0.0, 0.0],
            "uav_02": [0.0, 0.0, 0.0],
        },
        "obstacle_specs": [],
    }


class MeasurementUpdateEligibilityTests(unittest.TestCase):
    def test_next_window_mask_respects_sensor_period_and_keeps_none(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=10,
                gnss_mode="denied",
                measurement_update_eligibility_enabled=True,
            )
        )
        observations, info = env.reset(seed=2026, scenario_config=_scenario())
        self.assertTrue(info["measurement_update_eligibility_enabled"])

        # The next decision is at t=0.4 s.  A link updated at t=0.2 s must
        # wait until t=1.0 s because the default UWB period is 0.8 s.
        env.uwb_update_times[("uav_01", "uav_00")] = 0.2
        env.timestep = 1
        observations = env._observations()
        source_index = env.uav_ids.index("uav_01")
        self.assertFalse(bool(observations["uav_00"]["measurement_update_eligibility"][source_index]))
        self.assertFalse(bool(observations["uav_00"]["action_mask"][source_index]))
        self.assertTrue(bool(observations["uav_00"]["action_mask"][len(env.uav_ids)]))

    def test_bypass_is_normalized_to_none_with_a_timing_reason(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=10,
                gnss_mode="denied",
                measurement_update_eligibility_enabled=True,
            )
        )
        env.reset(seed=2026, scenario_config=_scenario())
        env.uwb_update_times[("uav_01", "uav_00")] = 0.2
        env.timestep = 1
        normalized = env._normalize_actions(
            {
                "uav_00": {
                    "measurement_uav_id": "uav_01",
                    "bandwidth": {},
                    "delta_v": [0.0, 0.0, 0.0],
                }
            }
        )
        self.assertEqual(normalized["uav_00"]["measurement_uav_id"], "none")
        self.assertEqual(
            env.invalid_measurement_action_events[-1]["reason"],
            "pre_action_update_period_not_elapsed",
        )

    def test_disabled_mode_preserves_historical_observation_contract(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=10, gnss_mode="denied")
        )
        observations, info = env.reset(seed=2026, scenario_config=_scenario())
        self.assertFalse(info["measurement_update_eligibility_enabled"])
        self.assertEqual(
            observations["uav_00"]["action_mask"].tolist(),
            env.graph_batch.local_observations["uav_00"]["action_mask"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
