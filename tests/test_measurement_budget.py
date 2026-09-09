"""Tests for the fixed train-fitted measurement budget."""

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


def _action(env: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
    return {
        uav_id: {
            "measurement_uav_id": "uav_01" if uav_id == "uav_00" else "none",
            "bandwidth": {
                target: 1.0 / (len(env.uav_ids) - 1)
                for target in env.uav_ids
                if target != uav_id
            },
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in env.uav_ids
    }


class MeasurementBudgetTests(unittest.TestCase):
    def test_budget_configuration_requires_complete_nonnegative_map(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(
                uav_count=3,
                measurement_budget_enabled=True,
            )
        with self.assertRaises(ValueError):
            SimulationConfig(
                uav_count=3,
                measurement_budget_enabled=True,
                measurement_budget_by_uav={"uav_00": 1, "uav_01": -1, "uav_02": 1},
            )

    def test_budget_consumption_masks_measurement_after_exhaustion(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=3,
                gnss_mode="denied",
                measurement_budget_enabled=True,
                measurement_budget_by_uav={"uav_00": 1, "uav_01": 0, "uav_02": 0},
            )
        )
        observations, reset_info = env.reset(seed=2026, scenario_config=_scenario())
        self.assertTrue(reset_info["measurement_budget_enabled"])
        self.assertEqual(observations["uav_00"]["measurement_budget_remaining"], 1.0)

        next_observations, _, _, _, info = env.step(_action(env))
        uav00_event = next(
            row for row in info["measurement_budget_events"] if row["uav_id"] == "uav_00"
        )
        self.assertEqual(uav00_event["measurement_uav_id"], "uav_01")
        self.assertEqual(uav00_event["budget_consumed"], 1)
        self.assertEqual(info["measurement_budget_remaining"]["uav_00"], 0)
        self.assertEqual(next_observations["uav_00"]["measurement_budget_remaining"], 0.0)
        mask = next_observations["uav_00"]["action_mask"]
        self.assertTrue(bool(mask[len(env.uav_ids)]))
        self.assertFalse(bool(mask[: len(env.uav_ids)].any()))

        _, _, _, _, second_info = env.step(_action(env))
        second_event = next(
            row for row in second_info["measurement_budget_events"] if row["uav_id"] == "uav_00"
        )
        self.assertEqual(second_event["measurement_uav_id"], "none")
        self.assertEqual(second_event["budget_consumed"], 0)
        self.assertEqual(second_info["measurement_budget_used"]["uav_00"], 1)


if __name__ == "__main__":
    unittest.main()
