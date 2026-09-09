"""Tests for the opt-in measurement-message communication overhead."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.formal import FormalExperimentConfig, _make_environment
from src.experiments.quality_aware_mappo import QualityAwareMAPPOConfig
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig


def _scenario() -> dict[str, object]:
    # Keep the vehicles above the default buildings so that this test isolates
    # queue loading rather than obstacle behavior.
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
            # uav_00 requests a measurement from uav_01.  The communication
            # resource consumed by that request is the directed uav_01 ->
            # uav_00 edge.
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


class MeasurementOverheadTests(unittest.TestCase):
    def test_measurement_mask_does_not_expose_non_potential_source(self) -> None:
        scenario = _scenario()
        scenario["initial_positions"] = {
            "uav_00": [100.0, 100.0, 50.0],
            "uav_01": [110.0, 100.0, 50.0],
            "uav_02": [190.0, 100.0, 50.0],
        }
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=1, gnss_mode="denied")
        )
        observations, _ = env.reset(seed=2026, scenario_config=scenario)
        measurement_mask = observations["uav_00"]["action_mask"]
        self.assertFalse(bool(measurement_mask[env.uav_ids.index("uav_02")]))
        self.assertTrue(bool(measurement_mask[-1]))

    def test_invalid_non_none_action_is_defensively_converted_to_none(self) -> None:
        scenario = _scenario()
        scenario["initial_positions"] = {
            "uav_00": [100.0, 100.0, 50.0],
            "uav_01": [110.0, 100.0, 50.0],
            "uav_02": [190.0, 100.0, 50.0],
        }
        env = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=1,
                gnss_mode="denied",
                measurement_overhead_enabled=True,
            )
        )
        env.reset(seed=2026, scenario_config=scenario)
        action = _action(env)
        action["uav_00"]["measurement_uav_id"] = "uav_02"
        _, _, _, _, info = env.step(action)

        self.assertEqual(info["actions"]["uav_00"]["measurement_uav_id"], "none")
        self.assertEqual(info["measurement_overhead_events"], [])
        self.assertEqual(info["invalid_measurement_action_count"], 1)
        self.assertEqual(
            info["invalid_measurement_action_events"][0]["reason"],
            "pre_action_not_actor_eligible",
        )

    def test_training_configs_propagate_overhead_switch(self) -> None:
        default_quality = QualityAwareMAPPOConfig(output_dir=Path("tmp-quality"))
        enabled_quality = QualityAwareMAPPOConfig(
            output_dir=Path("tmp-quality-enabled"),
            measurement_overhead_enabled=True,
        )
        self.assertFalse(default_quality.measurement_overhead_enabled)
        self.assertTrue(enabled_quality.measurement_overhead_enabled)

        default_formal = FormalExperimentConfig(output_dir=Path("tmp-formal"))
        enabled_formal = FormalExperimentConfig(
            output_dir=Path("tmp-formal-enabled"),
            measurement_overhead_enabled=True,
        )
        self.assertFalse(_make_environment(default_formal, seed=2026).config.measurement_overhead_enabled)
        self.assertTrue(_make_environment(enabled_formal, seed=2026).config.measurement_overhead_enabled)

    def test_default_switch_keeps_old_behavior(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=1, gnss_mode="denied")
        )
        env.reset(seed=2026, scenario_config=_scenario())
        _, _, _, _, info = env.step(_action(env))

        self.assertFalse(info["measurement_overhead_enabled"])
        self.assertEqual(info["measurement_overhead_events"], [])

    def test_enabled_switch_records_one_packet_load(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=1,
                gnss_mode="denied",
                measurement_overhead_enabled=True,
            )
        )
        env.reset(seed=2026, scenario_config=_scenario())
        _, _, _, _, info = env.step(_action(env))

        rows = info["measurement_overhead_events"]
        self.assertTrue(info["measurement_overhead_enabled"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source_uav_id"], "uav_01")
        self.assertEqual(row["target_uav_id"], "uav_00")
        self.assertEqual(row["payload_size_bytes"], 512)
        self.assertEqual(float(row["load_packets_requested"]), 1.0)
        self.assertEqual(float(row["load_packets_applied"]), 1.0)
        self.assertEqual(int(row["ekf_behavior_unchanged"]), 1)
        self.assertGreaterEqual(
            float(row["queue_with_measurement_packets"]),
            float(row["queue_before_packets"]),
        )
        self.assertTrue(
            float(row["queue_next_packets"]) >= 0.0
        )


if __name__ == "__main__":
    unittest.main()
