"""Tests for the paired cooperative-measurement density intervention."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.experiments.measurement_density_intervention import (
    MeasurementDensityInterventionConfig,
    actor_visible_measurement_candidates,
    intervene_measurement_selection,
)
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.sensor_model import SensorConfig
from src.synthetic.channel_model import ChannelConfig


class MeasurementDensityInterventionTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            MeasurementDensityInterventionConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="test",
            )

    def test_round_robin_uses_only_current_actor_mask(self) -> None:
        uav_ids = ("uav_00", "uav_01", "uav_02")
        observations = {
            "uav_00": {
                "action_mask": np.asarray([False, True, True, True]),
                "candidate_measurement_sources": ["uav_02", "uav_01"],
            },
            "uav_01": {
                "action_mask": np.asarray([True, False, False, True]),
                "candidate_measurement_sources": ["uav_00"],
            },
            "uav_02": {
                "action_mask": np.asarray([False, False, False, True]),
                "candidate_measurement_sources": [],
            },
        }
        recorded = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {peer: 0.5 for peer in uav_ids if peer != uav_id},
                "delta_v": [0.1, 0.2, 0.3],
            }
            for uav_id in uav_ids
        }
        cursors = {uav_id: 0 for uav_id in uav_ids}
        first, first_meta = intervene_measurement_selection(
            recorded, observations, uav_ids, "round_robin_candidate", cursors
        )
        second, _second_meta = intervene_measurement_selection(
            recorded, observations, uav_ids, "round_robin_candidate", cursors
        )
        self.assertEqual(actor_visible_measurement_candidates(observations["uav_00"], uav_ids), ["uav_01", "uav_02"])
        self.assertEqual(first["uav_00"]["measurement_uav_id"], "uav_01")
        self.assertEqual(second["uav_00"]["measurement_uav_id"], "uav_02")
        self.assertEqual(first["uav_01"]["measurement_uav_id"], "uav_00")
        self.assertEqual(first["uav_02"]["measurement_uav_id"], "none")
        self.assertTrue(first_meta["uav_00"]["actor_visible_only"])
        np.testing.assert_allclose(first["uav_00"]["delta_v"], [0.1, 0.2, 0.3])
        self.assertEqual(first["uav_00"]["bandwidth"], recorded["uav_00"]["bandwidth"])

    def test_environment_reports_sequential_measurement_gate_without_state_input(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=6,
                seed=321,
                gnss_mode="denied",
                sensor=SensorConfig(uwb_update_period_s=0.2),
                channel=ChannelConfig(
                    loss_beta0=-100.0,
                    shadowing_std_los_db=0.0,
                    shadowing_std_nlos_db=0.0,
                    rssi_measurement_std_db=0.0,
                ),
            )
        )
        observations, _info = environment.reset(seed=321)
        observed_events = []
        for _step in range(6):
            actions = {}
            for target in environment.uav_ids:
                candidates = actor_visible_measurement_candidates(
                    observations[target], environment.uav_ids
                )
                actions[target] = {
                    "measurement_uav_id": candidates[0] if candidates else "none",
                    "bandwidth": {
                        peer: 0.5 for peer in environment.uav_ids if peer != target
                    },
                    "delta_v": [0.0, 0.0, 0.0],
                }
            observations, _rewards, _terminated, _truncated, info = environment.step(actions)
            observed_events.extend(info["measurement_gate_events"])

        self.assertGreater(len(observed_events), 0)
        for event in observed_events:
            chain = [
                bool(event["uwb_valid"]),
                bool(event["effective_edge"]),
                bool(event["message_arrived"]),
                bool(event["update_period_passed"]),
                bool(event["new_packet"]),
                bool(event["ekf_update_attempted"]),
                bool(event["ekf_update_accepted"]),
            ]
            for earlier, later in zip(chain, chain[1:]):
                self.assertFalse(later and not earlier)
            self.assertNotIn("true_position", event)


if __name__ == "__main__":
    unittest.main()
