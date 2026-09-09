"""Tests for the paired cooperative EKF single-update quality audit."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.experiments.cooperative_update_quality_audit import (
    CooperativeUpdateQualityAuditConfig,
    PRACTICAL_EVENT_THRESHOLD_M,
    _quality_event_row,
    classify_update_effect,
)
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.sensor_model import SensorConfig
from src.synthetic.channel_model import ChannelConfig


class CooperativeUpdateQualityAuditTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            CooperativeUpdateQualityAuditConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="test",
            )

    def test_effect_classification_uses_fixed_one_millimetre_threshold(self) -> None:
        threshold = PRACTICAL_EVENT_THRESHOLD_M
        self.assertEqual(classify_update_effect(-threshold - 1e-6), "beneficial")
        self.assertEqual(classify_update_effect(threshold + 1e-6), "harmful")
        self.assertEqual(classify_update_effect(-threshold), "approximately_neutral")
        self.assertEqual(classify_update_effect(threshold), "approximately_neutral")
        self.assertEqual(classify_update_effect(0.0), "approximately_neutral")

    def test_truth_label_is_offline_and_formula_fields_close(self) -> None:
        environment = MultiUAVJointEnv(
            SimulationConfig(
                uav_count=3,
                max_steps=8,
                seed=456,
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
        environment.reset(seed=456)
        events = []
        for step in range(8):
            actions = {}
            for index, target in enumerate(environment.uav_ids):
                source = environment.uav_ids[(index - 1) % len(environment.uav_ids)]
                actions[target] = {
                    "measurement_uav_id": source,
                    "bandwidth": {
                        peer: 0.5 for peer in environment.uav_ids if peer != target
                    },
                    "delta_v": [0.0, 0.0, 0.0],
                }
            _obs, _reward, _terminated, _truncated, info = environment.step(actions)
            for index, event in enumerate(info["relative_range_update_events"]):
                events.append(
                    _quality_event_row(
                        event=event,
                        treatment="round_robin_candidate",
                        episode_id="episode_test",
                        scenario_id="scenario_test",
                        seed=456,
                        step=step,
                        event_index=index,
                    )
                )

        self.assertGreater(len(events), 0)
        for row in events:
            expected_change = (
                float(row["true_position_error_after_m"])
                - float(row["true_position_error_before_m"])
            )
            self.assertAlmostEqual(
                float(row["true_error_change_after_minus_before_m"]), expected_change
            )
            self.assertEqual(int(row["truth_used_offline_only"]), 1)
            self.assertTrue(np.isfinite(float(row["unknown_cross_bound_fraction"])))
            self.assertLessEqual(float(row["effective_variance_formula_residual"]), 1e-9)
            self.assertLessEqual(float(row["innovation_variance_formula_residual"]), 1e-9)
            self.assertLessEqual(float(row["state_correction_formula_residual"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
