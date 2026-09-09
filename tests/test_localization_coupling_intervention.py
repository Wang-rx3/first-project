"""Tests for the cooperative localization coupling intervention."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.experiments.localization_coupling_intervention import (
    LocalizationCouplingInterventionConfig,
)
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.sensor_model import SensorConfig
from src.synthetic.channel_model import ChannelConfig


class LocalizationCouplingInterventionTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            LocalizationCouplingInterventionConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="test",
            )

    def test_switch_blocks_only_cooperative_ekf_consumption(self) -> None:
        channel = ChannelConfig(
            loss_beta0=-100.0,
            shadowing_std_los_db=0.0,
            shadowing_std_nlos_db=0.0,
            rssi_measurement_std_db=0.0,
        )
        common = dict(
            uav_count=3,
            max_steps=6,
            seed=91,
            gnss_mode="denied",
            channel=channel,
            sensor=SensorConfig(uwb_update_period_s=0.2),
        )
        enabled = MultiUAVJointEnv(SimulationConfig(**common, cooperative_ekf_enabled=True))
        disabled = MultiUAVJointEnv(SimulationConfig(**common, cooperative_ekf_enabled=False))
        enabled.reset(seed=91)
        disabled.reset(seed=91)

        def actions(environment: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
            output: dict[str, dict[str, object]] = {}
            for index, uav_id in enumerate(environment.uav_ids):
                source = environment.uav_ids[(index - 1) % len(environment.uav_ids)]
                output[uav_id] = {
                    "measurement_uav_id": source,
                    "bandwidth": {
                        target: 1.0 / (len(environment.uav_ids) - 1)
                        for target in environment.uav_ids
                        if target != uav_id
                    },
                    "delta_v": [0.0, 0.0, 0.0],
                }
            return output

        for _ in range(6):
            _obs_e, _reward_e, _term_e, _trunc_e, info_e = enabled.step(actions(enabled))
            _obs_d, _reward_d, _term_d, _trunc_d, info_d = disabled.step(actions(disabled))
            for uav_id in enabled.uav_ids:
                np.testing.assert_allclose(
                    enabled.states[uav_id].position_m,
                    disabled.states[uav_id].position_m,
                )
            for key in enabled.snapshot.metrics:
                left = enabled.snapshot.metrics[key]
                right = disabled.snapshot.metrics[key]
                self.assertAlmostEqual(left.packet_loss, right.packet_loss)
                self.assertAlmostEqual(left.delay_s, right.delay_s)
                self.assertAlmostEqual(left.aoi_s, right.aoi_s)
                self.assertAlmostEqual(left.rate_bps, right.rate_bps)

        self.assertGreater(info_e["cooperative_ekf_update_opportunities"], 0)
        self.assertEqual(
            info_e["cooperative_ekf_update_opportunities"],
            info_d["cooperative_ekf_update_opportunities"],
        )
        self.assertEqual(info_e["cooperative_ekf_updates_blocked"], 0)
        self.assertEqual(
            info_d["cooperative_ekf_updates_blocked"],
            info_e["cooperative_ekf_update_opportunities"],
        )
        self.assertGreater(sum(ekf.valid_neighbor_count for ekf in enabled.ekfs.values()), 0)
        self.assertEqual(sum(ekf.valid_neighbor_count for ekf in disabled.ekfs.values()), 0)


if __name__ == "__main__":
    unittest.main()
