"""Tests for read-only cooperative EKF update-strength diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.experiments.ekf_update_strength_audit import EKFUpdateStrengthAuditConfig
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.sensor_model import SensorConfig
from src.synthetic.channel_model import ChannelConfig


class EKFUpdateStrengthAuditTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            EKFUpdateStrengthAuditConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="test",
            )

    def test_relative_range_trace_is_closed_and_read_only(self) -> None:
        common = dict(
            uav_count=3,
            max_steps=8,
            seed=123,
            gnss_mode="denied",
            channel=ChannelConfig(
                loss_beta0=-100.0,
                shadowing_std_los_db=0.0,
                shadowing_std_nlos_db=0.0,
                rssi_measurement_std_db=0.0,
            ),
            sensor=SensorConfig(uwb_update_period_s=0.2),
        )
        audited = MultiUAVJointEnv(SimulationConfig(**common))
        control = MultiUAVJointEnv(SimulationConfig(**common))
        audited.reset(seed=123)
        control.reset(seed=123)

        def actions(environment: MultiUAVJointEnv) -> dict[str, dict[str, object]]:
            result: dict[str, dict[str, object]] = {}
            for index, target in enumerate(environment.uav_ids):
                source = environment.uav_ids[(index - 1) % len(environment.uav_ids)]
                result[target] = {
                    "measurement_uav_id": source,
                    "bandwidth": {
                        peer: 0.5 for peer in environment.uav_ids if peer != target
                    },
                    "delta_v": [0.0, 0.0, 0.0],
                }
            return result

        observed_events: list[dict[str, object]] = []
        for _ in range(8):
            _oa, _ra, _ta, _tra, info_a = audited.step(actions(audited))
            _oc, _rc, _tc, _trc, _info_c = control.step(actions(control))
            observed_events.extend(info_a["relative_range_update_events"])
            for uav_id in audited.uav_ids:
                np.testing.assert_allclose(audited.ekfs[uav_id].state, control.ekfs[uav_id].state)
                np.testing.assert_allclose(
                    audited.ekfs[uav_id].covariance,
                    control.ekfs[uav_id].covariance,
                )

        self.assertGreater(len(observed_events), 0)
        for event in observed_events:
            effective = (
                float(event["measurement_variance_m2"])
                + float(event["neighbor_direction_variance_m2"])
                + float(event["unknown_cross_bound_m2"])
            )
            self.assertAlmostEqual(float(event["effective_measurement_variance_m2"]), effective)
            self.assertAlmostEqual(
                float(event["innovation_variance_m2"]),
                float(event["local_direction_variance_m2"]) + effective,
            )
            before = np.asarray(event["state_before"], dtype=float)
            after = np.asarray(event["state_after"], dtype=float)
            correction = np.asarray(event["state_correction"], dtype=float)
            np.testing.assert_allclose(after - before, correction)
            self.assertAlmostEqual(
                float(event["position_covariance_trace_before_m2"]),
                float(event["position_covariance_trace_after_final_m2"]),
            )
            self.assertAlmostEqual(float(event["j_loc_before"]), float(event["j_loc_after"]))


if __name__ == "__main__":
    unittest.main()
