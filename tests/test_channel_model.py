"""Tests for the effective statistical UAV-to-UAV channel model."""

from __future__ import annotations

import math
import random
import unittest

from src.synthetic.channel_model import ChannelConfig, compute_link_metrics


class ChannelModelTests(unittest.TestCase):
    def _metrics(self, **kwargs):
        defaults = {
            "distance_m": 20.0,
            "los": True,
            "bandwidth_ratio": 0.5,
            "queue_length_packets": 0.0,
            "previous_aoi_s": 0.0,
            "dt_s": 0.1,
            "rng": random.Random(4),
            "config": ChannelConfig(),
            "max_comm_range_m": 75.0,
        }
        defaults.update(kwargs)
        return compute_link_metrics(**defaults)

    def test_distance_and_nlos_reduce_received_power(self) -> None:
        near = self._metrics(distance_m=20.0, rng=random.Random(8))
        far = self._metrics(distance_m=50.0, rng=random.Random(8))
        nlos = self._metrics(distance_m=20.0, los=False, rng=random.Random(8))
        self.assertGreater(near.rx_power_dbm, far.rx_power_dbm)
        self.assertGreater(near.rate_bps, far.rate_bps)
        self.assertGreater(near.rx_power_dbm, nlos.rx_power_dbm)

    def test_transmit_power_improves_link(self) -> None:
        low = self._metrics(tx_power_dbm=0.0, rng=random.Random(9))
        high = self._metrics(tx_power_dbm=20.0, rng=random.Random(9))
        self.assertGreater(high.rx_power_dbm, low.rx_power_dbm)
        self.assertGreater(high.snr_db, low.snr_db)

    def test_bandwidth_share_changes_rate_and_delay(self) -> None:
        narrow = self._metrics(bandwidth_ratio=0.1, rng=random.Random(12))
        wide = self._metrics(bandwidth_ratio=0.8, rng=random.Random(12))
        self.assertGreater(wide.rate_bps, narrow.rate_bps)
        self.assertLess(wide.delay_s, narrow.delay_s)

    def test_zero_bandwidth_is_unavailable(self) -> None:
        metrics = self._metrics(bandwidth_ratio=0.0)
        self.assertEqual(metrics.rate_bps, 0.0)
        self.assertEqual(metrics.candidate_edge, 0)
        self.assertEqual(metrics.effective_edge, 0)
        self.assertEqual(metrics.packet_loss, 1.0)
        self.assertTrue(math.isfinite(metrics.delay_s))

    def test_aoi_resets_on_success_and_increments_on_failure(self) -> None:
        success_config = ChannelConfig(loss_beta0=-100.0, max_packet_loss=1.0)
        success = self._metrics(
            config=success_config,
            previous_aoi_s=5.0,
            rng=random.Random(10),
        )
        self.assertEqual(success.packet_success, 1)
        self.assertAlmostEqual(success.aoi_s, success.delay_s)

        failure_config = ChannelConfig(loss_beta0=100.0, max_packet_loss=1.0)
        failure = self._metrics(
            config=failure_config,
            previous_aoi_s=5.0,
            rng=random.Random(10),
        )
        self.assertEqual(failure.packet_success, 0)
        self.assertAlmostEqual(failure.aoi_s, 5.1)

    def test_environment_mode_defers_aoi_reset_until_arrival(self) -> None:
        success_config = ChannelConfig(loss_beta0=-100.0, max_packet_loss=1.0)
        success = self._metrics(
            config=success_config,
            previous_aoi_s=5.0,
            aoi_reset_on_success=False,
            rng=random.Random(10),
        )
        self.assertEqual(success.packet_success, 1)
        self.assertAlmostEqual(success.aoi_s, 5.1)

    def test_loss_is_bounded(self) -> None:
        for distance in (1.0, 20.0, 75.0, 100.0):
            metrics = self._metrics(distance_m=distance)
            self.assertGreaterEqual(metrics.packet_loss, 0.0)
            self.assertLessEqual(metrics.packet_loss, 1.0)


if __name__ == "__main__":
    unittest.main()
