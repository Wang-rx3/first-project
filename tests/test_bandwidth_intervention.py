"""Unit tests for the validation-only bandwidth intervention."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.bandwidth_intervention import (
    BandwidthInterventionConfig,
    intervene_bandwidth,
    load_recorded_actions,
)


class BandwidthInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorded = {
            "measurement_uav_id": "uav_02",
            "delta_v": [0.1, -0.2, 0.3],
            "bandwidth": {"uav_01": 0.2, "uav_02": 0.8},
        }

    def test_only_bandwidth_changes(self) -> None:
        for treatment in ("recorded_mappo", "equal_share", "permuted_share"):
            action, metadata = intervene_bandwidth(
                self.recorded, ("uav_01", "uav_02"), treatment
            )
            self.assertEqual(action["measurement_uav_id"], "uav_02")
            np.testing.assert_allclose(action["delta_v"], self.recorded["delta_v"])
            self.assertGreaterEqual(metadata["intervened_total"], 0.0)
            self.assertLessEqual(metadata["intervened_total"], 1.0 + 1e-12)

    def test_equal_and_permuted_allocations(self) -> None:
        equal, equal_meta = intervene_bandwidth(
            self.recorded, ("uav_01", "uav_02"), "equal_share"
        )
        permuted, permuted_meta = intervene_bandwidth(
            self.recorded, ("uav_01", "uav_02"), "permuted_share"
        )
        self.assertEqual(equal["bandwidth"], {"uav_01": 0.5, "uav_02": 0.5})
        self.assertEqual(permuted["bandwidth"], {"uav_01": 0.8, "uav_02": 0.2})
        self.assertTrue(equal_meta["effective_intervention"])
        self.assertTrue(permuted_meta["effective_intervention"])
        self.assertAlmostEqual(permuted_meta["recorded_total"], permuted_meta["intervened_total"])

    def test_one_or_zero_target_is_not_informative(self) -> None:
        for targets in ((), ("uav_01",)):
            _action, metadata = intervene_bandwidth(self.recorded, targets, "equal_share")
            self.assertFalse(metadata["eligible_for_intervention"])
            self.assertFalse(metadata["effective_intervention"])

    def test_test_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BandwidthInterventionConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="test",
            )

    def test_recorded_actions_are_grouped_by_window(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bandwidth_actions_") as temporary:
            episode = Path(temporary)
            rows = [
                {
                    "applied_window_start": start,
                    "uav_id": f"uav_{uav:02d}",
                    "action_raw": json.dumps(
                        {
                            "measurement_uav_id": "none",
                            "delta_v": [start, 0.0, 0.0],
                            "bandwidth": {f"uav_{(uav + 1) % 3:02d}": 1.0},
                        }
                    ),
                }
                for start in (0.0, 0.2)
                for uav in range(3)
            ]
            import csv

            with (episode / "actions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            grouped = load_recorded_actions(episode)
            self.assertEqual(len(grouped), 2)
            self.assertEqual(set(grouped[0]), {"uav_00", "uav_01", "uav_02"})
            self.assertAlmostEqual(grouped[1]["uav_00"]["delta_v"][0], 0.2)


if __name__ == "__main__":
    unittest.main()
