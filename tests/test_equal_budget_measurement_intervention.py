"""Tests for the equal-budget neighbor-selection intervention."""

from __future__ import annotations

import unittest

import numpy as np

from src.experiments.equal_budget_measurement_intervention import (
    _choose_equal_budget_actions,
)


def _thresholds() -> dict[str, object]:
    fields = (
        "rssi_dbm",
        "snr_db",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "queue_length_packets",
        "neighbor_covariance_trace_m2",
    )
    return {
        "quantiles": {
            field: {"q25": -100.0 if field in ("rssi_dbm", "snr_db") else 0.0,
                    "q75": 100.0}
            for field in fields
        }
    }


class EqualBudgetMeasurementInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uav_ids = ("uav_00", "uav_01", "uav_02")
        self.observations = {
            "uav_00": {
                "action_mask": np.asarray([False, True, True, True]),
                "candidate_measurement_sources": ["uav_01", "uav_02"],
                "candidate_incoming_links": {
                    "uav_01": {"link_state": "connected", "rssi_dbm": -50.0, "snr_db": 10.0,
                               "packet_loss": 0.0, "delay_s": 0.0, "aoi_s": 0.0, "queue_length_packets": 0.0},
                    "uav_02": {"link_state": "connected", "rssi_dbm": -45.0, "snr_db": 12.0,
                               "packet_loss": 0.0, "delay_s": 0.0, "aoi_s": 0.0, "queue_length_packets": 0.0},
                },
                "neighbor_messages": {"uav_01": {"covariance": np.eye(6)}, "uav_02": {"covariance": 2.0 * np.eye(6)}},
            },
            "uav_01": {"action_mask": np.asarray([True, False, False, True]), "candidate_measurement_sources": ["uav_00"]},
            "uav_02": {"action_mask": np.asarray([True, True, False, True]), "candidate_measurement_sources": ["uav_00", "uav_01"]},
        }
        self.recorded = {
            "uav_00": {"measurement_uav_id": "uav_01", "bandwidth": {"uav_01": 0.5, "uav_02": 0.5}, "delta_v": [0.1, 0.2, 0.3]},
            "uav_01": {"measurement_uav_id": "none", "bandwidth": {"uav_00": 0.5, "uav_02": 0.5}, "delta_v": [0.0, 0.0, 0.0]},
            "uav_02": {"measurement_uav_id": "uav_00", "bandwidth": {"uav_00": 0.5, "uav_01": 0.5}, "delta_v": [-0.1, 0.0, 0.1]},
        }

    def test_round_robin_preserves_none_pattern_and_controls(self) -> None:
        actions, metadata = _choose_equal_budget_actions(
            self.recorded, self.observations, self.uav_ids,
            "equal_budget_round_robin", _thresholds(), {uav_id: 0 for uav_id in self.uav_ids},
        )
        for uav_id in self.uav_ids:
            self.assertEqual(metadata[uav_id]["recorded_non_none"], metadata[uav_id]["selected_non_none"])
            self.assertEqual(metadata[uav_id]["budget_preserved"], 1)
            np.testing.assert_allclose(actions[uav_id]["delta_v"], self.recorded[uav_id]["delta_v"])
            self.assertEqual(actions[uav_id]["bandwidth"], self.recorded[uav_id]["bandwidth"])
        self.assertEqual(actions["uav_01"]["measurement_uav_id"], "none")

    def test_quality_aware_choice_is_actor_visible_and_equal_budget(self) -> None:
        actions, metadata = _choose_equal_budget_actions(
            self.recorded, self.observations, self.uav_ids,
            "equal_budget_quality_aware", _thresholds(), {uav_id: 0 for uav_id in self.uav_ids},
        )
        self.assertEqual(actions["uav_01"]["measurement_uav_id"], "none")
        self.assertTrue(metadata["uav_00"]["selected_measurement_uav_id"] in {"uav_01", "uav_02"})
        self.assertTrue(metadata["uav_02"]["selected_measurement_uav_id"] in {"uav_00", "uav_01"})
        self.assertTrue(all(metadata[uav_id]["budget_preserved"] == 1 for uav_id in self.uav_ids))


if __name__ == "__main__":
    unittest.main()
