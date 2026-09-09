"""Tests for train-fitted, actor-visible quality-aware measurement selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.quality_aware_measurement_intervention import (
    QualityAwareMeasurementInterventionConfig,
    fit_training_quality_thresholds,
    intervene_quality_measurement_selection,
    quality_candidate_scores,
)
from src.synthetic.episode_generator import _write_csv


def _thresholds() -> dict:
    return {
        "source_split": "train",
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "quantiles": {
            "rssi_dbm": {"q25": -90.0, "q75": -50.0},
            "snr_db": {"q25": 10.0, "q75": 50.0},
            "packet_loss": {"q25": 0.0, "q75": 0.4},
            "delay_s": {"q25": 0.001, "q75": 0.01},
            "aoi_s": {"q25": 0.001, "q75": 0.1},
            "queue_length_packets": {"q25": 0.0, "q75": 4.0},
            "neighbor_covariance_trace_m2": {"q25": 3.0, "q75": 12.0},
        },
    }


class QualityAwareMeasurementInterventionTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            QualityAwareMeasurementInterventionConfig(
                reference_scale=Path("reference"), output_dir=Path("output"), split="test"
            )

    def test_quality_selection_uses_visible_metrics_and_preserves_other_actions(self) -> None:
        ids = ("uav_00", "uav_01", "uav_02")
        observations = {
            "uav_00": {
                "action_mask": np.asarray([False, True, True, True]),
                "candidate_measurement_sources": ["uav_01", "uav_02"],
                "candidate_incoming_links": {
                    "uav_01": {
                        "rssi_dbm": -60.0, "snr_db": 40.0, "packet_loss": 0.05,
                        "delay_s": 0.002, "aoi_s": 0.01, "queue_length_packets": 0.0,
                        "link_state": "connected",
                    },
                    "uav_02": {
                        "rssi_dbm": -85.0, "snr_db": 12.0, "packet_loss": 0.35,
                        "delay_s": 0.009, "aoi_s": 0.09, "queue_length_packets": 3.0,
                        "link_state": "connected",
                    },
                },
                "neighbor_messages": {
                    "uav_01": {"covariance": np.eye(6)},
                    "uav_02": {"covariance": np.eye(6) * 3.0},
                },
                # Forbidden diagnostic fields must be ignored even if present.
                "physical_los": {"uav_01": False, "uav_02": True},
                "true_position_error": {"uav_01": 10.0, "uav_02": 0.0},
                "future_message_arrival": {"uav_01": False, "uav_02": True},
            },
            "uav_01": {"action_mask": np.asarray([False, False, False, True]), "candidate_measurement_sources": []},
            "uav_02": {"action_mask": np.asarray([False, False, False, True]), "candidate_measurement_sources": []},
        }
        recorded = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {peer: 0.5 for peer in ids if peer != uav_id},
                "delta_v": [0.1, 0.2, 0.3],
            }
            for uav_id in ids
        }
        actions, metadata = intervene_quality_measurement_selection(
            recorded, observations, ids, "quality_aware_candidate", _thresholds(), {key: 0 for key in ids}
        )
        self.assertEqual(actions["uav_00"]["measurement_uav_id"], "uav_01")
        self.assertEqual(actions["uav_01"]["measurement_uav_id"], "none")
        self.assertEqual(actions["uav_00"]["bandwidth"], recorded["uav_00"]["bandwidth"])
        np.testing.assert_allclose(actions["uav_00"]["delta_v"], recorded["uav_00"]["delta_v"])
        self.assertFalse(metadata["uav_00"]["uses_physical_los"])
        self.assertFalse(metadata["uav_00"]["uses_truth_error"])
        self.assertFalse(metadata["uav_00"]["uses_future_information"])

    def test_no_eligible_candidate_returns_none(self) -> None:
        observation = {
            "action_mask": np.asarray([False, True, False, True]),
            "candidate_measurement_sources": ["uav_01"],
            "candidate_incoming_links": {
                "uav_01": {
                    "rssi_dbm": -100.0, "snr_db": 1.0, "packet_loss": 0.9,
                    "delay_s": 0.1, "aoi_s": 1.0, "queue_length_packets": 10.0,
                    "link_state": "unstable",
                }
            },
        }
        rows = quality_candidate_scores(observation, ("uav_00", "uav_01", "uav_02"), _thresholds())
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["eligible"])

    def test_thresholds_are_fitted_from_train_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "train" / "episode_000001"
            episode.mkdir(parents=True)
            link_fields = (
                "candidate_edge", "rssi", "snr", "packet_loss", "delay", "aoi", "queue_length"
            )
            _write_csv(episode / "links.csv", link_fields, [
                {"candidate_edge": 1, "rssi": -80, "snr": 20, "packet_loss": 0.2, "delay": 0.01, "aoi": 0.1, "queue_length": 1},
                {"candidate_edge": 1, "rssi": -60, "snr": 40, "packet_loss": 0.0, "delay": 0.001, "aoi": 0.01, "queue_length": 0},
            ])
            covariance = np.eye(15)
            _write_csv(
                episode / "estimates.csv", ("covariance_flat_json",),
                [{"covariance_flat_json": json.dumps(covariance.reshape(-1).tolist())}],
            )
            thresholds = fit_training_quality_thresholds(root)
            self.assertEqual(thresholds["source_split"], "train")
            self.assertFalse(thresholds["validation_used_for_fitting"])
            self.assertFalse(thresholds["test_used_for_fitting"])
            self.assertEqual(thresholds["training_episode_count"], 1)
            self.assertEqual(thresholds["candidate_link_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
