"""Tests for train-fitted continuous measurement-quality scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.soft_quality_measurement_intervention import (
    SoftQualityMeasurementInterventionConfig,
    fit_training_soft_quality_model,
    intervene_soft_quality_measurement_selection,
    soft_quality_candidate_scores,
)
from src.synthetic.episode_generator import _write_csv


def _model(none_threshold: float = 0.9) -> dict:
    fields = (
        "rssi_dbm", "snr_db", "packet_loss", "delay_s", "aoi_s",
        "queue_length_packets", "neighbor_covariance_trace_m2",
    )
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
        "soft_score": {
            "weights": {field: 1.0 / len(fields) for field in fields},
            "missing_neighbor_covariance_risk": 1.0,
            "none_threshold": none_threshold,
        },
    }


class SoftQualityMeasurementInterventionTests(unittest.TestCase):
    def test_test_split_is_prohibited(self) -> None:
        with self.assertRaises(ValueError):
            SoftQualityMeasurementInterventionConfig(
                reference_scale=Path("reference"), output_dir=Path("output"), split="test"
            )

    def test_soft_score_ranks_visible_quality_and_ignores_forbidden_fields(self) -> None:
        observation = {
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
            "physical_los": {"uav_01": False, "uav_02": True},
            "true_position_error": {"uav_01": 10.0, "uav_02": 0.0},
            "future_message_arrival": {"uav_01": False, "uav_02": True},
        }
        rows = soft_quality_candidate_scores(observation, ("uav_00", "uav_01", "uav_02"), _model())
        self.assertEqual(rows[0]["source_uav_id"], "uav_01")
        self.assertLess(rows[0]["quality_risk_score"], rows[1]["quality_risk_score"])
        self.assertTrue(all(row["actor_visible_only"] for row in rows))

    def test_soft_selector_preserves_nonmeasurement_actions_and_uses_none_threshold(self) -> None:
        ids = ("uav_00", "uav_01", "uav_02")
        observation = {
            "action_mask": np.asarray([False, True, False, True]),
            "candidate_measurement_sources": ["uav_01"],
            "candidate_incoming_links": {
                "uav_01": {
                    "rssi_dbm": -80.0, "snr_db": 20.0, "packet_loss": 0.2,
                    "delay_s": 0.005, "aoi_s": 0.05, "queue_length_packets": 1.0,
                    "link_state": "connected",
                }
            },
            "neighbor_messages": {"uav_01": {"covariance": np.eye(6) * 2.0}},
        }
        observations = {
            "uav_00": observation,
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
        actions, metadata = intervene_soft_quality_measurement_selection(
            recorded, observations, ids, "soft_quality_candidate", _model(0.9), {key: 0 for key in ids}
        )
        self.assertEqual(actions["uav_00"]["measurement_uav_id"], "uav_01")
        self.assertEqual(actions["uav_00"]["bandwidth"], recorded["uav_00"]["bandwidth"])
        np.testing.assert_allclose(actions["uav_00"]["delta_v"], recorded["uav_00"]["delta_v"])
        self.assertFalse(metadata["uav_00"]["uses_physical_los"])
        strict_actions, _metadata = intervene_soft_quality_measurement_selection(
            recorded, observations, ids, "soft_quality_candidate", _model(0.01), {key: 0 for key in ids}
        )
        self.assertEqual(strict_actions["uav_00"]["measurement_uav_id"], "none")

    def test_soft_model_is_fitted_from_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "train" / "episode_000001"
            episode.mkdir(parents=True)
            _write_csv(
                episode / "links.csv",
                (
                    "timestamp", "src_uav_id", "dst_uav_id", "candidate_edge", "link_state",
                    "rssi", "snr", "packet_loss", "delay", "aoi", "queue_length",
                ),
                [
                    {"timestamp": 0.1, "src_uav_id": "uav_01", "dst_uav_id": "uav_00", "candidate_edge": 1, "link_state": "connected", "rssi": -80, "snr": 20, "packet_loss": 0.2, "delay": 0.01, "aoi": 0.1, "queue_length": 1},
                    {"timestamp": 0.2, "src_uav_id": "uav_01", "dst_uav_id": "uav_00", "candidate_edge": 1, "link_state": "connected", "rssi": -60, "snr": 40, "packet_loss": 0.0, "delay": 0.001, "aoi": 0.01, "queue_length": 0},
                ],
            )
            covariance = np.eye(15)
            _write_csv(
                episode / "estimates.csv",
                ("timestamp", "uav_id", "covariance_flat_json"),
                [
                    {"timestamp": 0.1, "uav_id": "uav_01", "covariance_flat_json": json.dumps(covariance.reshape(-1).tolist())},
                    {"timestamp": 0.2, "uav_id": "uav_01", "covariance_flat_json": json.dumps(covariance.reshape(-1).tolist())},
                ],
            )
            model = fit_training_soft_quality_model(root)
            self.assertEqual(model["source_split"], "train")
            self.assertFalse(model["validation_used_for_fitting"])
            self.assertFalse(model["test_used_for_fitting"])
            self.assertEqual(model["soft_score"]["connected_candidate_row_count"], 2)
            self.assertTrue(np.isfinite(model["soft_score"]["none_threshold"]))


if __name__ == "__main__":
    unittest.main()
