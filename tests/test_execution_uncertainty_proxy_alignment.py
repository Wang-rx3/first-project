"""Tests for the read-only execution-time proxy alignment audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.execution_uncertainty_proxy_alignment_audit import (
    ExecutionUncertaintyProxyAlignmentAuditConfig,
    run_execution_uncertainty_proxy_alignment_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _graph_row(split: str, timestamp: float) -> dict[str, object]:
    node_features = [
        [0.0, 0.0, 10.0] + [0.0] * 12 + [4.0, 4.0, 4.0] + [0.0] * 21,
        [0.0] * 39,
        [0.0] * 39,
    ]
    links = {
        "uav_01": {
            "rssi_dbm": -60.0, "snr_db": 30.0, "packet_loss": 0.02,
            "delay_s": 0.001, "aoi_s": 0.01, "queue_length_packets": 0.0,
            "candidate_edge": 1, "effective_edge": 0, "link_state": "connected",
        },
        "uav_02": {
            "rssi_dbm": -80.0, "snr_db": 8.0, "packet_loss": 0.25,
            "delay_s": 0.01, "aoi_s": 0.1, "queue_length_packets": 2.0,
            "candidate_edge": 1, "effective_edge": 0, "link_state": "connected",
        },
    }
    candidate_features = {
        "uav_01": [0.0, 0.0, 0.0, 10.0, -1.0, -60.0, 30.0, 0.02, 0.001, 0.01, 0.0],
        "uav_02": [0.0, 0.0, 0.0, 12.0, -1.0, -80.0, 8.0, 0.25, 0.01, 0.1, 2.0],
    }
    return {
        "episode_id": "episode_000001" if split == "train" else "episode_400001",
        "scenario_id": "scenario_000001",
        "seed": 2026,
        "timestamp": timestamp,
        "split": split,
        "uav_id": "uav_00",
        "node_features_json": json.dumps(node_features),
        "effective_edge_index_json": json.dumps([[], []]),
        "effective_edge_features_json": json.dumps([]),
        "candidate_edge_index_json": json.dumps([[0, 0], [1, 2]]),
        "candidate_edge_features_json": json.dumps(candidate_features),
        "action_mask_json": json.dumps({"measurement": [False, True, True, True]}),
        "potential_edge_index_json": json.dumps([[0, 0], [1, 2]]),
        "potential_edge_features_json": json.dumps(candidate_features),
        "candidate_incoming_edge_features_json": json.dumps(candidate_features),
        "candidate_incoming_links_json": json.dumps(links),
        "uwb_available_json": json.dumps({"uav_01": True, "uav_02": True}),
        "uwb_ranges_json": json.dumps({"uav_01": 10.0, "uav_02": 12.0}),
        "central_state_json": json.dumps({"true_position": [999.0, 999.0, 999.0]}),
    }


def _write_episode(scale: Path, split: str) -> None:
    episode_name = "episode_000001" if split == "train" else "episode_400001"
    episode = scale / split / episode_name
    _write_csv(episode / "graph_features.csv", [_graph_row(split, 0.2)])
    _write_csv(
        episode / "uwb_ranges.csv",
        [
            {"timestamp": 0.2, "src_uav_id": "uav_01", "dst_uav_id": "uav_00", "measurement_std": 0.1, "valid": 1},
            {"timestamp": 0.2, "src_uav_id": "uav_02", "dst_uav_id": "uav_00", "measurement_std": 0.5, "valid": 1},
        ],
    )
    if split == "validation":
        _write_csv(
            episode / "ekf_diagnostics.csv",
            [{"timestamp": 0.2, "uav_id": "uav_00", "position_error_m": 0.4}],
        )


def _write_update_log(scale: Path) -> None:
    path = scale / "diagnostics" / "cooperative_updates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "accepted": True,
        "episode_id": "episode_400001",
        "phase": "validation",
        "measurement_time_s": 0.2,
        "source_uav_id": "uav_01",
        "target_uav_id": "uav_00",
        "position_covariance_trace_before_m2": 12.0,
        "position_covariance_trace_after_linear_m2": 10.0,
        "position_covariance_trace_after_final_m2": 11.5,
        "state_correction_norm_m": 0.2,
        "position_correction_norm_m": 0.2,
        "true_position_error_before_m": 0.4,
        "true_position_error_after_m": 0.35,
        "nis": 1.0,
        "kalman_gain_norm": 0.2,
        "message_age_s": 0.1,
        "unknown_cross_bound_m2": 2.0,
        "los": True,
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")


class ExecutionUncertaintyProxyAlignmentTests(unittest.TestCase):
    def test_alignment_audit_is_validation_only_and_closes_linear_reduction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="proxy_alignment_") as temporary:
            root = Path(temporary) / "reference"
            for treatment in ("baseline_mappo", "quality_aware_mappo"):
                scale = root / "seed_2026" / treatment / "scale_3"
                _write_episode(scale, "train")
                _write_episode(scale, "validation")
                _write_update_log(scale)
                # A test directory is intentionally present; the audit must not enumerate it.
                (scale / "test" / "episode_500001").mkdir(parents=True, exist_ok=True)
                (scale / "test" / "episode_500001" / "true_position_should_not_be_read.txt").write_text(
                    "sealed", encoding="utf-8"
                )

            logs = Path(temporary) / "logs"
            plots = Path(temporary) / "plots"
            report = run_execution_uncertainty_proxy_alignment_audit(
                ExecutionUncertaintyProxyAlignmentAuditConfig(
                    reference_root=root,
                    logs_output=logs,
                    plots_output=plots,
                    bootstrap_samples=100,
                )
            )

            self.assertFalse(report["test_data_read"])
            self.assertTrue(report["checks"]["train_fit_separated"])
            self.assertTrue(report["checks"]["linear_covariance_reduction_formula_closed"])
            self.assertEqual(report["accepted_update_event_count"], 2)
            self.assertTrue((logs / "proxy_alignment_events.csv").is_file())
            self.assertTrue((logs / "proxy_alignment_episode_summary.csv").is_file())
            self.assertTrue((logs / "proxy_alignment_seed_summary.csv").is_file())
            self.assertTrue((logs / "proxy_alignment_strata.csv").is_file())
            self.assertTrue((logs / "proxy_alignment_correlations.csv").is_file())
            self.assertTrue((logs / "audit_report.json").is_file())
            self.assertTrue((plots / "proxy_alignment.png").is_file())


if __name__ == "__main__":
    unittest.main()
