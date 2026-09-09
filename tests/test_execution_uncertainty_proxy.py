"""Tests for the train-only execution-time uncertainty proxy audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.execution_uncertainty_proxy_audit import (
    ExecutionUncertaintyProxyAuditConfig,
    ProxyScales,
    _materialize_proxy_rows,
    _raw_information_gain,
    run_execution_uncertainty_proxy_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _graph_row(*, split: str, timestamp: float, rssi_1: float, rssi_2: float) -> dict[str, object]:
    node_features = [
        [0.0, 0.0, 10.0] + [0.0] * 12 + [4.0, 4.0, 4.0] + [0.0] * 21,
        [0.0] * 39,
        [0.0] * 39,
    ]
    links = {
        "uav_01": {
            "rssi_dbm": rssi_1, "snr_db": 35.0, "packet_loss": 0.02,
            "delay_s": 0.001, "aoi_s": 0.01, "queue_length_packets": 0.0,
            "candidate_edge": 1, "effective_edge": 0, "link_state": "connected",
        },
        "uav_02": {
            "rssi_dbm": rssi_2, "snr_db": 8.0, "packet_loss": 0.25,
            "delay_s": 0.01, "aoi_s": 0.1, "queue_length_packets": 2.0,
            "candidate_edge": 1, "effective_edge": 0, "link_state": "connected",
        },
    }
    candidate_features = {
        "uav_01": [0.0, 0.0, 0.0, 10.0, -1.0, rssi_1, 35.0, 0.02, 0.001, 0.01, 0.0],
        "uav_02": [0.0, 0.0, 0.0, 12.0, -1.0, rssi_2, 8.0, 0.25, 0.01, 0.1, 2.0],
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
        "bandwidth_mask_json": json.dumps([False, True, True]),
        "candidate_incoming_edge_features_json": json.dumps(candidate_features),
        "candidate_incoming_links_json": json.dumps(links),
        "uwb_available_json": json.dumps({"uav_01": True, "uav_02": True}),
        "uwb_ranges_json": json.dumps({"uav_01": 10.0, "uav_02": 12.0}),
        # The audit must not read this field because it is centralized state.
        "central_state_json": json.dumps({"true_position": [999.0, 999.0, 999.0]}),
    }


def _write_episode(scale: Path, split: str, *, validation_rssi: bool = False) -> None:
    episode = scale / split / ("episode_000001" if split == "train" else "episode_400001")
    timestamp = 0.2
    rssi_1, rssi_2 = (-60.0, -80.0) if not validation_rssi else (-5.0, -6.0)
    _write_csv(episode / "graph_features.csv", [_graph_row(split=split, timestamp=timestamp, rssi_1=rssi_1, rssi_2=rssi_2)])
    _write_csv(
        episode / "uwb_ranges.csv",
        [
            {"timestamp": timestamp, "src_uav_id": "uav_01", "dst_uav_id": "uav_00", "measurement_std": 0.1, "valid": 1},
            {"timestamp": timestamp, "src_uav_id": "uav_02", "dst_uav_id": "uav_00", "measurement_std": 0.5, "valid": 1},
        ],
    )
    if split == "validation":
        _write_csv(
            episode / "ekf_diagnostics.csv",
            [{"timestamp": timestamp, "uav_id": "uav_00", "position_error_m": 0.4}],
        )


class ExecutionUncertaintyProxyTests(unittest.TestCase):
    def test_forbidden_truth_fields_do_not_change_proxy(self) -> None:
        scales = ProxyScales(
            measurement_std_median=0.2,
            neighbor_covariance_trace_q95_m2=9.0,
            delay_q75_s=0.01,
            age_q75_s=0.1,
            queue_q75_packets=1.0,
            rssi_q05_dbm=-100.0,
            rssi_q95_dbm=-40.0,
            snr_q05_db=0.0,
            snr_q95_db=40.0,
            ig_q05=0.0,
            ig_q95=1.0,
        )
        row = {
            "own_cov_diag": np.asarray([2.0, 2.0, 2.0]),
            "neighbor_cov_diag": None,
            "u_hat": np.asarray([1.0, 0.0, 0.0]),
            "measurement_std_m": 0.2,
            "message_age_s": 0.01,
            "rssi_dbm": -60.0,
            "snr_db": 30.0,
            "packet_loss_pred": 0.1,
            "delay_pred_s": 0.001,
            "aoi_pred_s": 0.01,
            "queue_length_pred": 0.0,
            "link_state": "connected",
            "true_range": 10.0,
            "true_position_error": 99.0,
            "physical_los": 0,
        }
        first = _raw_information_gain(row, scales, 1.0)
        row["true_range"] = 99999.0
        row["true_position_error"] = -123.0
        row["physical_los"] = 1
        second = _raw_information_gain(row, scales, 1.0)
        self.assertTrue(np.isfinite(first))
        self.assertEqual(first, second)

    def test_missing_neighbor_covariance_has_finite_nonnegative_proxy(self) -> None:
        scales = ProxyScales(
            measurement_std_median=0.2,
            neighbor_covariance_trace_q95_m2=9.0,
            delay_q75_s=0.01,
            age_q75_s=0.1,
            queue_q75_packets=1.0,
            rssi_q05_dbm=-100.0,
            rssi_q95_dbm=-40.0,
            snr_q05_db=0.0,
            snr_q95_db=40.0,
            ig_q05=0.0,
            ig_q95=1.0,
        )
        row = {
            "own_cov_diag": np.asarray([1.0, 1.0, 1.0]),
            "neighbor_cov_diag": None,
            "u_hat": np.asarray([1.0, 0.0, 0.0]),
            "measurement_std_m": 0.1,
            "message_age_s": 0.0,
            "rssi_dbm": -60.0,
            "snr_db": 30.0,
            "packet_loss_pred": 0.0,
            "delay_pred_s": 0.001,
            "aoi_pred_s": 0.01,
            "queue_length_pred": 0.0,
            "link_state": "connected",
        }
        value = _raw_information_gain(row, scales, 1.0)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_ranking_is_deterministic_with_id_tie_break(self) -> None:
        scales = ProxyScales(
            measurement_std_median=0.2,
            neighbor_covariance_trace_q95_m2=9.0,
            delay_q75_s=0.01,
            age_q75_s=0.1,
            queue_q75_packets=1.0,
            rssi_q05_dbm=-100.0,
            rssi_q95_dbm=-40.0,
            snr_q05_db=0.0,
            snr_q95_db=40.0,
            ig_q05=0.0,
            ig_q95=1.0,
        )
        common = {
            "base_seed": 2026, "episode_seed": 1, "treatment": "baseline_mappo", "split": "validation",
            "episode_id": "episode_1", "scenario_id": "scenario_1", "timestamp": 0.2,
            "uav_id": "uav_00", "candidate_count": 2, "multi_candidate_window": 1,
            "own_position": np.zeros(3), "own_cov_diag": np.ones(3), "u_hat": np.asarray([1.0, 0.0, 0.0]),
            "geometry_source": "isotropic_no_neighbor_estimate", "neighbor_covariance_available": 0,
            "neighbor_cov_diag": None, "estimated_distance_m": 10.0, "measurement_std_m": 0.2,
            "uwb_available": 1.0, "message_age_s": 0.0, "rssi_dbm": -60.0, "snr_db": 20.0,
            "packet_loss_pred": 0.0, "delay_pred_s": 0.001, "aoi_pred_s": 0.01,
            "queue_length_pred": 0.0, "link_state": "connected", "candidate_edge": 1.0,
            "offline_position_error_m": 0.4, "offline_label_only": 1, "age_variance_coefficient": 1.0,
        }
        rows = [{**common, "source_uav_id": "uav_02"}, {**common, "source_uav_id": "uav_01"}]
        output = _materialize_proxy_rows(rows, scales, 1.0)
        ranked = [row["source_uav_id"] for row in sorted(output, key=lambda row: int(row["proxy_rank"]))]
        self.assertEqual(ranked, ["uav_01", "uav_02"])

    def test_audit_fits_train_only_and_keeps_test_sealed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="execution_proxy_") as temporary:
            root = Path(temporary) / "reference"
            for treatment in ("baseline_mappo", "quality_aware_mappo"):
                scale = root / "seed_2026" / treatment / "scale_3"
                _write_episode(scale, "train")
                _write_episode(scale, "validation", validation_rssi=True)
            logs = Path(temporary) / "logs"
            plots = Path(temporary) / "plots"
            report = run_execution_uncertainty_proxy_audit(
                ExecutionUncertaintyProxyAuditConfig(
                    reference_root=root, logs_output=logs, plots_output=plots,
                )
            )
            self.assertFalse(report["test_data_read"])
            self.assertTrue(report["checks"]["train_fit_separated"])
            self.assertEqual(report["candidate_row_count"], 4)
            params = json.loads((logs / "proxy_parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(params["scales_by_seed"]["2026"]["fit_split"], "train")
            self.assertLess(params["scales_by_seed"]["2026"]["scales"]["rssi_q95_dbm"], -10.0)
            for name in ("proxy_parameters.json", "candidate_proxy.csv", "episode_proxy_summary.csv", "seed_proxy_summary.csv", "proxy_strata.csv", "audit_report.json"):
                self.assertTrue((logs / name).is_file())
            self.assertTrue((plots / "execution_uncertainty_proxy.png").is_file())


if __name__ == "__main__":
    unittest.main()
