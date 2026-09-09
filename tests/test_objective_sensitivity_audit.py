"""Tests for the validation-only objective sensitivity audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.objective_sensitivity_audit import (
    ObjectiveSensitivityAuditConfig,
    run_objective_sensitivity_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_branch(root: Path, seed: int, treatment: str, error_m: float) -> None:
    scale = root / f"seed_{seed}" / treatment / "scale_3"
    diagnostics = scale / "diagnostics"
    episode = scale / "validation" / "episode_400001"
    diagnostics.mkdir(parents=True, exist_ok=True)
    episode.mkdir(parents=True, exist_ok=True)
    scenario = {
        "episode_id": "episode_400001",
        "scenario_id": "scenario_400001",
        "seed": seed + 400000,
        "split": "validation",
        "reward_config": {
            "weights": {
                "uncertainty": 0.35,
                "loss": 0.20,
                "aoi": 0.15,
                "delay": 0.15,
                "connectivity": 0.15,
            }
        },
    }
    (episode / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
    _write_csv(
        episode / "ground_truth.csv",
        [
            {
                "episode_id": "episode_400001",
                "scenario_id": "scenario_400001",
                "seed": seed + 400000,
                "timestamp": 0.2,
                "split": "validation",
                "uav_id": "uav_00",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
        ],
    )
    _write_csv(
        episode / "estimates.csv",
        [
            {
                "episode_id": "episode_400001",
                "scenario_id": "scenario_400001",
                "seed": seed + 400000,
                "timestamp": 0.2,
                "split": "validation",
                "uav_id": "uav_00",
                "x": error_m,
                "y": 0.0,
                "z": 0.0,
                "covariance_flat_json": json.dumps([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            }
        ],
    )
    # j_loc = 0.35 * 0.5 = 0.175; j_com = 0.1 + 0.03 + 0.03 - 0.15 = 0.01.
    reward_row = {
        "timestamp": 0.2,
        "uav_id": "uav_00",
        "reward_total": -0.186,
        "reward_localization": -1.0,
        "reward_packet_loss": -0.1,
        "reward_aoi": -0.03,
        "reward_delay": -0.03,
        "reward_connectivity": 1.0,
        "reward_collision": 0.0,
        "reward_energy": -0.001,
        "joint_objective": 0.185,
        "normalized_uncertainty": 0.5,
        "normalized_packet_loss": 0.5,
        "normalized_aoi": 0.2,
        "normalized_delay": 0.2,
        "normalized_connectivity": 1.0,
        "collision": 0,
        "task_complete": 0,
    }
    _write_csv(episode / "rewards.csv", [reward_row])
    metrics_row = {
        "episode_id": "episode_400001",
        "scenario_id": "scenario_400001",
        "split": "validation",
        "phase": "validation",
        "seed": seed + 400000,
        "uav_count": 3,
        "mean_reward": -0.186,
        "episode_return": -0.186,
        "j_loc": 0.175,
        "j_com": 0.01,
        "joint_objective": 0.185,
        "position_rmse_m": error_m,
        "covariance_position_trace": 3.0,
        "packet_loss": 0.5,
        "delay_s": 0.001,
        "aoi_s": 0.002,
        "connectivity": 1.0,
        "energy_proxy": 0.001,
        "collision_rate": 0.0,
        "task_complete": 0.0,
        "termination_reason": "time_limit",
    }
    _write_csv(diagnostics / "episode_metrics.csv", [metrics_row])
    (diagnostics / "cooperative_updates.jsonl").write_text("", encoding="utf-8")
    (diagnostics / "measurement_quality.jsonl").write_text("", encoding="utf-8")


class ObjectiveSensitivityAuditTests(unittest.TestCase):
    def test_validation_audit_closes_objective_and_keeps_test_sealed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="objective_sensitivity_") as temporary:
            root = Path(temporary) / "reference"
            for seed, error in ((2026, 0.4), (3035, 0.5)):
                _write_branch(root, seed, "baseline_mappo", error)
                _write_branch(root, seed, "quality_aware_mappo", error / 2.0)
            logs = Path(temporary) / "logs"
            plots = Path(temporary) / "plots"
            report = run_objective_sensitivity_audit(
                ObjectiveSensitivityAuditConfig(
                    reference_root=root,
                    logs_output=logs,
                    plots_output=plots,
                    bootstrap_samples=200,
                    bootstrap_seed=2026,
                )
            )
            self.assertTrue(report["audit_ok"])
            self.assertFalse(report["test_data_used"])
            self.assertTrue(report["checks"]["objective_and_reward_closure"])
            self.assertEqual(report["seed_count"], 2)
            self.assertTrue(
                (logs / "episode_sensitivity.csv").is_file()
                and (logs / "seed_sensitivity.csv").is_file()
                and (logs / "metric_correlations.csv").is_file()
                and (logs / "audit_report.json").is_file()
            )
            self.assertTrue((plots / "objective_sensitivity.png").is_file())

    def test_test_split_is_rejected_before_any_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="objective_sensitivity_test_") as temporary:
            with self.assertRaisesRegex(ValueError, "validation-only"):
                run_objective_sensitivity_audit(
                    ObjectiveSensitivityAuditConfig(
                        reference_root=Path(temporary),
                        split="test",
                        bootstrap_samples=200,
                    )
                )


if __name__ == "__main__":
    unittest.main()
