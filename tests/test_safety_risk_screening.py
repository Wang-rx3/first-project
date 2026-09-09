"""Tests for the independent safety-risk scenario screening."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.safety_risk_screening import (
    SafetyRiskScreeningConfig,
    run_safety_risk_screening,
)
from src.simulation.obstacle_model import collision_event
from src.simulation.scenario import generate_safety_risk_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PROFILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "scenario_screening_seed2026"
    / "selected_profiles"
    / "moderate.json"
)
EKF_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "python_sim_calibration_seed2026_v4"
    / "selected_ekf_config.json"
)


class SafetyRiskScreeningTests(unittest.TestCase):
    def test_scenario_has_safe_initial_state_and_overlapping_dynamic_obstacle(self) -> None:
        scenario = generate_safety_risk_scenario(4044, uav_count=3)
        positions = {
            uav_id: np.asarray(position, dtype=float)
            for uav_id, position in scenario["initial_positions"].items()
        }
        # The screening gate must not be satisfied by an initially invalid scene.
        self.assertEqual(collision_event(positions, [], 2.0)[0], 0)
        design = scenario["safety_risk_design"]
        self.assertTrue(design["initial_state_is_safe_by_construction"])
        self.assertTrue(design["dynamic_obstacle_altitude_overlap"])
        dynamic = next(
            obstacle
            for obstacle in scenario["obstacle_specs"]
            if obstacle["obstacle_type"] == "dynamic_vehicle"
        )
        self.assertEqual(dynamic["center_m"][2], design["flight_altitude_m"])
        self.assertGreater(dynamic["radius_m"], 0.0)

    def test_config_rejects_non_three_uav_or_duplicate_scenario_seeds(self) -> None:
        with self.assertRaises(ValueError):
            SafetyRiskScreeningConfig(output_dir=Path("unused"), uav_count=5)
        with self.assertRaises(ValueError):
            SafetyRiskScreeningConfig(output_dir=Path("unused"), scenario_seeds=(1, 1))

    def test_screening_writes_projection_diagnostic_and_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_risk_screening_") as temporary:
            root = Path(temporary)
            result = run_safety_risk_screening(
                SafetyRiskScreeningConfig(
                    output_dir=root / "data",
                    logs_dir=root / "logs",
                    plots_dir=root / "plots",
                    scenario_seeds=(2026, 3035, 4044, 5053, 6062),
                    max_steps=50,
                    scenario_profile_path=SCENARIO_PROFILE,
                    ekf_config_path=EKF_CONFIG,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"])
            self.assertTrue(result["screening_passed"])

            report_path = root / "logs" / "audit_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["audit_ok"])
            self.assertTrue(report["predeclared_gate"]["initial_state_safe"])
            self.assertTrue(report["predeclared_gate"]["preferred_rate_range_5_to_30_percent"])
            self.assertEqual(report["design"]["independent_unit"], "scenario_seed")
            self.assertTrue(report["design"]["time_windows_are_diagnostic_only"])

            with (root / "logs" / "scenario_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)  # five seeds x protected/diagnostic mode
            self.assertEqual({row["mode"] for row in rows}, {"projection_on", "projection_off"})
            self.assertTrue(all(int(row["initial_collision"]) == 0 for row in rows))
            self.assertTrue(all(int(row["finite_state"]) == 1 for row in rows))

            with (root / "logs" / "step_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                step_rows = list(csv.DictReader(handle))
            self.assertGreater(len(step_rows), 0)
            for row in step_rows:
                for field in (
                    "minimum_clearance_m",
                    "pre_projection_minimum_clearance_m",
                    "action_magnitude_mean",
                    "energy_proxy_step",
                    "path_length_step_m",
                ):
                    self.assertTrue(float(row[field]) >= 0.0)
                self.assertIn(row["termination_reason"], {"running", "collision", "time_limit"})

            self.assertTrue((root / "data" / "manifest.json").exists())
            self.assertTrue((root / "plots" / "safety_risk_screening.png").exists())

    def test_swept_projection_flag_is_forwarded_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_risk_screening_swept_") as temporary:
            root = Path(temporary)
            result = run_safety_risk_screening(
                SafetyRiskScreeningConfig(
                    output_dir=root / "data",
                    logs_dir=root / "logs",
                    plots_dir=root / "plots",
                    scenario_seeds=(2026,),
                    max_steps=10,
                    scenario_profile_path=SCENARIO_PROFILE,
                    ekf_config_path=EKF_CONFIG,
                    dynamic_obstacle_swept_projection_enabled=True,
                    include_unprotected_diagnostic=False,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"])
            report = json.loads((root / "logs" / "audit_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["design"]["dynamic_obstacle_swept_projection_enabled"])
            self.assertFalse(report["design"]["uses_neighbor_information"])
            self.assertEqual(report["design"]["trainer_updates"], 0)
            self.assertFalse(report["design"]["test_data_used_for_fitting_or_selection"])
            self.assertEqual(report["counts"]["selected_measurement_action_count"], 0)
            self.assertEqual(report["counts"]["cooperative_ekf_update_count"], 0)
            self.assertTrue(report["fixed_configuration"]["dynamic_obstacle_swept_projection_enabled"])
            with (root / "logs" / "scenario_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dynamic_obstacle_swept_projection_enabled"], "1")


if __name__ == "__main__":
    unittest.main()
