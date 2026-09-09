"""Tests for environment-only communication-localization screening."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.scenario_screening import (
    FACTOR_RANGES,
    ScenarioScreeningConfig,
    classify_parameter_summary,
    latin_hypercube_design,
    run_scenario_screening,
)


class ScenarioScreeningTests(unittest.TestCase):
    def test_latin_hypercube_is_reproducible_and_in_bounds(self) -> None:
        first = latin_hypercube_design(8, seed=2026)
        second = latin_hypercube_design(8, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(len({row["design_point_id"] for row in first}), 8)
        self.assertEqual(sorted(int(row["run_order"]) for row in first), list(range(1, 9)))
        for row in first:
            for name, (lower, upper) in FACTOR_RANGES.items():
                self.assertGreaterEqual(float(row[name]), lower)
                self.assertLessEqual(float(row[name]), upper)

    def test_predeclared_eligibility_gates(self) -> None:
        row = {
            "full_connectivity_fraction": 0.70,
            "partial_connectivity_fraction": 0.25,
            "disconnected_fraction": 0.05,
            "candidate_effective_ratio": 0.80,
            "mean_packet_loss": 0.10,
            "message_arrival_rate": 0.65,
            "uwb_update_acceptance_rate": 0.50,
            "neighbor_update_window_rate": 0.50,
            "collision_episode_fraction": 0.0,
            "finite_state_fraction": 1.0,
        }
        classification, failed = classify_parameter_summary(row)
        self.assertEqual(classification, "eligible")
        self.assertEqual(failed, [])

    def test_small_screening_run_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scenario_screening_") as temporary:
            root = Path(temporary)
            result = run_scenario_screening(
                ScenarioScreeningConfig(
                    output_dir=root / "data",
                    logs_dir=root / "logs",
                    plots_dir=root / "plots",
                    design_seed=17,
                    scenario_seeds=(17, 1026),
                    design_points=2,
                    max_steps=2,
                    force=True,
                    progress=False,
                )
            )
            self.assertEqual(result["independent_replication_unit"], "episode/scenario_seed")
            self.assertFalse(result["mappo_performance_used_for_selection"])
            self.assertTrue((root / "data" / "manifest.json").exists())
            self.assertTrue((root / "logs" / "design_matrix.csv").exists())
            self.assertTrue((root / "logs" / "run_schedule.csv").exists())
            self.assertTrue((root / "plots" / "feasible_region.png").exists())
            with (root / "logs" / "episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            payload = json.loads((root / "data" / "selected_profiles" / "moderate.json").read_text(encoding="utf-8"))
            self.assertIn("allowed_for_training", payload)


if __name__ == "__main__":
    unittest.main()
