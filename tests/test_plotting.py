"""Acceptance tests for the eight paper-oriented main figures."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.experiments.formal import FormalExperimentConfig, run_preliminary
from src.experiments.plotting import plot_main_figures


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(TORCH_AVAILABLE and MATPLOTLIB_AVAILABLE, "PyTorch and Matplotlib are required")
class MainPlotTests(unittest.TestCase):
    def test_main_figures_and_derived_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="main_figures_") as temporary:
            root = Path(temporary) / "experiment"
            run_preliminary(
                FormalExperimentConfig(
                    output_dir=root,
                    uav_count=3,
                    seed=2026,
                    max_steps=2,
                    warmup_episodes=1,
                    train_episodes=1,
                    validation_episodes=1,
                    test_episodes=1,
                    device="cpu",
                    progress=False,
                )
            )
            plot_root = root / "plots"
            log_root = root / "logs"
            summary = plot_main_figures(root, plot_root, logs_output=log_root)

            expected_figures = [
                "fig01_training_curves.png",
                "fig02_validation_test_comparison.png",
                "fig03_objective_reward_decomposition.png",
                "fig04_localization_time_series.png",
                "fig05_localization_cdf.png",
                "fig06_communication_metrics.png",
                "fig07_safety_energy.png",
                "fig08_trajectory_3d.png",
            ]
            for filename in expected_figures:
                target = plot_root / filename
                self.assertTrue(target.exists(), filename)
                self.assertGreater(target.stat().st_size, 0, filename)
            diagnostic_figure = plot_root.parent / "diagnostics" / "ekf_consistency.png"
            self.assertTrue(diagnostic_figure.exists())
            self.assertGreater(diagnostic_figure.stat().st_size, 0)
            self.assertTrue((log_root.parent / "diagnostics" / "ekf_consistency_metrics.csv").exists())

            self.assertEqual(summary["splits"], ["test", "validation"])
            with (log_root / "validation_test_summary.csv").open(encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in summary_rows}, {"validation", "test"})

            with (log_root / "objective_metrics.csv").open(encoding="utf-8") as handle:
                objective_rows = list(csv.DictReader(handle))
            self.assertTrue(objective_rows)
            for row in objective_rows:
                j_loc = float(row["j_loc"])
                j_com = float(row["j_com"])
                objective = float(row["joint_objective"])
                self.assertAlmostEqual(j_loc + j_com, objective, places=8)
                reward = float(row["mean_reward"])
                reconstructed = -(objective + float(row["collision_penalty"]) + float(row["energy_penalty"]))
                self.assertAlmostEqual(reward, reconstructed, places=7)

            with (log_root / "localization_metrics.csv").open(encoding="utf-8") as handle:
                localization_rows = list(csv.DictReader(handle))
            self.assertTrue(localization_rows)
            for row in localization_rows:
                for key in ("position_rmse_m", "velocity_rmse_mps", "covariance_position_trace"):
                    self.assertGreaterEqual(float(row[key]), 0.0)


if __name__ == "__main__":
    unittest.main()
