"""Tests for the frozen single-seed 41-dimensional safety evaluation."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.safety_motion_feature_evaluation import (
    BASELINE_BRANCH,
    BRANCHES,
    FULL_BRANCH,
    SafetyMotionFeatureEvaluationConfig,
    run_safety_motion_feature_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "safety_motion_feature_smoke_seed2026"
SCENARIO_ROOT = PROJECT_ROOT / "data" / "synthetic" / "safety_risk_screening_seed2026" / "scenarios"
HAS_BUNDLE = all(
    (
        CHECKPOINT_ROOT / branch / "scale_3" / "checkpoints" / "final.pt"
    ).exists()
    for branch in BRANCHES
)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SafetyMotionFeatureEvaluationConfigTests(unittest.TestCase):
    def _kwargs(self) -> dict[str, object]:
        return {
            "checkpoint_root": Path("checkpoint"),
            "scenario_root": Path("scenario"),
        }

    def test_config_is_fixed_to_single_seed_and_two_branches(self) -> None:
        with self.assertRaises(ValueError):
            SafetyMotionFeatureEvaluationConfig(**self._kwargs(), policy_seeds=(7071,))
        with self.assertRaises(ValueError):
            SafetyMotionFeatureEvaluationConfig(
                **self._kwargs(),
                branches=(BASELINE_BRANCH,),
            )
        with self.assertRaises(ValueError):
            SafetyMotionFeatureEvaluationConfig(
                **self._kwargs(),
                measurement_overhead_enabled=False,
            )


@unittest.skipUnless(TORCH_AVAILABLE and HAS_BUNDLE, "frozen 41-dimensional Smoke bundle is unavailable")
class SafetyMotionFeatureEvaluationRunTests(unittest.TestCase):
    def test_single_scenario_pair_is_read_only_and_finite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_motion_eval_") as temporary:
            root = Path(temporary)
            result = run_safety_motion_feature_evaluation(
                SafetyMotionFeatureEvaluationConfig(
                    checkpoint_root=CHECKPOINT_ROOT,
                    scenario_root=SCENARIO_ROOT,
                    output_dir=root / "data",
                    logs_dir=root / "logs",
                    plots_dir=root / "plots",
                    scenario_seeds=(2026,),
                    max_steps=1,
                    bootstrap_samples=100,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"], result)
            report = json.loads((root / "logs" / "audit_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["audit_ok"])
            self.assertFalse(report["decision_supported"])
            self.assertFalse(report["design"]["test_data_read"])
            self.assertFalse(report["design"]["trainer_update_called"])
            self.assertEqual(report["design"]["independent_unit"], "scenario_seed")
            self.assertEqual(report["source"]["checkpoint_layout"], "direct Smoke root / branch / scale_3")
            self.assertTrue((root / "plots" / "safety_motion_feature_comparison.png").exists())
            with (root / "logs" / "episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["branch"] for row in rows}, set(BRANCHES))
            self.assertTrue(all(int(row["actor_node_feature_dim_41"]) == 1 for row in rows))
            self.assertTrue(all(int(row["dynamic_motion_feature_finite"]) == 1 for row in rows))
            self.assertTrue(all(int(row["trainer_update_called"]) == 0 for row in rows))
            self.assertTrue(all(int(row["task_complete"]) == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
