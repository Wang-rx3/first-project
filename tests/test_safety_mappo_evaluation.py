"""Tests for deterministic frozen-checkpoint safety evaluation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.safety_mappo_evaluation import (
    BASELINE_BRANCH,
    FULL_BRANCH,
    SafetyMAPPOEvaluationConfig,
    run_safety_mappo_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "data" / "synthetic" / "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
SCENARIO_ROOT = PROJECT_ROOT / "data" / "synthetic" / "safety_risk_screening_seed2026" / "scenarios"
HAS_FROZEN_BUNDLE = (
    (MODEL_ROOT / "seed_7071" / BASELINE_BRANCH / "scale_3" / "checkpoints" / "final.pt").exists()
    and (MODEL_ROOT / "seed_7071" / FULL_BRANCH / "scale_3" / "checkpoints" / "final.pt").exists()
    and (SCENARIO_ROOT / "scenario_2026.json").exists()
)


class SafetyMAPPOEvaluationTests(unittest.TestCase):
    def test_config_rejects_invalid_evaluation_design(self) -> None:
        kwargs = {
            "model_root": Path("model"),
            "scenario_root": Path("scenarios"),
            "output_dir": Path("output"),
        }
        with self.assertRaises(ValueError):
            SafetyMAPPOEvaluationConfig(**kwargs, policy_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyMAPPOEvaluationConfig(**kwargs, scenario_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyMAPPOEvaluationConfig(**kwargs, branches=(BASELINE_BRANCH, "unknown"))
        with self.assertRaises(ValueError):
            SafetyMAPPOEvaluationConfig(**kwargs, safety_projection_enabled=False)
        with self.assertRaises(ValueError):
            SafetyMAPPOEvaluationConfig(**kwargs, device="cuda")

    @unittest.skipUnless(HAS_FROZEN_BUNDLE, "frozen evaluation bundle is not available")
    def test_single_pair_runs_without_training_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_mappo_eval_") as temporary:
            root = Path(temporary)
            result = run_safety_mappo_evaluation(
                SafetyMAPPOEvaluationConfig(
                    model_root=MODEL_ROOT,
                    scenario_root=SCENARIO_ROOT,
                    output_dir=root / "data",
                    logs_dir=root / "logs",
                    plots_dir=root / "plots",
                    policy_seeds=(7071,),
                    scenario_seeds=(2026,),
                    branches=(BASELINE_BRANCH, FULL_BRANCH),
                    max_steps=1,
                    bootstrap_samples=100,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"])
            self.assertFalse(result["decision_supported"])
            report = json.loads((root / "logs" / "audit_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["audit_ok"])
            self.assertFalse(report["design"]["test_data_read"])
            self.assertFalse(report["design"]["trainer_update_called"])
            self.assertEqual(report["design"]["independent_unit"], "scenario_seed")
            self.assertTrue((root / "plots" / "safety_energy_comparison.png").exists())
            with (root / "logs" / "episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["branch"] for row in rows}, {BASELINE_BRANCH, FULL_BRANCH})
            self.assertTrue(all(int(row["trainer_update_called"]) == 0 for row in rows))
            self.assertTrue(all(int(row["task_complete"]) == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
