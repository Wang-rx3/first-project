"""Unit tests for the independent-seed frozen-selector batch driver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.multiseed_soft_quality_validation import (
    DEFAULT_SEEDS,
    MultiSeedSoftQualityValidationConfig,
    _audit,
    _cross_seed_summary,
)


class MultiSeedSoftQualityValidationTests(unittest.TestCase):
    def test_default_design_has_five_unique_seeds_and_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = MultiSeedSoftQualityValidationConfig(output_dir=Path(temporary))
            self.assertEqual(config.seeds, DEFAULT_SEEDS)
            self.assertEqual(len(set(config.seeds)), 5)
            self.assertEqual(config.device, "cpu")

    def test_cross_seed_bootstrap_uses_seed_level_values(self) -> None:
        config = MultiSeedSoftQualityValidationConfig(
            output_dir=Path("unused"),
            seeds=(2026, 3035, 4044, 5053, 6062),
            bootstrap_samples=1000,
        )
        rows = []
        for seed, rmse, harmful in zip(config.seeds, (-.01, -.02, -.03, -.01, -.02), (-.01, -.02, -.01, -.03, -.02)):
            rows.extend(
                [
                    {
                        "base_seed": seed,
                        "metric": "position_rmse_m",
                        "mean_paired_difference": rmse,
                        "comparison": "soft_quality_candidate_minus_round_robin_candidate",
                    },
                    {
                        "base_seed": seed,
                        "metric": "harmful_fraction",
                        "mean_paired_difference": harmful,
                        "comparison": "soft_quality_candidate_minus_round_robin_candidate",
                    },
                ]
            )
        summary = _cross_seed_summary(rows, config)
        by_metric = {row["metric"]: row for row in summary}
        self.assertEqual(by_metric["position_rmse_m"]["seed_count"], 5)
        self.assertEqual(by_metric["position_rmse_m"]["improved_seed_count"], 5)
        self.assertTrue(by_metric["position_rmse_m"]["ci_supports_improvement"])

    def test_audit_requires_both_rmse_and_harmful_fraction(self) -> None:
        config = MultiSeedSoftQualityValidationConfig(
            output_dir=Path("unused"),
            seeds=(2026, 3035, 4044, 5053, 6062),
        )
        seed_runs = [{"audit_ok": True} for _ in config.seeds]
        means = [
            {"episode_count": 20, "treatment": treatment}
            for seed in config.seeds
            for treatment in ("recorded_mappo", "round_robin_candidate", "quality_aware_candidate", "soft_quality_candidate")
        ]
        contrasts = [
            {"comparison": "soft_quality_candidate_minus_round_robin_candidate"}
            for _ in range(5)
        ]
        cross_seed = [
            {
                "metric": "position_rmse_m",
                "improved_seed_count": 4,
                "ci_supports_improvement": True,
                "mean_across_seeds": -0.01,
                "bootstrap_95_ci_low": -0.02,
                "bootstrap_95_ci_high": -0.001,
            },
            {
                "metric": "harmful_fraction",
                "improved_seed_count": 3,
                "ci_supports_improvement": False,
                "mean_across_seeds": -0.01,
                "bootstrap_95_ci_low": -0.02,
                "bootstrap_95_ci_high": 0.001,
            },
        ]
        report = _audit(config, seed_runs, means, contrasts, cross_seed)
        self.assertFalse(report["decision_gate"] == "soft_quality_stable_across_seeds")
        self.assertTrue(report["overall_ok"])


if __name__ == "__main__":
    unittest.main()
