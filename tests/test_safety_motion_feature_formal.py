"""Fast structural tests for the formal safety-motion experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.formal import FormalExperimentConfig, _scenario_for
from src.experiments.safety_motion_feature_formal_training import (
    DEFAULT_TEST_SCENARIO_SEEDS,
    DEFAULT_TRAIN_SCENARIO_SEEDS,
    DEFAULT_VALIDATION_SCENARIO_SEEDS,
    FORMAL_BRANCHES,
    SafetyMotionFeatureFormalTrainingConfig,
    ensure_formal_scenario_pool,
)


PROFILE = Path("data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json")


class SafetyMotionFeatureFormalTests(unittest.TestCase):
    def test_pool_has_disjoint_frozen_splits_and_safe_initial_states(self) -> None:
        with tempfile.TemporaryDirectory(prefix="formal_safety_pool_") as temporary:
            root = Path(temporary) / "pool"
            manifest = ensure_formal_scenario_pool(
                root,
                scenario_profile_path=PROFILE,
                progress=False,
            )
            self.assertEqual(
                tuple(manifest["splits"]["train"]["seeds"]),
                DEFAULT_TRAIN_SCENARIO_SEEDS,
            )
            self.assertEqual(
                tuple(manifest["splits"]["validation"]["seeds"]),
                DEFAULT_VALIDATION_SCENARIO_SEEDS,
            )
            self.assertEqual(
                tuple(manifest["splits"]["test"]["seeds"]),
                DEFAULT_TEST_SCENARIO_SEEDS,
            )
            all_seeds = [
                item["seed"]
                for split in manifest["splits"].values()
                for item in split["scenarios"]
            ]
            self.assertEqual(len(all_seeds), len(set(all_seeds)))
            self.assertTrue(
                all(
                    item["initial_state_safe"]
                    for split in manifest["splits"].values()
                    for item in split["scenarios"]
                )
            )

            config = FormalExperimentConfig(
                output_dir=root / "run",
                scenario_profile_path=PROFILE,
                scenario_pool_root=root,
                warmup_episodes=1,
                train_episodes=1,
                validation_episodes=1,
                test_episodes=1,
            )
            train_first = _scenario_for(config, 123, split="train", episode_index=0)
            train_wrap = _scenario_for(config, 123, split="train", episode_index=20)
            validation_first = _scenario_for(config, 123, split="validation", episode_index=0)
            self.assertEqual(train_first["seed"], DEFAULT_TRAIN_SCENARIO_SEEDS[0])
            self.assertEqual(train_wrap["seed"], DEFAULT_TRAIN_SCENARIO_SEEDS[0])
            self.assertEqual(validation_first["seed"], DEFAULT_VALIDATION_SCENARIO_SEEDS[0])

    def test_training_config_requires_overhead_and_unique_policy_seeds(self) -> None:
        with self.assertRaises(ValueError):
            SafetyMotionFeatureFormalTrainingConfig(
                scenario_pool_root=Path("pool"),
                measurement_overhead_enabled=False,
            )
        with self.assertRaises(ValueError):
            SafetyMotionFeatureFormalTrainingConfig(
                scenario_pool_root=Path("pool"),
                policy_seeds=(2026, 2026),
            )

    def test_formal_branch_set_is_fixed(self) -> None:
        self.assertEqual(
            FORMAL_BRANCHES,
            ("baseline_37d", "baseline_41d", "budgeted_quality_aware_41d"),
        )


if __name__ == "__main__":
    unittest.main()
