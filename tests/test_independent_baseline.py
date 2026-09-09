"""Acceptance tests for the paired independent fixed-policy baseline."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.baseline import (
    IndependentBaselineConfig,
    independent_fixed_actions,
    run_independent_baseline,
)
from src.experiments.formal import FormalExperimentConfig, run_preliminary
from src.simulation.motion_model import MotionConfig


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class IndependentBaselineActionTests(unittest.TestCase):
    def test_validation_and_test_are_the_only_allowed_evaluation_splits(self) -> None:
        for split in ("validation", "test"):
            config = IndependentBaselineConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split=split,
            )
            self.assertEqual(config.split, split)
        with self.assertRaises(ValueError):
            IndependentBaselineConfig(
                reference_scale=Path("reference"),
                output_dir=Path("output"),
                split="train",
            )

    def test_action_uses_local_estimate_and_disables_cooperative_measurement(self) -> None:
        observation = {
            "uav_00": {
                "estimate": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, 0.0] + [0.0] * 9),
                "goal": np.asarray([10.0, 0.0, 2.0]),
                "potential_targets": ["uav_01", "uav_02"],
            }
        }
        motion = MotionConfig(dt_s=0.2, max_acceleration_mps2=3.0)
        action = independent_fixed_actions(observation, motion)["uav_00"]
        self.assertEqual(action["measurement_uav_id"], "none")
        self.assertEqual(action["bandwidth"], {"uav_01": 0.5, "uav_02": 0.5})
        self.assertLessEqual(
            float(np.linalg.norm(action["delta_v"])),
            motion.max_acceleration_mps2 * motion.dt_s + 1e-12,
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required to create the MAPPO reference")
class IndependentBaselineIntegrationTests(unittest.TestCase):
    def test_reference_test_episode_is_replayed_without_neighbor_ekf_updates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="independent_baseline_") as temporary:
            root = Path(temporary)
            reference = root / "reference"
            run_preliminary(
                FormalExperimentConfig(
                    output_dir=reference,
                    uav_count=3,
                    seed=2026,
                    max_steps=2,
                    warmup_episodes=1,
                    train_episodes=1,
                    validation_episodes=1,
                    test_episodes=1,
                    progress=False,
                )
            )
            first = root / "baseline_first"
            second = root / "baseline_second"
            first_report = run_independent_baseline(
                IndependentBaselineConfig(
                    reference_scale=reference / "scale_3",
                    output_dir=first,
                    bootstrap_samples=200,
                    progress=False,
                )
            )
            run_independent_baseline(
                IndependentBaselineConfig(
                    reference_scale=reference / "scale_3",
                    output_dir=second,
                    bootstrap_samples=200,
                    progress=False,
                )
            )

            self.assertTrue(first_report["validation_ok"])
            self.assertEqual(first_report["paired_episodes"], 1)
            manifest = json.loads((first / "scale_3" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["controller"]["uses_ground_truth"])
            self.assertFalse(manifest["controller"]["uses_neighbor_state"])
            self.assertFalse(manifest["controller"]["uses_uwb_cooperative_update"])
            self.assertEqual(manifest["episodes"][0]["max_neighbor_updates"], 0)

            episode = first / "scale_3" / "test" / "episode_500001"
            with (episode / "actions.csv").open(encoding="utf-8", newline="") as handle:
                actions = list(csv.DictReader(handle))
            self.assertTrue(actions)
            self.assertEqual({row["policy_source"] for row in actions}, {"independent_fixed_baseline"})
            self.assertEqual({row["measurement_uav_id"] for row in actions}, {"none"})

            with (first / "scale_3" / "diagnostics" / "episode_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                metrics = list(csv.DictReader(handle))
            self.assertTrue(metrics)
            for field in ("j_loc", "j_com", "joint_objective"):
                self.assertTrue(np.isfinite(float(metrics[0][field])))

            paired = list(
                csv.DictReader(
                    (first / "scale_3" / "diagnostics" / "paired_metrics.csv").open(
                        encoding="utf-8", newline=""
                    )
                )
            )
            self.assertEqual({int(row["seed"]) for row in paired}, {502026})
            self.assertTrue((first / "scale_3" / "diagnostics" / "baseline_comparison.png").exists())

            relative = Path("scale_3/test/episode_500001/actions.csv")
            self.assertEqual(
                hashlib.sha256((first / relative).read_bytes()).hexdigest(),
                hashlib.sha256((second / relative).read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
