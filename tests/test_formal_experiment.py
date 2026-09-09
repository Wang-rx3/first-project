"""Acceptance tests for the scalable preliminary UAV experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.formal import (
    FormalExperimentConfig,
    run_preliminary,
    summarize_preliminary,
    validate_preliminary,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for preliminary MAPPO experiments")
class FormalExperimentTests(unittest.TestCase):
    @staticmethod
    def _config(root: Path, uav_count: int) -> FormalExperimentConfig:
        return FormalExperimentConfig(
            output_dir=root,
            uav_count=uav_count,
            seed=2026,
            max_steps=2,
            warmup_episodes=1,
            train_episodes=1,
            validation_episodes=1,
            test_episodes=1,
            device="cpu",
        )

    def test_all_three_scales_generate_valid_episodes_metrics_and_plots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preliminary_scales_") as temporary:
            root = Path(temporary) / "experiment"
            summaries = [run_preliminary(self._config(root, count)) for count in (3, 5, 7)]
            aggregate = summarize_preliminary(
                root,
                logs_output=root / "logs",
                plots_output=root / "plots",
            )
            report = validate_preliminary(root)

            self.assertTrue(report["ok"], report)
            self.assertEqual(report["scales"].keys(), {"scale_3", "scale_5", "scale_7"})
            self.assertEqual(aggregate["scales"], [3, 5, 7])
            self.assertEqual(len(summaries), 3)
            for count in (3, 5, 7):
                scale = root / f"scale_{count}"
                self.assertTrue((scale / "checkpoints" / "final.pt").exists())
                self.assertGreater(float(summaries[count // 2 - 1]["parameter_delta"]["delta_l2"]), 0.0)
                with (scale / "diagnostics" / "episode_metrics.csv").open(encoding="utf-8") as handle:
                    metrics = list(csv.DictReader(handle))
                self.assertEqual({row["split"] for row in metrics}, {"train", "validation", "test"})
                self.assertTrue(all(float(row["position_rmse_m"]) >= 0.0 for row in metrics))
                self.assertTrue(all(float(row["position_rmse_norm_m"]) >= 0.0 for row in metrics))
                self.assertTrue(all(float(row["position_p95_norm_m"]) >= 0.0 for row in metrics))
                initial_hash_path = scale / "diagnostics" / "initial_parameter_hashes.json"
                self.assertTrue(initial_hash_path.exists())
                initial_hashes = json.loads(initial_hash_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    set(initial_hashes["modules"]),
                    {
                        "gat",
                        "actor_body",
                        "measurement_head",
                        "bandwidth_head",
                        "delta_mean_head",
                        "delta_logstd",
                        "critic",
                    },
                )

            for filename in (
                "learning_curves.png",
                "scale_comparison.png",
                "communication_localization_metrics.png",
            ):
                self.assertTrue((root / "plots" / filename).exists())
                self.assertGreater((root / "plots" / filename).stat().st_size, 0)

            training_metrics = [
                json.loads(line)
                for line in (root / "scale_3" / "diagnostics" / "training_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(training_metrics), 1)
            self.assertTrue(float(training_metrics[0]["gradient_norm"]) >= 0.0)

    def test_same_seed_reproduces_one_scale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preliminary_repeatability_") as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            run_preliminary(self._config(first, 3))
            run_preliminary(self._config(second, 3))
            for relative_path in (
                "scale_3/train/episode_000001/actions.csv",
                "scale_3/train/episode_000001/rewards.csv",
                "scale_3/diagnostics/episode_metrics.csv",
                "scale_3/diagnostics/attention_summary.csv",
            ):
                self.assertEqual(
                    _sha256(first / relative_path),
                    _sha256(second / relative_path),
                    relative_path,
                )

    def test_single_uav_sanity_run_has_no_cooperative_edges_and_updates_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="preliminary_single_uav_") as temporary:
            root = Path(temporary) / "experiment"
            summary = run_preliminary(
                FormalExperimentConfig(
                    output_dir=root,
                    uav_count=1,
                    seed=2026,
                    scenario_seed=2026,
                    max_steps=2,
                    warmup_episodes=1,
                    train_episodes=1,
                    validation_episodes=1,
                    test_episodes=1,
                    device="cpu",
                )
            )
            report = validate_preliminary(root)

            self.assertTrue(report["ok"], report)
            self.assertEqual(set(report["scales"]), {"scale_1"})
            self.assertEqual(summary["uav_count"], 1)
            self.assertGreater(float(summary["parameter_delta"]["delta_l2"]), 0.0)
            manifest = json.loads(
                (root / "scale_1" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["uav_count"], 1)
            self.assertEqual(manifest["result_status"], "single_uav_sanity_only")
            self.assertEqual(manifest["algorithm_role"], "single_agent_ppo_sanity")
            self.assertFalse(manifest["cooperative_links_expected"])
            self.assertEqual(manifest["config"]["scenario_seed"], 2026)
            with (root / "scale_1" / "diagnostics" / "episode_metrics.csv").open(
                encoding="utf-8"
            ) as handle:
                self.assertTrue(
                    all(row["effective_edges_seen"] == "0" for row in csv.DictReader(handle))
                )


if __name__ == "__main__":
    unittest.main()
