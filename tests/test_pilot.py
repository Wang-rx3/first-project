"""Acceptance tests for the reproducible three-UAV closed-loop pilot."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.schema import validate_episode_benchmark


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the MAPPO pilot")
class PilotTests(unittest.TestCase):
    def _run(self, root: Path) -> dict[str, object]:
        from src.experiments.pilot import PilotConfig, run_pilot

        return run_pilot(
            PilotConfig(
                output_dir=root,
                seed=2026,
                max_steps=2,
                warmup_episodes=1,
                integration_steps=2,
                train_episodes=1,
                validation_episodes=1,
                device="cpu",
            )
        )

    def test_pilot_closes_the_environment_learning_and_logging_loop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pilot_acceptance_") as temporary:
            root = Path(temporary) / "pilot"
            summary = self._run(root)

            report = validate_episode_benchmark(root)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["schema_version"], "episode-v1.3")
            self.assertEqual(report["episode_count"], 3)
            self.assertGreater(summary["probe"]["effective_edges_seen"], 0)
            self.assertGreater(summary["probe"]["neighbor_updates"], 0)
            self.assertGreater(summary["parameter_delta"]["delta_l2"], 0.0)

            self.assertTrue((root / "normalization_stats.json").exists())
            self.assertTrue((root / "diagnostics" / "attention_maps.jsonl").exists())
            self.assertTrue((root / "diagnostics" / "training_metrics.jsonl").exists())
            self.assertTrue((root / "diagnostics" / "parameter_delta.json").exists())
            self.assertTrue((root / "train" / "episode_000001" / "ekf_diagnostics.csv").exists())

            trace = json.loads(
                (root / "diagnostics" / "attention_maps.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )["trace"]
            self.assertEqual(trace["layer_count"], 2)
            self.assertTrue(trace["virtual_self_loops_added"])
            self.assertEqual(len(trace["physical_edge_index"]), 2)
            self.assertEqual(len(trace["attention_edge_index"]), 2)
            self.assertEqual(len(trace["attention_coefficients"]), 2)
            self.assertTrue(trace["attention_maps"])

            metrics = [
                json.loads(line)
                for line in (root / "diagnostics" / "training_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(metrics), 1)
            for key in (
                "episode",
                "update",
                "reward",
                "mean_reward",
                "loss",
                "actor_loss",
                "critic_loss",
                "entropy",
                "clip_fraction",
                "gradient_norm",
            ):
                self.assertIn(key, metrics[0])
                if key not in ("episode",):
                    self.assertTrue(float(metrics[0][key]) == float(metrics[0][key]))
            self.assertEqual(metrics[0]["phase"], "train")

            with (root / "train" / "episode_000001" / "actions.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                action_rows = list(csv.DictReader(handle))
            self.assertTrue(action_rows)
            self.assertEqual({row["policy_source"] for row in action_rows}, {"mappo"})
            self.assertTrue(
                all(float(row["bandwidth_ratio"]) <= 1.0 + 1e-6 for row in action_rows)
            )

    def test_same_seed_reproduces_integration_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pilot_repeatability_") as temporary:
            first_root = Path(temporary) / "first"
            second_root = Path(temporary) / "second"
            self._run(first_root)
            self._run(second_root)
            for relative_path in (
                "normalization_stats.json",
                "checks/probe_summary.json",
                "checks/determinism.json",
                "diagnostics/training_metrics.jsonl",
                "diagnostics/parameter_delta.json",
                "train/episode_000001/actions.csv",
                "train/episode_000001/rewards.csv",
            ):
                self.assertEqual(
                    _sha256(first_root / relative_path),
                    _sha256(second_root / relative_path),
                    relative_path,
                )


if __name__ == "__main__":
    unittest.main()
