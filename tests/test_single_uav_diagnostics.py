"""Tests for reproducible single-UAV episode diagnostic figures."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.single_uav_diagnostics import (
    SingleUAVDiagnosticsConfig,
    plot_single_uav_diagnostics,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class SingleUAVDiagnosticsTests(unittest.TestCase):
    def _make_episode(self, root: Path) -> Path:
        episode = root / "episode_000001"
        episode.mkdir(parents=True)
        (episode / "scenario.json").write_text(
            json.dumps(
                {
                    "episode_id": "episode_000001",
                    "scenario_id": "scenario_000001",
                    "seed": 1001,
                    "split": "test",
                    "uav_count": 1,
                    "scenario_generation": {"seed": 42},
                    "goals": {"uav_00": [2.0, 2.0, 10.0]},
                }
            ),
            encoding="utf-8",
        )
        common = {
            "episode_id": "episode_000001",
            "scenario_id": "scenario_000001",
            "seed": 1001,
            "split": "test",
            "uav_id": "uav_00",
        }
        timestamps = [0.2, 0.4, 0.6]
        _write_csv(
            episode / "trajectory.csv",
            ["episode_id", "scenario_id", "seed", "timestamp", "split", "uav_id", "x", "y", "z"],
            [
                {**common, "timestamp": timestamp, "x": index, "y": index * 0.5, "z": 10.0 + index * 0.1}
                for index, timestamp in enumerate(timestamps)
            ],
        )
        _write_csv(
            episode / "ground_truth.csv",
            ["episode_id", "scenario_id", "seed", "timestamp", "split", "uav_id", "x", "y", "z", "vx", "vy", "vz"],
            [
                {**common, "timestamp": timestamp, "x": index * 0.9, "y": index * 0.45, "z": 10.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
                for index, timestamp in enumerate(timestamps)
            ],
        )
        _write_csv(
            episode / "estimates.csv",
            ["episode_id", "scenario_id", "seed", "timestamp", "split", "uav_id", "x", "y", "z", "vx", "vy", "vz", "covariance_flat_json"],
            [
                {
                    **common,
                    "timestamp": timestamp,
                    "x": index,
                    "y": index * 0.5,
                    "z": 10.0 + index * 0.1,
                    "vx": 0.1,
                    "vy": 0.0,
                    "vz": 0.0,
                    "covariance_flat_json": json.dumps([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                }
                for index, timestamp in enumerate(timestamps)
            ],
        )
        _write_csv(
            episode / "rewards.csv",
            ["timestamp", "reward_total", "joint_objective", "normalized_uncertainty", "reward_collision", "reward_energy", "reward_connectivity"],
            [
                {
                    "timestamp": timestamp,
                    "reward_total": -0.5 + index * 0.1,
                    "joint_objective": 0.5 - index * 0.05,
                    "normalized_uncertainty": 0.8 - index * 0.1,
                    "reward_collision": 0.0,
                    "reward_energy": -0.01,
                    "reward_connectivity": 1.0,
                }
                for index, timestamp in enumerate(timestamps)
            ],
        )
        _write_csv(
            episode / "obstacles.csv",
            ["timestamp", "obstacle_id", "geometry_type", "x", "y", "z", "radius", "active"],
            [],
        )
        return episode

    def test_generates_time_series_trajectory_and_reward_figures(self) -> None:
        with tempfile.TemporaryDirectory(prefix="single_uav_diagnostics_") as temporary:
            episode = self._make_episode(Path(temporary) / "test")
            output = Path(temporary) / "plots"
            result = plot_single_uav_diagnostics(
                SingleUAVDiagnosticsConfig(episode_path=episode, output_dir=output)
            )

            for path in result["plots"].values():
                figure_path = Path(path)
                self.assertTrue(figure_path.exists(), figure_path)
                self.assertGreater(figure_path.stat().st_size, 0)
            manifest = json.loads((output / "diagnostics_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode"]["uav_count"], 1)
            self.assertEqual(manifest["episode"]["sample_count"], 3)
            self.assertEqual(manifest["episode"]["localization_points"], 3)
            self.assertEqual(manifest["episode"]["reward_points"], 3)
            self.assertEqual(
                manifest["x_axes"]["localization_error_timeseries"],
                "simulation_time_s",
            )
            self.assertIn("single-episode diagnostic only", manifest["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
