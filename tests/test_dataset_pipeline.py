"""Smoke tests for the dataset registry, schema, and benchmark generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dataset_registry import get_dataset, registry_as_dicts
from src.schema import EPISODE_SCHEMA_V1_1, EPISODE_SCHEMA_V1_2, validate_dataset_dir, validate_episode_benchmark, validate_episode_dir
from src.simulation.environment import MultiUAVJointEnv
from src.symbol_audit import audit_symbol_table
from src.synthetic.episode_generator import EpisodeBenchmarkConfig, _trajectory_rows, generate_episode_benchmark
from src.synthetic.flightmare_adapter import import_flightmare_episode
from src.synthetic.generator import SyntheticConfig, generate_benchmark


class DatasetPipelineTests(unittest.TestCase):
    def test_trajectory_compatibility_uses_numeric_no_link_sentinels(self) -> None:
        environment = MultiUAVJointEnv()
        truth = {
            "episode_id": "episode_000001",
            "scenario_id": "scenario_000001",
            "seed": 1,
            "timestamp": 0.2,
            "split": "train",
            "uav_id": "uav_00",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
        }
        row = _trajectory_rows(environment, [truth], [], [])[0]
        self.assertEqual(row["rssi"], environment.config.channel.min_rssi_dbm)
        self.assertEqual(row["snr"], environment.config.channel.min_snr_db)

    def test_registry_contains_only_python_generated_active_sources(self) -> None:
        keys = {entry["key"] for entry in registry_as_dicts()}
        self.assertTrue({"python_sim_benchmark", "python_sim_pilot", "python_sim_preliminary"} <= keys)
        self.assertFalse({"miluv", "uwb_uav_cluster", "aerpaw_aadm", "flightmare_benchmark"} & keys)
        self.assertEqual(get_dataset("PYTHON_SIM_BENCHMARK").key, "python_sim_benchmark")

    def test_generator_is_schema_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark"
            summary = generate_benchmark(
                SyntheticConfig(output_dir=output, scenarios=3, steps=5, dt=0.1, seed=7)
            )
            report = validate_dataset_dir(output)
            self.assertTrue(report["ok"], report)
            self.assertEqual(summary["trajectory_rows"], (3 + 5 + 7) * 5)
            self.assertTrue((output / "metadata.json").exists())
            first_bytes = (output / "trajectory.csv").read_bytes()

            second = Path(temp_dir) / "benchmark_second"
            generate_benchmark(
                SyntheticConfig(output_dir=second, scenarios=3, steps=5, dt=0.1, seed=7)
            )
            self.assertEqual(first_bytes, (second / "trajectory.csv").read_bytes())

    def test_episode_benchmark_contains_all_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "episodes"
            summary = generate_episode_benchmark(
                EpisodeBenchmarkConfig(
                    output_dir=output,
                    episodes=3,
                    duration_s=0.2,
                    physics_hz=10,
                    uwb_hz=5,
                    vision_hz=5,
                    gnss_hz=5,
                    link_hz=5,
                    decision_hz=5,
                    seed=11,
                )
            )
            report = validate_episode_benchmark(output)
            self.assertTrue(report["ok"], report)
            self.assertEqual(summary["episodes"], 3)
            self.assertTrue((output / "manifest.json").exists())
            episode = output / "train" / "episode_000001"
            self.assertTrue((episode / "packet_events.csv").exists())
            self.assertTrue((episode / "rewards.csv").exists())

    def test_episode_benchmark_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs = dict(
                episodes=1,
                duration_s=0.2,
                physics_hz=10,
                uwb_hz=5,
                vision_hz=5,
                gnss_hz=5,
                link_hz=5,
                decision_hz=5,
                seed=23,
            )
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            generate_episode_benchmark(EpisodeBenchmarkConfig(output_dir=first, **kwargs))
            generate_episode_benchmark(EpisodeBenchmarkConfig(output_dir=second, **kwargs))
            episode_files = (
                "links.csv",
                "actions.csv",
                "rewards.csv",
                "graph_features.csv",
            )
            for filename in episode_files:
                self.assertEqual(
                    (first / "train" / "episode_000001" / filename).read_bytes(),
                    (second / "train" / "episode_000001" / filename).read_bytes(),
                )

    def test_v11_episode_read_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "episodes"
            generate_episode_benchmark(
                EpisodeBenchmarkConfig(
                    output_dir=output,
                    episodes=1,
                    duration_s=0.1,
                    physics_hz=10,
                    uwb_hz=5,
                    vision_hz=5,
                    gnss_hz=5,
                    link_hz=5,
                    decision_hz=5,
                    seed=17,
                )
            )
            episode = output / "train" / "episode_000001"
            (episode / "estimates.csv").unlink()
            (episode / "graph_features.csv").unlink()
            report = validate_episode_dir(episode, schema_version=EPISODE_SCHEMA_V1_1)
            self.assertTrue(report["ok"], report)

    def test_v12_episode_read_compatibility_without_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "episodes"
            generate_episode_benchmark(
                EpisodeBenchmarkConfig(
                    output_dir=output,
                    episodes=1,
                    duration_s=0.1,
                    physics_hz=10,
                    uwb_hz=5,
                    vision_hz=5,
                    gnss_hz=5,
                    link_hz=5,
                    decision_hz=5,
                    seed=18,
                )
            )
            episode = output / "train" / "episode_000001"
            (episode / "ekf_diagnostics.csv").unlink()
            report = validate_episode_dir(episode, schema_version=EPISODE_SCHEMA_V1_2)
            self.assertTrue(report["ok"], report)

    def test_flightmare_contract_imports_valid_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_root = temp / "source"
            generate_episode_benchmark(
                EpisodeBenchmarkConfig(
                    output_dir=source_root,
                    episodes=1,
                    duration_s=0.1,
                    physics_hz=10,
                    uwb_hz=5,
                    vision_hz=5,
                    gnss_hz=5,
                    link_hz=5,
                    decision_hz=5,
                    seed=13,
                )
            )
            source = source_root / "train" / "episode_000001"
            target_root = temp / "imported"
            summary = import_flightmare_episode(
                source, target_root, "test", "episode_000999"
            )
            self.assertEqual(summary["engine_status"], "imported_flightmare_export")
            self.assertTrue(
                validate_episode_benchmark(target_root)["episodes"][0]["ok"]
            )

    def test_symbol_registry_has_no_duplicate_canonical_entries(self) -> None:
        report = audit_symbol_table(Path("sections/2_system_model.md"))
        self.assertTrue(report.ok, report.to_dict())
        self.assertGreaterEqual(report.canonical_symbols, 40)
        self.assertGreaterEqual(report.legacy_symbols, 4)


if __name__ == "__main__":
    unittest.main()
