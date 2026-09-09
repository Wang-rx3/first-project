"""Tests for frozen screening-profile integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.formal import FormalExperimentConfig, _make_environment, _scenario_for
from src.simulation.scenario import add_moving_los_screen, generate_scenario
from src.simulation.scenario_profile import load_scenario_profile


def _profile_payload(*, allowed: bool = True) -> dict[str, object]:
    return {
        "profile": "moderate",
        "uav_count": 3,
        "selection_status": "eligible_frozen_candidate",
        "allowed_for_training": allowed,
        "design_point_id": "point_022",
        "severity_score": 0.24,
        "parameters": {
            "max_comm_range_m": 27.7,
            "total_bandwidth_hz": 1_000_000.0,
            "nlos_extra_loss_db": 22.8,
            "message_arrivals_per_window": 4.8,
        },
        "screening_design_seed": 2026,
        "scenario_seeds": [2026, 3035],
        "mappo_performance_used_for_selection": False,
    }


class ScenarioProfileTests(unittest.TestCase):
    def test_profile_loads_and_applies_exact_channel_and_topology_parameters(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scenario_profile_") as temporary:
            path = Path(temporary) / "moderate.json"
            path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
            profile = load_scenario_profile(path, expected_uav_count=3)
            config = FormalExperimentConfig(
                output_dir=Path(temporary) / "output",
                uav_count=3,
                scenario_profile_path=path,
            )
            environment = _make_environment(config, seed=2026)
            parameters = profile["parameters"]
            self.assertEqual(environment.config.topology.max_comm_range_m, parameters["max_comm_range_m"])
            self.assertEqual(
                environment.config.topology.message_arrivals_per_window,
                parameters["message_arrivals_per_window"],
            )
            self.assertEqual(environment.config.channel.total_bandwidth_hz, parameters["total_bandwidth_hz"])
            self.assertEqual(environment.config.channel.nlos_extra_loss_db, parameters["nlos_extra_loss_db"])

    def test_disallowed_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scenario_profile_") as temporary:
            path = Path(temporary) / "hard.json"
            path.write_text(json.dumps(_profile_payload(allowed=False)), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scenario_profile(path, expected_uav_count=3)

    def test_formal_profile_adds_shared_moving_los_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scenario_profile_") as temporary:
            path = Path(temporary) / "moderate.json"
            path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
            config = FormalExperimentConfig(
                output_dir=Path(temporary) / "output",
                uav_count=3,
                scenario_profile_path=path,
            )
            scenario = _scenario_for(config, 2026)
            obstacle_ids = [item["obstacle_id"] for item in scenario["obstacle_specs"]]
            self.assertIn("screening_moving_nlos_screen", obstacle_ids)
            self.assertEqual(
                scenario["screening_geometry"]["type"],
                "moving_3d_los_screen",
            )

    def test_shared_geometry_is_deterministic_and_does_not_mutate_input(self) -> None:
        scenario = generate_scenario(2026, 3)
        original_count = len(scenario["obstacle_specs"])
        first = add_moving_los_screen(scenario)
        second = add_moving_los_screen(scenario)
        self.assertEqual(first, second)
        self.assertEqual(len(scenario["obstacle_specs"]), original_count)
        self.assertEqual(len(first["obstacle_specs"]), original_count + 1)


if __name__ == "__main__":
    unittest.main()
