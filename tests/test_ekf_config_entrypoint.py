"""Tests for the single EKF configuration entry point."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cli import _build_parser, _resolve_preliminary_ekf_parameters
from src.experiments.formal import load_selected_ekf_config


class EKFConfigEntrypointTests(unittest.TestCase):
    def test_selected_parameters_are_loaded_with_legacy_vo_alias(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ekf_config_") as temporary:
            path = Path(temporary) / "selected_ekf_config.json"
            path.write_text(
                json.dumps(
                    {
                        "selected": {
                            "vo_std_m": 0.05,
                            "process_position_std_m": 0.03,
                            "process_velocity_std_mps": 0.15,
                            "nlos_variance_multiplier": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_selected_ekf_config(path),
                {
                    "vo_position_std_m": 0.05,
                    "process_position_std_m": 0.03,
                    "process_velocity_std_mps": 0.15,
                    "nlos_variance_multiplier": 1.0,
                },
            )

    def test_run_preliminary_uses_selected_config_when_cli_values_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ekf_config_") as temporary:
            path = Path(temporary) / "selected_ekf_config.json"
            path.write_text(
                json.dumps(
                    {
                        "selected_parameters": {
                            "vo_position_std_m": 0.05,
                            "process_position_std_m": 0.03,
                            "process_velocity_std_mps": 0.15,
                            "nlos_variance_multiplier": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            parser = _build_parser()
            args = parser.parse_args(
                ["run-preliminary", "--uav-counts", "3", "--ekf-config", str(path)]
            )
            values, source = _resolve_preliminary_ekf_parameters(args, parser)
            self.assertEqual(values["vo_position_std_m"], 0.05)
            self.assertEqual(values["process_position_std_m"], 0.03)
            self.assertEqual(values["process_velocity_std_mps"], 0.15)
            self.assertEqual(values["nlos_variance_multiplier"], 1.0)
            self.assertEqual(source, str(path.resolve()))

    def test_conflicting_explicit_parameter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ekf_config_") as temporary:
            path = Path(temporary) / "selected_ekf_config.json"
            path.write_text(
                json.dumps(
                    {
                        "selected_parameters": {
                            "vo_position_std_m": 0.05,
                            "process_position_std_m": 0.03,
                            "process_velocity_std_mps": 0.15,
                            "nlos_variance_multiplier": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            parser = _build_parser()
            args = parser.parse_args(
                [
                    "run-preliminary",
                    "--uav-counts",
                    "3",
                    "--ekf-config",
                    str(path),
                    "--vo-std",
                    "0.03",
                ]
            )
            with self.assertRaises(SystemExit):
                _resolve_preliminary_ekf_parameters(args, parser)


if __name__ == "__main__":
    unittest.main()
