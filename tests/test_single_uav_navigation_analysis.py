from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.single_uav_navigation_analysis import (
    analyze_single_uav_navigation,
)


class SingleUAVNavigationAnalysisTests(unittest.TestCase):
    def _write_episode(
        self,
        scale: Path,
        split: str,
        episode_id: str,
        distances: list[float],
        speeds: list[float],
        stable: list[int],
    ) -> None:
        episode = scale / split / episode_id
        episode.mkdir(parents=True)
        (episode / "scenario.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "seed": 2026,
                    "uav_count": 1,
                    "termination_reason": "goal_reached" if any(stable) else "time_limit",
                    "reward_config": {
                        "single_uav_goal_threshold_m": 1.0,
                        "single_uav_near_goal_radius_m": 5.0,
                        "single_uav_goal_speed_threshold_mps": 0.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        with (episode / "rewards.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "current_goal_distance",
                    "single_uav_speed_mps",
                    "single_uav_goal_stable",
                    "single_uav_stable_steps",
                    "single_uav_stable_hold_steps",
                ],
            )
            writer.writeheader()
            stable_steps = 0
            for index, (distance, speed, stable_value) in enumerate(
                zip(distances, speeds, stable)
            ):
                stable_steps = stable_steps + 1 if stable_value else 0
                writer.writerow(
                    {
                        "timestamp": (index + 1) * 0.2,
                        "current_goal_distance": distance,
                        "single_uav_speed_mps": speed,
                        "single_uav_goal_stable": stable_value,
                        "single_uav_stable_steps": stable_steps,
                        "single_uav_stable_hold_steps": 5,
                    }
                )
        with (episode / "actions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "pre_speed_mps",
                    "post_speed_mps",
                    "speed_reduction_mps",
                    "applied_longitudinal_delta_v_mps",
                    "commanded_braking_delta_v_mps",
                    "braking_delta_v_mps",
                    "braking_action",
                    "brake_effective",
                    "brake_blocked",
                ],
            )
            writer.writeheader()
            for index, timestamp in enumerate(
                (index + 1) * 0.2 for index in range(len(speeds))
            ):
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "pre_speed_mps": speeds[index] + 0.1,
                        "post_speed_mps": speeds[index],
                        "speed_reduction_mps": 0.1,
                        "applied_longitudinal_delta_v_mps": -0.1,
                        "commanded_braking_delta_v_mps": 0.1,
                        "braking_delta_v_mps": 0.1,
                        "braking_action": 1,
                        "brake_effective": 1,
                        "brake_blocked": 0,
                    }
                )

    def test_reports_entry_rates_and_near_goal_speed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="single_uav_analysis_") as temporary:
            root = Path(temporary)
            scale = root / "scale_1"
            self._write_episode(
                scale,
                "train",
                "episode_000001",
                [6.0, 4.0, 0.8],
                [2.0, 1.5, 0.4],
                [0, 0, 1],
            )
            self._write_episode(
                scale,
                "validation",
                "episode_400001",
                [8.0, 7.0],
                [2.0, 2.2],
                [0, 0],
            )
            self._write_episode(
                scale,
                "test",
                "episode_500001",
                [6.0, 4.5],
                [2.0, 1.8],
                [0, 0],
            )

            output = root / "plots"
            metrics = root / "logs" / "navigation.csv"
            result = analyze_single_uav_navigation(
                scale,
                output_dir=output,
                metrics_output_path=metrics,
            )

            self.assertEqual(result["summary"]["train"]["entered_near_goal_episodes"], 1)
            self.assertEqual(result["summary"]["train"]["entered_goal_radius_episodes"], 1)
            self.assertEqual(result["summary"]["train"]["stable_goal_episodes"], 1)
            self.assertEqual(
                result["summary"]["train"]["maximum_consecutive_goal_hold_steps"],
                1.0,
            )
            self.assertAlmostEqual(
                result["summary"]["train"][
                    "mean_maximum_consecutive_goal_hold_steps"
                ],
                1.0,
            )
            self.assertEqual(
                result["summary"]["train"][
                    "maximum_consecutive_goal_hold_steps_max"
                ],
                1.0,
            )
            self.assertEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_1"],
                1,
            )
            self.assertEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_3"],
                0,
            )
            self.assertEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_5"],
                0,
            )
            self.assertAlmostEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_1_rate"],
                1.0,
            )
            self.assertAlmostEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_3_rate"],
                0.0,
            )
            self.assertAlmostEqual(
                result["summary"]["train"]["episodes_with_hold_steps_ge_5_rate"],
                0.0,
            )
            self.assertEqual(result["summary"]["test"]["entered_near_goal_episodes"], 1)
            self.assertEqual(result["summary"]["test"]["entered_goal_radius_episodes"], 0)
            self.assertAlmostEqual(
                result["summary"]["train"]["mean_speed_at_near_goal_entry_mps"],
                1.5,
            )
            self.assertEqual(result["summary"]["train"]["near_goal_braking_action_count"], 2)
            self.assertAlmostEqual(
                result["summary"]["train"]["near_goal_effective_braking_given_action_fraction"],
                1.0,
            )
            self.assertTrue(metrics.exists())
            header = metrics.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("maximum_consecutive_goal_hold_steps", header)
            self.assertTrue((output / "navigation_failure_analysis.png").exists())
            self.assertTrue((output / "speed_distance_relation.png").exists())
            self.assertTrue((output / "braking_action_summary.png").exists())


if __name__ == "__main__":
    unittest.main()
