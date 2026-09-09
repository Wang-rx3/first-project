"""Tests for the read-only dynamic-obstacle timing counterfactual audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.safety_mappo_evaluation import (
    BASELINE_BRANCH,
    FULL_BRANCH,
)
from src.experiments.safety_timing_counterfactual import (
    SafetyTimingCounterfactualConfig,
    _obstacles_from_payload,
    _collision_cause,
    _predict_obstacles,
    _snapshot_equal,
    run_safety_timing_counterfactual,
)
from src.simulation.obstacle_model import SphereObstacle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "data" / "synthetic" / (
    "python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826"
)
SCENARIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "safety_risk_screening_seed2026"
    / "scenarios"
)
HAS_FROZEN_BUNDLE = (
    (
        MODEL_ROOT
        / "seed_7071"
        / BASELINE_BRANCH
        / "scale_3"
        / "checkpoints"
        / "final.pt"
    ).exists()
    and (
        MODEL_ROOT
        / "seed_7071"
        / FULL_BRANCH
        / "scale_3"
        / "checkpoints"
        / "final.pt"
    ).exists()
    and (SCENARIO_ROOT / "scenario_2026.json").exists()
)


class SafetyTimingCounterfactualTests(unittest.TestCase):
    def test_config_is_deterministic_and_restricted_to_two_branches(self) -> None:
        kwargs = {
            "model_root": Path("model"),
            "scenario_root": Path("scenarios"),
        }
        with self.assertRaises(ValueError):
            SafetyTimingCounterfactualConfig(**kwargs, device="cuda")
        with self.assertRaises(ValueError):
            SafetyTimingCounterfactualConfig(**kwargs, deterministic=False)
        with self.assertRaises(ValueError):
            SafetyTimingCounterfactualConfig(
                **kwargs,
                branches=(BASELINE_BRANCH,),
            )
        with self.assertRaises(ValueError):
            SafetyTimingCounterfactualConfig(**kwargs, policy_seeds=(1, 1))

    def test_prediction_is_isolated_and_uses_obstacle_update_rule(self) -> None:
        obstacle = SphereObstacle(
            "vehicle_00",
            np.array([1.0, 2.0, 5.0]),
            radius_m=0.5,
            velocity_mps=np.array([2.0, 0.0, 0.0]),
            obstacle_type="dynamic_vehicle",
        )
        payload = [
            {
                "obstacle_id": "vehicle_00",
                "obstacle_type": "dynamic_vehicle",
                "geometry_type": "sphere",
                "active": 1,
                "center_m": obstacle.center_m.copy(),
                "velocity_mps": obstacle.velocity_mps.copy(),
                "radius_m": obstacle.radius_m,
            }
        ]
        original_center = obstacle.center_m.copy()
        predicted = _predict_obstacles(
            payload,
            dt_s=0.5,
            bounds_min_m=np.array([-100.0, -100.0, 0.0]),
            bounds_max_m=np.array([100.0, 100.0, 100.0]),
        )
        self.assertTrue(np.array_equal(original_center, obstacle.center_m))
        self.assertTrue(
            np.allclose(predicted[0]["center_m"], np.array([2.0, 2.0, 5.0]))
        )
        self.assertEqual(len(_obstacles_from_payload(predicted)), 1)
        self.assertFalse(_snapshot_equal(payload, predicted))

    def test_collision_cause_keeps_timing_and_projection_cases_distinct(self) -> None:
        self.assertEqual(
            _collision_cause({"actual_collision": 0}),
            "none",
        )
        self.assertEqual(
            _collision_cause(
                {
                    "actual_collision": 1,
                    "actual_collision_source_type": "uav_obstacle",
                    "dynamic_timing_collision": 1,
                }
            ),
            "dynamic_obstacle_motion",
        )
        self.assertEqual(
            _collision_cause(
                {
                    "actual_collision": 1,
                    "actual_collision_source_type": "uav_obstacle",
                    "pre_dynamic_actual_collision": 1,
                    "safety_projected": 0,
                }
            ),
            "projection_not_triggered",
        )
        self.assertEqual(
            _collision_cause(
                {
                    "actual_collision": 1,
                    "actual_collision_source_type": "uav_uav",
                }
            ),
            "uav_separation",
        )

    @unittest.skipUnless(HAS_FROZEN_BUNDLE, "frozen safety bundle is not available")
    def test_single_pair_writes_read_only_timing_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_timing_audit_") as temporary:
            root = Path(temporary)
            result = run_safety_timing_counterfactual(
                SafetyTimingCounterfactualConfig(
                    model_root=MODEL_ROOT,
                    scenario_root=SCENARIO_ROOT,
                    output_dir=root / "logs",
                    plots_dir=root / "plots",
                    reference_audit_dir=None,
                    policy_seeds=(7071,),
                    scenario_seeds=(2026,),
                    branches=(BASELINE_BRANCH, FULL_BRANCH),
                    max_steps=2,
                    bootstrap_samples=100,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"])
            report_path = root / "logs" / "timing_audit_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["checks"]["counterfactual_state_unchanged"])
            self.assertTrue(report["checks"]["obstacle_prediction_closure"])
            self.assertTrue(report["checks"]["test_data_read_zero"])
            self.assertTrue((root / "logs" / "timing_counterfactual_events.csv").exists())
            self.assertTrue((root / "logs" / "timing_step_diagnostics.csv").exists())
            self.assertTrue((root / "plots" / "safety_timing_counterfactual.png").exists())


if __name__ == "__main__":
    unittest.main()
