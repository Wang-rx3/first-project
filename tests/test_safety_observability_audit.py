"""Tests for the read-only dynamic-obstacle observability audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.safety_mappo_evaluation import BASELINE_BRANCH, FULL_BRANCH
from src.experiments.safety_observability_audit import (
    SafetyObservabilityAuditConfig,
    _feature_profile,
    _horizon_context,
    run_safety_observability_audit,
)


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
SOURCE_TIMING_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "preliminary"
    / "safety_mappo_evaluation_seed2026"
    / "safety_timing_counterfactual"
)
HAS_FROZEN_AUDIT = (
    (SOURCE_TIMING_ROOT / "timing_step_diagnostics.csv").exists()
    and (SOURCE_TIMING_ROOT / "timing_counterfactual_events.csv").exists()
    and (SOURCE_TIMING_ROOT / "timing_audit_report.json").exists()
    and (SCENARIO_ROOT / "scenario_2026.json").exists()
)


class SafetyObservabilityAuditTests(unittest.TestCase):
    def test_config_is_cpu_deterministic_and_restricted_to_two_branches(self) -> None:
        kwargs = {
            "model_root": Path("model"),
            "scenario_root": Path("scenarios"),
        }
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(**kwargs, device="cuda")
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(**kwargs, deterministic=False)
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(
                **kwargs,
                branches=(BASELINE_BRANCH,),
            )
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(**kwargs, policy_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(**kwargs, scenario_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyObservabilityAuditConfig(**kwargs, prediction_horizons=(1, 2))

    def test_current_obstacle_feature_is_aggregate_and_has_no_motion_field(self) -> None:
        profile = _feature_profile(
            {
                "actor_observation_json": json.dumps(
                    {"obstacle_features": [2.5, 1.0, 0.0, 7.0]}
                ),
                "actor_observation_truth_free": "1",
            }
        )
        self.assertEqual(profile["actor_obstacle_feature_present"], 1)
        self.assertEqual(profile["actor_obstacle_feature_finite"], 1)
        self.assertEqual(profile["actor_obstacle_feature_dimension"], 4)
        self.assertEqual(profile["actor_obstacle_motion_feature_present"], 0)
        self.assertEqual(profile["actor_obstacle_id_feature_present"], 0)
        self.assertEqual(
            profile["actor_obstacle_feature_semantics"],
            "aggregate_min_distance_box_count_sphere_count_timestep",
        )

    def test_horizon_context_decodes_json_positions_and_is_read_only(self) -> None:
        obstacle_payload = [
            {
                "obstacle_id": "vehicle_00",
                "obstacle_type": "dynamic_vehicle",
                "geometry_type": "sphere",
                "active": 1,
                "center_m": [2.0, 0.0, 5.0],
                "velocity_mps": [-1.0, 0.0, 0.0],
                "radius_m": 0.5,
            }
        ]
        row = {
            "uav_id": "uav_00",
            "before_obstacle_state_json": json.dumps(obstacle_payload),
            "current_position_json": json.dumps([0.0, 0.0, 5.0]),
            "nominal_position_json": json.dumps([0.0, 0.0, 5.0]),
            "actual_position_json": json.dumps([0.0, 0.0, 5.0]),
        }
        original = json.loads(row["before_obstacle_state_json"])
        context = _horizon_context([row], horizons=(1, 2, 3), dt_s=0.2)
        self.assertTrue(np.array_equal(context["current_positions"]["uav_00"], [0.0, 0.0, 5.0]))
        self.assertEqual(context["horizon"]["1"]["predicted_dynamic_collision"], 1)
        self.assertLess(
            context["horizon"]["3"]["predicted_dynamic_min_clearance_m"],
            context["horizon"]["1"]["predicted_dynamic_min_clearance_m"],
        )
        self.assertEqual(row["before_obstacle_state_json"], json.dumps(original))

    @unittest.skipUnless(HAS_FROZEN_AUDIT, "frozen timing replay is not available")
    def test_single_pair_audit_reproduces_collision_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_observability_audit_") as temporary:
            root = Path(temporary)
            result = run_safety_observability_audit(
                SafetyObservabilityAuditConfig(
                    model_root=MODEL_ROOT,
                    scenario_root=SCENARIO_ROOT,
                    source_timing_audit_dir=SOURCE_TIMING_ROOT,
                    output_dir=root / "logs",
                    plots_dir=root / "plots",
                    policy_seeds=(7071,),
                    scenario_seeds=(2026,),
                    branches=(BASELINE_BRANCH, FULL_BRANCH),
                    bootstrap_samples=100,
                    force=True,
                    progress=False,
                )
            )
            self.assertTrue(result["audit_ok"])
            report = json.loads(
                (root / "logs" / "audit_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["checks"]["source_timing_report_available"])
            self.assertTrue(report["checks"]["current_collision_events_reproduced"])
            self.assertTrue(report["checks"]["actor_observations_truth_free"])
            self.assertEqual(report["counts"]["observability_events"], 1)
            self.assertEqual(report["primary_cause"], "observable_but_no_motion_feature")
            for filename in (
                "observability_events.csv",
                "safety_envelope_diagnostics.csv",
                "scenario_summary.csv",
                "paired_contrasts.csv",
                "audit_report.json",
            ):
                self.assertTrue((root / "logs" / filename).exists(), filename)
            self.assertTrue((root / "plots" / "safety_observability_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
