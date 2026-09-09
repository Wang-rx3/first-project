"""Structural tests for the formal 37d/41d safety mechanism audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.safety_motion_feature_formal_mechanism_audit import (
    BASELINE_37D,
    BASELINE_41D,
    DEFAULT_POLICY_SEEDS,
    DEFAULT_SCENARIO_SEEDS,
    FORMAL_BRANCHES,
    FULL_BRANCH,
    SafetyMotionFeatureFormalMechanismAuditConfig,
    _attribution,
    _merge_safety_and_policy_rows,
)


class SafetyMotionFeatureFormalMechanismAuditTests(unittest.TestCase):
    def _kwargs(self) -> dict[str, object]:
        return {
            "checkpoint_root": Path("checkpoints"),
            "scenario_root": Path("scenarios"),
        }

    def test_config_requires_the_three_formal_branches(self) -> None:
        with self.assertRaises(ValueError):
            SafetyMotionFeatureFormalMechanismAuditConfig(
                **self._kwargs(),
                branches=(BASELINE_37D, BASELINE_41D),
            )
        with self.assertRaises(ValueError):
            SafetyMotionFeatureFormalMechanismAuditConfig(
                **self._kwargs(),
                policy_seeds=(7071, 7071),
            )
        with self.assertRaises(ValueError):
            SafetyMotionFeatureFormalMechanismAuditConfig(
                **self._kwargs(),
                measurement_overhead_enabled=False,
            )

    def test_default_design_has_expected_independent_units(self) -> None:
        config = SafetyMotionFeatureFormalMechanismAuditConfig(**self._kwargs())
        self.assertEqual(config.branches, FORMAL_BRANCHES)
        self.assertEqual(config.policy_seeds, DEFAULT_POLICY_SEEDS)
        self.assertEqual(config.scenario_seeds, DEFAULT_SCENARIO_SEEDS)
        self.assertEqual(config.device, "cpu")

    def test_merge_records_kinematic_closure_without_changing_safety_values(self) -> None:
        safety = {
            "policy_seed": 7071,
            "scenario_seed": 2026,
            "branch": BASELINE_37D,
            "step": 1,
            "uav_id": "uav_00",
            "delta_v_commanded": np.asarray([1.0, 0.0, 0.0]),
            "delta_v_applied": np.asarray([0.5, 0.0, 0.0]),
            "delta_v_projection": np.asarray([-0.5, 0.0, 0.0]),
            "projection_pre_minimum_clearance_m": 3.0,
            "dynamic_clearance_before_update_m": 3.0,
            "dynamic_clearance_after_update_m": 3.0,
            "collision": 0,
        }
        policy = {
            "policy_seed": 7071,
            "scenario_seed": 2026,
            "branch": BASELINE_37D,
            "step": 1,
            "uav_id": "uav_00",
            "actor_node_feature_dim": 37,
            "actor_dynamic_feature_present": 0,
            "actor_dynamic_feature_finite": 1,
            "actor_observation_truth_free": 1,
            "trainer_update_called": 0,
            "test_data_read": 0,
            "final_measurement_uav_id": "none",
            "overhead_event_present": 0,
            "budget_overrun": 0,
        }
        merged, errors = _merge_safety_and_policy_rows([safety], [policy])
        self.assertFalse(errors)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["kinematic_closure"], 1)
        self.assertEqual(merged[0]["nominal_action_risk"], 0)

    def test_attribution_returns_exactly_one_priority_category(self) -> None:
        scenario_rows = []
        for branch in FORMAL_BRANCHES:
            scenario_rows.append(
                {
                    "scenario_seed": 2026,
                    "branch": branch,
                    "minimum_clearance_m": 1.0,
                    "energy_proxy": 0.1,
                }
            )
        merged = [
            {
                "branch": FULL_BRANCH,
                "scenario_seed": 2026,
                "collision": 1,
                "actor_node_feature_dim": 41,
                "actor_dynamic_feature_present": 1,
                "actor_dynamic_feature_finite": 1,
                "nominal_action_risk": 0,
                "dynamic_post_update_risk": 1,
                "safety_projected": 0,
                "final_measurement_uav_id": "none",
            }
        ]
        result = _attribution(merged, [], scenario_rows)
        self.assertEqual(result["primary_cause"], "projection_coverage_insufficient")
        self.assertIn(result["primary_cause"], result["priority_order"])


if __name__ == "__main__":
    unittest.main()
