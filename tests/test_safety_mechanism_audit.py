"""Unit and smoke tests for the frozen safety-mechanism audit."""

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
from src.experiments.safety_mechanism_audit import (
    SafetyMechanismAuditConfig,
    _motion_risk,
    _scene_clearance_detail,
    run_safety_mechanism_audit,
)
from src.simulation.obstacle_model import BoxObstacle, SphereObstacle


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
    (MODEL_ROOT / "seed_7071" / BASELINE_BRANCH / "scale_3" / "checkpoints" / "final.pt").exists()
    and (MODEL_ROOT / "seed_7071" / FULL_BRANCH / "scale_3" / "checkpoints" / "final.pt").exists()
    and (SCENARIO_ROOT / "scenario_2026.json").exists()
)


class SafetyMechanismAuditTests(unittest.TestCase):
    def test_config_requires_two_frozen_comparison_branches(self) -> None:
        kwargs = {
            "model_root": Path("model"),
            "scenario_root": Path("scenarios"),
        }
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, branches=(BASELINE_BRANCH,))
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, branches=(BASELINE_BRANCH, "unknown"))
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, policy_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, scenario_seeds=(1, 1))
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, device="cuda")
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, deterministic=False)
        with self.assertRaises(ValueError):
            SafetyMechanismAuditConfig(**kwargs, safety_projection_enabled=False)

    def test_scene_clearance_attributes_obstacle_and_uav_sources(self) -> None:
        positions = {
            "uav_00": np.array([0.0, 0.0, 5.0]),
            "uav_01": np.array([1.0, 0.0, 5.0]),
        }
        detail = _scene_clearance_detail(positions, [], 2.0)
        self.assertEqual(detail["source_type"], "uav_uav")
        self.assertEqual(detail["collision"], 1)

        obstacle = BoxObstacle(
            "building_00",
            np.array([0.0, 0.0, 5.0]),
            np.array([2.0, 2.0, 2.0]),
        )
        detail = _scene_clearance_detail(
            {"uav_00": np.array([0.0, 0.0, 5.0])}, [obstacle], 2.0
        )
        self.assertEqual(detail["source_type"], "uav_obstacle")
        self.assertEqual(detail["source_id"], "building_00")

    def test_motion_risk_uses_three_dimensional_geometry_and_boundary(self) -> None:
        obstacle = SphereObstacle(
            "vehicle_00",
            np.array([2.0, 0.0, 5.0]),
            radius_m=0.5,
        )
        risk = _motion_risk(
            "uav_00",
            np.array([0.0, 0.0, 5.0]),
            np.array([4.0, 0.0, 5.0]),
            {},
            {"uav_00": np.array([0.0, 0.0, 5.0]), "uav_01": np.array([20.0, 0.0, 5.0])},
            [obstacle],
            2.0,
        )
        self.assertEqual(risk["reason"], "obstacle_segment")

        boundary_risk = _motion_risk(
            "uav_00",
            np.array([0.0, 0.0, 5.0]),
            np.array([0.0, 0.0, 5.0]),
            {"boundary_projected": True},
            {"uav_00": np.array([0.0, 0.0, 5.0])},
            [],
            2.0,
        )
        self.assertEqual(boundary_risk["reason"], "boundary")

    @unittest.skipUnless(HAS_FROZEN_BUNDLE, "frozen safety bundle is not available")
    def test_single_pair_audit_is_read_only_and_writes_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safety_mechanism_audit_") as temporary:
            root = Path(temporary)
            result = run_safety_mechanism_audit(
                SafetyMechanismAuditConfig(
                    model_root=MODEL_ROOT,
                    scenario_root=SCENARIO_ROOT,
                    output_dir=root / "logs",
                    plots_dir=root / "plots",
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
            report_path = root / "logs" / "audit_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["checks"]["actor_observations_truth_free"])
            self.assertTrue(report["checks"]["trainer_update_called_zero"])
            self.assertTrue(report["checks"]["test_data_read_zero"])
            self.assertTrue((root / "logs" / "collision_events.csv").exists())
            self.assertTrue((root / "logs" / "step_diagnostics.csv").exists())
            self.assertTrue((root / "plots" / "safety_mechanism_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
