"""Pure tests for encounter-time EKF horizontal assist-frame construction."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import unittest

import numpy as np

from src.experiments.single_uav_goal_recovery_correction_evaluation import (
    _build_encounter_horizontal_assist_frame,
)
from src.experiments.single_uav_lateral_exit_braking_assistance import (
    FROZEN_OFF_BASELINE_FINGERPRINT,
    preregistration_hash,
)
from src.experiments.single_uav_signed_bypass_observability import (
    SignedLateralAxisNotWellDefined,
    build_horizontal_goal_frame,
)


PROJECT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"


class HorizontalAssistFrameProjectionTests(unittest.TestCase):
    def test_nonzero_estimated_goal_ray_z_is_projected_before_frame_build(self) -> None:
        frame, reason, horizontal = _build_encounter_horizontal_assist_frame(
            goal_position_m=[10.0, 0.0, 0.2],
            estimated_position_m=[0.0, 0.0, 0.0],
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(horizontal, [10.0, 0.0, 0.0])

    def test_projected_lateral_axis_is_horizontal_unit_vector(self) -> None:
        frame, _, _ = _build_encounter_horizontal_assist_frame(
            goal_position_m=[5.0, 12.0, 2.0],
            estimated_position_m=[0.0, 0.0, -0.5],
        )
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(float(np.linalg.norm(frame.lateral_unit)), 1.0)
        self.assertEqual(float(frame.lateral_unit[2]), 0.0)

    def test_projection_removes_only_z_and_preserves_xy_direction(self) -> None:
        frame, _, horizontal = _build_encounter_horizontal_assist_frame(
            goal_position_m=[7.0, -4.0, 3.0],
            estimated_position_m=[2.0, -1.0, 1.5],
        )
        np.testing.assert_array_equal(horizontal[:2], [5.0, -3.0])
        self.assertEqual(float(horizontal[2]), 0.0)
        expected = horizontal / np.linalg.norm(horizontal)
        np.testing.assert_allclose(frame.radial_unit, expected)

    def test_signed_helper_nonhorizontal_fail_closed_contract_is_preserved(self) -> None:
        with self.assertRaises(SignedLateralAxisNotWellDefined):
            build_horizontal_goal_frame([1.0, 0.0, 0.1])

    def test_degenerate_xy_does_not_create_fake_frame(self) -> None:
        frame, reason, horizontal = _build_encounter_horizontal_assist_frame(
            goal_position_m=[0.0, 0.0, 0.2],
            estimated_position_m=[0.0, 0.0, 0.0],
        )
        self.assertIsNone(frame)
        self.assertEqual(reason, "horizontal_goal_frame_degenerate")
        np.testing.assert_array_equal(horizontal, [0.0, 0.0, 0.0])

    def test_frame_builder_accepts_no_actual_state_input(self) -> None:
        parameters = inspect.signature(
            _build_encounter_horizontal_assist_frame
        ).parameters
        self.assertEqual(
            tuple(parameters), ("goal_position_m", "estimated_position_m")
        )
        source = inspect.getsource(_build_encounter_horizontal_assist_frame)
        self.assertNotIn("actual", source)

    def test_off_action_fallback_path_is_still_the_existing_fusion(self) -> None:
        source = EVALUATOR.read_text(encoding="utf-8")
        start = source.index(
            "if assist_decision is not None and assist_decision.active:"
        )
        end = source.index("_assert_action_feasible", start)
        action_branch = source[start:end]
        self.assertIn("elif correction_result is not None:", action_branch)
        self.assertIn("fusion = fuse_actor_action(", action_branch)
        self.assertIn("actor_raw,", action_branch)

    def test_assistance_helper_hash_is_unchanged(self) -> None:
        source = (PROJECT / "src/control/lateral_exit_braking_assistance.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "19a729bed2cd301f216f94f770c19e9802a8c38dcba2b6ca5e5bac914fc3c383",
        )

    def test_active_preregistration_hash_is_unchanged(self) -> None:
        self.assertEqual(
            preregistration_hash(),
            "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2",
        )

    def test_off_baseline_fingerprint_constant_is_unchanged(self) -> None:
        self.assertEqual(
            FROZEN_OFF_BASELINE_FINGERPRINT,
            "d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b",
        )


if __name__ == "__main__":
    unittest.main()
