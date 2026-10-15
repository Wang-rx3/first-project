"""Pure offline tests for the signed bypass observability preparation.

These tests never instantiate an environment, load a checkpoint, call a
controller, or advance a simulator.  They exercise the side-channel math and
the static CLI/identity contracts only.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from src.experiments.single_uav_signed_bypass_observability import (
    LATERAL_AXIS_SOURCE,
    LATERAL_SIGN_CONVENTION,
    NOT_INDEPENDENTLY_VERIFIED,
    SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION,
    SignedLateralAxisNotWellDefined,
    build_horizontal_goal_frame,
    classify_signed_failure,
    compare_identity_component,
    compare_identity_components,
    decompose_signed_vector,
    frame_orthogonality,
    passed_geometry_fields,
    pipeline_signed_components,
    same_side_run_length,
    side_agreement,
    sign_change_count,
    signed_side,
    trajectory_fingerprint,
    update_side_run,
)


PROJECT = Path(__file__).resolve().parents[1]
EVAL_SOURCE = PROJECT / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"


class SignedBypassObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = build_horizontal_goal_frame([32.0, 0.0, 0.0])

    def test_schema_version_is_stable(self) -> None:
        self.assertEqual(SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION, "signed_bypass_observability_v1")

    def test_axis_source_is_goal_projected(self) -> None:
        self.assertEqual(LATERAL_AXIS_SOURCE, "goal_projected_xy_left")

    def test_sign_convention_is_explicit(self) -> None:
        self.assertIn("+1=goal-frame-left", LATERAL_SIGN_CONVENTION)

    def test_frame_radial_axis(self) -> None:
        np.testing.assert_allclose(self.frame.radial_unit, [1.0, 0.0, 0.0])

    def test_frame_lateral_axis(self) -> None:
        np.testing.assert_allclose(self.frame.lateral_unit, [0.0, 1.0, 0.0])

    def test_frame_vertical_axis(self) -> None:
        np.testing.assert_allclose(self.frame.vertical_unit, [0.0, 0.0, 1.0])

    def test_frame_orthogonality(self) -> None:
        checks = frame_orthogonality(self.frame)
        self.assertLess(max(abs(value) for value in checks.values()), 1.0e-12)

    def test_decomposition_reconstructs_vector(self) -> None:
        vector = np.asarray([2.0, -3.0, 4.0])
        parts = decompose_signed_vector(vector, self.frame)
        self.assertAlmostEqual(parts["reconstruction_residual"], 0.0)
        self.assertEqual(parts["radial"], 2.0)
        self.assertEqual(parts["lateral"], -3.0)
        self.assertEqual(parts["vertical"], 4.0)

    def test_degenerate_axis_fails_closed(self) -> None:
        with self.assertRaises(SignedLateralAxisNotWellDefined) as caught:
            build_horizontal_goal_frame([0.0, 0.0, 1.0])
        self.assertEqual(caught.exception.code, "SIGNED_LATERAL_AXIS_NOT_WELL_DEFINED")

    def test_nonhorizontal_goal_ray_fails_closed(self) -> None:
        with self.assertRaises(SignedLateralAxisNotWellDefined):
            build_horizontal_goal_frame([1.0, 0.0, 0.1])

    def test_left_right_center_labels(self) -> None:
        self.assertEqual(signed_side(2.0), "LEFT")
        self.assertEqual(signed_side(-2.0), "RIGHT")
        self.assertEqual(signed_side(0.0), "CENTER")

    def test_obstacle_signed_components(self) -> None:
        parts = decompose_signed_vector([4.0, 5.0, 1.0], self.frame)
        self.assertEqual(parts["radial"], 4.0)
        self.assertEqual(parts["lateral"], 5.0)

    def test_actor_signed_components(self) -> None:
        parts = decompose_signed_vector([-1.0, 0.5, -0.25], self.frame)
        self.assertEqual(parts["radial"], -1.0)
        self.assertEqual(parts["lateral"], 0.5)
        self.assertEqual(parts["vertical"], -0.25)

    def test_pipeline_contains_all_stages(self) -> None:
        result = pipeline_signed_components(
            {"pre_transform": [1.0, 2.0, 3.0], "final_applied": [0.0, 1.0, 0.0]},
            self.frame,
        )
        self.assertIn("pre_transform_lateral", result)
        self.assertIn("final_applied_radial", result)

    def test_passed_threshold_uses_extent_and_buffer(self) -> None:
        fields = passed_geometry_fields(-6.0, 4.5, 3.0, buffer_m=0.75, safe_distance_m=2.0)
        self.assertAlmostEqual(fields["passed_threshold_m"], 5.25)
        self.assertAlmostEqual(fields["passed_margin_m"], 0.75)

    def test_passed_condition_requires_clearance(self) -> None:
        fields = passed_geometry_fields(-6.0, 4.5, 2.1, buffer_m=0.75, safe_distance_m=2.0)
        self.assertFalse(fields["passed_condition_now"])
        self.assertTrue(fields["soft_recovery_condition_now"])

    def test_passed_condition_is_true_when_both_gates_hold(self) -> None:
        fields = passed_geometry_fields(-6.0, 4.5, 2.5, buffer_m=0.75, safe_distance_m=2.0)
        self.assertTrue(fields["passed_condition_now"])

    def test_sign_change_count_ignores_center(self) -> None:
        self.assertEqual(sign_change_count([1.0, 0.0, -1.0, -2.0, 2.0]), 2)

    def test_same_side_run_length(self) -> None:
        self.assertEqual(same_side_run_length([1.0, 2.0, -1.0, 3.0], reference_sign=1), 2)

    def test_update_side_run(self) -> None:
        self.assertEqual(update_side_run(2, 0.4, 1), 3)
        self.assertEqual(update_side_run(3, -0.4, 1), 0)

    def test_side_agreement(self) -> None:
        self.assertAlmostEqual(side_agreement([1.0, -1.0, 0.0], [2.0, -2.0, 3.0]), 1.0)

    def test_failure_wrong_side_label(self) -> None:
        label = classify_signed_failure(
            assessable=True, initial_obstacle_side="LEFT", recommended_side="RIGHT",
            passed=False, lateral_sign_changes=0, actor_motion_agreement=1.0,
            radial_progress_mean=1.0, lateral_valid=True,
        )
        self.assertEqual(label, "WRONG_INITIAL_BYPASS_SIDE")

    def test_failure_oscillation_precedes_radial(self) -> None:
        label = classify_signed_failure(
            assessable=True, initial_obstacle_side="LEFT", recommended_side="AMBIGUOUS",
            passed=False, lateral_sign_changes=2, actor_motion_agreement=1.0,
            radial_progress_mean=-1.0, lateral_valid=True,
        )
        self.assertEqual(label, "LATERAL_SIGN_OSCILLATION")

    def test_failure_actor_motion_mismatch(self) -> None:
        label = classify_signed_failure(
            assessable=True, initial_obstacle_side="LEFT", recommended_side="AMBIGUOUS",
            passed=False, lateral_sign_changes=0, actor_motion_agreement=0.2,
            radial_progress_mean=1.0, lateral_valid=True,
        )
        self.assertEqual(label, "ACTOR_TO_MOTION_DIRECTION_MISMATCH")

    def test_failure_radial_collapse(self) -> None:
        label = classify_signed_failure(
            assessable=True, initial_obstacle_side="LEFT", recommended_side="AMBIGUOUS",
            passed=False, lateral_sign_changes=0, actor_motion_agreement=1.0,
            radial_progress_mean=-0.2, lateral_valid=True,
        )
        self.assertEqual(label, "RADIAL_COLLAPSE_WITH_VALID_LATERAL")

    def test_failure_no_exit_label(self) -> None:
        label = classify_signed_failure(
            assessable=True, initial_obstacle_side="LEFT", recommended_side="AMBIGUOUS",
            passed=False, lateral_sign_changes=0, actor_motion_agreement=1.0,
            radial_progress_mean=0.2, lateral_valid=True,
        )
        self.assertEqual(label, "CORRECT_INITIAL_SIDE_BUT_NO_EXIT")

    def test_failure_not_assessable(self) -> None:
        label = classify_signed_failure(
            assessable=False, initial_obstacle_side="AMBIGUOUS", recommended_side="AMBIGUOUS",
            passed=False, lateral_sign_changes=0, actor_motion_agreement=float("nan"),
            radial_progress_mean=float("nan"), lateral_valid=False,
        )
        self.assertEqual(label, "NOT_ASSESSABLE")

    def test_fingerprint_is_deterministic(self) -> None:
        rows = [{"episode_id": "e", "step": 0, "value": 1.0}]
        self.assertEqual(trajectory_fingerprint(rows, keys=["episode_id", "step"]), trajectory_fingerprint(rows, keys=["episode_id", "step"]))

    def test_identity_component_uses_explicit_independent_field(self) -> None:
        current = {"identity_fields": {"reward_identity_hash": "reward-v1"}}
        baseline = {"identity_fields": {"reward_identity_hash": "reward-v1"}}
        result = compare_identity_component(current, baseline, component="reward_identity")
        self.assertIs(result["status"], True)
        self.assertEqual(result["current_value"], "reward-v1")
        self.assertEqual(result["baseline_value"], "reward-v1")

    def test_aggregate_check_is_not_independent_identity(self) -> None:
        current = {"previous_formal_identity_checks": {"same_reward": True}}
        baseline = {"previous_formal_identity_checks": {"same_reward": True}}
        result = compare_identity_component(current, baseline, component="reward_identity")
        self.assertEqual(result["status"], NOT_INDEPENDENTLY_VERIFIED)
        self.assertEqual(result["supporting_current_check"], True)

    def test_identity_component_preserves_false_check(self) -> None:
        current = {"identity_fields": {"dynamics_identity": False}}
        baseline = {"identity_fields": {"dynamics_identity": True}}
        result = compare_identity_component(current, baseline, component="dynamics_identity")
        self.assertIs(result["status"], False)

    def test_identity_component_missing_field_is_not_verified(self) -> None:
        result = compare_identity_component(
            {"previous_formal_identity_checks": {}},
            {},
            component="action_transform_identity",
        )
        self.assertEqual(result["status"], NOT_INDEPENDENTLY_VERIFIED)
        self.assertEqual(result["reason"], "independent_component_identity_field_unavailable")

    def test_all_component_identity_fields_are_reported(self) -> None:
        result = compare_identity_components({}, {})
        self.assertEqual(
            set(result),
            {
                "reward_identity",
                "dynamics_identity",
                "action_transform_identity",
                "safety_projection_identity",
                "boundary_projection_identity",
            },
        )
        self.assertTrue(
            all(item["status"] == NOT_INDEPENDENTLY_VERIFIED for item in result.values())
        )

    def test_cli_flag_is_declared(self) -> None:
        source = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("--single-uav-signed-bypass-observability-diagnostic", source)

    def test_logging_flag_is_not_observation_input(self) -> None:
        tree = ast.parse(EVAL_SOURCE.read_text(encoding="utf-8"))
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("signed_observability=bool(signed_bypass_observability_diagnostic)", text)
        self.assertNotIn("signed_observability", text.split("_actor_decision", 1)[0])
        self.assertIsInstance(tree, ast.Module)

    def test_signed_fields_are_row_side_channel(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("row.update(signed_step)", text)
        self.assertIn("if signed_observability:", text)

    def test_logging_helper_does_not_import_rng(self) -> None:
        helper = (PROJECT / "src/experiments/single_uav_signed_bypass_observability.py").read_text(encoding="utf-8")
        self.assertNotIn("np.random", helper)
        self.assertNotIn("import random", helper)

    def test_logging_fields_are_not_action_payload(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        action_start = text.index("next_observations, rewards, terminated, truncated, next_info = environment.step")
        action_end = text.index("stable_fields = extract_canonical_stable_fields", action_start)
        self.assertNotIn("signed_", text[action_start:action_end])

    def test_logging_fields_are_not_reward_expression(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        reward_start = text.index("reward = float(rewards.get")
        reward_end = text.index("row = {", reward_start)
        self.assertNotIn("signed_", text[reward_start:reward_end])

    def test_horizon_is_locked_to_120(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("HORIZON_DIAGNOSTIC_MAX_STEPS = 120", text)

    def test_checkpoint_sha_is_part_of_identity_lock(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("same_checkpoint_sha256", text)
        self.assertIn('"sha256"', text)

    def test_component_identity_is_not_hardcoded(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn(
            'identity_lock["same_reward_dynamics_action_safety_boundary"] = True',
            text,
        )
        self.assertIn("compare_identity_components", text)
        self.assertIn("NOT_INDEPENDENTLY_VERIFIED", (PROJECT / "src/experiments/single_uav_signed_bypass_observability.py").read_text(encoding="utf-8"))

    def test_only_logging_changed_keeps_all_identity_dependencies(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        start = text.index('identity_lock["only_logging_changed"] = bool(', text.index("component_identity"))
        end = text.index("identity_lock[\"trajectory_fingerprint_behavior_protection\"]", start)
        expression = text[start:end]
        for required in (
            "episode_identity_match",
            "behavior_match",
            "trajectory_match",
            "same_frozen_controller_identity",
            "same_goal_recovery_activation",
        ):
            self.assertIn(required, expression)

    def test_trajectory_fingerprint_protection_and_fail_fast_are_declared(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("trajectory_fingerprint_behavior_protection", text)
        self.assertIn('raise RuntimeError("LOGGING_PERTURBED_TRAJECTORY")', text)

    def test_scenario_and_order_identity_are_locked(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("same_scenario_manifest", text)
        self.assertIn("same_ordered_identity", text)

    def test_goal_recovery_activation_mode_is_checked(self) -> None:
        formal = PROJECT / "src/experiments/formal.py"
        self.assertIn("post_bypass_only", formal.read_text(encoding="utf-8"))

    def test_required_output_names_are_declared(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        for name in (
            "signed_bypass_summary.json",
            "signed_bypass_episodes.csv",
            "signed_bypass_steps.csv",
            "signed_bypass_focus_failures.csv",
            "signed_bypass_direction_transitions.csv",
        ):
            self.assertIn(name, text)

    def test_old_cli_config_is_still_present(self) -> None:
        text = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("terminal_capture_v23_authority_repair_horizon_diagnostic", text)
        self.assertIn("terminal_capture_v23_authority_repair_horizon120_diagnostic", text)

    def test_goal_recovery_activation_contract_is_locked(self) -> None:
        formal = PROJECT / "src/experiments/formal.py"
        self.assertIn('goal_recovery_activation_mode: str = "post_bypass_only"', formal.read_text(encoding="utf-8"))

    def test_independent_output_root_is_declared(self) -> None:
        source = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("single_uav_signed_bypass_observability_diagnostic_seed2026", source)

    def test_old_horizon_cli_remains_declared(self) -> None:
        source = EVAL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("--terminal-capture-v23-authority-repair-horizon120-diagnostic", source)


if __name__ == "__main__":
    unittest.main()
