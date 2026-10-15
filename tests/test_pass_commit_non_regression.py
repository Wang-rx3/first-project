from __future__ import annotations

import ast
import csv
import inspect
import math
from pathlib import Path
import unittest

from src.control.pass_commit_non_regression import (
    PASS_COMMIT_CLEARANCE_FLOOR_M,
    PASS_COMMIT_HARD_CAP_MPS,
    PASS_COMMIT_TRIGGER_DEADBAND_M,
    PassCommitNonRegression,
    derive_pass_geometry,
    pass_margin_direction_proof,
)


PROJECT = Path(__file__).resolve().parents[1]
GOAL = (10.0, 4.0, 0.0)
OBSTACLE = (5.0, 0.0, 0.0)
EXTENT = 1.0
BUFFER = 0.75


def _eval(
    controller: PassCommitNonRegression,
    step: int,
    x: float,
    *,
    enabled_context: bool = True,
    matches: bool = True,
    available: bool = True,
    clearance: float = 4.0,
    passed: bool = False,
    passed_now: bool = False,
    release: bool = False,
    post: bool = False,
    goal_recovery: bool = False,
    lateral_active: bool = False,
    past_closest: bool = True,
    increasing: bool = True,
    done: bool = False,
    actor: tuple[float, float, float] = (-0.1, 0.0, 0.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lateral_velocity: float = 0.1,
):
    position = (x, 4.0, 0.0)
    geometry = derive_pass_geometry(
        estimated_position_m=position,
        goal_position_m=GOAL,
        obstacle_center_m=OBSTACLE,
        obstacle_extent_m=EXTENT,
        pass_buffer_m=BUFFER,
    )
    return controller.evaluate(
        step=step,
        estimated_position_m=position,
        goal_position_m=GOAL,
        obstacle_center_m=OBSTACLE,
        obstacle_extent_m=EXTENT,
        pass_buffer_m=BUFFER,
        logged_along_obstacle_m=geometry.along_obstacle_m,
        logged_passed_margin_m=geometry.passed_margin_m,
        estimated_velocity_mps=velocity,
        baseline_candidate_delta_v_mps=actor,
        estimated_lateral_velocity_mps=lateral_velocity,
        encounter_context_established=enabled_context,
        frozen_obstacle_matches=matches,
        relevant_obstacle_available=available,
        relevant_obstacle_clearance_m=clearance,
        passed_obstacle=passed,
        passed_condition_now=passed_now,
        release_latched=release,
        post_bypass_recovery=post,
        goal_recovery_active=goal_recovery,
        lateral_exit_assist_active=lateral_active,
        past_closest_approach=past_closest,
        clearance_increasing=increasing,
        episode_done=done,
    )


def _latched_controller(*, enabled: bool = True, cap: float = PASS_COMMIT_HARD_CAP_MPS):
    controller = PassCommitNonRegression(enabled=enabled, cap_mps=cap)
    # First ready row is step 2; then two consecutive negative margin deltas.
    for step, x in enumerate((4.0, 3.8, 3.6, 3.4, 3.2)):
        decision = _eval(controller, step, x)
    return controller, decision


class PassCommitNonRegressionTests(unittest.TestCase):
    def test_prepass_only_can_become_active(self):
        controller, decision = _latched_controller()
        self.assertTrue(decision.active)
        self.assertFalse(decision.permanently_ended)

    def test_exact_pass_condition_immediately_permanently_closes(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, passed_now=True)
        self.assertFalse(decision.active)
        self.assertTrue(decision.permanently_ended)

    def test_release_latch_permanently_closes(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, release=True)
        self.assertTrue(decision.permanently_ended)
        self.assertFalse(decision.latched)

    def test_post_bypass_recovery_permanently_closes(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, post=True)
        self.assertTrue(decision.permanently_ended)

    def test_goal_recovery_overlap_permanently_closes(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, goal_recovery=True)
        self.assertTrue(decision.permanently_ended)
        self.assertEqual(decision.inactive_reason, "goal_recovery_active")

    def test_radial_assistance_is_absent_from_runtime_api(self):
        names = set(inspect.signature(PassCommitNonRegression.evaluate).parameters)
        self.assertFalse(any("radial_intent" in name for name in names))

    def test_lateral_exit_requires_three_contiguous_estimated_rows(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        ready = []
        for step, lateral in enumerate((0.4, 0.4, 0.4)):
            ready.append(_eval(controller, step, 4.0 - 0.1 * step, lateral_velocity=lateral).estimated_lateral_exit_ready)
        self.assertEqual(ready, [False, False, True])

    def test_nonready_lateral_velocity_breaks_contiguous_run(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        values = (0.4, 0.4, 0.6, 0.4, 0.4, 0.4)
        decisions = [
            _eval(controller, step, 4.0 - 0.05 * step, lateral_velocity=value)
            for step, value in enumerate(values)
        ]
        self.assertEqual([d.estimated_lateral_exit_ready for d in decisions], [False, False, False, False, False, True])

    def test_two_consecutive_regressions_trigger_and_latch(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        decisions = [_eval(controller, step, x) for step, x in enumerate((4.0, 3.8, 3.6, 3.4, 3.2))]
        self.assertFalse(decisions[3].triggered)
        self.assertTrue(decisions[4].triggered)
        self.assertTrue(decisions[4].latched)

    def test_single_regression_does_not_trigger(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        decisions = [_eval(controller, step, x) for step, x in enumerate((4.0, 3.8, 3.6, 3.4))]
        self.assertFalse(decisions[-1].latched)
        self.assertEqual(decisions[-1].regression_run, 1)

    def test_regression_deadband_is_fixed(self):
        self.assertEqual(PASS_COMMIT_TRIGGER_DEADBAND_M, 1.0e-8)
        self.assertEqual(PASS_COMMIT_CLEARANCE_FLOOR_M, 2.5)

    def test_running_best_uses_only_current_and_past_margins(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        decisions = [_eval(controller, step, x) for step, x in enumerate((4.0, 3.8, 3.6, 3.4, 3.2))]
        self.assertAlmostEqual(decisions[2].running_best_margin_m, decisions[2].passed_margin_m)
        self.assertAlmostEqual(decisions[4].running_best_margin_m, decisions[2].passed_margin_m)

    def test_current_goal_direction_increases_exact_pass_margin(self):
        before, after = pass_margin_direction_proof(
            estimated_position_m=(3.0, 4.0, 0.0),
            goal_position_m=GOAL,
            obstacle_center_m=OBSTACLE,
            obstacle_extent_m=EXTENT,
            pass_buffer_m=BUFFER,
        )
        self.assertGreater(after, before)
        self.assertAlmostEqual(after - before, 1.0e-5, places=9)

    def test_direction_derivation_rejects_degenerate_goal_geometry(self):
        with self.assertRaises(ValueError):
            derive_pass_geometry(
                estimated_position_m=(1, 2, 3), goal_position_m=(1, 2, 3),
                obstacle_center_m=(4, 5, 6), obstacle_extent_m=1, pass_buffer_m=0.75,
            )

    def test_nonregressing_candidate_requests_zero(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, actor=(0.2, 0.0, 0.0))
        self.assertFalse(decision.active)
        self.assertEqual(decision.requested_delta_v_mps, 0.0)

    def test_regressing_candidate_is_corrected_to_zero_when_cap_allows(self):
        controller, _ = _latched_controller(cap=PASS_COMMIT_HARD_CAP_MPS)
        decision = _eval(controller, 5, 3.0, actor=(-0.05, 0.0, 0.0))
        self.assertTrue(decision.active)
        self.assertAlmostEqual(decision.pass_progress_velocity_after_mps, 0.0)
        self.assertAlmostEqual(decision.applied_delta_v_mps, decision.requested_delta_v_mps)

    def test_hard_cap_leaves_negative_residual_without_retargeting(self):
        controller, _ = _latched_controller(cap=0.075)
        decision = _eval(controller, 5, 3.0, actor=(-0.2, 0.0, 0.0))
        self.assertEqual(PASS_COMMIT_HARD_CAP_MPS, 0.075)
        self.assertAlmostEqual(decision.applied_delta_v_mps, 0.075)
        self.assertLess(decision.pass_progress_velocity_after_mps, 0.0)

    def test_clearance_below_floor_permanently_fails_closed(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, clearance=2.49)
        self.assertTrue(decision.permanently_ended)
        self.assertFalse(decision.active)

    def test_obstacle_mismatch_permanently_fails_closed(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, matches=False)
        self.assertTrue(decision.permanently_ended)
        self.assertFalse(decision.active)

    def test_obstacle_unavailable_permanently_fails_closed(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, available=False)
        self.assertTrue(decision.permanently_ended)

    def test_logged_geometry_mismatch_permanently_fails_closed(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        good = derive_pass_geometry(
            estimated_position_m=(4, 4, 0), goal_position_m=GOAL,
            obstacle_center_m=OBSTACLE, obstacle_extent_m=EXTENT, pass_buffer_m=BUFFER,
        )
        decision = controller.evaluate(
            step=0, estimated_position_m=(4, 4, 0), goal_position_m=GOAL,
            obstacle_center_m=OBSTACLE, obstacle_extent_m=EXTENT, pass_buffer_m=BUFFER,
            logged_along_obstacle_m=good.along_obstacle_m,
            logged_passed_margin_m=good.passed_margin_m + 0.1,
            estimated_velocity_mps=(0, 0, 0), baseline_candidate_delta_v_mps=(-0.1, 0, 0),
            estimated_lateral_velocity_mps=0.1, encounter_context_established=True,
            frozen_obstacle_matches=True, relevant_obstacle_available=True,
            relevant_obstacle_clearance_m=4.0, passed_obstacle=False,
            passed_condition_now=False, release_latched=False,
            post_bypass_recovery=False, goal_recovery_active=False,
            lateral_exit_assist_active=False, past_closest_approach=True,
            clearance_increasing=True, episode_done=False,
        )
        self.assertTrue(decision.permanently_ended)
        self.assertEqual(decision.inactive_reason, "pass_geometry_fail_closed")

    def test_lateral_assistance_is_temporary_mutual_exclusion(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=PASS_COMMIT_HARD_CAP_MPS)
        for step, x in enumerate((4.0, 3.8, 3.6, 3.4, 3.2)):
            decision = _eval(controller, step, x, lateral_active=(step == 4))
        self.assertFalse(decision.active)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.permanently_ended)
        decision = _eval(controller, 5, 3.0, lateral_active=False)
        self.assertTrue(decision.active)

    def test_current_clearance_and_past_closest_gates_are_required(self):
        controller, _ = _latched_controller()
        no_increase = _eval(controller, 5, 3.0, increasing=False)
        self.assertFalse(no_increase.active)
        controller, _ = _latched_controller()
        not_past = _eval(controller, 5, 3.0, past_closest=False)
        self.assertFalse(not_past.active)

    def test_clearance_floor_equality_remains_eligible(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, clearance=2.5)
        self.assertFalse(decision.permanently_ended)
        self.assertTrue(decision.active)
        self.assertEqual(PASS_COMMIT_CLEARANCE_FLOOR_M, 2.5)

    def test_disabled_off_arm_never_applies_correction(self):
        controller, decision = _latched_controller(enabled=False)
        self.assertTrue(decision.eligible)
        self.assertFalse(decision.active)
        self.assertGreater(decision.requested_delta_v_mps, 0.0)
        self.assertEqual(decision.applied_delta_v_mps, 0.0)
        self.assertEqual(decision.applied_vector_mps, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            decision.pass_progress_velocity_after_mps,
            decision.pass_progress_velocity_before_mps,
        )
        self.assertEqual(decision.inactive_reason, "disabled_off_condition")

    def test_episode_end_permanently_closes(self):
        controller, _ = _latched_controller()
        decision = _eval(controller, 5, 3.0, done=True)
        self.assertTrue(decision.permanently_ended)

    def test_runtime_signature_does_not_accept_actual_state(self):
        names = set(inspect.signature(PassCommitNonRegression.evaluate).parameters)
        self.assertFalse(any("actual" in name or "simulator" in name for name in names))

    def test_pass_commit_row_additions_use_reserved_prefix(self):
        evaluator_path = PROJECT / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
        baseline_path = PROJECT / "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026/on/steps.csv"
        with baseline_path.open("r", encoding="utf-8", newline="") as stream:
            baseline_fields = set(next(csv.reader(stream)))
        module = ast.parse(evaluator_path.read_text(encoding="utf-8"))
        target = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "pass_commit_decision is not None"
        )
        update = next(
            node for node in ast.walk(target)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "row"
            and node.func.attr == "update"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        )
        added_fields = {
            key.value for key in update.args[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        unprefixed_new_fields = {
            field for field in added_fields
            if field not in baseline_fields and not field.startswith("pass_commit_")
        }
        self.assertEqual(unprefixed_new_fields, set())

    def test_action_transform_branch_calls_transform_once(self):
        path = PROJECT / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        target = None
        for node in ast.walk(module):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "pass_commit_decision is not None and pass_commit_decision.active":
                target = node
                break
        self.assertIsNotNone(target)
        calls = [
            node for statement in target.body for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_action_transform_once"
        ]
        self.assertEqual(len(calls), 1)
        evaluator = path.read_text(encoding="utf-8")
        self.assertIn("SingleCallActionTransform(actor_action_transform, action_limit_mps)", evaluator)
        self.assertIn("pass_commit_action_transform.assert_call_count(int(transform_applied))", evaluator)


if __name__ == "__main__":
    unittest.main()
