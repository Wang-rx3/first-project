from __future__ import annotations

import math
import ast
from pathlib import Path
import unittest

from src.control.pass_commit_action_decomposition import (
    SingleCallActionTransform,
    compose_pre_transform_action,
    post_safety_execution_residual_mps,
    post_transform_decomposition_residual_mps,
    pre_transform_decomposition_residual_mps,
)
from src.control.pass_commit_non_regression import (
    PASS_COMMIT_HARD_CAP_MPS,
    PassCommitNonRegression,
    derive_pass_geometry,
)


GOAL = (10.0, 4.0, 0.0)
OBSTACLE = (5.0, 0.0, 0.0)
EXTENT_M = 1.0
BUFFER_M = 0.75
TOLERANCE = 1.0e-9


def _evaluate(
    controller: PassCommitNonRegression,
    *,
    step: int,
    x: float,
):
    position = (x, 4.0, 0.0)
    geometry = derive_pass_geometry(
        estimated_position_m=position,
        goal_position_m=GOAL,
        obstacle_center_m=OBSTACLE,
        obstacle_extent_m=EXTENT_M,
        pass_buffer_m=BUFFER_M,
    )
    return controller.evaluate(
        step=step,
        estimated_position_m=position,
        goal_position_m=GOAL,
        obstacle_center_m=OBSTACLE,
        obstacle_extent_m=EXTENT_M,
        pass_buffer_m=BUFFER_M,
        logged_along_obstacle_m=geometry.along_obstacle_m,
        logged_passed_margin_m=geometry.passed_margin_m,
        estimated_velocity_mps=(0.0, 0.0, 0.0),
        baseline_candidate_delta_v_mps=(-0.2, 0.0, 0.0),
        estimated_lateral_velocity_mps=0.1,
        encounter_context_established=True,
        frozen_obstacle_matches=True,
        relevant_obstacle_available=True,
        relevant_obstacle_clearance_m=4.0,
        passed_obstacle=False,
        passed_condition_now=False,
        release_latched=False,
        post_bypass_recovery=False,
        goal_recovery_active=False,
        lateral_exit_assist_active=False,
        past_closest_approach=True,
        clearance_increasing=True,
        episode_done=False,
    )


def _active_decision(enabled: bool = True):
    controller = PassCommitNonRegression(
        enabled=enabled,
        cap_mps=PASS_COMMIT_HARD_CAP_MPS,
    )
    decisions = [
        _evaluate(controller, step=index, x=x)
        for index, x in enumerate((4.0, 3.8, 3.6, 3.4, 3.2))
    ]
    return controller, decisions[-1]


class PassCommitActionDecompositionTests(unittest.TestCase):
    def test_helper_is_pure_and_has_no_environment_rollout_or_training_call(self):
        source_path = Path(__file__).resolve().parents[1] / "src/control/pass_commit_action_decomposition.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = ("environment", "simulation", "rollout", "training", "formal")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.lower() for alias in node.names]
                self.assertFalse(any(token in name for token in forbidden for name in names))
            elif isinstance(node, ast.ImportFrom):
                name = str(node.module or "").lower()
                self.assertFalse(any(token in name for token in forbidden))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr.lower(), {"step", "backward", "rollout"})

    def test_evaluator_reconstruction_explicitly_passes_applied_vector(self):
        evaluator = Path(__file__).resolve().parents[1] / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
        tree = ast.parse(evaluator.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pre_transform_decomposition_residual_mps"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg for item in calls[0].keywords}
        self.assertIn("pass_commit_applied_mps", keywords)
        self.assertNotIn("pass_commit_requested_vector_mps", keywords)

    def test_old_missing_component_formula_reproduces_structural_failure(self):
        actor = (-0.2, 0.0, 0.0)
        requested = (0.2, 0.0, 0.0)
        applied = (0.075, 0.0, 0.0)
        combined = tuple(actor[i] + applied[i] for i in range(3))

        legacy_residual = math.sqrt(sum((combined[i] - actor[i]) ** 2 for i in range(3)))
        fixed_residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=combined,
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=applied,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )

        self.assertGreater(legacy_residual, TOLERANCE)
        self.assertAlmostEqual(legacy_residual, 0.075)
        self.assertEqual(fixed_residual, 0.0)
        self.assertNotEqual(requested, applied)

    def test_pass_commit_off_residual_and_transform_once(self):
        _, decision = _active_decision(enabled=False)
        self.assertGreater(decision.requested_delta_v_mps, decision.applied_delta_v_mps)
        self.assertEqual(decision.applied_delta_v_mps, 0.0)
        actor = (-0.2, 0.0, 0.0)
        lateral = (0.0, 0.04, 0.0)
        combined = compose_pre_transform_action(
            actor_base_candidate_mps=actor,
            lateral_exit_applied_mps=lateral,
            pass_commit_enabled=False,
            pass_commit_active=False,
            lateral_exit_active=True,
        )
        residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=combined,
            actor_base_candidate_mps=actor,
            lateral_exit_applied_mps=lateral,
            pass_commit_applied_mps=decision.applied_vector_mps,
            pass_commit_enabled=False,
            pass_commit_active=False,
            lateral_exit_active=True,
        )
        calls = []
        transform = SingleCallActionTransform(lambda value, limit: calls.append((value, limit)) or value, 0.6)
        transform(combined)
        transform.assert_call_count(1)
        self.assertEqual(residual, 0.0)
        self.assertEqual(len(calls), 1)

    def test_pass_commit_on_inactive_residual_and_transform_once(self):
        controller = PassCommitNonRegression(enabled=True, cap_mps=0.075)
        decision = _evaluate(controller, step=0, x=4.0)
        self.assertFalse(decision.active)
        actor = (-0.1, 0.0, 0.0)
        lateral = (0.0, -0.03, 0.0)
        combined = compose_pre_transform_action(
            actor_base_candidate_mps=actor,
            lateral_exit_applied_mps=lateral,
            pass_commit_enabled=True,
            pass_commit_active=False,
            lateral_exit_active=True,
        )
        residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=combined,
            actor_base_candidate_mps=actor,
            lateral_exit_applied_mps=lateral,
            pass_commit_applied_mps=decision.applied_vector_mps,
            pass_commit_enabled=True,
            pass_commit_active=False,
            lateral_exit_active=True,
        )
        calls = []
        transform = SingleCallActionTransform(lambda value, limit: calls.append(value) or value, 0.6)
        transform(combined)
        transform.assert_call_count(1)
        self.assertEqual(residual, 0.0)
        self.assertEqual(len(calls), 1)

    def test_pass_commit_on_active_applied_component_is_counted_once(self):
        _, decision = _active_decision()
        self.assertTrue(decision.active)
        self.assertGreater(decision.requested_delta_v_mps, decision.applied_delta_v_mps)
        self.assertEqual(decision.applied_delta_v_mps, 0.075)
        actor = (-0.2, 0.0, 0.0)
        combined = compose_pre_transform_action(
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=decision.applied_vector_mps,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )
        residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=combined,
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=decision.applied_vector_mps,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )
        calls = []
        transform = SingleCallActionTransform(
            lambda value, limit: calls.append((value, limit))
            or {"constrained_delta_v_mps": value},
            0.6,
        )
        result = transform(combined)
        transform.assert_call_count(1)
        self.assertEqual(residual, 0.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["constrained_delta_v_mps"], combined)
        self.assertEqual(combined, (-0.125, 0.0, 0.0))

    def test_requested_vector_cannot_replace_capped_applied_vector(self):
        _, decision = _active_decision()
        actor = (-0.2, 0.0, 0.0)
        combined = tuple(actor[i] + decision.applied_vector_mps[i] for i in range(3))
        residual = pre_transform_decomposition_residual_mps(
            combined_pre_transform_action_mps=combined,
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=decision.applied_vector_mps,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )
        misusing_requested_vector_residual = math.sqrt(
            sum(
                (combined[i] - actor[i] - decision.requested_vector_mps[i]) ** 2
                for i in range(3)
            )
        )
        self.assertEqual(residual, 0.0)
        self.assertAlmostEqual(misusing_requested_vector_residual, 0.125)

    def test_pass_commit_lateral_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            compose_pre_transform_action(
                actor_base_candidate_mps=(0.0, 0.0, 0.0),
                lateral_exit_applied_mps=(0.0, 0.1, 0.0),
                pass_commit_applied_mps=(0.05, 0.0, 0.0),
                pass_commit_enabled=True,
                pass_commit_active=True,
                lateral_exit_active=True,
            )

    def test_pass_commit_goal_recovery_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            compose_pre_transform_action(
                actor_base_candidate_mps=(0.0, 0.0, 0.0),
                goal_recovery_applied_mps=(0.0, 0.1, 0.0),
                pass_commit_applied_mps=(0.05, 0.0, 0.0),
                pass_commit_enabled=True,
                pass_commit_active=True,
                goal_recovery_active=True,
            )

    def test_pass_commit_radial_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            compose_pre_transform_action(
                actor_base_candidate_mps=(0.0, 0.0, 0.0),
                pass_commit_applied_mps=(0.05, 0.0, 0.0),
                radial_intent_applied_mps=(0.0, 0.05, 0.0),
                pass_commit_enabled=True,
                pass_commit_active=True,
                radial_intent_active=True,
            )

    def test_action_transform_proxy_fails_on_second_call(self):
        transform = SingleCallActionTransform(lambda value, limit: value, 0.6)
        transform((0.1, 0.0, 0.0))
        with self.assertRaisesRegex(AssertionError, "more than once"):
            transform((0.1, 0.0, 0.0))

    def test_post_transform_reconstruction_is_consistent(self):
        actor = (-0.2, 0.0, 0.0)
        applied = (0.075, 0.0, 0.0)
        combined = compose_pre_transform_action(
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=applied,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )
        scale = lambda value, limit: tuple(max(-limit, min(limit, x * 0.5)) for x in value)
        actual = SingleCallActionTransform(scale, 0.6)(combined)
        reconstructed_input = compose_pre_transform_action(
            actor_base_candidate_mps=actor,
            pass_commit_applied_mps=applied,
            pass_commit_enabled=True,
            pass_commit_active=True,
        )
        expected = scale(reconstructed_input, 0.6)
        residual = post_transform_decomposition_residual_mps(
            action_transform_input_mps=combined,
            action_transform_output_mps=actual,
            transformed_reconstruction_mps=expected,
        )
        self.assertEqual(residual, 0.0)

    def test_existing_post_safety_projection_identity_is_preserved(self):
        residual = post_safety_execution_residual_mps(
            pre_safety_action_mps=(0.1, 0.0, 0.0),
            projection_delta_mps=(-0.02, 0.01, 0.0),
            final_applied_action_mps=(0.08, 0.01, 0.0),
        )
        self.assertAlmostEqual(residual, 0.0)


if __name__ == "__main__":
    unittest.main()
