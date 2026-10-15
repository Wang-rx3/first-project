"""Pure synthetic tests for the frozen pre-pass handoff audit."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/experiments/single_uav_prepass_handoff_pass_progress_audit.py"
SPEC = importlib.util.spec_from_file_location("handoff_audit_under_test", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(AUDIT)


class HandoffAuditFormulaTests(unittest.TestCase):
    def test_finite_number(self):
        self.assertEqual(AUDIT.number("1.25"), 1.25)

    def test_nonfinite_is_missing(self):
        self.assertIsNone(AUDIT.number("nan"))
        self.assertIsNone(AUDIT.number("inf"))

    def test_integer_accepts_integral_float(self):
        self.assertEqual(AUDIT.integer("3.0"), 3)

    def test_integer_rejects_fractional(self):
        self.assertIsNone(AUDIT.integer("3.2"))

    def test_truthy_contract(self):
        self.assertTrue(AUDIT.flag("True"))
        self.assertTrue(AUDIT.flag("1"))
        self.assertFalse(AUDIT.flag("0"))

    def test_vector_json(self):
        self.assertEqual(AUDIT._vec("[1, 2, 0]"), (1.0, 2.0, 0.0))

    def test_vector_degenerate_missing(self):
        self.assertIsNone(AUDIT._vec("[]"))
        self.assertIsNone(AUDIT._vec("[NaN, 1, 0]"))

    def test_exact_pass_gate_positive_margin_and_clearance(self):
        row = {"passed_obstacle": "1", "passed_margin_m": "0.01", "relevant_obstacle_clearance_m": "2.5"}
        self.assertTrue(AUDIT._pass_gate(row))

    def test_pass_gate_requires_latch(self):
        row = {"passed_obstacle": "0", "passed_margin_m": "0.01", "relevant_obstacle_clearance_m": "3"}
        self.assertFalse(AUDIT._pass_gate(row))

    def test_pass_gate_strict_margin(self):
        row = {"passed_obstacle": "1", "passed_margin_m": "0", "relevant_obstacle_clearance_m": "3"}
        self.assertFalse(AUDIT._pass_gate(row))

    def test_pass_gate_clearance_floor(self):
        row = {"passed_obstacle": "1", "passed_margin_m": "0.1", "relevant_obstacle_clearance_m": "2.499"}
        self.assertFalse(AUDIT._pass_gate(row))

    def test_ols_slope(self):
        rows = [{"step": 0, "passed_margin_m": 1}, {"step": 1, "passed_margin_m": 3}, {"step": 2, "passed_margin_m": 5}]
        self.assertEqual(AUDIT._ols(rows, "passed_margin_m"), 2.0)

    def test_ols_needs_distinct_steps(self):
        self.assertIsNone(AUDIT._ols([{"step": 1, "x": 2}, {"step": 1, "x": 4}], "x"))

    def test_angle_same_direction(self):
        self.assertAlmostEqual(AUDIT._angle((1, 0), (4, 0)), 0.0)

    def test_angle_orthogonal(self):
        self.assertAlmostEqual(AUDIT._angle((1, 0), (0, 3)), 90.0)

    def test_angle_degenerate_is_structural_na(self):
        self.assertIsNone(AUDIT._angle((0, 0), (1, 0)))

    def test_negative_adjacent_count_and_fraction(self):
        rows = [{"step": i, "passed_margin_m": value} for i, value in enumerate((1, 2, 1, 0, 1))]
        result = AUDIT._negative_adjacent(rows)
        self.assertEqual(result["negative_adjacent_margin_count"], 2)
        self.assertEqual(result["negative_adjacent_margin_longest_run"], 2)
        self.assertEqual(result["negative_adjacent_margin_fraction"], 0.5)

    def test_negative_adjacent_requires_contiguous_rows(self):
        rows = [{"step": 0, "passed_margin_m": 2}, {"step": 2, "passed_margin_m": 1}]
        self.assertEqual(AUDIT._negative_adjacent(rows)["adjacent_pair_count"], 0)

    def test_goal_pass_improves_improves_quadrant(self):
        self.assertEqual(AUDIT._classify_goal_pass(-1, 1), "GOAL_IMPROVES_PASS_IMPROVES")

    def test_goal_pass_improves_worsens_quadrant(self):
        self.assertEqual(AUDIT._classify_goal_pass(-1, -1), "GOAL_IMPROVES_PASS_WORSENS")

    def test_goal_pass_worsens_improves_quadrant(self):
        self.assertEqual(AUDIT._classify_goal_pass(1, 1), "GOAL_WORSENS_PASS_IMPROVES")

    def test_goal_pass_worsens_worsens_quadrant(self):
        self.assertEqual(AUDIT._classify_goal_pass(1, -1), "GOAL_WORSENS_PASS_WORSENS")

    def test_goal_pass_zero_axis_is_explicit(self):
        self.assertEqual(AUDIT._classify_goal_pass(0, 1), "GOAL_ZERO_PASS_IMPROVES")
        self.assertEqual(AUDIT._classify_goal_pass(-1, 0), "GOAL_IMPROVES_PASS_ZERO")
        self.assertEqual(AUDIT._classify_goal_pass(0, 0), "GOAL_ZERO_PASS_ZERO")

    def test_goal_pass_missing_is_unavailable(self):
        self.assertEqual(AUDIT._classify_goal_pass(None, 0), "UNAVAILABLE")

    def test_assist_horizons_are_frozen(self):
        self.assertEqual(AUDIT.ASSIST_FIELDS[0], "passed_margin_m")
        self.assertEqual((1, 3, 5), (1, 3, 5))

    def test_exact_threshold_is_not_reacceleration(self):
        values = [False, False, False]
        self.assertEqual(AUDIT._longest_run(values), 0)

    def test_near_pass_band_is_frozen_descriptive_only(self):
        self.assertEqual(AUDIT.NEAR_PASS_BAND_M, (-1.0, 0.0))

    def test_fixed_cohorts(self):
        self.assertEqual(AUDIT.MAIN_IDS, tuple(f"test_r99_episode_{i:04d}" for i in (2, 4, 5, 6, 11, 20)))
        self.assertEqual(AUDIT.RESCUE_ID, "test_r99_episode_0012")
        self.assertEqual(AUDIT.SAFETY_ID, "test_r99_episode_0013")
        self.assertEqual(AUDIT.BOUNDARY_ID, "test_r99_episode_0014")

    def test_output_set_is_exactly_eight(self):
        self.assertEqual(len(AUDIT.OUTPUT_FILES), 8)
        self.assertEqual(len(set(AUDIT.OUTPUT_FILES)), 8)

    def test_atomic_output_names_match_contract(self):
        self.assertEqual(AUDIT.OUTPUT_FILES, (
            "prepass_handoff_summary.json", "prepass_handoff_episode_diagnostics.csv",
            "prepass_handoff_timing.csv", "prepass_handoff_pass_progress_windows.csv",
            "prepass_handoff_assist_event_response.csv", "prepass_handoff_goal_vs_pass_progress.csv",
            "prepass_handoff_special_cases.csv", "prepass_handoff_report.md",
        ))

    def test_canonical_hash_is_order_independent(self):
        self.assertEqual(AUDIT.canonical_hash({"a": 1, "b": 2}), AUDIT.canonical_hash({"b": 2, "a": 1}))

    def test_pass_limiter_classes(self):
        cases = ((-1, 3, "MARGIN_LIMITED"), (1, 2, "CLEARANCE_LIMITED"), (-1, 2, "BOTH_LIMITED"))
        for margin, clearance, expected in cases:
            if margin <= 0 and clearance >= AUDIT.PASS_CLEARANCE_M:
                result = "MARGIN_LIMITED"
            elif margin <= 0 and clearance < AUDIT.PASS_CLEARANCE_M:
                result = "BOTH_LIMITED"
            else:
                result = "CLEARANCE_LIMITED"
            self.assertEqual(result, expected)


class HandoffAuditWindowTests(unittest.TestCase):
    @staticmethod
    def _rows():
        margins = (-2.0, -1.0, 0.2, 0.1, -0.2)
        return [
            {"step": str(i), "passed_margin_m": str(margin), "relevant_obstacle_clearance_m": "3",
             "passed_obstacle": "1" if i >= 2 else "0", "passed_condition_now": "True" if margin > 0 else "False",
             "estimated_lateral_exit_ready": "1" if i >= 1 else "0", "radial_intent_assist_eligible": "1" if i == 1 else "0",
             "radial_intent_assist_active": "1" if i in (2, 4) else "0", "post_bypass_recovery": "0",
             "goal_recovery_correction_active": "0", "radial_intent_goal_recovery_active": "0",
             "safety_projection_applied": "0", "collision": "0", "goal_distance_m": str(10-i),
             "actual_lateral_velocity_mps": "0.4", "estimated_lateral_velocity_mps": "0.3",
             "actor_lateral_sign": "-1", "actor_lateral_sign_changes": "0", "along_obstacle_m": str(4-i),
             "obstacle_center_m": "[5,5,0]", "obstacle_relative_vector_m": "[4,5,0]",
             "signed_encounter_estimated_position_m": "[1,0,0]", "lateral_exit_assist_horizontal_goal_ray_m": "[1,0,0]",
             "radial_intent_assist_radial_unit": "[1,0,0]", "signed_axis_radial_unit": "[1,0,0]"}
            for i, margin in enumerate(margins)
        ]

    def test_pre_best_is_start_ready_through_best_inclusive(self):
        d, _, windows = AUDIT._episode_metrics("on", "e", self._rows(), {"passed_obstacle": "1"}, "MAIN_FAILURE")
        self.assertEqual([int(row["step"]) for row in windows["PRE_BEST"]], [1, 2])

    def test_post_best_excludes_best_row(self):
        _, _, windows = AUDIT._episode_metrics("on", "e", self._rows(), {"passed_obstacle": "1"}, "MAIN_FAILURE")
        self.assertEqual([int(row["step"]) for row in windows["POST_BEST"]], [3, 4])

    def test_best_tie_uses_earliest_step(self):
        rows = self._rows()
        rows[3]["passed_margin_m"] = rows[2]["passed_margin_m"]
        d, _, _ = AUDIT._episode_metrics("on", "e", rows, {}, "MAIN_FAILURE")
        self.assertEqual(d["best_margin_step"], 2)

    def test_pass_step_uses_exact_gate(self):
        d, _, _ = AUDIT._episode_metrics("on", "e", self._rows(), {}, "MAIN_FAILURE")
        self.assertEqual(d["first_pass_step"], 2)

    def test_assist_timing_labels_at_best(self):
        rows = self._rows()
        rows[2]["radial_intent_assist_active"] = "1"
        d, _, _ = AUDIT._episode_metrics("on", "e", rows, {}, "MAIN_FAILURE")
        self.assertEqual(d["assist_timing_vs_best"], "ASSIST_AT_BEST")

    def test_assist_timing_never(self):
        rows = self._rows()
        for row in rows:
            row["radial_intent_assist_active"] = "0"
        d, _, _ = AUDIT._episode_metrics("off", "e", rows, {}, "MAIN_FAILURE")
        self.assertEqual(d["assist_timing_vs_best"], "ASSIST_NEVER_ACTIVE")

    def test_timing_deltas_are_literal(self):
        _, timing, _ = AUDIT._episode_metrics("on", "e", self._rows(), {}, "MAIN_FAILURE")
        self.assertEqual(timing["assist_start_minus_best_margin_step"], 0)
        self.assertEqual(timing["lateral_exit_minus_best_margin_step"], -1)
        self.assertEqual(timing["best_margin_to_end_steps"], 2)
        self.assertEqual(timing["assist_start_to_end_steps"], 2)
        self.assertEqual(timing["best_margin_to_pass_steps"], 0)

    def test_source_consistency_at_best(self):
        d, _, _ = AUDIT._episode_metrics("on", "e", self._rows(), {}, "MAIN_FAILURE")
        self.assertTrue(d["pass_source_condition_consistent_at_best"])

    def test_post_best_margin_regression(self):
        d, _, _ = AUDIT._episode_metrics("on", "e", self._rows(), {}, "MAIN_FAILURE")
        self.assertTrue(d["best_to_final_regression"])
        self.assertGreater(d["post_best_negative_adjacent_margin_count"], 0)

    def test_frame_angle_is_zero_for_aligned_xy(self):
        rows = self._rows()
        self.assertAlmostEqual(AUDIT._angle_from_row(rows[2], use_assist_axis=True), 0.0)

    def test_structural_angle_na_for_degenerate(self):
        row = self._rows()[0]
        row["lateral_exit_assist_horizontal_goal_ray_m"] = "[0,0,0]"
        self.assertIsNone(AUDIT._angle_from_row(row, use_assist_axis=True))

    def test_goal_pass_window_rows_have_zero_axis(self):
        rows = self._rows()
        rows[0]["goal_distance_m"] = "10"
        rows[1]["goal_distance_m"] = "10"
        out = AUDIT._goal_pass_rows("on", "e", "MAIN_FAILURE", rows, 1, 2)
        self.assertEqual(out[0]["classification"], "GOAL_ZERO_PASS_IMPROVES")

    def test_assist_response_beyond_end_is_structural_na(self):
        rows = self._rows()
        rows[-1]["radial_intent_assist_active"] = "1"
        out = AUDIT._assist_response("e", "MAIN_FAILURE", rows)
        final_horizon = next(row for row in out if row["assist_step"] == 4 and row["horizon_steps"] == 5)
        self.assertEqual(final_horizon["response_status"], "STRUCTURAL_NA_BEYOND_EPISODE_END")

    def test_assist_response_internal_gap_fails_closed(self):
        rows = [row for row in self._rows() if row["step"] != "3"]
        with self.assertRaises(AUDIT.AuditError):
            AUDIT._assist_response("e", "MAIN_FAILURE", rows)

    def test_assist_response_uses_exact_direction_labels(self):
        out = AUDIT._assist_response("e", "MAIN_FAILURE", self._rows())
        labels = {row["pass_margin_response"] for row in out}
        self.assertTrue(labels <= {"PASS_MARGIN_IMPROVES", "PASS_MARGIN_WORSENS", "PASS_MARGIN_FLAT", "STRUCTURAL_NA"})


class HandoffAuditStaticSafetyTests(unittest.TestCase):
    def test_module_imports_only_standard_library(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] in {"argparse", "ast", "csv", "hashlib", "json", "math", "os", "re", "shutil", "sys", "tempfile", "typing", "io"} for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], {"__future__", "pathlib", "typing"})

    def test_no_environment_step_call(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "step" for node in calls))

    def test_no_rollout_or_training_call(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotRegex(node.func.id.lower(), r"(rollout|train|fit|update_policy)")
                if isinstance(node.func, ast.Attribute):
                    self.assertNotRegex(node.func.attr.lower(), r"(rollout|train|fit|update_policy)")

    def test_no_runtime_or_controller_imports(self):
        text = SOURCE.read_text(encoding="utf-8").lower()
        self.assertNotRegex(text, r"from\s+src\.(control|simulation|learning)\s+import")
        self.assertNotRegex(text, r"import\s+src\.(control|simulation|learning)")

    def test_source_has_no_controller_mutation_api(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        forbidden = {"load_state_dict", "state_dict", "optimizer", "environment", "controller"}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue(names.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
