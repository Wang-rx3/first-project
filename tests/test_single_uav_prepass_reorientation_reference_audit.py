"""Synthetic unit checks for the pre-pass reference audit's pure helpers."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "experiments" / "single_uav_prepass_reorientation_reference_audit.py"
SPEC = importlib.util.spec_from_file_location("prepass_reorientation_reference_audit_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(step: int, **values):
    data = {"step": step}
    data.update(values)
    return data


def metric(value, count=5):
    return audit.metric_record(value, count)


def default_failure_metrics(**updates):
    values = {name: metric(5.0) for name in audit.METRIC_NAMES}
    values.update(updates)
    return values


def default_references(**updates):
    values = {name: {"q1": 1.0, "q3": 10.0, "n": 11} for name in audit.REFERENCE_METRICS}
    values.update(updates)
    return values


class PrepassWindowTests(unittest.TestCase):
    def test_fixed_cohorts_and_six_file_contract(self):
        self.assertEqual(tuple(item[-4:] for item in audit.SUCCESS_IDS), ("0001", "0003", "0007", "0008", "0009", "0010", "0015", "0016", "0017", "0018", "0019"))
        self.assertEqual(tuple(item[-4:] for item in audit.FAILURE_IDS), ("0002", "0004", "0005", "0006", "0011", "0012", "0020"))
        self.assertEqual(audit.ALL_IDS, tuple(f"test_r99_episode_{index:04d}" for index in range(1, 21)))
        self.assertEqual(len(audit.OUTPUT_FILES), 6)
        self.assertEqual(len(set(audit.OUTPUT_FILES)), 6)

    def test_success_window_excludes_first_gate_transition(self):
        rows = [
            row(4, passed_margin_m=0.2, relevant_obstacle_clearance_m=2.49),
            row(5, passed_obstacle=True, passed_margin_m=0.3, relevant_obstacle_clearance_m=2.5),
            row(6, passed_obstacle=True, passed_margin_m=3.0, relevant_obstacle_clearance_m=3.0),
        ]
        selected, gate_step, transition, reason = audit.success_prepass_window(rows, 4)
        self.assertEqual(gate_step, 5)
        self.assertEqual([item["step"] for item in selected], [4])
        self.assertEqual(transition["step"], 5)
        self.assertIsNone(reason)

    def test_formal_pass_gate_uses_strict_margin_and_inclusive_clearance(self):
        self.assertFalse(audit.exact_pass_gate(row(1, passed_margin_m=0, relevant_obstacle_clearance_m=2.5)))
        self.assertFalse(audit.exact_pass_gate(row(1, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.499)))
        self.assertFalse(audit.exact_pass_gate(row(1, passed_obstacle=False, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.5)))
        self.assertTrue(audit.exact_pass_gate(row(1, passed_obstacle=True, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.5)))

    def test_transition_metadata_is_explicit_and_separate(self):
        transition = row(
            8, passed_margin_m=0.4, relevant_obstacle_clearance_m=2.7,
            passed_obstacle="true", post_bypass_recovery="1",
            goal_recovery_correction_eligible="yes", goal_recovery_correction_active="true",
            goal_recovery_correction_reason="eligible_and_active",
        )
        values = audit.pass_transition_metadata(transition)
        self.assertEqual(values, {
            "transition_passed_obstacle": True,
            "transition_post_bypass_recovery": True,
            "transition_goal_recovery_eligible": True,
            "transition_goal_recovery_active": True,
            "transition_goal_recovery_reason": "eligible_and_active",
            "transition_passed_margin": 0.4,
            "transition_clearance": 2.7,
        })

    def test_transition_flags_are_excluded_from_prepass_contamination_scan(self):
        rows = [
            row(4, passed_margin_m=-0.1, relevant_obstacle_clearance_m=2.0),
            row(5, passed_margin_m=0.2, relevant_obstacle_clearance_m=2.5,
                passed_obstacle="true", post_bypass_recovery="true", goal_recovery_correction_active="true"),
        ]
        window, first_pass, transition, reason = audit.success_prepass_window(rows, 4)
        audit.validate_success_window_contamination({audit.SUCCESS_IDS[0]: window})
        self.assertEqual([item["step"] for item in window], [4])
        self.assertEqual(first_pass, 5)
        self.assertEqual(transition["step"], 5)
        self.assertIsNone(reason)

    def test_edge_first_pass_before_local_exit_is_not_assessable(self):
        rows = [row(2, passed_obstacle=True, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.5), row(3, passed_obstacle=True, passed_margin_m=0.4, relevant_obstacle_clearance_m=3.0)]
        window, first_pass, transition, reason = audit.success_prepass_window(rows, 4)
        self.assertEqual(window, [])
        self.assertEqual(first_pass, 2)
        self.assertEqual(transition["step"], 2)
        self.assertEqual(reason, "NO_PREPASS_REFERENCE_WINDOW")

    def test_edge_first_pass_equal_to_local_exit_is_not_assessable(self):
        rows = [row(4, passed_obstacle=True, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.5)]
        window, first_pass, transition, reason = audit.success_prepass_window(rows, 4)
        self.assertEqual(window, [])
        self.assertEqual(first_pass, 4)
        self.assertEqual(transition["step"], 4)
        self.assertEqual(reason, "NO_PREPASS_REFERENCE_WINDOW")

    def test_missing_local_exit_and_missing_pass_remain_structural(self):
        passing = [row(2, passed_obstacle=True, passed_margin_m=0.1, relevant_obstacle_clearance_m=2.5)]
        no_exit = audit.success_prepass_window(passing, None)
        self.assertEqual(no_exit[0], [])
        self.assertEqual(no_exit[1], 2)
        self.assertEqual(no_exit[3], "LOCAL_EXIT_NOT_OBSERVED")
        no_pass = audit.success_prepass_window([row(2, passed_margin_m=-1, relevant_obstacle_clearance_m=2.5)], 1)
        self.assertEqual(no_pass, ([], None, None, "FIRST_PASS_NOT_OBSERVED"))

    def test_success_contamination_reports_exact_token_and_offender(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            windows = {audit.SUCCESS_IDS[0]: [row(3, goal_recovery_correction_active="true")]}
            with self.assertRaises(audit.PrepassContaminationError) as raised:
                audit.validate_success_window_contamination(windows)
            self.assertIn("PREPASS_REFERENCE_CONTROLLER_CONTAMINATION", str(raised.exception))
            self.assertEqual(raised.exception.details["episode_id"], audit.SUCCESS_IDS[0])
            self.assertEqual(raised.exception.details["first_offending_step"], 3)
            self.assertEqual(raised.exception.details["contaminating_fields"], ["goal_recovery_correction_active"])
            self.assertEqual(list(root.iterdir()), [])

    def test_post_bypass_recovery_in_prepass_triggers_contamination(self):
        with self.assertRaises(audit.PrepassContaminationError) as raised:
            audit.validate_success_window_contamination({audit.SUCCESS_IDS[0]: [row(3, post_bypass_recovery="true")]})
        self.assertEqual(raised.exception.details["contaminating_fields"], ["post_bypass_recovery"])

    def test_pass_latch_in_prepass_triggers_contamination(self):
        with self.assertRaises(audit.PrepassContaminationError) as raised:
            audit.validate_success_window_contamination({audit.SUCCESS_IDS[0]: [row(3, passed_obstacle="1")]})
        self.assertEqual(raised.exception.details["contaminating_fields"], ["passed_obstacle"])

    def test_summary_activity_counts_distinguish_prepass_from_transition(self):
        counts = audit.goal_recovery_activity_counts([
            {"cohort": "SUCCESS_REFERENCE", "goal_recovery_active_before_pass_row_count": 0, "transition_goal_recovery_active": True},
            {"cohort": "SUCCESS_REFERENCE", "goal_recovery_active_before_pass_row_count": 0, "transition_goal_recovery_active": False},
            {"cohort": "MAIN_FAILURE", "goal_recovery_active_before_pass_row_count": 99, "transition_goal_recovery_active": True},
        ])
        self.assertEqual(counts, {
            "active_before_transition_row_count": 0,
            "active_before_transition_episode_count": 0,
            "active_on_transition_row_count": 1,
            "active_on_transition_episode_count": 1,
        })
        self.assertEqual(audit.PREVIOUS_FAIL_FAST_REASON, "first pass transition row could already have Goal Recovery active by evaluator timing")

    def test_exit_search_requires_three_consecutive_frozen_frames(self):
        rows = [row(8, actual_lateral_velocity_mps=0.1), row(9, actual_lateral_velocity_mps=0.2), row(11, actual_lateral_velocity_mps=0.1), row(12, actual_lateral_velocity_mps=0.3), row(13, actual_lateral_velocity_mps=-0.5)]
        self.assertEqual(audit.first_local_exit_step(rows, 8), 11)
        self.assertIsNone(audit.first_local_exit_step(rows[:2], 8))

    def test_failure_best_tie_uses_earliest_step_and_partitions_exactly(self):
        rows = [row(2, passed_margin_m=-2), row(3, passed_margin_m=4), row(4, passed_margin_m=4), row(5, passed_margin_m=1)]
        segments = audit.make_segments(rows, 2)
        self.assertEqual([item["step"] for item in segments["PRE_BEST"]["rows"]], [2, 3])
        self.assertEqual([item["step"] for item in segments["POST_BEST"]["rows"]], [4, 5])
        self.assertEqual([item["step"] for item in segments["PRE_BEST"]["rows"] + segments["POST_BEST"]["rows"]], [2, 3, 4, 5])
        self.assertFalse(segments["POST_BEST"]["start_inclusive"])
        self.assertEqual(segments["POST_BEST"]["start_step"], 3)

    def test_failure_empty_post_best_retains_structural_window(self):
        segments = audit.make_segments([row(1, passed_margin_m=0), row(2, passed_margin_m=3)], 1)
        self.assertEqual([item["step"] for item in segments["PRE_BEST"]["rows"]], [1, 2])
        self.assertEqual(segments["POST_BEST"]["rows"], [])
        self.assertEqual(segments["POST_BEST"]["structural_reason"], "EMPTY_POST_BEST_AT_WINDOW_END")


class MetricTests(unittest.TestCase):
    def test_goalward_and_sustained_fractions_use_finite_observations_only(self):
        rows = [
            row(1, audit_actor_radial_delta_v_mps=2.0, audit_final_radial_delta_v_mps=1.0, audit_actual_radial_velocity_mps=1.0),
            row(2, audit_actor_radial_delta_v_mps=2.0, audit_final_radial_delta_v_mps=1.0, audit_actual_radial_velocity_mps=None),
            row(3, audit_actor_radial_delta_v_mps=2.0, audit_final_radial_delta_v_mps=1.0, audit_actual_radial_velocity_mps=1.0),
            row(4, audit_actor_radial_delta_v_mps=2.0, audit_final_radial_delta_v_mps=1.0, audit_actual_radial_velocity_mps=1.0),
            row(5, audit_actor_radial_delta_v_mps=-1.0, audit_final_radial_delta_v_mps=-1.0, audit_actual_radial_velocity_mps=-1.0),
        ]
        result = audit.compute_metrics(rows)
        self.assertEqual(result["actor_goalward_fraction"]["numerator"], 4)
        self.assertEqual(result["actor_goalward_fraction"]["denominator"], 5)
        self.assertEqual(result["actor_goalward_radial_step_fraction_prepass"], result["actor_goalward_fraction"])
        self.assertEqual(result["final_goalward_radial_step_fraction_prepass"], result["final_goalward_fraction"])
        self.assertEqual(result["actual_goalward_radial_step_fraction_prepass"], result["actual_goalward_fraction"])
        self.assertEqual(result["actor_goalward_sustained_window_fraction"]["value"], 1.0)
        self.assertEqual(result["actual_goalward_fraction"]["sample_count"], 4)
        self.assertEqual(result["actual_goalward_sustained_3step_fraction"]["sample_count"], 1)

    def test_zero_positive_actor_sum_makes_retention_undefined(self):
        result = audit.compute_metrics([
            row(1, actor_raw_radial_delta_v_mps=-1, final_applied_radial_delta_v_mps=2, actor_raw_lateral_delta_v_mps=0, final_applied_lateral_delta_v_mps=1),
            row(2, actor_raw_radial_delta_v_mps=-2, final_applied_radial_delta_v_mps=3, actor_raw_lateral_delta_v_mps=0, final_applied_lateral_delta_v_mps=1),
        ])
        retention = result["actor_to_final_radial_retention"]
        self.assertIsNone(retention["value"])
        self.assertEqual(retention["unavailable_reason"], "ACTOR_POSITIVE_RADIAL_SUM_NOT_ABOVE_EPSILON")
        self.assertEqual(retention["denominator"], 0.0)
        self.assertEqual(result["actor_radial_to_abs_lateral_ratio"]["unavailable_reason"], "MEAN_ABS_LATERAL_NOT_ABOVE_EPSILON")

    def test_paired_retention_ratio_and_positive_means(self):
        result = audit.compute_metrics([
            row(1, audit_actor_radial_delta_v_mps=2, audit_final_radial_delta_v_mps=1, audit_actual_radial_velocity_mps=3, audit_actor_lateral_delta_v_mps=2, audit_final_lateral_delta_v_mps=1),
            row(2, audit_actor_radial_delta_v_mps=-1, audit_final_radial_delta_v_mps=3, audit_actual_radial_velocity_mps=-2, audit_actor_lateral_delta_v_mps=-2, audit_final_lateral_delta_v_mps=-1),
        ])
        self.assertAlmostEqual(result["actor_to_final_radial_retention"]["value"], 2.0)
        self.assertAlmostEqual(result["actor_radial_positive_mean_mps"]["value"], 1.0)
        self.assertAlmostEqual(result["actor_radial_to_abs_lateral_ratio"]["value"], 0.5)
        self.assertAlmostEqual(result["actual_radial_mean_mps"]["value"], 0.5)

    def test_ols_requires_distinct_steps_and_margin_best_tie_is_earliest(self):
        same_step = [row(1, goal_distance_m=3), row(1, goal_distance_m=2)]
        self.assertEqual(audit.compute_metrics(same_step)["goal_distance_ols_slope_m_per_step"]["unavailable_reason"], "NO_DISTINCT_FINITE_STEPS")
        rows = [row(1, goal_distance_m=10, passed_margin_m=1), row(2, goal_distance_m=8, passed_margin_m=5), row(3, goal_distance_m=4, passed_margin_m=5)]
        result = audit.compute_metrics(rows)
        self.assertAlmostEqual(result["goal_distance_ols_slope_m_per_step"]["value"], -3.0)
        self.assertEqual(result["goal_distance_net_reduction_m"]["value"], 6.0)
        self.assertEqual(result["margin_best_step"]["value"], 2.0)
        self.assertEqual(result["margin_best_minus_final_m"]["value"], 0.0)

    def test_reacceleration_counts_only_consecutive_three_sample_runs(self):
        values = [0.6, 0.7, 0.8, 0.1, 0.9, 1.0, 1.1]
        rows = [row(index + 1, actual_lateral_velocity_mps=value) for index, value in enumerate(values)]
        result = audit.compute_metrics(rows)
        self.assertEqual(result["lateral_reacceleration_run_count"]["value"], 2.0)
        self.assertEqual(result["lateral_reacceleration_duration_steps"]["value"], 6.0)
        self.assertEqual(result["lateral_reacceleration_peak_abs_mps"]["value"], 1.1)

    def test_actor_side_fractions_and_sign_changes(self):
        rows = [row(1, actor_raw_lateral_delta_v_mps=1), row(2, actor_raw_lateral_delta_v_mps=-1), row(3, actor_raw_lateral_delta_v_mps=-2)]
        result = audit.compute_metrics(rows, initial_lateral_sign=1)
        self.assertEqual(result["actor_original_side_fraction"]["value"], 1.0 / 3.0)
        self.assertEqual(result["actor_counter_side_fraction"]["value"], 2.0 / 3.0)
        self.assertEqual(result["actor_lateral_sign_change_count"]["value"], 1.0)


class ReferenceAndDiagnosisTests(unittest.TestCase):
    def test_type7_quartiles_and_inclusive_iqr_edges(self):
        self.assertEqual(audit.type7_quantile([0, 10], 0.25), 2.5)
        reference = {"q1": 2.5, "q3": 7.5}
        self.assertEqual(audit.compare_iqr(2.5, reference), "WITHIN_IQR")
        self.assertEqual(audit.compare_iqr(7.5, reference), "WITHIN_IQR")
        self.assertEqual(audit.compare_iqr(2.49, reference), "BELOW_IQR")
        self.assertEqual(audit.compare_iqr(7.51, reference), "ABOVE_IQR")
        self.assertIsNone(audit.compare_iqr(None, reference))

    def test_reference_is_rebuilt_from_fixed_success_metrics_only(self):
        samples = {episode_id: {name: metric(index + 1) for name in audit.REFERENCE_METRICS} for index, episode_id in enumerate(audit.SUCCESS_IDS)}
        references = audit.build_success_references(samples)
        self.assertEqual(references["actor_goalward_fraction"]["eligible_episode_count"], 11)
        self.assertEqual(references["actor_goalward_fraction"]["n"], 11)
        self.assertEqual(references["actor_goalward_fraction"]["quantile_method"], "numpy_linear_type7")
        self.assertEqual(references["actor_goalward_fraction"]["q1"], 3.5)

    def test_diagnosis_paths_cover_all_required_labels(self):
        refs = default_references()
        def run(**updates):
            return audit.diagnose_failure(default_failure_metrics(**updates), refs, passed=False)

        self.assertEqual(run(actor_goalward_fraction=metric(0))["primary_diagnosis"], "RADIAL_INTENT_NOT_SUSTAINED")
        self.assertEqual(run(final_goalward_fraction=metric(0))["primary_diagnosis"], "PIPELINE_RADIAL_ATTENUATION")
        self.assertEqual(run(actor_radial_positive_mean_mps=metric(0))["primary_diagnosis"], "RADIAL_PROPULSION_INSUFFICIENT")
        self.assertEqual(run(actual_goalward_fraction=metric(0))["primary_diagnosis"], "RADIAL_MOTION_RESPONSE_WEAK")
        self.assertEqual(run(lateral_reacceleration_run_count=metric(1), lateral_reacceleration_duration_steps=metric(11))["primary_diagnosis"], "LATERAL_REACCELERATION")
        self.assertEqual(run(margin_best_minus_final_m=metric(11))["primary_diagnosis"], "PASS_PROGRESS_REGRESSION")
        self.assertEqual(run()["primary_diagnosis"], "PASS_GEOMETRY_NOT_REACHED")
        self.assertEqual(run(actual_goalward_fraction=metric(0), margin_best_minus_final_m=metric(11))["primary_diagnosis"], "MIXED")
        missing = default_failure_metrics(actual_radial_mean_mps=audit.null_metric("MISSING"))
        self.assertEqual(audit.diagnose_failure(missing, refs, passed=False)["primary_diagnosis"], "NOT_ASSESSABLE")

    def test_evidence_change_statuses(self):
        self.assertEqual(audit.evidence_change_status(True, True), "RETAINED_AFTER_PREPASS_CORRECTION")
        self.assertEqual(audit.evidence_change_status(True, False), "WITHDRAWN_AFTER_PREPASS_CORRECTION")
        self.assertEqual(audit.evidence_change_status(False, True), "NOT_PREVIOUSLY_ACTIVE")
        self.assertEqual(audit.evidence_change_status(None, False), "NOT_ASSESSABLE")

    def test_recommendation_requires_four_intent_below_and_pipeline_gate(self):
        refs = default_references()
        failure_data = {}
        for index, episode_id in enumerate(audit.FAILURE_IDS):
            updates = {}
            if index < 4:
                updates["actor_goalward_sustained_window_fraction"] = metric(0)
                if index == 0:
                    updates["final_goalward_fraction"] = metric(0)
            failure_data[episode_id] = default_failure_metrics(**updates)
        decision = audit.recommendation_decision(failure_data, refs)
        self.assertTrue(decision["all_seven_actor_persistence_comparisons_assessable"])
        self.assertEqual(decision["intent_below_reference_count"], 4)
        self.assertTrue(decision["pipeline_assessable_for_every_intent_below_episode"])
        self.assertEqual(decision["pipeline_active_count_among_intent_below"], 1)
        self.assertTrue(decision["authorized_minimal_radial_intent_persistence_diagnostic_preparation"])
        self.assertEqual(decision["single_suggestion"], "PREPARE_AUTHORIZED_MINIMAL_RADIAL_INTENT_PERSISTENCE_OFF_ON_CAUSAL_DIAGNOSTIC")
        failure_data[audit.FAILURE_IDS[0]]["actor_goalward_fraction"] = audit.null_metric("NO_SAMPLE")
        undecided = audit.recommendation_decision(failure_data, refs)
        self.assertIsNone(undecided["intent_below_reference_count"])
        self.assertIsNone(undecided["authorized_minimal_radial_intent_persistence_diagnostic_preparation"])
        self.assertEqual(undecided["single_suggestion"], "NO_INTERVENTION_CONTINUE_OFFLINE_MECHANISM_AUDIT_ONLY")

    def test_static_contract_has_no_runtime_or_step_execution_imports(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden_roots = {"src.simulation", "src.control", "src.learning", "gymnasium", "torch"}
        self.assertFalse(any(name == root or name.startswith(root + ".") for name in imported for root in forbidden_roots))
        forbidden_call = "." + "step("
        self.assertNotIn(forbidden_call, source)


if __name__ == "__main__":
    unittest.main()
