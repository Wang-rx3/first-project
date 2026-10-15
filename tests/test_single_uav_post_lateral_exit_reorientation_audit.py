"""Synthetic unit coverage for the post-lateral-exit offline audit."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments"
    / "single_uav_post_lateral_exit_reorientation_audit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "single_uav_post_lateral_exit_reorientation_audit", AUDIT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def step_row(step: int, *, speed: float = 0.0, eligible: bool = False) -> dict[str, str]:
    return {
        "episode_index": "1",
        "episode_id": "e1",
        "condition": "off",
        "step": str(step),
        "lateral_exit_assist_eligible": str(int(eligible)),
        "actual_lateral_velocity_mps": str(speed),
        "passed_margin_m": "-1",
        "relevant_obstacle_clearance_m": "3",
        "passed_condition_now": "False",
        "passed_obstacle": "0",
        "goal_distance_m": str(10 - step),
    }


class AuditMathTests(unittest.TestCase):
    def test_step_pair_identity_and_continuity(self) -> None:
        rows = [step_row(0), step_row(1)]
        grouped = audit.validate_step_continuity(rows, expected_episode_ids=("e1",), condition="off")
        self.assertEqual([int(row["step"]) for row in grouped["e1"]], [0, 1])
        with self.assertRaises(audit.SourceIdentityError):
            audit.validate_step_continuity([step_row(0), step_row(2)], expected_episode_ids=("e1",))
        with self.assertRaises(audit.SourceIdentityError):
            audit.validate_step_continuity([step_row(0), step_row(0)], expected_episode_ids=("e1",))

    def test_frame_and_vector_projection_integrity(self) -> None:
        rows = []
        for step in range(3):
            row = step_row(step)
            row.update({
                "lateral_exit_assist_horizontal_goal_ray_m": "[2,0,0]",
                "lateral_exit_assist_lateral_unit": "[0,1,0]",
                "lateral_exit_assist_frame_source": "encounter_time_ekf_estimated_goal_ray_projected_to_xy",
                "signed_axis_radial_unit": "[1,0,0]",
                "signed_axis_lateral_unit": "[0,1,0]",
                "signed_axis_vertical_unit": "[0,0,1]",
                "actor_raw_delta_v_mps": "[0.4,-0.1,0]",
                "applied_delta_v_mps": "[0.3,0.2,0]",
                "actor_raw_radial_delta_v_mps": "0.4",
                "actor_raw_lateral_delta_v_mps": "-0.1",
                "final_applied_radial_delta_v_mps": "0.3",
                "final_applied_lateral_delta_v_mps": "0.2",
                "estimated_radial_velocity_mps": "1",
                "actual_radial_velocity_mps": "0.8",
                "estimated_lateral_velocity_mps": "0.1",
                "estimated_vertical_velocity_mps": "0",
                "actual_vertical_velocity_mps": "0",
            })
            rows.append(row)
        rows[0]["actual_lateral_velocity_mps"] = "-0.2"
        rows[1]["actual_lateral_velocity_mps"] = "-0.1"
        rows[2]["actual_lateral_velocity_mps"] = "0.0"
        result = audit.validate_frame_and_decomposition(
            {"e1": rows}, [{"episode_id": "e1", "signed_encounter_step": "0"}]
        )
        self.assertEqual(result["status"], "PASS")
        broken = [dict(row) for row in rows]
        broken[1]["final_applied_lateral_delta_v_mps"] = "0.7"
        with self.assertRaises(audit.SourceIdentityError):
            audit.validate_frame_and_decomposition(
                {"e1": broken}, [{"episode_id": "e1", "signed_encounter_step": "0"}]
            )

    def test_off_anchor_and_common_window_are_shared(self) -> None:
        off = {"e1": [step_row(0), step_row(1), step_row(2, eligible=True), step_row(3)]}
        on_rows = [step_row(0), step_row(1), step_row(2), step_row(3)]
        for row in on_rows:
            row["condition"] = "on"
        on = {"e1": on_rows}
        pairs = [{
            "episode_id": "e1",
            "triggered_cohort_status": "INCLUDED",
            "off_defined_anchor_step": "2",
            "common_window_end_step": "3",
        }]
        selected = audit.select_common_windows(pairs, off, on)["e1"]
        self.assertEqual(selected["anchor_step"], 2)
        self.assertEqual(selected["common_end_step"], 3)
        self.assertEqual([int(row["step"]) for row in selected["off_rows"]], [2, 3])
        self.assertEqual([int(row["step"]) for row in selected["on_rows"]], [2, 3])

    def test_off_never_eligible_is_structural_na(self) -> None:
        off = {"e1": [step_row(0), step_row(1)]}
        on = {"e1": [dict(step_row(0), condition="on"), dict(step_row(1), condition="on")]}
        pair = [{
            "episode_id": "e1", "triggered_cohort_status": "OFF_NEVER_ELIGIBLE",
            "off_defined_anchor_step": "", "common_window_end_step": "",
        }]
        result = audit.select_common_windows(pair, off, on)["e1"]
        self.assertEqual(result["status"], "OFF_NEVER_ELIGIBLE")
        self.assertIsNone(result["anchor_step"])
        self.assertEqual(result["off_rows"], [])

    def test_local_exit_requires_three_consecutive_steps(self) -> None:
        rows = [step_row(index, speed=value) for index, value in enumerate((0.2, 0.9, 0.45, 0.40, 0.2, 0.8))]
        # An obsolete one-sample threshold would select step 0; the fixed rule selects step 2.
        self.assertEqual(
            audit.first_local_exit_step(rows, encounter_step=0, common_anchor_step=None), 2
        )
        self.assertIsNone(
            audit.first_local_exit_step(rows[:2], encounter_step=0, common_anchor_step=None)
        )

    def test_sustained_actor_and_actual_radial_rules(self) -> None:
        self.assertTrue(audit.sustained_positive([1, 1, 1, 1, 0], window=5, minimum=4))
        self.assertFalse(audit.sustained_positive([1, 1, 1, 0, 0], window=5, minimum=4))
        self.assertTrue(audit.sustained_positive([0, 0, 0, 0, -1], window=5, minimum=4, strict=False))
        self.assertFalse(audit.sustained_positive([0, 0, 0, 0, -1], window=5, minimum=4, strict=True))
        self.assertTrue(audit.sustained_actual_radial([2e-8, 2e-8, 3e-8, 0]))
        self.assertFalse(audit.sustained_actual_radial([2e-8, 1e-8, 3e-8]))

    def test_full_window_ols_slope_and_net_reduction(self) -> None:
        rows = [step_row(0), step_row(1), step_row(2)]
        rows[0]["goal_distance_m"] = "9"
        rows[1]["goal_distance_m"] = "7"
        rows[2]["goal_distance_m"] = "5"
        self.assertAlmostEqual(audit.ols_slope(rows), -2.0)
        metrics = audit.metrics_for_window(rows)
        self.assertAlmostEqual(metrics["goal_distance_net_reduction_m"], 4.0)

    def test_exact_pass_gate_and_latch_cross_check(self) -> None:
        rows = [step_row(0), step_row(1), step_row(2)]
        rows[0].update({"passed_margin_m": "0", "relevant_obstacle_clearance_m": "2.5"})
        rows[1].update({"passed_margin_m": "0.01", "relevant_obstacle_clearance_m": "2.5", "passed_condition_now": "True", "passed_obstacle": "1"})
        rows[2].update({"passed_margin_m": "-0.2", "relevant_obstacle_clearance_m": "4", "passed_obstacle": "1"})
        self.assertIs(audit.pass_gate(rows[0]), False)
        self.assertIs(audit.pass_gate(rows[1]), True)
        result = audit.validate_pass_latches({"e1": rows}, [{"episode_id": "e1", "passed_obstacle": "1"}])
        self.assertEqual(result["passed"], 1)

    def test_margin_regression_tracks_earliest_best_step(self) -> None:
        rows = [step_row(0), step_row(1), step_row(2), step_row(3)]
        for row, margin in zip(rows, (0.0, 2.0, 2.0, 1.0)):
            row["passed_margin_m"] = str(margin)
        metrics = audit.margin_summary(rows)
        self.assertEqual(metrics["margin_best_step"], 1)
        self.assertEqual(metrics["margin_regression_best_minus_final_m"], 1.0)
        self.assertEqual(metrics["margin_delta_anchor_to_final_m"], 1.0)

    def test_nonoverlapping_lateral_reacceleration_runs(self) -> None:
        speeds = (0.6, 0.7, 0.8, 0.1, -0.9, -0.8, 0.6, 0.2)
        rows = [step_row(index, speed=value) for index, value in enumerate(speeds)]
        result = audit.reacceleration_summary(rows)
        self.assertEqual(result["lateral_reacceleration_first_step"], 0)
        self.assertEqual(result["lateral_reacceleration_run_count"], 2)
        self.assertEqual(result["lateral_reacceleration_duration_steps"], 6)

    def test_type7_quantiles_and_inclusive_iqr_bounds(self) -> None:
        rows = [
            {"episode_id": episode_id, "actor_radial_mean_mps": float(index)}
            for index, episode_id in enumerate(audit.SUCCESS_REFERENCE_IDS)
        ]
        reference = audit.success_reference_statistics(rows, metrics=("actor_radial_mean_mps",))["actor_radial_mean_mps"]
        self.assertEqual(reference["n"], 11)
        self.assertEqual(reference["q1"], 2.5)
        self.assertEqual(reference["median"], 5.0)
        self.assertEqual(reference["q3"], 7.5)
        self.assertEqual(audit.compare_iqr(2.5, reference), "WITHIN_IQR")
        self.assertEqual(audit.compare_iqr(7.5, reference), "WITHIN_IQR")
        self.assertEqual(audit.compare_iqr(2.49, reference), "BELOW_IQR")

    def test_closed_taxonomy_and_special_annotations(self) -> None:
        with self.assertRaises(ValueError):
            audit.validate_diagnosis("OUTLIER_SCORE")
        safety = audit.diagnose_on_episode("test_r99_episode_0013", {}, {}, local_exit_found=False, passed=False)
        boundary = audit.diagnose_on_episode("test_r99_episode_0014", {}, {}, local_exit_found=False, passed=False)
        adverse = audit.diagnose_on_episode("test_r99_episode_0008", {}, {}, local_exit_found=True, passed=False)
        self.assertEqual(safety["special_case_diagnosis"], "SAFETY_DOMINATED_SPECIAL_CASE")
        self.assertEqual(boundary["special_case_diagnosis"], "BOUNDARY_DOMINATED_SPECIAL_CASE")
        self.assertEqual(safety["primary_diagnosis"], "NOT_ASSESSABLE")
        self.assertEqual(boundary["primary_diagnosis"], "NOT_ASSESSABLE")
        self.assertEqual(adverse["case_annotation"], "DOWNSTREAM_ADVERSE_SPECIAL_CASE")
        self.assertEqual(adverse["primary_diagnosis"], "NOT_ASSESSABLE")
        self.assertEqual(adverse["secondary_diagnoses"], [])

    def test_main_failure_goal_recovery_inactivity(self) -> None:
        rows = [{
            "episode_id": episode_id,
            "off_no_release": True,
            "on_no_release": True,
            "off_goal_recovery_inactive": True,
            "on_goal_recovery_inactive": True,
        } for episode_id in audit.MAIN_FAILURE_IDS]
        result = audit.validate_main_failure_pipeline(rows)
        self.assertTrue(result["no_release_all_main_failures"])
        self.assertTrue(result["goal_recovery_inactive_all_main_failures"])

    def test_preregistration_and_hash_guards_fail_closed(self) -> None:
        with self.assertRaises(audit.SourceIdentityError):
            audit.validate_preregistration({"schema_version": "wrong", "active_preregistration_hash": "bad"})
        trace = [{"episode_index": "1", "episode_id": "e1", "step": "0", "reward": "1"}]
        first = audit.off_trace_fingerprint(trace)
        changed = [dict(trace[0], reward="2")]
        self.assertNotEqual(first, audit.off_trace_fingerprint(changed))
        with self.assertRaises(audit.SourceIdentityError):
            audit._expected_no_training_guards({"training_performed": True})

    def test_bad_precheck_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing-source"
            output = Path(temporary) / "audit-output"
            with self.assertRaises(audit.SourceIdentityError):
                audit.run_audit(root, output)
            self.assertFalse(output.exists())

    def test_json_is_finite_and_csv_is_deterministic(self) -> None:
        encoded = audit.safe_json_dumps({"finite": 1.0, "missing": float("nan")})
        self.assertIsNone(json.loads(encoded)["missing"])
        rows = [{"a": 1, "b": True}, {"a": 2, "b": False}]
        self.assertEqual(audit.csv_text(rows), audit.csv_text(rows))

    def test_exact_eight_file_contract_has_no_ninth_artifact(self) -> None:
        self.assertEqual(len(audit.OUTPUT_FILES), 8)
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            audit._write_staged_files(stage, {name: "synthetic\n" for name in audit.OUTPUT_FILES})
            self.assertEqual({path.name for path in stage.iterdir()}, set(audit.OUTPUT_FILES))

    def test_static_module_has_no_runtime_imports_or_environment_step_call(self) -> None:
        path = Path(audit.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        forbidden = ("environment", "simulator", "controller", "learning", "training")
        self.assertFalse(any(any(token in name.lower() for token in forbidden) for name in imported))
        self.assertNotIn(".step(", source)


if __name__ == "__main__":
    unittest.main()
