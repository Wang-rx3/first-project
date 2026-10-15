"""Pure tests for the encounter-relative lateral exit authority audit."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from src.experiments.single_uav_lateral_exit_authority_audit import (
    DEFAULT_NEAR_ZERO_LATERAL_THRESHOLD_MPS,
    MAIN_FAILURE_IDS,
    REFERENCE_IDS,
    SPECIAL_FAILURE_IDS,
    analyze_episode,
    assert_source_hashes_unchanged,
    closest_approach_step,
    compare_to_reference,
    counter_impulses,
    decay_statistics,
    diagnose_episode,
    encounter_slice,
    extra_drift,
    first_countersteer_step,
    first_sustained_clearance_increase_step,
    first_sustained_countersteer_step,
    initial_nonzero_sign,
    pipeline_retention,
    require_unique_episode_ids,
    response_delay_steps,
    sign_change_count,
    SourceIdentityError,
    success_reference_statistics,
    validate_transition_rows,
)


PROJECT = Path(__file__).resolve().parents[1]
AUDIT_SOURCE = PROJECT / "src/experiments/single_uav_lateral_exit_authority_audit.py"


def row(
    step: int,
    *,
    actor: float = 0.0,
    actual: float = 1.0,
    clearance: float = 1.0,
    increasing: bool = False,
    displacement: float = 0.0,
    post_transform: float | None = None,
    pre_safety: float | None = None,
    post_boundary: float | None = None,
    post_safety: float | None = None,
    final: float | None = None,
) -> dict[str, object]:
    return {
        "step": step,
        "actor_raw_lateral_delta_v_mps": actor,
        "actual_lateral_velocity_mps": actual,
        "signed_obstacle_clearance_m": clearance,
        "clearance_increasing": increasing,
        "past_closest_approach": increasing,
        "actual_lateral_displacement_from_encounter_m": displacement,
        "post_transform_lateral_delta_v_mps": actor if post_transform is None else post_transform,
        "pre_safety_lateral_delta_v_mps": actor if pre_safety is None else pre_safety,
        "post_boundary_lateral_delta_v_mps": actor if post_boundary is None else post_boundary,
        "post_safety_lateral_delta_v_mps": actor if post_safety is None else post_safety,
        "final_applied_lateral_delta_v_mps": actor if final is None else final,
        "passed_condition_now": False,
        "passed_obstacle": False,
        "passed_margin_m": float(step),
        "safety_projection_applied": False,
        "boundary_projection_applied": False,
    }


class LateralExitAuthorityAuditTests(unittest.TestCase):
    def test_encounter_slicing_is_inclusive(self) -> None:
        rows = [row(step) for step in range(6)]
        self.assertEqual([item["step"] for item in encounter_slice(rows, 3)], [3, 4, 5])

    def test_closest_approach_is_encounter_relative_argmin(self) -> None:
        rows = [row(5, clearance=2.0), row(6, clearance=0.5), row(7, clearance=1.0)]
        self.assertEqual(closest_approach_step(rows), 6)

    def test_closest_approach_tie_uses_earliest_step(self) -> None:
        rows = [row(5, clearance=0.5), row(6, clearance=0.5)]
        self.assertEqual(closest_approach_step(rows), 5)

    def test_sustained_clearance_requires_three_steps(self) -> None:
        rows = [row(1, increasing=True), row(2, increasing=True), row(3, increasing=False)]
        self.assertIsNone(first_sustained_clearance_increase_step(rows))
        rows.append(row(4, increasing=True))
        rows.append(row(5, increasing=True))
        rows.append(row(6, increasing=True))
        self.assertEqual(first_sustained_clearance_increase_step(rows), 4)

    def test_initial_actual_sign_ignores_deadband(self) -> None:
        rows = [row(1, actual=0.0), row(2, actual=-0.4)]
        self.assertEqual(initial_nonzero_sign(rows, "actual_lateral_velocity_mps"), -1)

    def test_countersteer_is_opposite_initial_actual_sign(self) -> None:
        rows = [row(1, actor=-0.1), row(2, actor=0.2)]
        self.assertEqual(first_countersteer_step(rows, -1), 2)

    def test_sustained_countersteer_is_four_of_five(self) -> None:
        rows = [row(i, actor=value) for i, value in enumerate([0.2, 0.2, -0.1, 0.2, 0.2])]
        self.assertEqual(first_sustained_countersteer_step(rows, -1), 0)

    def test_sustained_countersteer_returns_first_counter_row(self) -> None:
        rows = [row(i, actor=value) for i, value in enumerate([-0.1, 0.2, 0.2, 0.2, 0.2])]
        self.assertEqual(first_sustained_countersteer_step(rows, -1), 1)

    def test_counter_impulse_direction_is_positive_magnitude(self) -> None:
        values = [row(0, actor=0.3, final=0.2), row(1, actor=-0.1, final=-0.05)]
        result = counter_impulses(values, -1)
        self.assertAlmostEqual(result["cumulative_actor_counter_lateral_delta_v_mps"], 0.3)
        self.assertAlmostEqual(result["net_actor_counter_lateral_impulse_mps"], 0.2)

    def test_same_direction_reacceleration_is_separate(self) -> None:
        values = [row(0, actor=0.4), row(1, actor=-0.25)]
        result = counter_impulses(values, -1)
        self.assertAlmostEqual(result["cumulative_actor_same_direction_lateral_delta_v_mps"], 0.25)

    def test_pipeline_retention_uses_stage_sums(self) -> None:
        values = [row(0, actor=0.4, post_transform=0.3, pre_safety=0.3, post_boundary=0.24, post_safety=0.12, final=0.12)]
        result = pipeline_retention(values, -1)
        self.assertAlmostEqual(result["actor_to_final_retention"], 0.3)
        self.assertAlmostEqual(result["transform_retention"], 0.75)
        self.assertAlmostEqual(result["boundary_retention"], 0.8)
        self.assertAlmostEqual(result["safety_retention"], 0.5)

    def test_near_zero_threshold_is_fixed(self) -> None:
        self.assertEqual(DEFAULT_NEAR_ZERO_LATERAL_THRESHOLD_MPS, 0.5)

    def test_decay_rate_is_speed_drop_per_step(self) -> None:
        result = decay_statistics([row(0, actual=1.0), row(1, actual=0.7), row(2, actual=0.5)])
        self.assertAlmostEqual(result["mean_lateral_speed_decay_mps_per_step"], 0.25)

    def test_response_delay_requires_three_decreases(self) -> None:
        rows = [row(i, actual=value) for i, value in enumerate([1.0, 1.1, 1.0, 0.9, 0.8])]
        self.assertEqual(response_delay_steps(rows, 0), 1)

    def test_response_delay_returns_none_without_sustained_drop(self) -> None:
        rows = [row(i, actual=value) for i, value in enumerate([1.0, 0.9, 1.0, 0.8])]
        self.assertIsNone(response_delay_steps(rows, 0))

    def test_extra_drift_is_projected_to_initial_direction(self) -> None:
        rows = [row(0, displacement=1.0), row(1, displacement=3.5)]
        initial_direction, signed_axis = extra_drift(rows, -1)
        self.assertAlmostEqual(signed_axis, 2.5)
        self.assertAlmostEqual(initial_direction, -2.5)

    def test_sign_changes_ignore_center_samples(self) -> None:
        self.assertEqual(sign_change_count([1.0, 0.0, -1.0, 0.0, 1.0]), 2)

    def test_reference_uses_only_success_ids(self) -> None:
        diagnostics = [
            {"episode_id": REFERENCE_IDS[0], "x": 1.0},
            {"episode_id": REFERENCE_IDS[1], "x": 3.0},
            {"episode_id": MAIN_FAILURE_IDS[0], "x": 100.0},
        ]
        result = success_reference_statistics(diagnostics, metrics=("x",))["x"]
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["median"], 2.0)

    def test_iqr_comparison(self) -> None:
        reference = {"q1": 2.0, "q3": 4.0}
        self.assertEqual(compare_to_reference(1.0, reference), "BELOW_REFERENCE_IQR")
        self.assertEqual(compare_to_reference(3.0, reference), "WITHIN_REFERENCE_IQR")
        self.assertEqual(compare_to_reference(5.0, reference), "ABOVE_REFERENCE_IQR")

    def test_whole_episode_label_is_not_diagnosis_input(self) -> None:
        source = AUDIT_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("signed_failure_label", source)

    def test_special_cases_are_excluded_from_main_cluster(self) -> None:
        references = {metric: {"q1": 0.0, "q3": 1.0} for metric in (
            "actor_to_final_retention", "mean_actor_countersteer_magnitude_mps",
            "net_actor_counter_lateral_impulse_mps", "mean_lateral_speed_decay_mps_per_step",
            "steps_to_near_zero_after_countersteer", "extra_lateral_drift_after_countersteer_m",
        )}
        diagnostic = {"episode_id": SPECIAL_FAILURE_IDS[0]}
        primary, _, _ = diagnose_episode(diagnostic, references)
        self.assertEqual(primary, "SAFETY_DOMINATED_SPECIAL_CASE")

    def test_exact_actor_to_final_retention_is_not_attenuation(self) -> None:
        references = {
            "actor_to_final_retention": {"q1": 1.1, "q3": 2.0},
            "mean_actor_countersteer_magnitude_mps": {"q1": 0.01, "q3": 0.10},
            "net_actor_counter_lateral_impulse_mps": {"q1": 0.01, "q3": 1.0},
            "mean_lateral_speed_decay_mps_per_step": {"q1": 0.01, "q3": 0.20},
            "steps_to_near_zero_after_countersteer": {"q1": 0.0, "q3": 20.0},
            "extra_lateral_drift_after_countersteer_m": {"q1": 0.0, "q3": 5.0},
        }
        diagnostic = {
            "episode_id": MAIN_FAILURE_IDS[0],
            "initial_actual_lateral_sign": -1,
            "closest_approach_step": 10,
            "first_sustained_countersteer_step": 11,
            "first_countersteer_step": 11,
            "post_closest_actor_countersteer_fraction": 1.0,
            "post_closest_same_direction_reacceleration_fraction": 0.0,
            "actor_sign_changes_after_closest_approach": 0,
            "actor_to_final_retention": 1.0,
            "mean_actor_countersteer_magnitude_mps": 0.05,
            "net_actor_counter_lateral_impulse_mps": 0.5,
            "mean_lateral_speed_decay_mps_per_step": 0.05,
            "steps_to_near_zero_after_countersteer": 5,
            "extra_lateral_drift_after_countersteer_m": 1.0,
            "near_zero_censored": 0,
        }
        _, _, gates = diagnose_episode(diagnostic, references)
        self.assertFalse(gates["pipeline_attenuation"])

    def test_source_hash_guard_accepts_unchanged(self) -> None:
        assert_source_hashes_unchanged({"a": "x"}, {"a": "x"})

    def test_source_hash_guard_rejects_change(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_source_hashes_unchanged({"a": "x"}, {"a": "y"})

    def test_duplicate_episode_identity_fails_closed(self) -> None:
        rows = [{"episode_id": "episode-a"}, {"episode_id": "episode-a"}]
        with self.assertRaises(SourceIdentityError):
            require_unique_episode_ids(rows, artifact_name="episodes.csv")

    def test_transition_validation_rejects_step_gap(self) -> None:
        template = {
            "episode_index": "1",
            "episode_id": "episode-a",
            "actor_lateral_sign": "1",
            "actual_lateral_sign": "1",
            "actor_lateral_sign_changes": "0",
            "actual_lateral_sign_changes": "0",
            "actor_same_side_run_length": "1",
            "actual_same_side_run_length": "1",
            "actor_actual_lateral_side_agree_now": "true",
            "actor_actual_lateral_side_agreement_fraction": "1",
            "clearance_increasing": "true",
            "past_closest_approach": "true",
        }
        rows = [{**template, "step": "1"}, {**template, "step": "3"}]
        with self.assertRaises(SourceIdentityError):
            validate_transition_rows(rows, expected_ids={"episode-a"})

    def test_source_file_hash_is_stable_under_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            _ = path.read_text(encoding="utf-8")
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_audit_script_does_not_import_environment_runtime(self) -> None:
        source = AUDIT_SOURCE.read_text(encoding="utf-8")
        for forbidden in ("src.simulation", ".formal import", "_make_environment", "torch", "pandas"):
            self.assertNotIn(forbidden, source)

    def test_audit_script_does_not_call_step(self) -> None:
        source = AUDIT_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("environment.step(", source)
        self.assertNotIn("env.step(", source)

    def test_analyze_episode_uses_encounter_relative_signs(self) -> None:
        rows = [row(i, actor=-0.2, actual=-1.0 + 0.2 * i, clearance=abs(i - 3) + 1.0) for i in range(6)]
        for item in rows:
            item["signed_encounter_step"] = 2
        summary = {"signed_encounter_step": 2, "passed_obstacle": False, "post_bypass_recovery": False}
        result = analyze_episode("test_r99_episode_0002", rows, summary, near_zero_threshold_mps=0.25)
        self.assertEqual(result["encounter_step"], 2)
        self.assertEqual(result["initial_actual_lateral_sign"], -1)

    def test_missing_reference_value_is_not_comparable(self) -> None:
        self.assertEqual(compare_to_reference(math.nan, {"q1": 1.0, "q3": 2.0}), "NOT_COMPARABLE")


if __name__ == "__main__":
    unittest.main()
