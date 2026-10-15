"""Pure contract tests for the prepared paired lateral-exit diagnostic."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import unittest

from src.experiments.single_uav_lateral_exit_braking_assistance import (
    ASSIST_CAP_MPS,
    FROZEN_OFF_BASELINE_FINGERPRINT,
    NOT_INDEPENDENTLY_VERIFIED,
    OFF_BASELINE_MISMATCH,
    PREREGISTERED_ENDPOINTS,
    SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH,
    aggregate_episode_endpoints,
    assert_paired_identity_report,
    build_paired_identity_report,
    build_paired_endpoint_rows,
    build_parser,
    off_trace_fingerprint,
    preregistration_hash,
    validate_off_against_frozen_baseline,
)


PROJECT = Path(__file__).resolve().parents[1]


def episode(index: int) -> dict[str, object]:
    return {
        "episode_index": index,
        "episode_id": f"test_r99_episode_{index:04d}",
        "scenario_seed": 1000 + index,
        "noise_seed": 2000 + index,
        "scenario_hash": f"hash-{index}",
        "max_steps": 120,
    }


def step(ep: dict[str, object], index: int, *, eligible: bool = False) -> dict[str, object]:
    return {
        **ep,
        "step": index,
        "actor_raw_delta_v_mps": [0.0, -0.1, 0.0],
        "goal_recovery_correction_enabled": 1,
        "goal_recovery_correction_eligible": 0,
        "goal_recovery_correction_active": 0,
        "goal_recovery_correction_reason": "not_post_bypass_recovery",
        "goal_recovery_correction_alpha": 0.0,
        "goal_recovery_correction_delta_v_mps": [0.0, 0.0, 0.0],
        "goal_recovery_applied_delta_v_mps": [0.0, 0.0, 0.0],
        "corrected_delta_v_pre_transform_mps": [0.0, -0.1, 0.0],
        "corrected_delta_v_pre_safety_mps": [0.0, -0.1, 0.0],
        "applied_delta_v_mps": [0.0, -0.1, 0.0],
        "motion_boundary_delta_v_mps": [0.0, 0.0, 0.0],
        "safety_projection_delta_v_mps": [0.0, 0.0, 0.0],
        "safety_projection_applied": 0,
        "boundary_projection_applied": 0,
        "action_clipped": 0,
        "action_transform_applied": 0,
        "estimated_radial_velocity_mps": 1.0,
        "estimated_lateral_velocity_mps": 1.0,
        "estimated_vertical_velocity_mps": 0.0,
        "estimated_radial_displacement_step_m": 0.2,
        "estimated_lateral_displacement_step_m": 0.2,
        "estimated_vertical_displacement_step_m": 0.0,
        "goal_distance_m": 10.0 - index,
        "goal_distance_after_m": 9.8 - index,
        "passed_obstacle": 0,
        "post_bypass_recovery": 0,
        "stable_goal": 0,
        "reward": 1.0,
        "collision": 0,
        "terminated": 0,
        "truncated": 0,
        "termination_reason": "running",
        "lateral_exit_assist_enabled": 0,
        "lateral_exit_assist_eligible": int(eligible),
        "lateral_exit_assist_encounter_step": 0,
        "lateral_exit_assist_initial_lateral_sign": 1,
        "lateral_exit_assist_minimum_clearance_m": 2.0,
        "lateral_exit_outcome_actual_lateral_velocity_mps": max(1.0 - 0.3 * index, 0.0),
        "lateral_exit_outcome_actual_lateral_displacement_from_encounter_m": 0.2 * index,
    }


def identity_result(
    episodes: list[dict[str, object]], *, enabled: bool
) -> dict[str, object]:
    return {
        "episodes": episodes,
        "provenance": {
            "checkpoint": {"sha256": "checkpoint-sha"},
            "policy_hash": "policy-hash",
            "observation_dimension": 14,
            "scenario_manifest_hash": "manifest-hash",
            "controller_config_hash": "controller-config-hash",
            "goal_recovery_config_hash": "goal-recovery-config-hash",
            "lateral_exit_treatment_config": {
                "enabled": enabled,
                "assist_cap_mps": ASSIST_CAP_MPS,
            },
            **(
                {
                    "off_baseline_check": {
                        "status": "PASS",
                        "fingerprint": FROZEN_OFF_BASELINE_FINGERPRINT,
                    }
                }
                if not enabled else {}
            ),
        },
    }


def valid_identity_report() -> dict[str, object]:
    episodes = [episode(index) for index in range(1, 21)]
    frozen = {"actor": "a", "critic": "c", "gat": "g", "policy": "p"}
    return build_paired_identity_report(
        off_result=identity_result(episodes, enabled=False),
        on_result=identity_result(copy.deepcopy(episodes), enabled=True),
        runtime_hashes={
            "before_off_or_first_condition": frozen,
            "after_off": copy.deepcopy(frozen),
            "after_on": copy.deepcopy(frozen),
        },
        expected_observation_dimension=14,
    )


class LateralExitBrakingDiagnosticContractTests(unittest.TestCase):
    def test_cli_conditions_are_locked(self) -> None:
        self.assertEqual(build_parser().parse_args([]).condition, "paired")
        self.assertEqual(build_parser().parse_args(["--condition", "off"]).condition, "off")
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--condition", "sweep"])

    def test_assistance_columns_do_not_change_off_fingerprint(self) -> None:
        rows = [step(episode(1), 0)]
        changed = copy.deepcopy(rows)
        changed[0]["lateral_exit_assist_inactive_reason"] = "new_logging_only"
        self.assertEqual(off_trace_fingerprint(rows), off_trace_fingerprint(changed))

    def test_behavior_change_does_change_off_fingerprint(self) -> None:
        rows = [step(episode(1), 0)]
        changed = copy.deepcopy(rows)
        changed[0]["applied_delta_v_mps"] = [0.0, -0.2, 0.0]
        self.assertNotEqual(off_trace_fingerprint(rows), off_trace_fingerprint(changed))

    def test_off_mismatch_uses_exact_fail_fast_token(self) -> None:
        episodes = [episode(index) for index in range(1, 21)]
        baseline_steps = [step(item, 0) for item in episodes]
        current_steps = copy.deepcopy(baseline_steps)
        current_steps[0]["reward"] = 2.0
        with self.assertRaisesRegex(RuntimeError, f"^{OFF_BASELINE_MISMATCH}$"):
            validate_off_against_frozen_baseline(
                current_episodes=episodes,
                current_steps=current_steps,
                baseline={"episodes": episodes, "steps": baseline_steps},
            )

    def test_threshold_endpoint_and_censoring(self) -> None:
        rows = [step(episode(1), index, eligible=index >= 2) for index in range(4)]
        metrics = aggregate_episode_endpoints(rows, anchor_step=2)
        self.assertEqual(metrics["steps_to_lateral_exit_threshold_0p5_mps"], 2)
        self.assertEqual(metrics["lateral_exit_threshold_0p5_mps_censored"], 0)
        self.assertEqual(metrics["anchor_step"], 2)

    def test_primary_excursion_name_matches_initial_direction_semantics(self) -> None:
        rows = [step(episode(1), index) for index in range(3)]
        rows[0]["lateral_exit_outcome_actual_lateral_displacement_from_encounter_m"] = 0.0
        rows[1]["lateral_exit_outcome_actual_lateral_displacement_from_encounter_m"] = 0.7
        rows[2]["lateral_exit_outcome_actual_lateral_displacement_from_encounter_m"] = -1.2
        metrics = aggregate_episode_endpoints(rows)
        name = "maximum_initial_direction_lateral_excursion_from_encounter_m"
        self.assertIn(name, PREREGISTERED_ENDPOINTS["primary"])
        self.assertNotIn(
            "maximum_actual_lateral_excursion_from_encounter_m",
            PREREGISTERED_ENDPOINTS["primary"],
        )
        self.assertEqual(metrics[name], 0.7)

    def test_secondary_threshold_is_measured_from_common_anchor(self) -> None:
        rows = [step(episode(1), index, eligible=index >= 2) for index in range(5)]
        velocities = [0.4, 0.3, 1.0, 0.8, 0.4]
        for row, velocity in zip(rows, velocities):
            row["lateral_exit_outcome_actual_lateral_velocity_mps"] = velocity
        metrics = aggregate_episode_endpoints(
            rows, anchor_step=2, common_end_step=4
        )
        self.assertEqual(metrics["steps_to_lateral_exit_threshold_0p5_mps"], 0)
        self.assertEqual(
            metrics["steps_to_lateral_exit_threshold_0p5_mps_after_common_anchor"],
            2,
        )
        self.assertEqual(
            metrics["after_common_anchor_threshold_status"],
            "THRESHOLD_REACHED_WITHIN_COMMON_WINDOW",
        )

    def test_off_only_anchor_is_used_for_on(self) -> None:
        episodes = [episode(index) for index in range(1, 21)]
        off_steps = []
        on_steps = []
        for item in episodes:
            off_steps.extend(step(item, idx, eligible=idx >= 2) for idx in range(4))
            on_steps.extend(step(item, idx, eligible=idx >= 1) for idx in range(4))
        pairs = build_paired_endpoint_rows(episodes, episodes, off_steps, on_steps)
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(row["off_defined_anchor_step"] == 2 for row in pairs))

    def test_continuous_secondary_uses_same_common_end(self) -> None:
        episodes = [episode(index) for index in range(1, 21)]
        off_steps: list[dict[str, object]] = []
        on_steps: list[dict[str, object]] = []
        for item in episodes:
            off_rows = [step(item, idx, eligible=idx >= 1) for idx in range(6)]
            on_rows = [step(item, idx, eligible=idx >= 1) for idx in range(4)]
            for row in off_rows + on_rows:
                row["lateral_exit_outcome_actual_lateral_velocity_mps"] = 1.0
            off_steps.extend(off_rows)
            on_steps.extend(on_rows)
        pair = build_paired_endpoint_rows(
            episodes, copy.deepcopy(episodes), off_steps, on_steps
        )[0]
        self.assertEqual(pair["common_window_end_step"], 3)
        self.assertAlmostEqual(
            pair["integrated_abs_lateral_velocity_after_assist_OFF"], 0.6
        )
        self.assertAlmostEqual(
            pair["integrated_abs_lateral_velocity_after_assist_ON"], 0.6
        )

    def test_right_censoring_occurs_at_common_window_end(self) -> None:
        rows = [step(episode(1), index, eligible=index >= 1) for index in range(5)]
        for row in rows:
            row["lateral_exit_outcome_actual_lateral_velocity_mps"] = 0.8
        metrics = aggregate_episode_endpoints(
            rows, anchor_step=1, common_end_step=3
        )
        self.assertEqual(metrics["after_common_anchor_threshold_censored"], 1)
        self.assertEqual(
            metrics["after_common_anchor_threshold_status"],
            "RIGHT_CENSORED_AT_COMMON_WINDOW_END",
        )
        self.assertEqual(metrics["common_window_end_step"], 3)

    def test_on_termination_before_off_anchor_has_distinct_status(self) -> None:
        episodes = [episode(index) for index in range(1, 21)]
        off_steps: list[dict[str, object]] = []
        on_steps: list[dict[str, object]] = []
        for item in episodes:
            off_steps.extend(step(item, idx, eligible=idx >= 3) for idx in range(5))
            on_steps.extend(step(item, idx, eligible=False) for idx in range(2))
        pair = build_paired_endpoint_rows(
            episodes, copy.deepcopy(episodes), off_steps, on_steps
        )[0]
        self.assertEqual(
            pair["triggered_cohort_status"], "ON_TERMINATED_BEFORE_OFF_ANCHOR"
        )
        self.assertEqual(pair["common_window_end_step"], "")

    def test_system_endpoints_use_full_episode_beyond_common_window(self) -> None:
        rows = [step(episode(1), index, eligible=index >= 1) for index in range(6)]
        rows[-1]["passed_obstacle"] = 1
        rows[-1]["post_bypass_recovery"] = 1
        rows[-1]["stable_goal"] = 1
        metrics = aggregate_episode_endpoints(
            rows, anchor_step=1, common_end_step=3
        )
        self.assertEqual(metrics["passed_obstacle"], 1)
        self.assertEqual(metrics["post_bypass_recovery"], 1)
        self.assertEqual(metrics["stable_goal"], 1)

    def test_identity_status_is_evidence_based_and_unverified_is_not_a_guard(self) -> None:
        report = valid_identity_report()
        statuses = report["identity_status"]
        guards = report["fail_fast_boolean_guards"]
        self.assertEqual(statuses["reward_identity"], NOT_INDEPENDENTLY_VERIFIED)
        self.assertNotIn("reward_identity", guards)
        self.assertTrue(report["only_lateral_exit_braking_assistance_changed"])

    def test_nonempty_status_string_cannot_pass_boolean_guard(self) -> None:
        report = valid_identity_report()
        report["fail_fast_boolean_guards"]["same_policy_hash"] = (
            NOT_INDEPENDENTLY_VERIFIED
        )
        with self.assertRaisesRegex(
            RuntimeError, "^LATERAL_EXIT_PAIRED_IDENTITY_MISMATCH$"
        ):
            assert_paired_identity_report(report)

    def test_only_treatment_changed_is_derived_from_exact_treatment_diff(self) -> None:
        episodes = [episode(index) for index in range(1, 21)]
        off_result = identity_result(episodes, enabled=False)
        on_result = identity_result(copy.deepcopy(episodes), enabled=True)
        on_result["provenance"]["lateral_exit_treatment_config"]["assist_cap_mps"] = 0.1
        frozen = {"actor": "same"}
        report = build_paired_identity_report(
            off_result=off_result,
            on_result=on_result,
            runtime_hashes={
                "before_off_or_first_condition": frozen,
                "after_off": copy.deepcopy(frozen),
                "after_on": copy.deepcopy(frozen),
            },
            expected_observation_dimension=14,
        )
        self.assertFalse(report["only_lateral_exit_braking_assistance_changed"])

    def test_assistance_helper_source_is_frozen(self) -> None:
        source = (PROJECT / "src/control/lateral_exit_braking_assistance.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "19a729bed2cd301f216f94f770c19e9802a8c38dcba2b6ca5e5bac914fc3c383",
        )

    def test_frozen_off_baseline_fingerprint_constant_is_unchanged(self) -> None:
        self.assertEqual(
            FROZEN_OFF_BASELINE_FINGERPRINT,
            "d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b",
        )

    def test_preregistration_hash_is_stable(self) -> None:
        self.assertEqual(
            preregistration_hash(),
            "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2",
        )
        self.assertEqual(
            SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH,
            "6804dc2e8207da57bbcfb202a7fb936379ea5a815adae771b74ceefd060a1679",
        )

    def test_preparation_modules_do_not_call_runtime(self) -> None:
        for relative in (
            "src/control/lateral_exit_braking_assistance.py",
            "src/experiments/single_uav_lateral_exit_braking_assistance.py",
        ):
            source = (PROJECT / relative).read_text(encoding="utf-8")
            self.assertNotIn("environment.step(", source)
            self.assertNotIn("env.step(", source)
            self.assertNotIn("actual_velocity_mps:", source)
            self.assertNotIn("actual_position_m:", source)


if __name__ == "__main__":
    unittest.main()
