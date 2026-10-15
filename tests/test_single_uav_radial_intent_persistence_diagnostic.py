from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
import unittest

from src.experiments.single_uav_radial_intent_persistence_diagnostic import (
    ALL_EPISODE_IDS,
    CANONICAL_ARTIFACT_FILES,
    FAILURE_EPISODE_IDS,
    FRAME_SOURCE,
    PARENT_LATERAL_EXIT_PREREGISTRATION_HASH,
    PREPARATION_ROOT,
    PREREGISTRATION_FILE,
    RUNTIME_SOURCE_LOCK_FILE,
    RADIAL_ASSIST_HARD_CAP_MPS,
    RADIAL_COMMAND_DEADBAND_MPS,
    RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED,
    TARGET_FLOOR_MPS,
    _csv_rows,
    _preregistration_payload,
    _type7,
    assert_paired_identity,
    build_off_baseline_contract,
    build_preparation_bundle,
    derive_assistance_cap,
    paired_itt_endpoint_summary,
    validate_off_against_frozen_baseline,
)


PROJECT = Path(__file__).resolve().parents[1]
PRIOR_ROOT = PROJECT / "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
PREPARATION_DIR = PROJECT / PREPARATION_ROOT
_ASSERTIONS = unittest.TestCase()


def _assert_approx(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(f"{actual!r} is not approximately {expected!r}")


def _minimal_baseline_rows() -> list[dict[str, str]]:
    keys = [
        "condition", "episode_index", "episode_id", "scenario_seed", "noise_seed", "step",
        "lateral_exit_assist_enabled", "lateral_exit_assist_active", "estimated_lateral_velocity_mps",
        "goal_recovery_correction_active", "terminal_capture_active", "terminal_v2_governor_active",
        "action_transform_applied", "corrected_delta_v_pre_transform_mps", "corrected_delta_v_pre_safety_mps",
        "safety_projection_applied", "boundary_projection_applied", "applied_delta_v_mps",
        "passed_obstacle", "post_bypass_recovery", "passed_margin_m", "relevant_obstacle_clearance_m",
        "collision", "terminated", "truncated", "stable_goal", "minimum_clearance_m",
    ]
    output = []
    for index, episode_id in enumerate(ALL_EPISODE_IDS, start=1):
        row = {key: "0" for key in keys}
        row.update({
            "condition": "on",
            "episode_index": str(index),
            "episode_id": episode_id,
            "scenario_seed": str(1000 + index),
            "noise_seed": str(2000 + index),
            "step": "0",
        })
        output.append(row)
    return output


def _episode_rows(passed_ids: set[str]) -> list[dict[str, object]]:
    return [
        {
            "episode_id": episode_id,
            "episode_index": index,
            "scenario_seed": index + 1000,
            "noise_seed": index + 2000,
            "scenario_hash": f"hash-{index}",
            "passed_obstacle": int(episode_id in passed_ids),
            "post_bypass_recovery": int(episode_id in passed_ids),
            "collision": 0,
            "minimum_clearance_m": 2.0,
            "stable_goal": int(episode_id.endswith("0008")),
        }
        for index, episode_id in enumerate(ALL_EPISODE_IDS, start=1)
    ]


def _step_rows(passed_ids: set[str], condition: str) -> list[dict[str, object]]:
    output = []
    for index, episode_id in enumerate(ALL_EPISODE_IDS, start=1):
        passed = episode_id in passed_ids
        for step in range(6):
            output.append({
                "episode_id": episode_id,
                "step": step,
                "passed_obstacle": int(passed and step >= 4),
                "post_bypass_recovery": int(passed and step >= 4),
                "estimated_lateral_exit_ready": int(step >= 2),
                "actor_radial_delta_v_mps": -0.02 if step < 4 else 0.03,
                "passed_margin_m": 0.4 if passed and step >= 4 else -0.1,
                "soft_pass_progress": (0.0, 0.3, 0.6, 0.5, 0.7, 0.8)[step],
                "applied_radial_assist_mps": 0.01 if condition == "on" and step == 3 else 0.0,
                "safety_projection_applied": int(step == 1),
                "boundary_projection_applied": int(step == 2),
                "collision": 0,
                "minimum_clearance_m": 2.0,
            })
    return output


def _eligible_cap_rows() -> list[dict[str, str]]:
    actor_radials = (-0.03, -0.02, -0.01, -0.04, -0.05, -0.06, -0.07)
    rows: list[dict[str, str]] = []
    for episode_index, (episode_id, actor_radial) in enumerate(
        zip(FAILURE_EPISODE_IDS, actor_radials), start=1
    ):
        for step in range(5):
            rows.append({
                "condition": "on",
                "episode_index": str(episode_index),
                "episode_id": episode_id,
                "step": str(step),
                "lateral_exit_assist_encounter_step": "0",
                "lateral_exit_assist_frame_source": FRAME_SOURCE,
                "lateral_exit_assist_horizontal_goal_ray_m": "[1.0,0.0,0.0]",
                "lateral_exit_assist_lateral_unit": "[0.0,1.0,0.0]",
                "lateral_exit_assist_frozen_obstacle_available": "1",
                "lateral_exit_assist_frozen_obstacle_id": "frozen-obstacle",
                "relevant_obstacle_id": "frozen-obstacle",
                "estimated_lateral_velocity_mps": "0.0",
                "actor_raw_delta_v_mps": f"[{actor_radial},0.2,-0.1]",
                "signed_axis_radial_unit": "[1.0,0.0,0.0]",
                "actor_raw_radial_delta_v_mps": str(actor_radial),
                "lateral_exit_assist_active": "0",
                "passed_obstacle": "0",
                "passed_condition_now": "False",
                "post_bypass_recovery": "0",
                "goal_recovery_correction_active": "0",
                "terminal_capture_active": "0",
                "terminal_v2_governor_active": "0",
                "terminal_v21_radial_conflict_active": "0",
                "terminal_v22_restorative_active": "0",
                "terminal_v23_authority_active": "False",
                "terminated": "0",
                "truncated": "0",
            })
    return rows


def test_fixed_parent_hash_floor_deadband_cap_and_frame_are_preregistered() -> None:
    assert PARENT_LATERAL_EXIT_PREREGISTRATION_HASH == "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2"
    assert TARGET_FLOOR_MPS == 0.031192317367950993
    assert RADIAL_COMMAND_DEADBAND_MPS == 1e-8
    assert RADIAL_ASSIST_HARD_CAP_MPS == 0.075
    assert FRAME_SOURCE == "encounter_time_ekf_estimated_goal_ray_projected_to_xy"
    _assert_approx(_type7((0.0, 1.0, 2.0, 3.0), 0.25), 0.75)


def test_fixed_all20_itt_population_and_seven_episode_mechanism_subgroup() -> None:
    assert len(ALL_EPISODE_IDS) == 20
    assert FAILURE_EPISODE_IDS == tuple(
        f"test_r99_episode_{index:04d}" for index in (2, 4, 5, 6, 11, 12, 20)
    )
    off = _episode_rows({f"test_r99_episode_{index:04d}" for index in range(1, 12)})
    on = _episode_rows({f"test_r99_episode_{index:04d}" for index in range(1, 12)})
    report = paired_itt_endpoint_summary(
        off_episodes=off,
        on_episodes=on,
        off_steps=_step_rows(set(), "off"),
        on_steps=_step_rows(set(), "on"),
    )
    assert report["episode_count"] == 20
    assert len(report["per_episode_paired_itt"]) == 20
    assert len(report["historical_main_failure_subgroup_descriptive_only"]) == 7
    assert sum(len(value) for value in report["paired_transitions"].values()) == 20
    assert report["paired_transitions"]["OFF pass -> ON pass"] == [
        f"test_r99_episode_{index:04d}" for index in range(1, 12)
    ]


def test_all_four_paired_transition_categories_are_explicit() -> None:
    off_pass = {f"test_r99_episode_{index:04d}" for index in range(1, 11)}
    on_pass = {f"test_r99_episode_{index:04d}" for index in range(2, 12)}
    report = paired_itt_endpoint_summary(
        off_episodes=_episode_rows(off_pass),
        on_episodes=_episode_rows(on_pass),
        off_steps=_step_rows(off_pass, "off"),
        on_steps=_step_rows(on_pass, "on"),
    )
    transitions = report["paired_transitions"]
    assert transitions["OFF fail -> ON pass"] == ["test_r99_episode_0011"]
    assert transitions["OFF pass -> ON fail"] == ["test_r99_episode_0001"]
    assert transitions["OFF pass -> ON pass"] == [
        f"test_r99_episode_{index:04d}" for index in range(2, 11)
    ]
    assert transitions["OFF fail -> ON fail"] == [
        f"test_r99_episode_{index:04d}" for index in range(12, 21)
    ]


def test_itt_endpoint_hierarchy_and_downstream_0008_rule_are_fixed() -> None:
    episodes = _episode_rows(set())
    report = paired_itt_endpoint_summary(
        off_episodes=episodes,
        on_episodes=episodes,
        off_steps=_step_rows(set(), "off"),
        on_steps=_step_rows(set(), "on"),
    )
    assert [item["tier"] for item in report["primary_endpoint_hierarchy"]] == [1, 2, 3]
    assert "episode_0008" in report["episode_0008_role"]
    assert report["stable_goal_role"] == "downstream descriptive endpoint only"
    assert report["per_episode_paired_itt"][7]["off"]["stable_goal_downstream_descriptive"] == 1


def test_off_baseline_contract_exactly_normalizes_condition_and_fails_on_mismatch() -> None:
    source_rows = _minimal_baseline_rows()
    contract = build_off_baseline_contract(source_rows)
    current = [dict(row, condition="off") for row in source_rows]
    result = validate_off_against_frozen_baseline(
        current_rows=current,
        baseline_contract=contract,
    )
    assert result["status"] == "PASS"
    changed = [dict(row) for row in current]
    changed[0]["estimated_lateral_velocity_mps"] = "0.0001"
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, "RADIAL_INTENT_OFF_BASELINE_MISMATCH"):
        validate_off_against_frozen_baseline(current_rows=changed, baseline_contract=contract)
    tiny_change = [dict(row) for row in current]
    tiny_change[0]["estimated_lateral_velocity_mps"] = "0.0000000000001"
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, "RADIAL_INTENT_OFF_BASELINE_MISMATCH"):
        validate_off_against_frozen_baseline(
            current_rows=tiny_change,
            baseline_contract=contract,
        )
    missing_key = [dict(row) for row in current]
    del missing_key[0]["estimated_lateral_velocity_mps"]
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, "RADIAL_INTENT_OFF_BASELINE_MISMATCH"):
        validate_off_against_frozen_baseline(
            current_rows=missing_key,
            baseline_contract=contract,
        )


def test_all20_itt_rejects_duplicate_episode_summary_rows() -> None:
    episodes = _episode_rows(set())
    duplicate = list(episodes) + [dict(episodes[-1])]
    with _ASSERTIONS.assertRaisesRegex(
        AssertionError,
        "RADIAL_INTENT_PRIMARY_POPULATION_MUST_BE_FIXED_ALL_20",
    ):
        paired_itt_endpoint_summary(
            off_episodes=duplicate,
            on_episodes=episodes,
            off_steps=_step_rows(set(), "off"),
            on_steps=_step_rows(set(), "on"),
        )


def test_prior_on_off_baseline_fingerprint_uses_every_ordered_inherited_source_row() -> None:
    source_rows = _csv_rows(PRIOR_ROOT / "on" / "steps.csv")
    contract = build_off_baseline_contract(source_rows)
    assert contract["row_count"] == len(source_rows)
    assert contract["episode_order"] == list(ALL_EPISODE_IDS)
    assert contract["normalized_field"] == "condition"
    assert "lateral_exit_assist_active" in contract["inherited_key_order"]
    assert "goal_recovery_correction_active" in contract["inherited_key_order"]
    assert "action_transform_applied" in contract["inherited_key_order"]
    assert "safety_projection_applied" in contract["inherited_key_order"]
    assert "passed_margin_m" in contract["inherited_key_order"]
    assert not any(key.startswith("radial_intent_") for key in contract["inherited_key_order"])
    assert len(contract["trace_sha256"]) == 64


def test_baseline_contract_requires_inherited_lateral_recovery_transform_projection_and_pass_keys() -> None:
    rows = _minimal_baseline_rows()
    del rows[0]["lateral_exit_assist_active"]
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, "RADIAL_INTENT_OFF_BASELINE_MISMATCH"):
        build_off_baseline_contract(rows)


def test_runtime_identity_requires_lateral_and_goal_recovery_on_in_both_arms() -> None:
    episodes = _episode_rows(set())
    common = {
        "scenario_manifest_hash": "manifest",
        "ordered_pool_identity": [(1, ALL_EPISODE_IDS[0], 2001)],
        "checkpoint": {"sha256": "checkpoint"},
        "policy_hash": "policy",
        "observation_dimension": 42,
        "controller_config_hash": "controller",
        "goal_recovery_config_hash": "recovery",
        "correction_enabled": True,
        "lateral_exit_braking_assistance_enabled": True,
    }
    off = dict(common, radial_intent_assistance_enabled=False)
    on = dict(common, radial_intent_assistance_enabled=True)
    assert_paired_identity(
        off_rows=episodes,
        on_rows=episodes,
        off_component_hashes={"actor": "a", "critic": "c", "gat": "g", "normalizer": "n"},
        on_component_hashes={"actor": "a", "critic": "c", "gat": "g", "normalizer": "n"},
        off_provenance=off,
        on_provenance=on,
    )
    changed_on = [dict(row) for row in episodes]
    changed_on[0]["noise_seed"] = -1
    with _ASSERTIONS.assertRaisesRegex(AssertionError, "RADIAL_INTENT_PAIRED_IDENTITY_MISMATCH"):
        assert_paired_identity(
            off_rows=episodes,
            on_rows=changed_on,
            off_component_hashes={"actor": "a"},
            on_component_hashes={"actor": "a"},
            off_provenance=off,
            on_provenance=on,
        )
    bad_off = dict(off, lateral_exit_braking_assistance_enabled=False)
    with _ASSERTIONS.assertRaisesRegex(AssertionError, "RADIAL_INTENT_LATERAL_BASELINE_NOT_FROZEN_ON"):
        assert_paired_identity(
            off_rows=episodes,
            on_rows=episodes,
            off_component_hashes={"actor": "a"},
            on_component_hashes={"actor": "a"},
            off_provenance=bad_off,
            on_provenance=on,
        )


def test_cap_derivation_fails_closed_without_runtime_equivalent_eligible_rows() -> None:
    rows = []
    for episode_index, episode_id in enumerate(FAILURE_EPISODE_IDS, start=1):
        for step in range(5):
            rows.append({"episode_id": episode_id, "episode_index": str(episode_index), "step": str(step)})
    with _ASSERTIONS.assertRaisesRegex(RuntimeError, RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED):
        derive_assistance_cap(rows)


def test_cap_statistics_are_type7_and_use_the_preregistered_hard_cap_rule() -> None:
    result = derive_assistance_cap(_eligible_cap_rows())
    statistics = result["required_assistance_statistics_mps"]
    _assert_approx(statistics["q1"], TARGET_FLOOR_MPS + 0.025)
    _assert_approx(statistics["median"], TARGET_FLOOR_MPS + 0.04)
    _assert_approx(statistics["q3"], TARGET_FLOOR_MPS + 0.055)
    _assert_approx(statistics["maximum"], TARGET_FLOOR_MPS + 0.07)
    assert result["max_radial_intent_assist_mps"] == RADIAL_ASSIST_HARD_CAP_MPS
    assert result["quantile_method"] == "numpy_linear_type7"
    assert result["fail_closed_policy"] is True
    assert result["fail_closed_triggered"] is False
    assert "fail_closed" not in result
    assert result["sweep_performed"] is False


def test_source_and_prepare_logic_have_no_rollout_training_or_optimizer_calls() -> None:
    source_path = PROJECT / "src/experiments/single_uav_radial_intent_persistence_diagnostic.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_preparation_bundle")
    prepare_source = ast.get_source_segment(source, prepare) or ""
    forbidden = (
        "environment.step(", "env.step(", "rollout(", "training(", "optimizer.step(",
        ".backward(", "simulation(", "formal_run(",
    )
    assert not any(token in prepare_source for token in forbidden)
    assert "environment.step(" not in source
    assert "optimizer.step(" not in source
    assert ".backward(" not in source


def test_prepare_path_is_lazy_and_canonical_artifact_paths_are_exact() -> None:
    source = (PROJECT / "src/experiments/single_uav_radial_intent_persistence_diagnostic.py").read_text(encoding="utf-8")
    prepare_body = source.split("def build_preparation_bundle", maxsplit=1)[1].split("def _csv_text", maxsplit=1)[0]
    assert "single_uav_goal_recovery_correction_evaluation" not in prepare_body
    assert str(PREPARATION_ROOT).replace("\\", "/") == "logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/preparation"
    assert CANONICAL_ARTIFACT_FILES == (
        "preregistered_radial_intent_persistence_endpoints.json",
        "radial_intent_preregistration_lock.json",
        "radial_intent_runtime_source_lock.json",
        "radial_intent_cap_derivation.json",
        "radial_intent_cap_derivation_rows.csv",
        "radial_intent_off_baseline_contract.json",
        "radial_intent_persistence_preparation_summary.json",
    )
    assert '"data_copies"' not in source


def test_preparation_payload_locks_identity_caps_and_no_sweep() -> None:
    cap = {
        "max_radial_intent_assist_mps": 0.05,
        "target_floor_mps": TARGET_FLOOR_MPS,
        "deadband_mps": RADIAL_COMMAND_DEADBAND_MPS,
        "frozen_frame_source": FRAME_SOURCE,
        "cap_rule": "min(Q3_required_assistance, 0.075 m/s)",
        "required_assistance_statistics_mps": {"median": 0.04, "q1": 0.02, "q3": 0.05, "maximum": 0.08},
        "fail_closed_policy": True,
        "fail_closed_triggered": False,
    }
    baseline = {"trace_sha256": "fingerprint", "row_count": 1, "source_steps_sha256": "steps"}
    payload = _preregistration_payload(cap=cap, baseline=baseline, source_hashes={"source.py": "hash"})
    assert payload["parent_lateral_exit_preregistration_hash"] == PARENT_LATERAL_EXIT_PREREGISTRATION_HASH
    assert payload["runtime_definition"]["max_radial_intent_assist_mps"] == 0.05
    assert payload["runtime_definition"]["assistance_cap_derivation"]["no_sweep"] is True
    assert payload["runtime_definition"]["assistance_cap_derivation"]["fail_closed"] is True
    assert payload["runtime_definition"]["assistance_cap_derivation"]["fail_closed_policy"] is True
    assert payload["runtime_definition"]["assistance_cap_derivation"]["fail_closed_triggered_at_preparation"] is False
    assert payload["preparation_only_flags"]["rollout_executed"] is False


def test_preparation_bundle_has_exact_artifact_set_without_data_copy_outputs() -> None:
    # This reads only fixed historical logs and builds in-memory canonical text.
    # The project preparation artifacts are written only by the explicit writer.
    if not PRIOR_ROOT.joinpath("on", "steps.csv").is_file():
        raise unittest.SkipTest("frozen lateral-assistance source log is not present")
    bundle = build_preparation_bundle(PROJECT)
    assert tuple(bundle) == CANONICAL_ARTIFACT_FILES
    assert all(name.startswith("radial_intent_") or name == PREREGISTRATION_FILE for name in bundle)
    assert not any("data" in name.lower() for name in bundle)
    payload = json.loads(bundle["preregistered_radial_intent_persistence_endpoints.json"])
    cap = json.loads(bundle["radial_intent_cap_derivation.json"])
    summary = json.loads(bundle["radial_intent_persistence_preparation_summary.json"])
    prereg_lock = json.loads(bundle["radial_intent_preregistration_lock.json"])
    runtime_lock = json.loads(bundle["radial_intent_runtime_source_lock.json"])
    assert payload["population"]["ordered_episode_ids"] == list(ALL_EPISODE_IDS)
    assert payload["runtime_definition"]["target_floor_mps"] == TARGET_FLOOR_MPS
    assert payload["primary_endpoint_hierarchy"][0]["tier"] == 1
    cap_contract = payload["runtime_definition"]["assistance_cap_derivation"]
    assert cap_contract["fail_closed"] is True
    assert cap_contract["fail_closed_policy"] is True
    assert cap_contract["fail_closed_triggered_at_preparation"] is False
    assert cap["fail_closed_policy"] is True
    assert cap["fail_closed_triggered"] is False
    assert "fail_closed" not in cap
    assert summary["cap_derivation_fail_closed_policy"] is True
    assert summary["cap_derivation_fail_closed_triggered"] is False
    for lock in (prereg_lock, runtime_lock):
        assert lock["cap_derivation_fail_closed_policy"] is True
        assert lock["cap_derivation_fail_closed_triggered"] is False


def test_existing_off_contract_if_present_has_exact_prior_fingerprint_and_hash_source() -> None:
    contract_path = PREPARATION_DIR / "radial_intent_off_baseline_contract.json"
    if not contract_path.is_file():
        raise unittest.SkipTest("canonical preparation artifacts have not been written")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = _csv_rows(PRIOR_ROOT / "on" / "steps.csv")
    assert contract["source_steps_sha256"]
    assert contract["row_count"] == len(rows)
    assert contract["baseline_condition"].startswith("prior lateral-exit assistance ON")
    assert contract["normalized_field"] == "condition"
    assert contract["trace_sha256"]


def test_preregistration_hash_lock_is_new_and_source_hashes_are_noncyclic_if_present() -> None:
    lock_path = PREPARATION_DIR / "radial_intent_preregistration_lock.json"
    runtime_path = PREPARATION_DIR / "radial_intent_runtime_source_lock.json"
    if not lock_path.is_file() or not runtime_path.is_file():
        raise unittest.SkipTest("canonical preparation artifacts have not been written")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    new_hash = lock["active_radial_intent_preregistration_hash"]
    assert new_hash != PARENT_LATERAL_EXIT_PREREGISTRATION_HASH
    assert runtime["new_radial_intent_preregistration_hash"] == new_hash
    assert RUNTIME_SOURCE_LOCK_FILE not in runtime["source_hashes"]


class SingleUavRadialIntentPersistenceDiagnosticTests(unittest.TestCase):
    """Expose the offline preparation checks to standard unittest discovery."""

    def test_fixed_parent_hash_floor_deadband_cap_and_frame_are_preregistered(self) -> None:
        test_fixed_parent_hash_floor_deadband_cap_and_frame_are_preregistered()

    def test_fixed_all20_itt_population_and_seven_episode_mechanism_subgroup(self) -> None:
        test_fixed_all20_itt_population_and_seven_episode_mechanism_subgroup()

    def test_all_four_paired_transition_categories_are_explicit(self) -> None:
        test_all_four_paired_transition_categories_are_explicit()

    def test_itt_endpoint_hierarchy_and_downstream_0008_rule_are_fixed(self) -> None:
        test_itt_endpoint_hierarchy_and_downstream_0008_rule_are_fixed()

    def test_off_baseline_contract_exactly_normalizes_condition_and_fails_on_mismatch(self) -> None:
        test_off_baseline_contract_exactly_normalizes_condition_and_fails_on_mismatch()

    def test_all20_itt_rejects_duplicate_episode_summary_rows(self) -> None:
        test_all20_itt_rejects_duplicate_episode_summary_rows()

    def test_prior_on_off_baseline_fingerprint_uses_every_ordered_inherited_source_row(self) -> None:
        test_prior_on_off_baseline_fingerprint_uses_every_ordered_inherited_source_row()

    def test_baseline_contract_requires_inherited_lateral_recovery_transform_projection_and_pass_keys(self) -> None:
        test_baseline_contract_requires_inherited_lateral_recovery_transform_projection_and_pass_keys()

    def test_runtime_identity_requires_lateral_and_goal_recovery_on_in_both_arms(self) -> None:
        test_runtime_identity_requires_lateral_and_goal_recovery_on_in_both_arms()

    def test_cap_derivation_fails_closed_without_runtime_equivalent_eligible_rows(self) -> None:
        test_cap_derivation_fails_closed_without_runtime_equivalent_eligible_rows()

    def test_cap_statistics_are_type7_and_use_the_preregistered_hard_cap_rule(self) -> None:
        test_cap_statistics_are_type7_and_use_the_preregistered_hard_cap_rule()

    def test_source_and_prepare_logic_have_no_rollout_training_or_optimizer_calls(self) -> None:
        test_source_and_prepare_logic_have_no_rollout_training_or_optimizer_calls()

    def test_prepare_path_is_lazy_and_canonical_artifact_paths_are_exact(self) -> None:
        test_prepare_path_is_lazy_and_canonical_artifact_paths_are_exact()

    def test_preparation_payload_locks_identity_caps_and_no_sweep(self) -> None:
        test_preparation_payload_locks_identity_caps_and_no_sweep()

    def test_preparation_bundle_has_exact_artifact_set_without_data_copy_outputs(self) -> None:
        test_preparation_bundle_has_exact_artifact_set_without_data_copy_outputs()

    def test_existing_off_contract_if_present_has_exact_prior_fingerprint_and_hash_source(self) -> None:
        test_existing_off_contract_if_present_has_exact_prior_fingerprint_and_hash_source()

    def test_preregistration_hash_lock_is_new_and_source_hashes_are_noncyclic_if_present(self) -> None:
        test_preregistration_hash_lock_is_new_and_source_hashes_are_noncyclic_if_present()
