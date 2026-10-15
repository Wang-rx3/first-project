from __future__ import annotations

import ast
import csv
import io
import json
import sys
import types
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENTS_PACKAGE = "src.experiments"
if EXPERIMENTS_PACKAGE not in sys.modules:
    package = types.ModuleType(EXPERIMENTS_PACKAGE)
    package.__path__ = [str(PROJECT / "src" / "experiments")]
    sys.modules[EXPERIMENTS_PACKAGE] = package

MODULE_NAME = "src.experiments.single_uav_pass_commit_diagnostic"
_spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
    MODULE_NAME, PROJECT / "src" / "experiments" / "single_uav_pass_commit_diagnostic.py"
)
pass_commit_diagnostic = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(_spec)
sys.modules[MODULE_NAME] = pass_commit_diagnostic
_spec.loader.exec_module(pass_commit_diagnostic)


def _minimal_contract_inputs():
    d = pass_commit_diagnostic
    step_rows = []
    episode_rows = []
    ordered_pool_identity = []
    for index, episode_id in enumerate(d.ALL_EPISODE_IDS, start=1):
        scenario_seed = 4020000 + index
        noise_seed = 4220000 + index
        scenario_hash = f"scenario-{index}"
        step_rows.append({
            "condition": "on",
            "episode_index": str(index),
            "episode_id": episode_id,
            "scenario_seed": str(scenario_seed),
            "noise_seed": str(noise_seed),
            "step": "0",
            "lateral_exit_assist_active": "0",
            "passed_margin_m": "-1.25",
            "passed_obstacle": "0",
            "action_transform_applied": "0",
            "safety_projection_applied": "0",
            "boundary_projection_applied": "0",
            "minimum_clearance_m": "3.0",
        })
        episode_rows.append({
            "condition": "on",
            "episode_index": str(index),
            "episode_id": episode_id,
            "scenario_seed": str(scenario_seed),
            "noise_seed": str(noise_seed),
            "scenario_hash": scenario_hash,
            "passed_obstacle": "0",
            "post_bypass_recovery": "0",
            "collision": "0",
        })
        ordered_pool_identity.append([index, episode_id, noise_seed])
    aggregate = {"episodes": 20, "passed_obstacle": 0, "collision": 0}
    summary = {
        "results": {
            "on": {
                "aggregate": aggregate,
                "provenance": {
                    "scenario_manifest_hash": "manifest-hash",
                    "ordered_pool_identity": ordered_pool_identity,
                    "checkpoint": {"sha256": "checkpoint-hash"},
                    "policy_hash": "policy-hash",
                    "observation_dimension": 42,
                    "controller_config_hash": "controller-hash",
                    "goal_recovery_config_hash": "goal-recovery-hash",
                },
            }
        },
        "frozen_runtime_component_hashes": {"actor": "actor-hash", "critic": "critic-hash"},
    }
    return step_rows, episode_rows, aggregate, summary


class SingleUAVPassCommitDiagnosticTests(unittest.TestCase):
    def test_fixed_population_is_twenty_ordered_episodes(self):
        d = pass_commit_diagnostic
        self.assertEqual(len(d.ALL_EPISODE_IDS), 20)
        self.assertEqual(d.ALL_EPISODE_IDS[0], "test_r99_episode_0001")
        self.assertEqual(d.ALL_EPISODE_IDS[-1], "test_r99_episode_0020")

    def test_main_and_special_cohorts_are_frozen(self):
        d = pass_commit_diagnostic
        self.assertEqual(
            d.MAIN_FAILURE_EPISODE_IDS,
            tuple(f"test_r99_episode_{i:04d}" for i in (2, 4, 5, 6, 11, 20)),
        )
        self.assertEqual(d.RESCUE_DESCRIPTIVE_EPISODE_ID, "test_r99_episode_0012")
        self.assertEqual(d.SAFETY_SPECIAL_EPISODE_ID, "test_r99_episode_0013")
        self.assertEqual(d.BOUNDARY_SPECIAL_EPISODE_ID, "test_r99_episode_0014")

    def test_manifest_preserves_frozen_seed_order(self):
        d = pass_commit_diagnostic
        scenarios, manifest = d._manifest_scenarios(PROJECT)
        ordered = manifest["identity"]["ordered_scenarios"]
        self.assertEqual([item["episode_id"] for item in ordered], list(d.ALL_EPISODE_IDS))
        self.assertEqual(len({int(item["scenario_seed"]) for item in ordered}), 20)
        self.assertEqual(set(scenarios), set(d.ALL_EPISODE_IDS))

    def test_parent_hash_is_identity_only_not_active_hash(self):
        d = pass_commit_diagnostic
        self.assertEqual(
            d.PARENT_RADIAL_INTENT_PREREGISTRATION_HASH,
            "bf70fb95188ca73c1ba87997a090d22331a3ac76a27daaaea1d86da09f59e40b",
        )
        self.assertNotEqual(d.PREREGISTRATION_FILE, "preregistered_radial_intent_persistence_endpoints.json")

    def test_action_decomposition_fix_has_explicit_parent_and_attempt_identity(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        payload = json.loads(bundle[d.PREREGISTRATION_FILE])
        self.assertEqual(
            payload["parent_pass_commit_preregistration_hash"],
            "74b999edb1158fc960374025f418d33c7812a57653c07f6d0e73fab8b69e02ac",
        )
        self.assertEqual(payload["attempt_id"], "action_decomposition_fix_v1")
        self.assertEqual(payload["schema_version"], "pass_commit_endpoints_v2_action_decomposition_fix")
        self.assertTrue(payload["previous_attempt"]["next_attempt_requires_fresh_paired_off_and_on"])

    def test_type7_quantiles_are_linear_and_reproducible(self):
        d = pass_commit_diagnostic
        values = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(d._type7(values, 0.25), 0.75)
        self.assertEqual(d._type7(values, 0.5), 1.5)
        self.assertEqual(d._type7(values, 0.75), 2.25)

    def test_cap_rule_uses_q3_and_hard_maximum_without_sweep(self):
        d = pass_commit_diagnostic
        rows = [
            {"episode_id": "a", "included": 1, "required_cancellation_mps": value}
            for value in (0.02, 0.04, 0.06, 0.08)
        ]
        # Complete the fixed cohort keys for the pure cap-summary helper.
        rows.extend(
            {"episode_id": episode_id, "included": 0, "required_cancellation_mps": 0.0}
            for episode_id in d.MAIN_FAILURE_EPISODE_IDS[1:]
        )
        result = d.derive_cap(rows)
        self.assertAlmostEqual(result["required_cancellation_statistics_mps"]["q3"], 0.065)
        self.assertEqual(result["pass_commit_max_delta_v_mps"], 0.065)
        self.assertFalse(result["sweep_performed"])

    def test_cap_derivation_preserves_every_source_row_and_reason(self):
        d = pass_commit_diagnostic
        prior_rows = d._csv_rows(PROJECT / d.PRIOR_ON_STEPS)
        scenarios, _ = d._manifest_scenarios(PROJECT)
        cap_rows = d.derive_cap_rows(prior_rows, scenarios)
        self.assertEqual(len(cap_rows), sum(row["episode_id"] in d.MAIN_FAILURE_EPISODE_IDS for row in prior_rows))
        self.assertTrue(all(row["inclusion_exclusion_reason"] for row in cap_rows))
        self.assertEqual(sum(int(row["included"]) for row in cap_rows), 54)

    def test_offline_cap_rows_use_runtime_trigger_and_both_existing_context_gates(self):
        d = pass_commit_diagnostic
        prior_rows = d._csv_rows(PROJECT / d.PRIOR_ON_STEPS)
        scenarios, _ = d._manifest_scenarios(PROJECT)
        cap_rows = d.derive_cap_rows(prior_rows, scenarios)
        included = [row for row in cap_rows if row["included"]]
        self.assertTrue(included)
        self.assertTrue(all(row["past_closest_approach"] for row in included))
        self.assertTrue(all(row["clearance_increasing_gate"] for row in included))
        self.assertTrue(all(row["pass_commit_latched"] for row in included))
        self.assertTrue(all(row["lateral_exit_assist_active"] == 0 for row in included))
        self.assertTrue(all(row["goal_recovery_active"] == 0 for row in included))

    def test_per_episode_zero_count_is_reportable_and_total_sample_is_positive(self):
        d = pass_commit_diagnostic
        prior_rows = d._csv_rows(PROJECT / d.PRIOR_ON_STEPS)
        scenarios, _ = d._manifest_scenarios(PROJECT)
        cap_rows = d.derive_cap_rows(prior_rows, scenarios)
        counts = {
            episode_id: sum(
                int(row["included"] == 1 and row["episode_id"] == episode_id)
                for row in cap_rows
            )
            for episode_id in d.MAIN_FAILURE_EPISODE_IDS
        }
        # The approved contract permits this cohort member to have no row.
        self.assertGreater(sum(counts.values()), 0)
        self.assertEqual(set(counts), set(d.MAIN_FAILURE_EPISODE_IDS))

    def test_real_cap_quantiles_and_frozen_value_are_locked(self):
        d = pass_commit_diagnostic
        rows = d._csv_rows(PROJECT / d.PRIOR_ON_STEPS)
        scenarios, _ = d._manifest_scenarios(PROJECT)
        result = d.derive_cap(d.derive_cap_rows(rows, scenarios))
        self.assertEqual(result["eligible_row_count"], 54)
        self.assertAlmostEqual(result["required_cancellation_statistics_mps"]["q1"], 0.04688886573620367)
        self.assertAlmostEqual(result["required_cancellation_statistics_mps"]["median"], 0.11859153512529347)
        self.assertAlmostEqual(result["required_cancellation_statistics_mps"]["q3"], 0.22226597384948757)
        self.assertAlmostEqual(result["required_cancellation_statistics_mps"]["maximum"], 3.227220032989568)
        self.assertEqual(result["pass_commit_max_delta_v_mps"], 0.075)

    def test_historical_off_contract_remains_identical_and_matches_frozen_trace(self):
        d = pass_commit_diagnostic
        contract = d._load_previous_active_contract(PROJECT)
        self.assertEqual(contract["fingerprint"], d.EXPECTED_OFF_BASELINE_FINGERPRINT)
        run_root = PROJECT / "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run/off"
        rows = d._csv_rows(run_root / "steps.csv")
        episodes = d._csv_rows(run_root / "episodes.csv")
        summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        result = d.validate_off_against_baseline(
            current_rows=rows,
            current_episodes=episodes,
            current_aggregate=summary["aggregate"],
            contract=contract,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["row_count"], 2185)
        self.assertEqual(result["episode_count"], 20)
        self.assertTrue(result["episode_trace_hashes_match"])
        self.assertTrue(result["episode_outcome_hashes_match"])
        self.assertTrue(result["aggregate_hash_matches"])

    def test_failed_attempt_is_marked_and_has_no_saved_on_rows(self):
        d = pass_commit_diagnostic
        marker = json.loads((PROJECT / d.ABORTED_RUN_STATUS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(marker["status"], "ABORTED_IN_ON_EPISODE_0001")
        self.assertEqual(marker["failure"], "ACTION_DECOMPOSITION_ASSERTION")
        self.assertEqual(marker["off"]["episodes_completed"], 20)
        self.assertEqual(marker["off"]["baseline_fingerprint"], d.EXPECTED_OFF_BASELINE_FINGERPRINT)
        self.assertEqual(list((PROJECT / "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run/on").iterdir()), [])
        self.assertFalse(marker["counted_as_formal_result"])

    def test_replacement_attempt_uses_distinct_output_root(self):
        d = pass_commit_diagnostic
        evaluator_path = PROJECT / "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
        tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
        roots = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {"PASS_COMMIT_DATA_ROOT", "PASS_COMMIT_LOG_ROOT", "PASS_COMMIT_PLOT_ROOT"}:
                    roots[node.targets[0].id] = ast.literal_eval(node.value.args[0])
        self.assertEqual(set(roots), {"PASS_COMMIT_DATA_ROOT", "PASS_COMMIT_LOG_ROOT", "PASS_COMMIT_PLOT_ROOT"})
        self.assertTrue(all("action_decomposition_fix_v1" in value for value in roots.values()))
        self.assertNotEqual(roots["PASS_COMMIT_LOG_ROOT"], "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run")
        self.assertNotIn("/run/", roots["PASS_COMMIT_LOG_ROOT"].replace("\\", "/"))

    def test_off_contract_normalizes_only_condition_and_excludes_new_prefix(self):
        d = pass_commit_diagnostic
        step_rows, episode_rows, aggregate, summary = _minimal_contract_inputs()
        contract = d.build_off_baseline_contract(
            prior_rows=step_rows,
            prior_episodes=episode_rows,
            prior_aggregate=aggregate,
            source_hashes={"input": "source-hash"},
            prior_summary=summary,
        )
        self.assertEqual(contract["normalized_field"], "condition")
        self.assertEqual(contract["normalized_value"], "off")
        self.assertIn("pass_commit_", contract["excluded_new_field_prefixes"])
        self.assertEqual(contract["fingerprint_name"], "PASS_COMMIT_OFF_BASELINE_FINGERPRINT")

    def test_off_contract_requires_all_twenty_and_exact_seeds(self):
        d = pass_commit_diagnostic
        step_rows, episode_rows, aggregate, summary = _minimal_contract_inputs()
        contract = d.build_off_baseline_contract(
            prior_rows=step_rows, prior_episodes=episode_rows,
            prior_aggregate=aggregate, source_hashes={}, prior_summary=summary,
        )
        current_rows = [dict(row, condition="off", pass_commit_active="0") for row in step_rows]
        current_episodes = [dict(row, condition="off", pass_commit_active_steps="0") for row in episode_rows]
        result = d.validate_off_against_baseline(
            current_rows=current_rows, current_episodes=current_episodes,
            current_aggregate=aggregate, contract=contract,
        )
        self.assertEqual(result["status"], "PASS")
        changed = [dict(row) for row in current_rows]
        changed[0]["scenario_seed"] = "999999"
        with self.assertRaisesRegex(RuntimeError, d.OFF_BASELINE_MISMATCH):
            d.validate_off_against_baseline(
                current_rows=changed, current_episodes=current_episodes,
                current_aggregate=aggregate, contract=contract,
            )

    def test_off_contract_fingerprint_changes_when_trace_outcome_or_aggregate_changes(self):
        d = pass_commit_diagnostic
        step_rows, episode_rows, aggregate, summary = _minimal_contract_inputs()
        contract = d.build_off_baseline_contract(
            prior_rows=step_rows, prior_episodes=episode_rows,
            prior_aggregate=aggregate, source_hashes={}, prior_summary=summary,
        )
        current_rows = [dict(row, condition="off") for row in step_rows]
        current_episodes = [dict(row, condition="off") for row in episode_rows]
        changed_aggregate = dict(aggregate, collision=1)
        with self.assertRaisesRegex(RuntimeError, d.OFF_BASELINE_MISMATCH):
            d.validate_off_against_baseline(
                current_rows=current_rows, current_episodes=current_episodes,
                current_aggregate=changed_aggregate, contract=contract,
            )

    def test_preparation_bundle_has_exact_seven_canonical_files(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        self.assertEqual(tuple(bundle), d.CANONICAL_ARTIFACT_FILES)
        self.assertEqual(len(bundle), 7)

    def test_new_active_preregistration_hash_is_distinct_and_verified(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        payload = json.loads(bundle[d.PREREGISTRATION_FILE])
        lock = json.loads(bundle[d.PREREGISTRATION_LOCK_FILE])
        source_lock = json.loads(bundle[d.RUNTIME_SOURCE_LOCK_FILE])
        active_hash = d._sha256_bytes(d._canonical(payload).encode("utf-8"))
        self.assertNotEqual(active_hash, d.PARENT_RADIAL_INTENT_PREREGISTRATION_HASH)
        self.assertEqual(lock["active_pass_commit_preregistration_hash"], active_hash)
        self.assertEqual(source_lock["active_pass_commit_preregistration_hash"], active_hash)

    def test_endpoints_lock_lateral_on_radial_off_and_four_tiers(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        payload = json.loads(bundle[d.PREREGISTRATION_FILE])
        base = payload["frozen_base_in_both_arms"]
        self.assertTrue(base["lateral_exit_braking_assistance_enabled"])
        self.assertFalse(base["radial_intent_persistence_assistance_enabled"])
        self.assertEqual([tier["tier"] for tier in payload["endpoint_hierarchy"]], [1, 2, 3, 4])

    def test_preparation_status_and_nonmutation_flags_are_frozen(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        summary = json.loads(bundle[d.PREPARATION_SUMMARY_FILE])
        flags = summary["execution_flags"]
        self.assertTrue(flags["pass_commit_diagnostic_prepared"])
        self.assertFalse(flags["pass_commit_diagnostic_executed"])
        self.assertFalse(flags["training_performed"])
        self.assertFalse(flags["rollout_executed"])
        self.assertFalse(flags["environment_step_called"])
        for key in ("actor_modified", "critic_modified", "gat_modified", "goal_recovery_modified", "terminal_modified", "reward_modified", "dynamics_modified", "safety_modified", "boundary_modified"):
            self.assertFalse(flags[key])

    def test_source_lock_hashes_runtime_inputs_without_generated_artifacts(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        source_lock = json.loads(bundle[d.RUNTIME_SOURCE_LOCK_FILE])
        hashed = source_lock["source_hashes"]
        self.assertIn("src/control/pass_commit_non_regression.py", hashed)
        self.assertIn("src/control/pass_commit_action_decomposition.py", hashed)
        self.assertIn("src/experiments/single_uav_goal_recovery_correction_evaluation.py", hashed)
        self.assertIn(d.PREVIOUS_OFF_BASELINE_CONTRACT_PATH.as_posix(), hashed)
        self.assertIn(d.PRIOR_ON_STEPS.as_posix(), hashed)
        self.assertNotIn(d.PREPARATION_ROOT.as_posix(), " ".join(hashed))

    def test_prepare_path_has_no_evaluator_or_forbidden_runtime_import(self):
        d = pass_commit_diagnostic
        source = (PROJECT / "src/experiments/single_uav_pass_commit_diagnostic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        top_level_imports = [node for node in imports if getattr(node, "col_offset", 0) == 0]
        for node in top_level_imports:
            name = node.module if isinstance(node, ast.ImportFrom) else ",".join(alias.name for alias in node.names)
            lowered = str(name).lower()
            self.assertFalse(any(token in lowered for token in ("formal", "simulation", "training", "evaluator")))
        human_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_human_evaluation")
        human_imports = [node for node in ast.walk(human_fn) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(len(human_imports), 1)
        self.assertIn("single_uav_goal_recovery_correction_evaluation", human_imports[0].module)

    def test_prepare_path_has_no_environment_step_optimizer_or_update_call(self):
        path = PROJECT / "src/experiments/single_uav_pass_commit_diagnostic.py"
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("environment.step", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)
        tree = ast.parse(source)
        prepare_names = {"build_preparation_bundle", "derive_cap_rows", "derive_cap", "write_preparation_artifacts", "main"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in prepare_names]
        self.assertTrue(functions)
        for function in functions:
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertFalse(node.func.attr in {"step", "backward"})

    def test_prepare_only_branch_has_no_rollout_training_or_human_run_call(self):
        source = (PROJECT / "src/experiments/single_uav_pass_commit_diagnostic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        prepare_branch = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.prepare_only"
        )
        calls = [node for statement in prepare_branch.body for node in ast.walk(statement) if isinstance(node, ast.Call)]
        call_names = []
        for node in calls:
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id.lower())
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr.lower())
        for forbidden in ("rollout", "training", "train", "step", "formal", "smoke", "human"):
            self.assertFalse(any(forbidden in name for name in call_names), (forbidden, call_names))
        self.assertIn("write_preparation_artifacts", call_names)

    def test_cap_csv_has_all_rows_with_explicit_reason(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        rows = list(csv.DictReader(io.StringIO(bundle[d.CAP_ROWS_FILE])))
        self.assertEqual(len(rows), 720)
        self.assertTrue(all(row["inclusion_exclusion_reason"] for row in rows))
        self.assertEqual(sum(int(row["included"]) for row in rows), 54)

    def test_new_attempt_keeps_radial_off_and_cap_frozen(self):
        d = pass_commit_diagnostic
        bundle = d.build_preparation_bundle(PROJECT)
        payload = json.loads(bundle[d.PREREGISTRATION_FILE])
        self.assertEqual(payload["single_treatment_difference"]["pass_commit_enabled"], {"OFF": False, "ON": True})
        self.assertFalse(payload["frozen_base_in_both_arms"]["radial_intent_persistence_assistance_enabled"])
        self.assertEqual(payload["authority_cap_derivation"]["pass_commit_max_delta_v_mps"], 0.075)
        self.assertEqual(payload["off_baseline"]["fingerprint"], d.EXPECTED_OFF_BASELINE_FINGERPRINT)


if __name__ == "__main__":
    unittest.main()
