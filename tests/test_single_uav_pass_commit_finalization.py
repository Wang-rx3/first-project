from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "src" / "pass_commit_finalization.py"
SPEC = importlib.util.spec_from_file_location("pass_commit_finalization_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _fixture():
    ids = ["episode_a", "episode_b", "episode_c"]
    prereg = {
        "attempt_id": "fixture_v1",
        "endpoint_hierarchy": [{"tier": 1, "name": "safety", "endpoints": ["collision"]}],
        "population": {
            "ordered_episode_ids": ids,
            "main_mechanism_subgroup": ["episode_b"],
            "rescue_descriptive_control": "episode_a",
            "safety_special": "episode_b",
            "boundary_special": "episode_c",
        },
    }
    outputs = {}
    for condition in ("off", "on"):
        episodes = []
        steps = []
        for index, episode_id in enumerate(ids, start=1):
            n_steps = 2 + (index == 3)
            passed = int(index == 1)
            active = int(condition == "on" and index == 2)
            local = []
            for step in range(n_steps):
                row = {
                    "condition": condition,
                    "episode_index": str(index),
                    "episode_id": episode_id,
                    "scenario_id": episode_id,
                    "scenario_seed": str(100 + index),
                    "noise_seed": str(200 + index),
                    "step": str(step),
                    "minimum_clearance_m": str(3.0 - step * 0.1),
                    "passed_obstacle": str(passed),
                    "post_bypass_recovery": str(passed),
                    "collision": "0",
                    "safety_projection_applied": "0",
                    "boundary_projection_applied": str(int(index == 3 and step == 0)),
                    "pass_commit_active": str(int(active and step == 1)),
                }
                local.append(row)
                steps.append(row)
            episodes.append({
                "condition": condition,
                "episode_index": str(index),
                "episode_id": episode_id,
                "scenario_seed": str(100 + index),
                "noise_seed": str(200 + index),
                "scenario_hash": f"hash-{index}",
                "steps": str(n_steps),
                "entered_8m": "1",
                "entered_5m": "1",
                "entered_3m": "1",
                "entered_1m": "1",
                "stable_goal": "0",
                "timeout": "0",
                "passed_obstacle": str(passed),
                "post_bypass_recovery": str(passed),
                "collision": "0",
                "safety_projection_steps": "0",
                "boundary_projection_steps": str(int(index == 3)),
                "pass_commit_active_steps": str(int(active)),
            })
        aggregate = {
            "episodes": 3,
            "entered_8m": 3,
            "entered_5m": 3,
            "entered_3m": 3,
            "entered_1m": 3,
            "stable_goal": 0,
            "timeout": 0,
            "passed_obstacle": 1,
            "post_bypass_recovery": 1,
            "collision": 0,
            "safety_projection_steps": 0,
            "boundary_projection_steps": 1,
            "pass_commit_active_steps": int(condition == "on"),
        }
        outputs[condition] = {
            "episodes": episodes,
            "step_rows": steps,
            "aggregate": aggregate,
            "provenance": {"scenario_manifest_hash": "manifest", "ordered_pool_identity": [[i, x, 200 + i] for i, x in enumerate(ids, 1)]},
        }
    return outputs, prereg


class PassCommitFinalizerTests(unittest.TestCase):
    def test_happy_path_is_pure_and_uses_preregistered_order(self):
        results, prereg = _fixture()
        before = copy.deepcopy(results)
        summary = MODULE.build_pass_commit_paired_summary(
            results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True
        )
        self.assertEqual(summary["population"], ["episode_a", "episode_b", "episode_c"])
        self.assertEqual(summary["on_pass_commit_active_steps"], 1)
        self.assertEqual(summary["paired_episode_outcomes"][1]["on_pass_commit_active_steps"], 1)
        self.assertEqual(results, before)

    def test_duplicate_order_subset_and_preregistration_mismatch_rejected(self):
        results, prereg = _fixture()
        results["on"]["episodes"][1]["episode_id"] = "episode_a"
        with self.assertRaises(MODULE.PassCommitValidationError):
            MODULE.build_pass_commit_paired_summary(results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True)
        results, prereg = _fixture()
        prereg["population"]["ordered_episode_ids"] = ["episode_a", "episode_b"]
        with self.assertRaises(MODULE.PassCommitValidationError):
            MODULE.build_pass_commit_paired_summary(results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True)

    def test_seed_step_count_and_counter_mismatch_rejected(self):
        for mutate in ("seed", "step", "counter"):
            results, prereg = _fixture()
            if mutate == "seed":
                results["on"]["episodes"][0]["scenario_seed"] = "999"
            elif mutate == "step":
                results["on"]["step_rows"].pop()
            else:
                results["on"]["episodes"][1]["pass_commit_active_steps"] = "0"
            with self.subTest(mutate=mutate), self.assertRaises(MODULE.PassCommitValidationError):
                MODULE.build_pass_commit_paired_summary(results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True)

    def test_malformed_counter_and_nonfinite_clearance_rejected(self):
        results, prereg = _fixture()
        results["on"]["step_rows"][0]["pass_commit_active"] = "NaN"
        with self.assertRaises(MODULE.PassCommitValidationError):
            MODULE.build_pass_commit_paired_summary(results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True)
        results, prereg = _fixture()
        results["off"]["step_rows"][0]["minimum_clearance_m"] = "nan"
        with self.assertRaises(MODULE.PassCommitValidationError):
            MODULE.build_pass_commit_paired_summary(results, preregistration=prereg, attempt_id="fixture_v1", off_baseline_fingerprint_match=True)

    def test_recovery_rejects_existing_and_overlapping_output_without_writing(self):
        results, prereg = _fixture()
        temp_parent = PROJECT / "logs" / "diagnostics" / "single_uav_ppo_pretraining_20261010_v1" / "executor_a_tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as root_name:
            root = Path(root_name)
            source = root / "source"
            preparation = root / "preparation"
            source.mkdir()
            preparation.mkdir()
            for condition, arm in results.items():
                arm_path = source / condition
                arm_path.mkdir()
                for filename, rows in (("episodes.csv", arm["episodes"]), ("steps.csv", arm["step_rows"])):
                    with (arm_path / filename).open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                (arm_path / "summary.json").write_text(json.dumps({"aggregate": arm["aggregate"], "provenance": arm["provenance"]}), encoding="utf-8")
            (preparation / "preregistered_pass_commit_endpoints.json").write_text(json.dumps(prereg), encoding="utf-8")
            (preparation / "pass_commit_preregistration_lock.json").write_text(json.dumps({"active_pass_commit_preregistration_hash": "x"}), encoding="utf-8")
            (preparation / "pass_commit_runtime_source_lock.json").write_text(json.dumps({"attempt_id": "fixture_v1", "source_hashes": {}}), encoding="utf-8")
            (preparation / "pass_commit_off_baseline_contract.json").write_text(json.dumps({"identity": {}, "fingerprint": "x"}), encoding="utf-8")
            existing = root / "existing"
            existing.mkdir()
            (existing / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(MODULE.PassCommitValidationError):
                MODULE.recover_pass_commit_summary(root, source_root=source, preparation_root=preparation, output_root=existing)
            with self.assertRaises(MODULE.PassCommitValidationError):
                MODULE.recover_pass_commit_summary(root, source_root=source, preparation_root=preparation, output_root=source / "child")
            self.assertEqual((existing / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_module_has_no_runtime_imports_and_cli_parser_is_stdlib(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("single_uav_goal_recovery_correction_evaluation", source)
        self.assertNotIn("src.simulation", source)
        parser = MODULE.build_parser()
        args = parser.parse_args(["--project-root", ".", "--output-root", "offline-result"])
        self.assertEqual(args.output_root, Path("offline-result"))


if __name__ == "__main__":
    unittest.main()
