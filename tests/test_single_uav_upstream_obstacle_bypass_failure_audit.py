"""Offline contract tests for the upstream obstacle-bypass audit."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path

_AUDIT_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "experiments"
    / "single_uav_upstream_obstacle_bypass_failure_audit.py"
)
_SPEC = importlib.util.spec_from_file_location("offline_upstream_audit", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)

DEFAULT_SOURCE_ROOT = _AUDIT.DEFAULT_SOURCE_ROOT
REFERENCE_EPISODES = _AUDIT.REFERENCE_EPISODES
TARGET_EPISODES = _AUDIT.TARGET_EPISODES
_encounter_step = _AUDIT._encounter_step
_episode_diagnosis = _AUDIT._episode_diagnosis
_episode_groups = _AUDIT._episode_groups
_identity_source = _AUDIT._identity_source
_metric_value = _AUDIT._metric_value
_phase_timeline = _AUDIT._phase_timeline
_reference_statistics = _AUDIT._reference_statistics
run_audit = _AUDIT.run_audit


def test_requested_cohorts_are_disjoint_and_complete() -> None:
    assert len(TARGET_EPISODES) == 9
    assert len(REFERENCE_EPISODES) == 11
    assert set(TARGET_EPISODES).isdisjoint(REFERENCE_EPISODES)
    assert sorted(TARGET_EPISODES + REFERENCE_EPISODES) == [
        f"test_r99_episode_{index:04d}" for index in range(1, 21)
    ]


def test_source_identity_gate_and_step_count() -> None:
    validation, groups, episode_data, _, _ = _identity_source(DEFAULT_SOURCE_ROOT)
    assert validation["status"] == "PASS"
    assert len(groups) == 20
    assert sum(len(rows) for rows in groups.values()) == 2182
    assert all(validation["identity_checks"].values())
    assert all(int(float(row["max_steps"])) == 120 for row in episode_data["rows"])


def test_target_and_reference_outcomes_match_source() -> None:
    _, _, episode_data, _, _ = _identity_source(DEFAULT_SOURCE_ROOT)
    rows = episode_data["map"]
    assert all(
        rows[episode_id]["passed_obstacle"] == "0"
        and rows[episode_id]["post_bypass_recovery"] == "0"
        for episode_id in TARGET_EPISODES
    )
    assert all(
        rows[episode_id]["passed_obstacle"] == "1"
        and rows[episode_id]["post_bypass_recovery"] == "1"
        for episode_id in REFERENCE_EPISODES
    )


def test_encounter_rule_uses_logged_extent_and_buffer() -> None:
    rows = [
        {
            "step": "0",
            "relevant_obstacle_id": "obs",
            "along_obstacle_m": "8",
            "obstacle_extent_m": "5",
        },
        {
            "step": "1",
            "relevant_obstacle_id": "obs",
            "along_obstacle_m": "5.7",
            "obstacle_extent_m": "5",
        },
    ]
    assert _encounter_step(rows) == 1


def test_norm_metric_aliases_use_vector_fields() -> None:
    row = {
        "actor_raw_delta_v_mps": "[3, 4, 0]",
        "corrected_delta_v_pre_safety_mps": "[0, 6, 0]",
        "applied_delta_v_mps": "[0, 8, 0]",
    }
    assert _metric_value(row, "actor_raw_norm_mps") == 5.0
    assert _metric_value(row, "pre_safety_norm_mps") == 6.0
    assert _metric_value(row, "applied_norm_mps") == 8.0


def test_phase_timeline_preserves_pass_and_projection_steps() -> None:
    rows = [
        {
            "step": str(index),
            "relevant_obstacle_clearance_m": str(4 - index * 0.1),
            "post_step_actual_tangential_velocity_mps": str(index * 0.2),
            "goal_distance_after_m": str(10 - index),
            "safety_projection_applied": "1" if index == 2 else "0",
            "boundary_projection_applied": "1" if index == 3 else "0",
            "action_clipped": "1" if index == 4 else "0",
            "collision": "0",
        }
        for index in range(6)
    ]
    timeline = _phase_timeline(rows, encounter=1, passed=4, recovery=4)
    assert timeline["first_obstacle_near_step"] == 1
    assert timeline["minimum_clearance_step"] == 4
    assert timeline["first_safety_projection_step"] == 2
    assert timeline["first_boundary_projection_step"] == 3
    assert timeline["first_action_clipping_step"] == 4
    assert timeline["passed_obstacle_step"] == 4
    assert timeline["post_bypass_recovery_step"] == 4


def test_reference_statistics_have_both_required_windows() -> None:
    _, groups, _, _, _ = _identity_source(DEFAULT_SOURCE_ROOT)
    reference, _ = _reference_statistics(groups)
    assert reference["episode_count"] == 11
    for window in ("encounter_m10_p30", "passed_m20_p0"):
        assert reference[window]["post_step_actual_radial_velocity_mps"]["n"] > 0
        assert reference[window]["post_step_actual_tangential_velocity_mps"]["q1"] <= reference[window]["post_step_actual_tangential_velocity_mps"]["q3"]


def test_full_audit_has_requested_files_and_header(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    expected = {
        "upstream_bypass_audit_summary.json",
        "upstream_bypass_episode_diagnostics.csv",
        "upstream_bypass_phase_timeline.csv",
        "upstream_bypass_obstacle_window_steps.csv",
        "upstream_bypass_reference_comparison.csv",
        "upstream_bypass_projection_breakdown.csv",
        "upstream_bypass_report.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert summary["source_step_count"] == 2182
    assert (output / "upstream_bypass_report.md").read_text(encoding="utf-8").splitlines()[0] == "UPSTREAM_BYPASS_AUDIT_READY"


def test_projection_totals_and_episode_concentration(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    rows = summary["per_episode_projection_counts"]
    assert sum(int(row["safety_projection_steps"]) for row in rows) == 5
    assert sum(int(row["boundary_projection_steps"]) for row in rows) == 41
    assert [row["episode_id"] for row in rows if row["boundary_projection_steps"]] == [
        "test_r99_episode_0013",
        "test_r99_episode_0014",
    ]


def test_goal_recovery_is_inactive_for_all_targets(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    assert all(item["active_steps"] == 0 for item in summary["goal_recovery_active_episode_breakdown"])
    assert all(item["activation_mode"] == "post_bypass_only" for item in summary["goal_recovery_active_episode_breakdown"])


def test_direction_is_explicitly_not_identifiable(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    for diagnosis in summary["per_episode_diagnosis"]:
        assert diagnosis["bypass_direction_audit"]["intended_side"] == "NOT_IDENTIFIABLE"
        assert diagnosis["bypass_direction_audit"]["direction_consistency"] == "NOT_ASSESSABLE"
        assert diagnosis["lateral_authority"]["raw_actor_lateral_component_observable"] is False


def test_extension_summary_has_first100_and_tail20(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    assert len(summary["extension_only_behavior_summary"]) == 9
    for item in summary["extension_only_behavior_summary"]:
        assert set(item) >= {"first_100", "extension_100_119"}


def test_source_hashes_are_unchanged_after_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    summary = run_audit(DEFAULT_SOURCE_ROOT, output)
    validation = summary["source_validation"]
    assert validation["source_unchanged_during_audit"] is True
    assert validation["source_hashes_before"] == validation["source_hashes_after"]


def test_json_is_strict_and_has_required_machine_keys(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    run_audit(DEFAULT_SOURCE_ROOT, output)
    summary = json.loads((output / "upstream_bypass_audit_summary.json").read_text(encoding="utf-8"))
    required = {
        "source_validation",
        "source_episode_count",
        "source_step_count",
        "target_episode_ids",
        "reference_episode_ids",
        "identity_checks",
        "per_episode_projection_counts",
        "per_episode_phase_timeline",
        "per_episode_diagnosis",
        "success_reference_statistics",
        "failure_vs_reference",
        "failure_cluster_counts",
        "safety_projection_episode_breakdown",
        "boundary_projection_episode_breakdown",
        "action_clip_episode_breakdown",
        "goal_recovery_active_episode_breakdown",
        "extension_only_behavior_summary",
    }
    assert required <= set(summary)
    assert summary["failure_cluster_counts"]["primary_total"] == 9


def test_csv_episode_foreign_keys_and_reference_relations(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    run_audit(DEFAULT_SOURCE_ROOT, output)
    with (output / "upstream_bypass_reference_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["episode_id"] for row in rows} == set(TARGET_EPISODES)
    assert {row["relation"] for row in rows} <= {
        "BELOW_IQR",
        "WITHIN_IQR",
        "ABOVE_IQR",
        "NOT_ASSESSABLE",
    }


def test_audit_module_has_no_runtime_imports() -> None:
    path = Path(__file__).parents[1] / "src" / "experiments" / "single_uav_upstream_obstacle_bypass_failure_audit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert all(not name.startswith("src.simulation") for name in imports)
    assert all(not name.startswith("src.control") for name in imports)
    assert all(not name.startswith("src.learning") for name in imports)
