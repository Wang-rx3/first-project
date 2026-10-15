from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/experiments/single_uav_terminal_capture_v23_preboundary_audit.py"
SPEC = importlib.util.spec_from_file_location("v23_preboundary_audit", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_row(step: int, distance: float, speed: float = 0.2, radial: float = -0.1, hold: int = 1, episode: str = "e"):
    candidate = distance <= 1.0 and speed <= 0.5
    counter = hold if candidate else 0
    return {
        "episode_id": episode,
        "step": str(step),
        "goal_distance_for_stable_m": str(distance),
        "goal_speed_for_stable_mps": str(speed),
        "stable_goal_distance_threshold_m": "1.0",
        "stable_speed_threshold_mps": "0.5",
        "stable_hold_steps_required": "5",
        "goal_distance_ok": str(int(distance <= 1.0)),
        "goal_speed_ok": str(int(speed <= 0.5)),
        "stable_candidate_step": str(int(candidate)),
        "stable_hold_counter": str(counter),
        "stable_goal_now": str(int(candidate and counter >= 5)),
        "post_step_actual_radial_velocity_mps": str(radial),
        "post_step_actual_tangential_velocity_mps": "0.1",
        "post_step_actual_speed_mps": str(speed),
        "goal_distance_m": str(distance),
        "goal_recovery_radial_delta_v_mps": "[0,0,0]",
        "goal_recovery_correction_norm_mps": "0.0",
        "terminal_correction_clipped": "0",
        "terminal_v21_actor_radial_delta_v_mps": "[0,0,0]",
        "terminal_v21_conflict_compensation_radial_mps": "0.0",
        "terminal_v21_net_radial_pre_transform_mps": "0.0",
        "terminal_v21_net_radial_post_transform_mps": "0.0",
        "terminal_v22_enabled": "1",
        "terminal_v22_hold_latched": "1",
        "terminal_v22_hold_enter_event": "0",
        "terminal_v22_hold_release_event": "0",
        "terminal_v22_hold_mode": "inside_canonical",
        "terminal_v22_restorative_active": "0",
        "terminal_v22_distance_outside_stable_m": "0.0",
        "terminal_v22_restorative_desired_radial_speed_mps": "0.0",
        "terminal_v22_restorative_radial_error_mps": "0.0",
        "terminal_v22_restorative_delta_v_mps": "[0,0,0]",
        "terminal_v22_inside_canonical_goal": str(int(distance <= 1.0)),
        "stable_goal": str(int(candidate and counter >= 5)),
        "goal_distance_source": "post_step_estimated_state",
        "goal_speed_source": "simulator_actual_state",
        "stable_criterion_timing": "post_environment_step_before_termination_return",
        "stable_log_schema_version": "goal_recovery_actual_stable_v1",
    }


def parsed(distances, speeds=None, radials=None, episode="e"):
    speeds = speeds or [0.2] * len(distances)
    radials = radials or [-0.1] * len(distances)
    rows = []
    counter = 0
    for step, (distance, speed, radial) in enumerate(zip(distances, speeds, radials)):
        candidate = distance <= 1.0 and speed <= 0.5
        counter = counter + 1 if candidate else 0
        rows.append(make_row(step, distance, speed, radial, counter, episode))
    return audit.parse_steps(rows)[episode]


def episode_metadata(episode: str, stable: int = 0, entered: int = 1):
    return {
        "episode_id": episode,
        "v22_hold_entered": str(entered),
        "stable_goal": str(stable),
    }


def test_01_formula_zero_at_target():
    assert audit.candidate_predicted_inward_speed(0.95, 0.95) == pytest.approx(0.0)


def test_02_formula_zero_below_target():
    assert audit.candidate_predicted_inward_speed(0.90, 0.95) == pytest.approx(0.0)


def test_03_formula_midpoint():
    assert audit.candidate_predicted_inward_speed(1.0, 0.95) == pytest.approx(0.25 * (0.05 / 0.10))


def test_04_formula_caps_at_quarter_mps():
    assert audit.candidate_predicted_inward_speed(1.5, 0.94) == pytest.approx(0.25)


def test_05_formula_rejects_bad_denominator():
    with pytest.raises(ValueError, match="below 0.98"):
        audit.candidate_predicted_inward_speed(1.0, 1.05)


def test_06_nearest_rank_p90_is_not_interpolated():
    assert audit.nearest_rank_percentile([0.1, 0.2, 0.3, 0.4, 0.5], 0.90) == pytest.approx(0.5)


def test_07_nearest_rank_empty():
    assert audit.nearest_rank_percentile([]) is None


def test_08_counter_reconstruction():
    rows = parsed([1.1, 0.9, 0.8, 0.7])
    assert [row["_counter"] for row in rows] == [0, 1, 2, 3]


def test_09_counter_reset_reconstruction():
    rows = parsed([0.9, 0.8, 1.1, 0.9])
    assert [row["_counter"] for row in rows] == [1, 2, 0, 1]


def test_10_duplicate_key_fails():
    rows = [make_row(0, 1.1), make_row(0, 1.1)]
    with pytest.raises(ValueError, match="duplicate"):
        audit.parse_steps(rows)


def test_11_non_contiguous_key_fails():
    with pytest.raises(ValueError, match="non-contiguous"):
        audit.parse_steps([make_row(0, 1.1), make_row(2, 1.1)])


def test_12_nan_fails_fast():
    bad = make_row(0, 1.1)
    bad["goal_distance_for_stable_m"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        audit.parse_steps([bad])


def test_13_stable_thresholds_are_frozen():
    bad = make_row(0, 1.1)
    bad["stable_speed_threshold_mps"] = "0.4"
    with pytest.raises(ValueError, match="thresholds"):
        audit.parse_steps([bad])


def test_14_outward_precursor_one_step():
    rows = parsed([0.9, 1.1])
    assert audit.is_outward_precursor(rows, 0)


def test_15_outward_precursor_two_step():
    rows = parsed([0.9, 0.95, 1.1])
    assert audit.is_outward_precursor(rows, 0)
    assert audit._precursor_horizon(rows, 0) == 2


def test_16_outward_precursor_requires_negative_radial_velocity():
    rows = parsed([0.9, 1.1], radials=[0.1, 0.1])
    assert not audit.is_outward_precursor(rows, 0)


def test_17_distance_only_reset_exact_transition():
    grouped = {"e": parsed([0.9, 0.8, 1.1])}
    resets = audit.detect_distance_only_resets(grouped, strict=False)
    assert len(resets) == 1 and resets[0]["reset_step"] == 2


def test_18_speed_failure_is_not_distance_only_reset():
    rows = parsed([0.9, 0.8, 0.9], speeds=[0.2, 0.2, 0.6])
    # The helper row's speed field is not a stable candidate, so this transition
    # cannot satisfy the current distance_ok=0, speed_ok=1 reset definition.
    assert audit.detect_distance_only_resets({"e": rows}, strict=False) == []


def test_19_reset_strict_count_gate():
    with pytest.raises(ValueError, match="exactly 9"):
        audit.detect_distance_only_resets({"e": parsed([0.9, 1.1])})


def test_20_reset_window_is_t_minus_three_through_t():
    rows = parsed([1.2, 1.1, 1.0, 0.9, 0.8, 1.1])
    reset = audit.detect_distance_only_resets({"e": rows}, strict=False)
    windows = audit.build_reset_windows({"e": rows}, reset)
    assert [int(row["relative_step"]) for row in windows] == [-3, -2, -1, 0]


def test_21_candidate_scan_contains_all_three_subsets():
    rows = parsed([0.98, 1.01, 0.99], episode="e")
    grouped = {"e": rows}
    reset = [{"episode_id": "e", "reset_step": 2, "precursor_indices": [0], "outward_precursor": True}]
    scan, population = audit.evaluate_candidates(grouped, [episode_metadata("e")], reset)
    assert {key for key in scan[0]["subset_metrics"]} == {"all_pop", "precursor_episodes", "stable_goal_episodes"}
    assert population["hold_entered_episode_count"] == 1


def test_22_candidate_coverage_uses_precursor_distance():
    rows = parsed([0.96, 1.1], episode="e")
    reset = [{"episode_id": "e", "reset_step": 1, "precursor_indices": [0], "outward_precursor": True}]
    scan, _ = audit.evaluate_candidates({"e": rows}, [episode_metadata("e")], reset)
    target_row = next(item for item in scan if item["target_radius_m"] == 0.96)
    assert target_row["coverage_count"] == 0


def test_23_candidate_gate_is_deepest_first():
    assert tuple(reversed(audit.CANDIDATE_TARGETS_M)) == (0.97, 0.96, 0.95, 0.94)


def test_24_status_flags_prove_no_rollout():
    assert audit.STATUS_FLAGS["rollout_executed"] is False
    assert audit.STATUS_FLAGS["offline_preboundary_audit_executed"] is True


def test_25_output_contract_is_four_files():
    assert len(audit.OUTPUT_FILES) == 4
    assert audit.OUTPUT_FILES[0].endswith("summary.json")
    assert audit.OUTPUT_FILES[-1].endswith("report.md")


def test_26_module_has_no_environment_or_controller_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("control" in name or "environment" in name or "formal" in name for name in imported)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "environment.step(" not in source
    assert "rollout(" not in source


def test_27_csv_reader_opens_read_only():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_csv")
    modes = [
        node.args[0].value
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert modes == ["r"]


def test_28_source_scope_names_exactly_four_inputs():
    assert audit.SOURCE_FILES == (
        "steps.csv",
        "episodes.csv",
        "terminal_capture_v22_validation_episodes.csv",
        "terminal_capture_v22_validation_summary.json",
    )
