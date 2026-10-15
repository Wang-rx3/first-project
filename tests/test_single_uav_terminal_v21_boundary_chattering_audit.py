from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/experiments/single_uav_terminal_v21_boundary_chattering_audit.py"
SPEC = importlib.util.spec_from_file_location("boundary_audit", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(step: int, distance: float, speed: float = 0.2, episode: str = "e", radial: float = 0.1, counter: int | None = None):
    candidate = distance <= 1.0 and speed <= 0.5
    return {
        "episode_id": episode, "step": str(step),
        "goal_distance_for_stable_m": str(distance), "goal_speed_for_stable_mps": str(speed),
        "stable_goal_distance_threshold_m": "1.0", "stable_speed_threshold_mps": "0.5",
        "stable_hold_steps_required": "5", "goal_distance_ok": str(int(distance <= 1.0)),
        "goal_speed_ok": str(int(speed <= 0.5)), "stable_candidate_step": str(int(candidate)),
        "stable_hold_counter": str(counter if counter is not None else 0),
        "stable_goal_now": str(int((counter or 0) >= 5)),
        "post_step_actual_radial_velocity_mps": str(radial),
        "post_step_actual_tangential_velocity_mps": "0.1", "post_step_actual_speed_mps": str(speed),
        "terminal_v2_inside_goal_hold": str(int(distance <= 1.0)),
        "terminal_v21_actor_radial_delta_v_mps": "0.0",
        "terminal_v21_conflict_compensation_radial_mps": "0.0",
        "terminal_v21_net_radial_post_transform_mps": "0.0",
    }


def parsed(distances, speeds=None, radial=None):
    speeds = speeds or [0.2] * len(distances)
    radial = radial or [0.1] * len(distances)
    counter = 0
    rows = []
    for i, (distance, speed, rv) in enumerate(zip(distances, speeds, radial)):
        counter = counter + 1 if distance <= 1 and speed <= .5 else 0
        rows.append(row(i, distance, speed, radial=rv, counter=counter))
    return audit.parse_steps(rows)["e"]


def test_01_stable_counter_offline_reconstruction():
    rows = parsed([1.1, 1, .9, .8, .7, .6])
    assert [r["_counter"] for r in rows] == [0, 1, 2, 3, 4, 5]


def test_02_reset_classification_all_categories():
    rows = parsed([.9, 1.1, .9, .9], [.2, .2, .2, .6])
    assert [x["type"] for x in audit.classify_resets(rows)] == ["distance_only_reset", "speed_only_reset"]
    synthetic = [
        {"_counter": 1, "_step": 0},
        {"_counter": 0, "_step": 1, "_distance_ok": False, "_speed_ok": False},
    ]
    assert audit.classify_resets(synthetic)[0]["type"] == "both_reset"


def test_03_inside_to_outside_crossing():
    assert audit.detect_crossings(parsed([.99, 1.01]))[0]["direction"] == "inside_to_outside"


def test_04_outside_to_inside_crossing():
    crossings = audit.detect_crossings(parsed([1.01, .99, 1.02, .98]))
    assert [item["direction"] for item in crossings] == ["inside_to_outside", "outside_to_inside"]


def test_05_excursion_grouping_and_open_censoring():
    items = audit.detect_excursions(parsed([.9, 1.01, 1.02, .99, 1.03]))
    assert len(items) == 2 and items[0]["return_to_1m"] and items[0]["steps_until_return"] == 2
    assert items[1]["censored"] and items[1]["steps_until_return"] is None


def test_06_max_excess_calculation():
    item = audit.detect_excursions(parsed([.9, 1.01, 1.04, .8]))[0]
    assert item["maximum_distance_m"] == 1.04 and item["peak_excess_over_1m"] == pytest.approx(.04)


def test_07_penetration_depth_before_first_reset():
    rows = parsed([1.1, .98, .9, 1.01])
    assert audit.penetration_before_first_reset(rows, audit.classify_resets(rows)) == pytest.approx(.1)


def test_08_candidate_radius_scan_is_reach_row_inclusive():
    rows = parsed([.98, .99, .99, .99, .99])
    result = audit.screen_enter(rows, .98)
    assert result["first_reach_step"] == 0 and result["window_start_step"] == 0


def test_09_release_screen_strict_and_no_resegmentation():
    excursions = [{"maximum_distance_m": 1.02}, {"maximum_distance_m": 1.03}]
    assert audit.screen_release(excursions, 1.02) == 1


def test_10_zero_crossing_case():
    assert audit.detect_crossings(parsed([1.2, 1.1, 1.05])) == []


def test_initial_entry_is_not_counted_as_post_entry_crossing():
    assert audit.detect_crossings(parsed([1.2, .99])) == []


def test_11_episode_never_enters_1m():
    rows = parsed([1.2, 1.1])
    assert audit.detect_excursions(rows) == [] and audit.screen_enter(rows, 1.0)["reached"] is False


def test_12_stable_episode():
    rows = parsed([.9] * 5)
    assert rows[-1]["_stable"] and rows[-1]["_counter"] == 5


def test_13_late_entry_is_step_95_or_later():
    rows = parsed([.9, .9], radial=[.1, .1])
    label, evidence = audit.classify_failure(rows, [], [], [], 95)
    assert evidence["late_entry"] and label == "LATE_ENTRY"


def test_14_nan_fails_fast():
    with pytest.raises(ValueError, match="finite"):
        audit.parse_steps([row(0, float("nan"), counter=0)])


def test_15_malformed_row_fails_fast():
    bad = row(0, 1.1, counter=0)
    del bad["goal_speed_ok"]
    with pytest.raises(ValueError, match="missing fields"):
        audit.parse_steps([bad])


def test_16_duplicate_step_fails_fast():
    with pytest.raises(ValueError, match="duplicate"):
        audit.parse_steps([row(0, 1.1, counter=0), row(0, 1.1, counter=0)])


def test_17_non_monotonic_step_fails_fast():
    with pytest.raises(ValueError, match="non-contiguous/non-monotonic"):
        audit.parse_steps([row(1, 1.1, counter=0), row(3, 1.1, counter=0)])


def test_18_source_files_are_opened_read_only_and_hashed():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "source_sha256" in source and "before != after" in source
    tree = ast.parse(source)
    read_csv_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_csv")
    modes = [node.args[0].value for node in ast.walk(read_csv_function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open" and node.args and isinstance(node.args[0], ast.Constant)]
    assert modes == ["r"]


def test_19_control_files_untouched_by_offline_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "src/control" not in source and "simulation.environment" not in source
    assert audit.STATUS_FLAGS["control_algorithm_modified"] is False


def test_20_no_rollout_path_and_exact_output_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "environment.step(" not in source and "rollout(" not in source
    assert audit.STATUS_FLAGS["rollout_executed"] is False
    assert audit.OUTPUT_FILES == (
        "boundary_chattering_summary.json", "boundary_chattering_episodes.csv", "boundary_crossings.csv",
        "boundary_excursions.csv", "hold_radius_screen.csv", "boundary_chattering_report.md",
    )
