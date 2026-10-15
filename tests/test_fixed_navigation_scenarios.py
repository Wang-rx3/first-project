"""Pure contract tests for the fixed-navigation scenario manifest."""
from __future__ import annotations

import copy
import math

from src.control.fixed_navigation import FixedWaypointNavigator
from src.experiments.single_uav_fixed_navigation_scenarios import build_fixed_navigation_scenarios


def _cases():
    manifest = build_fixed_navigation_scenarios()
    return manifest, [case for phase in manifest.values() for case in phase]


def _norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def test_manifest_cardinality_and_regime_coverage():
    manifest, all_cases = _cases()
    assert {key: len(value) for key, value in manifest.items()} == {
        "development": 6,
        "normal_evaluation": 36,
        "paired": 4,
        "obstacle": 4,
    }
    assert {case["regime"] for case in all_cases} == {"rest", "toward", "away", "tangent"}
    normal = manifest["normal_evaluation"]
    assert {regime: sum(case["regime"] == regime for case in normal) for regime in ("rest", "toward", "away", "tangent")} == {
        "rest": 9,
        "toward": 9,
        "away": 9,
        "tangent": 9,
    }


def test_ids_and_noise_seeds_are_unique_and_paired_reuse_is_intentional():
    manifest, all_cases = _cases()
    ids = [case["scenario_id"] for case in all_cases]
    seeds = [case["noise_seed"] for case in all_cases]
    assert len(ids) == len(set(ids))
    assert len(seeds) == len(set(seeds))
    fixed = manifest["paired"]
    coast_reuse = copy.deepcopy(fixed)
    assert [case["scenario_id"] for case in coast_reuse] == [case["scenario_id"] for case in fixed]
    assert [case["noise_seed"] for case in coast_reuse] == [case["noise_seed"] for case in fixed]


def test_mode_contract_and_finite_bounds():
    _manifest, all_cases = _cases()
    for case in all_cases:
        assert case["kind"] in {"normal", "single"}
        assert case["safety2_enabled"] is True
        assert case["assist_mask"] == "00000"
        assert all(case[name] is False for name in ("pass_commit", "lateral_assist", "radial_assist", "goal_recovery", "terminal_capture"))
        scene = case["scenario"]
        for field in ("initial_positions", "initial_velocities", "initial_attitudes", "goals"):
            vector = scene[field]["uav_00"]
            assert len(vector) == 3 and all(math.isfinite(float(value)) for value in vector)
        for point in case["references"]:
            assert len(point) == 3 and all(math.isfinite(float(value)) for value in point)
        for field in ("initial_positions", "goals"):
            point = scene[field]["uav_00"]
            assert 0.0 <= point[0] <= 200.0
            assert 0.0 <= point[1] <= 200.0
            assert 2.0 <= point[2] <= 60.0


def test_normal_and_obstacle_reset_geometry_contract():
    manifest, all_cases = _cases()
    normal = manifest["development"] + manifest["normal_evaluation"] + manifest["paired"]
    assert all(case["kind"] == "normal" for case in normal)
    assert all(case["scenario"]["obstacle_specs"] == [] for case in normal)
    assert all(case["scenario"]["obstacles"] == [] for case in normal)
    obstacles = manifest["obstacle"]
    assert all(case["kind"] == "single" for case in obstacles)
    assert all(len(case["scenario"]["obstacle_specs"]) == 1 for case in obstacles)
    assert all("obstacles" not in case["scenario"] for case in obstacles)
    assert all(case["scenario"]["obstacle_specs"][0]["geometry_type"] == "box" for case in obstacles)
    assert all(case["scenario"]["obstacle_specs"][0]["physical_collision_enabled"] is True for case in obstacles)
    assert all(case["scenario"]["obstacle_specs"][0]["velocity_mps"] == [0.0, 0.0, 0.0] for case in obstacles)
    hashes = [case["geometry_sha256"] for case in all_cases]
    assert all(len(value) == 64 for value in hashes)
    assert len({obstacles[0]["geometry_sha256"], obstacles[1]["geometry_sha256"]}) == 1
    assert len({obstacles[2]["geometry_sha256"], obstacles[3]["geometry_sha256"]}) == 1


def test_development_normal_geometry_sets_are_disjoint():
    manifest, _all_cases = _cases()
    development = {case["geometry_sha256"] for case in manifest["development"]}
    normal = {case["geometry_sha256"] for case in manifest["normal_evaluation"]}
    assert development.isdisjoint(normal)


def test_only_four_normal_cases_have_belief_offsets_with_required_norms():
    manifest, _all_cases = _cases()
    normal = manifest["normal_evaluation"]
    offset_cases = [case for case in normal if any(abs(float(value)) > 0.0 for value in case["belief_position_offset_m"])]
    assert [case["scenario_id"] for case in offset_cases] == [f"fixednav_normal{index:02d}" for index in range(32, 36)]
    for case in normal[:32]:
        assert case["belief_position_offset_m"] == [0.0, 0.0, 0.0]
        assert case["belief_velocity_offset_m"] == [0.0, 0.0, 0.0]
    for case in offset_cases:
        assert math.isclose(_norm(case["belief_position_offset_m"]), 0.4, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(_norm(case["belief_velocity_offset_m"]), 0.2, rel_tol=0.0, abs_tol=1e-12)


def test_manifest_is_deterministic_and_controller_does_not_mutate_goal():
    first = build_fixed_navigation_scenarios()
    second = build_fixed_navigation_scenarios()
    assert first == second
    case = first["obstacle"][0]
    before = copy.deepcopy(case["scenario"])
    navigator = FixedWaypointNavigator(case["references"])
    navigator.command([55.0, 110.0, 22.0, 0.0, 0.0, 0.0])
    assert case["scenario"] == before
    assert case["scenario"]["goals"]["uav_00"] == [79.0, 110.0, 22.0]
