"""Pure contract tests for the three-UAV fixed-navigation helpers."""

from __future__ import annotations

import math

import pytest

from src.experiments.three_uav_fixed_navigation_contract import (
    UIDS,
    StrictParkingTracker,
    controlled_communication_actions,
    joint_closest_approach,
    make_scenario_schedule,
)


def _local(*, candidates=(), effective=(), potential=(), mask=None):
    result = {
        "candidate_measurement_sources": list(candidates),
        "effective_sources": list(effective),
        "potential_targets": list(potential),
    }
    if mask is not None:
        result["action_mask"] = list(mask)
    return result


def test_schedule_has_exact_eight_rows_seeds_stages_and_stable_uid_maps():
    schedule = make_scenario_schedule()
    assert len(schedule) == 8
    assert [row["episode_id"] for row in schedule] == [
        "three_nav_dev_normal",
        "three_nav_c0_normal",
        "three_nav_c0_cut",
        "three_nav_c1_normal",
        "three_nav_c1_cut",
        "three_nav_c2_normal",
        "three_nav_c2_cut",
        "three_nav_c0_fusion_off",
    ]
    assert [row["case_id"] for row in schedule] == ["dev", "c0", "c0", "c1", "c1", "c2", "c2", "c0"]
    assert [row["variant"] for row in schedule] == [
        "normal", "normal", "cut", "normal", "cut", "normal", "cut", "fusion_off"
    ]
    assert [row["seed"] for row in schedule] == [
        2026115100, 2026115200, 2026115200, 2026115201,
        2026115201, 2026115202, 2026115202, 2026115200,
    ]
    assert schedule[0]["stage"] == "development"
    assert all(row["stage"] == "evaluation" for row in schedule[1:7])
    assert schedule[-1]["stage"] == "fusion_diagnostic"
    assert [row["cooperative_ekf_enabled"] for row in schedule] == [True] * 7 + [False]
    for row in schedule:
        scene = row["scenario"]
        assert list(scene["initial_positions"]) == list(UIDS)
        assert list(scene["initial_velocities"]) == list(UIDS)
        assert list(scene["initial_attitudes"]) == list(UIDS)
        assert list(scene["goals"]) == list(UIDS)
        assert scene["obstacle_specs"] == []
        assert scene["obstacles"] == []
        assert len(row["geometry_sha256"]) == 64


def test_schedule_geometry_clones_only_matched_variants_and_c2_has_stationary_uav():
    schedule = make_scenario_schedule()
    assert schedule[1]["scenario"] == schedule[2]["scenario"] == schedule[7]["scenario"]
    assert schedule[1]["geometry_sha256"] == schedule[2]["geometry_sha256"] == schedule[7]["geometry_sha256"]
    assert schedule[3]["scenario"] == schedule[4]["scenario"]
    assert schedule[5]["scenario"] == schedule[6]["scenario"]
    assert len({row["geometry_sha256"] for row in schedule}) == 4
    assert schedule[5]["scenario"]["initial_positions"] == {
        "uav_00": [40.0, 100.0, 25.0],
        "uav_01": [110.0, 100.0, 25.0],
        "uav_02": [77.0, 108.0, 31.0],
    }
    assert schedule[5]["scenario"]["goals"]["uav_02"] == schedule[5]["scenario"]["initial_positions"]["uav_02"]
    first = make_scenario_schedule()
    first[0]["scenario"]["goals"]["uav_00"][0] = -1.0
    assert make_scenario_schedule()[0]["scenario"]["goals"]["uav_00"] == [52.0, 40.0, 20.0]


def test_controlled_actions_use_incoming_intersection_mask_and_round_robin():
    observations = {
        "uav_00": _local(
            candidates=("uav_01", "uav_02"),
            effective=("uav_01", "uav_02"),
            potential=("uav_01", "uav_02"),
            mask=(True, False, True),
        ),
        "uav_01": _local(
            candidates=("uav_00", "uav_02"),
            effective=("uav_00",),
            potential=("uav_00", "uav_02"),
            mask=(True, True, False),
        ),
        "uav_02": _local(candidates=("uav_00",), effective=("uav_00",), potential=("uav_00",)),
    }
    first = controlled_communication_actions(observations, 1)
    second = controlled_communication_actions(observations, 2)
    assert first["uav_00"]["measurement_uav_id"] == "uav_02"
    assert second["uav_00"]["measurement_uav_id"] == "uav_02"
    assert first["uav_01"]["measurement_uav_id"] == "uav_00"
    assert first["uav_02"]["measurement_uav_id"] == "uav_00"
    assert set(first) == set(UIDS)
    assert all(set(action) == {"measurement_uav_id", "bandwidth"} for action in first.values())
    assert first["uav_00"]["bandwidth"] == {"uav_01": 0.5, "uav_02": 0.5}
    assert first["uav_01"]["bandwidth"] == {"uav_00": 0.5, "uav_02": 0.5}


def test_controlled_actions_exclude_self_none_fallback_and_cut_only_two_edges():
    observations = {
        uid: _local(
            candidates=(uid, "uav_01", "uav_02"),
            effective=(uid, "uav_01", "uav_02"),
            potential=("uav_00", "uav_01", "uav_02"),
        )
        for uid in UIDS
    }
    actions = controlled_communication_actions(observations, 20, cut_enabled=True)
    assert actions["uav_00"]["measurement_uav_id"] == "uav_02"
    assert actions["uav_01"]["measurement_uav_id"] == "uav_02"
    assert actions["uav_02"]["measurement_uav_id"] == "uav_01"
    assert actions["uav_00"]["bandwidth"] == {"uav_01": 0.5, "uav_02": 0.0}
    assert actions["uav_02"]["bandwidth"] == {"uav_00": 0.0, "uav_01": 0.5}
    assert controlled_communication_actions(observations, 51, cut_enabled=True)["uav_00"]["bandwidth"]["uav_02"] == 0.5
    none = controlled_communication_actions(
        {uid: _local(candidates=(uid,), effective=(uid,), potential=()) for uid in UIDS}, 1
    )
    assert all(action["measurement_uav_id"] == "none" for action in none.values())
    assert all(action["bandwidth"] == {} for action in none.values())


def test_joint_closest_approach_is_synchronized_and_detects_midpoint_crossing():
    pre = {"uav_00": [0, 0, 0], "uav_01": [2, 0, 0], "uav_02": [10, 10, 10]}
    post = {"uav_00": [2, 0, 0], "uav_01": [0, 0, 0], "uav_02": [10, 10, 10]}
    rows = joint_closest_approach(pre, post)
    pair = next(row for row in rows if (row["source"], row["target"]) == ("uav_00", "uav_01"))
    assert pair["time_fraction"] == pytest.approx(0.5)
    assert pair["distance_m"] == pytest.approx(0.0)
    stationary = joint_closest_approach(pre, pre)
    assert all(row["time_fraction"] == 0.0 for row in stationary)
    assert all(math.isfinite(row["distance_m"]) for row in stationary)
    with pytest.raises(ValueError):
        joint_closest_approach({**pre, "uav_02": [math.nan, 0, 0]}, post)


def test_strict_parking_tracker_individual_and_joint_holds_are_consecutive_and_sticky():
    tracker = StrictParkingTracker(hold_steps=5)
    distances = {uid: 1.0 for uid in UIDS}
    speeds = {uid: 0.5 for uid in UIDS}
    statuses = [tracker.update(distances, speeds) for _ in range(5)]
    assert statuses[0]["per_uav"]["uav_00"]["first_qualifying_step"] == 1
    assert statuses[4]["per_uav"]["uav_00"]["first_success_step"] == 5
    assert statuses[4]["joint_first_success_step"] == 5
    assert statuses[4]["joint_ever_success"] is True
    lost = tracker.update({uid: 1.01 for uid in UIDS}, speeds)
    assert lost["joint_hold_steps"] == 0
    assert lost["per_uav"]["uav_00"]["ever_success"] is True
    assert lost["per_uav"]["uav_00"]["hold_steps"] == 0
    recovered = tracker.update(distances, speeds)
    assert recovered["per_uav"]["uav_00"]["hold_steps"] == 1
    assert recovered["step"] == 7


def test_synchronized_distance_does_not_match_different_vehicle_times():
    pre = {"uav_00": [0, 0, 0], "uav_01": [1, -1, 0], "uav_02": [10, 10, 10]}
    post = {"uav_00": [2, 0, 0], "uav_01": [1, 3, 0], "uav_02": [10, 10, 10]}
    pair = joint_closest_approach(pre, post)[0]
    assert pair['time_fraction'] == pytest.approx(.3)
    assert pair['distance_m'] == pytest.approx(math.sqrt(.2))


def test_strict_parking_tracker_requires_simultaneous_three_uav_hold_and_latches_collision():
    tracker = StrictParkingTracker(hold_steps=2)
    good = {uid: 0.0 for uid in UIDS}
    one_bad = {uid: 0.0 for uid in UIDS}
    one_bad["uav_02"] = 2.0
    tracker.update(good, good)
    after_bad = tracker.update(one_bad, good)
    assert after_bad["joint_hold_steps"] == 0
    assert all(after_bad["per_uav"][uid]["hold_steps"] == (2 if uid != "uav_02" else 0) for uid in UIDS)
    collision = tracker.update(good, good, collision=True)
    assert all(not collision["per_uav"][uid]["qualifying"] for uid in UIDS)
    assert collision["joint_hold_steps"] == 0
    for _ in range(3):
        status = tracker.update(good, good, collision=False)
    assert all(not status["per_uav"][uid]["qualifying"] for uid in UIDS)
    assert status["joint_ever_success"] is False


def test_strict_parking_tracker_rejects_nonfinite_or_negative_inputs():
    tracker = StrictParkingTracker()
    good = {uid: 0.0 for uid in UIDS}
    with pytest.raises(ValueError):
        tracker.update({**good, "uav_00": math.nan}, good)
    with pytest.raises(ValueError):
        tracker.update(good, {**good, "uav_00": -1.0})
    with pytest.raises(ValueError):
        tracker.update({"uav_00": 0.0}, good)
