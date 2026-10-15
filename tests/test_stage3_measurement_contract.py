"""Pure tests for the Stage 3 three-UAV measurement contract."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from src.experiments.three_uav_measurement_selection_contract import (
    UIDS,
    admissible_mask,
    admissible_observations,
    calibration_readiness,
    communication_actions,
    expected_relative_credit,
    greedy_selection,
    make_scene_splits,
    random_selection,
    relative_linear_credit,
    reward_components,
    round_robin_selection,
)


def _message(position: tuple[float, float, float], covariance_scale: float = 1.0) -> dict[str, np.ndarray]:
    estimate = np.zeros(15, dtype=float)
    estimate[:3] = position
    return {"estimate": estimate, "covariance": covariance_scale * np.eye(15, dtype=float)}


def _local(uid: str = "uav_00", *, budget: float = 1.0, eligibility=True) -> dict[str, object]:
    return {
        "uav_id": uid,
        "candidate_measurement_sources": [source for source in UIDS if source != uid],
        "effective_sources": [source for source in UIDS if source != uid],
        "uwb_available": {source: source != uid for source in UIDS},
        "measurement_update_eligibility": np.asarray([eligibility] * 3, dtype=bool),
        "current_budget_remaining": budget,
        "potential_targets": [source for source in UIDS if source != uid],
        "action_mask": np.ones(4, dtype=bool),
        "estimate": _message((0.0, 0.0, 0.0))["estimate"],
        "covariance": np.eye(15, dtype=float),
        "neighbor_messages": {
            "uav_01": _message((10.0, 0.0, 0.0)),
            "uav_02": _message((0.0, 10.0, 0.0), 2.0),
        },
    }


def test_scene_splits_are_exactly_seeded_bounded_disjoint_and_safe():
    splits = make_scene_splits()
    assert list(splits) == ["train", "validation"]
    assert len(splits["train"]) == 4
    assert len(splits["validation"]) == 3
    assert [row["seed"] for row in splits["train"]] == [2026116100, 2026116101, 2026116102, 2026116103]
    assert [row["seed"] for row in splits["validation"]] == [2026116400, 2026116401, 2026116402]
    origins = []
    train_origins = set()
    for split, rows in splits.items():
        for row in rows:
            assert set(row) == {"scene_id", "split", "seed", "requires_graph_change", "scenario", "geometry_sha256"}
            scenario = row["scenario"]
            assert set(scenario) == {
                "initial_positions", "initial_velocities", "initial_attitudes", "goals", "obstacles", "obstacle_specs"
            }
            assert row["split"] == split
            assert len(row["geometry_sha256"]) == 64
            for uid in UIDS:
                start = np.asarray(scenario["initial_positions"][uid], dtype=float)
                goal = np.asarray(scenario["goals"][uid], dtype=float)
                assert np.all(start >= (0, 0, 2)) and np.all(start <= (200, 200, 60))
                assert np.all(goal >= (0, 0, 2)) and np.all(goal <= (200, 200, 60))
                assert 45.0 <= np.linalg.norm(goal - start) <= 55.0
                origins.append(tuple(start))
                if split == "train":
                    train_origins.add(tuple(start))
            if split == "validation":
                assert not any(tuple(scenario["initial_positions"][uid]) in train_origins for uid in UIDS)
    assert len(origins) == len(set(origins))


def test_admissibility_fails_closed_until_budget_eligibility_and_effective_data_arrive():
    local = _local(eligibility=False)
    assert np.array_equal(admissible_mask(local), np.asarray([False, False, False, True]))
    local["measurement_update_eligibility"] = np.asarray([True] * 3)
    local["current_budget_remaining"] = 0.0
    assert np.array_equal(admissible_mask(local), np.asarray([False, False, False, True]))
    local["current_budget_remaining"] = 1.0
    local["effective_sources"] = []
    assert np.array_equal(admissible_mask(local), np.asarray([False, False, False, True]))
    local["effective_sources"] = ["uav_01", "uav_02"]
    assert np.array_equal(admissible_mask(local), np.asarray([False, True, True, True]))
    local["neighbor_messages"]["uav_01"]["covariance"][0, 0] = -2.0
    assert not bool(admissible_mask(local)[1])


def test_observation_copy_and_action_validation_are_local_and_fail_closed():
    observations = {uid: _local(uid) for uid in UIDS}
    original = copy.deepcopy(observations)
    updated = admissible_observations(observations)
    assert set(updated) == set(UIDS)
    assert all(updated[uid]["uav_id"] == uid for uid in UIDS)
    assert all(updated[uid]["action_mask"].shape == (4,) for uid in UIDS)
    assert np.array_equal(observations["uav_00"]["estimate"], original["uav_00"]["estimate"])
    observations["uav_00"]["estimate"][0] = 99.0
    assert updated["uav_00"]["estimate"][0] == 0.0
    actions = communication_actions(observations, {uid: "uav_01" for uid in UIDS})
    assert actions["uav_00"]["measurement_uav_id"] == "uav_01"
    assert actions["uav_00"]["bandwidth"] == {"uav_01": 0.5, "uav_02": 0.5}
    blocked = dict(observations["uav_00"])
    blocked["current_budget_remaining"] = 0.0
    blocked_obs = dict(observations)
    blocked_obs["uav_00"] = blocked
    assert communication_actions(blocked_obs, {uid: "uav_01" for uid in UIDS})["uav_00"]["measurement_uav_id"] == "none"
    with pytest.raises(ValueError):
        communication_actions(observations, {"uav_00": "uav_01"})


def test_random_and_round_robin_only_use_current_shared_mask():
    local = _local()
    rng = np.random.default_rng(7)
    values = {random_selection(local, rng) for _ in range(300)}
    assert values <= {"uav_01", "uav_02", "none"}
    assert {"uav_01", "uav_02", "none"} <= values
    assert [round_robin_selection(local, step) for step in range(1, 5)] == ["uav_01", "uav_02", "uav_01", "uav_02"]
    assert round_robin_selection(_local(budget=0.0), 1) == "none"


def test_expected_credit_uses_direction_and_rejects_invalid_range_variance():
    local = _local()
    x_credit = expected_relative_credit(local, "uav_01", 1.0)
    y_credit = expected_relative_credit(local, "uav_02", 1.0)
    assert x_credit > 0.0 and y_credit >= 0.0
    changed = _local()
    changed["neighbor_messages"]["uav_01"]["covariance"] = 100.0 * np.eye(15)
    assert expected_relative_credit(changed, "uav_01", 1.0) < x_credit
    assert greedy_selection(local, 1.0, 1.0) in {"uav_01", "uav_02", "none"}
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            expected_relative_credit(local, "uav_01", bad)


def test_linear_credit_and_reward_use_accepted_proxy_actual_packets_and_shared_team_value():
    assert relative_linear_credit({"accepted": False, "position_covariance_trace_before_m2": 3.0, "position_covariance_trace_after_linear_m2": 1.0}) == 0.0
    assert relative_linear_credit({"accepted": True, "position_covariance_trace_before_m2": 3.0, "position_covariance_trace_after_linear_m2": 2.0, "position_covariance_trace_after_final_m2": 3.0}) == pytest.approx(1.0 / 3.0)
    info = {
        "learning_rewards": {uid: 999.0 for uid in UIDS},
        "relative_range_update_events": [{
            "target_uav_id": "uav_00", "accepted": True,
            "position_covariance_trace_before_m2": 3.0,
            "position_covariance_trace_after_linear_m2": 2.0,
            "position_covariance_trace_after_final_m2": 3.0,
        }],
        "measurement_budget_events": [{"uav_id": "uav_00", "budget_consumed": 1}],
        "measurement_overhead_events": [{"target_uav_id": "uav_00", "load_packets_applied": 1}],
    }
    rewards, terms = reward_components(info, 1.0)
    assert rewards["uav_00"] == rewards["uav_01"] == rewards["uav_02"]
    assert terms["uav_00"]["actual_overhead_bytes_equivalent"] == 512.0
    assert terms["uav_00"]["wasted_budget_units"] == 0.0
    poisoned = copy.deepcopy(info)
    poisoned["learning_rewards"] = {uid: -1e9 for uid in UIDS}
    assert reward_components(poisoned, 1.0) == reward_components(info, 1.0)
    dropped = copy.deepcopy(info)
    dropped["measurement_overhead_events"] = []
    _, dropped_terms = reward_components(dropped, 1.0)
    assert dropped_terms["uav_00"]["budget_consumed"] == 1.0
    assert dropped_terms["uav_00"]["actual_overhead_packets"] == 0.0
    assert dropped_terms["uav_00"]["wasted_budget_units"] == 1.0


def _calibration_inputs(*, weak_scene: str | None = None, positive_count: int = 30):
    scenes = [
        {"scene_id": f"train_{index}", "split": "train", "moving_fractions": {uid: 0.7 for uid in UIDS}, "physical_graph_changes": 1 if index == 0 else 0, "requires_graph_change": index == 0}
        for index in range(4)
    ]
    if weak_scene is not None:
        next(row for row in scenes if row["scene_id"] == weak_scene)["moving_fractions"]["uav_02"] = 0.49
    decisions = [{"scene_id": "train_0", "uav_id": "uav_00", "legal_peer_count": 2, "expected_credit_spread": 0.2, "greedy_positive": index == 0} for index in range(30)]
    return scenes, decisions, [0.1] * positive_count


def test_calibration_gates_are_train_scoped_and_report_both_separation_denominators():
    scenes, decisions, credits = _calibration_inputs()
    gates = {"min_moving_fraction": 0.5, "min_positive_credit_events": 30, "min_separated_decisions": 30, "expected_credit_spread_tolerance": 1e-6, "min_greedy_positive_decisions": 1}
    result = calibration_readiness(scenes, decisions, credits, 1.0, gates)
    assert result["ready"] is True
    assert result["aggregate"]["separated_decision_count"] == 30
    assert result["aggregate"]["separated_fraction_multi_peer"] == 1.0
    assert result["aggregate"]["separated_fraction_all_decisions"] == 1.0
    assert result["aggregate"]["physical_graph_changes_total"] == 1
    assert calibration_readiness(scenes, decisions, credits[:-1], 1.0, gates)["ready"] is False
    weak = calibration_readiness(_calibration_inputs(weak_scene="train_2")[0], decisions, credits, 1.0, gates)
    assert weak["ready"] is False
    val_only = [dict(row, split="validation") for row in scenes]
    assert calibration_readiness(val_only, decisions, credits, 1.0, gates)["ready"] is False
