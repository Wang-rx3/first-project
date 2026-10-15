from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict

import pytest

from src.experiments.single_uav_ppo_smoke_curriculum import (
    BOUNDS_MAX_M,
    BOUNDS_MIN_M,
    build_smoke_curriculum,
)
from src.experiments.single_uav_ppo_baseline import (
    SingleUAVPPOConfig,
    create_single_uav_ppo_environment,
    load_single_uav_ppo_config,
)
from src.simulation.obstacle_model import BoxObstacle, is_physical_obstacle


def _all_cases(curriculum: dict[str, object]) -> list[dict[str, object]]:
    return list(curriculum["train_pool"]) + list(curriculum["validation_pool"])


def test_curriculum_is_deterministic_and_json_serializable() -> None:
    first = build_smoke_curriculum(2026)
    second = build_smoke_curriculum(2026)
    assert first == second
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert build_smoke_curriculum(2027)["train_pool"] != first["train_pool"]


def test_pool_counts_balance_and_split_provenance() -> None:
    curriculum = build_smoke_curriculum()
    train = curriculum["train_pool"]
    validation = curriculum["validation_pool"]
    assert len(train) == 18
    assert len(validation) == 10
    assert Counter(case["kind"] for case in train) == {"normal": 6, "single": 6, "full": 6}
    assert Counter(case["kind"] for case in validation) == {"normal": 2, "single": 4, "full": 4}
    assert {case["split"] for case in train} == {"train"}
    assert {case["split"] for case in validation} == {"validation"}
    for field in ("episode_id", "scenario_id", "scenario_seed", "noise_seed", "geometry_sha256"):
        assert {case[field] for case in train}.isdisjoint({case[field] for case in validation})
    provenance = curriculum["provenance"]
    assert provenance["available_pool_vs_actually_exposed_distinct"] is True
    assert len(provenance["calibration_only_case_ids"]) == 6
    assert all(case_id in provenance["unscheduled_available_case_ids"] for case_id in provenance["calibration_only_case_ids"])


def test_reset_schema_bounds_and_direct_box_intersections() -> None:
    curriculum = build_smoke_curriculum()
    for case in _all_cases(curriculum):
        scenario = case["scenario"]
        expected_reset_keys = {
            "initial_positions",
            "initial_velocities",
            "initial_attitudes",
            "goals",
            "obstacle_specs",
        }
        assert set(scenario) == expected_reset_keys | ({"obstacles"} if case["kind"] == "normal" else set())
        for field in ("initial_positions", "goals"):
            vector = scenario[field]["uav_00"]
            assert all(low <= value <= high for value, low, high in zip(vector, BOUNDS_MIN_M, BOUNDS_MAX_M))
        obstacles = scenario["obstacle_specs"]
        if case["kind"] == "normal":
            assert obstacles == []
        else:
            assert len(obstacles) == 1
            obstacle = obstacles[0]
            assert set(obstacle) == {
                "obstacle_id",
                "geometry_type",
                "center_m",
                "size_m",
                "physical_collision_enabled",
            }
            assert obstacle["geometry_type"] == "box"
            assert obstacle["physical_collision_enabled"] is True
            start = scenario["initial_positions"]["uav_00"]
            goal = scenario["goals"]["uav_00"]
            center, size = obstacle["center_m"], obstacle["size_m"]
            x_low, x_high = center[0] - size[0] / 2, center[0] + size[0] / 2
            y_low, y_high = center[1] - size[1] / 2, center[1] + size[1] / 2
            assert x_low <= start[0] <= x_high or x_low <= goal[0] <= x_high or (
                min(start[0], goal[0]) <= x_high and max(start[0], goal[0]) >= x_low
            )
            assert y_low <= start[1] <= y_high
            assert y_low <= goal[1] <= y_high


def test_mirror_pairs_reflect_positions_goals_boxes_velocities_and_attitudes() -> None:
    curriculum = build_smoke_curriculum()
    for pool in (curriculum["train_pool"], curriculum["validation_pool"]):
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for case in pool:
            if case["mirror_pair"] is not None:
                groups[case["mirror_pair"]].append(case)
        assert all(len(pair) == 2 for pair in groups.values())
        for pair in groups.values():
            left = next(case for case in pair if case["side"] == "left")
            right = next(case for case in pair if case["side"] == "right")
            for field in ("initial_positions", "goals"):
                lvec = left["scenario"][field]["uav_00"]
                rvec = right["scenario"][field]["uav_00"]
                assert rvec[0] == lvec[0] and rvec[2] == lvec[2]
                assert rvec[1] == pytest.approx(200.0 - lvec[1])
            for field in ("initial_velocities", "initial_attitudes"):
                lvec = left["scenario"][field]["uav_00"]
                rvec = right["scenario"][field]["uav_00"]
                assert rvec[0] == lvec[0] and rvec[2] == lvec[2]
                assert rvec[1] == pytest.approx(-lvec[1])
            lbox = left["scenario"]["obstacle_specs"][0]
            rbox = right["scenario"]["obstacle_specs"][0]
            assert rbox["center_m"][0] == lbox["center_m"][0]
            assert rbox["center_m"][2] == lbox["center_m"][2]
            assert rbox["center_m"][1] == pytest.approx(200.0 - lbox["center_m"][1])
            assert rbox["size_m"] == lbox["size_m"]
            assert lbox["obstacle_id"] != rbox["obstacle_id"]


def test_geometry_hash_is_independent_of_obstacle_identifier() -> None:
    curriculum = build_smoke_curriculum()
    case = next(case for case in curriculum["train_pool"] if case["kind"] == "single")
    changed = copy.deepcopy(case["scenario"])
    changed["obstacle_specs"][0]["obstacle_id"] = "renamed_without_geometry_change"
    # The hash itself is defined over canonical geometry; changing an ID must
    # not make split leakage checks pass accidentally.
    from src.experiments.single_uav_ppo_smoke_curriculum import _geometry_hash

    assert _geometry_hash(case["scenario"]) == _geometry_hash(changed)


def test_schedule_stage_membership_order_and_budget() -> None:
    curriculum = build_smoke_curriculum()
    schedule = curriculum["schedule"]
    assert len(schedule) == 18
    assert [item["global_update"] for item in schedule] == list(range(1, 19))
    assert [item["stage_name"] for item in schedule] == ["stage1_normal"] * 6 + ["stage2_single"] * 6 + ["stage3_full"] * 6
    assert [item["kind"] for item in schedule[:6]] == ["normal"] * 6
    assert [item["kind"] for item in schedule[6:12]] == ["normal", "normal", "single", "single", "single", "single"]
    assert [item["side"] for item in schedule[6:12]] == ["none", "none", "left", "right", "left", "right"]
    assert [item["kind"] for item in schedule[12:]] == ["normal", "normal", "single", "single", "full", "full"]
    assert [item["side"] for item in schedule[12:]] == ["none", "none", "left", "right", "left", "right"]
    assert all(item["scenario_id"] for item in schedule)
    assert all(set(item) == {"global_update", "stage", "update_in_stage", "stage_name", "scenario_id", "kind", "side", "rollout_steps", "arms", "shared_both_arms"} for item in schedule)
    assert all(item["rollout_steps"] == 120 and item["shared_both_arms"] for item in schedule)
    assert all(item["arms"] == ["A", "B"] for item in schedule)
    assert sum(item["rollout_steps"] for item in schedule) == 2160
    assert curriculum["provenance"]["schedule_assertions"]["train_only_calibration_steps"] == 288
    exposed = {item["scenario_id"] for item in schedule}
    provenance = curriculum["provenance"]
    assert exposed == set(provenance["actually_exposed_case_ids"])
    assert exposed.issubset(set(provenance["available_train_case_ids"]))


def test_invalid_root_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_smoke_curriculum(-1)
    with pytest.raises(ValueError):
        build_smoke_curriculum(True)


def test_frozen_runtime_reset_uses_explicit_empty_normals_and_one_static_box() -> None:
    curriculum = build_smoke_curriculum()
    config = load_single_uav_ppo_config("data/configs/single_uav_ppo_baseline_v1.json")
    assert isinstance(config, SingleUAVPPOConfig)
    environment = create_single_uav_ppo_environment(config)
    for case in _all_cases(curriculum):
        environment.reset(seed=case["scenario_seed"], scenario_config=case["scenario"])
        if case["kind"] == "normal":
            assert environment.obstacles == []
            assert environment._strict_relevant_obstacles == {}
            continue
        assert len(environment.obstacles) == 1
        obstacle = environment.obstacles[0]
        assert isinstance(obstacle, BoxObstacle)
        assert is_physical_obstacle(obstacle)
        assert obstacle.velocity_mps is not None
        assert float((obstacle.velocity_mps ** 2).sum()) == pytest.approx(0.0)
        assert set(environment._strict_relevant_obstacles) == {str(obstacle.obstacle_id)}
