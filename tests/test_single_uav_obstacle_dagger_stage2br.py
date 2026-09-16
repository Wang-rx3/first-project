from types import SimpleNamespace

import numpy as np

from src.experiments.single_uav_obstacle_dagger_stage2br import (
    STATIC_CATEGORIES,
    _normal_no_obstacle_scenario,
    _round_episode_plan,
    audit_previous_data_bias,
    mirror_pair_manifest_rows,
    mirror_static_scenario,
    round_schedule_counts,
)


def test_stage2br_round_schedule_has_eight_normal_and_four_static() -> None:
    assert round_schedule_counts(3) == {
        "normal_far_distance": 24,
        "static_mirror_pressure": 12,
    }
    plan = _round_episode_plan(1)
    assert len(plan) == 12
    assert sum(item["kind"] == "normal_far_distance" for item in plan) == 8
    assert sum(item["kind"] == "static_mirror_pressure" for item in plan) == 4
    assert {item["category"] for item in plan if item["kind"] != "normal_far_distance"} == set(STATIC_CATEGORIES)
    assert all(item["split"] == "train" for item in plan[:9])
    assert all(item["split"] == "validation" for item in plan[9:])


def test_stage2br_manifest_contains_exact_mirror_pairs() -> None:
    rows = mirror_pair_manifest_rows(rounds=1)
    assert len(rows) == 4
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mirror_pair_id"], []).append(row)
    assert all({item["side"] for item in pair} == {"left", "right"} for pair in grouped.values())
    for pair in grouped.values():
        assert pair[0]["obstacle_dy_abs_m"] == pair[1]["obstacle_dy_abs_m"]
        assert abs(pair[0]["obstacle_dy_body_m"]) == abs(pair[1]["obstacle_dy_body_m"])
        assert pair[0]["obstacle_longitudinal_distance_m"] == pair[1]["obstacle_longitudinal_distance_m"]


def test_stage2br_mirror_geometry_changes_only_lateral_sign() -> None:
    config = SimpleNamespace()
    left = mirror_static_scenario(config, category="left", pair_index=1, side="left", seed=1, round_index=1)
    right = mirror_static_scenario(config, category="right", pair_index=1, side="right", seed=1, round_index=1)
    left_center = np.asarray(left["obstacle_specs"][0]["center_m"], dtype=float)
    right_center = np.asarray(right["obstacle_specs"][0]["center_m"], dtype=float)
    assert np.allclose(left_center[[0, 2]], right_center[[0, 2]])
    assert left_center[1] == -right_center[1] + 200.0
    assert left["obstacle_specs"][0]["radius_m"] == right["obstacle_specs"][0]["radius_m"]


def test_stage2br_normal_scenario_explicitly_disables_default_obstacle_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.single_uav_obstacle_dagger_stage2br._normal_scenario",
        lambda *args, **kwargs: {"obstacle_specs": [{"obstacle_id": "default"}]},
    )
    scenario = _normal_no_obstacle_scenario(
        SimpleNamespace(), seed=1, round_index=1, episode_index=1
    )
    assert scenario["obstacle_specs"] == []
    assert scenario["obstacles"] == []


def test_stage2br_previous_bias_audit_is_read_only_and_classifies_missing_right_queries() -> None:
    report = audit_previous_data_bias(
        "logs/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026_rerun4"
    )
    assert report["previous_aggregation_filtered_right"] is False
    assert report["previous_static_generator_explicit_mirror_pairs"] is False
    assert "mirror-balanced" in report["root_cause_classification"]
