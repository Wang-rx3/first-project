from __future__ import annotations

import numpy as np

from src.experiments.single_uav_obstacle_observation_audit import (
    actor_node_feature_layout,
    probe_scenarios,
    read_only_integrity_report,
)
from src.simulation.motion_model import MotionConfig


def test_direction_probe_changes_only_obstacle_direction() -> None:
    scenarios = probe_scenarios()
    reference = scenarios["front"]
    reference_uav = np.asarray(reference["initial_positions"]["uav_00"], dtype=float)
    reference_goal = np.asarray(reference["goals"]["uav_00"], dtype=float)

    centers: dict[str, np.ndarray] = {}
    for name, scenario in scenarios.items():
        assert scenario["initial_positions"] == reference["initial_positions"]
        assert scenario["initial_velocities"] == reference["initial_velocities"]
        assert scenario["goals"] == reference["goals"]
        obstacle = scenario["obstacle_specs"][0]
        center = np.asarray(obstacle["center_m"], dtype=float)
        centers[name] = center
        assert np.isclose(np.linalg.norm(center - reference_uav), 6.0)
        assert obstacle["radius_m"] == 1.0
        assert obstacle["obstacle_type"] == "static"
        assert np.allclose(reference_goal, [80.0, 100.0, 20.0])

    assert centers["left"][1] > centers["right"][1]
    assert centers["left_front"][1] > centers["right_front"][1]
    assert centers["front"][0] > centers["behind"][0]


def test_current_actor_layout_has_scalar_obstacle_aggregate_only() -> None:
    rows = actor_node_feature_layout(39)
    labels = [str(row["feature"]) for row in rows]
    assert len(rows) == 39
    assert "minimum_obstacle_clearance" in labels
    assert "obstacle_box_count" in labels
    assert "obstacle_sphere_count" in labels
    assert not any("obstacle_relative" in label for label in labels)
    assert not any("obstacle_bearing" in label for label in labels)


def test_motion_contract_has_lateral_and_braking_authority() -> None:
    contract = MotionConfig().action_contract()
    assert contract["vector_dimension"] == 3
    assert contract["negative_delta_v_allowed"] is True
    assert contract["braking_supported"] is True


def test_read_only_integrity_report_requires_unchanged_hashes() -> None:
    unchanged = read_only_integrity_report(
        policy_hash_before="p",
        policy_hash_after="p",
        checkpoint_sha256_before="c",
        checkpoint_sha256_after="c",
    )
    changed = read_only_integrity_report(
        policy_hash_before="p",
        policy_hash_after="q",
        checkpoint_sha256_before="c",
        checkpoint_sha256_after="d",
    )
    assert unchanged["policy_unchanged"] is True
    assert unchanged["checkpoint_unchanged"] is True
    assert changed["policy_unchanged"] is False
    assert changed["checkpoint_unchanged"] is False
