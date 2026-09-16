from __future__ import annotations

import numpy as np

from src.experiments.single_uav_stage2br4_bc_policy_preservation import (
    _distance_bin,
    _observation_vector,
    _parameter_group,
    _stratified_resample,
)


def _sample(distance: float, phase: str, index: int) -> dict:
    return {
        "episode_id": f"episode_{index}",
        "scenario_hash": f"scenario_{index}",
        "episode_index": index,
        "goal_distance_m": distance,
        "phase": phase,
        "kind": "normal_far_distance",
        "student_closed_loop": True,
        "observation": {
            "node_features": [[0.0] * 42],
            "action_mask": [False, True],
            "bandwidth_mask": [False],
            "remaining_time_fraction": 1.0,
        },
        "teacher_action_delta_v_mps": [0.0, 0.0, 0.0],
    }


def test_distance_bins_keep_boundary_semantics():
    assert _distance_bin(8.1) == ">8m"
    assert _distance_bin(8.0) == "8-5m"
    assert _distance_bin(5.0) == "5-3m"
    assert _distance_bin(3.0) == "3-1m"
    assert _distance_bin(1.0) == "<=1m"


def test_observation_vector_is_fixed_actor_visible_contract():
    vector = _observation_vector(_sample(10.0, "normal", 1)["observation"])
    assert vector.shape == (46,)
    assert np.isfinite(vector).all()


def test_parameter_groups_match_current_actor_names():
    assert _parameter_group("gat.layers.0.node_projection.weight") == "GAT"
    assert _parameter_group("actor_body.0.weight") == "actor_body_early"
    assert _parameter_group("actor_body.4.weight") == "actor_body_late"
    assert _parameter_group("delta_mean_head.bias") == "delta_mean_head"


def test_distance_phase_resampling_is_seeded_and_preserves_trajectory_ids():
    samples = [_sample(distance, phase, index) for index, (distance, phase) in enumerate([
        (20.0, "normal"),
        (7.0, "bypass"),
        (4.0, "recovery"),
        (2.0, "post_bypass"),
    ])]
    first, meta_first = _stratified_resample(samples, 12, seed=11)
    second, meta_second = _stratified_resample(samples, 12, seed=11)
    assert first == second
    assert meta_first == meta_second
    assert len(first) == 12
    assert {row["episode_id"] for row in first} <= {row["episode_id"] for row in samples}
