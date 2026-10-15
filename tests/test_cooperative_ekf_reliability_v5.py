"""Pure v5 diagnostic contracts; these tests never construct an environment."""
from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from src.experiments import three_uav_cooperative_ekf_reliability_v5 as v5


def _local(uid: str = "uav_00", *, legal: tuple[str, ...] = ("uav_01", "uav_02")) -> dict:
    messages = {
        source: {"estimate": np.arange(15, dtype=float) + index, "covariance": np.eye(15)}
        for index, source in enumerate(("uav_01", "uav_02"), start=1)
        if source in legal
    }
    return {
        "uav_id": uid,
        "action_mask": np.ones(4, dtype=bool),
        "base_action_mask": np.ones(4, dtype=bool),
        "candidate_measurement_sources": list(legal),
        "effective_sources": list(legal),
        "uwb_available": {source: True for source in legal},
        "measurement_update_eligibility": {source: True for source in legal},
        "current_budget_remaining": 30,
        "estimate": np.zeros(15, dtype=float),
        "covariance": np.eye(15),
        "neighbor_messages": messages,
    }


def test_fixed_and_random_selectors_cannot_honor_a_forged_mask() -> None:
    # uav_01 is the fixed preference, but the native physical/message contract
    # exposes only uav_02.  A forged all-true action mask must not widen it.
    local = _local(legal=("uav_02",))
    assert v5.select_source("fixed", local, step=2, env_seed=2026116100, credit_scale=0.08, range_variance_m2=0.0064) == "none"
    assert v5.select_source("random_peer", local, step=2, env_seed=2026116100, credit_scale=0.08, range_variance_m2=0.0064) == "uav_02"


def test_random_peer_uses_indexed_rng_without_none_when_a_peer_is_legal() -> None:
    local = _local(legal=("uav_01", "uav_02"))
    first = v5.select_source("random_peer", local, step=18, env_seed=2026116100, credit_scale=0.08, range_variance_m2=0.0064)
    second = v5.select_source("random_peer", copy.deepcopy(local), step=18, env_seed=2026116100, credit_scale=0.08, range_variance_m2=0.0064)
    assert first == second
    assert first in {"uav_01", "uav_02"}


def test_branch_window_helper_stops_at_earliest_two_windows() -> None:
    rows = [
        {"step": 2, "uav_id": "uav_00", "legal_sources": ("uav_01",)},
        {"step": 6, "uav_id": "uav_00", "legal_sources": ("uav_01", "uav_02")},
        {"step": 10, "uav_id": "uav_00", "legal_sources": ("uav_02", "uav_01")},
        {"step": 14, "uav_id": "uav_00", "legal_sources": ("uav_01", "uav_02")},
    ]
    selected = v5.select_branch_windows(rows)
    assert [row["step"] for row in selected] == [6, 10]


def test_metrics_keep_observed_denominator_under_censoring() -> None:
    rows = [
        {"step": 1, "uav_id": "uav_00", "completed_transition": True, "position_error_m": 1.0, "velocity_error_mps": 2.0, "position_error_vector_m": np.array([1.0, 0.0, 0.0]), "nees_position_3d": 1.0, "nees_position_velocity_6d": 2.0, "vo_nis_attempted": 1, "vo_nis_accepted": 1, "uwb_nis_attempted": 0, "uwb_nis_accepted": 0},
        {"step": 2, "uav_id": "uav_00", "completed_transition": True, "position_error_m": 3.0, "velocity_error_mps": 4.0, "position_error_vector_m": np.array([3.0, 0.0, 0.0]), "nees_position_3d": 9.0, "nees_position_velocity_6d": 4.0, "vo_nis_attempted": 1, "vo_nis_accepted": 0, "uwb_nis_attempted": 1, "uwb_nis_accepted": 1},
    ]
    result = v5.trajectory_summary(rows, horizon=120, censored=True, censor_reason="collision")
    assert result["observed_steps"] == 2
    assert result["uav_00"]["position_denominator"] == 2
    assert result["uav_00"]["vo_nis_attempted"] == 2
    assert v5.paired_post_improvement(4.0, 2.5) == 1.5


def test_parser_is_pure_and_defaults_to_locked_config() -> None:
    parser = v5.build_parser()
    args = parser.parse_args(["--run-id", "pure_test"])
    assert args.config == v5.CONFIG
    assert args.run_id == "pure_test"


def test_locked_run_matrix_covers_all_crossed_scene_seed_arm_rows() -> None:
    spec = v5.read_json(v5.CONFIG)
    rows = spec["run_matrix"]
    assert len(rows) == 16
    assert {(row["scene_id"], int(row["seed"])) for row in rows} == {
        ("train_compact_x", 2026116100),
        ("train_compact_x", 2026116102),
        ("train_bridge_x", 2026116100),
        ("train_bridge_x", 2026116102),
    }
    assert {row["arm"] for row in rows} == {"none", "fixed", "random_peer", "greedy"}
    assert all("scenario" in row for row in rows)


def test_normalized_summary_is_safe_json_without_missing_uid_nan() -> None:
    rows = [{
        "step": 1,
        "uav_id": "uav_00",
        "completed_transition": True,
        "position_error_m": 1.0,
        "velocity_error_mps": 1.0,
        "position_error_vector_m": np.array([1.0, 0.0, 0.0]),
        "nees_position_3d": 1.0,
        "nees_position_velocity_6d": 1.0,
        "vo_nis_attempted": 0,
        "vo_nis_accepted": 0,
        "uwb_nis_attempted": 0,
        "uwb_nis_accepted": 0,
    }]
    summary = v5.trajectory_summary(rows, horizon=120, censored=True, censor_reason="collision")
    encoded = v5.encode(summary)
    decoded = json.loads(encoded)
    assert decoded["uav_01"]["position_rmse_m"] is None
    assert decoded["uav_02"]["position_bias_xyz_m"] is None


def test_protected_file_content_rehash_rejects_change() -> None:
    import uuid
    target = v5.ROOT / ".tmp" / ("v5_test_protected_" + uuid.uuid4().hex + ".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text("before\n", encoding="utf-8")
        expected = {str(target): v5.file_hash(target)}
        target.write_text("after\n", encoding="utf-8")
        with pytest.raises(ValueError, match="protected input changed"):
            v5.verify_protected_inputs(expected)
    finally:
        target.unlink(missing_ok=True)


def test_snapshot_hash_serializes_native_topology_sets_deterministically() -> None:
    from src.simulation.topology import TopologySnapshot
    left = TopologySnapshot(potential_edges={("uav_01", "uav_00"), ("uav_02", "uav_00")})
    right = TopologySnapshot(potential_edges={("uav_02", "uav_00"), ("uav_01", "uav_00")})
    assert v5.content_hash(left) == v5.content_hash(right)
    assert json.loads(v5.encode(left))["potential_edges"] == [["uav_01", "uav_00"], ["uav_02", "uav_00"]]


def test_v5_training_guard_blocks_normalizer_fit_without_environment() -> None:
    with v5._forbid_training_runtime():
        with pytest.raises(RuntimeError, match="Forbidden v5 training operation"):
            v5.v3.FeatureNormalizer().fit(np.zeros((2, 3)))


def test_branch_and_episode_ids_are_joinable() -> None:
    case = {"scene_id": "train_compact_x", "seed": 2026116100}
    assert v5.episode_id("none", case) == "none_train_compact_x_s2026116100"
