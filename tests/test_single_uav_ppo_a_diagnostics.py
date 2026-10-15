"""Pure contract checks for the PPO-A scenario and witness helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.single_uav_ppo_a_scenarios import (
    build_a_foundation_scenarios,
    build_open_loop_witness,
)


def test_foundation_manifest_cardinality_and_freshness() -> None:
    manifest = build_a_foundation_scenarios()
    assert len(manifest["train_pool"]) == 18
    assert len(manifest["normal_train"]) == 6
    assert len(manifest["validation_normal"]) == 8
    assert len(manifest["validation_single"]) == 4
    assert len(manifest["validation_full"]) == 4
    assert len({item["episode_id"] for item in manifest["validation_normal"]}) == 8
    assert all(item["scenario"]["foundation_task_kind"] == "normal" for item in manifest["validation_normal"])
    assert all(not item["scenario"]["obstacles"] and not item["scenario"]["obstacle_specs"] for item in manifest["validation_normal"])


def test_open_loop_witness_is_pure_bounded_for_normal_validation() -> None:
    manifest = build_a_foundation_scenarios()
    for scene in manifest["normal_train"] + manifest["validation_normal"]:
        witness = build_open_loop_witness(scene, {})
        actions = np.asarray(witness["actions"], dtype=float)
        positions = np.asarray(witness["predicted_positions"], dtype=float)
        goal = np.asarray(scene["scenario"]["goals"]["uav_00"], dtype=float)
        assert actions.shape[1:] == (3,)
        assert actions.shape[0] <= 40
        assert np.max(np.linalg.norm(actions, axis=1)) <= 0.6 + 1e-8
        assert np.linalg.norm(positions[30] - goal) <= 1e-7
        assert np.all(np.isfinite(actions))
        assert np.all(np.isfinite(positions))


def test_open_loop_rejects_missing_state() -> None:
    with pytest.raises(ValueError):
        build_open_loop_witness({"scenario": {}}, {})
