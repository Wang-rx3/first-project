"""Pure and reset-only checks for the A normalization calibration APIs."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from src.experiments.single_uav_ppo_a_norm_calibration import (
    CALIBRATION_OBSERVATIONS,
    EXPECTED_CENTRAL_FEATURE_DIM,
    EXPECTED_NODE_FEATURE_DIM,
    build_split_manifest,
    fit_full_statistics,
    load_full_statistics,
    schema_gate,
)
from src.experiments.single_uav_ppo_a_support import load_shared_a_statistics
from src.experiments.single_uav_ppo_baseline import load_single_uav_ppo_config


def _bank(path) -> None:
    nodes = np.zeros((CALIBRATION_OBSERVATIONS, EXPECTED_NODE_FEATURE_DIM), dtype=np.float64)
    central = np.zeros((CALIBRATION_OBSERVATIONS, EXPECTED_CENTRAL_FEATURE_DIM), dtype=np.float64)
    values = np.linspace(-1.0, 1.0, CALIBRATION_OBSERVATIONS)
    nodes[:, 3] = values
    nodes[:, 4] = -values
    nodes[:, 5] = values * 0.5
    nodes[:, 30] = values * 4.0
    nodes[:, 31] = -values * 4.0
    nodes[:, 32] = values * 2.0
    central[:, :42] = nodes
    episode_ids, scenario_ids, arms, sources, splits, kinds, seeds, steps = [], [], [], [], [], [], [], []
    for arm in ("P", "G"):
        for case in range(12):
            episode = f"calibration_train_{arm}_{case:02d}"
            for step in range(121):
                episode_ids.append(episode)
                scenario_ids.append(f"case_{case:02d}")
                arms.append(arm)
                sources.append(arm)
                splits.append("calibration")
                kinds.append("normal")
                seeds.append(202611210000 + case)
                steps.append(step)
    np.savez(
        path,
        node_features=nodes,
        central_features=central,
        episode_id=np.asarray(episode_ids, dtype="U64"),
        scenario_id=np.asarray(scenario_ids, dtype="U64"),
        arm=np.asarray(arms, dtype="U2"),
        source=np.asarray(sources, dtype="U2"),
        split=np.asarray(splits, dtype="U16"),
        kind=np.asarray(kinds, dtype="U16"),
        noise_seed=np.asarray(seeds, dtype=np.int64),
        step=np.asarray(steps, dtype=np.int64),
    )


def test_manifest_has_balanced_independent_splits() -> None:
    manifest = build_split_manifest()
    assert {key: len(manifest[key]) for key in ("calibration", "train", "validation", "challenge")} == {
        "calibration": 12,
        "train": 6,
        "validation": 8,
        "challenge": 4,
    }
    calibration = manifest["calibration"]
    assert {(item["axis"], item["magnitude_m"]) for item in calibration} == {
        (axis, magnitude) for axis in ("x", "y", "z") for magnitude in (4.0, 8.0)
    } | {
        (axis, magnitude) for axis in ("x", "y", "z") for magnitude in (4.0, 8.0)
    }
    assert all(item["scenario"]["obstacles"] == [] for item in calibration)
    assert all(item["scenario"]["obstacle_specs"] == [] for item in calibration)
    assert {item["challenge_type"] for item in manifest["challenge"]} == {"stationary", "fast"}
    assert all(item["kind"] == "normal" for item in manifest["challenge"])


def test_schema_gate_is_reset_only() -> None:
    stats = load_shared_a_statistics("data/synthetic/single_uav_static_ppo_smoke_v1_seed2026/shared_train_statistics.json")
    base = replace(
        load_single_uav_ppo_config("data/configs/single_uav_ppo_baseline_v1.json"),
        train_potential_statistics=stats[2],
    )
    result = schema_gate(base, build_split_manifest()["calibration"][0])
    assert result["passed"] is True
    assert result["env_steps"] == 0
    assert result["actual_actor_node_shape"] == [1, 42]
    assert result["actual_central_shape"] == [174]


def test_full_statistics_preserve_signed_direction_and_are_read_only(tmp_path) -> None:
    bank_path = tmp_path / "calibration_bank.npz"
    artifact_path = tmp_path / "full_statistics.json"
    _bank(bank_path)
    potential = {"u_center": 1.0, "u_scale": 2.0, "c_center": 3.0, "c_scale": 4.0, "delta_u_scale": 0.5, "delta_goal_scale": 0.75, "sample_count": 288}
    norm, central, saved_potential, provenance = fit_full_statistics(bank_path, potential, artifact_path, tmp_path / "logs")
    assert saved_potential == potential
    node = norm.state_dict()["normalizers"]["node_features"]
    assert node["mean"][3:6] == [0.0, 0.0, 0.0]
    assert node["mean"][30:33] == [0.0, 0.0, 0.0]
    assert node["scale"][3] == node["scale"][4] == node["scale"][5]
    assert node["scale"][30] == node["scale"][31] == node["scale"][32]
    assert provenance["count"] == CALIBRATION_OBSERVATIONS
    loaded, loaded_central, loaded_potential, loaded_provenance = load_full_statistics(artifact_path)
    assert loaded_potential == potential
    assert loaded_provenance["source_sha256"] == provenance["source_sha256"]
    with pytest.raises(RuntimeError):
        loaded.fit([])
    with pytest.raises(RuntimeError):
        loaded_central.fit(np.zeros((2, EXPECTED_CENTRAL_FEATURE_DIM)))


def test_full_statistics_reject_incomplete_metadata(tmp_path) -> None:
    bank_path = tmp_path / "bad_bank.npz"
    _bank(bank_path)
    with np.load(bank_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["split"] = arrays["split"].copy()
    arrays["split"][0] = "validation"
    np.savez(bank_path, **arrays)
    with pytest.raises(ValueError, match="non-calibration"):
        fit_full_statistics(
            bank_path,
            {"u_center": 0, "u_scale": 1, "c_center": 0, "c_scale": 1, "delta_u_scale": 1, "delta_goal_scale": 1},
            tmp_path / "stats.json",
            tmp_path / "logs",
        )
