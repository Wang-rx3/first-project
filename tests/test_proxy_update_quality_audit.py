"""Unit tests for the frozen proxy/update-quality attribution audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.proxy_update_quality_audit import (
    DEFAULT_BRANCHES,
    ProxyUpdateQualityAuditConfig,
    _candidate_rows,
    _teacher_rows,
    _teacher_probabilities,
)


def test_audit_is_validation_only_and_accepts_the_three_formal_branches() -> None:
    config = ProxyUpdateQualityAuditConfig(
        reference_root="data",
        branches=DEFAULT_BRANCHES,
        split="validation",
    )

    assert config.split == "validation"
    assert config.branches == DEFAULT_BRANCHES

    with pytest.raises(ValueError, match="validation-only"):
        ProxyUpdateQualityAuditConfig(reference_root="data", split="test")


def test_teacher_masks_invalid_candidates_and_keeps_none_available() -> None:
    probabilities = _teacher_probabilities(
        np.asarray([0.2, 0.9, 0.4]),
        np.asarray([0.1, 0.1, 0.1]),
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([1.0, 0.0, 1.0, 1.0]),
        temperature=0.25,
        communication_cost_weight=1.0,
    )

    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[1] == 0.0
    assert probabilities[3] > 0.0
    assert int(np.argmax(probabilities)) != 1


def test_teacher_is_deterministic_and_invalid_proxy_is_not_selected() -> None:
    args = (
        np.asarray([0.1, 0.8, 0.3]),
        np.asarray([0.2, 0.1, 0.2]),
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0, 1.0]),
    )
    first = _teacher_probabilities(*args, temperature=0.25, communication_cost_weight=1.0)
    second = _teacher_probabilities(*args, temperature=0.25, communication_cost_weight=1.0)

    assert np.array_equal(first, second)
    assert first[1] == 0.0
    assert np.isclose(first.sum(), 1.0)


def test_candidate_rows_expand_finite_candidate_level_diagnostics() -> None:
    rows = _candidate_rows(
        [
            {
                "base_seed": 7071,
                "branch": "proxy_guided_adaptive_budgeted_quality_aware_mappo",
                "episode_id": "episode_400001",
                "episode_seed": 123,
                "split": "validation",
                "step": 0,
                "timestamp": 0.2,
                "uav_id": "uav_00",
                "candidate_uav_ids": ["uav_01", "uav_02"],
                "measurement_proxy_gain": [0.2, 0.8, 0.1],
                "measurement_proxy_cost": [0.3, 0.2, 0.4],
                "measurement_proxy_valid": [1.0, 1.0, 1.0],
                "measurement_probabilities": [0.2, 0.7, 0.05, 0.05],
                "proxy_teacher_probabilities": [0.1, 0.8, 0.05, 0.05],
                "submitted_measurement_uav_id": "uav_01",
                "proxy_features_in_actor": 1,
                "action_mask": [1, 1, 1, 1],
                "candidate_link_metrics": [
                    {"source_uav_id": "uav_01", "rssi_dbm": -60.0, "snr_db": 20.0},
                    {"source_uav_id": "uav_02", "rssi_dbm": -70.0, "snr_db": 15.0},
                ],
            }
        ],
        weight=1.0,
    )

    assert [row["source_uav_id"] for row in rows] == ["uav_01", "uav_02"]
    assert rows[0]["selected_by_actor"] == 1
    assert rows[0]["proxy_valid"] == 1
    assert all(np.isfinite(float(value)) for row in rows for value in row.values() if isinstance(value, (int, float)))
    assert all(row["truth_used_for_proxy"] == 0 for row in rows)


def test_none_selection_does_not_become_a_huge_uav_index() -> None:
    rows = _teacher_rows(
        [
            {
                "base_seed": 7071,
                "branch": "baseline_mappo",
                "episode_id": "episode_400001",
                "step": 0,
                "uav_id": "uav_00",
                "candidate_uav_ids": ["uav_01"],
                "measurement_proxy_gain": [0.1, 0.0, 0.0],
                "measurement_proxy_cost": [0.2, 1.0, 1.0],
                "measurement_proxy_valid": [1.0, 0.0, 0.0],
                "measurement_proxy_utility": [-0.1, -1.0, -1.0],
                "measurement_probabilities": [0.0, 0.2, 0.8, 0.0],
                "action_mask": [0, 1, 0, 1],
                "submitted_measurement_uav_id": "none",
                "proxy_teacher_choice": "none",
                "proxy_features_in_actor": 0,
            }
        ],
        weight=1.0,
        temperature=0.25,
    )

    assert len(rows) == 1
    assert rows[0]["selected_proxy_utility"] == 0.0
    assert np.isfinite(float(rows[0]["teacher_kl_actor"]))
