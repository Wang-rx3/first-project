"""Tests for canonical localization metrics and persistence NA semantics."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from src.experiments.ekf_update_persistence_audit import (
    _episode_rows,
    _seed_rows,
    _transition_type,
)
from src.experiments.metrics import position_error_metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_position_error_metrics_uses_one_euclidean_error_per_state_row() -> None:
    truth = [
        {"uav_id": "uav_00", "timestamp": 0.2, "x": 0.0, "y": 0.0, "z": 0.0},
        {"uav_id": "uav_01", "timestamp": 0.2, "x": 0.0, "y": 0.0, "z": 0.0},
    ]
    estimates = [
        {"uav_id": "uav_00", "timestamp": 0.2, "x": 1.0, "y": 2.0, "z": 2.0},
        {"uav_id": "uav_01", "timestamp": 0.2, "x": 0.0, "y": 0.0, "z": 4.0},
    ]

    result = position_error_metrics(truth, estimates)

    assert result["sample_count"] == 2.0
    assert np.isclose(result["rmse_norm_m"], math.sqrt(12.5))
    assert np.isclose(result["p95_norm_m"], 3.95)


def test_no_update_episode_reports_conditional_effects_as_na(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    scale = root / "seed_2026" / "baseline_mappo" / "scale_3"
    episode = scale / "validation" / "episode_400001"
    _write_csv(
        episode / "ground_truth.csv",
        [{"uav_id": "uav_00", "timestamp": 0.2, "x": 0.0, "y": 0.0, "z": 0.0}],
    )
    _write_csv(
        episode / "estimates.csv",
        [{"uav_id": "uav_00", "timestamp": 0.2, "x": 1.0, "y": 0.0, "z": 0.0}],
    )
    _write_csv(
        scale / "diagnostics" / "episode_metrics.csv",
        [
            {
                "episode_id": "episode_400001",
                "scenario_id": "scenario_400001",
                "split": "validation",
                "position_rmse_m": 1.0,
                "position_error_p95_m": 1.0,
                "j_loc": 0.3,
                "j_com": 0.2,
                "joint_objective": 0.5,
                "mean_reward": -0.5,
            }
        ],
    )

    episode_rows = _episode_rows(
        reference_root=root,
        seeds=(2026,),
        branches=("baseline_mappo",),
        split="validation",
        horizons=(1,),
        event_rows=[],
    )
    row = episode_rows[0]

    assert row["eventful_episode"] == 0
    assert row["accepted_update_count"] == 0
    assert row["beneficial_fraction"] == ""
    assert row["harmful_fraction"] == ""
    assert row["neutral_fraction"] == ""
    assert row["linear_cov_reduction_event_fraction"] == ""
    assert row["j_loc_immediate_change_abs_max"] == ""
    assert row["h1_censor_fraction"] == ""
    assert np.isclose(float(row["position_rmse_m"]), 1.0)

    seed_row = _seed_rows(
        episode_rows,
        seeds=(2026,),
        branches=("baseline_mappo",),
        horizons=(1,),
    )[0]
    assert seed_row["beneficial_fraction"] == ""
    assert seed_row["linear_cov_reduction_event_fraction"] == ""
    assert seed_row["j_loc_immediate_change_abs_max"] == ""
    assert seed_row["h1_censor_fraction"] == ""


def test_transition_type_distinguishes_sensor_and_cooperative_updates() -> None:
    assert _transition_type({}, timestamp=0.2, cooperative_times=set()) == "prediction_only"
    assert (
        _transition_type(
            {"vo_update_attempted": True},
            timestamp=0.2,
            cooperative_times=set(),
        )
        == "visual_odometry_update"
    )
    assert (
        _transition_type(
            {},
            timestamp=0.2,
            cooperative_times={0.2},
        )
        == "cooperative_uwb_update"
    )
