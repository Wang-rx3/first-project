"""Unit tests for the validation-only final mainline audit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.experiments.final_mainline_audit import (
    BASELINE_BRANCH,
    PROPOSED_BRANCH,
    FinalMainlineAuditConfig,
    _measurement_counts,
    _paired_contrasts,
    _training_aggregate,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path, **kwargs: object) -> FinalMainlineAuditConfig:
    values: dict[str, object] = {
        "reference_root": tmp_path / "reference",
        "bootstrap_samples": 100,
        "progress": False,
    }
    values.update(kwargs)
    return FinalMainlineAuditConfig(**values)


def test_measurement_overhead_invariants_are_counted_per_event(tmp_path: Path) -> None:
    episode = tmp_path / "episode_400001"
    _write_csv(
        episode / "actions.csv",
        [
            {"measurement_uav_id": "uav_01", "invalid_measurement_action": 0, "measurement_transition_invalidated": 0},
            {"measurement_uav_id": "none", "invalid_measurement_action": 0, "measurement_transition_invalidated": 0},
        ],
    )
    _write_csv(
        episode / "measurement_overhead.csv",
        [
            {
                "selected_non_none": 1,
                "payload_size_bytes": 512,
                "load_packets_requested": 1,
                "load_packets_applied": 1,
            }
        ],
    )

    counts = _measurement_counts(episode)

    assert counts["selected_non_none_actions"] == 1
    assert counts["overhead_event_count"] == 1
    assert counts["overhead_packets_applied"] == 1.0
    assert counts["overhead_invariant_failure_count"] == 0
    assert counts["action_overhead_count_mismatch"] == 0


def test_paired_contrasts_use_seed_level_rows() -> None:
    seeds = (101, 202, 303, 404, 505)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.extend(
            [
                {
                    "base_seed": seed,
                    "branch": BASELINE_BRANCH,
                    "position_rmse_norm_m": 1.0,
                    "position_p95_norm_m": 1.2,
                    "j_loc": 0.4,
                    "j_com": 0.2,
                    "mean_reward": -0.5,
                    "harmful_fraction": 0.3,
                    "beneficial_fraction": 0.5,
                    "accepted_updates_per_window": 0.4,
                    "update_coverage_any": 1.0,
                    "measurement_count": 10,
                    "measurement_overhead_packets": 10,
                    "actual_communication_cost": 0.5,
                    "expected_communication_cost": 0.5,
                    "packet_loss": 0.1,
                    "delay_s": 0.1,
                    "aoi_s": 0.2,
                    "connectivity": 0.8,
                    "energy_proxy": 0.1,
                },
                {
                    "base_seed": seed,
                    "branch": PROPOSED_BRANCH,
                    "position_rmse_norm_m": 0.9,
                    "position_p95_norm_m": 1.1,
                    "j_loc": 0.4,
                    "j_com": 0.2,
                    "mean_reward": -0.4,
                    "harmful_fraction": 0.2,
                    "beneficial_fraction": 0.6,
                    "accepted_updates_per_window": 0.5,
                    "update_coverage_any": 1.0,
                    "measurement_count": 10,
                    "measurement_overhead_packets": 10,
                    "actual_communication_cost": 0.5,
                    "expected_communication_cost": 0.5,
                    "packet_loss": 0.1,
                    "delay_s": 0.1,
                    "aoi_s": 0.2,
                    "connectivity": 0.8,
                    "energy_proxy": 0.1,
                },
            ]
        )

    contrasts = _paired_contrasts(
        rows,
        _config(Path("."), seeds=seeds, branches=(BASELINE_BRANCH, PROPOSED_BRANCH)),
    )
    rmse = next(row for row in contrasts if row["metric"] == "position_rmse_norm_m")

    assert rmse["seed_count"] == 5
    assert rmse["improved_seed_count"] == 5
    assert np.isclose(float(rmse["mean_difference"]), -0.1)


def test_training_aggregate_reports_mean_and_seed_interval() -> None:
    config = _config(Path("."), seeds=(1, 2, 3), branches=(BASELINE_BRANCH, PROPOSED_BRANCH))
    rows = [
        {"base_seed": 1, "branch": BASELINE_BRANCH, "update": 1, "mean_reward": -1.0},
        {"base_seed": 2, "branch": BASELINE_BRANCH, "update": 1, "mean_reward": -0.5},
        {"base_seed": 3, "branch": BASELINE_BRANCH, "update": 1, "mean_reward": 0.0},
    ]

    aggregate = _training_aggregate(rows, "mean_reward", config)

    assert len(aggregate) == 1
    assert np.isclose(float(aggregate[0]["mean"]), -0.5)
    assert float(aggregate[0]["ci95_lower"]) <= -0.5 <= float(aggregate[0]["ci95_upper"])


def test_final_audit_requires_baseline_and_proposed_branches(tmp_path: Path) -> None:
    try:
        _config(tmp_path, branches=(BASELINE_BRANCH,))
    except ValueError as exc:
        assert "proposed branch" in str(exc)
    else:
        raise AssertionError("configuration without proposed branch was accepted")
