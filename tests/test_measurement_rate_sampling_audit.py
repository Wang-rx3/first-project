"""Tests for the expected-rate versus realized-action audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiments.measurement_rate_sampling_audit import (
    MeasurementRateSamplingAuditConfig,
    run_measurement_rate_sampling_audit,
)


BRANCHES = (
    "baseline_mappo",
    "budgeted_quality_aware_mappo",
    "rate_controlled_quality_aware_mappo",
)
UAVS = ("uav_00", "uav_01", "uav_02")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _action_rows(episode_id: str, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = {
        "uav_00": json.dumps({"measurement": [False, True, True, True]}),
        "uav_01": json.dumps({"measurement": [True, False, True, True]}),
        "uav_02": json.dumps({"measurement": [True, True, False, True]}),
    }
    raw_actions = {
        "uav_00": ("uav_01", "none"),
        "uav_01": ("none", "none"),
        "uav_02": ("uav_00", "none"),
    }
    for step in (0, 1):
        for uav_id in UAVS:
            raw_id = raw_actions[uav_id][step]
            applied_id = (
                "none"
                if uav_id == "uav_02" and step == 0
                else raw_id
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "uav_id": uav_id,
                    "measurement_uav_id": applied_id,
                    "invalid_measurement_action": 0,
                    "measurement_transition_invalidated": int(
                        uav_id == "uav_02" and step == 0
                    ),
                    "timestamp": 0.2 + 0.2 * step,
                    "action_mask_json": masks[uav_id],
                    "action_raw": json.dumps(
                        {"measurement_uav_id": raw_id, "delta_v": [0.0, 0.0, 0.0]}
                    ),
                }
            )
    return rows


def _quality_rows(episode_id: str, split: str, branch: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = {
        "uav_00": ("uav_01", "none"),
        "uav_01": ("none", "none"),
        "uav_02": ("uav_00", "none"),
    }
    for step in (0, 1):
        for uav_id in UAVS:
            rows.append(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "phase": split,
                    "step": step,
                    "timestamp": 0.2 + 0.2 * step,
                    "uav_id": uav_id,
                    "measurement_uav_id": selected[uav_id][step],
                    "none_probability": 0.5,
                    "quality_feature_enabled": int(branch != "baseline_mappo"),
                    "measurement_budget_feature_enabled": int(branch != "baseline_mappo"),
                    "measurement_rate_constraint_enabled": int(
                        branch == "rate_controlled_quality_aware_mappo"
                    ),
                    "quality_actor_branch": branch,
                }
            )
    return rows


def _make_reference(root: Path, source_logs: Path, seed: int = 7071) -> None:
    applied_targets = {"uav_00": 0.5, "uav_01": 0.0, "uav_02": 0.0}
    for branch in BRANCHES:
        scale = root / f"seed_{seed}" / branch / "scale_3"
        configs = {
            "mappo_variant": branch,
            "measurement_quality_logit_penalty": (
                0.0 if branch == "baseline_mappo" else 1.0
            ),
            "measurement_budget_enabled": branch != "baseline_mappo",
            "measurement_rate_constraint_enabled": (
                branch == "rate_controlled_quality_aware_mappo"
            ),
            "measurement_rate_constraint_coef": (
                0.5 if branch == "rate_controlled_quality_aware_mappo" else 0.0
            ),
            "measurement_rate_tolerance": 0.05,
            "measurement_rate_target_by_uav": (
                applied_targets
                if branch == "rate_controlled_quality_aware_mappo"
                else None
            ),
            "measurement_budget_by_uav": {
                "uav_00": 1,
                "uav_01": 1,
                "uav_02": 1,
            },
        }
        (scale / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (scale / "manifest.json").write_text(
            json.dumps({"config": configs}),
            encoding="utf-8",
        )
        for split, episode_id in (
            ("train", "episode_000001"),
            ("validation", "episode_400001"),
        ):
            episode = scale / split / episode_id
            _write_csv(episode / "actions.csv", _action_rows(episode_id, split))
            if branch != "baseline_mappo":
                _write_csv(
                    episode / "measurement_budget.csv",
                    [
                        {
                            "episode_id": episode_id,
                            "step": step,
                            "uav_id": uav_id,
                            "budget_limit": 1,
                            "budget_remaining_before": 1 if step == 0 else 0,
                            "budget_used": 0 if step == 0 else 1,
                        }
                        for step in (0, 1)
                        for uav_id in UAVS
                    ],
                )
        quality_path = scale / "diagnostics" / "measurement_quality.jsonl"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality = (
            _quality_rows("episode_000001", "train", branch)
            + _quality_rows("episode_400001", "validation", branch)
        )
        quality_path.write_text(
            "".join(json.dumps(row) + "\n" for row in quality),
            encoding="utf-8",
        )

    source_logs.mkdir(parents=True, exist_ok=True)
    target_rows = [
        {
            "base_seed": seed,
            "uav_id": uav_id,
            "baseline_train_selected_measurements": int(rate * 2),
            "baseline_train_decision_windows": 2,
            "target_measurement_rate": rate,
            "source_split": "train",
            "source_branch": "baseline_mappo",
        }
        for uav_id, rate in applied_targets.items()
    ]
    _write_csv(source_logs / "measurement_rate_targets.csv", target_rows)
    _write_csv(
        source_logs / "training_rate_constraint.csv",
        [
            {
                "base_seed": seed,
                "treatment": "rate_controlled_quality_aware_mappo",
                "phase": "train",
                "update": 1,
                "episode_id": "episode_000001",
                "measurement_rate_constraint_enabled": 1,
                "measurement_rate_constraint_loss": 0.01,
            }
        ],
    )


def test_sampling_audit_separates_raw_from_applied_and_seals_test(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    source_logs = tmp_path / "source_logs"
    _make_reference(reference, source_logs)

    malformed_test = (
        reference
        / "seed_7071"
        / "baseline_mappo"
        / "scale_3"
        / "test"
        / "episode_500001"
        / "actions.csv"
    )
    malformed_test.parent.mkdir(parents=True, exist_ok=True)
    malformed_test.write_text("not,a,valid,audit,file\n", encoding="utf-8")

    output = tmp_path / "audit"
    plots = tmp_path / "plots"
    result = run_measurement_rate_sampling_audit(
        MeasurementRateSamplingAuditConfig(
            reference_root=reference,
            source_logs_dir=source_logs,
            output_dir=output,
            plots_dir=plots,
            seeds=(7071,),
            bootstrap_samples=200,
            force=True,
            progress=False,
        )
    )
    assert result["audit_ok"] is True
    report = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    assert report["test_data_read"] is False
    assert report["checks"]["validation_action_selection_mode"] == "deterministic_argmax"
    assert report["checks"]["sampling_reference_applicable"] is False
    assert report["checks"]["probability_action_alignment_mismatch_count"] == 0
    assert report["evidence"]["transition_invalidated_total"] > 0
    assert report["primary_reason"] != "probability_logging_sampling_mismatch"

    rows = list(csv.DictReader((output / "seed_summary.csv").open(encoding="utf-8-sig")))
    rate_row = next(
        row for row in rows
        if row["branch"] == "rate_controlled_quality_aware_mappo"
    )
    assert int(rate_row["raw_selected_measurements"]) == 2
    assert int(rate_row["applied_selected_measurements"]) == 1
    assert float(rate_row["raw_sampled_rate_all_windows"]) == 2 / 6
    assert float(rate_row["applied_measurement_rate_all_windows"]) < 2 / 6
    assert (output / "expected_vs_sampled.csv").is_file()
    assert (output / "target_recomputation.csv").is_file()
    assert (output / "mask_budget_audit.csv").is_file()
    assert (output / "training_constraint_audit.csv").is_file()
    assert (plots / "measurement_rate_sampling_audit.png").is_file()
