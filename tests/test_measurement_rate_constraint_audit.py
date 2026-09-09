"""Tests for the validation-only measurement-rate constraint audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiments.measurement_rate_constraint_audit import (
    MeasurementRateConstraintAuditConfig,
    run_measurement_rate_constraint_audit,
)
from src.experiments.measurement_rate_controlled_mappo import _fit_measurement_rate_targets


BRANCHES = (
    "baseline_mappo",
    "budgeted_quality_aware_mappo",
    "rate_controlled_quality_aware_mappo",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_reference(root: Path, source_logs: Path, seed: int = 7071) -> dict[str, float]:
    targets = {"uav_00": 0.5, "uav_01": 0.0, "uav_02": 0.5}
    for branch in BRANCHES:
        scale = root / f"seed_{seed}" / branch / "scale_3"
        for split in ("train", "validation"):
            episode = scale / split / ("episode_000001" if split == "train" else "episode_400001")
            rows = [
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_00",
                    "measurement_uav_id": "uav_01",
                    "timestamp": 0.2,
                    "action_mask_json": json.dumps({"measurement": [False, True, True, True]}),
                },
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_00",
                    "measurement_uav_id": "none",
                    "timestamp": 0.4,
                    "action_mask_json": json.dumps({"measurement": [False, True, True, True]}),
                },
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_01",
                    "measurement_uav_id": "none",
                    "timestamp": 0.2,
                    "action_mask_json": json.dumps({"measurement": [True, False, True, True]}),
                },
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_01",
                    "measurement_uav_id": "none",
                    "timestamp": 0.4,
                    "action_mask_json": json.dumps({"measurement": [True, False, True, True]}),
                },
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_02",
                    "measurement_uav_id": "uav_00",
                    "timestamp": 0.2,
                    "action_mask_json": json.dumps({"measurement": [True, True, False, True]}),
                },
                {
                    "episode_id": episode.name,
                    "split": split,
                    "uav_id": "uav_02",
                    "measurement_uav_id": "none",
                    "timestamp": 0.4,
                    "action_mask_json": json.dumps({"measurement": [True, True, False, True]}),
                },
            ]
            _write_csv(episode / "actions.csv", rows)
            (episode / "measurement_budget.csv").write_text(
                "episode_id,step,uav_id,budget_limit,budget_remaining_before,budget_consumed,budget_remaining_after,budget_used\n"
                + "\n".join(
                    f"{episode.name},{step},{uav_id},1,{1 if step == 0 else 0},{1 if step == 0 and uav_id != 'uav_01' else 0},{0 if step == 0 and uav_id != 'uav_01' else 1},{1 if step == 0 and uav_id != 'uav_01' else 0}"
                    for step in (0, 1)
                    for uav_id in ("uav_00", "uav_01", "uav_02")
                )
                + "\n",
                encoding="utf-8",
            )
        config = {
            "mappo_variant": branch,
            "measurement_quality_logit_penalty": 1.0 if branch != "baseline_mappo" else 0.0,
            "measurement_budget_enabled": branch != "baseline_mappo",
            "measurement_rate_constraint_enabled": branch == "rate_controlled_quality_aware_mappo",
            "measurement_rate_constraint_coef": 0.5 if branch == "rate_controlled_quality_aware_mappo" else 0.0,
            "measurement_rate_tolerance": 0.05,
            "measurement_rate_target_by_uav": targets if branch == "rate_controlled_quality_aware_mappo" else None,
        }
        (scale / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (scale / "manifest.json").write_text(json.dumps({"config": config}), encoding="utf-8")
        quality_path = scale / "diagnostics" / "measurement_quality.jsonl"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_rows: list[dict[str, object]] = []
        for split, episode_id in (("train", "episode_000001"), ("validation", "episode_400001")):
            for uav_id in ("uav_00", "uav_01", "uav_02"):
                expected = targets[uav_id]
                for step in (0, 1):
                    quality_rows.append(
                        {
                            "episode_id": episode_id,
                            "split": split,
                            "phase": split,
                            "step": step,
                            "uav_id": uav_id,
                            "none_probability": 1.0 - expected,
                            "quality_feature_enabled": int(branch != "baseline_mappo"),
                            "measurement_budget_feature_enabled": int(branch != "baseline_mappo"),
                            "measurement_rate_constraint_enabled": int(branch == "rate_controlled_quality_aware_mappo"),
                            "quality_actor_branch": branch,
                        }
                    )
        quality_path.write_text(
            "".join(json.dumps(row) + "\n" for row in quality_rows),
            encoding="utf-8",
        )
    source_logs.mkdir(parents=True, exist_ok=True)
    target_rows = [
        {
            "base_seed": seed,
            "uav_id": uav_id,
            "baseline_train_selected_measurements": 1 if target > 0 else 0,
            "baseline_train_decision_windows": 2,
            "target_measurement_rate": target,
            "source_split": "train",
            "source_branch": "baseline_mappo",
        }
        for uav_id, target in targets.items()
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
    return targets


def test_target_recomputation_is_seed_bound(tmp_path: Path) -> None:
    scale = tmp_path / "baseline" / "scale_3"
    _write_csv(
        scale / "train" / "episode_000001" / "actions.csv",
        [
            {"uav_id": "uav_00", "measurement_uav_id": "uav_01"},
            {"uav_id": "uav_00", "measurement_uav_id": "none"},
            {"uav_id": "uav_01", "measurement_uav_id": "none"},
            {"uav_id": "uav_02", "measurement_uav_id": "uav_00"},
        ],
    )
    targets, rows = _fit_measurement_rate_targets(scale, uav_count=3, base_seed=7071)
    assert targets == {"uav_00": 0.5, "uav_01": 0.0, "uav_02": 1.0}
    assert {row["base_seed"] for row in rows} == {7071}


def test_rate_constraint_audit_is_validation_only_and_complete(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    source_logs = tmp_path / "source_logs"
    _make_reference(reference, source_logs)
    # A malformed test file is intentional: the audit must not open it.
    test_file = reference / "seed_7071" / "baseline_mappo" / "scale_3" / "test" / "episode_500001" / "actions.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("not,csv,for,this,audit\n", encoding="utf-8")
    logs = tmp_path / "audit"
    plots = tmp_path / "plots"
    result = run_measurement_rate_constraint_audit(
        MeasurementRateConstraintAuditConfig(
            reference_root=reference,
            source_logs_dir=source_logs,
            output_dir=logs,
            plots_dir=plots,
            seeds=(7071,),
            bootstrap_samples=200,
            force=True,
            progress=False,
        )
    )
    assert result["audit_ok"] is True
    report = json.loads((logs / "audit_report.json").read_text(encoding="utf-8"))
    assert report["test_data_read"] is False
    assert report["checks"]["target_recomputation_complete"] is True
    assert report["checks"]["target_logged_seed_bound_match"] is True
    assert (logs / "training_constraint_audit.csv").is_file()
    assert (logs / "validation_rate_audit.csv").is_file()
    assert (plots / "measurement_rate_constraint_audit.png").is_file()
