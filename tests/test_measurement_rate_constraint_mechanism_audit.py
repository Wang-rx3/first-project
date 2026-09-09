"""Tests for the measurement-rate aggregation and gradient-path audit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.experiments.measurement_rate_constraint_mechanism_audit import (
    MeasurementRateConstraintMechanismAuditConfig,
    run_measurement_rate_constraint_mechanism_audit,
)
from src.learning.mappo import MAPPOConfig, MAPPOPolicy


SEED = 7071
UAVS = ("uav_00", "uav_01", "uav_02")
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


def _mask(uav_id: str) -> str:
    index = UAVS.index(uav_id)
    values = [True, True, True, True]
    values[index] = False
    return json.dumps({"measurement": values})


def _action_rows(episode_id: str, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in (0, 1):
        for uav_id in UAVS:
            selected = UAVS[(UAVS.index(uav_id) + 1) % len(UAVS)] if step == 0 else "none"
            rows.append(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "uav_id": uav_id,
                    "step": step,
                    "timestamp": 0.2 + 0.2 * step,
                    "measurement_uav_id": selected,
                    "invalid_measurement_action": 0,
                    "measurement_transition_invalidated": 0,
                    "action_mask_json": _mask(uav_id),
                    "action_raw": json.dumps(
                        {"measurement_uav_id": selected, "delta_v": [0.0, 0.0, 0.0]}
                    ),
                }
            )
    return rows


def _quality_rows(episode_id: str, split: str, branch: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                    "none_probability": 0.25,
                    "measurement_quality_penalty": json.dumps([0.2, 0.4, 0.6]),
                    "measurement_quality_logit_penalty": 1.0,
                    "measurement_quality_valid": json.dumps([1.0, 1.0, 1.0]),
                    "measurement_budget_remaining_ratio": 1.0,
                    "quality_feature_enabled": int(branch != "baseline_mappo"),
                    "measurement_budget_feature_enabled": int(branch != "baseline_mappo"),
                    "measurement_rate_constraint_enabled": int(
                        branch == "rate_controlled_quality_aware_mappo"
                    ),
                    "quality_actor_branch": branch,
                }
            )
    return rows


def _graph_rows(episode_id: str) -> list[dict[str, object]]:
    node_features = [[0.0] * 37 for _ in UAVS]
    edge_index = [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]
    edge_features = [[0.0] * 11 for _ in range(6)]
    rows: list[dict[str, object]] = []
    for step in (0, 1):
        for uav_id in UAVS:
            rows.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "uav_id": uav_id,
                    "node_features_json": json.dumps(node_features),
                    "effective_edge_index_json": json.dumps(edge_index),
                    "effective_edge_features_json": json.dumps(edge_features),
                    "candidate_edge_features_json": json.dumps({}),
                    "candidate_incoming_edge_features_json": json.dumps({}),
                    "potential_edge_features_json": json.dumps({}),
                    "action_mask_json": _mask(uav_id),
                    "bandwidth_mask_json": json.dumps([True, True, True]),
                }
            )
    return rows


def _make_reference(root: Path, source_logs: Path) -> None:
    target_map = {uav_id: 0.5 for uav_id in UAVS}
    rate_config = MAPPOConfig(
        uav_count=3,
        node_feature_dim=37,
        edge_feature_dim=11,
        gat_hidden_dim=4,
        gat_output_dim=4,
        gat_heads=1,
        gat_layers=1,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        max_obstacles=2,
        policy_epochs=1,
        minibatch_size=4,
        use_measurement_quality_features=True,
        use_measurement_budget_features=True,
        measurement_quality_logit_penalty=1.0,
        measurement_rate_constraint_enabled=True,
        measurement_rate_constraint_coef=0.5,
        measurement_rate_tolerance=0.05,
        measurement_rate_target_by_uav=target_map,
    )
    policy = MAPPOPolicy(rate_config, uav_ids=UAVS)

    for branch in BRANCHES:
        scale = root / f"seed_{SEED}" / branch / "scale_3"
        config = {
            "mappo_variant": branch,
            "uav_count": 3,
            "node_feature_dim": 37,
            "measurement_quality_logit_penalty": 0.0 if branch == "baseline_mappo" else 1.0,
            "measurement_budget_enabled": branch != "baseline_mappo",
            "measurement_rate_constraint_enabled": branch == "rate_controlled_quality_aware_mappo",
            "measurement_rate_constraint_coef": 0.5 if branch == "rate_controlled_quality_aware_mappo" else 0.0,
            "measurement_rate_tolerance": 0.05,
            "measurement_rate_target_by_uav": target_map if branch == "rate_controlled_quality_aware_mappo" else None,
        }
        (scale / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (scale / "manifest.json").write_text(
            json.dumps({"config": config}), encoding="utf-8"
        )
        for split, episode_id in (
            ("train", "episode_000001"),
            ("validation", "episode_400001"),
        ):
            episode = scale / split / episode_id
            _write_csv(episode / "actions.csv", _action_rows(episode_id, split))
            if branch == "rate_controlled_quality_aware_mappo" and split == "train":
                _write_csv(episode / "graph_features.csv", _graph_rows(episode_id))
        quality_path = scale / "diagnostics" / "measurement_quality.jsonl"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality = (
            _quality_rows("episode_000001", "train", branch)
            + _quality_rows("episode_400001", "validation", branch)
        )
        quality_path.write_text(
            "".join(json.dumps(row) + "\n" for row in quality), encoding="utf-8"
        )
        if branch == "rate_controlled_quality_aware_mappo":
            checkpoint = scale / "checkpoints" / "final.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "trainer_config": asdict(rate_config),
                    "policy_state_dict": policy.state_dict(),
                },
                checkpoint,
            )

    source_logs.mkdir(parents=True, exist_ok=True)
    _write_csv(
        source_logs / "measurement_rate_targets.csv",
        [
            {
                "base_seed": SEED,
                "uav_id": uav_id,
                "baseline_train_selected_measurements": 1,
                "baseline_train_decision_windows": 2,
                "target_measurement_rate": 0.5,
                "source_split": "train",
                "source_branch": "baseline_mappo",
            }
            for uav_id in UAVS
        ],
    )
    row: dict[str, object] = {
        "base_seed": SEED,
        "treatment": "rate_controlled_quality_aware_mappo",
        "phase": "train",
        "update": 1,
        "episode_id": "episode_000001",
        "measurement_rate_constraint_enabled": 1,
        "measurement_rate_constraint_loss": 0.04,
        "measurement_rate_constraint_weighted_loss": 0.02,
        "measurement_rate_expected_mean": 0.75,
        "measurement_rate_upper_residual_mean": 0.20,
        "measurement_rate_lower_residual_mean": 0.0,
        "measurement_rate_constraint_active_fraction": 1.0,
        "samples": 6,
    }
    for uav_id in UAVS:
        row[f"measurement_rate_expected_{uav_id}"] = 0.75
        row[f"measurement_rate_target_{uav_id}"] = 0.5
        row[f"measurement_rate_upper_residual_{uav_id}"] = 0.20
        row[f"measurement_rate_lower_residual_{uav_id}"] = 0.0
        row[f"measurement_rate_raw_loss_{uav_id}"] = 0.04
        row[f"measurement_rate_active_fraction_{uav_id}"] = 1.0
    _write_csv(source_logs / "training_rate_constraint.csv", [row])


def test_mechanism_audit_closes_per_uav_data_and_gradient_path(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    source_logs = tmp_path / "source_logs"
    _make_reference(reference, source_logs)

    # A malformed test file is intentional.  The audit must not enumerate or
    # parse the sealed test split.
    test_file = (
        reference
        / f"seed_{SEED}"
        / "baseline_mappo"
        / "scale_3"
        / "test"
        / "episode_500001"
        / "actions.csv"
    )
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("not,a,valid,audit,file\n", encoding="utf-8")

    output = tmp_path / "audit"
    plots = tmp_path / "plots"
    result = run_measurement_rate_constraint_mechanism_audit(
        MeasurementRateConstraintMechanismAuditConfig(
            reference_root=reference,
            source_logs_dir=source_logs,
            output_dir=output,
            plots_dir=plots,
            seeds=(SEED,),
            bootstrap_samples=200,
            force=True,
            progress=False,
        )
    )

    assert result["audit_ok"] is True
    report = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    assert report["test_data_read"] is False
    assert report["checks"]["target_recomputation_complete"] is True
    assert report["checks"]["per_uav_aggregation_closed"] is True
    assert report["checks"]["gradient_probe_complete"] is True
    assert report["checks"]["gradient_path_nonzero_when_active"] is True
    assert report["checks"]["checkpoint_unchanged"] is True
    assert report["primary_reason"] == "constraint_active_but_insufficient"

    aggregation = list(
        csv.DictReader(
            (output / "aggregation_audit.csv").open(encoding="utf-8-sig")
        )
    )
    assert aggregation and all(row["aggregation_ok"] == "1" for row in aggregation)
    gradient = list(
        csv.DictReader(
            (output / "gradient_path_audit.csv").open(encoding="utf-8-sig")
        )
    )
    assert gradient[0]["probe_ok"] == "1"
    assert float(gradient[0]["measurement_head_gradient_norm"]) > 0.0
    assert (plots / "measurement_rate_constraint_mechanism_audit.png").is_file()
