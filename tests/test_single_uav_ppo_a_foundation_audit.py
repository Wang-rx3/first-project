"""Pure tests for the independent PPO-A artifact audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.analysis.single_uav_ppo_a_foundation_audit import (
    _Audit,
    _check_action_and_updates,
    _check_conditional_gate,
    audit_artifacts,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_is_pending_when_future_phase_outputs_are_absent(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "synthetic" / "single_uav_ppo_a_foundation_v1_seed2026"
    log_root = tmp_path / "logs" / "preliminary" / "single_uav_ppo_a_foundation_v1_seed2026"
    data_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (data_root / "run_specification.lock.json").write_text(json.dumps({"schema_version": "test"}), encoding="utf-8")
    (data_root / "scenario_manifest.json").write_text(json.dumps({
        "normal_train": [{"scenario_id": f"t{index}"} for index in range(6)],
        "validation_normal": [{"scenario_id": f"v{index}"} for index in range(8)],
    }), encoding="utf-8")
    output = tmp_path / "audit.json"
    report = audit_artifacts(log_root, project_root=tmp_path, output=output)
    assert report["status"] == "pending"
    assert report["passed"] is False
    assert report["env_steps_executed"] == 0
    assert report["ppo_updates_executed"] == 0
    assert output.is_file()


def test_update_audit_requires_120_samples_and_eight_steps_per_update(tmp_path: Path) -> None:
    audit = _Audit(tmp_path, tmp_path / "logs", tmp_path / "data")
    arm_root = audit.log_root / "G"
    arm_root.mkdir(parents=True)
    train_ids = {f"t{index}" for index in range(6)}
    validation_ids = {f"v{index}" for index in range(8)}
    rows: list[dict[str, object]] = []
    for scenario_id in sorted(train_ids):
        for _ in range(120):
            rows.append({"phase": "online", "scenario_id": scenario_id, "logprob_action": "sampled", "sampled_delta_v": "[0.0, 0.0, 0.0]", "old_logprob": "-0.1"})
    for phase, identifiers in (("initial_train_eval", train_ids), ("final_train_eval", train_ids), ("initial_validation", validation_ids), ("final_validation", validation_ids)):
        rows.extend({"phase": phase, "scenario_id": scenario_id, "logprob_action": "sampled", "sampled_delta_v": "[0.0, 0.0, 0.0]", "old_logprob": "-0.1"} for scenario_id in sorted(identifiers))
    _write_csv(arm_root / "steps.csv", rows)
    for index in range(1, 7):
        (arm_root / f"update_{index:02d}.json").write_text(json.dumps({
            "expected_samples": 120,
            "record_count": 120,
            "optimizer_steps": index * 8,
            "on_policy_logprob_within_tolerance": True,
            "normalizers_unchanged": True,
            "all_weights_finite": True,
            "rollout_empty_after_update": True,
        }), encoding="utf-8")
    _check_action_and_updates(audit, "G", train_ids, validation_ids)
    assert audit.checks["G_sampled_action_logprob"] is True
    assert audit.checks["G_120_samples_x_6_updates_x_8_optimizer"] is True
    assert audit.checks["G_shared_six_train_fresh_eight_validation"] is True


def test_conditional_artifacts_are_blocked_when_normal_gate_failed(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    (log_root / "normal_training_result.json").write_text(json.dumps({"normal_release_gate_passed": False}), encoding="utf-8")
    (log_root / "single").mkdir()
    audit = _Audit(tmp_path, log_root, tmp_path / "data")
    _check_conditional_gate(audit)
    assert audit.checks["conditional_gate_blocks_single_full_when_failed"] is False


def test_audit_does_not_execute_environment_or_checkpoint_loads(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "synthetic" / "single_uav_ppo_a_foundation_v1_seed2026"
    log_root = tmp_path / "logs" / "preliminary" / "single_uav_ppo_a_foundation_v1_seed2026"
    data_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (data_root / "run_specification.lock.json").write_text(json.dumps({}), encoding="utf-8")
    (data_root / "scenario_manifest.json").write_text(json.dumps({
        "normal_train": [{"scenario_id": f"t{index}"} for index in range(6)],
        "validation_normal": [{"scenario_id": f"v{index}"} for index in range(8)],
    }), encoding="utf-8")
    report = audit_artifacts(log_root, project_root=tmp_path)
    assert report["read_only"] is True
    assert report["env_steps_executed"] == 0
    assert report["ppo_updates_executed"] == 0
    assert report["checkpoint_loads_executed"] == 0


def test_audit_output_is_exclusive_and_never_overwrites(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "synthetic" / "single_uav_ppo_a_foundation_v1_seed2026"
    log_root = tmp_path / "logs" / "preliminary" / "single_uav_ppo_a_foundation_v1_seed2026"
    data_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (data_root / "run_specification.lock.json").write_text(json.dumps({}), encoding="utf-8")
    (data_root / "scenario_manifest.json").write_text(json.dumps({
        "normal_train": [{"scenario_id": f"t{index}"} for index in range(6)],
        "validation_normal": [{"scenario_id": f"v{index}"} for index in range(8)],
    }), encoding="utf-8")
    output = tmp_path / "audit.json"
    audit_artifacts(log_root, project_root=tmp_path, output=output)
    original = output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit_artifacts(log_root, project_root=tmp_path, output=output)
    assert output.read_text(encoding="utf-8") == original
